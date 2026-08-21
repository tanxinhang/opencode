"""Gate F0-G8C: covariance-native conditional information (advice/018).

Replaces the compressed scalar `rho_s` with the full covariance-native
conditional information.  For the equal-covariance Gaussian evidence
`Y_q | H_h ~ N(mu_{h,q}, Sigma_q)` with `delta = mu_1 - mu_0`, the joint
KL of a coalition `S` is

    G_q(S) = (1/2) delta_S^T Sigma_SS^{-1} delta_S,

and the conditional marginal of UAV `i` given `S` has the
Schur-complement closed form

    Delta G_{i|S,q} = (1/2) delta_{i|S}^2 / v_{i|S},
    delta_{i|S} = delta_i - c^T Sigma_SS^{-1} delta_S,
    v_{i|S}     = sigma_i^2 - c^T Sigma_SS^{-1} c,     c = Sigma_{S i}.

This explains BOTH redundancy (`delta_{i|S} ~ 0` -> `Delta G ~ 0`) and
synergy (`v_{i|S} << sigma_i^2` with a surviving residual signal ->
`Delta G > g_i`); the *conditional innovation effect* replaces the
"correlation => penalty" narrative (Theorem 4.108).  For
`Sigma_0 != Sigma_1` the general Gaussian KL is used and
`Delta G = G(S union {i}) - G(S) >= 0` by the KL chain rule.

The covariance `Sigma_q` is derived from the OTFS/DD physics (bistatic
geometry similarity x Doppler-bin overlap x shared clutter), so the OTFS
DD-domain structure enters the scheduling value function directly
(advice/018 section 7).  First version: the covariance is built
offline and used online only via Schur updates -- no online correlation
learning (advice/018 section 8).
"""

from __future__ import annotations

import numpy as np

from uav_otfs_isac.scenario import uav_geometry


# ---------------------------------------------------------------------------
# Covariance-native conditional information
# ---------------------------------------------------------------------------


def joint_kl_equal_cov(delta_S: np.ndarray, sigma_ss: np.ndarray) -> float:
    """``G_q(S) = (1/2) delta_S^T Sigma_SS^{-1} delta_S`` (equal
    covariance under H0/H1)."""
    d = np.asarray(delta_S, dtype=float)
    if len(d) == 0:
        return 0.0
    s = np.asarray(sigma_ss, dtype=float)
    return 0.5 * float(np.dot(d, np.linalg.solve(s, d)))


def schur_conditional_gain(delta: np.ndarray, sigma: np.ndarray,
                           subset, i: int) -> float:
    """``Delta G_{i|S,q} = G(S union {i}) - G(S)`` by the Schur
    complement (equal covariance).  With an empty coalition it equals the
    marginal ``g_i = delta_i^2 / (2 sigma_i^2)``."""
    i = int(i)
    idx = [int(x) for x in subset if int(x) != i]
    d = np.asarray(delta, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if not idx:
        return 0.5 * d[i] ** 2 / max(float(s[i, i]), 1e-30)
    di = d[idx]
    s_ss = s[np.ix_(idx, idx)]
    c = s[idx, i]
    sii = float(s[i, i])
    sol = np.linalg.solve(s_ss, di)
    residual = float(d[i] - np.dot(c, sol))
    v = sii - float(np.dot(c, np.linalg.solve(s_ss, c)))
    if v <= 0.0:
        # numerically singular conditional covariance: clamp to a tiny
        # floor (the shrinkage guarantees SPD, but a near-singular block
        # can still underflow)
        v = max(v, 1e-12)
    return 0.5 * residual ** 2 / v


def joint_kl_general(delta_S: np.ndarray, sigma0_ss: np.ndarray,
                     sigma1_ss: np.ndarray) -> float:
    """Full Gaussian KL of `N(mu1, Sigma1)` vs `N(mu0, Sigma0)` for the
    coalition `S` (used when `Sigma_0 != Sigma_1`):

    ``G = (1/2)[ tr(S0^-1 S1) + delta^T S0^-1 delta - |S|
                  + log det(S0) - log det(S1) ]``.
    """
    d = np.asarray(delta_S, dtype=float)
    if len(d) == 0:
        return 0.0
    s0 = np.asarray(sigma0_ss, dtype=float)
    s1 = np.asarray(sigma1_ss, dtype=float)
    s0inv = np.linalg.inv(s0)
    tr = float(np.trace(np.dot(s0inv, s1)))
    quad = float(np.dot(d, np.dot(s0inv, d)))
    logdet = float(np.linalg.slogdet(s0)[1] - np.linalg.slogdet(s1)[1])
    return 0.5 * (tr + quad - len(d) + logdet)


def conditional_gain_general(delta: np.ndarray, sigma0: np.ndarray,
                             sigma1: np.ndarray, subset, i: int) -> float:
    """``Delta G_{i|S,q} = G(S union {i}) - G(S)`` via the full Gaussian
    KL (general `Sigma_0 != Sigma_1`); nonnegative by the KL chain rule."""
    i = int(i)
    idx = [int(x) for x in subset if int(x) != i]
    idx_wi = sorted(set(idx) | {i})
    g_wi = joint_kl_general(delta[idx_wi], sigma0[np.ix_(idx_wi, idx_wi)],
                            sigma1[np.ix_(idx_wi, idx_wi)])
    if not idx:
        return g_wi
    g_wo = joint_kl_general(delta[idx], sigma0[np.ix_(idx, idx)],
                            sigma1[np.ix_(idx, idx)])
    return max(0.0, g_wi - g_wo)


def marginal_kl(delta: np.ndarray, sigma: np.ndarray, i: int) -> float:
    """Marginal KL of UAV `i` under the equal-covariance model
    ``g_i = delta_i^2 / (2 sigma_i^2)``."""
    i = int(i)
    return 0.5 * float(delta[i]) ** 2 / max(float(sigma[i, i]), 1e-30)


# ---------------------------------------------------------------------------
# OTFS / DD-physics covariance source
# ---------------------------------------------------------------------------


def build_physics_covariance(
    positions: np.ndarray,
    target_pos: np.ndarray,
    fractional_doppler: np.ndarray,
    common_strength: float = 0.5,
    doppler_scale: float = 0.18,
    delay_scale: float = 0.2,
    include_delay: bool = True,
) -> np.ndarray:
    """Per-target OTFS/DD covariance over the UAVs.

    `Sigma_ij = common_strength * geometry_similarity_ij *
    doppler_overlap_ij * delay_overlap_ij` for `i != j`, diagonal 1:

    - geometry similarity: alignment of the two UAV->target line-of-
      sight unit vectors (bistatic geometry closeness);
    - Doppler overlap: `exp(-|nu_i - nu_j| / doppler_scale)` (fractional
      Doppler leakage overlap in the DD grid);
    - delay overlap: `exp(-|tau_i - tau_j| / delay_scale)` (delay-bin
      overlap).

    This gives the cross-UAV evidence covariance a physical OTFS/DD
    origin (advice/018 section 7); the first version builds it offline
    (calibration) and uses it online only through Schur updates.
    """
    p = np.asarray(positions, dtype=float)
    t = np.asarray(target_pos, dtype=float)
    k = p.shape[0]
    view = p - t
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    geom = np.clip(view @ view.T, 0.0, 1.0)
    dop = np.exp(-np.abs(fractional_doppler[:, None]
                         - fractional_doppler[None, :]) / doppler_scale)
    rng_d = np.linalg.norm(p - t, axis=1)
    if include_delay:
        # delay-bin overlap: normalize the bistatic range difference by a
        # characteristic scale (a fraction of the mean range) so the term
        # is dimensionally consistent with the DD-grid delay resolution
        delay_scale_norm = max(float(np.mean(rng_d)) * 0.3, 1e-9)
        delay = np.exp(-np.abs(rng_d[:, None] - rng_d[None, :])
                       / delay_scale_norm)
        corr = geom * dop * delay
    else:
        corr = geom * dop
    cov = float(common_strength) * corr
    np.fill_diagonal(cov, 1.0)
    # symmetric + SPD toward the diagonal
    cov = 0.5 * (cov + cov.T)
    return cov


def shrink_covariance(sigma: np.ndarray, lam: float) -> np.ndarray:
    """Shrinkage `(1-lambda) Sigma + lambda diag(Sigma)` guaranteeing
    positive definiteness (advice/018 section 8)."""
    sigma = np.asarray(sigma, dtype=float)
    return (1.0 - float(lam)) * sigma + float(lam) * np.diag(np.diag(sigma))


# ---------------------------------------------------------------------------
# Profile-driven scenario moments (homogeneous / heterogeneous / concentrated)
# ---------------------------------------------------------------------------


def profile_deltas(k: int, q: int, profile: str,
                   rng: np.random.Generator) -> np.ndarray:
    """Per-target marginal shift matrix (k, q) by profile type."""
    if profile == "homogeneous":
        return rng.uniform(0.35, 0.65, size=(k, q))
    if profile == "heterogeneous":
        return rng.uniform(0.1, 1.2, size=(k, q))
    if profile == "concentrated":
        delta = rng.uniform(0.05, 0.2, size=(k, q))
        for qq in range(q):
            dominant = int(rng.integers(0, k))
            delta[dominant, qq] = rng.uniform(1.5, 2.5)
        return delta
    raise ValueError(f"unknown profile {profile!r}")


def build_profile_moments(
    k: int, q: int, profile: str, rng: np.random.Generator,
    common_strength: float | None = None,
    doppler_scale: float = 0.15,
) -> dict:
    """Build per-target `(delta, Sigma)` for a profile draw.

    The marginal shifts follow the profile (homogeneous / heterogeneous /
    concentrated).  The covariance is the OTFS/DD physics covariance with
    a profile-controlled Doppler spread so the correlation level is
    meaningful (0.2-0.6, not the near-zero of a full-circle geometry):

    - homogeneous: the UAVs' fractional Doppler is tightly clustered
      (nearly the same DD Doppler bin) -> the Doppler-overlap weights are
      ~ 1 and the correlation is uniform at `common_strength`;
    - heterogeneous: the Doppler is spread over the grid -> the weights
      (and hence the correlation) vary strongly across pairs;
    - concentrated: the dominant UAV's Doppler is offset from the weak
      UAVs' cluster, so its correlation with the weak ones is low while
      the weak-weak correlation is high.

    The first version builds the covariance offline and uses it online
    only through Schur updates (no online correlation learning).
    """
    delta = profile_deltas(k, q, profile, rng)
    positions = uav_geometry(k)
    target = np.array([45.0, 55.0, 0.0])
    view = positions - target
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    geom = np.clip(view @ view.T, 0.0, 1.0)
    if profile == "homogeneous":
        frac = rng.uniform(0.28, 0.32, size=k)      # tightly clustered
    elif profile == "heterogeneous":
        frac = rng.uniform(0.0, 0.5, size=k)         # spread over the grid
    else:
        frac = rng.uniform(0.25, 0.3, size=k)        # weak UAVs clustered
        dominant = int(np.argmax(delta[:, 0]))
        frac[dominant] = rng.uniform(0.45, 0.5)      # dominant offset
    dop = np.exp(-np.abs(frac[:, None] - frac[None, :]) / doppler_scale)
    weights = np.clip(geom * dop, 0.0, 1.0)
    np.fill_diagonal(weights, 1.0)
    if common_strength is None:
        if profile == "homogeneous":
            common_strength = rng.uniform(0.4, 0.6)
        elif profile == "heterogeneous":
            common_strength = rng.uniform(0.2, 0.4)
        else:
            common_strength = rng.uniform(0.5, 0.7)
    sigma = float(common_strength) * weights
    np.fill_diagonal(sigma, 1.0)
    sigma = shrink_covariance(sigma, 0.05)
    return {
        "delta": delta,
        "sigma": sigma,
        "profile": profile,
        "common_strength": float(common_strength),
        "fractional_doppler": frac,
    }


def scalar_rho_from_covariance(sigma: np.ndarray) -> float:
    """The natural scalar compression of a covariance: the mean
    off-diagonal correlation (used by the scalar-rho method under the
    covariance world)."""
    s = np.asarray(sigma, dtype=float)
    k = s.shape[0]
    off = s[np.triu_indices(k, 1)]
    return float(np.mean(off)) if len(off) > 0 else 0.0


# ---------------------------------------------------------------------------
# Gaussian-evidence conditional FRIDS simulation (G8-C)
# ---------------------------------------------------------------------------


def _gaussian_joint_llr(y_s: np.ndarray, delta_s: np.ndarray,
                        sigma_ss: np.ndarray) -> float:
    """Joint LLR of the coalition observations `y_s`:
    `ell = delta' Sigma^-1 (y - delta/2)` (drift `G_q(S)` under H1)."""
    d = np.asarray(delta_s, dtype=float)
    if len(d) == 0:
        return 0.0
    s = np.asarray(sigma_ss, dtype=float)
    return float(np.dot(d, np.linalg.solve(s, np.asarray(y_s, dtype=float)
                                           - 0.5 * d)))


def _marginal_llr(y: float, delta: float, var: float = 1.0) -> float:
    """Marginal LLR of one UAV's observation `y` (N(delta, var) vs
    N(0, var)): `ell = (delta/var) y - delta^2/(2 var)`."""
    return float(delta) / max(float(var), 1e-30) * float(y) \
        - float(delta) ** 2 / (2.0 * max(float(var), 1e-30))


def simulate_gaussian_frids(
    delta: np.ndarray,
    sigma: np.ndarray,
    owner_of: list,
    alpha: float = 0.05,
    beta: float = 0.05,
    n_runs: int = 200,
    seed: int = 0,
    max_steps: int = 60,
    mu: float = 0.5,
    eps: float = 0.1,
    value_mode: str = "covariance",
    rho_s: float = 0.0,
    coalition_mode: str = "intent",
    delivery: np.ndarray | None = None,
    reliable: bool = False,
) -> dict:
    """Self-contained Gaussian-evidence FRIDS (G8-C).

    Per cycle every UAV chooses a target by the conditional-information
    index `J = y * value / (D + eps)` (value = singleton, scalar-rho
    common-factor, or covariance-native Schur `Delta G`), draws a
    correlated Gaussian observation, and the owner fuses the DELIVERED
    coalition via the JOINT LLR (drift `G_q(S)`).  The strictly-local
    coalition estimate comes from the received intents (or the true
    previous-cycle set for the oracle).  Stopping is the owner joint
    belief against the Wald two thresholds.

    The world (delivered joint information in the service gap) is always
    the covariance-native `G_q(S_del)`; only the VALUE differs across
    methods, so the comparison isolates the scheduling information.

    ``reliable`` weights the value by the U2U delivery success
    `s_{i, owner_q}` (the FRIDS-v2 "reliable information" convention,
    Theorem 4.94): value = `Delta G * s` (or `g * s` for the singleton).
    """
    k, q = delta.shape
    sigma = np.asarray(sigma, dtype=float)
    dmat = np.asarray(delta, dtype=float)
    deliv = delivery if delivery is not None else np.ones((k, k))
    np.fill_diagonal(deliv, 1.0)
    s_reliable = np.array([[deliv[i, owner_of[qq]] for qq in range(q)]
                           for i in range(k)])
    g_single = dmat ** 2 / 2.0
    if reliable:
        g_single = g_single * s_reliable
    a_thr = np.log((1.0 - beta) / alpha)
    b_thr = np.log(beta / (1.0 - alpha))

    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    sigma_chol = np.linalg.cholesky(sigma)

    for r in range(n_runs):
        H = rng.random(q) < 0.5
        H_all[r] = H
        L = np.zeros((k, q))          # per-UAV local belief
        L_own = np.zeros(q)           # owner joint belief
        decided = np.zeros(q, dtype=bool)
        intents_recv = np.full((k, k), -1, dtype=int)
        intents_all = np.full(k, -1, dtype=int)
        y = np.full((k, q), 1.0 / q)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            D_loc = np.maximum(a_thr - L, 0.0)
            perfect_set = {qq: [uav for uav in range(k)
                                if intents_all[uav] == qq]
                           for qq in range(q)} if coalition_mode == "perfect" \
                else None
            # choice by the conditional-information index
            intents = np.full(k, -1, dtype=int)
            served = {qq: [] for qq in range(q)}
            for uav in range(k):
                best_q = None
                best_g = -np.inf
                for qq in undecided:
                    if dmat[uav, qq] <= 0.0:
                        continue
                    if coalition_mode == "perfect":
                        coalition = perfect_set[qq] + [uav]
                    else:
                        coalition = [int(j) for j in range(k)
                                     if j == uav or int(intents_recv[uav, j])
                                     == qq]
                    if value_mode == "singleton":
                        val = g_single[uav, qq]
                    elif value_mode == "rho":
                        # common-factor conditional gain (the scalar-rho
                        # method): G(S union {i}) - G(S) under the
                        # common-factor joint KL
                        from uav_otfs_isac.evidence_correlation import (
                            joint_kl as rho_joint_kl)
                        idx_wi = sorted(set(coalition) | {uav})
                        idx_wo = sorted(set(coalition) - {uav})
                        val = rho_joint_kl(dmat[idx_wi, qq], rho_s) \
                            - (rho_joint_kl(dmat[idx_wo, qq], rho_s)
                               if idx_wo else 0.0)
                        if reliable:
                            val *= float(s_reliable[uav, qq])
                    elif value_mode in ("covariance", "oracle"):
                        val = schur_conditional_gain(dmat[:, qq], sigma,
                                                     coalition, uav)
                        if reliable:
                            val *= float(s_reliable[uav, qq])
                    else:
                        raise ValueError(f"unknown value_mode {value_mode!r}")
                    if val <= 0.0:
                        continue
                    score = y[uav, qq] * val / (D_loc[uav, qq] + eps)
                    if score > best_g:
                        best_g = score
                        best_q = qq
                if best_q is not None:
                    intents[uav] = int(best_q)
                    served[best_q].append(uav)
            intents_all = intents.copy()
            # correlated observation draws per target (independent across
            # targets; the covariance is within a target's sensing set)
            obs = {}
            for qq in undecided:
                sv = served[qq]
                if not sv:
                    continue
                idx = sorted(sv)
                n = len(idx)
                z = rng.standard_normal(n)
                if H[qq]:
                    obs[qq] = (idx, dmat[idx, qq]
                               + sigma_chol[np.ix_(idx, idx)] @ z)
                else:
                    obs[qq] = (idx, sigma_chol[np.ix_(idx, idx)] @ z)
            # local beliefs + delivery + owner joint fusion
            delivered_to_owner = {}
            for qq in undecided:
                if qq not in obs:
                    continue
                idx, y_obs = obs[qq]
                for pos, uav in enumerate(idx):
                    L[uav, qq] += _marginal_llr(
                        float(y_obs[pos]), float(dmat[uav, qq]),
                        float(sigma[uav, uav]))
                # delivery of the coalition to the owner (one rng draw per
                # link, shared by the fusion AND the service gap)
                d_owner = [uav for uav in idx
                           if rng.random() <= deliv[uav, owner_of[qq]]]
                delivered_to_owner[qq] = d_owner
                if d_owner:
                    d_idx = sorted(d_owner)
                    ell = _gaussian_joint_llr(
                        y_obs[[idx.index(u) for u in d_idx]],
                        dmat[d_idx, qq],
                        sigma[np.ix_(d_idx, d_idx)])
                    L_own[qq] += ell
            # mirror descent on the service gap: each UAV is credited its
            # MARGINAL contribution Delta G_{i|rest} to the delivered
            # coalition (FRIDS-v2 credits the singleton g_i; the
            # covariance-native credit is the Schur residual), so the
            # price dynamics do not fight the value
            S_loc = np.zeros((k, q))
            for qq in undecided:
                d_idx = sorted(delivered_to_owner.get(qq, []))
                if not d_idx:
                    continue
                for pos, uav in enumerate(d_idx):
                    others = [u for u in d_idx if u != uav]
                    S_loc[uav, qq] = schur_conditional_gain(
                        dmat[:, qq], sigma, others, uav)
            for uav in range(k):
                ratio = np.zeros(q)
                for qq in undecided:
                    ratio[qq] = S_loc[uav, qq] / (D_loc[uav, qq] + eps)
                rbar = float(np.mean(ratio[undecided]))
                e = rbar - ratio
                num = y[uav] * np.exp(mu * e)
                y[uav] = num / max(float(np.sum(num)), 1e-12)
            # stopping on the owner joint belief
            for qq in undecided:
                if L_own[qq] >= a_thr:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
                    if H[qq]:
                        declared_h1[r, qq] = 1.0
                elif L_own[qq] <= b_thr:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
            # deliver the intents for the next cycle's coalition estimate
            intents_next = np.full((k, k), -1, dtype=int)
            for qq in undecided:
                if qq not in obs:
                    continue
                idx, _ = obs[qq]
                for uav in idx:
                    for neighbor in range(k):
                        if neighbor == uav:
                            continue
                        if rng.random() <= deliv[uav, neighbor]:
                            intents_next[neighbor, uav] = int(intents[uav])
            intents_recv = intents_next

    e1 = [float(delays[H_all[:, qq], qq].mean()) for qq in range(q)]
    p_fa = [float(declared_h1[~H_all[:, qq], qq].mean()) for qq in range(q)]
    p_md = [float(1.0 - declared_h1[H_all[:, qq], qq].mean())
            for qq in range(q)]
    return {
        "worst_target_delay": float(np.max(e1)),
        "e1_delays": e1,
        "p_fa": p_fa,
        "p_md": p_md,
    }