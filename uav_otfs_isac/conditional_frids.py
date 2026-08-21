"""Gate F0-G8B: Conditional-reliable-information FRIDS (advice/013
section 7, advice/015 section 4).

The FRIDS-v2 scheduling value `g_iq` (the singleton reliable
information) is replaced by the KL-chain-rule conditional marginal

    J_{iq}^{corr} = y_q * Delta G_{i|S_q,q} / (D_q + eps),
    Delta G_{i|S_q,q} = G_q(S_q union {i}) - G_q(S_q),

where `G_q(S)` is the joint detection information of a coalition `S`
under the Gaussian common-factor model
(`evidence_correlation.joint_kl`) and `S_q^{(i)}` is UAV `i`'s STRICTLY
LOCAL coalition estimate for target `q`, inferred only from the intents
it actually received (the P-DIST constraint `a_{i,t} = pi_i(I_{i,t})`).

At `rho_s = 0` the conditional gain equals the singleton `g_iq`, so the
conditional scheduler reduces exactly to FRIDS-v2 (the sanity check).

`world_rho` is the physical correlation used in the DELIVERED reliable
information accounting (the service gap the mirror descent sees): when
`> 0` the per-cycle delivered joint information of the serving coalition
is `G_q(S_cycle)` instead of the singleton sum, so the scheduler stops
over-counting redundant reports.  `rho_s` is the correlation the
scheduler BELIEVES (used in the value).  Step 1 of the gate uses
`world_rho = 0` (independent world) and sweeps `rho_s` to isolate the
allocation effect of the conditional value; Step 2 sets
`rho_s = world_rho` for the consistent correlated world.

Everything else is frozen (FRIDS-v2 mirror descent, policy-matched
thresholds, compact token, fixed owner, full mesh, current scenario
generation).
"""

from __future__ import annotations

import numpy as np

from uav_otfs_isac.distributed_audit import (
    quantize_llr,
    quantize_with,
)
from uav_otfs_isac.evidence_correlation import joint_kl
from uav_otfs_isac.covariance_conditional import (
    joint_kl_equal_cov,
    schur_conditional_gain,
)


def reliable_delta_matrix(scenario: dict, owner_of: list,
                          s_for_g: np.ndarray | None = None) -> np.ndarray:
    """``delta_{iq} = sqrt(2 * g_reliable)`` -- the Gaussian-common-factor
    evidence scale of the RELIABLE information `g_iq = max_a i_plus * s`
    (delivery-inclusive).  Used for the SCHEDULING VALUE (the scheduler
    uses the expected reliable value, matching FRIDS-v2's `g_mat`)."""
    k, q = scenario["k"], scenario["q"]
    g_s = s_for_g if s_for_g is not None else scenario["u2u_success"]
    delta = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            owner = owner_of[qq]
            best = 0.0
            for act in scenario["by_host"][(i, qq)]:
                best = max(best, float(act["i_plus"]) * float(g_s[i, owner]))
            delta[i, qq] = float(np.sqrt(2.0 * max(best, 0.0)))
    return delta


def observation_delta_matrix(scenario: dict) -> np.ndarray:
    """``delta_{iq} = sqrt(2 * max_a i_plus)`` -- the RAW observation
    evidence scale (no delivery factor; the U2U delivery is realized by
    the per-link random draws).  Used for the DELIVERED reliable
    information accounting (matching FRIDS-v2's realized `obs_iplus`)."""
    k, q = scenario["k"], scenario["q"]
    delta = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            best = 0.0
            for act in scenario["by_host"][(i, qq)]:
                best = max(best, float(act["i_plus"]))
            delta[i, qq] = float(np.sqrt(2.0 * max(best, 0.0)))
    return delta


def conditional_gain(delta_col: np.ndarray, others: list, i: int,
                     rho_s: float) -> float:
    """``Delta G_{i|S,q} = G_q(S union {i}) - G_q(S)`` (KL chain rule),
    with `S` the OTHER UAVs already in the coalition.  At `rho_s = 0`
    this equals the singleton `g_i = delta_i^2 / 2` (independence)."""
    i = int(i)
    idx_wi = sorted(set(int(x) for x in others) | {i})
    idx_wo = sorted(set(int(x) for x in others) - {i})
    g_wi = joint_kl(np.asarray(delta_col, dtype=float)[idx_wi], rho_s)
    g_wo = (joint_kl(np.asarray(delta_col, dtype=float)[idx_wo], rho_s)
            if len(idx_wo) > 0 else 0.0)
    return max(0.0, g_wi - g_wo)


def coalition_from_intents(intents_recv: np.ndarray, i: int, q: int) -> list:
    """UAV ``i``'s strictly-local coalition estimate for target ``q``:
    the UAVs whose intents for ``q`` were actually received (1-cycle
    stale, delivery-limited) plus itself."""
    k = intents_recv.shape[0]
    return [int(j) for j in range(k)
            if j == i or int(intents_recv[i, j]) == int(q)]


def simulate_frids_v2_cond(
    scenario: dict,
    bounds: list,
    n_runs: int = 200,
    seed: int = 0,
    max_steps: int = 40,
    alpha: float = 0.05,
    beta: float = 0.05,
    mu: float = 0.5,
    eps: float = 0.1,
    quantizer: dict | None = None,
    delivery_matrix: np.ndarray | None = None,
    s_for_g: np.ndarray | None = None,
    rho_s: float = 0.0,
    world_rho: float = 0.0,
    coalition_mode: str = "intent",
    covariance: dict | None = None,
    world_covariance: dict | None = None,
) -> dict:
    """Conditional-reliable-information FRIDS (G8-B) / covariance-native
    conditional FRIDS (G8-C).

    The scheduling value uses the conditional marginal ``Delta G`` given
    the local coalition estimate; the delivered reliable information
    accounting uses the joint ``G_q(S_cycle)`` when ``world_rho > 0`` or
    ``world_covariance`` is set.  With ``rho_s = 0, world_rho = 0`` the
    decisions equal FRIDS-v2.

    ``covariance`` (dict of per-target `K x K` matrices) switches the
    value to the COVARIANCE-NATIVE Schur form
    ``Delta G = (1/2) delta_{i|S}^2 / v_{i|S}`` (Theorem 4.108) instead
    of the scalar-rho common-factor form.  ``world_covariance`` switches
    the service accounting to the covariance-native joint KL
    ``G_q(S_received)``.

    ``coalition_mode`` controls the local coalition estimate:
    ``"intent"`` uses the 1-cycle-stale, delivery-limited intent map
    (the deployable, strictly-local rule); ``"perfect"`` uses the true
    previous-cycle serving set (an offline oracle -- the audit compares
    the two to quantify the staleness/delivery cost).
    """
    k = scenario["k"]
    q = scenario["q"]
    owner_of = scenario["owner_of"]
    u2u = scenario["u2u_success"]
    delivery = delivery_matrix if delivery_matrix is not None else u2u
    g_s = s_for_g if s_for_g is not None else u2u
    delta = reliable_delta_matrix(scenario, owner_of, s_for_g)
    delta_raw = observation_delta_matrix(scenario)
    # singleton reliable information (for the rho=0 sanity and the
    # value at rho_s = 0)
    g_mat = delta ** 2 / 2.0
    a_thr = np.array([float(bounds[qq][0]) for qq in range(q)])

    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))

    for r in range(n_runs):
        H = rng.random(q) < 0.5
        H_all[r] = H
        L = np.zeros((k, q))
        decided = np.zeros(q, dtype=bool)
        intents_recv = np.full((k, k), -1, dtype=int)
        intents_all = np.full(k, -1, dtype=int)
        y = np.full((k, q), 1.0 / q)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            D_loc = np.maximum(a_thr[None, :] - L, 0.0)
            # true previous-cycle serving set (perfect-coalition oracle)
            perfect_set = {qq: [uav for uav in range(k)
                                if intents_all[uav] == qq]
                           for qq in range(q)} if coalition_mode == "perfect" \
                else None
            choices = [None] * k
            intents = np.full(k, -1, dtype=int)
            for uav in range(k):
                best_q = None
                best_g = -np.inf
                for qq in undecided:
                    if delta[uav, qq] <= 0.0:
                        continue
                    # strictly-local coalition estimate
                    if coalition_mode == "perfect":
                        coalition = perfect_set[qq] + [uav]
                    else:
                        coalition = coalition_from_intents(intents_recv,
                                                           uav, qq)
                    if covariance is not None:
                        sig = covariance.get(qq, covariance)
                        dg = schur_conditional_gain(delta[:, qq], sig,
                                                    coalition, uav)
                    else:
                        dg = conditional_gain(delta[:, qq], coalition, uav,
                                              rho_s)
                    if dg <= 0.0:
                        continue
                    score = y[uav, qq] * dg / (D_loc[uav, qq] + eps)
                    if score > best_g:
                        best_g = score
                        best_q = qq
                if best_q is not None:
                    owner = owner_of[best_q]
                    rel = (1.0 if uav == owner else float(u2u[uav, owner]))
                    best_act = None
                    best_i = -np.inf
                    for act in scenario["by_host"][(uav, best_q)]:
                        v = float(act["i_plus"]) * rel
                        if v > best_i:
                            best_i = v
                            best_act = act
                    choices[uav] = (int(best_q), best_act)
                    intents[uav] = int(best_q)
            intents_all = intents.copy()
            # sensing (frozen)
            obs_target = np.full(k, -1, dtype=int)
            obs_llr = np.zeros(k)
            obs_iplus = np.zeros(k)
            for uav in range(k):
                choice = choices[uav]
                if choice is None or choice[1] is None:
                    continue
                qq, act = choice
                p_obs = act["p1"] if H[qq] else act["p0"]
                y_obs = int(rng.choice(len(p_obs), p=p_obs))
                llr_obs = (quantize_with(quantizer, float(act["llr"][y_obs]))
                           if quantizer is not None
                           else quantize_llr(float(act["llr"][y_obs])))
                obs_target[uav] = qq
                obs_llr[uav] = llr_obs
                obs_iplus[uav] = float(act["i_plus"])
                L[uav, qq] += llr_obs
            # token exchange with per-receiver delivery; the delivered
            # reliable information to each receiver is the joint
            # G_q(S_received) when the world is correlated (world_rho>0)
            S_loc = np.zeros((k, q))
            recv_sets = {qq: {uav: [] for uav in range(k)}
                         for qq in range(q)}
            intents_next = np.full((k, k), -1, dtype=int)
            for uav in range(k):
                qq = int(obs_target[uav])
                if qq < 0:
                    continue
                for neighbor in range(k):
                    if neighbor == uav:
                        continue
                    if rng.random() > delivery[uav, neighbor]:
                        continue
                    if not decided[qq]:
                        L[neighbor, qq] += obs_llr[uav]
                        recv_sets[qq][neighbor].append(uav)
                    intents_next[neighbor, uav] = int(intents[uav])
            intents_recv = intents_next
            for qq in undecided:
                for uav in range(k):
                    sv = sorted(set(recv_sets[qq][uav]))
                    if not sv:
                        continue
                    if world_covariance is not None:
                        sig = world_covariance.get(qq, world_covariance)
                        S_loc[uav, qq] = joint_kl_equal_cov(
                            delta_raw[sv, qq],
                            sig[np.ix_(sv, sv)])
                    elif world_rho > 0.0:
                        S_loc[uav, qq] = joint_kl(
                            delta_raw[sv, qq], world_rho)
                    else:
                        S_loc[uav, qq] = float(
                            np.sum(delta_raw[sv, qq] ** 2) / 2.0)
            # mirror descent per UAV on the normalized service gap
            for uav in range(k):
                ratio = np.zeros(q)
                for qq in undecided:
                    ratio[qq] = S_loc[uav, qq] / (D_loc[uav, qq] + eps)
                rbar = float(np.mean(ratio[undecided]))
                e = rbar - ratio
                num = y[uav] * np.exp(mu * e)
                y[uav] = num / max(float(np.sum(num)), 1e-12)
            # stopping on the owner belief
            for qq in undecided:
                l_own = L[owner_of[qq], qq]
                if l_own >= bounds[qq][0]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
                    if H[qq]:
                        declared_h1[r, qq] = 1.0
                elif l_own <= bounds[qq][1]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)

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