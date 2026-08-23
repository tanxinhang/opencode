"""FRIDS: Feasibility-Aware Reliable-Information Dual Scheduling
(advice/009 Gate F0-G3).

The heuristic ``normalized dual-G + congestion price`` is replaced by a
scheduler derived from the min-max detection-deficit relaxation:

    max z  s.t.  sum_i x_iq g_iq >= z D_q,  sum_q x_iq <= 1,

with ``D_q`` the residual detection deficit and ``g_iq`` the reliable
post-communication marginal detection information.  The Lagrangian gives
the local index

    J_iq(t) = nu_q(t) * g_iq(t),

where ``g_iq = max_a I+_{iqa}^post * s_{i, owner_q}`` (evidence quality
AND delivery reliability enter the same marginal value -- sensing and
communication are not stitched afterwards), and the target price is a
projected subgradient on the simplex,

    nu_q(t+1) = Pi_Delta[ nu_q(t) + mu ( Dbar_q - Sbar_q ) ],

with ``Dbar_q`` the smoothed residual deficit ``[A_q - L_own,q]_+`` and
``Sbar_q`` the smoothed reliable information actually received.  The
price therefore rises for starved targets and falls for well-served ones
-- starvation correction + bottleneck focusing + load balancing, without
any scale-fitted parameter (no eta(K)).

The feasibility certificate prevents resource sinks: if the remaining
horizon times the max reliable information rate cannot close the
information deficit, the target is flagged currently-infeasible and its
price is capped (it is served at the capped price, but cannot drag the
other targets' prices to infinity).

All other mechanisms are frozen (fixed owner, full mesh, 19-bit token,
communication-domain beliefs, calibrated two-threshold stopping, current
scenario generation).  Only the action-selection index changes.
"""

from __future__ import annotations

import numpy as np

from uav_otfs_isac.crn_tape import ExogenousTape, draw_atom
from uav_otfs_isac.distributed_audit import (
    calibrate_target_bounds,
    quantize_llr,
    quantize_with,
    uniform_quantizer,
)
from uav_otfs_isac.difficulty_decomposition import d_kl_binary


def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    rho = int(np.max(np.where(u - css / ind > 0)[0]))
    theta = css[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def simplex_projection_lb(v: np.ndarray, lb: float) -> np.ndarray:
    """Projection onto the simplex with a lower bound per coordinate:
    ``{x : sum x = 1, x >= lb}``.  The floor is the dual lower bound of
    the min-max relaxation -- it guarantees every undecided target keeps
    a baseline price share, which is what prevents starvation-driven
    censored misses in the deployed runs."""
    n = len(v)
    if lb * n >= 1.0:
        return np.full(n, 1.0 / n)
    w = np.maximum(v - lb, 0.0)
    slack = 1.0 - lb * n
    if np.sum(w) <= slack + 1e-12:
        scale = slack / max(np.sum(w), 1e-12)
        return lb + w * scale
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - slack
    ind = np.arange(1, n + 1)
    rho = int(np.max(np.where(u - css / ind > 0)[0]))
    theta = css[rho] / (rho + 1)
    return lb + np.maximum(w - theta, 0.0)


def g_reliable(scenario: dict, uav: int, q: int, owner_of: list) -> float:
    """Reliable post-communication marginal detection information of UAV
    ``uav`` about target ``q``: the best kernel's I+ scaled by the
    delivery success into the owner's belief (owner itself: success 1)."""
    owner = owner_of[q]
    best = 0.0
    for act in scenario["by_host"][(uav, q)]:
        rel = float(scenario["u2u_success"][uav, owner])
        best = max(best, float(act["i_plus"]) * rel)
    return best


def capacity(scenario: dict, q: int, owner_of: list) -> float:
    """Max reliable information rate of target q (sum over UAVs)."""
    return float(sum(g_reliable(scenario, i, q, owner_of)
                     for i in range(scenario["k"])))


def simulate_frids(
    scenario: dict,
    bounds: list,
    n_runs: int = 250,
    seed: int = 0,
    max_steps: int = 40,
    alpha: float = 0.05,
    beta: float = 0.05,
    mu: float = 0.2,
    ema: float = 0.5,
    nu_cap: float = 0.5,
    nu_floor: float = 0.0,
    quantizer: dict | None = None,
    b_cycle: float = 8.0,
) -> dict:
    """MC run of the FRIDS-v1 mainline (compact-token machinery, FRIDS
    action index, owner-belief deficit).  ``nu_floor`` is the dual lower
    bound per undecided target (anti-starvation price floor; default 0).
    Returns delays, realized errors, and the feasibility audit
    (infeasible-target statistics)."""
    k = scenario["k"]
    q = scenario["q"]
    owner_of = scenario["owner_of"]
    u2u = scenario["u2u_success"]
    cap = {qq: capacity(scenario, qq, owner_of) for qq in range(q)}
    g_mat = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            g_mat[i, qq] = g_reliable(scenario, i, qq, owner_of)
    info_floor = float(d_kl_binary(1.0 - beta, alpha))

    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    infeasible_cycles = np.zeros((n_runs, q))
    feasible = np.ones((n_runs, q))

    for r in range(n_runs):
        H = rng.random(q) < 0.5
        H_all[r] = H
        L = np.zeros((k, q))
        decided = np.zeros(q, dtype=bool)
        intents_recv = np.full((k, k), -1, dtype=int)
        nu = np.full(q, 1.0 / q)
        db = np.zeros(q)          # smoothed residual deficit
        sb = np.zeros(q)          # smoothed received reliable info
        i_acc = np.zeros(q)       # accumulated reliable info (delivered)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            # feasibility certificate per undecided target
            for qq in undecided:
                d_info = max(info_floor - i_acc[qq], 0.0)
                h_rem = max(max_steps - t, 1)
                if h_rem * cap[qq] < d_info:
                    infeasible_cycles[r, qq] += 1.0
                    feasible[r, qq] = 0.0
                    nu[qq] = min(nu[qq], nu_cap)   # cap, don't sink
            # action selection: argmax_q nu_q * g_iq over own kernels
            choices = [None] * k
            intents = np.full(k, -1, dtype=int)
            for uav in range(k):
                best_q = None
                best_g = -np.inf
                for qq in undecided:
                    if g_mat[uav, qq] <= 0.0:
                        continue
                    score = nu[qq] * g_mat[uav, qq]
                    if score > best_g:
                        best_g = score
                        best_q = qq
                if best_q is not None:
                    best_act = None
                    best_i = -np.inf
                    owner = owner_of[best_q]
                    rel = (1.0 if uav == owner
                           else float(u2u[uav, owner]))
                    for act in scenario["by_host"][(uav, best_q)]:
                        v = float(act["i_plus"]) * rel
                        if v > best_i:
                            best_i = v
                            best_act = act
                    choices[uav] = (int(best_q), best_act)
                    intents[uav] = int(best_q)
            # sensing + tokens (compact-token machinery)
            obs_target = np.full(k, -1, dtype=int)
            obs_llr = np.zeros(k)
            obs_iplus = np.zeros(k)
            for uav in range(k):
                choice = choices[uav]
                if choice is None or choice[1] is None:
                    continue
                qq, act = choice
                p = act["p1"] if H[qq] else act["p0"]
                y_obs = int(rng.choice(len(p), p=p))
                llr_obs = (quantize_with(quantizer,
                                         float(act["llr"][y_obs]))
                           if quantizer is not None
                           else quantize_llr(float(act["llr"][y_obs])))
                obs_target[uav] = qq
                obs_llr[uav] = llr_obs
                obs_iplus[uav] = float(act["i_plus"])
                L[uav, qq] += llr_obs
            # token exchange (compact-token machinery); s_cycle
            # accumulates the reliable info actually delivered to each
            # owner this cycle
            s_cycle = np.zeros(q)
            intents_next = np.full((k, k), -1, dtype=int)
            for uav in range(k):
                qq = int(obs_target[uav])
                if qq < 0:
                    continue
                for neighbor in range(k):
                    if neighbor == uav:
                        continue
                    if rng.random() > u2u[uav, neighbor]:
                        continue
                    if not decided[qq]:
                        L[neighbor, qq] += obs_llr[uav]
                        if neighbor == owner_of[qq]:
                            i_acc[qq] += obs_iplus[uav]
                            s_cycle[qq] += obs_iplus[uav]
                    intents_next[neighbor, uav] = int(intents[uav])
            intents_recv = intents_next
            # price update (projected subgradient on the simplex, with
            # smoothed deficit and smoothed received info); decided
            # targets are excluded from the simplex (their price cannot
            # consume probability mass from the undecided ones)
            for qq in undecided:
                d_here = max(bounds[qq][0] - L[owner_of[qq], qq], 0.0)
                db[qq] = ema * d_here + (1.0 - ema) * db[qq]
                sb[qq] = ema * s_cycle[qq] + (1.0 - ema) * sb[qq]
            nu_full = np.zeros(q)
            nu_full[undecided] = simplex_projection_lb(
                nu[undecided] + mu * (db[undecided] - sb[undecided]),
                nu_floor)
            nu = nu_full
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
    active = np.maximum((delays < max_steps).sum(axis=0), 1)
    return {
        "worst_target_delay": float(np.max(e1)),
        "e1_delays": e1,
        "p_fa": p_fa,
        "p_md": p_md,
        "infeasible_cycle_fraction": [
            float(np.mean(infeasible_cycles[:, qq] / active[qq]))
            for qq in range(q)],
        "infeasible_target_fraction": [float(np.mean(1.0 - feasible[:, qq]))
                                       for qq in range(q)],
    }


def load_cut(scenario: dict, owner_of: list, subset, horizon: int,
             beta: float = 0.05, alpha: float = 0.05,
             info_floor: float | None = None) -> float:
    """Information-load cut of a target subset S (advice/010 section 5,
    Theorem 4): ``rho(S) = sum_{q in S} D_q^info / (H * sum_i max_{q in S}
    g_iq)`` with ``D_q^info`` the residual information deficit at t=0
    (``d(1-beta||alpha)``, no evidence accumulated yet) and the
    denominator the max reliable information the UAVs can deliver to the
    subset per cycle (each UAV serves the subset's best target).  If
    ``rho(S) > 1``, no allocation can finish all targets of S within the
    horizon -- a necessary infeasibility condition that turns the
    empirical Q/K feasibility region into a theory-grounded boundary."""
    if info_floor is None:
        info_floor = float(d_kl_binary(1.0 - beta, alpha))
    num = len(subset) * info_floor
    den = 0.0
    for i in range(scenario["k"]):
        den += max(g_reliable(scenario, i, qq, owner_of)
                   for qq in subset)
    return float(num / (horizon * max(den, 1e-12)))


def _audit_cycle(undecided, y, y_common, D_loc, L, g_mat, eps, owner_of,
                 a_thr, aud, price_mode="local"):
    """F0-G9A per-cycle diagnostics (advice/020): local price
    disagreement, normalized-value disagreement, owner-vs-local deficit
    gap, and the distributed action-invariance certificate fraction
    ``P(m_i > 2 E_i)`` (Theorem 4.109).  ``eps_y`` is the ACTION-relevant
    price error: ``|y^{(i)} - y_common|`` in the local mode, exactly zero
    in the common-price oracle mode (the action then uses the common
    price)."""
    k, q = y.shape
    if not undecided:
        return
    # D_y: mean over UAV pairs of the L1 price disagreement
    d_y, cnt = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            d_y += float(np.sum(np.abs(y[i] - y[j])))
            cnt += 1
    aud["d_y"] += d_y / max(cnt, 1)
    # owner-anchored deficit (the stopping decision state)
    d_own = np.maximum(
        a_thr - np.array([L[owner_of[qq], qq] for qq in range(q)]), 0.0)
    # per-UAV local vs ideal value errors and the deficit gap
    eps_y = np.zeros(k)
    eps_v = np.zeros(k)
    v_max = 0.0
    gap, gcnt = 0.0, 0
    for i in range(k):
        for qq in undecided:
            if g_mat[i, qq] <= 0.0:
                continue
            v_hat = g_mat[i, qq] / (D_loc[i, qq] + eps)
            v_ideal = g_mat[i, qq] / (d_own[qq] + eps)
            v_max = max(v_max, float(v_ideal))
            if price_mode == "common":
                eps_y[i] = 0.0
            else:
                eps_y[i] = max(eps_y[i],
                               abs(float(y[i, qq] - y_common[qq])))
            eps_v[i] = max(eps_v[i], abs(float(v_hat - v_ideal)))
            gap += abs(float(d_own[qq] - D_loc[i, qq]))
            gcnt += 1
    # D_v: mean over (UAV pairs, undecided targets) of the local
    # normalized-value disagreement
    d_v, vcnt = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            for qq in undecided:
                if g_mat[i, qq] <= 0.0 or g_mat[j, qq] <= 0.0:
                    continue
                d_v += abs(g_mat[i, qq] / (D_loc[i, qq] + eps)
                           - g_mat[j, qq] / (D_loc[j, qq] + eps))
                vcnt += 1
    aud["d_v"] += d_v / max(vcnt, 1)
    aud["deficit_gap"] += gap / max(gcnt, 1)
    aud["eps_y"] = max(aud["eps_y"], float(np.max(eps_y)))
    aud["eps_v"] = max(aud["eps_v"], float(np.max(eps_v)))
    aud["v_max"] = max(aud["v_max"], float(v_max))
    # action-invariance certificate: ideal margin m_i vs 2 E_i
    for i in range(k):
        ideal = {}
        local = {}
        for qq in undecided:
            if g_mat[i, qq] <= 0.0:
                continue
            ideal[qq] = (y_common[qq] * g_mat[i, qq]
                         / (d_own[qq] + eps))
            local[qq] = (y[i, qq] * g_mat[i, qq]
                         / (D_loc[i, qq] + eps))
        if len(ideal) < 2:
            continue
        qs = sorted(ideal, key=ideal.get, reverse=True)
        m_i = float(ideal[qs[0]] - ideal[qs[1]])
        e_i = v_max * eps_y[i] + eps_v[i] + eps_y[i] * eps_v[i]
        aud["margin_total"] += 1.0
        if m_i > 2.0 * e_i:
            aud["margin_ok"] += 1.0
        # realized action change decomposition: local argmax vs (i) the
        # common-price-only ideal (y_common, D_loc) -- isolates the PRICE
        # disagreement; (ii) the owner-deficit-only ideal (y_i, D_own) --
        # isolates the DEFICIT disagreement; (iii) both (y_common, D_own)
        if max(ideal, key=ideal.get) != max(local, key=local.get):
            aud["action_change"] += 1.0
        aud["action_total"] += 1.0
        price_ideal = {qq: y_common[qq] * g_mat[i, qq]
                       / (D_loc[i, qq] + eps) for qq in ideal}
        deficit_ideal = {qq: y[i, qq] * g_mat[i, qq]
                         / (d_own[qq] + eps) for qq in ideal}
        if max(price_ideal, key=price_ideal.get) \
                != max(local, key=local.get):
            aud["action_change_price"] += 1.0
        if max(deficit_ideal, key=deficit_ideal.get) \
                != max(local, key=local.get):
            aud["action_change_deficit"] += 1.0
        aud["action_decomp_total"] += 1.0
    aud["n_cycles"] += 1.0


def simulate_frids_v2(
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
    price_mode: str = "local",
    audit: bool = False,
    mobility: float | None = None,
    power_cap: np.ndarray | None = None,
    bridge: bool = False,
    rx_cap_tokens: np.ndarray | None = None,
    raw_counts: bool = False,
    exog: ExogenousTape | None = None,
) -> dict:
    """FRIDS-v2 (advice/010): demand-normalized primal + dual-consistent
    mirror descent, strictly distributed.

    Every UAV i maintains ITS OWN price vector ``y^{(i)}`` (its own dual
    estimate on the ordinary simplex), its own residual deficit
    ``D^{(i)}_q = [A_q* - L_{i,q}]_+`` from its own local belief, and its
    own received reliable information ``S^{(i)}_q`` (what was delivered
    to IT) -- no access to the owner belief or any other global quantity
    (provenance audit, P-DIST ``a_i = pi_i(I_{i,t})``).  The local score
    is the demand-normalized index

        J_{iq} = y^{(i)}_q * g_{iq} / (D^{(i)}_q + eps),

    dual-consistent with the primal ``max z s.t. sum x g / D_q >= z``
    (the dual weight lives on the ordinary simplex).  The price update
    is exponentiated-gradient (mirror descent) on the normalized service
    gap ``e_q = rbar - S_q/(D_q + eps)`` with ``rbar`` the average
    normalized service; no ``nu_floor`` is needed (y > 0 by
    construction).

    Robust-FRIDS (advice/011): ``s_for_g`` is the reliability matrix the
    scheduler BELIEVES (used inside ``g``), while ``delivery_matrix`` is
    the TRUE matrix used for the delivery draws.  The robust variant
    passes ``s_for_g = max(0, s_hat - delta)`` (worst point of the
    interval uncertainty set; exact because ``g(s) = s I+`` is monotone
    linear in ``s``).  With both ``None`` the two coincide (nominal).

    Local-Dual Consistency Audit (advice/020, Gate F0-G9A):
    ``price_mode`` selects the price used in the action index --
    ``"local"`` (the frozen per-UAV price ``y^{(i)}``) or ``"common"``
    (the offline oracle price ``y = mean_i y^{(i)}``, everything else
    stays local/unchanged -- the minimal oracle that isolates the cost
    of local dual disagreement).  ``audit`` additionally tracks, per
    cycle, the local price disagreement ``D_y``, the normalized-value
    disagreement ``D_v``, the ideal action margin ``m_i`` vs the
    distributed action-invariance error bound ``E_i`` (Theorem 4.109),
    and the owner-vs-local deficit disagreement (the token-age bound).
    The default path (``price_mode="local", audit=False``) is
    byte-identical to the frozen FRIDS-v2.

    Frozen-policy mobility stress (advice/022 section 5): ``mobility``
    (relative per-cycle evidence change bound, e.g. 0.1 = +/- 10%) adds a
    bounded random walk to every (UAV, target) reliable information
    `g_iq` and scales the observation LLR atoms by the same factor -- the
    UAVs move but the FRIDS policy is unchanged.  Default ``None`` keeps
    the static-geometry behavior.

    Sensing-resource audit (advice/024, G10): ``power_cap`` (per-target
    max sensing power) restricts every UAV's kernel to powers at or below
    the cap; the energy-conserving oracle sets a high cap on the weak
    target and a low cap elsewhere (same total sensing energy) to measure
    how much of the +4 dB sensing headroom is re-allocatable.  Default
    ``None`` keeps the frozen behavior (the best kernel of every power).

    Service-delay bridge (advice/003, P1): ``bridge=True`` records, per
    run and target, the owner LLR trajectory ``L_q(t)``, the cumulative
    reliable service ``A_q(t)`` (delivered ``i_plus``), the cumulative
    LLR variance proxy ``V_q(t) ~ 2 A_q(t)``, the normalized service
    ``r_q(t)`` and the stopping time ``T_q``, so the LLR decomposition
    ``L_q = L_q(0) + A_q + M_q`` (M a martingale) and the Freedman-type
    stopping tail bound (Theorem 4.110) can be verified numerically.
    Default ``False`` keeps the output identical.

    Hard receiver admission (P2.1-1, advice/006 section 6):
    ``rx_cap_tokens`` (per-UAV, token units) turns the receive side into a
    HARD per-receiver, per-cycle budget: physical Bernoulli offers first,
    then an overloaded receiver admits ``floor(cap_i)`` tokens uniformly
    without replacement plus one more with the fractional-surplus
    probability, so PATHWISE ``sum_{j!=i} b_tok z_{ji,t} <= ceil(cap_i)``
    and ``E[N_adm] = B_rx / b_tok`` -- ``rho_C > 1`` becomes real
    contention -> token loss -> ``g^eff`` -> ``T_q``.  Default ``None``
    keeps the frozen independent-delivery path byte-identical.
    """
    k = scenario["k"]
    q = scenario["q"]
    owner_of = scenario["owner_of"]
    u2u = scenario["u2u_success"]
    delivery = delivery_matrix if delivery_matrix is not None else u2u
    g_s = s_for_g if s_for_g is not None else u2u
    g_mat = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            owner = owner_of[qq]
            best = 0.0
            for act in scenario["by_host"][(i, qq)]:
                if power_cap is not None and float(act["power"]) \
                        > float(power_cap[qq]):
                    continue
                best = max(best, float(act["i_plus"])
                           * float(g_s[i, owner]))
            g_mat[i, qq] = best

    # bridge (advice/003 P1): per-rule H1 moments of the deployed
    # quantized atoms and the TRUE delivery reliability, so the
    # predictable drift A_q(t) = sum_tau sum_i x_iq E_1[Z|F] is exact
    # (the martingale M_q = L_q - A_q is zero-mean by construction; the
    # theory does not require a Gaussian shortcut -- the increments are
    # bounded by the finite alphabet, b_q, and concentration is the
    # Freedman-type martingale bound).
    b_llr = np.zeros((k, q))           # max |atom| of the chosen kernel
    mu_llr = np.zeros((k, q))          # E_1[quantized atom]
    v_llr = np.zeros((k, q))           # Var_1[quantized atom]
    # analytic deterministic V upper bound (advice/005 section 3): the
    # ``v`` side of the Freedman joint event is PRE-REGISTERED as
    # ``V_q(t) <= t * sum_i max_a [ s sigma^2 + s (1-s) g~^2 ]`` (per-UAV
    # max over the quantized-kernel atoms), NOT the sample-max of the
    # audit MC draws.
    v2_max = np.zeros((k, q))          # max_a Var_1[Q(atom)]
    mu2_max = np.zeros((k, q))         # max_a (E_1[Q(atom)])^2
    rel_mat = np.zeros((k, q))         # TRUE delivery to the owner
    v_up_analytic = np.zeros(q)        # nats^2/cycle upper bound per target
    if bridge:
        for i in range(k):
            for qq in range(q):
                owner = owner_of[qq]
                rel = 1.0 if i == owner else float(delivery[i, owner])
                rel_mat[i, qq] = rel
                best_act = None
                best_v = -np.inf
                for act in scenario["by_host"][(i, qq)]:
                    if power_cap is not None and float(act["power"]) \
                            > float(power_cap[qq]):
                        continue
                    v = float(act["i_plus"]) * rel
                    # analytic bound needs the max over actions of the
                    # deployed-atom variance and drift^2
                    atoms_a = np.array([
                        (quantize_with(quantizer, float(x))
                         if quantizer is not None else quantize_llr(float(x)))
                        for x in act["llr"]
                    ])
                    p1a = np.asarray(act["p1"], dtype=float)
                    mua = float(np.sum(p1a * atoms_a))
                    vva = float(np.sum(p1a * atoms_a * atoms_a)) - mua * mua
                    mu2_max[i, qq] = max(mu2_max[i, qq], mua * mua)
                    v2_max[i, qq] = max(v2_max[i, qq], max(vva, 0.0))
                    if v > best_v:
                        best_v = v
                        best_act = act
                if best_act is None:
                    continue
                atoms = np.array([
                    (quantize_with(quantizer, float(x))
                     if quantizer is not None else quantize_llr(float(x)))
                    for x in best_act["llr"]
                ])
                p1 = np.asarray(best_act["p1"], dtype=float)
                mu = float(np.sum(p1 * atoms))
                vv = float(np.sum(p1 * atoms * atoms)) - mu * mu
                mu_llr[i, qq] = mu
                v_llr[i, qq] = max(vv, 0.0)
                b_llr[i, qq] = float(np.max(np.abs(atoms)))
        # PRE-REGISTERED per-target per-cycle variance upper bound
        # (advice/005 section 3): ``V_q(t) <= t * sum_i max_a [ s sigma^2
        # + s(1-s) g~^2 ]`` (each UAV serves <= 1 target/cycle, so the
        # conditional variance of the Bernoulli-scaled deliverable atom
        # is bounded by the action-max; the v grid is fixed before the
        # audit MC draws, not derived from their max).
        for qq in range(q):
            v_up_analytic[qq] = float(sum(
                rel_mat[i, qq] * v2_max[i, qq]
                + rel_mat[i, qq] * (1.0 - rel_mat[i, qq]) * mu2_max[i, qq]
                for i in range(k)))
    # realized reliable info delivered to the owner (own observation
    # counts, token deliveries count), the i_plus-based counterpart of S
    decided_upper = np.zeros((n_runs, q)) if raw_counts else None
    if bridge:
        br = {
            "L": np.zeros((n_runs, max_steps, q)),
            "A": np.zeros((n_runs, max_steps, q)),
            "A_raw": np.zeros((n_runs, max_steps, q)),
            "V": np.zeros((n_runs, max_steps, q)),
            "M": np.zeros((n_runs, max_steps, q)),
            "S": np.zeros((n_runs, max_steps, q)),
            "r_pred": np.zeros((n_runs, max_steps, q)),
            "r_real": np.zeros((n_runs, max_steps, q)),
            "n_served": np.zeros((n_runs, max_steps, q)),
            "delivery_matrix": delivery,
        }
        br["a_thr"] = np.array([float(bounds[qq][0]) for qq in range(q)])

    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    infeasible_cycles = np.zeros((n_runs, q))
    # audit accumulators (Gate F0-G9A)
    aud = {
        "d_y": 0.0, "d_v": 0.0, "deficit_gap": 0.0,
        "margin_ok": 0.0, "margin_total": 0.0,
        "action_change": 0.0, "action_total": 0.0,
        "action_change_price": 0.0, "action_change_deficit": 0.0,
        "action_decomp_total": 0.0,
        "eps_y": 0.0, "eps_v": 0.0, "v_max": 0.0,
        "n_cycles": 0.0, "n_deficit": 0.0,
    }
    a_thr = np.array([float(bounds[qq][0]) for qq in range(q)])
    power_used = 0.0
    power_cycles = 0.0

    for r in range(n_runs):
        # CRN tape (advice/010 P0-2): both schedulers read the SAME target
        # presence uniforms; legacy keeps the sequential stream.
        H = (exog.U_H[r] < 0.5) if exog is not None \
            else (rng.random(q) < 0.5)
        H_all[r] = H
        L = np.zeros((k, q))
        decided = np.zeros(q, dtype=bool)
        intents_recv = np.full((k, k), -1, dtype=int)
        y = np.full((k, q), 1.0 / q)          # per-UAV price vector
        mfac = np.ones((k, q)) if mobility is not None else None
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            # bridge: record the owner LLR at cycle start so the realized
            # increment Z_q(t) can be measured
            if bridge:
                l_own_before = np.array(
                    [L[owner_of[qq], qq] for qq in range(q)])
            # frozen-policy mobility: bounded random walk of the evidence
            if mfac is not None:
                m = float(mobility)
                # CRN: the frozen-mobility walk uses the pre-registered
                # tape uniforms when supplied (legacy keeps rng.uniform).
                mfac = np.clip(
                    mfac + ((-m + 2.0 * m * exog.U_mfac[r, t])
                            if exog is not None
                            else rng.uniform(-m, m, (k, q))),
                    1.0 - m, 1.0 + m)
            g_eff = g_mat if mfac is None else g_mat * mfac
            # per-UAV local deficit from its OWN belief
            D_loc = np.maximum(
                np.array([bounds[qq][0] for qq in range(q)])[None, :]
                - L, 0.0)                      # (k, q), strictly local
            # common oracle price (mean of the local prices)
            y_common = np.mean(y, axis=0)
            # action selection: argmax over q of price_iq * g_iq / (D+eps)
            choices = [None] * k
            intents = np.full(k, -1, dtype=int)
            for uav in range(k):
                best_q = None
                best_g = -np.inf
                for qq in undecided:
                    if g_eff[uav, qq] <= 0.0:
                        continue
                    price = (y_common[qq] if price_mode == "common"
                             else y[uav, qq])
                    score = price * g_eff[uav, qq] \
                        / (D_loc[uav, qq] + eps)
                    if score > best_g:
                        best_g = score
                        best_q = qq
                if best_q is not None:
                    owner = owner_of[best_q]
                    rel = (1.0 if uav == owner
                           else float(u2u[uav, owner]))
                    best_act = None
                    best_i = -np.inf
                    for act in scenario["by_host"][(uav, best_q)]:
                        if power_cap is not None and float(act["power"]) \
                                > float(power_cap[best_q]):
                            continue
                        v = float(act["i_plus"]) * rel
                        if v > best_i:
                            best_i = v
                            best_act = act
                    choices[uav] = (int(best_q), best_act)
                    intents[uav] = int(best_q)
            # sensing + tokens (compact-token machinery)
            obs_target = np.full(k, -1, dtype=int)
            obs_llr = np.zeros(k)
            obs_iplus = np.zeros(k)
            if power_cap is not None:
                power_used += sum(float(c[1]["power"]) for c in choices
                                  if c is not None and c[1] is not None)
                power_cycles += 1.0
            for uav in range(k):
                choice = choices[uav]
                if choice is None or choice[1] is None:
                    continue
                qq, act = choice
                p = act["p1"] if H[qq] else act["p0"]
                # CRN: same base observation uniform mapped through whatever
                # kernel this scheduler chose (legacy uses rng.choice).
                y_obs = (
                    draw_atom(p, float(exog.U_obs[r, t, uav, qq]))
                    if exog is not None
                    else int(rng.choice(len(p), p=p))
                )
                llr_obs = (quantize_with(quantizer,
                                         float(act["llr"][y_obs]))
                           if quantizer is not None
                           else quantize_llr(float(act["llr"][y_obs])))
                if mfac is not None:
                    llr_obs *= float(mfac[uav, qq])
                obs_target[uav] = qq
                obs_llr[uav] = llr_obs
                obs_iplus[uav] = float(act["i_plus"])
                L[uav, qq] += llr_obs
            # token exchange: each UAV's received info S^{(i)} is what
            # was delivered to IT (strictly local)
            S_loc = np.zeros((k, q))
            intents_next = np.full((k, k), -1, dtype=int)
            if rx_cap_tokens is None:
                for uav in range(k):
                    qq = int(obs_target[uav])
                    if qq < 0:
                        continue
                    for neighbor in range(k):
                        if neighbor == uav:
                            continue
                        if (exog is not None
                                and exog.U_link[r, t, uav, neighbor]
                                > delivery[uav, neighbor]) or (
                            exog is None
                            and rng.random() > delivery[uav, neighbor]):
                            continue
                        if not decided[qq]:
                            L[neighbor, qq] += obs_llr[uav]
                            S_loc[neighbor, qq] += obs_iplus[uav]
                        intents_next[neighbor, uav] = int(intents[uav])
            else:
                # P2.1-1 (advice/006 section 6): HARD per-receiver,
                # per-cycle receive admission.  Phase 1: physical
                # Bernoulli offers (identical draws).  Phase 2: an
                # overloaded receiver admits ``m_i = floor(cap_i)`` tokens
                # uniformly WITHOUT replacement, plus one more with the
                # fractional surplus probability -- so ``E[N_adm] =
                # B_rx/b_tok`` and ``sum_{j!=i} b_tok z_{ji,t} <=
                # ceil(cap_i)`` pathwise.  This makes ``rho_C > 1`` a real
                # contention -> token loss -> g^eff -> T_q, not a label.
                offer = [[] for _ in range(k)]   # per receiver: senders
                for uav in range(k):
                    qq = int(obs_target[uav])
                    if qq < 0:
                        continue
                    for neighbor in range(k):
                        if neighbor == uav:
                            continue
                        if (exog is not None
                                and exog.U_link[r, t, uav, neighbor]
                                > delivery[uav, neighbor]) or (
                            exog is None
                            and rng.random() > delivery[uav, neighbor]):
                            continue
                        offer[neighbor].append(uav)
                admit = [[] for _ in range(k)]
                for nb in range(k):
                    offered = offer[nb]
                    if not offered:
                        continue
                    cap = float(rx_cap_tokens[nb])
                    m_i = int(np.floor(cap))
                    theta = cap - m_i
                    keep = offered
                    if len(offered) > m_i:
                        # CRN: the m_i subsample is selected by the same
                        # pre-registered admission uniforms when the tape is
                        # present; legacy keeps rng.choice.
                        if exog is not None:
                            order = np.argsort(exog.U_adm[r, t, offered])
                            keep = [offered[i] for i in order[:m_i]]
                        else:
                            idx = rng.choice(len(offered), size=m_i,
                                             replace=False)
                            keep = [offered[i] for i in sorted(idx)]
                    theta_gate = (
                        exog is not None
                        and float(exog.U_adm[r, t, nb]) < theta
                    ) or (exog is None and rng.random() < theta) \
                        and theta > 0.0 and len(offered) > m_i
                    if theta_gate:
                        extra = [u for u in offered if u not in keep]
                        if extra and len(keep) < len(offered):
                            # P1-14 (advice/009 section 14): the fractional
                            # extra slot must be chosen EXCHANGEABLY.
                            # With the tape, the exchangeable pick is the
                            # same pre-registered uniform (legacy rng.choice).
                            if exog is not None:
                                pick = extra[int(exog.U_adm_extra[r, t, nb]
                                                 * len(extra))]
                                pick = int(pick)
                            else:
                                pick = int(rng.choice(extra))
                            keep = keep + [pick]
                    admit[nb] = keep
                for nb in range(k):
                    for uav in admit[nb]:
                        qq = int(obs_target[uav])
                        if qq < 0:
                            continue
                        if not decided[qq]:
                            L[nb, qq] += obs_llr[uav]
                            S_loc[nb, qq] += obs_iplus[uav]
                        intents_next[nb, uav] = int(intents[uav])
            intents_recv = intents_next
            if bridge:
                # P1 service-delay bridge: per-cycle, per-undecided-target
                # the realized owner-LLR increment, the predictable drift
                # (exact quantized-domain conditional mean), the
                # predictable conditional variance (Bernoulli-scaled
                # atoms), the realized i_plus service delivered to the
                # owner (own observation always counts), the scheduler
                # g-bookkeeping (A_raw, what the theory of Theorem 4.95
                # uses), and the normalized service ratios.
                for qq in undecided:
                    owner = owner_of[qq]
                    a_inc = 0.0
                    a_raw_inc = 0.0
                    v_inc = 0.0
                    n_serv = 0.0
                    for uav in range(k):
                        if int(obs_target[uav]) != qq:
                            continue
                        rel = rel_mat[uav, qq]
                        mf = (float(mfac[uav, qq]) if mobility is not None
                              else 1.0)
                        mu = float(mu_llr[uav, qq]) * mf
                        vv = float(v_llr[uav, qq]) * mf * mf
                        a_inc += rel * mu
                        a_raw_inc += float(g_eff[uav, qq])
                        # Var of Bern(rel)*atom under H1
                        v_inc += rel * vv + rel * (1.0 - rel) * mu * mu
                        n_serv += 1.0
                    s_inc = float(S_loc[owner, qq])
                    if int(obs_target[owner]) == qq:
                        s_inc += float(obs_iplus[owner])
                    d_here = max(float(a_thr[qq] - L[owner, qq]), 0.0)
                    br["L"][r, t, qq] = float(L[owner, qq])
                    br["A"][r, t, qq] = (
                        (br["A"][r, t - 1, qq] if t > 0 else 0.0) + a_inc)
                    br["A_raw"][r, t, qq] = (
                        (br["A_raw"][r, t - 1, qq] if t > 0 else 0.0)
                        + a_raw_inc)
                    br["V"][r, t, qq] = (
                        (br["V"][r, t - 1, qq] if t > 0 else 0.0) + v_inc)
                    br["M"][r, t, qq] = (
                        float(L[owner, qq])
                        - (br["A"][r, t - 1, qq] if t > 0 else 0.0)
                        - a_inc)
                    br["S"][r, t, qq] = (
                        (br["S"][r, t - 1, qq] if t > 0 else 0.0) + s_inc)
                    br["r_pred"][r, t, qq] = a_inc / max(d_here + eps, 1e-12)
                    br["r_real"][r, t, qq] = s_inc / max(d_here + eps, 1e-12)
                    br["n_served"][r, t, qq] = n_serv
            # mirror descent per UAV on the normalized service gap
            for uav in range(k):
                ratio = np.zeros(q)
                for qq in undecided:
                    ratio[qq] = S_loc[uav, qq] \
                        / (D_loc[uav, qq] + eps)
                rbar = float(np.mean(ratio[undecided]))
                e = rbar - ratio
                num = y[uav] * np.exp(mu * e)
                y[uav] = num / max(np.sum(num), 1e-12)
            if audit:
                _audit_cycle(undecided, y, y_common, D_loc, L, g_mat, eps,
                             owner_of, a_thr, aud, price_mode)
            # stopping on the owner belief (the system detection rule)
            for qq in undecided:
                l_own = L[owner_of[qq], qq]
                if l_own >= bounds[qq][0]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
                    if H[qq]:
                        declared_h1[r, qq] = 1.0
                    if raw_counts:
                        decided_upper[r, qq] = 1.0
                elif l_own <= bounds[qq][1]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
            # bridge (advice/004 P0.5-1): fill-forward the recorded
            # processes after the target stops, so the audit verifies the
            # standard STOPPED process ``M_{t wedge T}`` (M_t stays at
            # M_T after T) rather than a process that abruptly returns to
            # zero -- optional-stopping form of the martingale claim.
            if bridge:
                for qq in range(q):
                    tt = int(delays[r, qq])
                    if tt > max_steps:
                        continue
                    last = max(tt - 1, 0)
                    for name in ("L", "A", "A_raw", "V", "M", "S",
                                 "r_pred", "r_real", "n_served"):
                        val = float(br[name][r, last, qq])
                        for tt_ in range(tt, max_steps):
                            br[name][r, tt_, qq] = val

    e1 = [float(delays[H_all[:, qq], qq].mean()) for qq in range(q)]
    p_fa = [float(declared_h1[~H_all[:, qq], qq].mean()) for qq in range(q)]
    p_md = [float(1.0 - declared_h1[H_all[:, qq], qq].mean())
            for qq in range(q)]
    out = {
        "worst_target_delay": float(np.max(e1)),
        "e1_delays": e1,
        "p_fa": p_fa,
        "p_md": p_md,
        "infeasible_cycle_fraction": [0.0] * q,
    }
    # pooled-J (advice/010 P0-5): raw per-target H1 delay statistics pooled
    # across ALL runs of this cell, so a geometry can compute the pooled
    # worst-target E[T_q | H1] = max_q sum_h1_delay[q]/n_h1[q] without the
    # per-run worst-then-average selection bias (E[max_q hatT_q] >=
    # max_q E[hatT_q]).
    out["pool"] = {
        "n_h1": [int(np.sum(H_all[:, qq])) for qq in range(q)],
        "sum_h1_delay": [float(np.sum(delays[H_all[:, qq], qq]))
                         for qq in range(q)],
        "sum2_h1_delay": [float(np.sum(delays[H_all[:, qq], qq] ** 2))
                          for qq in range(q)],
    }
    if audit:
        out["audit"] = {
            "d_y": float(aud["d_y"] / max(aud["n_cycles"], 1)),
            "d_v": float(aud["d_v"] / max(aud["n_cycles"], 1)),
            "deficit_gap": float(aud["deficit_gap"]
                                 / max(aud["n_cycles"], 1)),
            "eps_y_max": float(aud["eps_y"]),
            "eps_v_max": float(aud["eps_v"]),
            "v_max": float(aud["v_max"]),
            "margin_ok_fraction": float(
                aud["margin_ok"] / max(aud["margin_total"], 1)),
            "margin_samples": float(aud["margin_total"]),
            "action_change_rate": float(
                aud["action_change"] / max(aud["action_total"], 1)),
            "action_change_price_rate": float(
                aud["action_change_price"] / max(aud["action_decomp_total"],
                                                 1)),
            "action_change_deficit_rate": float(
                aud["action_change_deficit"] / max(aud["action_decomp_total"],
                                                   1)),
        }
    if power_cap is not None:
        out["sensing_power_per_uav"] = float(
            power_used / max(power_cycles * k, 1))
    if bridge:
        out["bridge"] = {
            "L": br["L"], "A": br["A"], "A_raw": br["A_raw"],
            "V": br["V"], "M": br["M"], "S": br["S"],
            "r_pred": br["r_pred"], "r_real": br["r_real"],
            "n_served": br["n_served"],
            "a_thr": br["a_thr"].tolist(),
            "delivery_matrix": np.asarray(br["delivery_matrix"]),
            "H": H_all, "T": delays,
            "b_llr": b_llr, "mu_llr": mu_llr, "v_llr": v_llr,
            "v2_max": v2_max, "mu2_max": mu2_max,
            "v_up_analytic": v_up_analytic,
        }
    if raw_counts:
        # P2.1a (advice/008 section 13): raw per-target conditional
        # counts so the QoS certificate uses the ACTUAL binomial
        # denominators (the frozen non-raw output is byte-identical).
        # ``decided_upper`` is the RAW decision record (upper-threshold
        # crossing), independent of the declared_h1 bookkeeping that only
        # counts H1 runs -- N_FA = H0 runs crossing the upper bound,
        # N_MD = H1 runs NOT crossing it (lower-side or never decided
        # within the horizon).  N_H0/N_H1 are the realized per-target
        # run counts (random denominators, NOT equal to n_runs, so the
        # P2.1 QoS must NOT invert the conditional error probability
        # times n_runs -- that was the P0 flaw advice/008 section 13
        # flagged).
        n_H0, n_H1 = [], []
        n_fa, n_md = [], []
        for qq in range(q):
            h0 = int(np.sum(~H_all[:, qq]))
            h1 = int(np.sum(H_all[:, qq]))
            n_H0.append(h0)
            n_H1.append(h1)
            n_fa.append(int(np.sum(decided_upper[~H_all[:, qq], qq])))
            n_md.append(int(h1 - np.sum(decided_upper[H_all[:, qq], qq])))
        out["raw_counts"] = {
            "n_H0": n_H0, "n_H1": n_H1, "n_FA": n_fa, "n_MD": n_md,
        }
    return out
