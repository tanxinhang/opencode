"""Gate F0: distributed information audit (advice/005.md).

The project-level correction of advice/005 repositions the system as
``Communication-Constrained Distributed Multi-UAV Task-Oriented Cooperative
Detection``: every UAV acts on its own information set

    I_{i,t} = { o_{i,0:t}, m_{j->i,0:t} (delivered tokens), own history }

with local beliefs ``L_{i,q,t}``, a compact U2U token interface, bounded
coordination rounds (R_coord <= 2), per-node resource budgets, and
``a_{i,t} = pi_i(I_{i,t})``.  The centralized fusion center survives only
as an offline audit oracle.

This module implements the F0 audit: the *same* distributed decision rule
(per-target dual G-values from the calibrated two-threshold delay values,
plus a neighbor-intent congestion price psi) is evaluated under four
information structures that differ only in what each UAV knows:

A. ``centralized``   -- global belief, same-cycle complete intents, perfect
                        delivery (oracle / upper reference).
B. ``full_message``  -- local information sets; every broadcast carries the
                        exact observation LLR of this cycle + next intent;
                        delivery subject to per-link success.
C. ``compact_token`` -- local information sets; every broadcast carries the
                        token ``{q, L_hat, u, r, chi, intent, t_stamp}``
                        with the observation LLR uniformly quantized to
                        ``token_llr_bits``; delivery subject to per-link
                        success.
D. ``local_only``    -- no tokens at all (zero-communication baseline).

The audit answers the three questions of advice/005 section 15 on the
worst-target H1 detection delay ``max_q E_1[T_q]`` (constraints: calibrated
per-target two-threshold rule with ``P_FA <= alpha``, ``P_MD <= beta``):

- cooperation value   ``Delta_coop    = J_D - J_C`` (is cooperation worth it?)
- communication loss  ``Delta_comm    = J_C - J_B`` (is token design the
  research bottleneck?)
- decentralization    ``Delta_decent  = J_B - J_A`` (how costly is removing
  the fusion center?)

plus the distributed stability metrics of advice/005 section 18: decision
conflict rate, duplicate sensing rate, role switch rate (fixed owners -> 0),
and the belief disagreement ``D_L(t) = mean |L_{i,q,t} - L_{j,q,t}|``.

Notation and kernels follow ``active_detection_bellman.py`` (post-
communication kernels ``(p0, p1, llr)`` through quantization + BSC +
detectable erasure) and ``SYSTEM_MODEL.md``.
"""

from __future__ import annotations

import numpy as np

from uav_otfs_isac.active_detection_bellman import (
    action_kernels,
    belief_from_log_odds,
    calibrate_sprt_boundaries,
    delay_value_iteration,
    sprt_boundary_policy,
    _evaluate_single,
)

# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------


def build_distributed_scenario(
    rng: np.random.Generator,
    k_uavs: int = 6,
    q_targets: int = 3,
    bits_list: tuple = (1, 2, 3),
    powers: tuple = (1.0, 2.0),
    l_acc: float = 4.0,
    weak_targets: tuple = (0,),
    snr_shift: float = 0.0,
    dd_grid: tuple | None = None,
    dd_physics: dict | None = None,
) -> dict:
    """Randomized ``K x Q`` report-link kernels plus the U2U reliability
    matrix and the fixed per-target owner roles.

    Every UAV hosts one report link per target; each link offers one
    kernel per entry of ``powers`` (the sensing-power lever) with a shared
    quantizer-bit count.  ``weak_targets`` draw their sensing SNR from a
    lower band so that coordination pressure (weak-target starvation) is
    present by construction.  ``snr_shift`` (dB) shifts every sensing SNR
    up (the System-Bottleneck ideal-evidence oracle: uniformly strong
    sensing isolates the sensing headroom of the worst-target delay).

    Fixed-TB OTFS evidence audit (advice/024, G10-C): ``dd_grid`` is the
    DD grid shape ``(N_doppler, N_delay)``; ``dd_physics`` maps each
    target to its physical normalized (Doppler, delay).  When both are
    given, the per-UAV sensing SNR is attenuated by the standard OTFS
    fractional-bin leakage

        L = sinc^2( frac_nu ) * sinc^2( frac_tau ),
        frac_nu  = ((nu_q + jitter_iq) * N_doppler) mod 1,
        frac_tau = ((tau_q + jitter_iq) * N_delay)  mod 1,

    so changing the DD resource SHAPE under a fixed time-bandwidth
    product (N_doppler * N_delay = const) changes the evidence I+ only
    through the grid resolution.  With ``dd_grid=None`` (default) the
    frozen behavior is unchanged.
    """
    q = q_targets
    if dd_grid is not None:
        n_d, n_l = int(dd_grid[0]), int(dd_grid[1])
    else:
        n_d = n_l = None
    links: dict = {qq: [] for qq in range(q)}          # all kernels of q
    by_host: dict = {
        (i, qq): [] for i in range(k_uavs) for qq in range(q)
    }                                                   # UAV-local kernels
    for i in range(k_uavs):
        for qq in range(q):
            if qq in weak_targets:
                snr_db = float(rng.uniform(-4.5, -1.0)) + float(snr_shift)
            else:
                snr_db = float(rng.uniform(-3.0, 3.0)) + float(snr_shift)
            if dd_grid is not None:
                nu_q, tau_q = dd_physics[qq]
                fr_nu = ((float(nu_q) + 0.02 * float(rng.standard_normal()))
                         * n_d) % 1.0
                fr_tau = ((float(tau_q) + 0.02 * float(rng.standard_normal()))
                          * n_l) % 1.0
                leak = float(np.sinc(fr_nu) ** 2 * np.sinc(fr_tau) ** 2)
                snr_db += 10.0 * np.log10(max(leak, 1e-6))
            noncentrality = l_acc * 10 ** (snr_db / 10.0)
            bits = int(rng.integers(bits_list[0], bits_list[-1] + 1))
            flip = float(rng.uniform(0.02, 0.15))
            success = float(rng.uniform(0.5, 0.9))
            for power in powers:
                mu1 = l_acc + power * noncentrality
                var1 = l_acc + 2.0 * power * noncentrality
                kernel = action_kernels(
                    l_acc, l_acc, mu1, var1, bits, flip, success,
                )
                kernel["cost"] = float(bits + (power - 1.0))
                kernel["bits"] = bits
                kernel["power"] = float(power)
                kernel["host"] = i
                kernel["target"] = qq
                links[qq].append(kernel)
                by_host[(i, qq)].append(kernel)
    u2u = 0.6 + 0.35 * rng.random((k_uavs, k_uavs))
    u2u = (u2u + u2u.T) / 2.0
    np.fill_diagonal(u2u, 1.0)
    # cyclic owner assignment: for Q > K multiple targets share
    # an owner (consistent for every K, Q); for Q <= K this is
    # the original o_q = q role
    owner_of = [int(qq % k_uavs) for qq in range(q)]
    return {
        "k": k_uavs, "q": q, "l_acc": l_acc,
        "links": links, "by_host": by_host, "u2u_success": u2u,
        "owner_of": owner_of,
    }


def _wald_grid(alpha: float, beta: float, margin: float = 1.0,
               points: int = 7):
    """The (A, B) scan grid around the Wald values used by the
    calibration."""
    a_wald = float(np.log((1.0 - beta) / alpha))
    b_wald = float(np.log(beta / (1.0 - alpha)))
    a_grid = np.unique(np.concatenate([
        np.linspace(a_wald - margin, a_wald + margin, points),
        [a_wald],
    ]))
    b_grid = np.unique(np.concatenate([
        np.linspace(b_wald - margin, b_wald + margin, points),
        [b_wald],
    ]))
    return a_grid, b_grid


def stabilized_calibrate(
    actions: list,
    alpha: float = 0.05,
    beta: float = 0.05,
    budget: int = 4 * 64,
    scan_runs: int = 300,
    verify_runs: int = 2000,
    seed: int = 100,
    margin: float = 1.0,
    points: int = 7,
) -> dict:
    """Two-stage calibrated two thresholds ``(A*, B*)`` with a
    high-MC-stable selection.

    Stage 1 scans the (A, B) grid around the Wald values with
    ``scan_runs`` MC evaluations and keeps the feasible candidate with the
    smallest H1 delay.  Stage 2 re-evaluates the 3x3 neighborhood of that
    candidate with ``verify_runs`` (independent seed) and picks the
    feasible minimum-delay boundary deterministically.

    The plain one-stage calibration flips between near-tied feasible
    candidates when its MC budget is small, and that realization noise
    leaks into the F0 comparison (it dominated the compact-token vs
    full-message gap); the two-stage selection removes it.
    """
    a_grid, b_grid = _wald_grid(alpha, beta, margin, points)
    best = None
    for a_bound in a_grid:
        for b_bound in b_grid:
            if a_bound <= b_bound:
                continue
            pol = sprt_boundary_policy(actions, float(a_bound),
                                       float(b_bound))
            row = _evaluate_single(pol, actions, budget, alpha, beta,
                                   scan_runs, seed)
            if row["p_fa"] <= alpha + 1e-9 and row["p_md"] <= beta + 1e-9:
                if best is None or row["e1_delay"] < best[0]:
                    best = (row["e1_delay"], float(a_bound),
                            float(b_bound))
    if best is None:
        raise ValueError(
            f"no (A, B) on the calibration grid meets P_FA <= {alpha}, "
            f"P_MD <= {beta}"
        )
    _, a0, b0 = best
    if len(a_grid) > 1:
        step = float(a_grid[1] - a_grid[0])
    else:
        step = 0.5
    a_local = np.unique(np.clip(
        a0 + np.array([-step, 0.0, step]), a_grid[0], a_grid[-1]))
    b_local = np.unique(np.clip(
        b0 + np.array([-step, 0.0, step]), b_grid[0], b_grid[-1]))

    def _verify(pool):
        """High-MC feasibility check of a (A, B) pool; the feasible
        candidate with the smallest H1 delay wins (deterministic)."""
        found = None
        for a_bound in pool[0]:
            for b_bound in pool[1]:
                if a_bound <= b_bound:
                    continue
                pol = sprt_boundary_policy(actions, float(a_bound),
                                           float(b_bound))
                row = _evaluate_single(pol, actions, budget, alpha, beta,
                                       verify_runs, seed + 1000)
                if row["p_fa"] <= alpha + 1e-9 \
                        and row["p_md"] <= beta + 1e-9:
                    if found is None or row["e1_delay"] < found["e1_delay"]:
                        row["a_bound"] = float(a_bound)
                        row["b_bound"] = float(b_bound)
                        found = row
        return found

    best = _verify((a_local, b_local))
    if best is None:
        # the coarse winner's neighborhood is empty under the high-MC
        # verification (near-tie feasibility flip at large scales);
        # fall back to the full grid (rare, ~5x the local cost)
        best = _verify((a_grid, b_grid))
    if best is None:
        raise ValueError(
            f"no (A, B) meets P_FA <= {alpha}, P_MD <= {beta} under the "
            f"high-MC verification"
        )
    return best


def calibrate_target_bounds(
    scenario: dict,
    alpha: float = 0.05,
    beta: float = 0.05,
    n_runs: int = 300,
    seed: int = 100,
    margin: float = 1.0,
    points: int = 7,
    llr_bits: int | None = None,
    verify_runs: int = 0,
    quantizer: dict | None = None,
    power_cap: np.ndarray | None = None,
) -> list:
    """Per-target numerically calibrated two thresholds ``(A_q, B_q)``.

    Each target calibrates against the union of all kernels available to
    it (all hosts), so the thresholds are valid for any observation stream
    that a host may contribute; the realized errors under the *actual*
    distributed observation policies are reported separately by the audit.

    With ``llr_bits`` (or ``quantizer``) given, the calibration runs on
    the token-quantized kernels (the compact-token information structure
    maintains its belief in the communication domain, so its thresholds
    must be calibrated on quantized atoms -- the D2 lesson that a
    constraint must enter the deployed rule, not be applied afterwards).
    With ``verify_runs > 0`` the two-stage stabilized selection is used
    (see :func:`stabilized_calibrate`); the F0 audit requires it so that
    the compact-token vs full-message comparison is not dominated by
    calibration realization noise.

    The calibration budget is set so that it never binds inside the 64-step
    evaluation horizon (``budget = 4 * 64``, max kernel cost 4): the
    boundaries are then a property of the two-threshold rule alone, not of
    a total observation budget.
    """
    bounds = []
    for qq in range(scenario["q"]):
        actions = scenario["links"][qq]
        if power_cap is not None:
            actions = [a for a in actions
                       if float(a["power"]) <= float(power_cap[qq])]
        if llr_bits is not None or quantizer is not None:
            actions = quantized_kernels(actions, llr_bits,
                                        quantizer=quantizer)
        if verify_runs > 0:
            cal = stabilized_calibrate(
                actions, alpha, beta, 4 * 64,
                scan_runs=n_runs, verify_runs=verify_runs,
                seed=seed + 10 * qq, margin=margin, points=points,
            )
        else:
            cal = calibrate_sprt_boundaries(
                actions, alpha, beta, 4 * 64,
                n_runs=n_runs, seed=seed + 10 * qq,
                margin=margin, points=points,
            )
        bounds.append((float(cal["a_bound"]), float(cal["b_bound"])))
    return bounds


def build_target_values(
    scenario: dict,
    bounds: list,
    horizon: int = 40,
    budget: int = 8,
    nu: tuple | None = None,
    lam: float = 1.0,
    grid: int = 101,
    l_max: float = 8.0,
) -> list:
    """Per-target detection-delay values ``V_q(l, b)`` (built once, linear
    in Q; the shared building block of the dual G-value scheduler)."""
    q = scenario["q"]
    if nu is None:
        nu = tuple([1.0 / q] * q)
    return [
        delay_value_iteration(
            scenario["links"][qq], horizon, budget,
            64.0, 64.0, lam=lam, grid=grid, l_max=l_max,
            cycle_cost=float(nu[qq]), bounds=bounds[qq],
        )
        for qq in range(q)
    ]


# ---------------------------------------------------------------------------
# Distributed decision rule
# ---------------------------------------------------------------------------


def action_gain(
    v_delay: dict,
    act: dict,
    l: float,
    step: int,
    b_remaining: float,
    nu_q: float,
    lam: float = 1.0,
) -> float:
    """Dual G-value gain of playing ``act`` at belief ``l`` (the D2-D dual
    index, public reimplementation of ``active_detection_bellman._delay_gain``
    scaled by the min-max weight ``nu_q`` and the cost price ``lam``):

    ``J = nu_q * [ V(l, b) - (cycle + c + E V(l + llr, b - c)) ] - lam * c``
    """
    ls = v_delay["ls"]
    values = v_delay["values"]
    horizon = v_delay["horizon"]
    budget_max = v_delay["budget"]
    cycle = float(v_delay.get("cycle_cost", 1.0))
    rem = int(np.clip(horizon - step, 0, horizon))
    b = int(np.clip(int(round(b_remaining)), 0, budget_max))
    c = float(act.get("cost", 0.0))
    b_next = int(np.clip(b - int(round(c)), 0, budget_max))
    l_c = float(np.clip(l, ls[0], ls[-1]))
    pi = belief_from_log_odds(l_c)
    v_now = float(np.interp(l_c, ls, values[rem, b]))
    exp = 0.0
    for k in range(len(act["p0"])):
        target = float(np.clip(l_c + act["llr"][k], ls[0], ls[-1]))
        v_next = float(np.interp(target, ls, values[rem - 1, b_next]))
        exp += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) * v_next
    return nu_q * (v_now - (cycle + c + exp)) - lam * c


def _best_kernel_gain(uav, qq, l, step, b_cycle, scenario, singles, nu, lam):
    """Best dual-G-value over the kernels UAV ``uav`` hosts for target
    ``qq`` at local belief ``l``."""
    best = None
    best_g = -np.inf
    for act in scenario["by_host"][(uav, qq)]:
        g = action_gain(singles[qq], act, l, step, b_cycle,
                        float(nu[qq]), lam=lam)
        if g > best_g:
            best_g = g
            best = act
    return best, best_g


def choose_actions(
    mode: str,
    beliefs,
    undecided: list,
    scenario: dict,
    singles: list,
    nu: tuple,
    lam: float,
    step: int,
    b_cycle: float,
    intents_recv: np.ndarray,
    eta: float,
    centralized_belief=None,
    psi_gamma: float = 1.0,
    eta_A: float = 0.0,
    ages: np.ndarray | None = None,
    normalize_gains: bool = False,
    counts_override: np.ndarray | None = None,
) -> tuple:
    """Distributed action selection (advice/005 section 8).

    Every UAV scores each undecided target by its best dual G-value at its
    own belief, adds the neighbor-intent congestion price
    ``psi = -eta * (# received intents for q) ** psi_gamma`` (stale by one
    cycle in the token modes; same-cycle and complete in ``centralized``)
    and the optional detection-age bonus ``eta_A * age_q`` (cycles since
    the last service of q; the anti-starvation term of advice/007 section
    8), and senses the best.  Returns ``(choices, intents)`` with
    ``choices[uav] = (target, kernel)`` and ``intents[uav]`` the chosen
    target (for the next broadcast).

    The default parameters are the frozen F0/F0-S mechanism (``psi_gamma =
    1`` linear price, ``eta_A = 0`` no age term, ``normalize_gains =
    False``).  The F0-A competition audit found that the additive price
    and age terms are numerically inert against the 1e9-scaled in-band
    dual-G gains; with ``normalize_gains = True`` each UAV divides its
    candidates by the local gain scale (``max |J|`` over its undecided
    targets, a per-cycle, per-UAV constant), which makes the additive
    terms comparable -- the theoretical requirement for any additive
    congestion/latency price (the marginal value of the n-th concurrent
    observer decays like n^{-2}, so the price must live on the decision
    scale, not in absolute units).
    """
    k = scenario["k"]
    q = scenario["q"]
    choices = [None] * k
    intents = np.full(k, -1, dtype=int)
    if mode == "centralized":
        # round 1: everyone announces the target with the largest base gain
        # (computed on the global belief; complete and same-cycle)
        base_intents = []
        for uav in range(k):
            best_q = undecided[0]
            best_g = -np.inf
            for qq in undecided:
                _, g = _best_kernel_gain(
                    uav, qq, centralized_belief[qq],
                    step, b_cycle, scenario, singles, nu, lam,
                )
                if g > best_g:
                    best_g = g
                    best_q = qq
            base_intents.append(best_q)
        counts = np.zeros(q)
        for u in base_intents:
            counts[u] += 1.0
    elif mode == "local_only":
        counts = np.zeros(q)
    elif counts_override is not None:
        # the two-round (fresh-intent) coordination provides the per-target
        # congestion counts of THIS cycle (received base intents); the
        # price is then computed on fresh information instead of the
        # 1-cycle-stale intent map
        counts = np.asarray(counts_override, dtype=float)
    else:
        counts = np.zeros(q)
        for uav in range(k):
            for neighbor in range(k):
                if neighbor == uav:
                    continue
                it = intents_recv[uav, neighbor]
                if it >= 0:
                    counts[it] += 1.0
    for uav in range(k):
        best_q = None
        best_g = -np.inf
        raw = {}
        for qq in undecided:
            if mode == "centralized":
                l_here = centralized_belief[qq]
            else:
                l_here = float(beliefs[uav, qq])
            _, g = _best_kernel_gain(
                uav, qq, l_here, step, b_cycle, scenario, singles, nu, lam,
            )
            raw[qq] = g
        scale = 1.0
        if normalize_gains:
            scale = max(1e-12, max(abs(v) for v in raw.values()))
        for qq in undecided:
            g = raw[qq] / scale if normalize_gains else raw[qq]
            psi = (-eta * counts[qq] ** psi_gamma
                   if mode != "local_only" else 0.0)
            age_term = eta_A * float(ages[qq]) if ages is not None else 0.0
            g_total = g + psi + age_term
            if g_total > best_g:
                best_g = g_total
                best_q = qq
        if best_q is not None:
            choices[uav] = (int(best_q), None)
            intents[uav] = int(best_q)
    # resolve the kernel inside each choice after ties are settled
    for uav in range(k):
        if choices[uav] is None:
            continue
        qq = choices[uav][0]
        if mode == "centralized":
            l_here = centralized_belief[qq]
        else:
            l_here = float(beliefs[uav, qq])
        best, _ = _best_kernel_gain(
            uav, qq, l_here, step, b_cycle, scenario, singles, nu, lam,
        )
        choices[uav] = (qq, best)
    return choices, intents


# ---------------------------------------------------------------------------
# Token interface (advice/005 section 6)
# ---------------------------------------------------------------------------

TOKEN_LLR_BITS = 5          # 4 bits is infeasible at alpha=beta=0.05
TOKEN_LLR_RANGE = 6.0
TOKEN_FULL_LLR_BITS = 32
TOKEN_Q_BITS = 2
TOKEN_CHI_BITS = 2
TOKEN_INTENT_BITS = 2
TOKEN_STAMP_BITS = 4
TOKEN_U_BITS = 2
TOKEN_R_BITS = 2


def token_bits(q_targets: int = 4) -> dict:
    """Bit accounting of the compact token
    ``{q, L_hat, u, r, chi, intent, t_stamp}`` (advice/005 section 6),
    scale-aware (advice/010): the target field and the intent field need
    ``b_q = ceil(log2 Q)`` bits each, and the dead payload fields
    ``u/r`` are dropped first so the total stays <= 19 bits for every Q.

    (Q=4 and below keep the original layout; Q=8 moves 2 bits from
    ``u/r`` to ``q/intent``.)
    """
    b_q = max(2, int(np.ceil(np.log2(max(q_targets, 2)))))
    b_q = min(b_q, 8)
    b_intent = b_q
    q_bits = b_q
    llr_bits = TOKEN_LLR_BITS
    # the dead payload fields (u/r/chi/stamp are never read by the
    # algorithm) are dropped in cascade order until the layout fits 19
    # bits: first u/r, then chi, then stamp (advice/010-012)
    dropped = []
    u_bits, r_bits, chi_bits, stamp_bits = (
        TOKEN_U_BITS, TOKEN_R_BITS, TOKEN_CHI_BITS, TOKEN_STAMP_BITS)
    for field, size in (("u", TOKEN_U_BITS), ("r", TOKEN_R_BITS),
                        ("chi", TOKEN_CHI_BITS), ("stamp",
                                                  TOKEN_STAMP_BITS)):
        if (q_bits + llr_bits + u_bits + r_bits + chi_bits + b_intent
                + stamp_bits) <= 19:
            break
        if field == "u":
            u_bits = 0
        elif field == "r":
            r_bits = 0
        elif field == "chi":
            chi_bits = 0
        else:
            stamp_bits = 0
        dropped.append(field)
    total = (q_bits + llr_bits + u_bits + r_bits + chi_bits + b_intent
             + stamp_bits)
    if total > 19:
        raise ValueError(
            f"no token layout fits 19 bits for Q={q_targets} "
            f"(q={b_q}, intent={b_intent}, llr={llr_bits})")
    layout = {
        "q": q_bits,
        "llr": llr_bits,
        "u": u_bits, "r": r_bits,
        "chi": chi_bits,
        "intent": b_intent,
        "stamp": stamp_bits,
        "total": total,
        "dropped": dropped,
    }
    layout["full_message_total"] = (
        TOKEN_FULL_LLR_BITS + layout["q"] + layout["chi"]
        + layout["intent"] + layout["stamp"])
    return layout


def quantize_llr(llr: float, bits: int | None = None,
                 llr_range: float | None = None) -> float:
    """Uniform mid-rise quantization of one LLR over
    ``[-llr_range, llr_range]`` (the token evidence field ``L_hat``).

    The defaults are read from the module-level token layout at call time,
    so experiments can sweep ``TOKEN_LLR_BITS`` without redefining the
    function.
    """
    if bits is None:
        bits = TOKEN_LLR_BITS
    if llr_range is None:
        llr_range = TOKEN_LLR_RANGE
    levels = 2 ** bits
    step = 2.0 * llr_range / levels
    clipped = float(np.clip(llr, -llr_range, llr_range - 1e-12))
    idx = int(np.floor((clipped + llr_range) / step))
    idx = int(np.clip(idx, 0, levels - 1))
    return -llr_range + (idx + 0.5) * step


def uniform_quantizer(bits: int | None = None,
                      llr_range: float | None = None) -> dict:
    """The frozen mid-rise uniform codebook as an explicit ``{edges,
    centroids}`` dict (the uniform special case of the encoder
    abstraction used by the token-fidelity designs)."""
    if bits is None:
        bits = TOKEN_LLR_BITS
    if llr_range is None:
        llr_range = TOKEN_LLR_RANGE
    levels = 2 ** bits
    edges = np.linspace(-llr_range, llr_range, levels + 1)
    centroids = 0.5 * (edges[:-1] + edges[1:])
    return {
        "edges": np.asarray(edges),
        "centroids": np.asarray(centroids),
        "bits": int(bits),
        "range": float(llr_range),
    }


def lloyd_max_quantizer(
    atoms: np.ndarray,
    weights: np.ndarray,
    bits: int | None = None,
    llr_range: float | None = None,
    iters: int = 300,
) -> dict:
    """Lloyd-Max (k-means style) non-uniform quantizer on the empirical
    LLR atom distribution (the F0-D weak-target-fidelity token design).

    Theory: for a scalar source with known distribution, the MSE-optimal
    quantizer satisfies the nearest-neighbor and centroid conditions.  The
    centroid condition ``l_k = E[L | L in bin_k]`` makes the per-bin error
    zero-mean, so the accumulated communication-domain belief is unbiased
    increment-wise (the drift that drives the sequential-test delay is
    preserved), unlike mid-rise quantization whose per-bin error follows
    the within-bin asymmetry of the atoms.  Weighting the distortion by
    ``weights`` (the H1 atom masses) targets the H1 drift
    ``E_1[sum L_hat]`` that determines the detection delay ``E_1[T]``.

    The codebook lives inside the 19-bit token (the ``L_hat`` field keeps
    its ``bits`` levels); only the encoder changes -- no protocol change
    (communication principle: the receiver knows the shared codebook).
    """
    if bits is None:
        bits = TOKEN_LLR_BITS
    if llr_range is None:
        llr_range = TOKEN_LLR_RANGE
    levels = 2 ** bits
    x = np.asarray(atoms, dtype=float)
    w = np.asarray(weights, dtype=float)
    lo, hi = -llr_range, llr_range
    # init: uniform midpoints over the range
    centroids = np.linspace(lo + (hi - lo) / (2.0 * levels),
                            hi - (hi - lo) / (2.0 * levels), levels)
    for _ in range(iters):
        dist = np.abs(x[:, None] - centroids[None, :])
        idx = np.argmin(dist, axis=1)
        new = centroids.copy()
        for k in range(levels):
            mask = idx == k
            wsum = float(np.sum(w[mask]))
            if mask.sum() > 0 and wsum > 0:
                new[k] = float(np.sum(w[mask] * x[mask]) / wsum)
        # empty-bin repair: keep the previous centroid (unused level)
        centroids = np.sort(new)
    edges = np.concatenate((
        [lo],
        0.5 * (centroids[:-1] + centroids[1:]),
        [hi],
    ))
    return {
        "edges": np.asarray(edges),
        "centroids": np.asarray(centroids),
        "bits": int(bits),
        "range": float(llr_range),
    }


def quantize_with(quantizer: dict, llr: float) -> float:
    """Quantize one LLR with a precomputed (possibly non-uniform)
    codebook: nearest centroid in the bin that contains the value."""
    edges = quantizer["edges"]
    centroids = quantizer["centroids"]
    clipped = float(np.clip(llr, edges[0], edges[-1]))
    idx = int(np.clip(np.searchsorted(edges, clipped, side="right") - 1,
                      0, len(centroids) - 1))
    return float(centroids[idx])


def build_token_quantizer(scenario: dict, bits: int | None = None,
                          llr_range: float | None = None,
                          weight: str = "h1",
                          per_target_equal: bool = True) -> dict:
    """System-wide token codebook fitted on the pooled empirical LLR atom
    distribution of the scenario (all targets, all hosts), weighted by the
    H1 atom masses (``weight="h1"``: the delay-relevant drift) or by the
    pooled H0/H1 masses (``weight="pooled"``).

    With ``per_target_equal=True`` (default) every target contributes the
    same total codebook weight (per-target normalization), so the weak
    target's small atoms are represented on equal footing with the strong
    targets' atoms -- the worst-target metric cares about the weakest
    link, not the average link.  The mass-weighted pooled design
    (``per_target_equal=False``) optimizes the average target and starves
    the weak one (measured: only 2/32 levels near zero).
    """
    atoms = []
    weights = []
    for qq in range(scenario["q"]):
        for act in scenario["links"][qq]:
            p1 = np.asarray(act["p1"], dtype=float)
            p0 = np.asarray(act["p0"], dtype=float)
            llr = np.asarray(act["llr"], dtype=float)
            w = p1 if weight == "h1" else 0.5 * (p1 + p0)
            if per_target_equal:
                w = w / max(float(np.sum(w)), 1e-12) / scenario["q"]
            atoms.extend(llr.tolist())
            weights.extend(w.tolist())
    return lloyd_max_quantizer(np.asarray(atoms), np.asarray(weights),
                               bits=bits, llr_range=llr_range)


def mu_law_quantizer(bits: int | None = None,
                     llr_range: float | None = None,
                     mu: float = 100.0) -> dict:
    """mu-law companding codebook for the L_hat evidence field.

    The companded domain ``y = sign(l) * log(1 + mu*|l|/R) / log(1+mu)``
    is quantized uniformly; the linear-domain codebook ``{edges,
    centroids}`` is what the rest of the pipeline consumes.  mu-law
    guarantees fine resolution near zero *by construction* (classical
    telephony companding), which is exactly where the weak target's small
    LLR atoms live -- distribution-free, in contrast to the Lloyd-Max
    codebook whose mass-weighted pooling starves the small-atom region
    (F0-D: the 5-bit uniform token costs the weak target ~22% delay).
    """
    if bits is None:
        bits = TOKEN_LLR_BITS
    if llr_range is None:
        llr_range = TOKEN_LLR_RANGE
    levels = 2 ** bits
    r = float(llr_range)
    # uniform grid in the companded domain
    y = np.linspace(-1.0, 1.0, levels + 1)
    y_mid = 0.5 * (y[:-1] + y[1:])
    centroids = np.sign(y_mid) * (
        (1.0 + mu) ** np.abs(y_mid) - 1.0) / mu * r
    edges = np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu * r
    edges[0], edges[-1] = -r, r
    return {
        "edges": np.asarray(edges),
        "centroids": np.asarray(centroids),
        "bits": int(bits),
        "range": r,
        "mu": float(mu),
    }


def quantized_kernels(actions: list, bits: int | None = None,
                      llr_range: float | None = None,
                      quantizer: dict | None = None) -> list:
    """The same kernels with the LLR atoms passed through the token
    encoder.  The compact-token stopping rule is calibrated on these
    (the D2 lesson: a constraint must enter the rule that is actually
    deployed -- here the belief is maintained in the communication
    domain, so its thresholds must be calibrated on quantized atoms).
    With ``quantizer`` given, the non-uniform (Lloyd-Max) codebook is
    used instead of the default mid-rise uniform grid.
    """
    out = []
    for act in actions:
        k = dict(act)
        if quantizer is not None:
            k["llr"] = np.asarray([
                quantize_with(quantizer, float(x)) for x in act["llr"]
            ])
        else:
            k["llr"] = np.asarray([
                quantize_llr(float(x), bits, llr_range) for x in act["llr"]
            ])
        out.append(k)
    return out


# ---------------------------------------------------------------------------
# F0 simulation
# ---------------------------------------------------------------------------

MODES = ("centralized", "full_message", "compact_token", "local_only")


def simulate_system(
    mode: str,
    scenario: dict,
    bounds: list,
    singles: list,
    alpha: float = 0.05,
    beta: float = 0.05,
    n_runs: int = 400,
    seed: int = 0,
    max_steps: int = 40,
    nu: tuple | None = None,
    lam: float = 1.0,
    eta: float = 0.5,
    b_cycle: float = 8.0,
    psi_gamma: float = 1.0,
    eta_A: float = 0.0,
    normalize_gains: bool = False,
    quantizer: dict | None = None,
    delivery_override: float | None = None,
    fresh_intents: bool = False,
) -> dict:
    """Monte-Carlo run of one information structure.

    Per run the targets' truths are drawn independently (P(H1) = 0.5); every
    cycle each UAV senses one undecided target with the distributed dual
    G-value rule and (modes B/C) broadcasts one token; a target stops when
    the *owner* belief crosses the calibrated two-threshold rule.  Returns
    delays, realized errors, stability metrics, and bit accounting.

    ``psi_gamma`` / ``eta_A`` / ``normalize_gains`` are the F0-A corrected
    coordination options (defaults reproduce the frozen F0/F0-S mechanism;
    ``normalize_gains=True`` is the corrected mainline of advice/007-008).
    ``quantizer`` swaps the token's L_hat encoder for a non-uniform
    (Lloyd-Max) codebook (F0-D weak-target-fidelity design); the compact
    thresholds must be calibrated on the same codebook.  ``delivery_override``
    replaces the per-link success for the diagnostic (e.g. perfect
    delivery); ``fresh_intents`` enables the two-round coordination
    (R_coord = 2, within the per-cycle U2U budget): round 1 broadcasts the
    base intents (congestion counts of THIS cycle), round 2 carries the
    evidence token -- the congestion price then uses fresh intents instead
    of the 1-cycle-stale map.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    k = scenario["k"]
    q = scenario["q"]
    if nu is None:
        nu = tuple([1.0 / q] * q)
    owner_of = scenario["owner_of"]
    u2u = scenario["u2u_success"]
    tbits = token_bits(q)
    if mode == "centralized":
        bits_per_broadcast = 0.0
    elif mode == "full_message":
        bits_per_broadcast = float(tbits["full_message_total"])
    elif mode == "compact_token":
        bits_per_broadcast = float(tbits["total"])
    else:
        bits_per_broadcast = 0.0

    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    sense_cost = np.zeros((n_runs, q))
    conflict_pairs = np.zeros(n_runs)          # pairs of UAVs on same target
    dup_cycles = np.zeros(n_runs)              # cycles with >=2 on one target
    n_cycles = np.zeros(n_runs)
    dL_sum = np.zeros(n_runs)
    dL_count = np.zeros(n_runs)
    bits_total = np.zeros(n_runs)
    costs_total = np.zeros(n_runs)

    for r in range(n_runs):
        H = rng.random(q) < 0.5
        H_all[r] = H
        L = np.zeros((k, q))
        L_g = np.zeros(q)
        decided = np.zeros(q, dtype=bool)
        intents_recv = np.full((k, k), -1, dtype=int)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            n_cycles[r] += 1.0
            counts_override = None
            if fresh_intents and mode in ("full_message", "compact_token"):
                # round 1: everyone announces its base intent (no price),
                # received with delivery -> fresh congestion counts
                _, base_intents = choose_actions(
                    mode, L, undecided, scenario, singles, nu, lam, t,
                    b_cycle, intents_recv, eta,
                    centralized_belief=L_g,
                    psi_gamma=psi_gamma, eta_A=0.0,
                    normalize_gains=normalize_gains,
                    counts_override=np.zeros(q),
                )
                fresh_counts = np.zeros(q)
                for uav in range(k):
                    bi = int(base_intents[uav])
                    if bi < 0:
                        continue
                    for neighbor in range(k):
                        if neighbor == uav:
                            continue
                        succ = (delivery_override
                                if delivery_override is not None
                                else u2u[uav, neighbor])
                        if rng.random() <= succ:
                            fresh_counts[bi] += 1.0
                counts_override = fresh_counts
            choices, intents = choose_actions(
                mode, L, undecided, scenario, singles, nu, lam, t,
                b_cycle, intents_recv, eta, centralized_belief=L_g,
                psi_gamma=psi_gamma, eta_A=eta_A,
                normalize_gains=normalize_gains,
                counts_override=counts_override,
            )
            # sensing (each UAV samples its chosen kernel exactly once;
            # the token phase reuses the same realization)
            target_count = np.zeros(q)
            obs_target = np.full(k, -1, dtype=int)
            obs_llr = np.zeros(k)
            for uav in range(k):
                choice = choices[uav]
                if choice is None or choice[1] is None:
                    continue
                qq, act = choice
                target_count[qq] += 1.0
                p = act["p1"] if H[qq] else act["p0"]
                y = int(rng.choice(len(p), p=p))
                llr_obs = float(act["llr"][y])
                if mode == "compact_token":
                    # communication-domain belief: every increment, local
                    # or received, passes the token encoder (the compact
                    # thresholds are calibrated on quantized atoms)
                    llr_obs = (quantize_with(quantizer, llr_obs)
                               if quantizer is not None
                               else quantize_llr(llr_obs))
                obs_target[uav] = qq
                obs_llr[uav] = llr_obs
                if mode == "centralized":
                    L_g[qq] += llr_obs
                else:
                    L[uav, qq] += llr_obs
                sense_cost[r, qq] += float(act["cost"])
                costs_total[r] += float(act["cost"])
            # conflict / duplicate accounting
            for i in range(k):
                if obs_target[i] < 0:
                    continue
                for j in range(i + 1, k):
                    if obs_target[j] == obs_target[i]:
                        conflict_pairs[r] += 1.0
            if np.any(target_count >= 2.0):
                dup_cycles[r] += 1.0
            # token exchange (B/C) and intent bookkeeping
            if mode in ("full_message", "compact_token"):
                intents_next = np.full((k, k), -1, dtype=int)
                for uav in range(k):
                    if obs_target[uav] < 0:
                        continue
                    qq = int(obs_target[uav])
                    if mode == "full_message":
                        llr_sent = obs_llr[uav]
                    else:
                        llr_sent = quantize_llr(obs_llr[uav])
                    bits_total[r] += bits_per_broadcast
                    for neighbor in range(k):
                        if neighbor == uav:
                            continue
                        succ = (delivery_override
                                if delivery_override is not None
                                else u2u[uav, neighbor])
                        if rng.random() > succ:
                            continue
                        if not decided[qq]:
                            L[neighbor, qq] += llr_sent
                        intents_next[neighbor, uav] = int(intents[uav])
                intents_recv = intents_next
            # stopping on the owner belief (global belief for centralized)
            for qq in undecided:
                l_own = L_g[qq] if mode == "centralized" else L[owner_of[qq], qq]
                if l_own >= bounds[qq][0]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
                    if H[qq]:
                        declared_h1[r, qq] = 1.0
                elif l_own <= bounds[qq][1]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
                    if not H[qq]:
                        declared_h1[r, qq] = 0.0
            # belief disagreement over active pairs (B/C/D only)
            if mode != "centralized":
                for qq in undecided:
                    for i in range(k):
                        for j in range(i + 1, k):
                            dL_sum[r] += abs(L[i, qq] - L[j, qq])
                            dL_count[r] += 1.0

    # aggregates
    e1_delays = []
    p_fa = []
    p_md = []
    for qq in range(q):
        h1 = H_all[:, qq]
        h0 = ~h1
        e1_delays.append(float(delays[h1, qq].mean()))
        p_fa.append(float(declared_h1[h0, qq].mean()))
        p_md.append(float(1.0 - declared_h1[h1, qq].mean()))
    active_mean = max(float(np.mean(n_cycles)), 1.0)
    dL_mean = (float(np.sum(dL_sum) / np.sum(dL_count))
               if dL_count.sum() > 0 else None)
    return {
        "mode": mode,
        "e1_delays": e1_delays,
        "worst_target_delay": float(np.max(e1_delays)),
        "mean_e1_delay": float(np.mean(e1_delays)),
        "p_fa": p_fa,
        "p_md": p_md,
        "mean_sense_cost": float(np.mean(costs_total)),
        "mean_u2u_bits_per_cycle": float(np.mean(bits_total) / active_mean),
        "mean_u2u_bits_total": float(np.mean(bits_total)),
        "conflict_rate": float(np.mean(conflict_pairs) / active_mean
                               * 2.0 / max(k * (k - 1), 1)),
        "duplicate_sensing_rate": float(np.mean(dup_cycles) / active_mean),
        "role_switch_rate": 0.0,
        "belief_disagreement": dL_mean,
    }


def run_audit(
    scenario: dict,
    alpha: float = 0.05,
    beta: float = 0.05,
    n_runs: int = 400,
    seeds: int = 4,
    max_steps: int = 40,
    eta: float = 0.5,
    calib_n_runs: int = 300,
    calib_seed: int = 100,
    calib_verify_runs: int = 2000,
) -> dict:
    """Full F0 audit: calibrate, build values, simulate the four
    information structures, and answer the three questions of advice/005
    section 15 with the gap decomposition of section 13.

    Every mode meets the same error constraints with its *own* calibrated
    two-threshold rule: exact kernels for ``centralized`` / ``full_message``
    / ``local_only``, token-quantized kernels for ``compact_token`` (its
    belief lives in the communication domain).  The calibration uses the
    two-stage stabilized selection at a fixed seed (``calib_seed``) with a
    high-MC verification (``calib_verify_runs``); the residual
    calibration-seed sensitivity of the gaps is a documented boundary (a
    near-tie among feasible boundaries can shift a gap by a few tenths of
    a cycle).
    """
    q = scenario["q"]
    bounds_exact = calibrate_target_bounds(
        scenario, alpha, beta, n_runs=calib_n_runs,
        seed=calib_seed, verify_runs=calib_verify_runs,
    )
    bounds_token = calibrate_target_bounds(
        scenario, alpha, beta, n_runs=calib_n_runs,
        llr_bits=TOKEN_LLR_BITS, seed=calib_seed,
        verify_runs=calib_verify_runs,
    )
    nu = tuple([1.0 / q] * q)
    singles = build_target_values(scenario, bounds_exact, horizon=max_steps,
                                  nu=nu)
    bounds_for_mode = {
        "centralized": bounds_exact,
        "full_message": bounds_exact,
        "compact_token": bounds_token,
        "local_only": bounds_exact,
    }
    per_mode = {m: {key: [] for key in (
        "worst_target_delay", "e1_delays", "p_fa", "p_md",
        "mean_u2u_bits_per_cycle", "conflict_rate",
        "duplicate_sensing_rate", "belief_disagreement",
        "mean_e1_delay", "mean_sense_cost",
    )} for m in MODES}
    for seed in range(seeds):
        for mode in MODES:
            out = simulate_system(
                mode, scenario, bounds_for_mode[mode], singles,
                alpha=alpha, beta=beta, n_runs=n_runs,
                seed=seed * 1000 + 7, max_steps=max_steps,
                nu=nu, eta=eta,
            )
            for key in per_mode[mode]:
                val = out[key]
                per_mode[mode][key].append(val)
    summary = {}
    for mode in MODES:
        summary[mode] = {
            "worst_target_delay": float(np.mean(
                per_mode[mode]["worst_target_delay"])),
            "e1_delays": [float(np.mean([r[i] for r in per_mode[mode]
                                         ["e1_delays"]]))
                          for i in range(q)],
            "p_fa": [float(np.mean([r[i] for r in per_mode[mode]["p_fa"]]))
                     for i in range(q)],
            "p_md": [float(np.mean([r[i] for r in per_mode[mode]["p_md"]]))
                     for i in range(q)],
            "mean_u2u_bits_per_cycle": float(np.mean(
                [r for r in per_mode[mode]["mean_u2u_bits_per_cycle"]
                 if r is not None])) if any(
                     r is not None for r in per_mode[mode]
                     ["mean_u2u_bits_per_cycle"]) else 0.0,
            "conflict_rate": float(np.mean(
                per_mode[mode]["conflict_rate"])),
            "duplicate_sensing_rate": float(np.mean(
                per_mode[mode]["duplicate_sensing_rate"])),
            "role_switch_rate": 0.0,
            "belief_disagreement": float(np.mean([
                r for r in per_mode[mode]["belief_disagreement"]
                if r is not None])) if any(
                    r is not None for r in per_mode[mode]
                    ["belief_disagreement"]) else None,
            "mean_sense_cost": float(np.mean(
                per_mode[mode]["mean_sense_cost"])),
        }
    j_a = summary["centralized"]["worst_target_delay"]
    j_b = summary["full_message"]["worst_target_delay"]
    j_c = summary["compact_token"]["worst_target_delay"]
    j_d = summary["local_only"]["worst_target_delay"]
    gaps = {
        "decentralization": {
            "definition": "J_full_message - J_centralized",
            "value": float(j_b - j_a),
            "relative": float((j_b - j_a) / max(j_a, 1e-12)),
        },
        "communication": {
            "definition": "J_compact_token - J_full_message",
            "value": float(j_c - j_b),
            "relative": float((j_c - j_b) / max(j_b, 1e-12)),
        },
        "cooperation": {
            "definition": "J_local_only - J_compact_token",
            "value": float(j_d - j_c),
            "relative": float((j_d - j_c) / max(j_c, 1e-12)),
        },
    }
    tol_cycles = 0.20   # MC tolerance on the mean worst-target delays
    ordered = (j_a <= j_b + tol_cycles and j_b <= j_c + tol_cycles
               and j_c <= j_d + tol_cycles)
    tol = 0.02
    # the error constraint is required for the operational systems (A/B/C);
    # local-only is a diagnostic baseline whose (likely) infeasibility is
    # itself a finding (cooperation is necessary, not merely helpful)
    operational = ("centralized", "full_message", "compact_token")
    error_ok = all(
        float(np.max(summary[m]["p_fa"])) <= alpha + tol
        and float(np.max(summary[m]["p_md"])) <= beta + tol
        for m in operational
    )
    d_error_ok = bool(
        float(np.max(summary["local_only"]["p_fa"])) <= alpha + tol
        and float(np.max(summary["local_only"]["p_md"])) <= beta + tol
    )
    material = 0.02  # relative materiality threshold of a gap
    passed = ordered and error_ok
    return {
        "bounds_exact": [[round(float(b[0]), 3), round(float(b[1]), 3)]
                         for b in bounds_exact],
        "bounds_token": [[round(float(b[0]), 3), round(float(b[1]), 3)]
                         for b in bounds_token],
        "calibration": {
            "seed": calib_seed,
            "scan_runs": calib_n_runs,
            "verify_runs": calib_verify_runs,
        },
        "modes": summary,
        "gaps": gaps,
        "questions": {
            "cooperation_value": {
                "answer": bool(gaps["cooperation"]["relative"] > material),
                "comment": ("cooperation (tokens) materially helps the "
                            "worst target"
                            if gaps["cooperation"]["relative"] > material
                            else "no material cooperation value in this "
                                 "scenario"),
            },
            "communication_loss": {
                "answer": bool(gaps["communication"]["relative"] > material),
                "comment": ("token quantization is material; token design "
                            "is the research bottleneck"
                            if gaps["communication"]["relative"] > material
                            else "token quantization at the tested layout "
                                 "is nearly free; not the bottleneck"),
            },
            "decentralization_penalty": {
                "answer": bool(gaps["decentralization"]["relative"]
                               > material),
                "comment": ("removing the fusion center costs materially; "
                            "study ownership / conflict / distributed price"
                            if gaps["decentralization"]["relative"]
                            > material
                            else "decentralization is nearly free"),
            },
        },
        "stability": {
            "conflict_rate": {m: summary[m]["conflict_rate"]
                              for m in MODES},
            "duplicate_sensing_rate": {
                m: summary[m]["duplicate_sensing_rate"] for m in MODES},
            "role_switch_rate": {m: summary[m]["role_switch_rate"]
                                 for m in MODES},
            "belief_disagreement": {
                m: summary[m]["belief_disagreement"] for m in MODES},
        },
        "token_bits": token_bits(q),
        "passed": passed,
        "ordering_holds": ordered,
        "error_constraints_met": error_ok,
        "local_only_error_constraints_met": d_error_ok,
        "local_only_finding": {
            "met_constraints": d_error_ok,
            "comment": ("the zero-communication baseline cannot meet the "
                        "error constraints at the tested horizon; "
                        "cooperation is necessary, not merely helpful"
                        if not d_error_ok
                        else "the zero-communication baseline meets the "
                             "error constraints"),
        },
    }
