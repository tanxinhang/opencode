"""Gate F0-G7: Physical-airtime reporting for FRIDS-v2 (advice/013).

Upgrades the U2U ledger from a bit-count abstraction to a waveform-derived
airtime constraint.  The link rate is the Shannon capacity UPPER BOUND
(honest -- never a claimed throughput):

    R_ij(t) = W_c * log2(1 + gamma_ij(t))          bits/s
    tau_ij(t) = b_tok(Q) / R_ij(t)                 seconds (token airtime)

and exactly ONE decision variable is added, the report/no-report gate

    z_i(t) = 1{ max_q U_iq(t) > 0 },
    U_iq(t) = y_q * g_iq / (D_q + eps) - lambda_i(t) * c_air(i, o_q),

with ``c_air(i, j) = tau_ij / T_air`` the fractional airtime budget one
token consumes at receiver ``j`` and ``lambda_i`` the per-UAV airtime
price.  The price is the sum of a baseline task-opportunity cost
``lambda_base`` (the sensing/spectrum opportunity cost of a U2U report,
positive even without congestion, per the F0-A n^-2 law) and a dual
ascent term driven by the UAV's own observed receive-load ratio:

    lambda_dual_i(t+1) = max(0, lambda_dual_i(t) + mu_c * (rho_i(t) - 1)),
    rho_i(t) = Lbar_i(t) / T_air,

with ``Lbar_i`` the EMA of the committed receive airtime at UAV ``i``.

The critical mathematical requirement (Lemma 4.99, FORMAL_PROOFS 5C): a
common additive communication price without the no-report (idle) action
never changes any argmax -- the price would be a NO-OP.  The price here
is paired with the idle option ``z_i = 0`` (score 0), so it decides
whether the best net value exceeds the airtime cost.

The full-mesh receive load at UAV ``i`` is ``L_i = sum_{j != i} z_j
tau_ji``.  If a cycle's committed load exceeds the budget, every token is
delivered at receiver ``i`` with overflow survival
``min(1, T_air / L_i)`` (queue-overflow thinning, physical decode loss);
``tau_ij`` is still charged (the transmission consumed the airtime).

All sensing/task machinery is frozen: the FRIDS-v2 target choice
``argmax_q y*g/(D+eps)``, the mirror-descent prices, the policy-matched
two-threshold stopping on the communication-domain belief, the compact
scale-aware token, fixed owner, full mesh, current scenario generation.
The airtime price affects ONLY the report gate ``z_i`` (the always-report
baseline is therefore price-independent).  Only the reporting changes.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from uav_otfs_isac.distributed_audit import (
    quantize_llr,
    quantize_with,
)
from uav_otfs_isac.difficulty_decomposition import d_kl_binary


# ---------------------------------------------------------------------------
# Physical airtime model
# ---------------------------------------------------------------------------


def link_rate_from_snr(snr_db: float | np.ndarray, bandwidth: float) -> float | np.ndarray:
    """Shannon capacity ``W_c log2(1 + gamma)`` -- an UPPER BOUND on the
    link rate, never a claimed throughput (advice/013 section 1.1)."""
    gamma = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    return float(bandwidth) * np.log2(1.0 + gamma)


def token_airtime(bits: float, rate: float | np.ndarray) -> float | np.ndarray:
    """``tau = b_tok / R`` seconds per token."""
    return float(bits) / np.asarray(rate, dtype=float)


def snr_from_outage_success(
    success: float | np.ndarray,
    threshold_db: float = 5.0,
    shadowing_db: float = 3.0,
) -> float | np.ndarray:
    """Invert the log-normal outage model ``s = Phi((SNR - threshold) /
    sigma)`` to recover the per-link SNR from the delivery success.  This
    couples the capacity rate and the outage success to the SAME U2U link
    statistics (physical: better link -> higher rate AND higher survival),
    while keeping the sensing channel independent (SYSTEM_MODEL section
    5).  Successes at the clipping edges map to a bounded SNR band."""
    s = np.clip(np.asarray(success, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(threshold_db) + float(shadowing_db) * norm.ppf(s)


def build_airtime_model(
    scenario: dict,
    bandwidth: float = 1e6,
    threshold_db: float = 5.0,
    shadowing_db: float = 3.0,
    t_air: float | None = None,
    rho_target: float | None = None,
) -> dict:
    """Per-link SNR, rate, token airtime and the per-cycle airtime budget.

    ``T_air`` is either given explicitly or derived so that the
    always-report full-mesh receive load ratio ``rho_target`` holds:
    ``T_air = max_i sum_{j != i} tau_ji / rho_target``.  Returns the
    ``tau`` (seconds), ``c_air = tau / T_air`` (fractional budget), and
    the ``rho_full`` load ratio of the always-report frame.
    """
    k = scenario["k"]
    b_tok = float(scenario.get("token_bits", token_bits_default(scenario["q"])))
    s = np.asarray(scenario["u2u_success"], dtype=float)
    snr_db = np.asarray(
        snr_from_outage_success(s, threshold_db, shadowing_db), dtype=float)
    rate = np.asarray(link_rate_from_snr(snr_db, bandwidth), dtype=float)
    tau = np.asarray(token_airtime(b_tok, rate), dtype=float)
    np.fill_diagonal(tau, 0.0)
    full_load = np.array([float(np.sum(tau[:, i])) for i in range(k)])
    if t_air is None:
        if rho_target is None:
            raise ValueError("give either t_air or rho_target")
        t_air = float(np.max(full_load) / max(float(rho_target), 1e-12))
    rho_full = float(np.max(full_load) / max(float(t_air), 1e-30))
    return {
        "snr_db": snr_db,
        "rate": rate,
        "tau": tau,
        "t_air": float(t_air),
        "b_tok": b_tok,
        "c_air": tau / max(float(t_air), 1e-30),
        "rho_full": rho_full,
        "bandwidth": float(bandwidth),
        "threshold_db": float(threshold_db),
        "shadowing_db": float(shadowing_db),
    }


def token_bits_default(q_targets: int) -> float:
    """Total token bits at ``Q`` targets (scale-aware layout)."""
    from uav_otfs_isac.distributed_audit import token_bits
    return float(token_bits(q_targets)["total"])


def receive_load(admitted: list, tau: np.ndarray, k: int) -> np.ndarray:
    """Committed receive airtime at every receiver: ``L_i = sum_{j != i,
    j in admitted} tau_ji``."""
    load = np.zeros(k)
    for uav in admitted:
        for neighbor in range(k):
            if neighbor == uav:
                continue
            load[neighbor] += tau[uav, neighbor]
    return load


def overflow_survival(load: np.ndarray, t_air: float) -> np.ndarray:
    """Per-receiver overflow survival ``min(1, T_air / L_i)`` (queue
    overflow thinning when the committed frame exceeds the budget)."""
    return np.minimum(1.0, t_air / np.maximum(load, 1e-15))


def report_score(
    y: np.ndarray,
    g: np.ndarray,
    deficit: float,
    eps: float,
    lam: float,
    c_air: float,
) -> float:
    """Net value of reporting ``U_iq = y*g/(D+eps) - lambda * c_air``."""
    return float(y * g / (deficit + eps) - lam * c_air)


def update_airtime_price(
    lam_dual: np.ndarray,
    load_smooth: np.ndarray,
    t_air: float,
    mu_c: float,
    lam_cap: float = 2.0,
) -> np.ndarray:
    """Dual ascent on the receive-load constraint
    ``lambda += mu_c * (rho - 1)`` clipped to ``[0, lam_cap]``
    (rho = L/T_air)."""
    rho = np.asarray(load_smooth, dtype=float) / max(float(t_air), 1e-30)
    return np.clip(
        lam_dual + float(mu_c) * (rho - 1.0), 0.0, float(lam_cap))


def oracle_admission(values: np.ndarray, tau: np.ndarray, t_air: float) -> np.ndarray:
    """Central admission oracle (offline reference): greedy by
    value/airtime density over the per-receiver load constraint
    ``max_i sum_{j != i} z_j tau_ji <= T_air``.  Values are the GLOBAL
    (owner-belief) values -- the oracle is not deployable."""
    k = len(values)
    densities = np.asarray(values, dtype=float) / np.maximum(
        np.sum(tau, axis=1), 1e-30)
    order = np.argsort(-densities)
    z = np.zeros(k, dtype=bool)
    load = np.zeros(k)
    for uav in order:
        cand = load + tau[uav]
        cand[uav] = 0.0
        if float(np.max(cand)) <= float(t_air) + 1e-12:
            z[uav] = True
            load = cand
    return z


# ---------------------------------------------------------------------------
# Airtime-constrained FRIDS-v2 simulation (all reporting methods)
# ---------------------------------------------------------------------------


def _g_matrix(scenario: dict, s_for_g: np.ndarray | None) -> np.ndarray:
    k, q = scenario["k"], scenario["q"]
    owner_of = scenario["owner_of"]
    g_s = s_for_g if s_for_g is not None else scenario["u2u_success"]
    g = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            best = 0.0
            for act in scenario["by_host"][(i, qq)]:
                best = max(best, float(act["i_plus"]) * float(g_s[i, owner_of[qq]]))
            g[i, qq] = best
    return g


def simulate_frids_v2_air(
    scenario: dict,
    bounds: list,
    airtime: dict,
    n_runs: int = 200,
    seed: int = 0,
    max_steps: int = 40,
    alpha: float = 0.05,
    beta: float = 0.05,
    mu: float = 0.5,
    mu_c: float = 0.2,
    eps: float = 0.1,
    quantizer: dict | None = None,
    delivery_matrix: np.ndarray | None = None,
    s_for_g: np.ndarray | None = None,
report_mode: str = "value",
    lambda_base: float = 0.0,
    p: float = 1.0,
    period: int = 1,
    ema_rho: float = 0.8,
    lam_cap: float = 2.0,
    value_mode: str = "info",
) -> dict:
    """FRIDS-v2 with the report/no-report airtime gate.

    ``report_mode``:

    - ``always`` -- every sensing UAV broadcasts (the frozen mainline;
      overload thinned by overflow survival);
    - ``value``  -- ``z_i = 1`` iff the best report value clears the
      airtime price (see ``value_mode``);
    - ``random`` -- ``z_i ~ Bernoulli(p)`` (communication-volume-matched
      fair baseline);
    - ``periodic`` -- ``z_i = 1{(t + i) mod period == 0}`` (fixed low-
      overhead baseline);
    - ``oracle``  -- central admission by value density on the global
      values (offline upper reference).

    ``value_mode`` selects the report value used by ``value``:

    - ``"deficit"`` -- the strict joint-LP dual of advice/013:
      ``y*g/(D+eps)`` vs ``lambda * c_air``.  This concentrates the
      airtime loss on the high-deficit target (its reports have small
      *normalized* value) and cannot beat uniform overflow thinning on
      the worst-target delay (F0-G7 finding);
    - ``"info"`` (default) -- the deficit-normalized price
      ``lambda * c_air / (D+eps)``, which cancels ``D`` (Lemma 4.101)
      and reduces the gate to comparing the min-max-weighted reliable
      information ``y*g`` to the airtime price ``lambda * c_air``.

    Returns the standard metrics plus the communication ledger
    (``comm``): airtime per cycle, tx reports per UAV, rx load per UAV,
    max load ratio, budget-feasible cycle fraction and the overflow-
    thinned token fraction.
    """
    k = scenario["k"]
    q = scenario["q"]
    owner_of = scenario["owner_of"]
    u2u = scenario["u2u_success"]
    delivery = delivery_matrix if delivery_matrix is not None else u2u
    tau = np.asarray(airtime["tau"], dtype=float)
    c_air = np.asarray(airtime["c_air"], dtype=float)
    t_air = float(airtime["t_air"])
    # locally-computable scarcity forecast: the full-report receive-load
    # ratio at the UAV's own receiver.  In the symmetric full mesh the
    # load into receiver ``i`` equals UAV ``i``'s own transmit row sum
    # (tau_ji = tau_ij), so every UAV can compute ``rho_est(i)`` from its
    # own channel state alone at t = 0 -- the dual cold-start (binds
    # immediately in the congested regime instead of ramping over the
    # short detection horizon).
    rho_est = np.sum(tau, axis=1) / max(float(t_air), 1e-30)
    g = _g_matrix(scenario, s_for_g)
    info_floor = float(d_kl_binary(1.0 - beta, alpha))
    a_thr = np.array([float(bounds[qq][0]) for qq in range(q)])

    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    comm_airtime = np.zeros(n_runs)     # committed airtime per cycle
    comm_tx = np.zeros(n_runs)          # admitted reports per cycle
    comm_rx = np.zeros(n_runs)          # mean receive load per cycle
    comm_max_ratio = np.zeros(n_runs)   # max_i L_i / T_air per cycle
    comm_feasible = np.zeros(n_runs)    # cycles with max L_i <= T_air
    comm_thinned = np.zeros(n_runs)     # tokens dropped by overflow
    comm_cycles = np.zeros(n_runs)

    for r in range(n_runs):
        H = rng.random(q) < 0.5
        H_all[r] = H
        L = np.zeros((k, q))
        decided = np.zeros(q, dtype=bool)
        y = np.full((k, q), 1.0 / q)
        lam_dual = mu_c * np.maximum(rho_est - 1.0, 0.0)
        # seed the smoothed load with the scarcity forecast so the dual
        # starts at its stationary point instead of decaying immediately
        load_smooth = rho_est * max(float(t_air), 1e-30)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            comm_cycles[r] += 1.0
            # price: baseline opportunity cost + dual term.  The dual uses
            # the EMA of the committed load of the PREVIOUS cycle (the
            # price reacts to the congestion it helped create, one cycle
            # stale -- consistent with the 1-cycle-stale intent map).
            lam_dual = update_airtime_price(
                lam_dual, load_smooth, t_air, mu_c, lam_cap)
            lam = lambda_base + lam_dual
            # local deficit from the UAV's OWN belief (strictly local)
            D_loc = np.maximum(a_thr[None, :] - L, 0.0)
            # choice + report gate.  The SENSING target is the frozen
            # FRIDS-v2 argmax ``argmax_q y*g/(D+eps)`` in every mode; the
            # airtime price only decides the report ``z_i`` (the frozen
            # mainline baseline is therefore price-independent).
            choices = [None] * k
            z = np.zeros(k, dtype=bool)
            for uav in range(k):
                best_q = None
                best_j = -np.inf
                for qq in undecided:
                    if g[uav, qq] <= 0.0:
                        continue
                    j_val = y[uav, qq] * g[uav, qq] / (D_loc[uav, qq] + eps)
                    if j_val > best_j:
                        best_j = j_val
                        best_q = qq
                if best_q is None:
                    continue
                if report_mode == "value":
                    if value_mode == "deficit":
                        net = best_j - lam[uav] * c_air[uav, owner_of[best_q]]
                    elif value_mode == "info":
                        net = (y[uav, best_q] * g[uav, best_q]
                               - lam[uav] * c_air[uav, owner_of[best_q]])
                    else:
                        raise ValueError(f"unknown value_mode {value_mode!r}")
                    z[uav] = bool(net > 0.0)
                elif report_mode == "always":
                    z[uav] = True
                elif report_mode == "random":
                    z[uav] = rng.random() < p
                elif report_mode == "periodic":
                    z[uav] = ((t + uav) % max(period, 1)) == 0
                elif report_mode == "oracle":
                    # the oracle decides the admission set globally below;
                    # here only the sensing target is fixed
                    z[uav] = True
                else:
                    raise ValueError(f"unknown report_mode {report_mode!r}")
                # best kernel for the chosen target (frozen FRIDS-v2)
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
            # oracle mode: recompute the admission set on GLOBAL values
            if report_mode == "oracle":
                d_g = np.maximum(a_thr - np.array(
                    [L[owner_of[qq], qq] for qq in range(q)]), 0.0)
                y_g = np.mean(y, axis=0)
                values = np.zeros(k)
                for uav in range(k):
                    choice = choices[uav]
                    if choice is None or choice[1] is None:
                        continue
                    best_q = choice[0]
                    values[uav] = y_g[best_q] * g[uav, best_q] \
                        / (d_g[best_q] + eps)
                z = oracle_admission(values, tau, t_air)
            # sensing (each UAV samples its chosen kernel exactly once;
            # the report gate only decides the broadcast)
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
            # committed receive load of the admitted set
            admitted = [uav for uav in range(k)
                        if z[uav] and obs_target[uav] >= 0]
            load_now = receive_load(admitted, tau, k)
            thin = overflow_survival(load_now, t_air)
            # ledger
            airtime_now = float(np.sum(load_now))
            comm_airtime[r] += airtime_now
            comm_tx[r] += len(admitted)
            comm_rx[r] += float(np.mean(load_now))
            comm_max_ratio[r] = max(
                comm_max_ratio[r], float(np.max(load_now) / max(t_air, 1e-30)))
            comm_feasible[r] += float(np.max(load_now) <= t_air + 1e-12)
            # token exchange with per-receiver overflow thinning
            S_loc = np.zeros((k, q))
            for uav in admitted:
                qq = int(obs_target[uav])
                for neighbor in range(k):
                    if neighbor == uav:
                        continue
                    if rng.random() > delivery[uav, neighbor] * thin[neighbor]:
                        continue
                    if not decided[qq]:
                        L[neighbor, qq] += obs_llr[uav]
                        S_loc[neighbor, qq] += obs_iplus[uav]
            comm_thinned[r] += float(
                sum((1.0 - thin[neighbor])
                    for uav in admitted for neighbor in range(k)
                    if neighbor != uav))
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
            # EMA of the committed load (used by the price next cycle)
            load_smooth = ema_rho * load_now + (1.0 - ema_rho) * load_smooth

    e1 = [float(delays[H_all[:, qq], qq].mean()) for qq in range(q)]
    p_fa = [float(declared_h1[~H_all[:, qq], qq].mean()) for qq in range(q)]
    p_md = [float(1.0 - declared_h1[H_all[:, qq], qq].mean())
            for qq in range(q)]
    active = np.maximum(comm_cycles, 1.0)
    return {
        "worst_target_delay": float(np.max(e1)),
        "e1_delays": e1,
        "p_fa": p_fa,
        "p_md": p_md,
        "comm": {
            "airtime_per_cycle": float(np.mean(comm_airtime / active)),
            "tx_reports_per_uav": float(np.mean(comm_tx / active) / k),
            "rx_load_per_uav": float(np.mean(comm_rx)),
            "max_load_ratio": float(np.mean(comm_max_ratio)),
            "budget_feasible_fraction": float(np.mean(comm_feasible / active)),
            "thinned_tokens_per_cycle": float(np.mean(comm_thinned / active)),
        },
    }