"""Gate F0-A: target competition audit (advice/007.md).

F0-S established that the scale loss concentrates in the distributed
task-decision layer (J(C) +13.1% over (6,3)->(16,8) while the centralized
oracle grows only +5.6%).  F0-A does NOT change any decision mechanism; it
adds five diagnostic quantities, measured per cycle on the frozen mainline
(compact token, dual-G, fixed owner, full mesh), at the same paired scales
(K, Q) = (6,3), (8,4), (12,6), (16,8):

1. per-target service rate  r_q = #{t : exists i, a_i(t)->q} / T,
   with r_min = min_q r_q   (target starvation detector);
2. longest consecutive idle run  H_q^idle, H_max^idle = max_q H_q^idle
   (sequential detection is gap-sensitive: an idle run freezes the belief,
   so E[T_q] grows even at equal mean service);
3. per-target concurrent UAV count  n_q(t) = sum_i 1[a_i(t)->q]:
   mean nbar_q, 95th percentile, max_t n_q(t) (starvation +
   over-concentration detector);
4. urgency-allocation Spearman correlation
   rho_alloc = corr( U_q(t), n_q(t) )  over undecided targets and cycles,
   with U_q(t) = max_i max_a J_{i,q,a}(t) the target-level dual-G aggregate
   the scheduler itself uses (does local value translate into the right
   global allocation?);
5. per-cycle allocation regret (offline audit only, same local candidates
   J_{i,q,a}):
   Phi_oracle(t) = sum_i max_q J_{i,q}(t)   (per-UAV argmax reference,
   no global information added to the deployment)
   Phi_dist(t)   = sum_i J_{i,q_i(t)}(t)
   R_alloc(t) = Phi_oracle(t) - Phi_dist(t),  plus the distorted-choice
   rate (fraction of UAV-cycles where the deployed choice differs from the
   unconstrained argmax).

Verdict (advice/007 section 6): exactly three allowed conclusions.

- Case 1 (resources insufficient): r_min drops with Q, rho_alloc stays
  high, regret stays small, and the coordination probes (age term,
  super-linear price) do not help -> the problem is Q/K load feasibility,
  not the scheduler.
- Case 2 (starvation, allocation fixable): H_max^idle grows with Q and
  the offline argmax reference shows material regret -> the prescribed fix
  is the single starvation-age term  J' = J + eta_A * A_q(t).
- Case 3 (over-concentration): rho_alloc drops with Q and some targets
  get n_q >> 1 while others get 0 -> the prescribed fix is the
  super-linear congestion price  psi = -eta * n_q^gamma, gamma > 1
  (marginal delay benefit of the n-th concurrent observer decays like
  n^{-2}, so the price curvature gamma > 1 is the right shape).

All other mechanisms stay frozen (fixed owner, full mesh, 19-bit token,
current thresholds, dual-G base value, sensing model).  The audit only
observes the UAV -> target allocation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
    choose_actions,
    quantize_llr,
    quantize_with,
)


def _best_gain_matrix(beliefs, undecided, scenario, singles, nu, lam,
                      step, b_cycle):
    """Per-UAV x per-target best dual-G values (the scheduler's own local
    candidates).  ``beliefs[uav, q]`` is UAV uav's local belief."""
    k = scenario["k"]
    J = np.full((k, len(undecided)), -np.inf)
    for uav in range(k):
        for j, qq in enumerate(undecided):
            best = -np.inf
            for act in scenario["by_host"][(uav, qq)]:
                g = _gain(singles[qq], act, float(beliefs[uav, qq]),
                          step, b_cycle, float(nu[qq]), lam)
                if g > best:
                    best = g
            J[uav, j] = best
    return J


def _gain(v_delay, act, l, step, b_remaining, nu_q, lam):
    """Same dual G-value as ``distributed_audit.action_gain`` (kept local
    to avoid recomputing through the public API in the audit loop)."""
    from uav_otfs_isac.distributed_audit import action_gain
    return action_gain(v_delay, act, l, step, b_remaining, nu_q, lam)


def simulate_competition_audit(
    scenario: dict,
    bounds: list,
    singles: list,
    nu: tuple,
    n_runs: int = 400,
    seed: int = 0,
    max_steps: int = 40,
    eta: float = 0.5,
    psi_gamma: float = 1.0,
    eta_A: float = 0.0,
    normalize_gains: bool = False,
    quantizer: dict | None = None,
    delivery_override: float | None = None,
    fresh_intents: bool = False,
    b_cycle: float = 8.0,
) -> dict:
    """MC run of the frozen compact-token mainline with per-cycle
    allocation diagnostics (service rates, idle runs, concurrency,
    urgency-allocation correlation, allocation regret).  The optional
    (eta_A, psi_gamma, normalize_gains) triple is the F0-A prescribed fix
    family; defaults reproduce the frozen mechanism exactly.
    ``quantizer`` swaps the token's L_hat encoder for the non-uniform
    (Lloyd-Max) codebook of the F0-D weak-target-fidelity design;
    ``delivery_override`` / ``fresh_intents`` are the F0-F delivery /
    coordination diagnostics (perfect delivery; two-round fresh-intent
    coordination)."""
    k = scenario["k"]
    q = scenario["q"]
    owner_of = scenario["owner_of"]
    u2u = scenario["u2u_success"]
    rng = np.random.default_rng(seed)

    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    service = np.zeros((n_runs, q))          # cycles served per target
    idle_max = np.zeros((n_runs, q))         # longest idle run per target
    nq_series = np.zeros((max_steps, q))     # per-cycle concurrency log
    nq_mean_all = np.zeros((n_runs, q))      # per-run mean concurrency
    nq_p95_all = np.zeros((n_runs, q))       # per-run 95th percentile
    nq_max_all = np.zeros((n_runs, q))       # per-run max concurrency
    j_med_all = np.zeros(n_runs)             # median |J| (gain scale)
    j_spread_all = np.zeros(n_runs)          # cross-target spread of max-J
    rho_sum = np.zeros(n_runs)
    rho_n = np.zeros(n_runs)
    regret_sum = np.zeros(n_runs)            # Phi_oracle - Phi_dist
    regret_n = np.zeros(n_runs)
    distorted = np.zeros(n_runs)             # UAV-cycles with non-argmax
    distorted_n = np.zeros(n_runs)
    n_cycles = np.zeros(n_runs)

    for r in range(n_runs):
        H = rng.random(q) < 0.5
        H_all[r] = H
        L = np.zeros((k, q))
        decided = np.zeros(q, dtype=bool)
        intents_recv = np.full((k, k), -1, dtype=int)
        last_served = np.zeros(q)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            n_cycles[r] += 1.0
            ages = np.array([float(t - last_served[qq])
                             for qq in range(q)])  # full-length; only
            # undecided entries are read by choose_actions
            counts_override = None
            if fresh_intents:
                _, base_intents = choose_actions(
                    "compact_token", L, undecided, scenario, singles, nu,
                    1.0, t, b_cycle, intents_recv, eta,
                    normalize_gains=normalize_gains,
                    counts_override=np.zeros(scenario["q"]),
                )
                fresh_counts = np.zeros(scenario["q"])
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
                "compact_token", L, undecided, scenario, singles, nu,
                1.0, t, b_cycle, intents_recv, eta,
                psi_gamma=psi_gamma, eta_A=eta_A, ages=ages,
                normalize_gains=normalize_gains,
                counts_override=counts_override,
            )
            # per-cycle candidate matrix for regret (offline audit)
            Jmat = _best_gain_matrix(L, undecided, scenario, singles, nu,
                                     1.0, t, b_cycle)
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
                llr_obs = (quantize_with(quantizer,
                                         float(act["llr"][y]))
                           if quantizer is not None
                           else quantize_llr(float(act["llr"][y])))
                obs_target[uav] = qq
                obs_llr[uav] = llr_obs
                L[uav, qq] += llr_obs
            # service / concurrency / regret bookkeeping
            for uav in range(k):
                qq = int(obs_target[uav])
                if qq < 0:
                    continue
                j = undecided.index(qq)
                j_best = int(np.argmax(Jmat[uav]))
                regret_sum[r] += float(Jmat[uav, j_best] - Jmat[uav, j])
                regret_n[r] += 1.0
                if j_best != j:
                    distorted[r] += 1.0
                distorted_n[r] += 1.0
            for qq in undecided:
                served = target_count[qq] >= 1
                service[r, qq] += 1.0 if served else 0.0
                if served:
                    last_served[qq] = float(t)
                else:
                    idle_max[r, qq] = max(idle_max[r, qq],
                                          t - last_served[qq])
                nq_series[t, qq] = target_count[qq]
            # urgency-allocation correlation on undecided targets
            if len(undecided) >= 2:
                U = Jmat.max(axis=0)
                nvec = np.array([target_count[qq] for qq in undecided])
                j_med_all[r] += float(np.median(np.abs(U)))
                j_spread_all[r] += float(np.ptp(U))
                if np.any(nvec > 0) and np.std(U) > 1e-12 \
                        and np.std(nvec) > 0:
                    rho_sum[r] += float(np.corrcoef(
                        rankdata(U), rankdata(nvec))[0, 1])
                    rho_n[r] += 1.0
            # token exchange
            intents_next = np.full((k, k), -1, dtype=int)
            for uav in range(k):
                qq = int(obs_target[uav])
                if qq < 0:
                    continue
                for neighbor in range(k):
                    if neighbor == uav:
                        continue
                    succ = (delivery_override
                            if delivery_override is not None
                            else u2u[uav, neighbor])
                    if rng.random() > succ:
                        continue
                    if not decided[qq]:
                        L[neighbor, qq] += obs_llr[uav]
                    intents_next[neighbor, uav] = int(intents[uav])
            intents_recv = intents_next
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
        nc = int(n_cycles[r])
        nq_run = nq_series[:nc]
        nq_mean_all[r] = nq_run.mean(axis=0)
        nq_p95_all[r] = np.percentile(nq_run, 95, axis=0)
        nq_max_all[r] = nq_run.max(axis=0)

    # aggregates
    e1 = [float(delays[H_all[:, qq], qq].mean()) for qq in range(q)]
    p_fa = [float(declared_h1[~H_all[:, qq], qq].mean()) for qq in range(q)]
    p_md = [float(1.0 - declared_h1[H_all[:, qq], qq].mean())
            for qq in range(q)]
    active = np.maximum(n_cycles, 1.0)
    r_per_target = [float(service[:, qq].sum() / active.sum())
                    for qq in range(q)]
    return {
        "worst_target_delay": float(np.max(e1)),
        "e1_delays": e1,
        "p_fa": p_fa,
        "p_md": p_md,
        "r_min": float(np.min(r_per_target)),
        "r_mean": float(np.mean(r_per_target)),
        "r_per_target": r_per_target,
        "H_max_idle": float(np.max(idle_max)),
        "H_idle_per_target": [float(np.max(idle_max[:, qq]))
                              for qq in range(q)],
        "nbar_per_target": [float(nq_mean_all[:, qq].mean())
                            for qq in range(q)],
        "n95_per_target": [float(nq_p95_all[:, qq].mean())
                           for qq in range(q)],
        "n_max_per_target": [float(nq_max_all[:, qq].max())
                             for qq in range(q)],
        "concurrency_max": float(np.max(nq_max_all)),
        "j_median_scale": float(np.mean(j_med_all)),
        "j_cross_target_spread": float(np.mean(j_spread_all)),
        "rho_alloc": float(np.sum(rho_sum) / np.sum(rho_n))
        if rho_n.sum() > 0 else float("nan"),
        "mean_regret": float(np.sum(regret_sum) / np.sum(regret_n))
        if regret_n.sum() > 0 else float("nan"),
        "distorted_choice_rate": float(np.sum(distorted)
                                       / np.sum(distorted_n))
        if distorted_n.sum() > 0 else float("nan"),
    }


def classify_case(rows: dict) -> dict:
    """The only three allowed conclusions of F0-A (advice/007 section 6).

    Evidence: r_min trend, H_max_idle trend, rho_alloc trend, regret /
    distortion trend, concurrency concentration.  Returns the primary case
    with the full evidence table.
    """
    keys = list(rows)
    first, last = keys[0], keys[-1]
    r0 = rows[first]
    r1 = rows[last]

    def trend(key):
        v0 = r0[key]
        v1 = r1[key]
        if v0 is None or v1 is None or abs(v0) < 1e-12:
            return float("nan")
        return (v1 - v0) / abs(v0)

    ev = {
        "r_min": (r0["r_min"], r1["r_min"],
                  round(trend("r_min"), 3)),
        "r_mean": (r0["r_mean"], r1["r_mean"],
                   round(trend("r_mean"), 3)),
        "H_max_idle": (r0["H_max_idle"], r1["H_max_idle"],
                       round(trend("H_max_idle"), 3)),
        "rho_alloc": (r0["rho_alloc"], r1["rho_alloc"],
                      round(trend("rho_alloc"), 3)),
        "distorted_choice_rate": (
            r0["distorted_choice_rate"], r1["distorted_choice_rate"],
            round(trend("distorted_choice_rate"), 3)),
    }
    # starvation evidence: idle runs grow, service drops
    starvation = ev["H_max_idle"][2] > 0.3 and ev["r_min"][2] < -0.1
    # over-concentration evidence: rho drops and concurrency is high
    overconc = ev["rho_alloc"][2] < -0.1 and r1["concurrency_max"] >= 3
    # insufficiency evidence: rho stays high, regret small, service drops
    insufficiency = ev["rho_alloc"][0] >= 0.7 and \
        r1["distorted_choice_rate"] < 0.05 and ev["r_min"][2] < -0.1
    if insufficiency and not starvation:
        case = "case_1_resources_insufficient"
        next_step = ("the scheduler maps urgency to allocation correctly; "
                     "the loss is Q/K load feasibility -- study the Q/K "
                     "feasible region, do not change the algorithm")
    elif starvation and overconc:
        case = "case_2_3_starvation_and_overconcentration"
        next_step = ("starvation with concentration: apply the "
                     "starvation-age term first (one scalar eta_A), keep "
                     "the linear price; revisit the price curvature only "
                     "if age alone is not enough")
    elif starvation:
        case = "case_2_starvation"
        next_step = ("anti-starvation target scheduling: add the single "
                     "starvation-age term J' = J + eta_A * A_q(t) and "
                     "sweep only eta_A")
    elif overconc:
        case = "case_3_overconcentration"
        next_step = ("distributed load balancing: change the congestion "
                     "price curvature only, psi = -eta * n_q^gamma with "
                     "gamma > 1, and sweep only gamma")
    else:
        case = "case_1_resources_insufficient"
        next_step = ("no starvation/concentration signature; the loss is "
                     "Q/K load feasibility -- study the Q/K feasible "
                     "region")
    return {
        "primary_case": case,
        "next_step": next_step,
        "evidence": ev,
        "starvation_signature": bool(starvation),
        "overconcentration_signature": bool(overconc),
        "insufficiency_signature": bool(insufficiency),
    }
