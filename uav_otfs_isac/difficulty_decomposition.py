"""Gate F0-D: target difficulty decomposition (advice/008.md).

The remaining question after F0-A is whether the 16/8 worst-target delay
still contains optimizable distributed loss, or whether it is dominated by
intrinsic target difficulty and the max-over-Q extreme-sample effect.  F0-D
is a pure paired diagnostic -- no algorithm change.  The frozen system is
the corrected mainline (compact token, dual-G with the normalized
coordination of F0-A: per-UAV per-cycle gain normalization + linear
congestion price eta = 1, fixed owner, full mesh, 19-bit token, calibrated
two-threshold stopping, current scenario generation).

For every target q of a scale (K, Q) the gate computes three numbers on
the SAME realization (same kernels, owners, resource limits, thresholds):

- J_q^iso : the target run alone (only q's kernels active; the same K
  UAVs, the same owner, the same per-UAV action budget and the same
  thresholds) -- the intrinsic difficulty;
- J_q^cent: the multi-target run under the centralized audit oracle
  (global belief, perfect delivery) -- intrinsic difficulty + multi-target
  resource competition;
- J_q^dist: the deployed distributed run -- the full decomposition

    J_q^dist = J_q^iso + Delta_q^comp + Delta_q^dec,
    Delta_q^comp = J_q^cent - J_q^iso   (competition cost),
    Delta_q^dec  = J_q^dist - J_q^cent  (decentralization cost),

plus, for the hardest targets, the difficulty fingerprint
(I^+_{q,max}, C_{q,max}, N_q^useful) and the information-theoretic delay
floor T_q^LB = d(1-beta||alpha) / Ibar_q^max (Wald/relative-entropy lower
bound) as a sanity check of how close J^iso is to the information-physical
limit.

Verdict on the hardest target (max J^dist), by share of J^dist:

- Case A (intrinsic dominates): J^iso / J^dist >= 0.9  -> stop optimizing
  the scheduler; study the (Q/K, difficulty-distribution) feasibility
  region;
- Case B (competition dominates): (J^cent - J^iso)/J^dist >= 0.1 and
  larger than the decentralization share -> resource capacity / load
  feasibility problem;
- Case C (decentralization dominates): (J^dist - J^cent)/J^dist >= 0.1
  and larger than the competition share -> there is still distributed
  allocation headroom worth optimizing.

The distribution of {J_q^iso} over q (median / p90 / max) is reported per
scale: if the median stays flat while the max rises with Q, the F0-S
worst-target growth is mainly the max-over-Q extreme-sample effect.
"""

from __future__ import annotations

import numpy as np

from uav_otfs_isac.competition_audit import simulate_competition_audit
from uav_otfs_isac.distributed_audit import (
    build_target_values,
    calibrate_target_bounds,
    simulate_system,
)

EPS_USEFUL = 0.05   # min best-kernel I+ for a UAV to count as informative


def d_kl_binary(a: float, b: float) -> float:
    """Binary relative entropy ``a log(a/b) + (1-a) log((1-a)/(1-b))``."""
    a = float(np.clip(a, 1e-12, 1 - 1e-12))
    b = float(np.clip(b, 1e-12, 1 - 1e-12))
    return a * np.log(a / b) + (1.0 - a) * np.log((1.0 - a) / (1.0 - b))


def isolated_scenario(scenario: dict, q: int) -> dict:
    """The same realization with only target ``q`` active: same UAVs, the
    same per-(UAV, q) kernels, the same owner, the same U2U reliability
    matrix, and the same per-UAV action budget (one action per cycle).
    The single target takes index 0; ``owner_of = [q]`` keeps the same
    owner UAV."""
    k = scenario["k"]
    links = {}
    by_host = {}
    for i in range(k):
        copied = [dict(act) for act in scenario["by_host"][(i, q)]]
        for act in copied:
            act["target"] = 0
        by_host[(i, 0)] = copied
        if 0 not in links:
            links[0] = []
        links[0].extend(copied)
    return {
        "k": k, "q": 1, "l_acc": scenario["l_acc"],
        "links": links, "by_host": by_host,
        "u2u_success": scenario["u2u_success"],
        "owner_of": [q],
    }


def difficulty_fingerprint(scenario: dict, q: int) -> dict:
    """I^+_{q,max} (best post-communication evidence rate), C_{q,max}
    (best Chernoff information), N_q^useful (UAVs whose best kernel keeps
    I+ >= EPS_USEFUL)."""
    i_max = -np.inf
    c_max = -np.inf
    n_useful = 0
    for i in range(scenario["k"]):
        best_i = -np.inf
        for act in scenario["by_host"][(i, q)]:
            i_max = max(i_max, float(act["i_plus"]))
            c_max = max(c_max, float(act["chernoff"]))
            best_i = max(best_i, float(act["i_plus"]))
        if best_i >= EPS_USEFUL:
            n_useful += 1
    return {
        "i_plus_max": float(i_max),
        "chernoff_max": float(c_max),
        "n_useful_uavs": int(n_useful),
        "t_lb_per_obs": float(d_kl_binary(1.0 - 0.05, 0.05)
                              / max(i_max, 1e-12)),
    }


def _e1_of(sim_out: dict):
    return list(sim_out["e1_delays"])


def run_decomposition(
    scenario: dict,
    bounds: list,
    singles: list,
    n_runs: int = 300,
    seeds: int = 3,
    max_steps: int = 40,
    eta: float = 1.0,
    beta: float = 0.05,
    alpha: float = 0.05,
) -> dict:
    """Per-target three-way decomposition on the corrected mainline
    (normalized dual-G, linear price eta=1, compact token, fixed owner,
    full mesh, calibrated thresholds)."""
    q = scenario["q"]
    k = scenario["k"]
    nu = tuple([1.0 / q] * q)

    def run(mode, sc, bnd, sing, nuv):
        acc = [[] for _ in range(sc["q"])]
        for seed in range(seeds):
            if mode == "dist":
                out = simulate_competition_audit(
                    sc, bnd, sing, nuv, n_runs=n_runs,
                    seed=seed * 1000 + 7, max_steps=max_steps,
                    eta=eta, normalize_gains=True,
                )
            else:
                # the centralized / full-message references run the SAME
                # corrected index (normalized gains); the centralized
                # oracle needs no congestion price (it has no
                # coordination friction), so eta = 0 there, while the
                # full-message distributed mode keeps the deployed price
                # (methodology correction 2026-08-17: the earlier dec
                # split ran these modes on the raw unnormalized index and
                # the centralized mode with the price, confounding the
                # composition)
                out = simulate_system(
                    mode, sc, bnd, sing, n_runs=n_runs,
                    seed=seed * 1000 + 7, max_steps=max_steps, nu=nuv,
                    eta=0.0 if mode == "centralized" else eta,
                    normalize_gains=True,
                )
            e1 = _e1_of(out)
            for j in range(sc["q"]):
                acc[j].append(e1[j])
        return [float(np.mean(v)) for v in acc]

    # J^dist and J^cent on the full scenario (per-target e1)
    j_dist = run("dist", scenario, bounds, singles, nu)
    j_cent = run("centralized", scenario, bounds, singles, nu)
    # mode B (full-precision messages) splits the decentralization gap
    # into quantization loss (C vs B) and delivery/local-decision loss
    # (B vs centralized), on the same per-target basis
    j_fullmsg = run("full_message", scenario, bounds, singles, nu)

    # J^iso per target: same realization, only q active
    j_iso = []
    iso_singles_cache = {}
    for qq in range(q):
        iso = isolated_scenario(scenario, qq)
        if qq not in iso_singles_cache:
            iso_singles_cache[qq] = build_target_values(
                iso, [bounds[qq]], horizon=max_steps,
                nu=(1.0,),
            )
        j_iso.append(run("dist", iso, [bounds[qq]],
                         iso_singles_cache[qq], (1.0,))[0])

    # difficulty fingerprints and info-theoretic floor
    fingerprints = [difficulty_fingerprint(scenario, qq)
                    for qq in range(q)]

    # decomposition on the hardest target (max J^dist)
    q_star = int(np.argmax(j_dist))
    jd = j_dist[q_star]
    ji = j_iso[q_star]
    jc = j_cent[q_star]
    shares = {
        "J_iso_share": float(ji / max(jd, 1e-12)),
        "comp_share": float((jc - ji) / max(jd, 1e-12)),
        "dec_share": float((jd - jc) / max(jd, 1e-12)),
    }
    if shares["J_iso_share"] >= 0.9:
        case = "case_A_intrinsic_dominates"
        next_step = ("worst-target delay is ~90%+ intrinsic target "
                     "difficulty; stop optimizing the scheduler; study the "
                     "(Q/K, difficulty-distribution) feasibility region")
    elif shares["comp_share"] >= 0.1 and \
            shares["comp_share"] > shares["dec_share"]:
        case = "case_B_competition_dominates"
        next_step = ("multi-target resource competition is the main loss; "
                     "study the capacity / load feasibility region "
                     "(Q/K, B, E) under P_FA <= alpha, P_MD <= beta")
    elif shares["dec_share"] >= 0.1 and \
            shares["dec_share"] > shares["comp_share"]:
        case = "case_C_decentralization_dominates"
        next_step = ("the distributed gap on the hardest target is still "
                     "material; there is distributed allocation headroom "
                     "-- only then consider re-expressing the min-max "
                     "urgency via nu_q")
    else:
        case = "mixed"
        next_step = ("shares are comparable; report the decomposition "
                     "honestly and keep the scheduler frozen")

    iso_arr = np.asarray(j_iso)
    return {
        "j_iso": j_iso,
        "j_cent": j_cent,
        "j_fullmsg": j_fullmsg,
        "j_dist": j_dist,
        "q_star": int(q_star),
        "dec_split": {
            "quantization_share": [float(
                (j_dist[qq] - j_fullmsg[qq]) / max(j_dist[qq], 1e-12))
                for qq in range(q)],
            "delivery_local_share": [float(
                (j_fullmsg[qq] - j_cent[qq]) / max(j_dist[qq], 1e-12))
                for qq in range(q)],
        },
        "iso_distribution": {
            "median": float(np.median(iso_arr)),
            "p90": float(np.percentile(iso_arr, 90)),
            "max": float(np.max(iso_arr)),
        },
        "hardest": {
            "q": int(q_star),
            "J_iso": float(ji),
            "J_cent": float(jc),
            "J_dist": float(jd),
            "shares": shares,
            "fingerprint": fingerprints[q_star],
            "t_lb_per_obs": fingerprints[q_star]["t_lb_per_obs"],
        },
        "fingerprints": fingerprints,
        "case": case,
        "next_step": next_step,
    }
