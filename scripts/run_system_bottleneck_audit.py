"""System Bottleneck Audit v2 (advice/022).

FRIDS-v2 is FROZEN.  Four ISOLATED oracle gaps measure the headroom of
each system layer on the worst-target delay, plus the already-audited
local-dual gap (G9A):

    Delta_sensing = J_current - J_ideal_evidence
                    (all sensing SNRs boosted: the sensing headroom)
    Delta_comm    = J_current - J_perfect_U2U
                    (perfect U2U delivery into the owner belief)
    Delta_owner   = J_current - J_best_static_owner
                    (offline best static owner assignment)
    Delta_mobility= J_mobile - J_static
                    (frozen-policy response to bounded evidence drift)
    Delta_dual    ~ 1.8%  (G9A common-price oracle)

The layer with the largest gap (> 5%) is the only candidate for the next
one-variable repair; everything else is frozen (audit -> oracle gap ->
bottleneck classification -> one-variable repair -> life gate).

Also reports the across-scenario statistics (median / p90 / max of
J_s = max_q E1[T_q]) and a paired bootstrap CI on the mean gaps, per the
statistical-generalization requirement (advice/022 section 1).

Writes ``results/system_bottleneck_audit.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import g_reliable, simulate_frids_v2

SNR_SHIFT = 4.0        # dB, ideal-evidence oracle
MOBILITY = 0.05        # relative per-cycle evidence change (mobile stress)
OWNER_CANDIDATES = 4   # static owner assignments tried per scenario


def eval_sim(sc, bounds, n_runs, seeds, max_steps, **kw):
    J = []
    for seed in range(seeds):
        out = simulate_frids_v2(sc, bounds, n_runs=n_runs,
                                seed=seed * 1000 + 7, max_steps=max_steps,
                                **kw)
        J.append(out["worst_target_delay"])
    return float(np.mean(J))


def best_static_owner(sc, bounds, n_runs, seeds, max_steps, rng):
    """Best static owner assignment among the fixed + greedy + random."""
    k, q = sc["k"], sc["q"]
    owner_of = sc["owner_of"]
    cand = [list(owner_of)]
    # greedy: owner maximizes the aggregate reliable info into it
    greedy = []
    for qq in range(q):
        best_o, best_g = owner_of[qq], -np.inf
        for o in range(k):
            g_tot = sum(g_reliable(sc, i, qq, [o] * q) for i in range(k))
            if g_tot > best_g:
                best_g = g_tot
                best_o = o
        greedy.append(best_o)
    cand.append(greedy)
    for _ in range(max(1, OWNER_CANDIDATES - 2)):
        perm = list(range(k))
        rng.shuffle(perm)
        cand.append([int(perm[qq % k]) for qq in range(q)])
    best_J = np.inf
    best_o = None
    for o in cand:
        sc_o = dict(sc)
        sc_o["owner_of"] = list(o)
        J = eval_sim(sc_o, bounds, n_runs, seeds, max_steps)
        if J < best_J:
            best_J = J
            best_o = o
    return best_J, best_o


def bootstrap_ci(gaps, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = np.asarray(gaps, dtype=float)
    draws = np.array([rng.choice(g, size=len(g), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/system_bottleneck_audit.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=400)
    parser.add_argument("--b-delta", type=float, default=1.0)
    parser.add_argument("--scenario-seeds", type=int, default=3)
    parser.add_argument("--dual-gap", type=float, default=0.018,
                        help="G9A local-dual gap (fraction)")
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q
    per_scenario = {}
    for s in range(args.scenario_seeds):
        rng = np.random.default_rng(s)
        sc = build_distributed_scenario(rng, k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(sc, args.alpha, args.beta, n_runs=300,
                                     seed=args.calib_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=args.calib_verify)
        bounds = [[bt[qq][0], bt[qq][1] - args.b_delta] for qq in range(q)]

        J_base = eval_sim(sc, bounds, args.n_runs, args.seeds,
                          args.max_steps)
        # perfect U2U (delivery into the owner belief)
        ones = np.ones((k, k))
        J_comm = eval_sim(sc, bounds, args.n_runs, args.seeds,
                          args.max_steps, delivery_matrix=ones,
                          s_for_g=ones)
        # ideal evidence: uniformly strong sensing (recalibrated)
        sc_hi = build_distributed_scenario(np.random.default_rng(s),
                                           k_uavs=k, q_targets=q,
                                           snr_shift=SNR_SHIFT)
        bt_hi = calibrate_target_bounds(sc_hi, args.alpha, args.beta,
                                        n_runs=300, seed=args.calib_seed,
                                        llr_bits=TOKEN_LLR_BITS,
                                        verify_runs=args.calib_verify)
        bounds_hi = [[bt_hi[qq][0], bt_hi[qq][1] - args.b_delta]
                     for qq in range(q)]
        J_sensing = eval_sim(sc_hi, bounds_hi, args.n_runs, args.seeds,
                             args.max_steps)
        # best static owner
        J_owner, best_o = best_static_owner(
            sc, bounds, args.n_runs, args.seeds, args.max_steps,
            np.random.default_rng(1000 + s))
        # mobile stress (frozen policy)
        J_mobile = eval_sim(sc, bounds, args.n_runs, args.seeds,
                            args.max_steps, mobility=MOBILITY)

        per_scenario[str(s)] = {
            "J_current": J_base, "J_perfect_u2u": J_comm,
            "J_ideal_evidence": J_sensing, "J_best_owner": J_owner,
            "J_mobile": J_mobile, "best_owner": best_o,
            "d_sensing": float(J_base - J_sensing),
            "d_comm": float(J_base - J_comm),
            "d_owner": float(J_base - J_owner),
            "d_mobility": float(J_mobile - J_base),
            "rel_sensing": float((J_base - J_sensing) / max(J_base, 1e-12)),
            "rel_comm": float((J_base - J_comm) / max(J_base, 1e-12)),
            "rel_owner": float((J_base - J_owner) / max(J_base, 1e-12)),
            "rel_mobility": float((J_mobile - J_base) / max(J_base, 1e-12)),
        }
        print(f"  scenario {s}: base {J_base:.2f} | sensing {J_sensing:.2f} "
              f"({per_scenario[str(s)]['d_sensing']:+.2f}) | u2u "
              f"{J_comm:.2f} ({per_scenario[str(s)]['d_comm']:+.2f}) | "
              f"owner {J_owner:.2f} ({per_scenario[str(s)]['d_owner']:+.2f}) "
              f"| mobile {J_mobile:.2f} "
              f"({per_scenario[str(s)]['d_mobility']:+.2f}) "
              f"({time.time()-t0:.0f}s)", flush=True)

    # across-scenario statistics (median / p90 / max of J_s)
    def stats(key):
        vals = [per_scenario[s][key] for s in per_scenario]
        return {"median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90)),
                "max": float(np.max(vals))}

    def gap_stats(rel_key):
        vals = [per_scenario[s][rel_key] for s in per_scenario]
        lo, hi = bootstrap_ci(vals)
        return {"mean": float(np.mean(vals)),
                "ci95": [float(lo), float(hi)],
                "per_scenario": [float(x) for x in vals]}

    gaps = {
        "sensing": gap_stats("rel_sensing"),
        "comm": gap_stats("rel_comm"),
        "owner": gap_stats("rel_owner"),
        "mobility": gap_stats("rel_mobility"),
        "dual": {"mean": args.dual_gap, "ci95": [0.0, args.dual_gap],
                 "per_scenario": [args.dual_gap] * args.scenario_seeds},
    }
    means = {k: gaps[k]["mean"] for k in gaps}
    argmax_layer = max(means, key=lambda k: means[k])
    largest = means[argmax_layer]
    repair = bool(largest > 0.05)
    gate = {
        "headroom_decomposition": {
            layer: {"gap_mean": float(gaps[layer]["mean"]),
                    "ci95": gaps[layer]["ci95"]}
            for layer in gaps},
        "largest_gap_layer": argmax_layer,
        "largest_gap": float(largest),
        "exceeds_5pct": bool(repair),
        "next_repair": (
            f"one-variable repair on the {argmax_layer} layer "
            f"(gap {largest:.1%} > 5%)"
            if repair
            else "no layer exceeds 5%; freeze all layers and write up"),
        "J_stats": {layer: stats(f"J_{layer}")
                    for layer in ("current", "perfect_u2u",
                                  "ideal_evidence", "best_owner", "mobile")},
    }

    payload = {
        "audit": "system-bottleneck-v2",
        "params": {
            "K": k, "Q": q, "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps, "alpha": args.alpha,
            "beta": args.beta, "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify, "b_delta": args.b_delta,
            "scenario_seeds": args.scenario_seeds,
            "snr_shift_db": SNR_SHIFT, "mobility": MOBILITY,
            "owner_candidates": OWNER_CANDIDATES,
            "dual_gap_from": "F0-G9A common-price oracle",
            "frozen": ["FRIDS-v2", "token", "owner", "full mesh",
                       "calibration protocol", "(K,Q)", "all params"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "per_scenario": per_scenario,
        "gaps": gaps,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()