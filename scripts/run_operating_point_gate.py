"""Gate F0-G1: policy-matched operating point (advice/009).

The thresholds are currently calibrated on a single-stream
information-per-cost selector, so the DEPLOYED policy realizes P_FA = 0
(far below alpha): the reported delays are not at the achievable
operating point.  This gate scans the upper threshold A_q per target
(greedy coordinate descent, others fixed at the reference) under the
DEPLOYED mainline (compact token, normalized dual-G, scale-adaptive
price), keeps the feasible candidate minimizing the worst-target E1[T]
(P_FA <= alpha, P_MD <= beta with tolerance), and verifies the winner at
high MC.  The lower threshold B_q stays at the reference (second-order
for the delay metric).  Everything else is frozen.

Writes ``results/operating_point_gate.json``.
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

from uav_otfs_isac.competition_audit import simulate_competition_audit
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
)

SCALES = ((12, 6), (16, 8))
A_STEPS = (0.0, -0.3, -0.6, -0.9, -1.2)   # candidate A reductions


def eval_bounds(sc, q, bounds, singles, nu, n_runs, seeds, max_steps,
                eta, alpha, beta):
    J, md, fa = [], [], []
    for seed in range(seeds):
        out = simulate_competition_audit(
            sc, bounds, singles, nu, n_runs=n_runs,
            seed=seed * 1000 + 7, max_steps=max_steps,
            eta=eta, normalize_gains=True)
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        fa.append(max(out["p_fa"]))
    return {"J": float(np.mean(J)), "p_md_max": float(np.max(md)),
            "p_fa_max": float(np.max(fa))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/operating_point_gate.json")
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--verify-runs", type=int, default=500)
    parser.add_argument("--verify-seeds", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.time()
    rows = {}
    for (k, q) in SCALES:
        rng = np.random.default_rng(args.scenario_seed)
        sc = build_distributed_scenario(rng, k_uavs=k, q_targets=q)
        nu = tuple([1.0 / q] * q)
        ref = calibrate_target_bounds(sc, args.alpha, args.beta,
                                      n_runs=300, seed=args.calib_seed,
                                      llr_bits=TOKEN_LLR_BITS,
                                      verify_runs=2000)
        eta_use = args.eta if k <= 12 else 0.0   # scale-adaptive price
        bounds = [list(b) for b in ref]
        # greedy coordinate scan over A_q (B fixed at the reference)
        for qq in range(q):
            best_row = None
            for step in A_STEPS:
                cand = [list(b) for b in bounds]
                cand[qq][0] = max(ref[qq][0] + step, ref[qq][1] + 0.1)
                singles = build_target_values(sc, cand,
                                              horizon=args.max_steps,
                                              nu=nu)
                row = eval_bounds(sc, q, cand, singles, nu,
                                  args.n_runs, args.seeds,
                                  args.max_steps, eta_use,
                                  args.alpha, args.beta)
                feasible = (row["p_fa_max"] <= args.alpha + 0.02
                            and row["p_md_max"] <= args.beta + 0.02)
                if feasible and (best_row is None
                                 or row["J"] < best_row[1]["J"]):
                    best_row = (cand, row)
            if best_row is not None:
                bounds = best_row[0]
        # verify: reference vs matched, high MC
        singles_ref = build_target_values(sc, ref,
                                          horizon=args.max_steps, nu=nu)
        singles_new = build_target_values(sc, bounds,
                                          horizon=args.max_steps, nu=nu)
        ref_v = eval_bounds(sc, q, ref, singles_ref, nu,
                            args.verify_runs, args.verify_seeds,
                            args.max_steps, eta_use, args.alpha,
                            args.beta)
        new_v = eval_bounds(sc, q, bounds, singles_new, nu,
                            args.verify_runs, args.verify_seeds,
                            args.max_steps, eta_use, args.alpha,
                            args.beta)
        # the moderate-MC scan can pick a candidate that violates at the
        # authoritative MC; fall back to the reference in that case
        feasible_new = (new_v["p_fa_max"] <= args.alpha + 0.02
                        and new_v["p_md_max"] <= args.beta + 0.02)
        if not feasible_new or new_v["J"] >= ref_v["J"]:
            bounds = [list(b) for b in ref]
            new_v = ref_v
            improved = False
        else:
            improved = True
        rows[f"{k}_{q}"] = {
            "k": k, "q": q, "eta_used": eta_use,
            "ref_bounds": [[round(b[0], 3), round(b[1], 3)] for b in ref],
            "matched_bounds": [[round(b[0], 3), round(b[1], 3)]
                               for b in bounds],
            "ref": ref_v,
            "matched": new_v,
            "improved": bool(improved),
        }
        print(f"({k},{q}) eta {eta_use}: ref J {ref_v['J']:.3f} "
              f"(P_FA {ref_v['p_fa_max']:.3f}, P_MD {ref_v['p_md_max']:.3f})"
              f" -> {'matched' if improved else 'fallback'} "
              f"J {new_v['J']:.3f} "
              f"(P_FA {new_v['p_fa_max']:.3f}, P_MD {new_v['p_md_max']:.3f})"
              f"  delta {new_v['J']-ref_v['J']:+.3f}", flush=True)
    payload = {
        "gate": "f0g1-policy-matched-operating-point",
        "params": {
            "scales": [list(s) for s in SCALES],
            "n_runs": args.n_runs, "seeds": args.seeds,
            "verify_runs": args.verify_runs,
            "verify_seeds": args.verify_seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "scenario_seed": args.scenario_seed,
            "calib_seed": args.calib_seed,
            "a_steps": list(A_STEPS),
            "frozen": ["fixed owner", "full mesh", "19-bit token",
                       "normalized dual-G + scale-adaptive price",
                       "current scenario gen"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
