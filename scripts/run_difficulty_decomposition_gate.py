"""Gate F0-D: target difficulty decomposition (advice/008.md).

Pure paired diagnostic on the corrected mainline (normalized dual-G,
eta = 1, compact token, fixed owner, full mesh, calibrated thresholds):
for every target q of the failed scales, decompose

    J_q^dist = J_q^iso + Delta_q^comp + Delta_q^dec

with J^iso from the same realization with only q active, J^cent from the
centralized audit oracle, and J^dist from the deployed system; report the
difficulty fingerprint (I+_max, Chernoff_max, N_useful) and the
information-theoretic delay floor T^LB per observation as a sanity check;
and classify the hardest target as Case A (intrinsic dominates, stop
optimizing the scheduler), Case B (competition dominates, capacity /
load feasibility), or Case C (decentralization dominates, allocation
headroom remains).  No algorithm is modified.

Writes ``results/difficulty_decomposition_gate.json``.
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

from uav_otfs_isac.difficulty_decomposition import run_decomposition
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
)

SCALES = ((6, 3), (12, 6), (16, 8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/difficulty_decomposition_gate.json")
    parser.add_argument("--n-runs", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0,
                        help="congestion price scale on the normalized "
                             "index (the F0-A corrected mainline)")
    args = parser.parse_args()

    t0 = time.time()
    scales = {}
    for (k, q) in SCALES:
        rng = np.random.default_rng(args.scenario_seed)
        scenario = build_distributed_scenario(rng, k_uavs=k,
                                              q_targets=q)
        bt = calibrate_target_bounds(scenario, args.alpha, args.beta,
                                     n_runs=300, seed=args.calib_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=2000)
        nu = tuple([1.0 / q] * q)
        singles = build_target_values(scenario, bt, horizon=args.max_steps,
                                      nu=nu)
        dec = run_decomposition(
            scenario, bt, singles,
            n_runs=args.n_runs, seeds=args.seeds,
            max_steps=args.max_steps,
            alpha=args.alpha, beta=args.beta, eta=args.eta,
        )
        scales[f"{k}_{q}"] = {
            "k": k, "q": q, **dec,
            "runtime_s": round(time.time() - t0, 1),
        }
    payload = {
        "gate": "f0d-target-difficulty-decomposition",
        "params": {
            "scales": [list(s) for s in SCALES],
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "scenario_seed": args.scenario_seed,
            "calib_seed": args.calib_seed,
            "eta": args.eta,
            "token_llr_bits": TOKEN_LLR_BITS,
            "frozen": ["fixed owner", "full mesh", "19-bit token",
                       "dual-G + normalized coordination (eta=1)",
                       "calibrated two-threshold stopping", "current "
                       "scenario gen", "all resource constraints"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "scales": scales,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for key, row in payload["scales"].items():
        print(f"scale {key}: case={row['case']}")
        print(f"  {'q':>3}{'J_iso':>8}{'J_cent':>8}{'J_dist':>8}"
              f"{'iso%':>7}{'comp%':>7}{'dec%':>7}")
        for qq in range(row["q"]):
            ji = row["j_iso"][qq]
            jc = row["j_cent"][qq]
            jd = row["j_dist"][qq]
            print(f"  {qq:>3}{ji:>8.2f}{jc:>8.2f}{jd:>8.2f}"
                  f"{100*ji/max(jd,1e-12):>7.1f}"
                  f"{100*(jc-ji)/max(jd,1e-12):>7.1f}"
                  f"{100*(jd-jc)/max(jd,1e-12):>7.1f}")
        d = row["iso_distribution"]
        print(f"  iso distribution: median {d['median']:.2f}, "
              f"p90 {d['p90']:.2f}, max {d['max']:.2f}")
        h = row["hardest"]
        print(f"  hardest q{h['q']}: iso {h['J_iso']:.2f} / cent "
              f"{h['J_cent']:.2f} / dist {h['J_dist']:.2f}  "
              f"shares { {k: round(v, 3) for k, v in h['shares'].items()} }")
        fp = h["fingerprint"]
        print(f"  fingerprint: I+_max {fp['i_plus_max']:.3f}, "
              f"Chernoff_max {fp['chernoff_max']:.3f}, "
              f"N_useful {fp['n_useful_uavs']}, "
              f"T_LB/obs {fp['t_lb_per_obs']:.2f}")
        print(f"  next: {row['next_step']}")


if __name__ == "__main__":
    main()
