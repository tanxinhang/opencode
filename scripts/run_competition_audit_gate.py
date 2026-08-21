"""Gate F0-A: target competition audit (advice/007.md).

Runs the frozen compact-token mainline at the paired scales
(K, Q) = (6,3), (8,4), (12,6), (16,8) and records five per-cycle
allocation diagnostics (service rate, max idle run, concurrent UAV
counts, urgency-allocation correlation rho_alloc, allocation regret /
distorted-choice rate).  No decision mechanism is changed.  The verdict is
one of the three allowed conclusions (advice/007 section 6): resources
insufficient (Q/K feasibility), starvation (prescribed fix: one
starvation-age term eta_A), or over-concentration (prescribed fix:
super-linear congestion price gamma > 1).

Writes ``results/competition_audit_gate.json``.
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

from uav_otfs_isac.competition_audit import (
    classify_case,
    simulate_competition_audit,
)
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
)

SCALES = ((6, 3), (8, 4), (12, 6), (16, 8))


def run_competition_audit(scales=SCALES, n_runs=400, seeds=4,
                          max_steps=40, alpha=0.05, beta=0.05,
                          scenario_seed=0, calib_seed=100,
                          calib_verify_runs=2000):
    """Frozen mainline + diagnostics at every scale; returns (rows,
    verdict)."""
    rows = {}
    for (k, q) in scales:
        t0 = time.time()
        rng = np.random.default_rng(scenario_seed)
        scenario = build_distributed_scenario(rng, k_uavs=k,
                                              q_targets=q)
        bt = calibrate_target_bounds(scenario, alpha, beta,
                                     n_runs=300, seed=calib_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=calib_verify_runs)
        nu = tuple([1.0 / q] * q)
        singles = build_target_values(scenario, bt, horizon=max_steps,
                                      nu=nu)
        agg = {key: [] for key in (
            "worst_target_delay", "r_min", "r_mean", "r_per_target",
            "H_max_idle", "H_idle_per_target", "nbar_per_target",
            "n95_per_target", "n_max_per_target", "concurrency_max",
            "j_median_scale", "j_cross_target_spread",
            "rho_alloc", "mean_regret", "distorted_choice_rate",
            "p_fa", "p_md",
        )}
        for seed in range(seeds):
            out = simulate_competition_audit(
                scenario, bt, singles, nu, n_runs=n_runs,
                seed=seed * 1000 + 7, max_steps=max_steps,
            )
            for key in agg:
                agg[key].append(out[key])
        def mean(key):
            vals = agg[key]
            if key in ("r_per_target", "H_idle_per_target",
                       "nbar_per_target", "n95_per_target",
                       "n_max_per_target", "p_fa", "p_md"):
                return [float(np.mean([v[i] for v in vals]))
                        for i in range(q)]
            return float(np.mean(vals))
        rows[f"{k}_{q}"] = {
            "k": k, "q": q,
            **{key: mean(key) for key in agg},
            "runtime_s": round(time.time() - t0, 1),
        }
    verdict = classify_case(rows)
    return rows, verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/competition_audit_gate.json")
    parser.add_argument("--n-runs", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify-runs", type=int, default=2000)
    args = parser.parse_args()

    t0 = time.time()
    rows, verdict = run_competition_audit(
        scales=SCALES, n_runs=args.n_runs, seeds=args.seeds,
        max_steps=args.max_steps, alpha=args.alpha, beta=args.beta,
        scenario_seed=args.scenario_seed, calib_seed=args.calib_seed,
        calib_verify_runs=args.calib_verify_runs,
    )
    payload = {
        "gate": "f0a-target-competition-audit",
        "params": {
            "scales": [list(s) for s in SCALES],
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "scenario_seed": args.scenario_seed,
            "calib_seed": args.calib_seed,
            "calib_verify_runs": args.calib_verify_runs,
            "token_llr_bits": TOKEN_LLR_BITS,
            "frozen": ["fixed owner", "full mesh", "19-bit token",
                       "dual-G + linear congestion price", "calibrated "
                       "two-threshold stopping", "current scenario gen"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'scale':<8}{'J':>7}{'r_min':>7}{'r_mean':>8}"
          f"{'H_idle':>8}{'n_max':>7}{'rho':>7}{'distort':>9}")
    for key, row in rows.items():
        print(f"{key:<8}{row['worst_target_delay']:>7.2f}"
              f"{row['r_min']:>7.3f}{row['r_mean']:>8.3f}"
              f"{row['H_max_idle']:>8.1f}{row['concurrency_max']:>7.1f}"
              f"{row['rho_alloc']:>7.3f}"
              f"{row['distorted_choice_rate']:>9.4f}")
    print(f"primary case: {verdict['primary_case']}")
    print(f"next step: {verdict['next_step']}")
    print("evidence:", json.dumps(verdict["evidence"], indent=1))
    print(f"signatures: starvation={verdict['starvation_signature']}, "
          f"overconcentration={verdict['overconcentration_signature']}, "
          f"insufficiency={verdict['insufficiency_signature']}")


if __name__ == "__main__":
    main()
