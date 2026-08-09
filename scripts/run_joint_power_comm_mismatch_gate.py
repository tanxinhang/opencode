"""Communication-channel mismatch gate for NOMP power-bit refinement.

The sensing channel determines the per-report deflection while the
communication channel adds BSC bit flips and link erasures.  The robust exact
oracle evaluates every option at the worst endpoint `(flip_hi, success_lo)`,
and the online NOMP refinement uses the same endpoint P_D for its
lexicographic max-min moves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.nomp_refinement import (
    nomp_wta_greedy_joint_multi,
    wta_greedy_joint_multi,
)
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_robust_power_bit_options,
    pareto_options,
)
from scripts.run_joint_power_comparison import make_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_power_comm_mismatch_gate.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--reports", type=int, default=2)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--flip-hi", type=float, default=0.2)
    parser.add_argument("--success-lo", type=float, default=0.7)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    summary = []
    for budget in args.budgets:
        wta_worsts = []
        nomp_worsts = []
        exact_worsts = []
        nomp_rounds = []
        for seed in range(args.seeds):
            scenario = make_scenario(
                10000 + seed,
                args.reports,
                args.targets,
                heterogeneous=True,
            )
            wta_worsts.append(float(wta_greedy_joint_multi(
                scenario,
                budget,
                min_cover=False,
                flip_probability=args.flip_hi,
                success_probability=args.success_lo,
            )["worst_pd"]))
            result = nomp_wta_greedy_joint_multi(
                scenario,
                budget,
                flip_probability=args.flip_hi,
                success_probability=args.success_lo,
            )
            nomp_worsts.append(float(result["worst_pd"]))
            nomp_rounds.append(int(result["refine_rounds"]))
            groups = []
            for target in scenario:
                options = enumerate_robust_power_bit_options(
                    float(target[0]),
                    target[1:],
                    power_levels=np.arange(budget + 1, dtype=float),
                    bit_options=np.arange(3, dtype=int),
                    budget=budget,
                    flip_interval=(0.0, args.flip_hi),
                    success_interval=(args.success_lo, 1.0),
                    grid=args.grid,
                )
                groups.append(pareto_options(options, "robust_pd"))
            exact_worsts.append(float(exact_joint_power_bit_maxmin(
                groups, budget
            )))
        wta_mean = float(np.mean(wta_worsts))
        nomp_mean = float(np.mean(nomp_worsts))
        exact_mean = float(np.mean(exact_worsts))
        summary.append({
            "budget": budget,
            "wta_greedy_worst_mean": wta_mean,
            "nomp_greedy_worst_mean": nomp_mean,
            "robust_exact_worst_mean": exact_mean,
            "wta_gap_to_exact": float(exact_mean - wta_mean),
            "nomp_gap_to_exact": float(exact_mean - nomp_mean),
            "nomp_mean_refine_rounds": float(np.mean(nomp_rounds)),
        })

    payload = {
        "gate": "joint-power-comm-mismatch",
        "flip_hi": args.flip_hi,
        "success_lo": args.success_lo,
        "seeds": args.seeds,
        "targets": args.targets,
        "reports": args.reports,
        "grid": args.grid,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
