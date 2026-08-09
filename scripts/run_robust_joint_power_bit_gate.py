"""Robust joint sensing-communication resource allocation gate."""

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
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_robust_power_bit_options,
    pareto_options,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robust_joint_power_bit_gate.json")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--grid", type=int, default=32)
    args = parser.parse_args()

    targets = [
        (0.4, np.array([1.8, 2.0])),
        (0.3, np.array([1.2, 1.4])),
    ]
    power_levels = np.array([0.0, 1.0, 2.0])
    bit_options = np.array([0, 1, 2])
    summary = []
    for budget in args.budgets:
        robust_groups = []
        clean_groups = []
        target_options = []
        for owner, deltas in targets:
            options = enumerate_robust_power_bit_options(
                owner,
                deltas,
                power_levels=power_levels,
                bit_options=bit_options,
                budget=budget,
                flip_interval=(0.0, 0.2),
                success_interval=(0.5, 1.0),
                grid=args.grid,
            )
            target_options.append(options)
            robust_groups.append(pareto_options(options, "robust_pd"))
            clean_groups.append(pareto_options(options, "clean_pd"))
        robust_value = exact_joint_power_bit_maxmin(robust_groups, budget)
        clean_value = exact_joint_power_bit_maxmin(clean_groups, budget)
        clean_schedule_robust = []
        for options in target_options:
            candidates = [
                option for option in options
                if option.clean_pd >= clean_value - 1e-9
            ]
            chosen = min(
                candidates, key=lambda option: option.cost_bits
            )
            clean_schedule_robust.append(chosen.robust_pd)
        clean_schedule_robust_worst = float(np.min(clean_schedule_robust))
        summary.append({
            "budget": budget,
            "robust_worst_pd": float(robust_value),
            "clean_optimal_clean_worst_pd": float(clean_value),
            "clean_schedule_robust_worst_pd": clean_schedule_robust_worst,
            "robust_improvement_pp": float(
                (robust_value - clean_schedule_robust_worst) * 100.0
            ),
        })
    payload = {
        "gate": "robust-joint-power-bit",
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
