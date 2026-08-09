"""Joint sensing-power and communication-bit allocation gate.

Each target can spend a shared resource budget on sensing power (which
scales evidence separation) and quantizer bits (which improve report
fidelity).  The joint exact allocation is compared with sensing-only and
communication-only baselines under the same budget.
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

from uav_otfs_isac.joint_power_bit import (
    exact_joint_power_bit_maxmin,
    power_bit_target_options,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_power_bit_gate.json")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--max-power", type=float, default=2.0)
    parser.add_argument("--max-bits", type=int, default=2)
    args = parser.parse_args()

    targets = [
        (0.4, np.array([1.8, 2.0])),
        (0.3, np.array([1.2, 1.4])),
    ]
    power_levels = np.linspace(0.0, args.max_power, 3)
    bit_options = np.arange(args.max_bits + 1, dtype=int)
    summary = []
    for budget in args.budgets:
        joint_groups = [
            power_bit_target_options(
                owner, deltas,
                power_levels=power_levels,
                bit_options=bit_options,
                budget=budget,
                grid=args.grid,
            )
            for owner, deltas in targets
        ]
        sensing_groups = [
            power_bit_target_options(
                owner, deltas,
                power_levels=power_levels,
                bit_options=np.array([1]),
                budget=budget,
                grid=args.grid,
            )
            for owner, deltas in targets
        ]
        comm_groups = [
            power_bit_target_options(
                owner, deltas,
                power_levels=np.array([1.0]),
                bit_options=bit_options,
                budget=budget,
                grid=args.grid,
            )
            for owner, deltas in targets
        ]
        joint = exact_joint_power_bit_maxmin(joint_groups, budget)
        sensing = exact_joint_power_bit_maxmin(sensing_groups, budget)
        comm = exact_joint_power_bit_maxmin(comm_groups, budget)
        summary.append({
            "budget": budget,
            "joint_worst_pd": float(joint),
            "sensing_only_worst_pd": float(sensing),
            "communication_only_worst_pd": float(comm),
            "joint_vs_sensing_pp": float((joint - sensing) * 100.0),
            "joint_vs_communication_pp": float((joint - comm) * 100.0),
        })
    payload = {
        "gate": "joint-power-bit-allocation",
        "max_power": args.max_power,
        "max_bits": args.max_bits,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
