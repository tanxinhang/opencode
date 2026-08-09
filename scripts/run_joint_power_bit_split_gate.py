"""Optimal sensing-power versus communication-bit resource split.

The exact max-min solution now returns the concrete chosen options.  This
gate reports how much of the budget is spent on sensing power and how much
on communication bits at each budget.
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

from uav_otfs_isac.joint_allocation import exact_joint_maxmin_selection
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_robust_power_bit_options,
    pareto_options,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_power_bit_split_gate.json")
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
        groups = []
        option_lists = []
        for owner, deltas in targets:
            options = enumerate_robust_power_bit_options(
                owner,
                deltas,
                power_levels=power_levels,
                bit_options=bit_options,
                budget=budget,
                flip_interval=(0.0, 0.0),
                success_interval=(1.0, 1.0),
                grid=args.grid,
            )
            option_lists.append(options)
            groups.append(pareto_options(options, "clean_pd"))
        _, chosen = exact_joint_maxmin_selection(groups, budget)
        power_cost = 0
        bit_cost = 0
        selected = []
        for q, (cost, pd) in enumerate(chosen):
            option = next(
                item for item in option_lists[q]
                if item.cost_bits == cost
                and np.isclose(item.clean_pd, pd)
            )
            power_cost += int(sum(option.powers))
            bit_cost += int(sum(option.bits))
            selected.append({
                "target": q,
                "powers": list(option.powers),
                "bits": list(option.bits),
                "cost_bits": cost,
                "clean_pd": float(pd),
            })
        total = power_cost + bit_cost
        summary.append({
            "budget": budget,
            "power_share": float(power_cost / max(total, 1)),
            "bit_share": float(bit_cost / max(total, 1)),
            "power_cost": power_cost,
            "bit_cost": bit_cost,
            "total_used": total,
            "selected": selected,
        })
    payload = {
        "gate": "joint-power-bit-split",
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
