"""Joint winner-take-all versus full power-bit allocation (proportional)."""

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
from uav_otfs_isac.power_split_theory import (
    proportional_power_bit_options,
    winner_take_all_proportional_options,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/winner_take_all_joint_proportional_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        deltas_a = rng.uniform(0.8, 2.0, 3)
        deltas_b = rng.uniform(0.8, 2.0, 3)
        full_groups = [
            proportional_power_bit_options(
                0.4, deltas_a,
                power_levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                bit_options=np.array([0, 1, 2]),
                budget=args.budget, grid=32,
            ),
            proportional_power_bit_options(
                0.3, deltas_b,
                power_levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                bit_options=np.array([0, 1, 2]),
                budget=args.budget, grid=32,
            ),
        ]
        winner_groups = [
            winner_take_all_proportional_options(
                0.4, deltas_a,
                bit_options=np.array([0, 1, 2]),
                budget=args.budget, grid=32,
            ),
            winner_take_all_proportional_options(
                0.3, deltas_b,
                bit_options=np.array([0, 1, 2]),
                budget=args.budget, grid=32,
            ),
        ]
        full_value = exact_joint_power_bit_maxmin(full_groups, args.budget)
        winner_value = exact_joint_power_bit_maxmin(
            winner_groups, args.budget
        )
        rows.append({
            "seed": seed,
            "full_worst_pd": float(full_value),
            "winner_worst_pd": float(winner_value),
            "equal": bool(abs(winner_value - full_value) < 1e-9),
        })
    payload = {
        "gate": "winner-take-all-joint-proportional",
        "passed": all(row["equal"] for row in rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": payload["passed"],
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
