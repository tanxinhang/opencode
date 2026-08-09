"""Scaling benchmark: winner-take-all versus full power-bit enumeration."""

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

from uav_otfs_isac.power_split_theory import (
    proportional_power_bit_options,
    winner_take_all_proportional_options,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/winner_take_all_scaling_benchmark.json")
    parser.add_argument("--reports", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    rows = []
    for reports in args.reports:
        deltas = np.linspace(1.0, 2.0, reports)
        start = time.perf_counter()
        full = proportional_power_bit_options(
            0.4,
            deltas,
            power_levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            bit_options=np.array([0, 1, 2]),
            budget=args.budget,
            grid=args.grid,
        )
        full_seconds = time.perf_counter() - start
        start = time.perf_counter()
        winner = winner_take_all_proportional_options(
            0.4,
            deltas,
            bit_options=np.array([0, 1, 2]),
            budget=args.budget,
            grid=args.grid,
        )
        winner_seconds = time.perf_counter() - start
        full_dict = dict(full)
        winner_dict = dict(winner)
        equal = (
            set(full_dict) == set(winner_dict)
            and all(
                abs(full_dict[cost] - winner_dict[cost]) < 1e-9
                for cost in full_dict
            )
        )
        rows.append({
            "reports": reports,
            "full_seconds": full_seconds,
            "winner_seconds": winner_seconds,
            "speedup": full_seconds / max(winner_seconds, 1e-9),
            "equal_frontier": bool(equal),
            "frontier_options": len(full_dict),
        })
    payload = {
        "gate": "winner-take-all-scaling",
        "passed": all(row["equal_frontier"] for row in rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
