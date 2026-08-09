"""Scaling benchmark for Exact Joint option enumeration and max-min solve.

Per-target option enumeration is linear in the target count, while the
max-min threshold search now uses per-target Pareto frontiers and binary
search, so the threshold feasibility cost is ``O(Q log O)`` instead of
``O(Q O)`` per candidate threshold.
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

from uav_otfs_isac.joint_allocation import (
    exact_joint_maxmin,
    target_options,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/exact_joint_scaling_benchmark.json")
    parser.add_argument("--reports", type=int, default=4)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 4, 8, 16])
    args = parser.parse_args()

    rows = []
    for q_count in args.targets:
        groups = []
        enum_start = time.perf_counter()
        for q in range(q_count):
            deltas = np.linspace(
                0.6 + 0.05 * q,
                2.0 + 0.05 * q,
                args.reports,
            )
            groups.append(target_options(
                0.3 + 0.05 * q, deltas, grid=args.grid,
            ))
        enum_seconds = time.perf_counter() - enum_start
        budget = int(4 * q_count)
        solve_start = time.perf_counter()
        value = exact_joint_maxmin(groups, budget)
        solve_seconds = time.perf_counter() - solve_start
        rows.append({
            "targets": q_count,
            "reports_per_target": args.reports,
            "grid": args.grid,
            "option_enumeration_seconds": enum_seconds,
            "maxmin_solve_seconds": solve_seconds,
            "total_seconds": enum_seconds + solve_seconds,
            "worst_pd": float(value),
        })

    payload = {
        "gate": "exact-joint-scaling",
        "reports_per_target": args.reports,
        "grid": args.grid,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
