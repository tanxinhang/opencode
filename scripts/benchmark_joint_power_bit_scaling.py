"""Scaling benchmark for joint power-bit option enumeration."""

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

from uav_otfs_isac.joint_power_bit import power_bit_target_options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_power_bit_scaling_benchmark.json")
    parser.add_argument("--reports", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--grid", type=int, default=32)
    args = parser.parse_args()

    rows = []
    for reports in args.reports:
        deltas = np.linspace(1.0, 2.0, reports)
        start = time.perf_counter()
        options = power_bit_target_options(
            0.4,
            deltas,
            power_levels=np.array([0.0, 1.0, 2.0]),
            bit_options=np.array([0, 1, 2]),
            budget=3 * reports,
            grid=args.grid,
        )
        seconds = time.perf_counter() - start
        rows.append({
            "reports": reports,
            "grid": args.grid,
            "seconds": seconds,
            "frontier_options": len(options),
        })
    payload = {
        "gate": "joint-power-bit-scaling",
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
