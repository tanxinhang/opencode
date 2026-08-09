"""Erasure stochastic-dominance gate.

The gate checks that lower link success probabilities never produce a
received set outside a monotone coupling of the cleaner law, and that the
exact expected P_D is nonincreasing as erasure grows.
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

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.erasure_dominance import (
    verify_expected_pd_monotonicity,
    verify_monotone_coupling,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/erasure_dominance_gate.json")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--grid", type=int, default=128)
    args = parser.parse_args()

    high = np.array([0.9, 0.8, 0.7, 0.6])
    lows = [
        np.array([0.6, 0.5, 0.4, 0.3]),
        np.array([0.45, 0.4, 0.35, 0.3]),
    ]
    coupling = []
    for index, low in enumerate(lows):
        coupling.append({
            "pair": index,
            **verify_monotone_coupling(
                high, low, samples=args.samples,
            ),
        })
    pd_rows = []
    for low_success in (0.7, 0.6, 0.5):
        clean = symmetric_diversity_model(
            np.full(4, 1.4), success_probability=0.9
        )
        degraded = symmetric_diversity_model(
            np.full(4, 1.4), success_probability=low_success
        )
        row = verify_expected_pd_monotonicity(
            clean, degraded, grid=args.grid
        )
        pd_rows.append({
            "low_success": low_success,
            **row,
        })
    payload = {
        "gate": "erasure-dominance",
        "coupling": coupling,
        "expected_pd": pd_rows,
        "passed": all(row["passed"] for row in coupling)
        and all(row["passed"] and row["in_theorem_scope"] for row in pd_rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "coupling_passed": [row["passed"] for row in coupling],
        "expected_pd": [
            {
                "low_success": row["low_success"],
                "gap": row["gap"],
                "in_theorem_scope": row["in_theorem_scope"],
            }
            for row in pd_rows
        ],
        "passed": payload["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
