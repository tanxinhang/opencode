"""G8-S gate: scaled exact-threshold max-min selection.

The exact max-min selector enumerates every per-target report subset, which
becomes infeasible as the report count grows.  G8-S replaces the subset
enumeration for operating points with P_D >= 0.5 by a minimum-cost
branch-and-bound: a node is pruned only when even adding all remaining
reports cannot reach the threshold, or when the current and cheapest next
cost cannot beat the best known feasible subset.

The gate verifies the scaled selector against exhaustive enumeration on small
controlled models, then demonstrates the branch-and-bound on a synthetic
12-report model that would require 4096 subsets by exhaustive enumeration.
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

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.exact_quota_selection import exact_maxmin_select
from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.scalable_selection import (
    minimum_cost_to_threshold,
    scaled_maxmin_select,
)


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


def _aggregate_by_budget(rows, budget_key, numeric_fields):
    budgets = sorted({row[budget_key] for row in rows})
    out = []
    for budget in budgets:
        cell = [row for row in rows if row[budget_key] == budget]
        entry = {"budget_bits": budget, "n_seeds": len(cell)}
        for field in numeric_fields:
            mean, std = _mean_std([row[field] for row in cell])
            entry[f"{field}_mean"] = mean
            entry[f"{field}_std"] = std
        out.append(entry)
    return out


def _synthetic_model(num_reports: int) -> TargetEvidenceModel:
    n = num_reports + 1
    delta = np.linspace(2.0, 0.2, n)
    model = TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(n),
        mu1=delta,
        sigma0=np.eye(n),
        sigma1=np.eye(n),
        success_prob=np.ones(n),
        report_bits=np.array(
            [0] + [1 + (index % 4) for index in range(1, n)],
            dtype=int,
        ),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )
    model.validate()
    return model


def _scalability_benchmark(grid: int):
    rows = []
    for num_reports in (8, 12, 16, 20, 24, 28, 32, 40):
        model = _synthetic_model(num_reports)
        started = time.perf_counter()
        result = minimum_cost_to_threshold(
            model, 0.8, 0.05, grid=grid, max_cost=30,
            max_exhaustive_reports=0,
        )
        elapsed = time.perf_counter() - started
        rows.append({
            "num_reports": num_reports,
            "exhaustive_subsets": 2 ** num_reports,
            "found": result is not None,
            "min_cost": None if result is None else result[0],
            "scheduled": None if result is None else sorted(result[1]),
            "wall_seconds": elapsed,
        })
    return rows


def run_gate(*, output: Path, seeds: int, budgets, grid: int) -> None:
    rows = []
    max_error = 0.0
    for seed_offset in range(seeds):
        rng = np.random.default_rng(20260808 + seed_offset)
        for budget in budgets:
            models = [
                symmetric_diversity_model(
                    np.sort(rng.uniform(0.6, 1.8, size=4))[::-1],
                    success_probability=float(rng.uniform(0.5, 0.95)),
                    report_bits=np.array([1, 2, 3, 5]),
                )
                for _ in range(3)
            ]
            exact = exact_maxmin_select(
                models, budget, 0.05, grid=grid
            )
            scaled = scaled_maxmin_select(
                models, budget, 0.05, grid=grid, tolerance=1e-7
            )
            error = abs(
                float(np.min(exact.expected_pd) - np.min(scaled.expected_pd))
            )
            max_error = max(max_error, error)
            rows.append({
                "seed_offset": seed_offset,
                "budget_bits": budget,
                "exact_worst": float(np.min(exact.expected_pd)),
                "scaled_worst": float(np.min(scaled.expected_pd)),
                "abs_error": error,
            })

    large_model = _synthetic_model(12)
    large_result = minimum_cost_to_threshold(
        large_model, 0.8, 0.05, grid=grid, max_cost=30
    )
    scalability = _scalability_benchmark(grid)
    payload = {
        "gate": "G8-S-scaled-maxmin-selection",
        "seeds": seeds,
        "grid": grid,
        "controlled_rows": rows,
        "controlled_by_budget": _aggregate_by_budget(
            rows, "budget_bits", ["exact_worst", "scaled_worst", "abs_error"]
        ),
        "controlled_summary": {
            "max_abs_error": max_error,
            "within_tolerance": bool(max_error <= 1e-6),
        },
        "large_report_set": {
            "num_reports": 12,
            "found": large_result is not None,
            "min_cost": None if large_result is None else large_result[0],
            "scheduled": None if large_result is None else sorted(
                large_result[1]
            ),
        },
        "scalability_benchmark": scalability,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "controlled_summary": payload["controlled_summary"],
        "large_report_set": payload["large_report_set"],
        "scalability_benchmark": scalability,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scaled_maxmin_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 9])
    parser.add_argument("--grid", type=int, default=64)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        grid=args.grid,
    )


if __name__ == "__main__":
    main()
