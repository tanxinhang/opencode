"""Target-count scalability audit for the exact selection selectors.

The exact budget/max-min selectors are proved exact for any target count,
but the DP state set grows with Q.  This gate measures wall time and keeps
the exhaustive-oracle comparison at Q=3/4/5 so the exactness claim is
re-verified when the target count grows.
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
from uav_otfs_isac.exact_quota_selection import (
    exact_budget_select,
    exact_maxmin_select,
    subset_expected_pd_map,
)
from uav_otfs_isac.expected_pd import expected_pd_greedy_select


def _exhaustive_oracle(models, budget_bits, false_alarm_rate, qos, qos_w, perf_w, grid):
    subsets = [
        subset_expected_pd_map(model, false_alarm_rate, grid=grid)
        for model in models
    ]
    best = None

    def score(values):
        values = np.asarray(values, dtype=float)
        gap = float(np.sum(
            qos_w * np.maximum(qos - values, 0.0) / np.maximum(qos, 1e-12)
        ))
        return (
            -gap,
            float(np.sum(perf_w * values)),
            float(np.min(values)),
        )

    def recurse(index, groups, used):
        nonlocal best
        if index == len(models):
            values = np.asarray([
                subsets[q][groups[q]] for q in range(len(models))
            ], dtype=float)
            current = score(values)
            if best is None or current > best[0]:
                best = (current, values)
            return
        model = models[index]
        for scheduled, _ in subsets[index].items():
            cost = sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            if used + cost <= budget_bits:
                recurse(index + 1, groups + [scheduled], used + cost)

    recurse(0, [], 0)
    assert best is not None
    return best[0], best[1]


def _exhaustive_maxmin_oracle(models, budget_bits, false_alarm_rate, grid):
    subsets = [
        subset_expected_pd_map(model, false_alarm_rate, grid=grid)
        for model in models
    ]
    best = None

    def recurse(index, groups, used):
        nonlocal best
        if index == len(models):
            values = np.asarray([
                subsets[q][groups[q]] for q in range(len(models))
            ], dtype=float)
            current = float(np.min(values))
            if best is None or current > best[0]:
                best = (current, values)
            return
        model = models[index]
        for scheduled, _ in subsets[index].items():
            cost = sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            if used + cost <= budget_bits:
                recurse(index + 1, groups + [scheduled], used + cost)

    recurse(0, [], 0)
    assert best is not None
    return best[0], best[1]


def run_gate(*, output: Path, seeds: int, budgets, grid: int) -> None:
    qos = np.array([0.85, 0.82, 0.88, 0.84, 0.86])
    qos_w = np.array([0.4, 0.3, 0.3, 0.2, 0.25])
    perf_w = np.array([1.0, 0.8, 0.9, 0.85, 0.95])
    rows = []
    for num_targets in (3, 4, 5):
        qos_cell = qos[:num_targets]
        qos_w_cell = qos_w[:num_targets]
        perf_w_cell = perf_w[:num_targets]
        for budget in budgets:
            for seed_offset in range(seeds):
                rng = np.random.default_rng(20260809 + seed_offset)
                models = [
                    symmetric_diversity_model(
                        np.sort(rng.uniform(0.6, 1.8, size=4))[::-1],
                        success_probability=float(rng.uniform(0.5, 0.95)),
                        report_bits=np.array([1, 2, 3, 5]),
                    )
                    for _ in range(num_targets)
                ]
                started = time.perf_counter()
                exact_budget = exact_budget_select(
                    models, budget, 0.05, qos_pd=qos_cell,
                    qos_weights=qos_w_cell, performance_weights=perf_w_cell,
                    grid=grid,
                )
                budget_wall = time.perf_counter() - started
                started = time.perf_counter()
                exact_maxmin = exact_maxmin_select(
                    models, budget, 0.05, qos_pd=qos_cell,
                    qos_weights=qos_w_cell, performance_weights=perf_w_cell,
                    grid=grid,
                )
                maxmin_wall = time.perf_counter() - started
                greedy = expected_pd_greedy_select(
                    models, budget, 0.05, qos_pd=qos_cell,
                    qos_weights=qos_w_cell, performance_weights=perf_w_cell,
                    grid=grid,
                )
                budget_oracle_score, budget_oracle_values = _exhaustive_oracle(
                    models, budget, 0.05, qos_cell, qos_w_cell, perf_w_cell,
                    grid,
                )
                maxmin_oracle_score, maxmin_oracle_values = (
                    _exhaustive_maxmin_oracle(models, budget, 0.05, grid)
                )

                def budget_score(values):
                    values = np.asarray(values, dtype=float)
                    gap = float(np.sum(
                        qos_w_cell * np.maximum(qos_cell - values, 0.0)
                        / np.maximum(qos_cell, 1e-12)
                    ))
                    return (
                        -gap,
                        float(np.sum(perf_w_cell * values)),
                        float(np.min(values)),
                    )

                rows.append({
                    "num_targets": num_targets,
                    "budget_bits": budget,
                    "seed_offset": seed_offset,
                    "budget_wall_seconds": budget_wall,
                    "maxmin_wall_seconds": maxmin_wall,
                    "budget_oracle_match": bool(
                        budget_score(exact_budget.expected_pd)
                        == budget_oracle_score
                    ),
                    "maxmin_oracle_match": bool(
                        np.isclose(
                            np.min(exact_maxmin.expected_pd),
                            maxmin_oracle_score,
                        )
                    ),
                    "budget_never_worse": bool(
                        budget_score(exact_budget.expected_pd)
                        >= budget_score(greedy.expected_pd)
                    ),
                    "maxmin_never_worse": bool(
                        np.min(exact_maxmin.expected_pd)
                        >= np.min(greedy.expected_pd) - 1e-12
                    ),
                    "budget_used_bits": int(exact_budget.used_bits),
                    "maxmin_used_bits": int(exact_maxmin.used_bits),
                })

    by_cell = {}
    for row in rows:
        key = (row["num_targets"], row["budget_bits"])
        by_cell.setdefault(key, []).append(row)
    summary = []
    for (num_targets, budget), cell in sorted(by_cell.items()):
        summary.append({
            "num_targets": num_targets,
            "budget_bits": budget,
            "n_seeds": len(cell),
            "budget_wall_mean_ms": 1000.0 * float(np.mean([
                row["budget_wall_seconds"] for row in cell
            ])),
            "maxmin_wall_mean_ms": 1000.0 * float(np.mean([
                row["maxmin_wall_seconds"] for row in cell
            ])),
            "budget_oracle_match_rate": float(np.mean([
                1.0 if row["budget_oracle_match"] else 0.0 for row in cell
            ])),
            "maxmin_oracle_match_rate": float(np.mean([
                1.0 if row["maxmin_oracle_match"] else 0.0 for row in cell
            ])),
            "budget_never_worse_rate": float(np.mean([
                1.0 if row["budget_never_worse"] else 0.0 for row in cell
            ])),
            "maxmin_never_worse_rate": float(np.mean([
                1.0 if row["maxmin_never_worse"] else 0.0 for row in cell
            ])),
        })
    payload = {
        "gate": "G8-target-scalability",
        "seeds": seeds,
        "grid": grid,
        "budgets": budgets,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/exact_selection_target_scalability.json")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--grid", type=int, default=32)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        grid=args.grid,
    )


if __name__ == "__main__":
    main()
