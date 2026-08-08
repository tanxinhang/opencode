"""Target-count scalability of exact joint bit allocation.

The exact joint formulation is target-separable, so its theoretical
exactness holds for any target count.  This gate verifies the empirical
scaling: Q = 2/3/5/8/12 targets with strong/medium/weak profiles share one
budget that grows linearly with Q, and the exact joint max-min is compared
with the greedy per-report allocation.  Frontier sizes and DP wall times are
recorded so the scalability claim is backed by data rather than by a
two-target anecdote.
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
    greedy_bits,
    model_from_bits,
    subset_options,
    target_options,
    vectorized_target_options,
)


PROFILES = (
    (0.4, 1.8, 2.2),
    (0.35, 1.5, 1.8),
    (0.3, 1.2, 1.6),
)


def _profiles(
    target_count: int,
    reports_per_target: int,
    rng: np.random.Generator,
):
    result = []
    for q in range(target_count):
        owner, lo, hi = PROFILES[q % len(PROFILES)]
        deltas = np.concatenate((
            [owner], rng.uniform(lo, hi, reports_per_target),
        ))
        result.append(deltas)
    return result


def run_gate(
    *,
    output: Path,
    seeds: int,
    grid: int,
    reports_per_target: int,
) -> None:
    target_counts = (2, 3, 5, 8, 12, 20)
    budget_multiplier = 3 + reports_per_target
    rows = []
    for target_count in target_counts:
        budget = budget_multiplier * target_count
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            deltas_list = _profiles(target_count, reports_per_target, rng)
            pattern = np.array(
                [0] + list(range(1, reports_per_target + 1)),
            )

            fixed = exact_joint_maxmin(
                [
                    subset_options(float(deltas[0]), deltas[1:], pattern[1:], grid)
                    for deltas in deltas_list
                ],
                budget,
            )

            greedy_option_sets = []
            for deltas in deltas_list:
                bits = np.concatenate((
                    [0], greedy_bits(deltas[1:], budget, grid),
                ))
                greedy_option_sets.append(
                    subset_options(float(deltas[0]), deltas[1:], bits[1:], grid)
                )
            greedy = exact_joint_maxmin(greedy_option_sets, budget)

            if reports_per_target >= 6:
                option_sets = [
                    vectorized_target_options(
                        float(deltas[0]), deltas[1:], grid=grid,
                    )
                    for deltas in deltas_list
                ]
            else:
                option_sets = [
                    target_options(float(deltas[0]), deltas[1:], grid)
                    for deltas in deltas_list
                ]
            started = time.perf_counter()
            joint = exact_joint_maxmin(option_sets, budget)
            dp_wall = time.perf_counter() - started
            rows.append({
                "target_count": target_count,
                "budget_bits": budget,
                "seed": seed,
                "fixed_pd": fixed,
                "greedy_pd": greedy,
                "exact_joint_pd": joint,
                "joint_over_greedy_pp": float((joint - greedy) * 100.0),
                "median_frontier_size": float(np.median([
                    len(options) for options in option_sets
                ])),
                "max_frontier_size": int(max(len(o) for o in option_sets)),
                "dp_wall_seconds": dp_wall,
            })

    summary = []
    for target_count in target_counts:
        cell = [r for r in rows if r["target_count"] == target_count]
        gain = [r["joint_over_greedy_pp"] for r in cell]
        summary.append({
            "target_count": target_count,
            "budget_bits": budget_multiplier * target_count,
            "n_seeds": len(cell),
            "fixed_pd_mean": float(np.mean([r["fixed_pd"] for r in cell])),
            "greedy_pd_mean": float(np.mean([r["greedy_pd"] for r in cell])),
            "exact_joint_pd_mean": float(np.mean([r["exact_joint_pd"] for r in cell])),
            "joint_over_greedy_mean_pp": float(np.mean(gain)),
            "joint_over_greedy_min_pp": float(np.min(gain)),
            "joint_over_greedy_max_pp": float(np.max(gain)),
            "median_frontier_size": float(np.mean([
                r["median_frontier_size"] for r in cell
            ])),
            "max_frontier_size": int(max(
                r["max_frontier_size"] for r in cell
            )),
            "dp_wall_seconds_mean": float(np.mean([
                r["dp_wall_seconds"] for r in cell
            ])),
        })

    payload = {
        "gate": "joint-bit-allocation-target-count-scalability",
        "seeds": seeds,
        "grid": grid,
        "reports_per_target": reports_per_target,
        "target_counts": list(target_counts),
        "rows": rows,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_scale_gate.json")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--reports", type=int, default=4)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        grid=args.grid,
        reports_per_target=args.reports,
    )


if __name__ == "__main__":
    main()
