"""Where does Exact Joint beat Greedy?

The gate compares the concrete bit schedules and the budget used by the two
methods.  It reports the P_D gap, the budget-usage difference, and the
correlation between the exact-vs-greedy gap and Exact's extra budget usage.
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

import scripts.run_mappo_baseline as mappo
from uav_otfs_isac.joint_allocation import (
    exact_joint_maxmin,
    exact_joint_maxmin_selection,
    greedy_bits,
    subset_options,
    target_options,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/exact_vs_greedy_config_gate.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--budget-multiplier", type=int, default=8)
    args = parser.parse_args()

    summary = []
    all_rows = []
    for q_count in args.targets:
        mappo.N_TARGETS = q_count
        mappo.N_REPORTS = 4
        budget = args.budget_multiplier * q_count
        scenarios = [mappo._scenario(10000 + seed) for seed in range(args.seeds)]
        rows = []
        for scenario in scenarios:
            greedy_vectors = [
                np.concatenate((
                    [0], greedy_bits(target[1:], budget, mappo.GRID),
                ))
                for target in scenario
            ]
            greedy = exact_joint_maxmin(
                [
                    subset_options(
                        float(target[0]), target[1:], vector[1:], mappo.GRID,
                    )
                    for target, vector in zip(scenario, greedy_vectors)
                ],
                budget,
            )
            exact_options = [
                target_options(float(target[0]), target[1:], mappo.GRID)
                for target in scenario
            ]
            exact = exact_joint_maxmin(exact_options, budget)
            _, exact_chosen = exact_joint_maxmin_selection(
                exact_options, budget
            )
            greedy_used = int(sum(
                int(vector[1:].sum()) for vector in greedy_vectors
            ))
            exact_used = int(sum(cost for cost, _ in exact_chosen))
            rows.append({
                "targets": q_count,
                "greedy_worst_pd": float(greedy),
                "exact_worst_pd": float(exact),
                "gap_pp": float((exact - greedy) * 100.0),
                "greedy_used_bits": greedy_used,
                "exact_used_bits": exact_used,
                "budget_delta": exact_used - greedy_used,
            })
        all_rows.extend(rows)
        gaps = np.asarray([row["gap_pp"] for row in rows])
        deltas = np.asarray([row["budget_delta"] for row in rows])
        summary.append({
            "targets": q_count,
            "budget_bits": budget,
            "mean_gap_pp": float(np.mean(gaps)),
            "mean_exact_used_bits": float(np.mean([
                row["exact_used_bits"] for row in rows
            ])),
            "mean_greedy_used_bits": float(np.mean([
                row["greedy_used_bits"] for row in rows
            ])),
            "mean_budget_delta": float(np.mean(deltas)),
            "gap_budget_correlation": float(np.corrcoef(gaps, deltas)[0, 1]),
            "exact_more_budget_rate": float(np.mean(deltas > 0)),
        })
    payload = {
        "gate": "exact-vs-greedy-config",
        "summary": summary,
        "rows": all_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
