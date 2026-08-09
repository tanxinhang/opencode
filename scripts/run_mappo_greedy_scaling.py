"""MAPPO / Greedy / Exact Joint scaling comparison with a figure.

The same scenario generator, report count, grid, training seeds, test seeds,
and budget-per-target ratio are used for every target count.  The result is
written as JSON and plotted as worst-P_D versus target count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_mappo_baseline as mappo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/mappo_greedy_scaling.json")
    parser.add_argument("--figure", default="paper_figures/mappo_greedy_scaling.png")
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--train-seeds", type=int, default=20)
    parser.add_argument("--test-seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--budget-multiplier", type=int, default=8)
    parser.add_argument("--reports", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for q_count in args.targets:
        mappo.N_TARGETS = q_count
        mappo.N_REPORTS = args.reports
        budget = args.budget_multiplier * q_count
        temp = PROJECT_ROOT / "results" / f"mappo_scaling_q{q_count}.json"
        mappo.run_baseline(
            output=temp,
            train_seeds=args.train_seeds,
            test_seeds=args.test_seeds,
            episodes=args.episodes,
            budgets=(budget,),
            exact_max_reports=None,
            exact_max_bits=4,
        )
        payload = json.loads(temp.read_text(encoding="utf-8"))
        row = payload["summary"][0]
        rows.append({
            "targets": q_count,
            "budget_bits": budget,
            "mappo_worst_mean": row["mappo_worst_mean"],
            "greedy_worst_mean": row["greedy_worst_mean"],
            "exact_joint_worst_mean": row["exact_joint_worst_mean"],
            "mappo_used_bits_mean": row["mappo_used_bits_mean"],
            "train_seconds": row["train_seconds"],
        })

    payload = {
        "gate": "mappo-greedy-exact-scaling",
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "episodes": args.episodes,
        "budget_multiplier": args.budget_multiplier,
        "reports_per_target": args.reports,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    qs = [row["targets"] for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(qs, [row["mappo_worst_mean"] for row in rows],
             "o-", label="MAPPO")
    plt.plot(qs, [row["greedy_worst_mean"] for row in rows],
             "s-", label="Greedy")
    plt.plot(qs, [row["exact_joint_worst_mean"] for row in rows],
             "^-", label="Exact Joint")
    plt.xlabel("Number of targets Q")
    plt.ylabel("Mean worst-target P_D")
    plt.title("MAPPO vs Greedy vs Exact Joint under same per-target budget")
    plt.grid(alpha=0.3)
    plt.legend()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
