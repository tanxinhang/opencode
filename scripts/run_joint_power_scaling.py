"""Joint power-bit scaling: worst P_D versus target count.

The same per-target budget multiplier, report count, training seeds, test
seeds, and episode count are used for every Q.  Exact uses the
winner-take-all frontier, which is exact under the proportional model and
keeps the oracle computable as Q and the per-target budget grow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_joint_power_comparison import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_power_scaling.json")
    parser.add_argument("--figure", default="paper_figures/joint_power_scaling.png")
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--train-seeds", type=int, default=20)
    parser.add_argument("--test-seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--budget-multiplier", type=int, default=4)
    parser.add_argument("--reports", type=int, default=2)
    args = parser.parse_args()

    rows = []
    for q_count in args.targets:
        budget = args.budget_multiplier * q_count
        comparison_args = SimpleNamespace(
            mode="heterogeneous",
            reports=args.reports,
            targets=q_count,
            budgets=(budget,),
            episodes=args.episodes,
            train_seeds=args.train_seeds,
            test_seeds=args.test_seeds,
            exact_mode="wta",
        )
        payload = run_comparison(comparison_args)
        row = payload["summary"][0]
        rows.append({
            "targets": q_count,
            "budget": budget,
            "mappo_worst_mean": row["mappo_worst_mean"],
            "greedy_worst_mean": row["greedy_worst_mean"],
            "wta_greedy_worst_mean": row["wta_greedy_worst_mean"],
            "ucb_wta_greedy_worst_mean": row["ucb_wta_greedy_worst_mean"],
            "ucb_wta_certificate_stop_rate": (
                row["ucb_wta_certificate_stop_rate"]
            ),
            "nomp_greedy_worst_mean": row["nomp_greedy_worst_mean"],
            "ucb_nomp_greedy_worst_mean": row["ucb_nomp_greedy_worst_mean"],
            "ucb_nomp_certificate_stop_rate": (
                row["ucb_nomp_certificate_stop_rate"]
            ),
            "exact_worst_mean": row["exact_worst_mean"],
            "train_seconds": row["train_seconds"],
        })
        print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "joint-power-scaling",
        "mode": "heterogeneous",
        "exact_mode": "wta",
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
    plt.figure(figsize=(8, 5))
    plt.plot(qs, [row["mappo_worst_mean"] for row in rows],
             "o-", color="#7f6d5f", label="MAPPO")
    plt.plot(qs, [row["greedy_worst_mean"] for row in rows],
             "s-", color="#bdbdbd", label="Greedy")
    plt.plot(qs, [row["wta_greedy_worst_mean"] for row in rows],
             "v-", color="#3182bd", label="WTA-Greedy")
    plt.plot(qs, [row["ucb_wta_greedy_worst_mean"] for row in rows],
             "^--", color="#9ecae1", label="UCB-WTA")
    plt.plot(qs, [row["nomp_greedy_worst_mean"] for row in rows],
             "D-", color="#31a354", label="NOMP-Greedy")
    plt.plot(qs, [row["ucb_nomp_greedy_worst_mean"] for row in rows],
             "d--", color="#a1d99b", label="UCB-NOMP")
    plt.plot(qs, [row["exact_worst_mean"] for row in rows],
             "*-", color="#e6550d", label="WTA-Exact")
    plt.xlabel("Number of targets Q")
    plt.ylabel("Mean worst-target P_D")
    plt.title("Joint power-bit allocation under per-target budget 4Q")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
