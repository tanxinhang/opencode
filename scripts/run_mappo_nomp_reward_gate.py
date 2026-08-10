"""MAPPO-NOMP reward gate: training MAPPO on NOMP-final P_D.

Instead of rewarding the raw MAPPO proposal, the actor is trained with the
worst P_D after NOMP refines the proposal.  This aligns the policy objective
with the system objective and measures whether the proposal improves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_joint_power_comparison import (
    evaluate_mappo,
    evaluate_mappo_nomp,
    make_scenario,
    train_mappo,
    train_mappo_nomp_reward,
    train_mappo_ppo,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/mappo_nomp_reward_gate.json")
    parser.add_argument("--figure", default="paper_figures/mappo_nomp_reward.png")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--test-seeds", type=int, default=10)
    args = parser.parse_args()

    train_scenarios = [
        make_scenario(seed, 2, 2, heterogeneous=True)
        for seed in range(args.train_seeds)
    ]
    test_scenarios = [
        make_scenario(10000 + seed, 2, 2, heterogeneous=True)
        for seed in range(args.test_seeds)
    ]
    rows = []
    for budget in args.budgets:
        vanilla = train_mappo(train_scenarios, budget, args.episodes, 2)
        informed = train_mappo_nomp_reward(
            train_scenarios, budget, args.episodes, 2
        )
        ppo_informed = train_mappo_ppo(
            train_scenarios,
            budget,
            args.episodes,
            2,
            ppo_epochs=2,
            entropy_coef=0.05,
            learning_rate=1e-3,
            reward_mode="nomp",
        )
        rows.append({
            "budget": budget,
            "mappo_worst_mean": float(evaluate_mappo(
                vanilla, test_scenarios, budget, 2
            )),
            "mappo_nomp_worst_mean": float(evaluate_mappo_nomp(
                vanilla, test_scenarios, budget, 2
            )),
            "informed_mappo_worst_mean": float(evaluate_mappo(
                informed, test_scenarios, budget, 2
            )),
            "informed_mappo_nomp_worst_mean": float(evaluate_mappo_nomp(
                informed, test_scenarios, budget, 2
            )),
            "ppo_informed_mappo_worst_mean": float(evaluate_mappo(
                ppo_informed, test_scenarios, budget, 2
            )),
            "ppo_informed_mappo_nomp_worst_mean": float(evaluate_mappo_nomp(
                ppo_informed, test_scenarios, budget, 2
            )),
        })
        print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "mappo-nomp-reward",
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    budgets = [row["budget"] for row in rows]
    x = np.arange(len(rows))
    width = 0.2
    fields = [
        ("mappo_worst_mean", "MAPPO"),
        ("informed_mappo_worst_mean", "MAPPO-NOMP-reward"),
        ("ppo_informed_mappo_worst_mean", "PPO-NOMP-reward"),
        ("mappo_nomp_worst_mean", "MAPPO+NOMP"),
        ("informed_mappo_nomp_worst_mean", "Informed+NOMP"),
        ("ppo_informed_mappo_nomp_worst_mean", "PPO-Informed+NOMP"),
    ]
    plt.figure(figsize=(8, 5))
    for i, (field, name) in enumerate(fields):
        plt.bar(x + (i - len(fields) / 2) * width,
                [row[field] for row in rows], width, label=name)
    plt.xticks(x, budgets)
    plt.xlabel("Budget B")
    plt.ylabel("Mean worst P_D")
    plt.title("MAPPO trained with NOMP-final reward")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
