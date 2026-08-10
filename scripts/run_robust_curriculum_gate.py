"""Robust curriculum gate: MAPPO trained on weak/poor-channel curriculum.

The channel-aware MAPPO is trained on heterogeneous, weak, strong, and
channel-shifted scenarios, then evaluated on unseen weak targets and harsh
channels.  NOMP remains the online robust allocator and exact is the oracle.
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

from uav_otfs_isac.nomp_refinement import nomp_wta_greedy_joint_multi

from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)
from scripts.run_joint_power_comparison import (
    evaluate_mappo_bandit_adapter_nomp,
    evaluate_robust_mappo,
    make_scenario,
    robust_state,
    train_robust_mappo,
)
from scripts.run_unknown_environment_gate import (
    exact_clean,
    exact_comm,
    make_shifted_clean,
)


def clean_to_tuple(scenario):
    out = []
    for target in scenario:
        row = np.asarray(target, dtype=float)
        owner = float(row[0])
        deltas = row[1:]
        out.append((
            owner,
            deltas,
            np.zeros_like(deltas),
            np.ones_like(deltas),
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robust_curriculum_gate.json")
    parser.add_argument("--figure", default="paper_figures/robust_curriculum.png")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-seeds", type=int, default=20)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    train_scenarios = []
    for seed in range(args.train_seeds):
        train_scenarios.append(clean_to_tuple(
            make_scenario(seed, 2, 2, heterogeneous=True)
        ))
        train_scenarios.append(clean_to_tuple(
            make_shifted_clean(seed, 2, 2, "weak")
        ))
        train_scenarios.append(clean_to_tuple(
            make_shifted_clean(seed, 2, 2, "strong")
        ))
        train_scenarios.append(make_comm_mismatch_scenario(
            seed, 2, 2, flip_hi=0.4, success_lo=0.4
        ))

    actors = {
        budget: train_robust_mappo(
            train_scenarios, budget, args.episodes, 2
        )
        for budget in args.budgets
    }

    environments = {
        "weak": [
            clean_to_tuple(make_shifted_clean(10000 + seed, 2, 2, "weak"))
            for seed in range(args.test_seeds)
        ],
        "channel_shift": [
            make_comm_mismatch_scenario(
                10000 + seed, 2, 2, flip_hi=0.4, success_lo=0.4
            )
            for seed in range(args.test_seeds)
        ],
        "in_distribution": [
            clean_to_tuple(
                make_scenario(10000 + seed, 2, 2, heterogeneous=True)
            )
            for seed in range(args.test_seeds)
        ],
    }

    rows = []
    for env, scenarios in environments.items():
        for budget in args.budgets:
            robust_mappo = evaluate_robust_mappo(
                actors[budget], scenarios, budget, 2
            )
            robust_bandit = evaluate_mappo_bandit_adapter_nomp(
                actors[budget],
                scenarios,
                budget,
                2,
                state_builder=robust_state,
            )
            nomp = []
            exact = []
            for scenario in scenarios:
                nomp.append(float(nomp_wta_greedy_joint_multi(
                    scenario, budget
                )["worst_pd"]))
                if env == "channel_shift":
                    exact.append(exact_comm(
                        scenario, budget, args.grid
                    ))
                else:
                    exact.append(exact_clean(
                        scenario, budget, args.grid
                    ))
            rows.append({
                "environment": env,
                "budget": budget,
                "robust_mappo_worst_mean": float(robust_mappo),
                "robust_bandit_worst_mean": float(robust_bandit),
                "nomp_worst_mean": float(np.mean(nomp)),
                "exact_worst_mean": float(np.mean(exact)),
            })
            print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "robust-curriculum",
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = ["in_distribution", "weak", "channel_shift"]
    methods = [
        ("robust_mappo_worst_mean", "Robust MAPPO"),
        ("robust_bandit_worst_mean", "Robust Bandit"),
        ("nomp_worst_mean", "NOMP"),
        ("exact_worst_mean", "Exact"),
    ]
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.2
    for i, (field, name) in enumerate(methods):
        values = [
            float(np.mean([
                row[field] for row in rows
                if row["environment"] == env
            ]))
            for env in labels
        ]
        plt.bar(x + (i - len(methods) / 2) * width, values,
                width, label=name)
    plt.xticks(x, labels)
    plt.ylabel("Mean worst P_D (average over budgets)")
    plt.title("Robust curriculum MAPPO in unseen environments")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
