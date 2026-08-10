"""Robustness of PPO trainer with NOMP-final reward.

Vanilla MAPPO and PPO+NOMP-final-reward MAPPO are trained on the same
heterogeneous distribution, then evaluated on unseen weak targets and
increasing channel difficulty.  NOMP, the proposal-plus-refine hybrid, and
the bandit adapter (with NOMP fallback) are compared on the same draws.
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

from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)
from scripts.run_joint_power_comparison import (
    evaluate_mappo,
    evaluate_mappo_bandit_adapter_nomp,
    evaluate_mappo_nomp,
    evaluate_robust_mappo,
    evaluate_robust_mappo_nomp,
    make_scenario,
    robust_state,
    train_mappo,
    train_mappo_ppo,
)
from scripts.run_unknown_environment_gate import (
    exact_clean,
    exact_comm,
    evaluate_mappo_true,
    make_shifted_clean,
)
from uav_otfs_isac.nomp_refinement import nomp_wta_greedy_joint_multi


def clean_to_tuple(scenario):
    out = []
    for target in scenario:
        row = np.asarray(target, dtype=float)
        out.append((
            float(row[0]),
            row[1:],
            np.zeros_like(row[1:]),
            np.ones_like(row[1:]),
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ppo_nomp_reward_robustness_gate.json")
    parser.add_argument("--figure", default="paper_figures/ppo_nomp_reward_robustness.png")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    train_scenarios = [
        make_scenario(seed, 2, 2, heterogeneous=True)
        for seed in range(args.train_seeds)
    ]

    environments = {
        "in_distribution": [
            clean_to_tuple(
                make_scenario(10000 + seed, 2, 2, heterogeneous=True)
            )
            for seed in range(args.test_seeds)
        ],
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
        "hard": [
            make_comm_mismatch_scenario(
                10000 + seed, 2, 2, flip_hi=0.5, success_lo=0.3
            )
            for seed in range(args.test_seeds)
        ],
    }

    rows = []
    for budget in args.budgets:
        vanilla = train_mappo(
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
        curriculum = []
        for seed in range(args.train_seeds):
            curriculum.append(clean_to_tuple(
                make_scenario(seed, 2, 2, heterogeneous=True)
            ))
            curriculum.append(clean_to_tuple(
                make_shifted_clean(seed, 2, 2, "weak")
            ))
            curriculum.append(clean_to_tuple(
                make_shifted_clean(seed, 2, 2, "strong")
            ))
            curriculum.append(make_comm_mismatch_scenario(
                seed, 2, 2, flip_hi=0.5, success_lo=0.3
            ))
        robust_ppo_informed = train_mappo_ppo(
            curriculum,
            budget,
            args.episodes,
            2,
            ppo_epochs=2,
            entropy_coef=0.05,
            learning_rate=1e-3,
            reward_mode="nomp",
            state_builder=robust_state,
        )
        for env, scenarios in environments.items():
            mappo_scenarios = []
            for scenario in scenarios:
                converted = []
                for target in scenario:
                    if isinstance(target, tuple):
                        owner = float(target[0])
                        deltas = np.asarray(target[1], dtype=float)
                    else:
                        row = np.asarray(target, dtype=float)
                        owner = float(row[0])
                        deltas = row[1:]
                    converted.append(
                        np.concatenate(([owner], deltas))
                    )
                mappo_scenarios.append(converted)
            mappo = evaluate_mappo_true(
                vanilla, mappo_scenarios, scenarios, budget, 2
            )
            informed = evaluate_mappo_true(
                ppo_informed, mappo_scenarios, scenarios, budget, 2
            )
            informed_nomp = evaluate_mappo_nomp(
                ppo_informed,
                scenarios,
                budget,
                2,
                state_scenarios=mappo_scenarios,
            )
            informed_bandit = evaluate_mappo_bandit_adapter_nomp(
                ppo_informed,
                scenarios,
                budget,
                2,
                state_scenarios=mappo_scenarios,
            )
            robust_informed = evaluate_robust_mappo(
                robust_ppo_informed, scenarios, budget, 2
            )
            robust_informed_nomp = evaluate_robust_mappo_nomp(
                robust_ppo_informed, scenarios, budget, 2
            )
            robust_informed_bandit = evaluate_mappo_bandit_adapter_nomp(
                robust_ppo_informed,
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
                if env in ("channel_shift", "hard"):
                    exact.append(exact_comm(
                        scenario, budget, args.grid
                    ))
                else:
                    exact.append(exact_clean(
                        scenario, budget, args.grid
                    ))
            rows.append({
                "budget": budget,
                "environment": env,
                "mappo_worst_mean": float(mappo),
                "ppo_informed_worst_mean": float(informed),
                "ppo_informed_nomp_worst_mean": float(informed_nomp),
                "ppo_informed_bandit_worst_mean": float(informed_bandit),
                "robust_ppo_informed_worst_mean": float(robust_informed),
                "robust_ppo_informed_nomp_worst_mean": float(
                    robust_informed_nomp
                ),
                "robust_ppo_informed_bandit_worst_mean": float(
                    robust_informed_bandit
                ),
                "nomp_worst_mean": float(np.mean(nomp)),
                "exact_worst_mean": float(np.mean(exact)),
            })
            print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "ppo-nomp-reward-robustness",
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = ["in_distribution", "weak", "channel_shift", "hard"]
    methods = [
        ("mappo_worst_mean", "MAPPO"),
        ("ppo_informed_worst_mean", "PPO-Informed"),
        ("ppo_informed_nomp_worst_mean", "PPO-Informed+NOMP"),
        ("ppo_informed_bandit_worst_mean", "PPO-Informed Bandit"),
        ("robust_ppo_informed_worst_mean", "Robust PPO-Informed"),
        ("robust_ppo_informed_nomp_worst_mean", "Robust PPO+NOMP"),
        ("robust_ppo_informed_bandit_worst_mean", "Robust PPO Bandit"),
        ("nomp_worst_mean", "NOMP"),
        ("exact_worst_mean", "Exact"),
    ]
    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.14
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
    plt.title("PPO + NOMP-final reward robustness")
    plt.grid(alpha=0.3, axis="y")
    plt.legend(ncol=3, fontsize=8)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
