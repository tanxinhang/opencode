"""Hard channel gate: robustness under increasing channel difficulty.

Robust MAPPO is trained on a curriculum that includes hard channels, then
evaluated on baseline, hard, and extreme channel draws.  NOMP/UCB-NOMP and
the robust bandit are evaluated on the same unseen draws.
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
    evaluate_mappo_bandit_adapter_nomp,
    evaluate_robust_mappo,
    make_scenario,
    robust_state,
    train_robust_mappo,
    ucb_wta_greedy_joint_multi,
)
from scripts.run_unknown_environment_gate import (
    exact_comm,
    make_shifted_clean,
)
from uav_otfs_isac.nomp_refinement import (
    nomp_wta_greedy_joint_multi,
    wta_greedy_joint_multi,
)


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
    parser.add_argument("--output", default="results/hard_channel_gate.json")
    parser.add_argument("--figure", default="paper_figures/hard_channel.png")
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
            seed, 2, 2, flip_hi=0.5, success_lo=0.3
        ))

    actors = {
        budget: train_robust_mappo(
            train_scenarios, budget, args.episodes, 2
        )
        for budget in args.budgets
    }

    environments = {
        "baseline": dict(flip_hi=0.2, success_lo=0.7),
        "hard": dict(flip_hi=0.5, success_lo=0.3),
        "extreme": dict(flip_hi=0.7, success_lo=0.15),
    }
    rows = []
    for env, params in environments.items():
        for budget in args.budgets:
            scenarios = [
                make_comm_mismatch_scenario(
                    10000 + seed, 2, 2, **params
                )
                for seed in range(args.test_seeds)
            ]
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
            wta = []
            ucb = []
            nomp = []
            exact = []
            for scenario in scenarios:
                wta.append(float(wta_greedy_joint_multi(
                    scenario, budget, min_cover=False
                )["worst_pd"]))
                ucb.append(float(ucb_wta_greedy_joint_multi(
                    scenario,
                    budget,
                    noise_scale=0.2,
                    seed=0,
                    min_cover=True,
                    refine=True,
                )["worst_pd"]))
                nomp.append(float(nomp_wta_greedy_joint_multi(
                    scenario, budget
                )["worst_pd"]))
                exact.append(exact_comm(scenario, budget, args.grid))
            rows.append({
                "environment": env,
                "budget": budget,
                "wta_worst_mean": float(np.mean(wta)),
                "ucb_nomp_worst_mean": float(np.mean(ucb)),
                "nomp_worst_mean": float(np.mean(nomp)),
                "robust_mappo_worst_mean": float(robust_mappo),
                "robust_bandit_worst_mean": float(robust_bandit),
                "exact_worst_mean": float(np.mean(exact)),
            })
            print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "hard-channel",
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = list(environments)
    methods = [
        ("wta_worst_mean", "WTA-Greedy"),
        ("robust_mappo_worst_mean", "Robust MAPPO"),
        ("robust_bandit_worst_mean", "Robust Bandit"),
        ("ucb_nomp_worst_mean", "UCB-NOMP"),
        ("nomp_worst_mean", "NOMP"),
        ("exact_worst_mean", "Exact"),
    ]
    plt.figure(figsize=(9, 5))
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
    plt.title("Robustness under increasing channel difficulty")
    plt.grid(alpha=0.3, axis="y")
    plt.legend(ncol=3, fontsize=8)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
