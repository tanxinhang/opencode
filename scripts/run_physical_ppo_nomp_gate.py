"""PPO + NOMP under physical hard channels with UAV count sweep.

PPO is trained per UAV count R with the NOMP-final reward, then evaluated on
physical baseline/hard/extreme channels.  The bandit variant includes the
pure NOMP fallback.
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
    evaluate_mappo_bandit_adapter_nomp,
    evaluate_mappo_nomp,
    make_scenario,
    train_mappo_ppo,
)
from scripts.run_physical_hard_channel_gate import make_physical_scenario
from scripts.run_unknown_environment_gate import (
    evaluate_mappo_true,
    exact_comm,
)
from uav_otfs_isac.nomp_refinement import nomp_wta_greedy_joint_multi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/physical_ppo_nomp_gate.json")
    parser.add_argument("--figure", default="paper_figures/physical_ppo_nomp.png")
    parser.add_argument("--reports", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--train-seeds", type=int, default=20)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--budget-multiplier", type=int, default=4)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    environments = {
        "baseline": dict(reference_snr_db=25.0, threshold_db=5.0, shadowing_db=3.0),
        "hard": dict(reference_snr_db=8.0, threshold_db=10.0, shadowing_db=4.0),
        "extreme": dict(reference_snr_db=3.0, threshold_db=12.0, shadowing_db=5.0),
    }
    rows = []
    for reports in args.reports:
        budget = args.budget_multiplier * reports * 2
        train_scenarios = [
            make_scenario(seed, reports, 2, heterogeneous=True)
            for seed in range(args.train_seeds)
        ]
        actor = train_mappo_ppo(
            train_scenarios,
            budget,
            args.episodes,
            reports,
            ppo_epochs=2,
            entropy_coef=0.05,
            learning_rate=1e-3,
            reward_mode="raw",
        )
        for difficulty, params in environments.items():
            scenarios = [
                make_physical_scenario(
                    10000 + seed, reports, 2, **params
                )
                for seed in range(args.test_seeds)
            ]
            mappo_scenarios = []
            for scenario in scenarios:
                converted = []
                for target in scenario:
                    owner = float(target[0])
                    deltas = np.asarray(target[1], dtype=float)
                    converted.append(
                        np.concatenate(([owner], deltas))
                    )
                mappo_scenarios.append(converted)
            ppo = evaluate_mappo_true(
                actor, mappo_scenarios, scenarios, budget, reports
            )
            ppo_nomp = evaluate_mappo_nomp(
                actor,
                scenarios,
                budget,
                reports,
                state_scenarios=mappo_scenarios,
            )
            ppo_bandit = evaluate_mappo_bandit_adapter_nomp(
                actor,
                scenarios,
                budget,
                reports,
                state_scenarios=mappo_scenarios,
            )
            nomp = []
            exact = []
            for scenario in scenarios:
                nomp.append(float(nomp_wta_greedy_joint_multi(
                    scenario,
                    budget,
                    max_rounds=50,
                    samples=512,
                    candidate_budget=8,
                )["worst_pd"]))
                if reports == 2:
                    exact.append(exact_comm(scenario, budget, args.grid))
            rows.append({
                "reports": reports,
                "budget": budget,
                "difficulty": difficulty,
                "ppo_worst_mean": float(ppo),
                "ppo_nomp_worst_mean": float(ppo_nomp),
                "ppo_bandit_worst_mean": float(ppo_bandit),
                "nomp_worst_mean": float(np.mean(nomp)),
                "exact_worst_mean": (
                    float(np.mean(exact)) if exact else None
                ),
            })
            print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "physical-ppo-nomp",
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    colors = {
        "baseline": "#31a354",
        "hard": "#e6550d",
        "extreme": "#756bb1",
    }
    plt.figure(figsize=(7.5, 5))
    for difficulty in environments:
        for field, style in (
            ("ppo_worst_mean", "--"),
            ("ppo_nomp_worst_mean", "-."),
            ("nomp_worst_mean", "-"),
        ):
            series = [
                next(
                    row for row in rows
                    if row["difficulty"] == difficulty
                    and row["reports"] == reports
                )
                for reports in args.reports
            ]
            plt.plot(
                args.reports,
                [row[field] for row in series],
                style,
                color=colors[difficulty],
                marker="o",
                label=f"{difficulty} {field.split('_')[0]}",
            )
    plt.xlabel("UAV count R")
    plt.ylabel("Mean worst P_D")
    plt.title("PPO + NOMP under physical channel difficulty")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
