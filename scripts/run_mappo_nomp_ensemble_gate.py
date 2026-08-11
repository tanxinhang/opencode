"""Multi-temperature proposal ensemble for PPO+NOMP.

Sampling proposals at different temperatures explores both the argmax and
high-entropy regions; NOMP refines each proposal and keeps the best.  Early
stopping after no improvement keeps the loop finite and the best-of-K
monotone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_joint_power_comparison import (
    evaluate_mappo_bandit_adapter_nomp,
    evaluate_mappo_nomp,
    evaluate_mappo_nomp_multi,
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
    parser.add_argument("--output", default="results/mappo_nomp_ensemble_gate.json")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--train-seeds", type=int, default=15)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(0)
    train_scenarios = [
        make_scenario(seed, 2, 2, heterogeneous=True)
        for seed in range(args.train_seeds)
    ]
    actor = train_mappo_ppo(
        train_scenarios,
        args.budget,
        args.episodes,
        2,
        ppo_epochs=2,
        entropy_coef=0.05,
        learning_rate=1e-3,
        reward_mode="raw",
    )
    environments = {
        "hard": dict(reference_snr_db=8.0, threshold_db=10.0, shadowing_db=4.0),
        "extreme": dict(reference_snr_db=3.0, threshold_db=12.0, shadowing_db=5.0),
    }
    rows = []
    for difficulty, params in environments.items():
        scenarios = [
            make_physical_scenario(
                10000 + seed, 2, 2, **params
            )
            for seed in range(args.test_seeds)
        ]
        mappo_scenarios = []
        for scenario in scenarios:
            converted = [
                np.concatenate(([float(t[0])], np.asarray(t[1], dtype=float)))
                for t in scenario
            ]
            mappo_scenarios.append(converted)
        ppo = evaluate_mappo_true(
            actor, mappo_scenarios, scenarios, args.budget, 2
        )
        single = evaluate_mappo_nomp(
            actor,
            scenarios,
            args.budget,
            2,
            state_scenarios=mappo_scenarios,
        )
        ensemble = evaluate_mappo_nomp_multi(
            actor,
            scenarios,
            args.budget,
            2,
            samples=9,
            max_rounds=50,
            candidate_budget=8,
            state_scenarios=mappo_scenarios,
            temperatures=(1.0, 2.0, 3.0),
            seed=0,
            residual_adaptive=True,
            alpha=4.0,
        )
        bandit = evaluate_mappo_bandit_adapter_nomp(
            actor,
            scenarios,
            args.budget,
            2,
            state_scenarios=mappo_scenarios,
        )
        nomp = []
        exact = []
        for scenario in scenarios:
            nomp.append(float(nomp_wta_greedy_joint_multi(
                scenario,
                args.budget,
                max_rounds=50,
                samples=512,
                candidate_budget=8,
            )["worst_pd"]))
            exact.append(exact_comm(scenario, args.budget, args.grid))
        rows.append({
            "difficulty": difficulty,
            "ppo_worst_mean": float(ppo),
            "ppo_nomp_single_worst_mean": float(single),
            "ppo_nomp_ensemble_worst_mean": float(ensemble),
            "ppo_bandit_worst_mean": float(bandit),
            "nomp_worst_mean": float(np.mean(nomp)),
            "exact_worst_mean": float(np.mean(exact)),
        })
        print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "mappo-nomp-ensemble",
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
