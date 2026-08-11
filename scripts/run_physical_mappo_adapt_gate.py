"""Physical-difficulty curriculum for MAPPO, with NOMP-final reward.

The PPO policy is trained on a mixture of physical baseline/hard/extreme
channels using the NOMP-final reward with low-cost refinement, so it learns
proposals that adapt to difficult channels.  Evaluation uses the same
multi-temperature residual-adaptive ensemble.
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
    evaluate_mappo_nomp_multi,
    evaluate_robust_mappo,
    make_scenario,
    robust_state,
    train_mappo_ppo,
)
from scripts.run_physical_hard_channel_gate import make_physical_scenario
from scripts.run_unknown_environment_gate import exact_comm
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
    parser.add_argument("--output", default="results/physical_mappo_adapt_gate.json")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--train-seeds", type=int, default=15)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(0)
    curriculum = []
    for seed in range(args.train_seeds):
        curriculum.append(clean_to_tuple(
            make_scenario(seed, 2, 2, heterogeneous=True)
        ))
        curriculum.append(make_physical_scenario(
            seed, 2, 2,
            reference_snr_db=25.0, threshold_db=5.0, shadowing_db=3.0,
        ))
        curriculum.append(make_physical_scenario(
            seed, 2, 2,
            reference_snr_db=8.0, threshold_db=10.0, shadowing_db=4.0,
        ))
        curriculum.append(make_physical_scenario(
            seed, 2, 2,
            reference_snr_db=3.0, threshold_db=12.0, shadowing_db=5.0,
        ))
    actor = train_mappo_ppo(
        curriculum,
        args.budget,
        args.episodes,
        2,
        ppo_epochs=2,
        entropy_coef=0.05,
        learning_rate=1e-3,
        reward_mode="nomp",
        state_builder=robust_state,
        reward_max_rounds=30,
        reward_candidate_budget=8,
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
        ppo = evaluate_robust_mappo(actor, scenarios, args.budget, 2)
        ensemble = evaluate_mappo_nomp_multi(
            actor,
            scenarios,
            args.budget,
            2,
            samples=9,
            max_rounds=50,
            candidate_budget=8,
            temperatures=(1.0, 2.0, 3.0),
            seed=0,
            residual_adaptive=True,
            alpha=4.0,
            state_builder=robust_state,
        )
        bandit = evaluate_mappo_bandit_adapter_nomp(
            actor,
            scenarios,
            args.budget,
            2,
            state_builder=robust_state,
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
            "adapted_ppo_worst_mean": float(ppo),
            "adapted_ensemble_worst_mean": float(ensemble),
            "adapted_bandit_worst_mean": float(bandit),
            "nomp_worst_mean": float(np.mean(nomp)),
            "exact_worst_mean": float(np.mean(exact)),
        })
        print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "physical-mappo-adapt",
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
