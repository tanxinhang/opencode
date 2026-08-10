"""Generalization gate: performance in unseen environments.

MAPPO is trained on the heterogeneous training distribution, then evaluated
on shifted channels, weaker targets, and stronger targets.  The online NOMP
family and exact oracle are evaluated on the same unseen draws.
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
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac import nomp_refinement as nomp
from uav_otfs_isac.nomp_refinement import (
    nomp_wta_greedy_joint_multi,
    wta_greedy_joint_multi,
)
from uav_otfs_isac.power_split_theory import (
    winner_take_all_proportional_options,
)
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_heterogeneous_robust_power_bit_options,
    pareto_options,
)
from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)
from scripts.run_joint_power_comparison import (
    _feasible_from_mappo,
    evaluate_mappo_adapter_nomp,
    evaluate_mappo_bandit_adapter_nomp,
    greedy_joint_multi,
    make_scenario,
    train_mappo,
    ucb_wta_greedy_joint_multi,
)


def make_shifted_clean(seed, reports, targets, kind):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(targets):
        if kind == "weak":
            owner = rng.uniform(0.2, 0.3)
            deltas = rng.uniform(0.5, 1.2, reports)
        else:
            owner = rng.uniform(0.4, 0.6)
            deltas = rng.uniform(1.5, 2.5, reports)
        out.append(np.concatenate(([owner], deltas)))
    return out


def exact_clean(scenario, budget, grid):
    groups = [
        winner_take_all_proportional_options(
            float(target[0]),
            target[1:],
            bit_options=np.arange(3, dtype=int),
            budget=budget,
            grid=grid,
        )
        for target in scenario
    ]
    return float(exact_joint_power_bit_maxmin(groups, budget))


def exact_comm(scenario, budget, grid):
    groups = []
    for owner, deltas, flips, successes in scenario:
        options = enumerate_heterogeneous_robust_power_bit_options(
            owner,
            deltas,
            [(0.0, float(value)) for value in flips],
            [(float(value), 1.0) for value in successes],
            power_levels=np.arange(budget + 1, dtype=float),
            bit_options=np.arange(3, dtype=int),
            budget=budget,
            grid=grid,
        )
        groups.append(pareto_options(options, "robust_pd"))
    return float(exact_joint_power_bit_maxmin(groups, budget))


def evaluate_mappo_true(
    actor, state_scenarios, eval_scenarios, budget, reports
):
    """MAPPO proposes from projected state; P_D is scored on the true env."""
    worsts = []
    for state_scenario, eval_scenario in zip(
        state_scenarios, eval_scenarios
    ):
        states = [
            np.concatenate((
                [float(target[0]) / 2.0],
                np.asarray(target[1:], dtype=float) / 2.0,
                [float(budget) / 20.0],
            ))
            for target in state_scenario
        ]
        with torch.no_grad():
            logits_b, logits_p = actor(torch.as_tensor(
                np.stack(states), dtype=torch.float32
            ))
            bits = torch.stack([
                torch.argmax(logits_b[:, r, :], dim=1)
                for r in range(reports)
            ], dim=1).numpy()
            powers = torch.stack([
                torch.argmax(logits_p[:, r, :], dim=1)
                for r in range(reports)
            ], dim=1).numpy()
        powers, bits = _feasible_from_mappo(
            eval_scenario, powers, bits, budget
        )
        worsts.append(float(min(nomp.target_scores(
            eval_scenario, powers, bits, 16
        ))))
    return float(np.mean(worsts))


def plot_rows(rows, figure, labels):
    methods = [
        ("greedy_worst_mean", "Greedy"),
        ("wta_greedy_worst_mean", "WTA-Greedy"),
        ("mappo_worst_mean", "MAPPO"),
        ("mappo_adapter_worst_mean", "MAPPO-Adapter"),
        ("mappo_bandit_worst_mean", "MAPPO-Bandit"),
        ("ucb_nomp_worst_mean", "UCB-NOMP"),
        ("nomp_greedy_worst_mean", "NOMP"),
        ("exact_worst_mean", "Exact"),
    ]
    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.1
    for i, (field, name) in enumerate(methods):
        values = []
        for env in labels:
            filtered = [
                row[field] for row in rows
                if row["environment"] == env
                and row[field] is not None
            ]
            if not filtered:
                values.append(np.nan)
            else:
                values.append(float(np.mean(filtered)))
        if all(np.isnan(value) for value in values):
            continue
        plt.bar(x + (i - len(methods) / 2) * width, values,
                width, label=name)
    plt.xticks(x, labels)
    plt.ylabel("Mean worst P_D (average over budgets)")
    plt.title("Performance in unseen environments")
    plt.grid(alpha=0.3, axis="y")
    plt.legend(ncol=4, fontsize=8)
    figure = Path(figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/unknown_environment_gate.json")
    parser.add_argument("--figure", default="paper_figures/unknown_environment.png")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()

    labels = ["in_distribution", "channel_shift", "weak", "strong"]
    if args.figure_only:
        payload = json.loads(Path(args.output).read_text(encoding="utf-8"))
        plot_rows(payload["rows"], args.figure, labels)
        return

    train_scenarios = [
        make_scenario(seed, 2, 2, heterogeneous=True)
        for seed in range(args.train_seeds)
    ]
    actors = {
        budget: train_mappo(
            train_scenarios, budget, args.episodes, 2
        )
        for budget in args.budgets
    }

    environments = ["in_distribution", "channel_shift", "weak", "strong"]
    rows = []
    for env in environments:
        for budget in args.budgets:
            if env == "in_distribution":
                scenarios = [
                    make_scenario(10000 + seed, 2, 2, heterogeneous=True)
                    for seed in range(args.test_seeds)
                ]
                is_comm = False
            elif env == "channel_shift":
                scenarios = [
                    make_comm_mismatch_scenario(
                        10000 + seed,
                        2,
                        2,
                        flip_hi=0.4,
                        success_lo=0.4,
                    )
                    for seed in range(args.test_seeds)
                ]
                is_comm = True
            else:
                scenarios = [
                    make_shifted_clean(10000 + seed, 2, 2, env)
                    for seed in range(args.test_seeds)
                ]
                is_comm = False
            mappo_scenarios = []
            for seed_scenario in scenarios:
                converted = []
                for target in seed_scenario:
                    if isinstance(target, tuple):
                        owner = float(target[0])
                        deltas = np.asarray(target[1], dtype=float)
                    else:
                        row = np.asarray(target, dtype=float)
                        owner = float(row[0])
                        deltas = row[1:]
                    converted.append(np.concatenate(([owner], deltas)))
                mappo_scenarios.append(converted)

            greedy = []
            wta = []
            ucb = []
            nomp = []
            exact = []
            mappo = evaluate_mappo_true(
                actors[budget],
                mappo_scenarios,
                scenarios,
                budget,
                2,
            )
            mappo_adapter = evaluate_mappo_adapter_nomp(
                actors[budget],
                scenarios,
                budget,
                2,
                state_scenarios=mappo_scenarios,
            )
            mappo_bandit = evaluate_mappo_bandit_adapter_nomp(
                actors[budget],
                scenarios,
                budget,
                2,
                state_scenarios=mappo_scenarios,
            )
            for scenario in scenarios:
                if not is_comm:
                    greedy.append(greedy_joint_multi(scenario, budget))
                    exact.append(exact_clean(scenario, budget, args.grid))
                else:
                    exact.append(exact_comm(scenario, budget, args.grid))
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
            rows.append({
                "environment": env,
                "budget": budget,
                "greedy_worst_mean": (
                    float(np.mean(greedy)) if greedy else None
                ),
                "wta_greedy_worst_mean": float(np.mean(wta)),
                "ucb_nomp_worst_mean": float(np.mean(ucb)),
                "nomp_greedy_worst_mean": float(np.mean(nomp)),
                "exact_worst_mean": float(np.mean(exact)),
                "mappo_worst_mean": float(mappo),
                "mappo_adapter_worst_mean": float(mappo_adapter),
                "mappo_bandit_worst_mean": float(mappo_bandit),
            })
            print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "unknown-environment",
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plot_rows(rows, args.figure, labels)


if __name__ == "__main__":
    main()
