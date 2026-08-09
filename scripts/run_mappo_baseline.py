"""MAPPO baseline for multi-target bit allocation and report selection.

Each target is an agent that chooses, for each of its four reports, a bit
count in {0, 1, 2, 3, 4}.  Agents share one actor (parameter-sharing MAPPO)
and one centralized critic; the reward is worst-target P_D minus a linear
penalty for exceeding the common budget.  The trained policy is evaluated
deterministically on held-out scenarios and compared with the greedy
per-report allocation and the exact joint max-min oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.joint_allocation import (
    exact_joint_maxmin,
    exact_joint_maxmin_selection,
    greedy_bits,
    moments,
    subset_options,
    target_options,
)


MAX_BITS = 4
N_REPORTS = 4
N_TARGETS = 2
GRID = 32


def _target_pd(owner_delta: float, report_deltas: np.ndarray, bits: np.ndarray) -> float:
    from uav_otfs_isac.fusion import optimal_gaussian_detection_probability

    mu0 = [0.0]
    mu1 = [owner_delta]
    var0 = [1.0]
    var1 = [1.0]
    for delta, bit_count in zip(report_deltas, bits):
        if bit_count == 0:
            continue
        m0, m1, v0, v1 = moments(float(delta), int(bit_count))
        mu0.append(m0)
        mu1.append(m1)
        var0.append(v0)
        var1.append(v1)
    return float(optimal_gaussian_detection_probability(
        np.asarray(mu0), np.asarray(mu1),
        np.diag(var0), np.diag(var1),
        set(range(len(mu0))), 0.05, grid=GRID,
    ))


def _scenario(seed: int):
    rng = np.random.default_rng(seed)
    scenario = []
    for q in range(N_TARGETS):
        strong = q % 2 == 0
        owner_delta = 0.4 if strong else 0.3
        lo, hi = (1.8, 2.2) if strong else (1.2, 1.6)
        scenario.append(np.concatenate((
            [owner_delta],
            rng.uniform(lo, hi, N_REPORTS),
        )))
    return scenario


def _state(owner_delta: float, report_deltas: np.ndarray, budget: int) -> np.ndarray:
    return np.concatenate((
        [owner_delta / 2.0],
        np.asarray(report_deltas, dtype=float) / 2.0,
        [budget / 20.0],
    ))


class Actor(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(64, MAX_BITS + 1) for _ in range(N_REPORTS)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = self.net(state)
        return torch.stack([head(hidden) for head in self.heads], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def _rollout(
    actor: Actor,
    critic: Critic,
    budget: int,
    scenarios,
) -> tuple[list, list, list, list, list]:
    states = []
    actions = []
    logprobs = []
    rewards = []
    values = []
    dones = []
    for scenario in scenarios:
        state_list = [
            _state(float(target[0]), target[1:], budget)
            for target in scenario
        ]
        global_state = np.concatenate(state_list)
        with torch.no_grad():
            logits = actor(torch.as_tensor(
                np.stack(state_list), dtype=torch.float32,
            ))
            distributions = [torch.distributions.Categorical(
                logits=logits[:, r, :],
            ) for r in range(N_REPORTS)]
            chosen = torch.stack([
                dist.sample() for dist in distributions
            ], dim=1).numpy()
            chosen_tensor = torch.as_tensor(chosen, dtype=torch.int64)
            value = critic_value(critic, global_state)
        bits = [chosen[q] for q in range(N_TARGETS)]
        costs = [int(bits[q].sum()) for q in range(N_TARGETS)]
        used = sum(costs)
        pds = [
            _target_pd(float(target[0]), target[1:], bits[q])
            for q, target in enumerate(scenario)
        ]
        worst = float(min(pds))
        reward = (
            worst
            - 0.1 * max(0, used - budget)
            - 0.05 * max(0, budget - used) / max(budget, 1)
        )
        lp = 0.0
        for r in range(N_REPORTS):
            lp += float(distributions[r].log_prob(chosen_tensor[:, r]).sum())
        states.append(state_list)
        actions.append(chosen)
        logprobs.append(lp)
        rewards.append(reward)
        values.append(value)
        dones.append(1.0)
    return states, actions, logprobs, rewards, values, dones


def critic_value(critic: Critic, global_state: np.ndarray) -> float:
    with torch.no_grad():
        return float(critic(torch.as_tensor(global_state, dtype=torch.float32)))


def _evaluate(
    actor: Actor,
    budget: int,
    scenarios,
) -> tuple[float, float, float]:
    worsts = []
    used = []
    for scenario in scenarios:
        state_list = [
            _state(float(target[0]), target[1:], budget)
            for target in scenario
        ]
        with torch.no_grad():
            logits = actor(torch.as_tensor(
                np.stack(state_list), dtype=torch.float32,
            ))
            chosen = torch.stack([
                torch.argmax(logits[:, r, :], dim=1)
                for r in range(N_REPORTS)
            ], dim=1).numpy()
        bits = [chosen[q] for q in range(N_TARGETS)]
        pds = [
            _target_pd(float(target[0]), target[1:], bits[q])
            for q, target in enumerate(scenario)
        ]
        worsts.append(float(min(pds)))
        used.append(int(bits[0].sum() + bits[1].sum()))
    return float(np.mean(worsts)), float(np.mean(used)), float(
        sum(u > budget for u in used) / len(used)
    )


def _reference(
    budget: int,
    scenarios,
    *,
    exact_max_reports: int | None = None,
    exact_max_bits: int = 4,
):
    greedy_worst = []
    exact_worst = []
    greedy_schedules = []
    exact_schedules = []
    pattern = np.array([0, 1, 2, 3, 4])
    for scenario in scenarios:
        greedy_vectors = [
            np.concatenate((
                [0], greedy_bits(target[1:], budget, GRID),
            ))
            for target in scenario
        ]
        greedy = exact_joint_maxmin(
            [
                subset_options(
                    float(target[0]), target[1:], vector[1:], GRID,
                )
                for target, vector in zip(scenario, greedy_vectors)
            ],
            budget,
        )
        exact_options = [
                target_options(
                    float(target[0]), target[1:], GRID,
                    max_bits=exact_max_bits,
                    max_reports=exact_max_reports,
                )
                for target in scenario
        ]
        exact = exact_joint_maxmin(exact_options, budget)
        _, exact_chosen = exact_joint_maxmin_selection(
            exact_options,
            budget,
        )
        greedy_worst.append(greedy)
        exact_worst.append(exact)
        greedy_schedules.append([
            vector.tolist() for vector in greedy_vectors
        ])
        exact_schedules.append([
            {"cost": cost, "value": value}
            for cost, value in exact_chosen
        ])
    return (
        float(np.mean(greedy_worst)),
        float(np.mean(exact_worst)),
        greedy_schedules,
        exact_schedules,
    )


def run_baseline(
    *,
    output: Path,
    train_seeds: int,
    test_seeds: int,
    episodes: int,
    budgets,
    exact_max_reports: int | None,
    exact_max_bits: int,
) -> None:
    summary = []
    for budget in budgets:
        train_scenarios = [_scenario(seed) for seed in range(train_seeds)]
        test_scenarios = [
            _scenario(10000 + seed) for seed in range(test_seeds)
        ]
        state_dim = N_REPORTS + 2
        actor = Actor(state_dim)
        critic = Critic(state_dim * N_TARGETS)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
        started = time.perf_counter()
        for episode in range(episodes):
            states, actions, logprobs, rewards, values, _ = _rollout(
                actor, critic, budget, [train_scenarios[episode % len(train_scenarios)]],
            )
            state_flat = torch.as_tensor(
                np.concatenate(states[0]), dtype=torch.float32,
            )
            agent_states = torch.as_tensor(
                np.stack(states[0]), dtype=torch.float32,
            )
            returns = torch.as_tensor([rewards[0]], dtype=torch.float32)
            advantages = returns - torch.as_tensor(
                [values[0]], dtype=torch.float32,
            )
            old_logprob = torch.as_tensor([logprobs[0]], dtype=torch.float32)
            action_tensor = torch.as_tensor(actions[0], dtype=torch.int64)
            for _ in range(4):
                logits = actor(agent_states)
                new_logprob = 0.0
                entropy = 0.0
                for r in range(N_REPORTS):
                    dist = torch.distributions.Categorical(logits=logits[:, r, :])
                    new_logprob += dist.log_prob(action_tensor[:, r]).sum()
                    entropy += dist.entropy().mean()
                ratio = torch.exp(new_logprob - old_logprob)
                clip = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                actor_loss = -torch.min(ratio * advantages, clip * advantages).mean()
                value_pred = critic(state_flat.unsqueeze(0))
                critic_loss = nn.functional.mse_loss(value_pred, returns)
                loss = actor_loss + critic_loss - 0.01 * entropy
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                loss.backward()
                actor_opt.step()
                critic_opt.step()
        train_seconds = time.perf_counter() - started
        mappo_worst, used_mean, over_rate = _evaluate(
            actor, budget, test_scenarios,
        )
        (
            greedy_worst,
            exact_worst,
            greedy_schedules,
            exact_schedules,
        ) = _reference(
            budget,
            test_scenarios,
            exact_max_reports=exact_max_reports,
            exact_max_bits=exact_max_bits,
        )
        summary.append({
            "budget_bits": budget,
            "exact_max_reports": exact_max_reports,
            "exact_max_bits": exact_max_bits,
            "train_episodes": episodes,
            "train_seconds": train_seconds,
            "mappo_worst_mean": mappo_worst,
            "mappo_used_bits_mean": used_mean,
            "mappo_over_budget_rate": over_rate,
            "greedy_worst_mean": greedy_worst,
            "exact_joint_worst_mean": exact_worst,
            "greedy_schedules": greedy_schedules,
            "exact_schedules": exact_schedules,
            "mappo_gap_to_exact_pp": float((exact_worst - mappo_worst) * 100.0),
            "greedy_gap_to_exact_pp": float((exact_worst - greedy_worst) * 100.0),
        })
        print(json.dumps(summary[-1], indent=2), flush=True)

    payload = {
        "gate": "mappo-baseline-multi-target",
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "episodes": episodes,
        "grid": GRID,
        "exact_max_reports": exact_max_reports,
        "exact_max_bits": exact_max_bits,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/mappo_baseline.json")
    parser.add_argument("--train-seeds", type=int, default=40)
    parser.add_argument("--test-seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--reports", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+", default=[14, 16, 18])
    parser.add_argument("--exact-max-reports", type=int, default=None)
    parser.add_argument("--exact-max-bits", type=int, default=4)
    args = parser.parse_args()
    global N_TARGETS, N_REPORTS
    N_TARGETS = args.targets
    N_REPORTS = args.reports
    run_baseline(
        output=Path(args.output),
        train_seeds=args.train_seeds,
        test_seeds=args.test_seeds,
        episodes=args.episodes,
        budgets=tuple(args.budgets),
        exact_max_reports=args.exact_max_reports,
        exact_max_bits=args.exact_max_bits,
    )


if __name__ == "__main__":
    main()
