"""Joint power-bit comparison: MAPPO vs Greedy vs winner-take-all exact.

Every method allocates both sensing power and communication bits under the
same budget.  The proposed exact method uses the winner-take-all power
reduction; Greedy uses marginal P_D gain per resource unit; MAPPO selects
bits and power with a small parameter-sharing policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.fusion import optimal_gaussian_detection_probability
from uav_otfs_isac.joint_allocation import model_from_bits
from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.power_split_theory import (
    proportional_power_bit_options,
    winner_take_all_proportional_options,
)


GRID = 16
MAX_POWER = 2
MAX_BITS = 2
POWER_OPTIONS = (0, 1, 2)
BIT_OPTIONS = (0, 1, 2)


def make_scenario(seed: int, reports: int, targets: int):
    rng = np.random.default_rng(seed)
    out = []
    for q in range(targets):
        strong = q % 2 == 0
        owner = 0.4 if strong else 0.3
        lo, hi = (1.8, 2.2) if strong else (1.2, 1.6)
        out.append(np.concatenate((
            [owner],
            rng.uniform(lo, hi, reports),
        )))
    return out


def state(owner, deltas, budget):
    return np.concatenate((
        [owner / 2.0],
        np.asarray(deltas, dtype=float) / 2.0,
        [budget / 20.0],
    ))


def pd_value(owner, deltas, powers, bits, grid=GRID):
    full_deltas = np.concatenate((
        [owner],
        np.asarray(deltas, dtype=float)
        * np.sqrt(np.maximum(np.asarray(powers, dtype=float), 0.0)),
    ))
    full_bits = np.concatenate(([0], np.asarray(bits, dtype=int)))
    model = model_from_bits(
        full_deltas, full_bits, bit_flip_probability=0.0
    )
    model = replace(
        model,
        success_prob=np.ones(model.num_uavs),
        sigma1=model.sigma0,
    )
    return float(optimal_gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1,
        set(range(model.num_uavs)), 0.05, grid=grid,
    ))


class Actor(nn.Module):
    def __init__(self, state_dim, reports):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.bit_heads = nn.ModuleList([
            nn.Linear(64, len(BIT_OPTIONS)) for _ in range(reports)
        ])
        self.power_heads = nn.ModuleList([
            nn.Linear(64, len(POWER_OPTIONS)) for _ in range(reports)
        ])

    def forward(self, x):
        h = self.net(x)
        bits = torch.stack([head(h) for head in self.bit_heads], dim=1)
        powers = torch.stack([head(h) for head in self.power_heads], dim=1)
        return bits, powers


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rollout(actor, critic, scenario, budget):
    states = [state(float(t[0]), t[1:], budget) for t in scenario]
    global_state = np.concatenate(states)
    with torch.no_grad():
        logits_b, logits_p = actor(torch.as_tensor(
            np.stack(states), dtype=torch.float32
        ))
        bits = torch.stack([
            torch.distributions.Categorical(logits=logits_b[:, r, :]).sample()
            for r in range(len(scenario[0]) - 1)
        ], dim=1).numpy()
        powers = torch.stack([
            torch.distributions.Categorical(logits=logits_p[:, r, :]).sample()
            for r in range(len(scenario[0]) - 1)
        ], dim=1).numpy()
        value = float(critic(torch.as_tensor(
            global_state, dtype=torch.float32
        )))
    pds = [
        pd_value(float(t[0]), t[1:], powers[q], bits[q])
        for q, t in enumerate(scenario)
    ]
    used = int(powers.sum() + bits.sum())
    worst = float(np.min(pds))
    reward = worst - 0.1 * max(0, used - budget)
    return reward, value, bits, powers


def train_mappo(scenarios, budget, episodes, reports):
    state_dim = reports + 2
    actor = Actor(state_dim, reports)
    critic = Critic(state_dim * len(scenarios[0]))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        reward, value, bits, powers = rollout(
            actor, critic, scenario, budget
        )
        states = [state(float(t[0]), t[1:], budget) for t in scenario]
        agent_states = torch.as_tensor(np.stack(states), dtype=torch.float32)
        global_state = torch.as_tensor(
            np.concatenate(states), dtype=torch.float32
        )
        returns = torch.as_tensor([reward], dtype=torch.float32)
        advantage = returns - torch.as_tensor([value], dtype=torch.float32)
        bit_target = torch.as_tensor(bits, dtype=torch.int64)
        power_target = torch.as_tensor(powers, dtype=torch.int64)
        logits_b, logits_p = actor(agent_states)
        lp = 0.0
        for r in range(reports):
            lp += torch.distributions.Categorical(
                logits=logits_b[:, r, :]
            ).log_prob(bit_target[:, r]).sum()
            lp += torch.distributions.Categorical(
                logits=logits_p[:, r, :]
            ).log_prob(power_target[:, r]).sum()
        loss = -advantage * lp
        vp = critic(global_state.unsqueeze(0))
        loss = loss + nn.functional.mse_loss(vp, returns)
        actor_opt.zero_grad(); critic_opt.zero_grad()
        loss.backward(); actor_opt.step(); critic_opt.step()
    return actor


def evaluate_mappo(actor, scenarios, budget, reports):
    worsts = []
    for scenario in scenarios:
        states = [state(float(t[0]), t[1:], budget) for t in scenario]
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
        pds = [
            pd_value(float(t[0]), t[1:], powers[q], bits[q])
            for q, t in enumerate(scenario)
        ]
        worsts.append(float(np.min(pds)))
    return float(np.mean(worsts))


def greedy_joint(owner, deltas, budget):
    reports = len(deltas)
    powers = np.zeros(reports, dtype=int)
    bits = np.ones(reports, dtype=int)
    used = reports
    if used > budget:
        bits = np.zeros(reports, dtype=int)
        used = 0
    while True:
        best = None
        for r in range(reports):
            for action in ("power", "bit"):
                trial_p = powers.copy(); trial_b = bits.copy()
                if action == "power":
                    if trial_p[r] >= MAX_POWER:
                        continue
                    trial_p[r] += 1
                else:
                    if trial_b[r] >= MAX_BITS:
                        continue
                    trial_b[r] += 1
                cost = used + 1
                if cost > budget:
                    continue
                gain = pd_value(
                    owner, deltas, trial_p, trial_b
                ) - pd_value(owner, deltas, powers, bits)
                key = (gain, r, action)
                if best is None or key > best[0]:
                    best = (key, trial_p, trial_b)
        if best is None or best[0][0] <= 0:
            break
        _, powers, bits = best
        used += 1
    return pd_value(owner, deltas, powers, bits)


def greedy_joint_multi(scenario, budget):
    reports = len(scenario[0]) - 1
    powers = [np.zeros(reports, dtype=int) for _ in scenario]
    bits = [np.ones(reports, dtype=int) for _ in scenario]
    used = 2 * reports * len(scenario)
    if used > budget:
        powers = [np.zeros(reports, dtype=int) for _ in scenario]
        bits = [np.zeros(reports, dtype=int) for _ in scenario]
        used = 0
    else:
        powers = [np.ones(reports, dtype=int) for _ in scenario]
        bits = [np.ones(reports, dtype=int) for _ in scenario]

    def worst():
        return min(
            pd_value(float(t[0]), t[1:], powers[q], bits[q])
            for q, t in enumerate(scenario)
        )

    while True:
        current = worst()
        best = None
        for q, target in enumerate(scenario):
            for r in range(reports):
                for action in ("power", "bit"):
                    trial_p = powers[q].copy()
                    trial_b = bits[q].copy()
                    if action == "power":
                        if trial_p[r] >= MAX_POWER:
                            continue
                        trial_p[r] += 1
                    else:
                        if trial_b[r] >= MAX_BITS:
                            continue
                        trial_b[r] += 1
                    if used + 1 > budget:
                        continue
                    old = powers[q].copy(), bits[q].copy()
                    powers[q], bits[q] = trial_p, trial_b
                    new_worst = worst()
                    gain = current - new_worst
                    powers[q], bits[q] = old
                    key = (gain, q, r, action)
                    if best is None or key > best[0]:
                        best = (key, q, r, action)
        if best is None or best[0][0] <= 0:
            break
        _, q, r, action = best
        if action == "power":
            powers[q][r] += 1
        else:
            bits[q][r] += 1
        used += 1
    return worst()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_power_comparison.json")
    parser.add_argument("--reports", type=int, default=2)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--test-seeds", type=int, default=20)
    args = parser.parse_args()

    train_scenarios = [
        make_scenario(seed, args.reports, args.targets)
        for seed in range(args.train_seeds)
    ]
    test_scenarios = [
        make_scenario(10000 + seed, args.reports, args.targets)
        for seed in range(args.test_seeds)
    ]
    summary = []
    for budget in args.budgets:
        start = time.perf_counter()
        actor = train_mappo(
            train_scenarios, budget, args.episodes, args.reports
        )
        mappo_worst = evaluate_mappo(
            actor, test_scenarios, budget, args.reports
        )
        train_seconds = time.perf_counter() - start
        greedy_worsts = []
        exact_worsts = []
        winner_worsts = []
        for scenario in test_scenarios:
            greedy_worsts.append(greedy_joint_multi(scenario, budget))
            full_groups = [
                proportional_power_bit_options(
                    float(t[0]), t[1:],
                    power_levels=np.arange(budget + 1, dtype=float),
                    bit_options=np.arange(MAX_BITS + 1, dtype=int),
                    budget=budget, grid=GRID,
                )
                for t in scenario
            ]
            winner_groups = [
                winner_take_all_proportional_options(
                    float(t[0]), t[1:],
                    bit_options=np.arange(MAX_BITS + 1, dtype=int),
                    budget=budget, grid=GRID,
                )
                for t in scenario
            ]
            exact_worsts.append(exact_joint_power_bit_maxmin(
                full_groups, budget
            ))
            winner_worsts.append(exact_joint_power_bit_maxmin(
                winner_groups, budget
            ))
        summary.append({
            "budget": budget,
            "mappo_worst_mean": mappo_worst,
            "greedy_worst_mean": float(np.mean(greedy_worsts)),
            "exact_worst_mean": float(np.mean(exact_worsts)),
            "winner_worst_mean": float(np.mean(winner_worsts)),
            "train_seconds": train_seconds,
        })
    payload = {
        "gate": "joint-power-comparison",
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
