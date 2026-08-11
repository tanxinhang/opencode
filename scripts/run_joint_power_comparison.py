"""Joint power-bit comparison: MAPPO vs Greedy vs winner-take-all exact.

Every method allocates both sensing power and communication bits under the
same budget.  The proposed exact method uses the winner-take-all power
reduction; Greedy uses marginal P_D gain per resource unit; MAPPO selects
bits and power with a small parameter-sharing policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.mappo_nomp_adapter import (
    MappoNompAdapter,
    ModeBanditAdapter,
    NompRequirement,
)
from uav_otfs_isac import nomp_refinement as nomp
from uav_otfs_isac.power_split_theory import (
    power_gain_coefficient,
    proportional_power_bit_options,
    proportional_target_pd,
    winner_take_all_proportional_options,
)


GRID = 16
MAX_BITS = 2
BIT_OPTIONS = (0, 1, 2)


def make_scenario(seed: int, reports: int, targets: int, heterogeneous=False):
    rng = np.random.default_rng(seed)
    out = []
    for q in range(targets):
        if heterogeneous:
            owner = rng.uniform(0.2, 0.5)
            out.append(np.concatenate((
                [owner],
                rng.uniform(0.5, 2.5, reports),
            )))
            continue
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


def robust_state(target, budget):
    """Channel-aware MAPPO state including per-report flip/success."""
    owner, deltas, flips, successes = nomp.parse_target(target)
    return np.concatenate((
        [float(owner) / 2.0],
        np.asarray(deltas, dtype=float) / 2.0,
        [float(budget) / 20.0],
        np.asarray(flips, dtype=float) / 0.5,
        np.asarray(successes, dtype=float),
    ))


def pd_value(owner, deltas, powers, bits, grid=GRID):
    return float(proportional_target_pd(
        float(owner),
        np.asarray(deltas, dtype=float),
        np.asarray(powers, dtype=float),
        np.asarray(bits, dtype=int),
        grid,
    ))


class Actor(nn.Module):
    def __init__(self, state_dim, reports, power_options=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.bit_heads = nn.ModuleList([
            nn.Linear(64, len(BIT_OPTIONS)) for _ in range(reports)
        ])
        self.power_heads = nn.ModuleList([
            nn.Linear(64, power_options) for _ in range(reports)
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
    actor = Actor(state_dim, reports, power_options=budget + 1)
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
        bit_target = torch.as_tensor(
            np.asarray(bits), dtype=torch.int64
        )
        power_target = torch.as_tensor(
            np.asarray(powers), dtype=torch.int64
        )
        logits_b, logits_p = actor(agent_states)
        lp = 0.0
        entropy = 0.0
        for r in range(reports):
            dist_b = torch.distributions.Categorical(
                logits=logits_b[:, r, :]
            )
            dist_p = torch.distributions.Categorical(
                logits=logits_p[:, r, :]
            )
            lp += dist_b.log_prob(bit_target[:, r]).sum()
            lp += dist_p.log_prob(power_target[:, r]).sum()
            entropy += dist_b.entropy().sum() + dist_p.entropy().sum()
        loss = -advantage * lp - 0.01 * entropy
        vp = critic(global_state.unsqueeze(0))
        loss = loss + nn.functional.mse_loss(vp, returns)
        actor_opt.zero_grad(); critic_opt.zero_grad()
        loss.backward(); actor_opt.step(); critic_opt.step()
    return actor


def train_mappo_nomp_reward(scenarios, budget, episodes, reports):
    """Train MAPPO with NOMP-final P_D as reward (middleware-informed)."""
    state_dim = reports + 2
    actor = Actor(state_dim, reports, power_options=budget + 1)
    critic = Critic(state_dim * len(scenarios[0]))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        states = [state(float(t[0]), t[1:], budget) for t in scenario]
        global_state = torch.as_tensor(
            np.concatenate(states), dtype=torch.float32
        )
        agent_states = torch.as_tensor(
            np.stack(states), dtype=torch.float32
        )
        with torch.no_grad():
            logits_b, logits_p = actor(agent_states)
            bits = torch.stack([
                torch.distributions.Categorical(
                    logits=logits_b[:, r, :]
                ).sample()
                for r in range(reports)
            ], dim=1).numpy()
            powers = torch.stack([
                torch.distributions.Categorical(
                    logits=logits_p[:, r, :]
                ).sample()
                for r in range(reports)
            ], dim=1).numpy()
            value = float(critic(global_state))
        powers, bits = _feasible_from_mappo(
            scenario, powers, bits, budget
        )
        powers, bits, _ = nomp.maxmin_refine(
            scenario,
            powers,
            bits,
            max_power=budget,
            max_bits=MAX_BITS,
            max_rounds=50,
            grid=GRID,
        )
        reward = float(min(nomp.target_scores(
            scenario, powers, bits, GRID
        )))
        returns = torch.as_tensor([reward], dtype=torch.float32)
        advantage = returns - torch.as_tensor([value], dtype=torch.float32)
        bit_target = torch.as_tensor(
            np.asarray(bits), dtype=torch.int64
        )
        power_target = torch.as_tensor(
            np.asarray(powers), dtype=torch.int64
        )
        logits_b, logits_p = actor(agent_states)
        lp = 0.0
        entropy = 0.0
        for r in range(reports):
            dist_b = torch.distributions.Categorical(
                logits=logits_b[:, r, :]
            )
            dist_p = torch.distributions.Categorical(
                logits=logits_p[:, r, :]
            )
            lp += dist_b.log_prob(bit_target[:, r]).sum()
            lp += dist_p.log_prob(power_target[:, r]).sum()
            entropy += dist_b.entropy().sum() + dist_p.entropy().sum()
        loss = -advantage * lp - 0.01 * entropy
        vp = critic(global_state.unsqueeze(0))
        loss = loss + nn.functional.mse_loss(vp, returns)
        actor_opt.zero_grad(); critic_opt.zero_grad()
        loss.backward(); actor_opt.step(); critic_opt.step()
    return actor


def train_mappo_ppo(
    scenarios,
    budget,
    episodes,
    reports,
    *,
    ppo_epochs: int = 4,
    batch_size: int = 8,
    mini_batch_size: int = 4,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    learning_rate: float = 1e-3,
    reward_mode: str = "raw",
    state_builder=None,
):
    """Real PPO: clipped surrogate, mini-batches, normalized advantages."""
    if state_builder is None:
        state_builder = lambda target, budget: state(
            float(target[0]), target[1:], budget
        )
    state_dim = len(state_builder(scenarios[0][0], budget))
    actor = Actor(state_dim, reports, power_options=budget + 1)
    critic = Critic(state_dim * len(scenarios[0]))
    actor_opt = torch.optim.Adam(
        actor.parameters(), lr=learning_rate
    )
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=learning_rate
    )
    buffer = []

    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        states = [state_builder(t, budget) for t in scenario]
        global_state = np.concatenate(states)
        agent_states = torch.as_tensor(
            np.stack(states), dtype=torch.float32
        )
        with torch.no_grad():
            logits_b, logits_p = actor(agent_states)
            dist_b = [
                torch.distributions.Categorical(logits=logits_b[:, r, :])
                for r in range(reports)
            ]
            dist_p = [
                torch.distributions.Categorical(logits=logits_p[:, r, :])
                for r in range(reports)
            ]
            bits = torch.stack([
                dist_b[r].sample() for r in range(reports)
            ], dim=1)
            powers = torch.stack([
                dist_p[r].sample() for r in range(reports)
            ], dim=1)
            old_lp = sum(
                dist_b[r].log_prob(bits[:, r]).sum()
                + dist_p[r].log_prob(powers[:, r]).sum()
                for r in range(reports)
            )
            value = float(critic(torch.as_tensor(
                global_state, dtype=torch.float32
            )))
        if reward_mode == "raw":
            pds = [
                pd_value(
                    float(t[0]),
                    t[1:],
                    powers[q].numpy(),
                    bits[q].numpy(),
                )
                for q, t in enumerate(scenario)
            ]
            used = int(powers.sum().item() + bits.sum().item())
            reward = float(np.min(pds)) - 0.1 * max(0, used - budget)
        else:
            proposal_powers = powers.numpy()
            proposal_bits = bits.numpy()
            proposal_powers, proposal_bits = _feasible_from_mappo(
                scenario, proposal_powers, proposal_bits, budget
            )
            proposal_powers, proposal_bits, _ = nomp.maxmin_refine(
                scenario,
                proposal_powers,
                proposal_bits,
                max_power=budget,
                max_bits=MAX_BITS,
                max_rounds=50,
                grid=GRID,
            )
            reward = float(min(nomp.target_scores(
                scenario, proposal_powers, proposal_bits, GRID
            )))
        buffer.append({
            "agent_states": agent_states,
            "global_state": torch.as_tensor(
                global_state, dtype=torch.float32
            ),
            "bits": bits,
            "powers": powers,
            "old_lp": old_lp,
            "reward": reward,
            "value": value,
        })
        if len(buffer) < batch_size:
            continue

        agent_batch = torch.stack([item["agent_states"] for item in buffer])
        global_batch = torch.stack([item["global_state"] for item in buffer])
        bits_batch = torch.stack([item["bits"] for item in buffer])
        powers_batch = torch.stack([item["powers"] for item in buffer])
        old_lp_batch = torch.stack([item["old_lp"] for item in buffer])
        returns = torch.as_tensor(
            [item["reward"] for item in buffer], dtype=torch.float32
        )
        values = torch.as_tensor(
            [item["value"] for item in buffer], dtype=torch.float32
        )
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (
            advantage.std() + 1e-8
        )

        for _ in range(ppo_epochs):
            order = torch.randperm(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                idx = order[start:start + mini_batch_size]
                logits_b, logits_p = actor(agent_batch[idx])
                new_lp = 0.0
                entropy = 0.0
                for r in range(reports):
                    dist_b = torch.distributions.Categorical(
                        logits=logits_b[:, r, :]
                    )
                    dist_p = torch.distributions.Categorical(
                        logits=logits_p[:, r, :]
                    )
                    new_lp = new_lp + dist_b.log_prob(
                        bits_batch[idx][:, :, r]
                    ).sum(dim=1) + dist_p.log_prob(
                        powers_batch[idx][:, :, r]
                    ).sum(dim=1)
                    entropy = entropy + dist_b.entropy().sum(dim=1) + (
                        dist_p.entropy().sum(dim=1)
                    )
                ratio = torch.exp(new_lp - old_lp_batch[idx])
                adv = advantage[idx]
                clip = torch.clamp(
                    ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                )
                policy_loss = -torch.mean(torch.min(
                    ratio * adv, clip * adv
                )) - entropy_coef * torch.mean(entropy)
                vpred = critic(global_batch[idx])
                value_loss = nn.functional.mse_loss(
                    vpred, returns[idx]
                )
                loss = policy_loss + value_loss
                actor_opt.zero_grad(); critic_opt.zero_grad()
                loss.backward()
                actor_opt.step(); critic_opt.step()
        buffer.clear()
    return actor


def robust_rollout(actor, critic, scenario, budget):
    states = [robust_state(t, budget) for t in scenario]
    global_state = np.concatenate(states)
    reports = nomp._report_count(scenario[0])
    with torch.no_grad():
        logits_b, logits_p = actor(torch.as_tensor(
            np.stack(states), dtype=torch.float32
        ))
        bits = torch.stack([
            torch.distributions.Categorical(
                logits=logits_b[:, r, :]
            ).sample()
            for r in range(reports)
        ], dim=1).numpy()
        powers = torch.stack([
            torch.distributions.Categorical(
                logits=logits_p[:, r, :]
            ).sample()
            for r in range(reports)
        ], dim=1).numpy()
        value = float(critic(torch.as_tensor(
            global_state, dtype=torch.float32
        )))
    pds = nomp.target_scores(scenario, powers, bits, GRID)
    used = int(powers.sum() + bits.sum())
    worst = float(np.min(pds))
    reward = worst - 0.1 * max(0, used - budget)
    return reward, value, bits, powers


def train_robust_mappo(scenarios, budget, episodes, reports):
    """Curriculum-trained MAPPO with channel-aware state."""
    state_dim = 2 + 3 * reports
    actor = Actor(state_dim, reports, power_options=budget + 1)
    critic = Critic(state_dim * len(scenarios[0]))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        reward, value, bits, powers = robust_rollout(
            actor, critic, scenario, budget
        )
        states = [robust_state(t, budget) for t in scenario]
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
        entropy = 0.0
        for r in range(reports):
            dist_b = torch.distributions.Categorical(
                logits=logits_b[:, r, :]
            )
            dist_p = torch.distributions.Categorical(
                logits=logits_p[:, r, :]
            )
            lp += dist_b.log_prob(bit_target[:, r]).sum()
            lp += dist_p.log_prob(power_target[:, r]).sum()
            entropy += dist_b.entropy().sum() + dist_p.entropy().sum()
        loss = -advantage * lp - 0.01 * entropy
        vp = critic(global_state.unsqueeze(0))
        loss = loss + nn.functional.mse_loss(vp, returns)
        actor_opt.zero_grad(); critic_opt.zero_grad()
        loss.backward(); actor_opt.step(); critic_opt.step()
    return actor


def evaluate_robust_mappo(actor, scenarios, budget, reports):
    worsts = []
    for scenario in scenarios:
        states = [robust_state(t, budget) for t in scenario]
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
            scenario, powers, bits, budget
        )
        pds = nomp.target_scores(scenario, powers, bits, GRID)
        worsts.append(float(np.min(pds)))
    return float(np.mean(worsts))


def evaluate_robust_mappo_nomp(actor, scenarios, budget, reports):
    """Channel-aware MAPPO proposal refined by NOMP."""
    worsts = []
    for scenario in scenarios:
        states = [robust_state(t, budget) for t in scenario]
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
            scenario, powers, bits, budget
        )
        powers, bits, _ = nomp.maxmin_refine(
            scenario,
            powers,
            bits,
            max_power=budget,
            max_bits=MAX_BITS,
            max_rounds=100,
            grid=GRID,
        )
        worsts.append(float(min(nomp.target_scores(
            scenario, powers, bits, GRID
        ))))
    return float(np.mean(worsts))


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


def _feasible_from_mappo(scenario, powers, bits, budget):
    """Drop power units until the MAPPO proposal is budget feasible."""
    q_count = len(scenario)
    reports = len(powers[0])
    powers = [np.asarray(row, dtype=int).copy() for row in powers]
    bits = [np.asarray(row, dtype=int).copy() for row in bits]
    used = int(sum(powers[q].sum() + bits[q].sum() for q in range(q_count)))

    def score(row_p, row_b):
        return float(np.min(nomp.target_scores(
            scenario,
            row_p,
            row_b,
            GRID,
        )))

    while used > budget:
        best = None
        for q in range(q_count):
            for r in range(reports):
                if powers[q][r] <= 0:
                    continue
                trial_p = [row.copy() for row in powers]
                trial_p[q][r] -= 1
                loss = score(powers, bits) - score(trial_p, bits)
                key = (loss, q, r)
                if best is None or key < best[0]:
                    best = (key, q, r)
        if best is None:
            break
        _, q, r = best
        powers[q][r] -= 1
        used -= 1
    return powers, bits


def evaluate_mappo_nomp(
    actor, scenarios, budget, reports, state_scenarios=None
):
    """MAPPO proposes report activation/bits, NOMP refines sensing power."""
    worsts = []
    for index, scenario in enumerate(scenarios):
        state_scenario = (
            state_scenarios[index] if state_scenarios is not None
            else scenario
        )
        states = [state(float(t[0]), t[1:], budget) for t in state_scenario]
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
            scenario, powers, bits, budget
        )
        powers, bits, _ = nomp.maxmin_refine(
            scenario,
            powers,
            bits,
            max_power=budget,
            max_bits=MAX_BITS,
            max_rounds=100,
            grid=GRID,
        )
        worsts.append(float(min(nomp.target_scores(
            scenario, powers, bits, GRID
        ))))
    return float(np.mean(worsts))


def evaluate_mappo_nomp_multi(
    actor,
    scenarios,
    budget,
    reports,
    *,
    samples: int = 6,
    state_scenarios=None,
    max_rounds: int = 100,
    candidate_budget: int = 32,
    temperatures=(1.0, 2.0),
    patience: int = 2,
    seed: int = 0,
):
    """PPO+NOMP with multiple sampled proposals, keeping the best refined one.

    This is the Bandit-style mechanism without an explicit mode registry:
    NOMP refines several PPO proposals and the best max-min schedule is kept,
    so a single bad argmax proposal cannot drag the hybrid down.
    """
    torch.manual_seed(seed)
    worsts = []
    for index, scenario in enumerate(scenarios):
        state_scenario = (
            state_scenarios[index] if state_scenarios is not None
            else scenario
        )
        states = [state(float(t[0]), t[1:], budget) for t in state_scenario]
        with torch.no_grad():
            logits_b, logits_p = actor(torch.as_tensor(
                np.stack(states), dtype=torch.float32
            ))
            dist_b = [
                torch.distributions.Categorical(logits=logits_b[:, r, :])
                for r in range(reports)
            ]
            dist_p = [
                torch.distributions.Categorical(logits=logits_p[:, r, :])
                for r in range(reports)
            ]
            best_score = -1.0
            stalled = 0
            per_temperature = max(int(np.ceil(samples / len(temperatures))), 1)
            for temperature in temperatures:
                for _ in range(per_temperature):
                    if temperature is None:
                        bits = torch.stack([
                            torch.argmax(logits_b[:, r, :], dim=1)
                            for r in range(reports)
                        ], dim=1).numpy()
                        powers = torch.stack([
                            torch.argmax(logits_p[:, r, :], dim=1)
                            for r in range(reports)
                        ], dim=1).numpy()
                    else:
                        bits = torch.stack([
                            torch.distributions.Categorical(
                                logits=logits_b[:, r, :] / float(temperature)
                            ).sample()
                            for r in range(reports)
                        ], dim=1).numpy()
                        powers = torch.stack([
                            torch.distributions.Categorical(
                                logits=logits_p[:, r, :] / float(temperature)
                            ).sample()
                            for r in range(reports)
                        ], dim=1).numpy()
                powers, bits = _feasible_from_mappo(
                    scenario, powers, bits, budget
                )
                powers, bits, _ = nomp.maxmin_refine(
                    scenario,
                    powers,
                    bits,
                    max_power=budget,
                    max_bits=MAX_BITS,
                    max_rounds=max_rounds,
                    grid=GRID,
                    candidate_budget=candidate_budget,
                )
                score = float(min(nomp.target_scores(
                    scenario, powers, bits, GRID
                )))
                if not np.isfinite(score):
                    continue
                if score <= best_score + 1e-12:
                    stalled += 1
                else:
                    stalled = 0
                best_score = max(best_score, score)
                if stalled >= patience:
                    break
            if stalled >= patience:
                break
        if best_score < 0.0:
            with torch.no_grad():
                bits = torch.stack([
                    torch.argmax(logits_b[:, r, :], dim=1)
                    for r in range(reports)
                ], dim=1).numpy()
                powers = torch.stack([
                    torch.argmax(logits_p[:, r, :], dim=1)
                    for r in range(reports)
                ], dim=1).numpy()
            powers, bits = _feasible_from_mappo(
                scenario, powers, bits, budget
            )
            powers, bits, _ = nomp.maxmin_refine(
                scenario,
                powers,
                bits,
                max_power=budget,
                max_bits=MAX_BITS,
                max_rounds=max_rounds,
                grid=GRID,
                candidate_budget=candidate_budget,
            )
            fallback = float(min(nomp.target_scores(
                scenario, powers, bits, GRID
            )))
            if np.isfinite(fallback):
                best_score = fallback
        worsts.append(best_score)
    return float(np.mean(worsts))


def evaluate_mappo_probe_nomp(actor, scenarios, budget, reports):
    """MAPPO chooses which reports to probe, NOMP allocates bits/power."""
    worsts = []
    for scenario in scenarios:
        states = [state(float(t[0]), t[1:], budget) for t in scenario]
        with torch.no_grad():
            logits_b, _ = actor(torch.as_tensor(
                np.stack(states), dtype=torch.float32
            ))
            bits = torch.stack([
                torch.argmax(logits_b[:, r, :], dim=1)
                for r in range(reports)
            ], dim=1).numpy()
        mask = [np.asarray(bits[q] > 0, dtype=int) for q in range(len(scenario))]
        if any(int(row.sum()) > 0 for row in mask):
            result = nomp.nomp_wta_greedy_joint_multi(
                scenario, budget, probe_mask=mask
            )
        else:
            result = nomp.nomp_wta_greedy_joint_multi(scenario, budget)
        worsts.append(float(result["worst_pd"]))
    return float(np.mean(worsts))


def evaluate_mappo_adapter_nomp(
    actor,
    scenarios,
    budget,
    reports,
    iters=3,
    state_scenarios=None,
    state_builder=None,
):
    """Adapter repeatedly translates MAPPO rollouts into NOMP inputs."""
    worsts = []
    adapter = MappoNompAdapter(actor, state_builder=state_builder)
    for scenario_index, scenario in enumerate(scenarios):
        state_scenario = (
            state_scenarios[scenario_index]
            if state_scenarios is not None
            else scenario
        )
        requirement = NompRequirement(
            modes="auto",
            budget=budget,
        )
        result = adapter.propose_and_allocate(
            scenario,
            requirement,
            seed=scenario_index,
            sample=True,
            iters=iters,
            state_scenario=state_scenario,
        )
        worsts.append(float(result["worst_pd"]))
    return float(np.mean(worsts))


def evaluate_mappo_bandit_adapter_nomp(
    actor,
    scenarios,
    budget,
    reports,
    iters=5,
    state_scenarios=None,
    state_builder=None,
):
    """UCB adapter learns which MAPPO information NOMP should request."""
    worsts = []
    adapter = ModeBanditAdapter(actor, state_builder=state_builder)
    for scenario_index, scenario in enumerate(scenarios):
        state_scenario = (
            state_scenarios[scenario_index]
            if state_scenarios is not None
            else scenario
        )
        requirement = NompRequirement(
            modes="auto",
            budget=budget,
        )
        result = adapter.propose_and_allocate(
            scenario,
            requirement,
            seed=scenario_index,
            sample=True,
            iters=iters,
            state_scenario=state_scenario,
        )
        worsts.append(float(result["worst_pd"]))
    return float(np.mean(worsts))


def greedy_joint_multi(scenario, budget, initialization="equal", max_power=None):
    if max_power is None:
        max_power = budget
    reports = len(scenario[0]) - 1
    if initialization == "equal":
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
    else:
        powers = [np.zeros(reports, dtype=int) for _ in scenario]
        bits = [np.zeros(reports, dtype=int) for _ in scenario]
        used = 0
        for q, target in enumerate(scenario):
            coefficients = np.asarray([
                power_gain_coefficient(
                    float(target[r + 1]), 1, 0.0, 1.0
                )
                for r in range(reports)
            ])
            winner = int(np.argmax(coefficients))
            bits[q][winner] = 1
            powers[q][winner] = 1
            used += 2
        if used > budget:
            powers = [np.zeros(reports, dtype=int) for _ in scenario]
            bits = [np.zeros(reports, dtype=int) for _ in scenario]
            used = 0

    def scores():
        return np.asarray([
            pd_value(float(t[0]), t[1:], powers[q], bits[q])
            for q, t in enumerate(scenario)
        ])

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
                        if trial_p[r] >= max_power:
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
    # Reallocation phase: move one unit from a surplus report to a bottleneck
    # target/report when it improves the global worst target.
    while True:
        current = worst()
        best_move = None
        for qa in range(len(scenario)):
            for ra in range(reports):
                for qb in range(len(scenario)):
                    for rb in range(reports):
                        if qa == qb and ra == rb:
                            continue
                        for kind in ("bit", "power"):
                            if kind == "bit":
                                if bits[qa][ra] <= 1 or bits[qb][rb] >= MAX_BITS:
                                    continue
                                old_a, old_b = bits[qa].copy(), bits[qb].copy()
                                bits[qa][ra] -= 1
                                bits[qb][rb] += 1
                            else:
                                if (
                                    powers[qa][ra] <= 1
                                    or powers[qb][rb] >= max_power
                                ):
                                    continue
                                old_a, old_b = powers[qa].copy(), powers[qb].copy()
                                powers[qa][ra] -= 1
                                powers[qb][rb] += 1
                            new_worst = worst()
                            gain = current - new_worst
                            if kind == "bit":
                                bits[qa], bits[qb] = old_a, old_b
                            else:
                                powers[qa], powers[qb] = old_a, old_b
                            if gain > 0:
                                key = (gain, qa, ra, qb, rb, kind)
                                if best_move is None or key > best_move[0]:
                                    best_move = (key, qa, ra, qb, rb, kind)
        if best_move is None:
            break
        _, qa, ra, qb, rb, kind = best_move
        if kind == "bit":
            bits[qa][ra] -= 1
            bits[qb][rb] += 1
        else:
            powers[qa][ra] -= 1
            powers[qb][rb] += 1
    return worst()


def wta_greedy_joint_multi(scenario, budget):
    """Online WTA greedy wrapper (power action space matches the oracle)."""
    return float(nomp.wta_greedy_joint_multi(
        scenario, budget, min_cover=False
    )["worst_pd"])


def ucb_wta_greedy_joint_multi(
    scenario,
    budget,
    *,
    noise_scale,
    seed,
    max_steps: int = 100,
    min_cover: bool = False,
    refine: bool = False,
    max_refine_rounds: int = 100,
    max_power=None,
    confidence: float = 0.05,
    max_feedback_rounds: int = 20,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    floors=None,
    weights=None,
):
    """Online WTA-Greedy whose winner/activation use UCB error estimates."""
    reports = nomp._report_count(scenario[0])
    rng = np.random.default_rng(seed)
    power_cap = budget if max_power is None else int(max_power)
    if min_cover:
        powers, bits, used = nomp.initial_min_cover(
            scenario,
            budget,
            flip_probability=flip_probability,
            success_probability=success_probability,
            grid=GRID,
        )
    else:
        powers = [np.zeros(reports, dtype=int) for _ in scenario]
        bits = [np.zeros(reports, dtype=int) for _ in scenario]
        used = 0
    means = []
    counts = []
    for q, target in enumerate(scenario):
        _, deltas, flips, successes = nomp.parse_target(
            target, flip_probability, success_probability
        )
        true = np.asarray([
            power_gain_coefficient(
                float(deltas[r]),
                1,
                float(flips[r]),
                float(successes[r]),
            )
            for r in range(reports)
        ])
        means.append(
            true + noise_scale * rng.standard_normal(reports)
        )
        counts.append(np.ones(reports, dtype=float))
    beta = float(norm.ppf(1.0 - confidence / (2.0 * reports)))
    prior_noise_scale = max(noise_scale, 0.1)

    def scores():
        raw = np.asarray(nomp.target_scores(
            scenario,
            powers,
            bits,
            GRID,
            flip_probability,
            success_probability,
        ))
        if floors is not None:
            return np.asarray(nomp.qos_scores(raw, floors, weights))
        return raw

    def width(q):
        return beta * prior_noise_scale / np.sqrt(counts[q])

    def ucb(q):
        return means[q] + width(q)

    def observe(q, r):
        _, deltas, flips, successes = nomp.parse_target(
            scenario[q], flip_probability, success_probability
        )
        observed = (
            power_gain_coefficient(
                float(deltas[r]),
                1,
                float(flips[r]),
                float(successes[r]),
            )
            + noise_scale * rng.standard_normal()
        )
        means[q][r] = (
            means[q][r] * counts[q][r] + observed
        ) / (counts[q][r] + 1.0)
        counts[q][r] += 1.0

    def certificate_status():
        for q in range(len(scenario)):
            active = [r for r in range(reports) if bits[q][r] > 0]
            if not active:
                return False
            values = ucb(q)[active]
            winner = active[int(np.argmax(values))]
            if powers[q][winner] <= 0:
                return False
            lcb_best = means[q][winner] - width(q)[winner]
            all_ucb = ucb(q).copy()
            all_ucb[winner] = -np.inf
            second = int(np.argmax(all_ucb))
            ucb_second = all_ucb[second]
            if lcb_best <= ucb_second:
                return False
        return True

    steps_used = 0
    while True:
        current = float(np.mean(scores()))
        best = None
        for q, target in enumerate(scenario):
            active = [r for r in range(reports) if bits[q][r] > 0]
            for r in range(reports):
                if bits[q][r] > 0 or used + 2 > budget:
                    continue
                old_b, old_p = bits[q].copy(), powers[q].copy()
                bits[q][r] = 1
                powers[q][r] = 1
                new_score = float(np.mean(scores()))
                gain = new_score - current
                bits[q], powers[q] = old_b, old_p
                if gain > 0:
                    key = (gain / 2.0, gain, ucb(q)[r], q, "activate", r)
                    if best is None or key > best[0]:
                        best = (key, q, "activate", r)
            for r in active:
                if bits[q][r] >= MAX_BITS or used + 1 > budget:
                    continue
                old_b = bits[q].copy()
                bits[q][r] += 1
                new_score = float(np.mean(scores()))
                gain = new_score - current
                bits[q] = old_b
                if gain > 0:
                    key = (gain, gain, ucb(q)[r], q, "bit", r)
                    if best is None or key > best[0]:
                        best = (key, q, "bit", r)
            if active:
                winner = active[int(np.argmax(ucb(q)[active]))]
                if powers[q][winner] < power_cap and used + 1 <= budget:
                    old_p = powers[q].copy()
                    powers[q][winner] += 1
                    new_score = float(np.mean(scores()))
                    gain = new_score - current
                    powers[q] = old_p
                    if gain > 0:
                        key = (gain, gain, ucb(q)[winner], q, "power", winner)
                        if best is None or key > best[0]:
                            best = (key, q, "power", winner)
        if best is None:
            break
        _, q, action, index = best
        if action == "activate":
            bits[q][index] = 1
            powers[q][index] = 1
            used += 2
            observe(q, index)
        elif action == "bit":
            bits[q][index] += 1
            used += 1
            observe(q, index)
        else:
            powers[q][index] += 1
            used += 1
        steps_used += 1
        if steps_used >= max_steps:
            break
    refine_rounds = 0
    if refine:
        powers, bits, refine_rounds = nomp.maxmin_refine(
            scenario,
            powers,
            bits,
            max_power=power_cap,
            max_bits=MAX_BITS,
            max_rounds=max_refine_rounds,
            grid=GRID,
            flip_probability=flip_probability,
            success_probability=success_probability,
            floors=floors,
            weights=weights,
        )
        for q, target in enumerate(scenario):
            for r in range(reports):
                if bits[q][r] > 0:
                    observe(q, r)
    stopped_by_certificate = False
    feedback_rounds = 0
    while feedback_rounds < max_feedback_rounds:
        if certificate_status():
            stopped_by_certificate = True
            break
        for q, target in enumerate(scenario):
            active = [r for r in range(reports) if bits[q][r] > 0]
            if active:
                winner = active[int(np.argmax(ucb(q)[active]))]
                observe(q, winner)
            all_ucb = ucb(q).copy()
            for r in active:
                all_ucb[r] = -np.inf
            if np.isfinite(all_ucb).any():
                probe = int(np.argmax(all_ucb))
                observe(q, probe)
        feedback_rounds += 1
    worst_pd = min(
        nomp.target_scores(
            scenario,
            powers,
            bits,
            GRID,
            flip_probability,
            success_probability,
        )
    )
    qos_worst = None
    if floors is not None:
        qos_worst = float(np.min(nomp.qos_scores(
            np.asarray(nomp.target_scores(
                scenario,
                powers,
                bits,
                GRID,
                flip_probability,
                success_probability,
            )),
            floors,
            weights,
        )))
    return {
        "worst_pd": worst_pd,
        "qos_worst": qos_worst,
        "steps_used": steps_used,
        "stopped_by_certificate": stopped_by_certificate,
        "refine_rounds": refine_rounds,
        "feedback_rounds": feedback_rounds,
    }


def nomp_greedy_joint_multi(
    scenario, budget, *, max_rounds: int = 100
):
    """NOMP-inspired online greedy: min cover, WTA addition, leximin refine."""
    return float(nomp.nomp_wta_greedy_joint_multi(
        scenario, budget, max_rounds=max_rounds
    )["worst_pd"])


def run_comparison(args) -> dict:
    exact_mode = args.exact_mode
    if exact_mode == "auto":
        exact_mode = "full" if args.reports <= 2 else "wta"

    train_scenarios = [
        make_scenario(
            seed, args.reports, args.targets,
            heterogeneous=args.mode == "heterogeneous",
        )
        for seed in range(args.train_seeds)
    ]
    test_scenarios = [
        make_scenario(
            10000 + seed, args.reports, args.targets,
            heterogeneous=args.mode == "heterogeneous",
        )
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
        ppo_actor = train_mappo_ppo(
            train_scenarios,
            budget,
            args.episodes,
            args.reports,
            ppo_epochs=2,
            entropy_coef=0.05,
            learning_rate=1e-3,
        )
        mappo_ppo_worst = evaluate_mappo(
            ppo_actor, test_scenarios, budget, args.reports
        )
        mappo_nomp_worst = evaluate_mappo_nomp(
            actor, test_scenarios, budget, args.reports
        )
        mappo_probe_nomp_worst = evaluate_mappo_probe_nomp(
            actor, test_scenarios, budget, args.reports
        )
        mappo_adapter_nomp_worst = evaluate_mappo_adapter_nomp(
            actor, test_scenarios, budget, args.reports
        )
        mappo_bandit_adapter_nomp_worst = evaluate_mappo_bandit_adapter_nomp(
            actor, test_scenarios, budget, args.reports
        )
        train_seconds = time.perf_counter() - start
        greedy_worsts = []
        greedy_winner_init_worsts = []
        wta_greedy_worsts = []
        ucb_wta_greedy_worsts = []
        ucb_steps = []
        ucb_certificates = []
        ucb_feedback_rounds = []
        ucb_nomp_worsts = []
        ucb_nomp_steps = []
        ucb_nomp_certificates = []
        ucb_nomp_refine_rounds = []
        ucb_nomp_feedback_rounds = []
        nomp_greedy_worsts = []
        exact_worsts = []
        winner_worsts = []
        for scenario_index, scenario in enumerate(test_scenarios):
            greedy_worsts.append(greedy_joint_multi(scenario, budget))
            greedy_winner_init_worsts.append(greedy_joint_multi(
                scenario, budget, initialization="winner"
            ))
            wta_greedy_worsts.append(wta_greedy_joint_multi(scenario, budget))
            ucb_result = ucb_wta_greedy_joint_multi(
                scenario,
                budget,
                noise_scale=0.2,
                seed=scenario_index,
            )
            ucb_wta_greedy_worsts.append(ucb_result["worst_pd"])
            ucb_steps.append(ucb_result["steps_used"])
            ucb_certificates.append(
                ucb_result["stopped_by_certificate"]
            )
            ucb_feedback_rounds.append(ucb_result["feedback_rounds"])
            ucb_nomp_result = ucb_wta_greedy_joint_multi(
                scenario,
                budget,
                noise_scale=0.2,
                seed=scenario_index,
                min_cover=True,
                refine=True,
                max_refine_rounds=100,
            )
            ucb_nomp_worsts.append(ucb_nomp_result["worst_pd"])
            ucb_nomp_steps.append(ucb_nomp_result["steps_used"])
            ucb_nomp_certificates.append(
                ucb_nomp_result["stopped_by_certificate"]
            )
            ucb_nomp_refine_rounds.append(
                ucb_nomp_result["refine_rounds"]
            )
            ucb_nomp_feedback_rounds.append(
                ucb_nomp_result["feedback_rounds"]
            )
            nomp_greedy_worsts.append(nomp_greedy_joint_multi(
                scenario, budget
            ))
            winner_groups = [
                winner_take_all_proportional_options(
                    float(t[0]), t[1:],
                    bit_options=np.arange(MAX_BITS + 1, dtype=int),
                    budget=budget, grid=GRID,
                )
                for t in scenario
            ]
            winner_value = exact_joint_power_bit_maxmin(
                winner_groups, budget
            )
            if exact_mode == "full":
                full_groups = [
                    proportional_power_bit_options(
                        float(t[0]), t[1:],
                        power_levels=np.arange(budget + 1, dtype=float),
                        bit_options=np.arange(MAX_BITS + 1, dtype=int),
                        budget=budget, grid=GRID,
                    )
                    for t in scenario
                ]
                exact_worsts.append(exact_joint_power_bit_maxmin(
                    full_groups, budget
                ))
            else:
                exact_worsts.append(winner_value)
            winner_worsts.append(winner_value)
        summary.append({
            "budget": budget,
            "mappo_worst_mean": mappo_worst,
            "mappo_ppo_worst_mean": mappo_ppo_worst,
            "mappo_nomp_worst_mean": mappo_nomp_worst,
            "mappo_probe_nomp_worst_mean": mappo_probe_nomp_worst,
            "mappo_adapter_nomp_worst_mean": mappo_adapter_nomp_worst,
            "mappo_bandit_adapter_nomp_worst_mean": (
                mappo_bandit_adapter_nomp_worst
            ),
            "greedy_worst_mean": float(np.mean(greedy_worsts)),
            "greedy_winner_init_worst_mean": float(np.mean(
                greedy_winner_init_worsts
            )),
            "wta_greedy_worst_mean": float(np.mean(wta_greedy_worsts)),
            "ucb_wta_greedy_worst_mean": float(np.mean(
                ucb_wta_greedy_worsts
            )),
            "ucb_wta_mean_steps": float(np.mean(ucb_steps)),
            "ucb_wta_certificate_stop_rate": float(np.mean(
                ucb_certificates
            )),
            "ucb_wta_mean_feedback_rounds": float(np.mean(
                ucb_feedback_rounds
            )),
            "ucb_nomp_greedy_worst_mean": float(np.mean(
                ucb_nomp_worsts
            )),
            "ucb_nomp_mean_steps": float(np.mean(ucb_nomp_steps)),
            "ucb_nomp_certificate_stop_rate": float(np.mean(
                ucb_nomp_certificates
            )),
            "ucb_nomp_mean_refine_rounds": float(np.mean(
                ucb_nomp_refine_rounds
            )),
            "ucb_nomp_mean_feedback_rounds": float(np.mean(
                ucb_nomp_feedback_rounds
            )),
            "nomp_greedy_worst_mean": float(np.mean(
                nomp_greedy_worsts
            )),
            "exact_worst_mean": float(np.mean(exact_worsts)),
            "winner_worst_mean": float(np.mean(winner_worsts)),
            "train_seconds": train_seconds,
        })
    payload = {
        "gate": "joint-power-comparison",
        "mode": args.mode,
        "exact_mode": exact_mode,
        "targets": args.targets,
        "reports": args.reports,
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "summary": summary,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_power_comparison.json")
    parser.add_argument("--reports", type=int, default=2)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--test-seeds", type=int, default=20)
    parser.add_argument("--mode", choices=["homogeneous", "heterogeneous"], default="homogeneous")
    parser.add_argument(
        "--exact-mode",
        choices=["full", "wta", "auto"],
        default="auto",
        help="full enumerates power vectors; wta uses the closed-form frontier; auto switches at reports>2",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    payload = run_comparison(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
