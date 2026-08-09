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
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.fusion import optimal_gaussian_detection_probability
from uav_otfs_isac.joint_allocation import model_from_bits
from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.power_split_theory import (
    power_gain_coefficient,
    proportional_power_bit_options,
    winner_take_all_proportional_options,
)


GRID = 16
MAX_POWER = 2
MAX_BITS = 2
POWER_OPTIONS = (0, 1, 2)
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


def greedy_joint_multi(scenario, budget, initialization="equal"):
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
                                if powers[qa][ra] <= 1 or powers[qb][rb] >= MAX_POWER:
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
    """Online winner-take-all greedy.

    Bits are added by marginal gain per resource unit.  Every power increment
    is given to the current best report of that target, so the power rule is
    winner-take-all at every step and no power-vector enumeration is used.
    """
    reports = len(scenario[0]) - 1
    powers = [np.zeros(reports, dtype=int) for _ in scenario]
    bits = [np.zeros(reports, dtype=int) for _ in scenario]
    used = 0

    def worst():
        return min(
            pd_value(float(t[0]), t[1:], powers[q], bits[q])
            for q, t in enumerate(scenario)
        )

    def scores():
        return np.asarray([
            pd_value(float(t[0]), t[1:], powers[q], bits[q])
            for q, t in enumerate(scenario)
        ])

    while True:
        current = float(np.mean(scores()))
        best = None
        for q, target in enumerate(scenario):
            active = [r for r in range(reports) if bits[q][r] > 0]
            # Activate a new report with 1 bit + 1 power.
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
                    key = (gain / 2.0, gain, q, "activate", r)
                    if best is None or key > best[0]:
                        best = (key, q, "activate", r)
            # Add one bit to an active report.
            for r in active:
                if bits[q][r] >= MAX_BITS or used + 1 > budget:
                    continue
                old_b = bits[q].copy()
                bits[q][r] += 1
                new_score = float(np.mean(scores()))
                gain = new_score - current
                bits[q] = old_b
                if gain > 0:
                    key = (gain, gain, q, "bit", r)
                    if best is None or key > best[0]:
                        best = (key, q, "bit", r)
            # Add one power to the current winner.
            if active:
                coefficients = np.asarray([
                    power_gain_coefficient(
                        float(target[r + 1]), int(bits[q][r]), 0.0, 1.0
                    )
                    for r in active
                ])
                winner = active[int(np.argmax(coefficients))]
                if powers[q][winner] < MAX_POWER and used + 1 <= budget:
                    old_p = powers[q].copy()
                    powers[q][winner] += 1
                    new_score = float(np.mean(scores()))
                    gain = new_score - current
                    powers[q] = old_p
                    if gain > 0:
                        key = (gain, gain, q, "power", winner)
                        if best is None or key > best[0]:
                            best = (key, q, "power", winner)
        if best is None:
            break
        _, q, action, index = best
        if action == "activate":
            bits[q][index] = 1
            powers[q][index] = 1
            used += 2
        elif action == "bit":
            bits[q][index] += 1
            used += 1
        else:
            powers[q][index] += 1
            used += 1
    return worst()


def ucb_wta_greedy_joint_multi(
    scenario, budget, *, noise_scale, seed, max_steps: int = 100
):
    """Online WTA-Greedy whose winner/activation use UCB error estimates."""
    reports = len(scenario[0]) - 1
    rng = np.random.default_rng(seed)
    powers = [np.zeros(reports, dtype=int) for _ in scenario]
    bits = [np.zeros(reports, dtype=int) for _ in scenario]
    means = []
    counts = []
    for q, target in enumerate(scenario):
        true = np.asarray([
            power_gain_coefficient(
                float(target[r + 1]), 1, 0.0, 1.0
            )
            for r in range(reports)
        ])
        means.append(
            true + noise_scale * rng.standard_normal(reports)
        )
        counts.append(np.ones(reports, dtype=float))
    used = 0
    beta = float(norm.ppf(0.975))

    def scores():
        return np.asarray([
            pd_value(float(t[0]), t[1:], powers[q], bits[q])
            for q, t in enumerate(scenario)
        ])

    def ucb(q):
        return means[q] + beta * noise_scale / np.sqrt(counts[q])

    def observe(q, r, delta, bit_count):
        observed = power_gain_coefficient(
            float(delta), int(bit_count), 0.0, 1.0
        )
        means[q][r] = (
            means[q][r] * counts[q][r] + observed
        ) / (counts[q][r] + 1.0)
        counts[q][r] += 1.0

    steps_used = 0
    stopped_by_certificate = False
    while True:
        certificate_ok = True
        active_targets = 0
        for q in range(len(scenario)):
            active = [r for r in range(reports) if bits[q][r] > 0]
            if not active:
                continue
            if len(active) < 2:
                certificate_ok = False
                break
            active_targets += 1
            values = ucb(q)[active]
            order = np.argsort(-values, kind="stable")
            best = active[order[0]]
            second = active[order[1]] if len(active) > 1 else None
            lcb_best = (
                means[q][best]
                - beta * noise_scale / np.sqrt(counts[q][best])
            )
            if second is None:
                ucb_second = -np.inf
            else:
                ucb_second = (
                    means[q][second]
                    + beta * noise_scale / np.sqrt(counts[q][second])
                )
            if lcb_best <= ucb_second:
                certificate_ok = False
                break
        if certificate_ok and active_targets == len(scenario):
            stopped_by_certificate = True
            break
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
                if powers[q][winner] < MAX_POWER and used + 1 <= budget:
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
            observe(q, index, scenario[q][index + 1], 1)
        elif action == "bit":
            bits[q][index] += 1
            used += 1
            observe(q, index, scenario[q][index + 1], bits[q][index])
        else:
            powers[q][index] += 1
            used += 1
        steps_used += 1
        if steps_used >= max_steps:
            break
    worst_pd = min(
        pd_value(float(t[0]), t[1:], powers[q], bits[q])
        for q, t in enumerate(scenario)
    )
    return {
        "worst_pd": worst_pd,
        "steps_used": steps_used,
        "stopped_by_certificate": stopped_by_certificate,
    }


def nomp_greedy_joint_multi(
    scenario, budget, *, max_steps: int = 100
):
    """NOMP-inspired online greedy with power refinement.

    Greedy selects bit/activation actions; after every action a refinement
    loop moves power from low-gain reports to the current winner when it
    improves the average P_D.  The residual is recomputed each round.
    """
    reports = len(scenario[0]) - 1
    powers = [np.zeros(reports, dtype=int) for _ in scenario]
    bits = [np.zeros(reports, dtype=int) for _ in scenario]
    used = 0

    def scores():
        return np.asarray([
            pd_value(float(t[0]), t[1:], powers[q], bits[q])
            for q, t in enumerate(scenario)
        ])

    def refine():
        improved = True
        while improved:
            improved = False
            for q, target in enumerate(scenario):
                active = [r for r in range(reports) if bits[q][r] > 0]
                if len(active) < 2:
                    continue
                coefficients = np.asarray([
                    power_gain_coefficient(
                        float(target[r + 1]), int(bits[q][r]), 0.0, 1.0
                    )
                    for r in active
                ])
                winner = active[int(np.argmax(coefficients))]
                before = float(np.mean(scores()))
                for source in active:
                    if source == winner or powers[q][source] <= 1:
                        continue
                    old_p = powers[q].copy()
                    powers[q][source] -= 1
                    powers[q][winner] += 1
                    after = float(np.mean(scores()))
                    if after > before + 1e-12:
                        improved = True
                        break
                    powers[q] = old_p
                if improved:
                    continue
                # Bit refinement: move one bit to the winner if it improves
                # the average P_D.
                for source in active:
                    if source == winner or bits[q][source] <= 1:
                        continue
                    if bits[q][winner] >= MAX_BITS:
                        continue
                    old_b = bits[q].copy()
                    bits[q][source] -= 1
                    bits[q][winner] += 1
                    after = float(np.mean(scores()))
                    if after > before + 1e-12:
                        improved = True
                        break
                    bits[q] = old_b

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
                    key = (gain / 2.0, gain, q, "activate", r)
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
                    key = (gain, gain, q, "bit", r)
                    if best is None or key > best[0]:
                        best = (key, q, "bit", r)
            if active:
                coefficients = np.asarray([
                    power_gain_coefficient(
                        float(target[r + 1]), int(bits[q][r]), 0.0, 1.0
                    )
                    for r in active
                ])
                winner = active[int(np.argmax(coefficients))]
                if powers[q][winner] < MAX_POWER and used + 1 <= budget:
                    old_p = powers[q].copy()
                    powers[q][winner] += 1
                    new_score = float(np.mean(scores()))
                    gain = new_score - current
                    powers[q] = old_p
                    if gain > 0:
                        key = (gain, gain, q, "power", winner)
                        if best is None or key > best[0]:
                            best = (key, q, "power", winner)
        if best is None:
            break
        _, q, action, index = best
        if action == "activate":
            bits[q][index] = 1
            powers[q][index] = 1
            used += 2
        elif action == "bit":
            bits[q][index] += 1
            used += 1
        else:
            powers[q][index] += 1
            used += 1
        refine()
        steps_used += 1
        if steps_used >= max_steps:
            break
    return min(
        pd_value(float(t[0]), t[1:], powers[q], bits[q])
        for q, t in enumerate(scenario)
    )


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
    args = parser.parse_args()

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
        train_seconds = time.perf_counter() - start
        greedy_worsts = []
        greedy_winner_init_worsts = []
        wta_greedy_worsts = []
        ucb_wta_greedy_worsts = []
        ucb_steps = []
        ucb_certificates = []
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
            nomp_greedy_worsts.append(nomp_greedy_joint_multi(
                scenario, budget
            ))
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
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
