"""Adapter between MAPPO proposals and NOMP allocation requirements.

NOMP declares what input it needs (``probe_mask`` or ``proposal``) and the
adapter repeatedly translates MAPPO rollouts into that input, runs NOMP, and
keeps the best max-min schedule under a finite iteration cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import nomp_refinement as nomp


@dataclass(frozen=True)
class NompRequirement:
    """What NOMP needs from the MAPPO side."""

    modes: tuple[str, ...] | str = "auto"
    budget: int = 8
    max_bits: int = 2
    grid: int = 16
    max_refine_rounds: int = 100
    max_exact_reports: int = 8
    samples: int = 2048
    candidate_budget: int = 32


def _probe_mask_allocate(outputs, scenario, requirement):
    bits = outputs["bits"]
    mask = [
        np.asarray(bits[q] > 0, dtype=int)
        for q in range(len(scenario))
    ]
    if any(int(row.sum()) > 0 for row in mask):
        return nomp.nomp_wta_greedy_joint_multi(
            scenario,
            requirement.budget,
            max_bits=requirement.max_bits,
            grid=requirement.grid,
            max_rounds=requirement.max_refine_rounds,
            probe_mask=mask,
        )
    return nomp.nomp_wta_greedy_joint_multi(
        scenario,
        requirement.budget,
        max_bits=requirement.max_bits,
        grid=requirement.grid,
        max_rounds=requirement.max_refine_rounds,
    )


def _entropy_probe_allocate(outputs, scenario, requirement):
    logits = outputs["bit_logits"]
    probs = torch.softmax(logits, dim=-1).numpy()
    mask = [
        np.asarray(probs[q, :, 1:].sum(axis=1) >= 0.5, dtype=int)
        for q in range(len(scenario))
    ]
    if any(int(row.sum()) > 0 for row in mask):
        return nomp.nomp_wta_greedy_joint_multi(
            scenario,
            requirement.budget,
            max_bits=requirement.max_bits,
            grid=requirement.grid,
            max_rounds=requirement.max_refine_rounds,
            probe_mask=mask,
        )
    return nomp.nomp_wta_greedy_joint_multi(
        scenario,
        requirement.budget,
        max_bits=requirement.max_bits,
        grid=requirement.grid,
        max_rounds=requirement.max_refine_rounds,
    )


def _proposal_allocate(outputs, scenario, requirement):
    powers, bits = _feasible(
        scenario,
        outputs["powers"],
        outputs["bits"],
        requirement.budget,
        requirement.grid,
    )
    powers, bits, _ = nomp.maxmin_refine(
        scenario,
        powers,
        bits,
        max_power=requirement.budget,
        max_bits=requirement.max_bits,
        max_rounds=requirement.max_refine_rounds,
        grid=requirement.grid,
    )
    return {
        "worst_pd": float(min(nomp.target_scores(
            scenario,
            powers,
            bits,
            requirement.grid,
        ))),
    }


def _nomp_allocate(outputs, scenario, requirement):
    """Pure NOMP fallback: guarantees the hybrid is never below NOMP."""
    return nomp.nomp_wta_greedy_joint_multi(
        scenario,
        requirement.budget,
        max_bits=requirement.max_bits,
        grid=requirement.grid,
        max_rounds=requirement.max_refine_rounds,
    )


MODE_REGISTRY = {
    "nomp": _nomp_allocate,
    "probe_mask": _probe_mask_allocate,
    "entropy_probe": _entropy_probe_allocate,
    "proposal": _proposal_allocate,
}


def select_modes(scenario, budget):
    """Pick information modes from NOMP's current operating regime."""
    target_count = len(scenario)
    if budget <= 4 * target_count:
        return ("nomp", "probe_mask")
    if budget >= 8 * target_count:
        return ("nomp", "proposal")
    return ("nomp", "probe_mask", "entropy_probe", "proposal")


def ucb_index(mean, count, total_pulls, beta=1.0):
    """UCB index for a mode with bounded [0, 1] rewards."""
    if count <= 0.0:
        return np.inf
    return float(mean) + beta * np.sqrt(
        np.log(max(float(total_pulls), 1.0)) / count
    )


def build_state(owner, deltas, budget):
    """MAPPO state used during training (kept compatible with the actor)."""
    return np.concatenate((
        [float(owner) / 2.0],
        np.asarray(deltas, dtype=float) / 2.0,
        [float(budget) / 20.0],
    ))


def build_state_channel_aware(target, budget):
    """Channel-aware state including per-report flip and success."""
    owner, deltas, flips, successes = nomp.parse_target(target)
    return np.concatenate((
        [float(owner) / 2.0],
        np.asarray(deltas, dtype=float) / 2.0,
        [float(budget) / 20.0],
        np.asarray(flips, dtype=float) / 0.5,
        np.asarray(successes, dtype=float),
    ))


def _default_state_builder(target, budget):
    owner, deltas, _, _ = nomp.parse_target(target)
    return build_state(owner, deltas, budget)


def _feasible(scenario, powers, bits, budget, grid):
    """Drop power units until the proposal is budget feasible."""
    q_count = len(scenario)
    reports = len(powers[0])
    powers = [np.asarray(row, dtype=int).copy() for row in powers]
    bits = [np.asarray(row, dtype=int).copy() for row in bits]
    used = int(sum(
        powers[q].sum() + bits[q].sum() for q in range(q_count)
    ))

    def score(row_p, row_b):
        return float(np.min(nomp.target_scores(
            scenario, row_p, row_b, grid
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


class MappoNompAdapter:
    """Translate MAPPO rollouts into NOMP inputs and keep the best schedule."""

    def __init__(self, actor, state_builder=None):
        self.actor = actor
        self.state_builder = (
            state_builder or _default_state_builder
        )

    def propose_and_allocate(
        self,
        scenario,
        requirement: NompRequirement,
        *,
        seed: int = 0,
        sample: bool = True,
        iters: int = 3,
        state_scenario=None,
    ):
        state_scenario = scenario if state_scenario is None else state_scenario
        reports = nomp._report_count(scenario[0])
        if requirement.modes == "auto":
            modes = select_modes(scenario, requirement.budget)
        elif isinstance(requirement.modes, str):
            modes = (requirement.modes,)
        else:
            modes = requirement.modes
        best = None
        trace = []
        for _ in range(iters):
            states = np.stack([
                self.state_builder(t, requirement.budget)
                for t in state_scenario
            ])
            with torch.no_grad():
                logits_b, logits_p = self.actor(torch.as_tensor(
                    states, dtype=torch.float32
                ))
                outputs = {}
                if sample:
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
                else:
                    bits = torch.stack([
                        torch.argmax(logits_b[:, r, :], dim=1)
                        for r in range(reports)
                    ], dim=1).numpy()
                    powers = torch.stack([
                        torch.argmax(logits_p[:, r, :], dim=1)
                        for r in range(reports)
                    ], dim=1).numpy()
                outputs.update({
                    "bits": bits,
                    "powers": powers,
                    "bit_logits": logits_b,
                })
            for mode in modes:
                if mode not in MODE_REGISTRY:
                    raise ValueError(f"unknown mode {mode}")
                result = MODE_REGISTRY[mode](
                    outputs, scenario, requirement
                )
                if best is None or result["worst_pd"] > best["worst_pd"]:
                    best = result
                trace.append({
                    "mode": mode,
                    "worst_pd": float(result["worst_pd"]),
                })
        return {
            "worst_pd": float(best["worst_pd"]),
            "modes": list(modes),
            "trace": trace,
        }


class ModeBanditAdapter:
    """UCB over MAPPO information modes, with NOMP max-min as reward."""

    def __init__(self, actor, beta: float = 1.0, state_builder=None):
        self.actor = actor
        self.beta = beta
        self.state_builder = (
            state_builder or _default_state_builder
        )

    def propose_and_allocate(
        self,
        scenario,
        requirement: NompRequirement,
        *,
        seed: int = 0,
        sample: bool = True,
        iters: int = 5,
        state_scenario=None,
    ):
        state_scenario = scenario if state_scenario is None else state_scenario
        reports = nomp._report_count(scenario[0])
        if requirement.modes == "auto":
            modes = list(select_modes(scenario, requirement.budget))
        elif isinstance(requirement.modes, str):
            modes = [requirement.modes]
        else:
            modes = list(requirement.modes)
        means = {mode: 0.0 for mode in modes}
        counts = {mode: 0 for mode in modes}
        best = None
        best_mode = None
        for pull in range(1, iters + 1):
            chosen = None
            if pull <= len(modes):
                chosen = modes[pull - 1]
            else:
                scores = {
                    mode: ucb_index(
                        means[mode], counts[mode], pull, self.beta
                    )
                    for mode in modes
                }
                chosen = max(scores, key=lambda mode: scores[mode])
            states = np.stack([
                self.state_builder(t, requirement.budget)
                for t in state_scenario
            ])
            with torch.no_grad():
                logits_b, logits_p = self.actor(torch.as_tensor(
                    states, dtype=torch.float32
                ))
                if sample:
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
                else:
                    bits = torch.stack([
                        torch.argmax(logits_b[:, r, :], dim=1)
                        for r in range(reports)
                    ], dim=1).numpy()
                    powers = torch.stack([
                        torch.argmax(logits_p[:, r, :], dim=1)
                        for r in range(reports)
                    ], dim=1).numpy()
            outputs = {
                "bits": bits,
                "powers": powers,
                "bit_logits": logits_b,
            }
            result = MODE_REGISTRY[chosen](outputs, scenario, requirement)
            reward = float(result["worst_pd"])
            counts[chosen] += 1
            means[chosen] += (reward - means[chosen]) / counts[chosen]
            if best is None or reward > best["worst_pd"]:
                best = result
                best_mode = chosen
        return {
            "worst_pd": float(best["worst_pd"]),
            "best_mode": best_mode,
            "means": means,
            "counts": counts,
        }


class PriorityPolicy(nn.Module):
    """Small policy that chooses which target to prioritize each round."""

    def __init__(self, state_dim, q_count):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, q_count),
        )

    def forward(self, states):
        return self.net(states)


class PriorityNompAdapter:
    """MAPPO-guided priority middleware for weighted NOMP solving.

    Each round the policy picks the target that receives higher QoS weight;
    NOMP then solves the weighted max-min problem.  The middleware feeds the
    resulting unweighted worst P_D back as reward, so the policy learns which
    priority vector helps NOMP escape its own local optima.
    """

    def __init__(self, state_dim, state_builder=None, floor: float = 0.2):
        self.state_dim = state_dim
        self.state_builder = state_builder or _default_state_builder
        self.floor = floor

    def propose_and_allocate(
        self,
        scenario,
        requirement: NompRequirement,
        *,
        episodes: int = 8,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        q_count = len(scenario)
        policy = PriorityPolicy(self.state_dim, q_count)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        baseline = 0.0
        best = nomp.nomp_wta_greedy_joint_multi(
            scenario,
            requirement.budget,
            max_bits=requirement.max_bits,
            grid=requirement.grid,
            max_rounds=requirement.max_refine_rounds,
            max_exact_reports=requirement.max_exact_reports,
            samples=requirement.samples,
            candidate_budget=requirement.candidate_budget,
        )
        trace = []
        trace.append({
            "priority_target": None,
            "weights": None,
            "worst_pd": float(best["worst_pd"]),
        })
        floors = [self.floor] * q_count
        for _ in range(episodes):
            states = np.stack([
                self.state_builder(t, requirement.budget)
                for t in scenario
            ])
            logits = policy(torch.as_tensor(
                states, dtype=torch.float32
            )).mean(dim=0)
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())
            weights = [0.5] * q_count
            weights[action] = 2.0
            result = nomp.nomp_wta_greedy_joint_multi(
                scenario,
                requirement.budget,
                max_bits=requirement.max_bits,
                grid=requirement.grid,
                max_rounds=requirement.max_refine_rounds,
                max_exact_reports=requirement.max_exact_reports,
                samples=requirement.samples,
                candidate_budget=requirement.candidate_budget,
                floors=floors,
                weights=weights,
            )
            reward = float(result["worst_pd"])
            advantage = reward - baseline
            baseline = baseline + 0.2 * (reward - baseline)
            loss = -advantage * dist.log_prob(
                torch.as_tensor(action, dtype=torch.int64)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if best is None or reward > best["worst_pd"]:
                best = result
            trace.append({
                "priority_target": action,
                "weights": weights,
                "worst_pd": reward,
            })
        return {
            "worst_pd": float(best["worst_pd"]),
            "trace": trace,
        }
