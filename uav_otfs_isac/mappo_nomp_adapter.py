"""Adapter between MAPPO proposals and NOMP allocation requirements.

NOMP declares what input it needs (``probe_mask`` or ``proposal``) and the
adapter repeatedly translates MAPPO rollouts into that input, runs NOMP, and
keeps the best max-min schedule under a finite iteration cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import nomp_refinement as nomp


@dataclass(frozen=True)
class NompRequirement:
    """What NOMP needs from the MAPPO side."""

    modes: tuple[str, ...] | str = "auto"
    budget: int = 8
    max_bits: int = 2
    grid: int = 16
    max_refine_rounds: int = 100


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


MODE_REGISTRY = {
    "probe_mask": _probe_mask_allocate,
    "entropy_probe": _entropy_probe_allocate,
    "proposal": _proposal_allocate,
}


def select_modes(scenario, budget):
    """Pick information modes from NOMP's current operating regime."""
    target_count = len(scenario)
    if budget <= 4 * target_count:
        return ("probe_mask",)
    if budget >= 8 * target_count:
        return ("proposal",)
    return ("probe_mask", "entropy_probe", "proposal")


def build_state(owner, deltas, budget):
    """MAPPO state used during training (kept compatible with the actor)."""
    return np.concatenate((
        [float(owner) / 2.0],
        np.asarray(deltas, dtype=float) / 2.0,
        [float(budget) / 20.0],
    ))


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

    def __init__(self, actor):
        self.actor = actor

    def propose_and_allocate(
        self,
        scenario,
        requirement: NompRequirement,
        *,
        seed: int = 0,
        sample: bool = True,
        iters: int = 3,
    ):
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
                build_state(float(t[0]), t[1:], requirement.budget)
                for t in scenario
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
