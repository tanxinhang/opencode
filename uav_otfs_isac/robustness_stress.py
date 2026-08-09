"""Progressive stress profiles and survival envelopes.

The module is deliberately small so that robustness axes can be added one at
a time.  Every :class:`StressProfile` applies degradations on top of the same
seed-resolved scenario; the survival envelope then reports worst-target
expected ``P_D`` under the budgeted greedy selector.  New axes should first
get their own focused tests, then be enabled in the envelope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np

from .config import ExperimentConfig
from .expected_pd import expected_pd_greedy_select
from .models import TargetEvidenceModel
from .scenario import build_models, target_geometry, uav_geometry


@dataclass(frozen=True)
class StressProfile:
    """Named combination of independent degradation axes.

    ``bit_flip_probability=None`` keeps the scenario-native per-link flip
    profile; otherwise every reporting link is set to the supplied value.
    The owner link is never degraded.  ``mobility_std`` perturbs each target
    position with a seed-split Gaussian offset, leaving the scenario RNG for
    the rest of the model unchanged.
    """

    label: str
    interference_to_noise: float = 0.0
    interference_sources: tuple[tuple[float, float, float], ...] = ()
    interference_reference_inr: tuple[float, ...] = ()
    bit_flip_probability: float | None = None
    success_probability_scale: float = 1.0
    mobility_std: float = 0.0
    max_displacement_std: float = 3.0
    velocity_limit_mps: float = 0.0
    frame_duration_s: float = 1.0


def inr_from_sources(
    sources: Sequence[Sequence[float]],
    reference_inr: Sequence[float],
    transmitter_positions: np.ndarray,
    reference_distance: float = 100.0,
) -> np.ndarray:
    """Free-space path-loss INR from independent interference sources."""
    if len(sources) != len(reference_inr):
        raise ValueError("one reference INR is required per interference source")
    positions = np.asarray(transmitter_positions, dtype=float)
    total = np.zeros(positions.shape[0], dtype=float)
    for source, inr_ref in zip(sources, reference_inr):
        distance = np.linalg.norm(
            positions - np.asarray(source, dtype=float), axis=1
        )
        total += float(inr_ref) * (
            reference_distance / np.maximum(distance, 1e-9)
        ) ** 2
    return total


def stress_target_positions(
    cfg: ExperimentConfig,
    seed: int,
    profile: StressProfile,
) -> list[np.ndarray]:
    """Bounded isotropic target displacement for a single stress frame."""
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    if profile.mobility_std <= 0.0:
        return targets
    if profile.max_displacement_std <= 0.0:
        raise ValueError("max_displacement_std must be positive")
    rng = np.random.default_rng(seed + 100_000)
    if profile.velocity_limit_mps > 0.0:
        if profile.frame_duration_s <= 0.0:
            raise ValueError("frame_duration_s must be positive")
        max_displacement = profile.velocity_limit_mps * profile.frame_duration_s
    else:
        max_displacement = profile.mobility_std * profile.max_displacement_std
    result = []
    for base in targets:
        direction = rng.standard_normal(3)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            result.append(np.asarray(base, dtype=float))
            continue
        radius = float(np.minimum(
            profile.mobility_std * norm,
            max_displacement,
        ))
        result.append(
            np.asarray(base, dtype=float)
            + (direction / norm) * radius
        )
    return result


def build_stress_models(
    cfg: ExperimentConfig,
    seed: int,
    profile: StressProfile,
) -> list[TargetEvidenceModel]:
    """Build moment-matched models with the requested stress applied."""
    rng = np.random.default_rng(seed)
    targets = stress_target_positions(cfg, seed, profile)
    effective_cfg = cfg
    if profile.bit_flip_probability is not None:
        flip = float(profile.bit_flip_probability)
        effective_cfg = replace(
            cfg,
            reporting=replace(
                cfg.reporting,
                bit_flip_probability_range=(flip, flip),
            ),
        )
    if profile.interference_sources:
        interference = inr_from_sources(
            profile.interference_sources,
            profile.interference_reference_inr,
            uav_geometry(cfg.num_uavs),
        )
    else:
        interference = np.full(
            cfg.num_uavs, profile.interference_to_noise, dtype=float
        )
    models = build_models(
        effective_cfg,
        rng,
        target_positions=targets,
        interference_to_noise=interference,
    )
    return [_apply_channel_stress(model, profile) for model in models]


def _apply_channel_stress(
    model: TargetEvidenceModel,
    profile: StressProfile,
) -> TargetEvidenceModel:
    success = np.clip(
        model.success_prob * profile.success_probability_scale, 0.0, 1.0
    )
    success[model.owner] = 1.0
    flip = model.bit_flip_prob.copy()
    if profile.bit_flip_probability is not None:
        flip = np.full(model.num_uavs, profile.bit_flip_probability)
    flip[model.owner] = 0.0
    stressed = replace(model, success_prob=success, bit_flip_prob=flip)
    stressed.validate()
    return stressed


def evaluate_stress_profile(
    cfg: ExperimentConfig,
    seed: int,
    profile: StressProfile,
    *,
    budget_bits: int,
    false_alarm_rate: float = 0.05,
    grid: int = 64,
    models: Sequence[TargetEvidenceModel] | None = None,
) -> dict:
    """Run the budgeted selector on one stress cell and report the envelope."""
    if models is None:
        models = build_stress_models(cfg, seed, profile)
    selection = expected_pd_greedy_select(
        models,
        budget_bits,
        false_alarm_rate,
        grid=grid,
    )
    return {
        "seed": seed,
        "label": profile.label,
        "worst_pd": float(np.min(selection.expected_pd)),
        "mean_pd": float(np.mean(selection.expected_pd)),
        "used_bits": int(selection.used_bits),
        "scheduled": [
            sorted(group) for group in selection.scheduled
        ],
    }


def survival_envelope(
    cfg: ExperimentConfig,
    profiles: Sequence[StressProfile],
    seeds: int,
    *,
    budget_bits: int,
    false_alarm_rate: float = 0.05,
    grid: int = 64,
    qos_target: float | None = None,
) -> list[dict]:
    """Return per-profile mean/min worst-``P_D`` over the seed block."""
    rows = []
    for profile in profiles:
        cells = [
            evaluate_stress_profile(
                cfg,
                cfg.seed + offset,
                profile,
                budget_bits=budget_bits,
                false_alarm_rate=false_alarm_rate,
                grid=grid,
            )
            for offset in range(seeds)
        ]
        worst = np.asarray([cell["worst_pd"] for cell in cells])
        row = {
            "label": profile.label,
            "profile": asdict(profile),
            "worst_pd_mean": float(np.mean(worst)),
            "worst_pd_min": float(np.min(worst)),
            "qos_rate": (
                float(np.mean(worst >= qos_target - 1e-9))
                if qos_target is not None else None
            ),
            "cells": cells,
        }
        rows.append(row)
    return rows
