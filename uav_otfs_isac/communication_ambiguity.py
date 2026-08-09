"""Communication-parameter ambiguity and endpoint reduction.

For a fixed schedule, if the violation probability is nondecreasing in the
BSC flip probability and nonincreasing in the link success probability, then
the worst case over a rectangular ambiguity set is attained at
``(flip_hi, success_lo)``.  The BSC cascade ordering and erasure stochastic
dominance provide the exact-LRT basis; this module verifies the moment-model
monotonicity on a grid and exposes the endpoint scenario for robust DP.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .joint_allocation import model_from_bits
from .models import TargetEvidenceModel
from .risk import gaussian_pd_loss_distribution


def build_endpoint_models(
    owner_delta: float,
    deltas: np.ndarray,
    bits: np.ndarray,
    flip_interval: tuple[float, float],
    success_interval: tuple[float, float],
) -> list[TargetEvidenceModel]:
    """Four corner models of the communication ambiguity rectangle."""
    flip_lo, flip_hi = flip_interval
    success_lo, success_hi = success_interval
    full_deltas = np.concatenate(([float(owner_delta)], np.asarray(deltas)))
    full_bits = np.concatenate(([0], np.asarray(bits, dtype=int)))
    models = []
    for flip in (flip_lo, flip_hi):
        base = model_from_bits(
            full_deltas, full_bits, bit_flip_probability=flip
        )
        for success in (success_hi, success_lo):
            success_vector = np.array(
                [1.0] + [float(success)] * deltas.size
            )
            models.append(replace(base, success_prob=success_vector))
    return models


def verify_endpoint_dominance(
    owner_delta: float,
    deltas: np.ndarray,
    bits: np.ndarray,
    flip_interval: tuple[float, float],
    success_interval: tuple[float, float],
    *,
    scheduled: set[int],
    minimum_pd: float,
    false_alarm_rate: float = 0.05,
    grid: int = 32,
    p_steps: int = 5,
    s_steps: int = 5,
) -> dict:
    """Check that the endpoint (flip_hi, success_lo) dominates the grid."""
    deltas = np.asarray(deltas, dtype=float)
    bits = np.asarray(bits, dtype=int)
    endpoints = build_endpoint_models(
        owner_delta, deltas, bits, flip_interval, success_interval
    )
    endpoint_violation = max(
        gaussian_pd_loss_distribution(
            model, scheduled, minimum_pd, false_alarm_rate
        ).violation_probability()
        for model in endpoints
    )
    worst = endpoint_violation
    worst_at = None
    flip_lo, flip_hi = flip_interval
    success_lo, success_hi = success_interval
    for flip in np.linspace(flip_lo, flip_hi, p_steps):
        for success in np.linspace(success_lo, success_hi, s_steps):
            model = build_endpoint_models(
                owner_delta, deltas, bits,
                (float(flip), float(flip)),
                (float(success), float(success)),
            )[0]
            violation = gaussian_pd_loss_distribution(
                model, scheduled, minimum_pd, false_alarm_rate
            ).violation_probability()
            if violation > worst + 1e-12:
                worst = violation
                worst_at = (float(flip), float(success))
    return {
        "endpoint_violation": float(endpoint_violation),
        "grid_worst_violation": float(worst),
        "grid_worst_at": worst_at,
        "passed": worst <= endpoint_violation + 1e-12,
    }


def build_endpoint_scenario_groups(
    targets,
    flip_interval: tuple[float, float],
    success_interval: tuple[float, float],
) -> tuple[list[list[TargetEvidenceModel]], list[list[TargetEvidenceModel]]]:
    """Return (four-corner groups, endpoint-reduced groups) per target.

    The endpoint-reduced group contains only the ``(flip_hi, success_lo)``
    model, which is the worst corner under the monotonicity closure.
    """
    full_groups = []
    reduced_groups = []
    for owner_delta, deltas, bits in targets:
        models = build_endpoint_models(
            owner_delta, deltas, bits, flip_interval, success_interval
        )
        full_groups.append(models)
        reduced_groups.append([models[-1]])
    return full_groups, reduced_groups
