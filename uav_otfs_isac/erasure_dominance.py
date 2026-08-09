"""Erasure stochastic dominance and expected-P_D closure.

If link success probabilities satisfy ``p_a >= p_b`` componentwise, the
``p_b`` reception process can be coupled to the ``p_a`` process by first
receiving with ``p_a`` and then dropping every received report independently
with probability ``1 - p_b / p_a``.  In that coupling the ``p_b`` received
set is always a subset of the ``p_a`` received set.  Consequently every
decision rule on the degraded output is a randomized decision rule on the
clean output, and, at operating points where P_D is set-monotone, the
expected P_D is nonincreasing as erasure grows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .expected_pd import expected_gaussian_detection_probability
from .models import TargetEvidenceModel


def erasure_cascade_factor(high_success: float, low_success: float) -> float:
    """Return the drop factor r with low = high * r."""
    high = float(high_success)
    low = float(low_success)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("success probabilities must satisfy 0 <= low <= high <= 1")
    if high <= 0.0:
        return 0.0
    return float(low / high)


def verify_monotone_coupling(
    high_success: np.ndarray,
    low_success: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Check the coupled event ``received_low subset received_high``."""
    high = np.asarray(high_success, dtype=float)
    low = np.asarray(low_success, dtype=float)
    if high.shape != low.shape:
        raise ValueError("success probability vectors must have equal shape")
    if np.any((high < 0.0) | (high > 1.0)) or np.any(
        (low < 0.0) | (low > 1.0)
    ):
        raise ValueError("success probabilities must lie in [0, 1]")
    if np.any(low > high + 1e-12):
        raise ValueError("low_success must be componentwise no larger than high_success")
    rng = np.random.default_rng(seed)
    uniforms = rng.random((samples, high.size))
    high_received = uniforms < high[None, :]
    low_received = uniforms < low[None, :]
    violations = int(np.any(low_received & ~high_received, axis=1).sum())
    return {
        "samples": samples,
        "coupling_violations": violations,
        "passed": violations == 0,
    }


def expected_pd_at_success(
    model: TargetEvidenceModel,
    scheduled: Sequence[int],
    false_alarm_rate: float,
    grid: int,
) -> float:
    return float(expected_gaussian_detection_probability(
        model,
        scheduled,
        false_alarm_rate,
        pd_mode="optimal",
        grid=grid,
    ))


def verify_expected_pd_monotonicity(
    clean_model: TargetEvidenceModel,
    degraded_model: TargetEvidenceModel,
    *,
    false_alarm_rate: float = 0.05,
    grid: int = 256,
) -> dict:
    """Check that a cleaner erasure law never reduces expected P_D."""
    if clean_model.num_uavs != degraded_model.num_uavs:
        raise ValueError("models must share the UAV count")
    if clean_model.owner != degraded_model.owner:
        raise ValueError("models must share the owner")
    scheduled = frozenset(range(clean_model.num_uavs))
    clean_pd = expected_pd_at_success(
        clean_model, scheduled, false_alarm_rate, grid
    )
    degraded_pd = expected_pd_at_success(
        degraded_model, scheduled, false_alarm_rate, grid
    )
    return {
        "clean_expected_pd": clean_pd,
        "degraded_expected_pd": degraded_pd,
        "gap": clean_pd - degraded_pd,
        "in_theorem_scope": bool(
            clean_pd >= 0.5 - 1e-9 and degraded_pd >= 0.5 - 1e-9
        ),
        "passed": clean_pd >= degraded_pd - 1e-9,
    }
