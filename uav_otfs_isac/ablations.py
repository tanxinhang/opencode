from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

import numpy as np

from .expectation import expected_deflection
from .models import SelectionResult, TargetEvidenceModel


def diagonal_covariance_models(
    models: Sequence[TargetEvidenceModel],
) -> list[TargetEvidenceModel]:
    """Selection model that incorrectly assumes independent UAV evidence."""
    return [
        replace(model, sigma0=np.diag(np.diag(model.sigma0)), sigma1=np.diag(np.diag(model.sigma1)))
        for model in models
    ]


def deterministic_link_models(
    models: Sequence[TargetEvidenceModel],
) -> list[TargetEvidenceModel]:
    """Selection model that treats every scheduled report as received."""
    return [replace(model, success_prob=np.ones(model.num_uavs)) for model in models]


def evaluate_schedule_on_truth(
    truth_models: Sequence[TargetEvidenceModel],
    selection: SelectionResult,
    qos_min: Sequence[float],
    qos_weights: Sequence[float],
    *,
    mode: str = "exact",
    max_exact_reports: int = 14,
    rng: np.random.Generator | None = None,
) -> SelectionResult:
    """Re-score a schedule using the complete physical/statistical model."""
    rng = np.random.default_rng(0) if rng is None else rng
    quality = np.asarray([
        expected_deflection(
            model,
            selection.scheduled[q],
            mode=mode,
            max_exact_reports=max_exact_reports,
            rng=rng,
        )
        for q, model in enumerate(truth_models)
    ])
    qos = np.asarray(qos_min, dtype=float)
    weights = np.asarray(qos_weights, dtype=float)
    gap = float(np.sum(weights * np.maximum(qos - quality, 0.0) / np.maximum(qos, 1e-12)))
    return replace(selection, expected_deflection=quality, normalized_qos_gap=gap)
