from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np

from .fusion import optimal_deflection
from .models import TargetEvidenceModel


def expected_deflection_exact(model: TargetEvidenceModel, scheduled: Iterable[int]) -> float:
    reports = sorted(set(scheduled) - {model.owner})
    total = 0.0
    for pattern in product((0, 1), repeat=len(reports)):
        probability = 1.0
        received = {model.owner}
        for uav, success in zip(reports, pattern):
            p = float(model.success_prob[uav])
            probability *= p if success else 1.0 - p
            if success:
                received.add(uav)
        total += probability * optimal_deflection(model.delta, model.sigma0, received)
    return float(total)


def expected_deflection_saa(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    rng: np.random.Generator,
    samples: int = 2048,
) -> float:
    reports = sorted(set(scheduled) - {model.owner})
    values = np.empty(samples, dtype=float)
    for sample in range(samples):
        received = {model.owner}
        for uav in reports:
            if rng.random() < model.success_prob[uav]:
                received.add(uav)
        values[sample] = optimal_deflection(model.delta, model.sigma0, received)
    return float(values.mean())


def expected_deflection(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    *,
    mode: str = "exact",
    max_exact_reports: int = 14,
    rng: np.random.Generator | None = None,
    samples: int = 2048,
) -> float:
    reports = len(set(scheduled) - {model.owner})
    if mode == "exact" and reports <= max_exact_reports:
        return expected_deflection_exact(model, scheduled)
    if rng is None:
        rng = np.random.default_rng(0)
    return expected_deflection_saa(model, scheduled, rng, samples=samples)

