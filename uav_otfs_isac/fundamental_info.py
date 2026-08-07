"""Unified information-budget abstraction for ISAC fusion.

The central idea is that system-level detection is governed by an abstract
information budget:

``J = sensing information + communication information - erasure/overhead loss``.

For moment-matched Gaussian evidence, local sensing information is measured
by deflection; soft reporting preserves a fraction of it; 1-bit hard reports
preserve the KL divergence between their H0/H1 decision laws; and consensus
uses all local decisions without consuming report bits.  The same budget
contains the RIS aperture gain through the evidence SNR.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.stats import norm

from .fusion import optimal_deflection
from .models import TargetEvidenceModel
from .sota_baselines import hard_decision_local_probabilities


def full_info_deflection(models: Sequence[TargetEvidenceModel]) -> np.ndarray:
    """Deflection using all UAV evidence (no reporting loss)."""
    return np.asarray([
        optimal_deflection(model.delta, model.sigma0, set(range(model.num_uavs)))
        for model in models
    ])


def schedule_deflection(
    models: Sequence[TargetEvidenceModel],
    scheduled: Sequence[Iterable[int]],
) -> np.ndarray:
    """Deflection of a concrete report schedule after reporting loss."""
    return np.asarray([
        optimal_deflection(
            model.delta, model.sigma0, scheduled[q]
        )
        for q, model in enumerate(models)
    ])


def hard_kl_information(
    model: TargetEvidenceModel,
    uav: int,
    local_false_alarm_rate: float = 0.1,
) -> float:
    """KL divergence between H0/H1 1-bit decision laws."""
    p0, p1 = hard_decision_local_probabilities(
        model, uav, local_false_alarm_rate
    )
    p0 = float(np.clip(p0, 1e-12, 1.0 - 1e-12))
    p1 = float(np.clip(p1, 1e-12, 1.0 - 1e-12))
    return float(
        p1 * np.log(p1 / p0)
        + (1.0 - p1) * np.log((1.0 - p1) / (1.0 - p0))
    )


def hard_consensus_information(
    models: Sequence[TargetEvidenceModel],
    local_false_alarm_rate: float = 0.1,
) -> np.ndarray:
    """Total KL information available to peer consensus per target."""
    return np.asarray([
        sum(
            hard_kl_information(model, uav, local_false_alarm_rate)
            for uav in range(model.num_uavs)
            if uav != model.owner
        )
        for model in models
    ])


def effective_deflection(
    pd: float,
    false_alarm_rate: float,
    variance_ratio: float = 1.0,
) -> float:
    """Effective deflection that reproduces an observed Gaussian P_D.

    Under ``Sigma1 = variance_ratio * Sigma0`` the detection probability is
    ``P_D = Phi((sqrt(D) - z_FA) / sqrt(c))``, so inverting the strictly
    monotone Gaussian CDF gives
    ``D_eff = (sqrt(c) * Phi^{-1}(P_D) + z_FA)^2``.
    """
    pd = float(np.clip(pd, 1e-9, 1.0 - 1e-9))
    z = norm.ppf(1.0 - false_alarm_rate)
    return float((norm.ppf(pd) * np.sqrt(variance_ratio) + z) ** 2)
