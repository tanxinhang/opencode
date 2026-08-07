"""Exact soft/hard hybrid linear fusion.

For a target, soft reports form a moment-matched Gaussian score, while hard
reports contribute independent log-likelihood terms.  The combined score is

``T = w^T x_soft + sum_{i in H} log( p_{b_i,i} / q_{b_i,i} )``,

where `b_i in {0,1}` is the received hard decision and `p/q` are the
post-BSC decision probabilities under H1/H0.  The soft threshold is found by
binary search so that the total false-alarm probability equals the target.
The resulting P_D is exact over the enumerated hard-decision patterns.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np
from scipy.stats import norm

from .fusion import optimal_gaussian_weights
from .models import TargetEvidenceModel
from .sota_baselines import hard_decision_local_probabilities


def hybrid_gaussian_hard_pd(
    model: TargetEvidenceModel,
    soft_reports: Iterable[int],
    hard_reports: Iterable[int],
    false_alarm_rate: float,
    *,
    grid: int = 512,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Exact P_FA/P_D of soft-plus-hard LLR fusion for one target."""
    soft_reports = sorted(set(soft_reports))
    hard_reports = sorted(set(hard_reports) - set(soft_reports))
    if model.owner not in soft_reports:
        soft_reports = [model.owner, *soft_reports]
    if not soft_reports:
        raise ValueError("at least the owner soft report is required")
    weights = optimal_gaussian_weights(
        model.mu0, model.mu1, model.sigma0, model.sigma1,
        soft_reports, false_alarm_rate, grid=grid,
    )
    mean0 = float(weights @ model.mu0[soft_reports])
    mean1 = float(weights @ model.mu1[soft_reports])
    var0 = float(weights @ model.sigma0[np.ix_(soft_reports, soft_reports)] @ weights)
    var1 = float(weights @ model.sigma1[np.ix_(soft_reports, soft_reports)] @ weights)
    if min(var0, var1) <= 1e-14:
        raise FloatingPointError("degenerate hybrid score variance")
    per_hard = {
        uav: hard_decision_local_probabilities(model, uav, 0.1)
        for uav in hard_reports
    }
    patterns = []
    for bits in product((0, 1), repeat=len(hard_reports)):
        llr = 0.0
        p0_pattern = 1.0
        p1_pattern = 1.0
        for bit, uav in zip(bits, hard_reports):
            p0, p1 = per_hard[uav]
            llr += float(
                np.log(p1 / p0) if bit else np.log((1.0 - p1) / (1.0 - p0))
            )
            p0_pattern *= p0 if bit else 1.0 - p0
            p1_pattern *= p1 if bit else 1.0 - p1
        patterns.append((llr, p0_pattern, p1_pattern))

    def pfa_for_threshold(threshold: float) -> float:
        total = 0.0
        for llr, p0_pattern, _ in patterns:
            total += p0_pattern * norm.sf(
                (threshold - mean0 - llr) / np.sqrt(var0)
            )
        return total

    lo = mean0 - 20.0 * np.sqrt(var0)
    hi = mean0 + 20.0 * np.sqrt(var0)
    if pfa_for_threshold(lo) < false_alarm_rate:
        threshold = lo
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if pfa_for_threshold(mid) > false_alarm_rate:
                lo = mid
            else:
                hi = mid
        threshold = 0.5 * (lo + hi)
    pfa = pfa_for_threshold(threshold)
    pd = 0.0
    for llr, _, p1_pattern in patterns:
        pd += p1_pattern * norm.sf(
            (threshold - mean1 - llr) / np.sqrt(var1)
        )
    return {
        "pfa": pfa,
        "pd": pd,
        "threshold": threshold,
        "soft_reports": len(soft_reports),
        "hard_reports": len(hard_reports),
    }
