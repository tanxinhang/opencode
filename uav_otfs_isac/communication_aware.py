"""Communication-aware sensing score and its optimality certificate.

For a diagonal, proportional-covariance target model with independent
erasures and equal report costs, the expected received deflection of a
scheduled set is

``J_i = s_i * delta_i^2 / sigma0_ii``.

Therefore the subset with the largest ``J_i`` maximizes the expected received
deflection.  In the concave region of the P_D-deflection map, this also
maximizes the upper-bound surrogate ``P_D(E[deflection])``.  Exact P_D
optimality with heterogeneous erasure survival may still require DP, which
is why this score is presented as a certificate-optimal surrogate, not as a
claim of exact P_D optimality.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import TargetEvidenceModel


def communication_aware_sensing_score(
    model: TargetEvidenceModel,
    uav: int,
) -> float:
    """Per-report communication-aware sensing score."""
    return float(
        model.success_prob[uav]
        * model.delta[uav] ** 2
        / max(model.sigma0[uav, uav], 1e-12)
    )


def expected_received_deflection(
    model: TargetEvidenceModel,
    scheduled: Sequence[int],
) -> float:
    """Expected received deflection over independent erasures."""
    total = 0.0
    for uav in scheduled:
        if uav == model.owner:
            continue
        total += communication_aware_sensing_score(model, uav)
    return total


def communication_aware_top_k(
    model: TargetEvidenceModel,
    budget_bits: int,
) -> frozenset[int]:
    """Select reports by the communication-aware score under equal costs."""
    candidates = [
        i for i in range(model.num_uavs)
        if i != model.owner and int(model.report_bits[i]) > 0
    ]
    candidates.sort(
        key=lambda i: communication_aware_sensing_score(model, i),
        reverse=True,
    )
    selected = {model.owner}
    used = 0
    for i in candidates:
        cost = int(model.report_bits[i])
        if used + cost > budget_bits:
            continue
        selected.add(i)
        used += cost
    return frozenset(selected)
