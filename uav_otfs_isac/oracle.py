from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from .expectation import expected_deflection
from .models import SelectionResult, TargetEvidenceModel


def exhaustive_oracle(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    qos_min: Sequence[float],
    qos_weights: Sequence[float],
    performance_weights: Sequence[float],
    *,
    max_candidates: int = 22,
    mode: str = "exact",
    max_exact_reports: int = 14,
    rng: np.random.Generator | None = None,
) -> SelectionResult:
    """Lexicographic exact oracle for small instances.

    First minimizes normalized QoS gap, then maximizes weighted expected
    deflection among solutions with that minimum gap.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    candidates = [
        (q, i, int(model.report_bits[i]))
        for q, model in enumerate(models)
        for i in range(model.num_uavs)
        if i != model.owner
    ]
    if len(candidates) > max_candidates:
        raise ValueError(f"oracle has {len(candidates)} candidates; max_candidates={max_candidates}")
    qos = np.asarray(qos_min, dtype=float); qos_w = np.asarray(qos_weights, dtype=float)
    perf_w = np.asarray(performance_weights, dtype=float)
    best_key = None; best = None
    for mask in range(1 << len(candidates)):
        used = 0; scheduled = [{m.owner} for m in models]
        feasible = True
        for idx, (q, i, cost) in enumerate(candidates):
            if mask & (1 << idx):
                used += cost
                if used > budget_bits:
                    feasible = False; break
                scheduled[q].add(i)
        if not feasible:
            continue
        quality = np.asarray([
            expected_deflection(m, scheduled[q], mode=mode, max_exact_reports=max_exact_reports, rng=rng)
            for q, m in enumerate(models)
        ])
        gap = float(np.sum(qos_w * np.maximum(qos - quality, 0.0) / np.maximum(qos, 1e-12)))
        utility = float(perf_w @ quality)
        key = (-gap, utility, -used)
        if best_key is None or key > best_key:
            best_key = key; best = (scheduled, quality, used, gap)
    assert best is not None
    scheduled, quality, used, gap = best
    return SelectionResult(tuple(frozenset(x) for x in scheduled), quality, used, gap, tuple())
