from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .expectation import expected_deflection
from .models import SelectionResult, TargetEvidenceModel


def _result(
    models: Sequence[TargetEvidenceModel],
    scheduled: list[set[int]],
    qos_min: Sequence[float],
    qos_weights: Sequence[float],
    mode: str,
    max_exact_reports: int,
    rng: np.random.Generator,
) -> SelectionResult:
    quality = np.asarray([
        expected_deflection(m, scheduled[q], mode=mode, max_exact_reports=max_exact_reports, rng=rng)
        for q, m in enumerate(models)
    ])
    qos = np.asarray(qos_min, dtype=float); weights = np.asarray(qos_weights, dtype=float)
    gap = float(np.sum(weights * np.maximum(qos - quality, 0.0) / np.maximum(qos, 1e-12)))
    used = sum(int(models[q].report_bits[i]) for q, group in enumerate(scheduled) for i in group if i != models[q].owner)
    return SelectionResult(tuple(frozenset(x) for x in scheduled), quality, used, gap, tuple())


def no_cooperation(
    models: Sequence[TargetEvidenceModel], qos_min, qos_weights, *, mode="exact", max_exact_reports=14, rng=None
) -> SelectionResult:
    rng = np.random.default_rng(0) if rng is None else rng
    return _result(models, [{m.owner} for m in models], qos_min, qos_weights, mode, max_exact_reports, rng)


def ranked_baseline(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    qos_min,
    qos_weights,
    score: Callable[[TargetEvidenceModel, int], float],
    *,
    mode="exact",
    max_exact_reports=14,
    rng=None,
) -> SelectionResult:
    rng = np.random.default_rng(0) if rng is None else rng
    scheduled = [{m.owner} for m in models]
    candidates = []
    for q, model in enumerate(models):
        for i in range(model.num_uavs):
            if i != model.owner:
                candidates.append((float(score(model, i)), q, i, int(model.report_bits[i])))
    candidates.sort(reverse=True)
    used = 0
    for _, q, i, cost in candidates:
        if used + cost <= budget_bits:
            scheduled[q].add(i); used += cost
    return _result(models, scheduled, qos_min, qos_weights, mode, max_exact_reports, rng)


def sensing_quality_score(model: TargetEvidenceModel, i: int) -> float:
    return float(model.delta[i] ** 2 / model.sigma0[i, i])


def communication_score(model: TargetEvidenceModel, i: int) -> float:
    return float(model.success_prob[i] * (1.0 - model.bit_flip_prob[i]))


def independent_post_report_score(model: TargetEvidenceModel, i: int) -> float:
    return float(model.success_prob[i] * model.delta[i] ** 2 / model.sigma0[i, i])


def random_score_factory(rng: np.random.Generator):
    scores: dict[tuple[int, int], float] = {}
    def score(model: TargetEvidenceModel, i: int) -> float:
        key = (model.target_id, i)
        if key not in scores:
            scores[key] = float(rng.random())
        return scores[key]
    return score


def all_scheduled(
    models: Sequence[TargetEvidenceModel], qos_min, qos_weights, *, mode="exact", max_exact_reports=14, rng=None
) -> SelectionResult:
    rng = np.random.default_rng(0) if rng is None else rng
    return _result(models, [set(range(m.num_uavs)) for m in models], qos_min, qos_weights, mode, max_exact_reports, rng)

