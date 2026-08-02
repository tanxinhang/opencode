from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .expectation import expected_deflection
from .fusion import conditional_marginal_deflection
from .models import SelectionResult, TargetEvidenceModel


def _quality(
    models: Sequence[TargetEvidenceModel],
    scheduled: list[set[int]],
    mode: str,
    max_exact_reports: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray([
        expected_deflection(
            model,
            scheduled[q],
            mode=mode,
            max_exact_reports=max_exact_reports,
            rng=rng,
        )
        for q, model in enumerate(models)
    ])


def _normalized_gap(quality: np.ndarray, qos: np.ndarray, weights: np.ndarray) -> float:
    denominator = np.maximum(qos, 1e-12)
    return float(np.sum(weights * np.maximum(qos - quality, 0.0) / denominator))


def greedy_select(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    qos_min: Sequence[float],
    qos_weights: Sequence[float],
    performance_weights: Sequence[float],
    *,
    mode: str = "exact",
    max_exact_reports: int = 14,
    rng: np.random.Generator | None = None,
    gain_mode: str = "exact",
    qos_first: bool = True,
) -> SelectionResult:
    """Two-stage greedy approximation to revised equations (20a)-(21).

    Stage 1 minimizes normalized QoS shortfall. Stage 2 maximizes expected
    deflection without worsening the achieved minimum shortfall.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if gain_mode not in {"exact", "first_order"}:
        raise ValueError("gain_mode must be 'exact' or 'first_order'")
    q_count = len(models)
    qos = np.asarray(qos_min, dtype=float)
    qos_w = np.asarray(qos_weights, dtype=float)
    perf_w = np.asarray(performance_weights, dtype=float)
    scheduled = [{m.owner} for m in models]
    used = 0
    trace: list[dict[str, float | int | str]] = []
    quality = _quality(models, scheduled, mode, max_exact_reports, rng)

    def candidates(stage: str):
        current_gap = _normalized_gap(quality, qos, qos_w)
        for q, model in enumerate(models):
            if stage == "qos" and quality[q] >= qos[q] - 1e-12:
                continue
            for k in range(model.num_uavs):
                if k in scheduled[q] or k == model.owner:
                    continue
                cost = int(model.report_bits[k])
                if used + cost > budget_bits:
                    continue
                trial = set(scheduled[q]); trial.add(k)
                trial_quality = expected_deflection(
                    model, trial, mode=mode, max_exact_reports=max_exact_reports, rng=rng
                )
                true_gain = max(trial_quality - quality[q], 0.0)
                conditional_gain = conditional_marginal_deflection(
                    model.delta, model.sigma0, scheduled[q], k
                )
                first_order = float(model.success_prob[k]) * conditional_gain
                ranking_gain = true_gain if gain_mode == "exact" else first_order
                if stage == "qos":
                    trial_all = quality.copy()
                    trial_all[q] = quality[q] + ranking_gain
                    gap_reduction = current_gap - _normalized_gap(trial_all, qos, qos_w)
                    score = gap_reduction / max(cost, 1)
                else:
                    score = perf_w[q] * ranking_gain / max(cost, 1)
                yield score, true_gain, first_order, q, k, cost, trial_quality

    stages = ("qos", "performance") if qos_first else ("performance",)
    for stage in stages:
        while True:
            options = list(candidates(stage))
            if not options:
                break
            options.sort(key=lambda x: (x[0], x[1], -x[5]), reverse=True)
            score, true_gain, first_order, q, k, cost, trial_quality = options[0]
            if score <= 1e-14:
                break
            scheduled[q].add(k)
            used += cost
            quality[q] = trial_quality
            trace.append({
                "stage": stage,
                "target": q,
                "uav": k,
                "cost_bits": cost,
                "score": float(score),
                "expected_gain": float(true_gain),
                "first_order_gain": float(first_order),
                "expected_deflection": float(trial_quality),
            })

    return SelectionResult(
        scheduled=tuple(frozenset(x) for x in scheduled),
        expected_deflection=quality,
        used_bits=used,
        normalized_qos_gap=_normalized_gap(quality, qos, qos_w),
        trace=tuple(trace),
    )
