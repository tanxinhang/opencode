"""Discrete coordinate-ascent refinement for report scheduling.

The greedy selector is a forward-only construction.  This module adds a
discrete analogue of gradient ascent: starting from any feasible schedule,
it repeatedly evaluates the best single-report add, remove, or swap move
that preserves the bit budget and improves the expected-P_D objective.  This
catches budget reallocation and redundancy removal that forward greedy
cannot express.  The result is a local optimum with respect to all
single-report moves; no universal approximation ratio is claimed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .expected_pd import expected_gaussian_detection_probability
from .models import ExpectedPdSelectionResult, TargetEvidenceModel


def discrete_gradient_select(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    false_alarm_rate: float,
    *,
    init_schedule: Sequence[Sequence[int]] | None = None,
    qos_pd: Sequence[float] | None = None,
    qos_weights: Sequence[float] | None = None,
    performance_weights: Sequence[float] | None = None,
    pd_mode: str = "optimal",
    max_exact_reports: int = 14,
    rng: np.random.Generator | None = None,
    samples: int = 2048,
    grid: int = 512,
    max_rounds: int = 12,
) -> ExpectedPdSelectionResult:
    """Coordinate ascent over add/remove/swap moves from an initial schedule."""
    if budget_bits < 0:
        raise ValueError("budget_bits must be nonnegative")
    if pd_mode not in {"optimal", "deflection"}:
        raise ValueError("pd_mode must be 'optimal' or 'deflection'")
    count = len(models)
    qos = np.zeros(count, dtype=float) if qos_pd is None else np.asarray(
        qos_pd, dtype=float
    )
    qos_w = np.ones(count, dtype=float) if qos_weights is None else np.asarray(
        qos_weights, dtype=float
    )
    perf_w = (
        np.ones(count, dtype=float)
        if performance_weights is None
        else np.asarray(performance_weights, dtype=float)
    )
    if qos.shape != (count,) or qos_w.shape != (count,) or perf_w.shape != (count,):
        raise ValueError("per-target arrays must have one entry per target")
    if init_schedule is None:
        scheduled = [{model.owner} for model in models]
    else:
        scheduled = [set(group) for group in init_schedule]
        if len(scheduled) != count:
            raise ValueError("init_schedule must have one group per model")
        for q, model in enumerate(models):
            if model.owner not in scheduled[q]:
                raise ValueError("owner report must be present in every group")
            if any(i < 0 or i >= model.num_uavs for i in scheduled[q]):
                raise ValueError("init_schedule contains an invalid report index")
    used = 0
    for q, model in enumerate(models):
        used += sum(
            int(model.report_bits[i]) for i in scheduled[q] if i != model.owner
        )
    if used > budget_bits:
        raise ValueError("init_schedule already exceeds budget_bits")

    cache: dict[tuple[int, frozenset[int]], float] = {}

    def expected(q: int, selected: frozenset[int]) -> float:
        key = (q, selected)
        if key not in cache:
            cache[key] = expected_gaussian_detection_probability(
                models[q], selected, false_alarm_rate, pd_mode=pd_mode,
                max_exact_reports=max_exact_reports, rng=rng, samples=samples,
                grid=grid,
            )
        return cache[key]

    quality = [
        expected(q, frozenset(scheduled[q])) for q in range(count)
    ]

    def normalized_gap(values: Sequence[float]) -> float:
        return float(np.sum(
            qos_w * np.maximum(qos - np.asarray(values), 0.0)
            / np.maximum(qos, 1e-12)
        ))

    trace: list[dict[str, float | int | str]] = []
    rounds = 0
    while rounds < max_rounds:
        current_gap = normalized_gap(quality)
        best_move = None
        best_score = 1e-12
        for q, model in enumerate(models):
            current_quality = quality[q]
            for remove_i in list(scheduled[q]):
                if remove_i == model.owner:
                    continue
                remove_cost = int(model.report_bits[remove_i])
                for add_i in range(model.num_uavs):
                    if add_i == model.owner or add_i in scheduled[q]:
                        continue
                    add_cost = int(model.report_bits[add_i])
                    new_cost = used - remove_cost + add_cost
                    if new_cost > budget_bits:
                        continue
                    trial_set = set(scheduled[q])
                    trial_set.remove(remove_i)
                    trial_set.add(add_i)
                    trial_quality = expected(q, frozenset(trial_set))
                    trial_values = list(quality)
                    trial_values[q] = trial_quality
                    gap_reduction = current_gap - normalized_gap(trial_values)
                    if gap_reduction > 1e-12:
                        score = gap_reduction
                    else:
                        score = perf_w[q] * max(
                            trial_quality - current_quality, 0.0
                        )
                    if score > best_score:
                        best_score = score
                        best_move = (
                            "swap", q, remove_i, add_i, trial_quality, new_cost
                        )
                # Removal-only move.
                trial_set = set(scheduled[q])
                trial_set.remove(remove_i)
                trial_quality = expected(q, frozenset(trial_set))
                trial_values = list(quality)
                trial_values[q] = trial_quality
                gap_reduction = current_gap - normalized_gap(trial_values)
                if gap_reduction > 1e-12:
                    score = gap_reduction
                else:
                    score = perf_w[q] * max(
                        trial_quality - current_quality, 0.0
                    )
                if score > best_score:
                    best_score = score
                    best_move = (
                        "remove", q, remove_i, None, trial_quality,
                        used - remove_cost,
                    )
            # Addition-only move.
            for add_i in range(model.num_uavs):
                if add_i == model.owner or add_i in scheduled[q]:
                    continue
                add_cost = int(model.report_bits[add_i])
                if used + add_cost > budget_bits:
                    continue
                trial_set = set(scheduled[q])
                trial_set.add(add_i)
                trial_quality = expected(q, frozenset(trial_set))
                trial_values = list(quality)
                trial_values[q] = trial_quality
                gap_reduction = current_gap - normalized_gap(trial_values)
                if gap_reduction > 1e-12:
                    score = gap_reduction
                else:
                    score = perf_w[q] * max(
                        trial_quality - current_quality, 0.0
                    )
                if score > best_score:
                    best_score = score
                    best_move = (
                        "add", q, None, add_i, trial_quality, used + add_cost
                    )
        if best_move is None:
            break
        move_type, q, remove_i, add_i, trial_quality, new_cost = best_move
        if move_type in ("swap", "remove"):
            scheduled[q].remove(remove_i)
        if move_type in ("swap", "add"):
            scheduled[q].add(add_i)
        used = new_cost
        quality[q] = trial_quality
        trace.append({
            "round": rounds,
            "move": move_type,
            "target": q,
            "remove_uav": -1 if remove_i is None else remove_i,
            "add_uav": -1 if add_i is None else add_i,
            "used_bits": used,
            "expected_pd": float(trial_quality),
            "score": float(best_score),
        })
        rounds += 1

    return ExpectedPdSelectionResult(
        scheduled=tuple(frozenset(group) for group in scheduled),
        expected_pd=np.asarray(quality, dtype=float),
        used_bits=used,
        normalized_qos_gap=normalized_gap(quality),
        trace=tuple(trace),
    )
