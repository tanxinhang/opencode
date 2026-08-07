"""Expected-P_D selection over the exact post-communication reception law.

The system objective is the expected detection probability

``E_PD(q, S) = E_gamma[ P_D(owner | received(S, gamma)) ]``

where ``gamma`` follows the model's independent or correlated reception law
and ``P_D`` is evaluated with the Gate G3 ``P_D``-optimal linear fusion
family.  Because every fixed-pattern ``P_D`` is set-monotone at operating
points with ``P_D >= 0.5``, the expectation is also set-monotone in the
scheduled set.  In the proportional-covariance regime the per-pattern ``P_D``
is a concave function of a modular deflection when the inflection condition
``c + D - z_FA sqrt(D) >= 0`` holds, so the expected objective is monotone
submodular and cardinality-greedy retains the classical ``1 - 1/e`` property.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.stats import norm

from .fusion import (
    gaussian_detection_probability,
    optimal_gaussian_detection_probability,
)
from .models import ExpectedPdSelectionResult, TargetEvidenceModel
from .risk import received_pattern_distribution


def _sample_received_sets(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    rng: np.random.Generator,
    samples: int,
) -> list[frozenset[int]]:
    reports = sorted(set(scheduled) - {model.owner})
    result = []
    for _ in range(samples):
        received = {model.owner}
        if model.reception_state_probabilities is not None:
            state = int(rng.choice(
                model.reception_state_probabilities.size,
                p=model.reception_state_probabilities,
            ))
            conditional = model.conditional_success_probabilities[state]
            for uav in reports:
                if rng.random() < conditional[uav]:
                    received.add(uav)
        elif model.reception_patterns is not None:
            pattern_index = int(rng.choice(
                model.pattern_probabilities.size,
                p=model.pattern_probabilities,
            ))
            pattern = model.reception_patterns[pattern_index]
            received.update(uav for uav in reports if pattern[uav] == 1)
        else:
            for uav in reports:
                if rng.random() < model.success_prob[uav]:
                    received.add(uav)
        result.append(frozenset(received))
    return result


def _pd_for_received(
    model: TargetEvidenceModel,
    received: Iterable[int],
    false_alarm_rate: float,
    pd_mode: str,
    grid: int,
) -> float:
    if pd_mode == "optimal":
        return optimal_gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            received, false_alarm_rate, grid=grid,
        )
    if pd_mode == "deflection":
        return gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            received, false_alarm_rate,
        )
    raise ValueError("pd_mode must be 'optimal' or 'deflection'")


def expected_gaussian_detection_probability(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    false_alarm_rate: float,
    *,
    pd_mode: str = "optimal",
    max_exact_reports: int = 14,
    rng: np.random.Generator | None = None,
    samples: int = 2048,
    grid: int = 4096,
) -> float:
    """Expected P_D over the exact or sampled post-communication reception law."""
    if pd_mode not in {"optimal", "deflection"}:
        raise ValueError("pd_mode must be 'optimal' or 'deflection'")
    reports = sorted(set(scheduled) - {model.owner})
    if len(reports) <= max_exact_reports:
        total = 0.0
        for received, probability in received_pattern_distribution(model, scheduled):
            total += probability * _pd_for_received(
                model, received, false_alarm_rate, pd_mode, grid
            )
        return float(total)
    if rng is None:
        rng = np.random.default_rng(0)
    received_sets = _sample_received_sets(model, scheduled, rng, samples)
    return float(np.mean([
        _pd_for_received(model, received, false_alarm_rate, pd_mode, grid)
        for received in received_sets
    ]))


def pd_inflection_condition(
    deflection: float,
    variance_ratio: float,
    false_alarm_rate: float,
    tolerance: float = 1e-9,
) -> bool:
    """Whether P_D is concave in deflection at the given operating point.

    In the proportional regime ``Sigma1 = variance_ratio * Sigma0``,
    ``P_D = Phi((sqrt(D) - z_FA) / sqrt(c))``.  The second derivative is
    nonpositive exactly when ``c + D - z_FA sqrt(D) >= 0``, which is the
    region where the expected-P_D set function is submodular.
    """
    if deflection < 0.0:
        raise ValueError("deflection must be nonnegative")
    if variance_ratio <= 0.0:
        raise ValueError("variance_ratio must be positive")
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must lie in (0, 1)")
    threshold = norm.ppf(1.0 - false_alarm_rate)
    return bool(
        variance_ratio + deflection - threshold * np.sqrt(deflection)
        >= -tolerance
    )


def expected_pd_greedy_select(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    false_alarm_rate: float,
    *,
    qos_pd: Sequence[float] | None = None,
    qos_weights: Sequence[float] | None = None,
    performance_weights: Sequence[float] | None = None,
    pd_mode: str = "optimal",
    max_exact_reports: int = 14,
    rng: np.random.Generator | None = None,
    samples: int = 2048,
    grid: int = 4096,
) -> ExpectedPdSelectionResult:
    """Budgeted greedy on expected P_D gain per report bit.

    The QoS stage first minimizes normalized miss-deficit, then the
    performance stage maximizes expected-P_D gain per bit without worsening
    the achieved deficit, mirroring the two-stage structure of
    :func:`uav_otfs_isac.selection.greedy_select`.
    """
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
    scheduled = [{model.owner} for model in models]
    used = 0
    trace: list[dict[str, float | int | str]] = []
    cache: dict[tuple[int, frozenset[int]], float] = {}

    def expected(q: int, selected: Iterable[int]) -> float:
        key = (q, frozenset(selected))
        if key not in cache:
            cache[key] = expected_gaussian_detection_probability(
                models[q], key[1], false_alarm_rate, pd_mode=pd_mode,
                max_exact_reports=max_exact_reports, rng=rng, samples=samples,
                grid=grid,
            )
        return cache[key]

    quality = [expected(q, scheduled[q]) for q in range(count)]

    def normalized_gap(values: Sequence[float]) -> float:
        return float(np.sum(
            qos_w * np.maximum(qos - np.asarray(values), 0.0) / np.maximum(qos, 1e-12)
        ))

    def candidates(stage: str):
        current_gap = normalized_gap(quality)
        for q, model in enumerate(models):
            current = quality[q]
            for i in range(model.num_uavs):
                if i == model.owner or i in scheduled[q]:
                    continue
                cost = int(model.report_bits[i])
                if used + cost > budget_bits:
                    continue
                trial = scheduled[q] | {i}
                trial_quality = expected(q, trial)
                true_gain = max(trial_quality - current, 0.0)
                relative_gain = max(
                    (max(1.0 - current, 1e-6) - max(1.0 - trial_quality, 0.0))
                    / max(1.0 - current, 1e-6),
                    0.0,
                )
                if stage == "qos":
                    trial_values = list(quality)
                    trial_values[q] = current + true_gain
                    gap_reduction = current_gap - normalized_gap(trial_values)
                    score = gap_reduction / max(cost, 1)
                else:
                    score = perf_w[q] * true_gain / max(cost, 1)
                yield score, true_gain, relative_gain, q, i, cost, trial_quality

    for stage in ("qos", "performance"):
        while True:
            options = list(candidates(stage))
            if not options:
                break
            options.sort(key=lambda item: (item[0], item[1], -item[5]), reverse=True)
            score, true_gain, relative_gain, q, i, cost, trial_quality = options[0]
            if score <= 1e-14:
                break
            scheduled[q].add(i)
            used += cost
            quality[q] = trial_quality
            trace.append({
                "stage": stage,
                "target": q,
                "uav": i,
                "cost_bits": cost,
                "score": float(score),
                "expected_pd_gain": float(true_gain),
                "relative_miss_deficit_gain": float(relative_gain),
                "expected_pd": float(trial_quality),
            })

    return ExpectedPdSelectionResult(
        scheduled=tuple(frozenset(group) for group in scheduled),
        expected_pd=np.asarray(quality, dtype=float),
        used_bits=used,
        normalized_qos_gap=normalized_gap(quality),
        trace=tuple(trace),
    )
