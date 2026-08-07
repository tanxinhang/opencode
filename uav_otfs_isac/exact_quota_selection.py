"""Exact budget-constrained expected-P_D selection.

For each target the expected-P_D value of every report subset can be evaluated
exactly.  When every non-owner report has the same bit cost, the budget
constraint reduces to a cardinality constraint and the exact selector can be
organized as a search over per-target report quotas.  In general the costs are
heterogeneous, so the exact selector is a multiple-choice knapsack dynamic
program over targets and total report bits.  ``exact_budget_select`` solves
the lexicographic objective (QoS gap, weighted mean, worst target), while
``exact_maxmin_select`` solves the worst-target max-min objective used by
the system-level gates.

Theorem (exactness of the budget DP): let ``O_q`` be the set of all
``(cost(S), P_D(S))`` pairs obtained by enumerating every subset of non-owner
reports for target ``q``.  Every global schedule with total cost at most ``B``
is a path through the DP in which one ``O_q`` option is chosen per target.
For any two partial schedules with the same accumulated cost, if one has
``P_D`` no smaller in every processed target, then every monotone completion
(lower QoS gap, larger weighted sum, or larger minimum) of the dominated
schedule is no better than the corresponding completion of the dominating one.
Keeping only componentwise-Pareto value vectors therefore preserves at least
one optimal path, and the DP terminates with the exact lexicographic optimum.

The exhaustive subset enumeration is limited to ``max_exhaustive_reports``
non-owner reports per target; larger models fall back to the forward greedy
schedule.  The current audited system has seven non-owner reports per target.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from .expected_pd import expected_gaussian_detection_probability
from .expected_pd import expected_pd_greedy_select
from .models import ExpectedPdSelectionResult, TargetEvidenceModel


def subset_expected_pd_map(
    model: TargetEvidenceModel,
    false_alarm_rate: float,
    *,
    pd_mode: str = "optimal",
    grid: int = 512,
) -> dict[frozenset[int], float]:
    """Expected P_D for every subset of non-owner reports."""
    reports = [
        i for i in range(model.num_uavs) if i != model.owner
    ]
    result = {}
    for size in range(len(reports) + 1):
        for subset in combinations(reports, size):
            scheduled = frozenset([model.owner, *subset])
            result[scheduled] = expected_gaussian_detection_probability(
                model, scheduled, false_alarm_rate,
                pd_mode=pd_mode, grid=grid,
            )
    return result


def best_per_size(
    model: TargetEvidenceModel,
    subset_values: dict[frozenset[int], float],
) -> list[tuple[frozenset[int], float]]:
    """Best subset and expected P_D for each report count."""
    max_reports = model.num_uavs - 1
    best = [(frozenset([model.owner]), 0.0)] * (max_reports + 1)
    for scheduled, value in subset_values.items():
        size = len(scheduled) - 1
        if value > best[size][1] + 1e-12:
            best[size] = (scheduled, value)
    return best


def best_by_cost(
    model: TargetEvidenceModel,
    subset_values: dict[frozenset[int], float],
) -> dict[int, tuple[frozenset[int], float]]:
    """Best report subset and expected P_D for each total bit cost."""
    result: dict[int, tuple[frozenset[int], float]] = {}
    for scheduled, value in subset_values.items():
        cost = sum(
            int(model.report_bits[i])
            for i in scheduled
            if i != model.owner
        )
        if cost not in result or value > result[cost][1] + 1e-12:
            result[cost] = (scheduled, float(value))
    return result


def _pareto_dominated_options(
    options: Sequence[tuple[int, frozenset[int], float]],
) -> list[tuple[int, frozenset[int], float]]:
    """Drop per-target options dominated in both cost and value.

    If option A has cost at most cost(B) and value at least value(B), then B
    is dominated: replacing B by A in any global schedule preserves the bit
    budget and never lowers any target's expected P_D.  Because options are
    processed in increasing cost, keeping only options whose value strictly
    exceeds every earlier option implements exactly this dominance rule.
    """
    ordered = sorted(
        options,
        key=lambda item: (item[0], -item[2]),
    )
    kept: list[tuple[int, frozenset[int], float]] = []
    best_value = -np.inf
    for cost, scheduled, value in ordered:
        if value > best_value + 1e-12:
            kept.append((cost, scheduled, float(value)))
            best_value = float(value)
    return kept


def _pareto_frontier(
    states: list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]],
) -> list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]]:
    """Drop states dominated componentwise by another state of equal cost."""
    unique: dict[tuple[float, ...], tuple[frozenset[int], ...]] = {}
    for vector, choices in states:
        if vector not in unique:
            unique[vector] = choices
    keep: list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]] = []
    vectors = list(unique)
    for index, vector in enumerate(vectors):
        dominated = False
        for other_index, other in enumerate(vectors):
            if index == other_index:
                continue
            if all(
                other[k] >= vector[k] - 1e-12
                for k in range(len(vector))
            ):
                dominated = True
                break
        if not dominated:
            keep.append((vector, unique[vector]))
    return keep


def exact_budget_select(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    false_alarm_rate: float,
    *,
    qos_pd: Sequence[float] | None = None,
    qos_weights: Sequence[float] | None = None,
    performance_weights: Sequence[float] | None = None,
    pd_mode: str = "optimal",
    grid: int = 512,
    max_exhaustive_reports: int = 10,
    allow_greedy_fallback: bool = True,
) -> ExpectedPdSelectionResult:
    """Global exact selection under heterogeneous per-report bit costs."""
    if budget_bits < 0:
        raise ValueError("budget_bits must be nonnegative")
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

    max_report_counts = [model.num_uavs - 1 for model in models]
    if any(size > max_exhaustive_reports for size in max_report_counts):
        if not allow_greedy_fallback:
            raise ValueError(
                "report count exceeds max_exhaustive_reports and greedy "
                "fallback is disabled"
            )
        return expected_pd_greedy_select(
            models, budget_bits, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights,
            performance_weights=perf_w, pd_mode=pd_mode, grid=grid,
        )

    def gap(values: Sequence[float]) -> float:
        return float(np.sum(
            qos_w * np.maximum(qos - np.asarray(values), 0.0)
            / np.maximum(qos, 1e-12)
        ))

    target_options = []
    for model in models:
        values = subset_expected_pd_map(
            model, false_alarm_rate, pd_mode=pd_mode, grid=grid
        )
        options = []
        for cost, (scheduled, value) in best_by_cost(model, values).items():
            if cost <= budget_bits:
                options.append((cost, scheduled, value))
        target_options.append(_pareto_dominated_options(options))

    best: dict[int, list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]]] = {
        0: [((), ())]
    }
    for options in target_options:
        next_best: dict[
            int, list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]]
        ] = {}
        for cost, states in best.items():
            for vector, choices in states:
                for option_cost, scheduled, value in options:
                    new_cost = cost + option_cost
                    if new_cost > budget_bits:
                        continue
                    new_vector = vector + (float(value),)
                    new_choices = choices + (scheduled,)
                    next_best.setdefault(new_cost, []).append(
                        (new_vector, new_choices)
                    )
        best = {}
        for cost, states in next_best.items():
            frontier = _pareto_frontier(states)
            if frontier:
                best[cost] = frontier

    best_score: tuple[float, float, float] | None = None
    best_state: tuple[int, tuple[frozenset[int], ...]] | None = None
    for cost, states in best.items():
        for vector, choices in states:
            values = np.asarray(vector, dtype=float)
            score = (
                -gap(values),
                float(np.sum(perf_w * values)),
                float(np.min(values)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_state = (cost, choices)

    if best_state is None:
        scheduled = tuple(frozenset([model.owner]) for model in models)
        values = np.asarray([
            expected_gaussian_detection_probability(
                model, set([model.owner]), false_alarm_rate,
                pd_mode=pd_mode, grid=grid,
            )
            for model in models
        ])
        return ExpectedPdSelectionResult(
            scheduled=scheduled,
            expected_pd=values,
            used_bits=0,
            normalized_qos_gap=gap(values),
            trace=tuple({} for _ in models),
        )

    used_bits, choices = best_state
    scheduled = tuple(choices)
    values = np.asarray([
        expected_gaussian_detection_probability(
            model, group, false_alarm_rate, pd_mode=pd_mode, grid=grid
        )
        for model, group in zip(models, scheduled)
    ])
    return ExpectedPdSelectionResult(
        scheduled=scheduled,
        expected_pd=values,
        used_bits=used_bits,
        normalized_qos_gap=gap(values),
        trace=tuple({
            "target": q,
            "reports": sorted(group),
            "cost_bits": sum(
                int(models[q].report_bits[i])
                for i in group
                if i != models[q].owner
            ),
            "expected_pd": float(values[q]),
        } for q, group in enumerate(scheduled)),
    )


def exact_maxmin_select(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    false_alarm_rate: float,
    *,
    qos_pd: Sequence[float] | None = None,
    qos_weights: Sequence[float] | None = None,
    performance_weights: Sequence[float] | None = None,
    pd_mode: str = "optimal",
    grid: int = 512,
    max_exhaustive_reports: int = 10,
    allow_greedy_fallback: bool = True,
) -> ExpectedPdSelectionResult:
    """Global exact max-min selection under heterogeneous report costs.

    The objective is the worst-target expected `P_D`:

    ``max_{S_q: sum cost(S_q) <= B} min_q f_q(S_q)``.

    For a fixed threshold `t`, feasibility is a multiple-choice knapsack
    problem: choose one enumerated subset per target with cost sum at most
    `B` and value at least `t` for every target.  The threshold is monotone
    in the feasible set, so the exact optimum is found by binary search over
    the finite set of enumerated per-target values, and the returned
    schedule is a feasible schedule at the optimal threshold.
    """
    if budget_bits < 0:
        raise ValueError("budget_bits must be nonnegative")
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

    max_report_counts = [model.num_uavs - 1 for model in models]
    if any(size > max_exhaustive_reports for size in max_report_counts):
        if not allow_greedy_fallback:
            raise ValueError(
                "report count exceeds max_exhaustive_reports and greedy "
                "fallback is disabled"
            )
        return expected_pd_greedy_select(
            models, budget_bits, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights,
            performance_weights=perf_w, pd_mode=pd_mode, grid=grid,
        )

    def gap(values: Sequence[float]) -> float:
        return float(np.sum(
            qos_w * np.maximum(qos - np.asarray(values), 0.0)
            / np.maximum(qos, 1e-12)
        ))

    target_options = []
    candidates: set[float] = set()
    for model in models:
        values = subset_expected_pd_map(
            model, false_alarm_rate, pd_mode=pd_mode, grid=grid
        )
        options = []
        for cost, (scheduled, value) in best_by_cost(model, values).items():
            if cost <= budget_bits:
                options.append((cost, scheduled, float(value)))
        kept = _pareto_dominated_options(options)
        for _, _, value in kept:
            candidates.add(value)
        target_options.append(kept)

    thresholds = sorted(candidates)
    if not thresholds:
        raise ValueError("no feasible report subset exists")

    def feasible(
        threshold: float,
    ) -> dict[int, list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]]] | None:
        dp: dict[
            int, list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]]
        ] = {0: [((), ())]}
        for options in target_options:
            next_dp: dict[
                int, list[tuple[tuple[float, ...], tuple[frozenset[int], ...]]]
            ] = {}
            for cost, choices in dp.items():
                for vector, selected in choices:
                    for option_cost, scheduled, value in options:
                        if value < threshold - 1e-12:
                            continue
                        new_cost = cost + option_cost
                        if new_cost > budget_bits:
                            continue
                        next_dp.setdefault(new_cost, []).append(
                            (vector + (float(value),), selected + (scheduled,))
                        )
            dp = {}
            for cost, states in next_dp.items():
                frontier = _pareto_frontier(states)
                if frontier:
                    dp[cost] = frontier
            if not dp:
                return None
        return dp

    low = 0
    high = len(thresholds) - 1
    best_dp = feasible(thresholds[low])
    assert best_dp is not None
    while low < high:
        mid = (low + high + 1) // 2
        candidate_dp = feasible(thresholds[mid])
        if candidate_dp is not None:
            best_dp = candidate_dp
            low = mid
        else:
            high = mid - 1

    best_score: tuple[float, float, float] | None = None
    best_state: tuple[int, tuple[frozenset[int], ...]] | None = None
    assert best_dp is not None
    for cost, states in best_dp.items():
        for vector, choices in states:
            values = np.asarray(vector, dtype=float)
            score = (
                -gap(values),
                float(np.sum(perf_w * values)),
                float(np.min(values)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_state = (cost, choices)

    if best_state is None:
        scheduled = tuple(frozenset([model.owner]) for model in models)
        values = np.asarray([
            expected_gaussian_detection_probability(
                model, set([model.owner]), false_alarm_rate,
                pd_mode=pd_mode, grid=grid,
            )
            for model in models
        ])
        return ExpectedPdSelectionResult(
            scheduled=scheduled,
            expected_pd=values,
            used_bits=0,
            normalized_qos_gap=gap(values),
            trace=tuple({} for _ in models),
        )

    used_bits, choices = best_state
    scheduled = tuple(choices)
    values = np.asarray([
        expected_gaussian_detection_probability(
            model, group, false_alarm_rate, pd_mode=pd_mode, grid=grid
        )
        for model, group in zip(models, scheduled)
    ])
    return ExpectedPdSelectionResult(
        scheduled=scheduled,
        expected_pd=values,
        used_bits=used_bits,
        normalized_qos_gap=gap(values),
        trace=tuple({
            "target": q,
            "reports": sorted(group),
            "cost_bits": sum(
                int(models[q].report_bits[i])
                for i in group
                if i != models[q].owner
            ),
            "expected_pd": float(values[q]),
        } for q, group in enumerate(scheduled)),
    )


def exact_quota_select(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    false_alarm_rate: float,
    *,
    qos_pd: Sequence[float] | None = None,
    qos_weights: Sequence[float] | None = None,
    performance_weights: Sequence[float] | None = None,
    pd_mode: str = "optimal",
    grid: int = 512,
    max_exhaustive_reports: int = 10,
    allow_greedy_fallback: bool = True,
) -> ExpectedPdSelectionResult:
    """Global exact selection under equal per-report costs."""
    costs = [
        sorted({int(model.report_bits[i]) for i in range(model.num_uavs)
                if i != model.owner})
        for model in models
    ]
    if any(len(set(cost)) != 1 for cost in costs):
        raise ValueError("exact_quota_select requires equal report costs")
    return exact_budget_select(
        models, budget_bits, false_alarm_rate, qos_pd=qos_pd,
        qos_weights=qos_weights,
        performance_weights=performance_weights, pd_mode=pd_mode, grid=grid,
        max_exhaustive_reports=max_exhaustive_reports,
        allow_greedy_fallback=allow_greedy_fallback,
    )


def _compositions(
    total: int,
    count: int,
    capacities: Sequence[int],
):
    """All integer quotas with sum <= total and per-target capacities."""
    quotas = [0] * count

    def recurse(index: int, remaining: int):
        if index == count - 1:
            for value in range(min(remaining, capacities[index]) + 1):
                quotas[index] = value
                yield tuple(quotas)
            quotas[index] = 0
            return
        for value in range(min(remaining, capacities[index]) + 1):
            quotas[index] = value
            yield from recurse(index + 1, remaining - value)
        quotas[index] = 0

    yield from recurse(0, total)
