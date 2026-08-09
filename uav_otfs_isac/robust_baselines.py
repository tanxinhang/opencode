"""Strong baselines for the worst-scenario robust allocation study.

All baselines are evaluated on the same worst-case excess scale as the
exact robust DP: for every candidate schedule the maximum scenario-weighted
violation excess is computed from the robust target options.  This keeps the
comparison fair even when the baseline itself does not carry scenario
vectors through the DP.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import TargetEvidenceModel
from .robust_portfolio import (
    RobustPortfolioOption,
    enumerate_robust_target_portfolios,
)


@dataclass(frozen=True)
class RobustBaselineResult:
    name: str
    scheduled: tuple[frozenset[int], ...]
    used_bits: int
    worst_excess: float
    worst_violation_probability_per_target: tuple[float, ...]


def evaluate_robust_schedule_worst_excess(
    option_groups: Sequence[Sequence[RobustPortfolioOption]],
    scheduled: Sequence[frozenset[int]],
    target_weights: np.ndarray,
    violation_limits: np.ndarray,
) -> float:
    """Maximum scenario-weighted violation excess of one schedule."""
    scenario_count = len(option_groups[0][0].scenario_violations)
    excesses = []
    for scenario in range(scenario_count):
        total = 0.0
        for q, group in enumerate(option_groups):
            option = next(
                option for option in group
                if option.scheduled == scheduled[q]
            )
            total += target_weights[q] * max(
                option.scenario_violations[scenario] - violation_limits[q],
                0.0,
            )
        excesses.append(total)
    return float(max(excesses))


def evaluate_robust_schedule_worst_violation(
    option_groups: Sequence[Sequence[RobustPortfolioOption]],
    scheduled: Sequence[frozenset[int]],
) -> tuple[float, ...]:
    """Per-target worst violation probability over all scenarios."""
    values = []
    for q, group in enumerate(option_groups):
        option = next(
            option for option in group
            if option.scheduled == scheduled[q]
        )
        values.append(float(max(option.scenario_violations)))
    return tuple(values)


def _option_for_schedule(
    group: Sequence[RobustPortfolioOption],
    scheduled: frozenset[int],
) -> RobustPortfolioOption:
    return next(option for option in group if option.scheduled == scheduled)


def worst_case_sensing_top_k(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    budget_bits: int,
    target_weights: np.ndarray,
    violation_limits: np.ndarray,
    *,
    minimum_quality: Sequence[float],
    false_alarm_rate: float = 0.05,
    option_groups: Sequence[Sequence[RobustPortfolioOption]] | None = None,
) -> RobustBaselineResult:
    """Top-K ranking by worst-case per-report sensing deflection."""
    if option_groups is None:
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                minimum_quality[q],
                float(target_weights[q]),
                0.9,
                1.0,
                quality_mode="gaussian_pd",
                false_alarm_rate=false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_models_by_target)
        ]
    option_groups = list(option_groups)
    candidates = []
    for q, scenarios in enumerate(scenario_models_by_target):
        owner = scenarios[0].owner
        for i in range(scenarios[0].num_uavs):
            if i == owner:
                continue
            score = min(
                float(model.delta[i] ** 2 / model.sigma0[i, i])
                for model in scenarios
            )
            candidates.append((
                score, q, i, int(scenarios[0].report_bits[i])
            ))
    candidates.sort(reverse=True)
    scheduled = [
        frozenset({scenario_models_by_target[q][0].owner})
        for q in range(len(scenario_models_by_target))
    ]
    used = 0
    for _, q, i, cost in candidates:
        if used + cost <= budget_bits:
            scheduled[q] = scheduled[q] | {i}
            used += cost
    return RobustBaselineResult(
        name="worst_case_sensing_top_k",
        scheduled=tuple(scheduled),
        used_bits=used,
        worst_excess=evaluate_robust_schedule_worst_excess(
            option_groups, scheduled, target_weights, violation_limits
        ),
        worst_violation_probability_per_target=(
            evaluate_robust_schedule_worst_violation(
                option_groups, scheduled
            )
        ),
    )


def worst_case_communication_top_k(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    budget_bits: int,
    target_weights: np.ndarray,
    violation_limits: np.ndarray,
    *,
    minimum_quality: Sequence[float],
    false_alarm_rate: float = 0.05,
    option_groups: Sequence[Sequence[RobustPortfolioOption]] | None = None,
) -> RobustBaselineResult:
    """Top-K ranking by worst-case report reliability."""
    if option_groups is None:
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                minimum_quality[q],
                float(target_weights[q]),
                0.9,
                1.0,
                quality_mode="gaussian_pd",
                false_alarm_rate=false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_models_by_target)
        ]
    option_groups = list(option_groups)
    candidates = []
    for q, scenarios in enumerate(scenario_models_by_target):
        owner = scenarios[0].owner
        for i in range(scenarios[0].num_uavs):
            if i == owner:
                continue
            score = min(
                float(
                    model.success_prob[i]
                    * (1.0 - model.bit_flip_prob[i])
                )
                for model in scenarios
            )
            candidates.append((
                score, q, i, int(scenarios[0].report_bits[i])
            ))
    candidates.sort(reverse=True)
    scheduled = [
        frozenset({scenario_models_by_target[q][0].owner})
        for q in range(len(scenario_models_by_target))
    ]
    used = 0
    for _, q, i, cost in candidates:
        if used + cost <= budget_bits:
            scheduled[q] = scheduled[q] | {i}
            used += cost
    return RobustBaselineResult(
        name="worst_case_communication_top_k",
        scheduled=tuple(scheduled),
        used_bits=used,
        worst_excess=evaluate_robust_schedule_worst_excess(
            option_groups, scheduled, target_weights, violation_limits
        ),
        worst_violation_probability_per_target=(
            evaluate_robust_schedule_worst_violation(
                option_groups, scheduled
            )
        ),
    )


def worst_case_independent_post_top_k(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    budget_bits: int,
    target_weights: np.ndarray,
    violation_limits: np.ndarray,
    *,
    minimum_quality: Sequence[float],
    false_alarm_rate: float = 0.05,
    option_groups: Sequence[Sequence[RobustPortfolioOption]] | None = None,
) -> RobustBaselineResult:
    """Top-K by worst-case post-report independent deflection score."""
    if option_groups is None:
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                minimum_quality[q],
                float(target_weights[q]),
                0.9,
                1.0,
                quality_mode="gaussian_pd",
                false_alarm_rate=false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_models_by_target)
        ]
    option_groups = list(option_groups)
    candidates = []
    for q, scenarios in enumerate(scenario_models_by_target):
        owner = scenarios[0].owner
        for i in range(scenarios[0].num_uavs):
            if i == owner:
                continue
            score = min(
                float(
                    model.success_prob[i]
                    * model.delta[i] ** 2
                    / model.sigma0[i, i]
                )
                for model in scenarios
            )
            candidates.append((
                score, q, i, int(scenarios[0].report_bits[i])
            ))
    candidates.sort(reverse=True)
    scheduled = [
        frozenset({scenario_models_by_target[q][0].owner})
        for q in range(len(scenario_models_by_target))
    ]
    used = 0
    for _, q, i, cost in candidates:
        if used + cost <= budget_bits:
            scheduled[q] = scheduled[q] | {i}
            used += cost
    return RobustBaselineResult(
        name="worst_case_independent_post_top_k",
        scheduled=tuple(scheduled),
        used_bits=used,
        worst_excess=evaluate_robust_schedule_worst_excess(
            option_groups, scheduled, target_weights, violation_limits
        ),
        worst_violation_probability_per_target=(
            evaluate_robust_schedule_worst_violation(
                option_groups, scheduled
            )
        ),
    )


def worst_case_random_top_k(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    budget_bits: int,
    target_weights: np.ndarray,
    violation_limits: np.ndarray,
    *,
    minimum_quality: Sequence[float],
    false_alarm_rate: float = 0.05,
    seed: int = 20260809,
    option_groups: Sequence[Sequence[RobustPortfolioOption]] | None = None,
) -> RobustBaselineResult:
    """Deterministic random Top-K schedule evaluated on worst-case scale."""
    if option_groups is None:
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                minimum_quality[q],
                float(target_weights[q]),
                0.9,
                1.0,
                quality_mode="gaussian_pd",
                false_alarm_rate=false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_models_by_target)
        ]
    option_groups = list(option_groups)
    rng = np.random.default_rng(seed)
    candidates = []
    for q, scenarios in enumerate(scenario_models_by_target):
        owner = scenarios[0].owner
        for i in range(scenarios[0].num_uavs):
            if i == owner:
                continue
            candidates.append((
                float(rng.random()), q, i,
                int(scenarios[0].report_bits[i]),
            ))
    candidates.sort(reverse=True)
    scheduled = [
        frozenset({scenario_models_by_target[q][0].owner})
        for q in range(len(scenario_models_by_target))
    ]
    used = 0
    for _, q, i, cost in candidates:
        if used + cost <= budget_bits:
            scheduled[q] = scheduled[q] | {i}
            used += cost
    return RobustBaselineResult(
        name="worst_case_random_top_k",
        scheduled=tuple(scheduled),
        used_bits=used,
        worst_excess=evaluate_robust_schedule_worst_excess(
            option_groups, scheduled, target_weights, violation_limits
        ),
        worst_violation_probability_per_target=(
            evaluate_robust_schedule_worst_violation(
                option_groups, scheduled
            )
        ),
    )


def worst_case_no_cooperation(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    target_weights: np.ndarray,
    violation_limits: np.ndarray,
    *,
    minimum_quality: Sequence[float],
    false_alarm_rate: float = 0.05,
    option_groups: Sequence[Sequence[RobustPortfolioOption]] | None = None,
) -> RobustBaselineResult:
    """Owner-only schedule evaluated on the worst-case scale."""
    if option_groups is None:
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                minimum_quality[q],
                float(target_weights[q]),
                0.9,
                1.0,
                quality_mode="gaussian_pd",
                false_alarm_rate=false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_models_by_target)
        ]
    option_groups = list(option_groups)
    scheduled = [
        frozenset({scenario_models_by_target[q][0].owner})
        for q in range(len(scenario_models_by_target))
    ]
    return RobustBaselineResult(
        name="worst_case_no_cooperation",
        scheduled=tuple(scheduled),
        used_bits=0,
        worst_excess=evaluate_robust_schedule_worst_excess(
            option_groups, scheduled, target_weights, violation_limits
        ),
        worst_violation_probability_per_target=(
            evaluate_robust_schedule_worst_violation(
                option_groups, scheduled
            )
        ),
    )


def robust_greedy_worst_case(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    budget_bits: int,
    target_weights: np.ndarray,
    violation_limits: np.ndarray,
    *,
    minimum_quality: Sequence[float],
    false_alarm_rate: float = 0.05,
    option_groups: Sequence[Sequence[RobustPortfolioOption]] | None = None,
) -> RobustBaselineResult:
    """Greedy that adds the best worst-excess reduction per bit."""
    if option_groups is None:
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                minimum_quality[q],
                float(target_weights[q]),
                0.9,
                1.0,
                quality_mode="gaussian_pd",
                false_alarm_rate=false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_models_by_target)
        ]
    option_groups = list(option_groups)
    scheduled = [
        frozenset({scenario_models_by_target[q][0].owner})
        for q in range(len(scenario_models_by_target))
    ]
    used = 0
    while True:
        current_excess = evaluate_robust_schedule_worst_excess(
            option_groups, scheduled, target_weights, violation_limits
        )
        best = None
        for q, group in enumerate(option_groups):
            current_option = _option_for_schedule(group, scheduled[q])
            for option in group:
                if option.scheduled == scheduled[q]:
                    continue
                if not scheduled[q].issubset(option.scheduled):
                    continue
                incremental_cost = option.cost_bits - current_option.cost_bits
                if incremental_cost <= 0 or used + incremental_cost > budget_bits:
                    continue
                trial = list(scheduled)
                trial[q] = option.scheduled
                trial_excess = evaluate_robust_schedule_worst_excess(
                    option_groups, trial, target_weights, violation_limits
                )
                score = (current_excess - trial_excess) / incremental_cost
                key = (
                    score,
                    trial_excess,
                    -incremental_cost,
                    q,
                    option.scheduled,
                )
                if best is None or key > best[0]:
                    best = (key, trial)
        if best is None:
            break
        _, trial = best
        scheduled = trial
        used = sum(
            _option_for_schedule(option_groups[q], scheduled[q]).cost_bits
            for q in range(len(option_groups))
        )
    return RobustBaselineResult(
        name="robust_greedy_worst_case",
        scheduled=tuple(scheduled),
        used_bits=used,
        worst_excess=evaluate_robust_schedule_worst_excess(
            option_groups, scheduled, target_weights, violation_limits
        ),
        worst_violation_probability_per_target=(
            evaluate_robust_schedule_worst_violation(
                option_groups, scheduled
            )
        ),
    )
