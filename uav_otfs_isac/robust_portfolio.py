"""Exact worst-case chance-constrained portfolio allocation.

The nominal DP in :mod:`uav_otfs_isac.risk` optimizes against one fixed
model per target.  This module extends it to a finite scenario set, for
example the same physical scene under different interference, BSC, erasure,
or mobility realizations.  Each target-level schedule is scored under every
scenario, and the DP minimizes the maximum scenario-weighted violation
excess, then the maximum scenario risk objective.

The DP is exact for a finite scenario set: every scenario's total excess and
risk are monotone sums over the chosen target options, so a componentwise
nondominated label at a given cost cannot be improved by any future target.
When degradation states are independent across targets, the exact problem
separates per target; see
:func:`optimize_independent_robust_chance_constrained_portfolio`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import SelectionResult, TargetEvidenceModel
from .risk import enumerate_target_portfolios


@dataclass(frozen=True)
class RobustPortfolioOption:
    target: int
    scheduled: frozenset[int]
    cost_bits: int
    scenario_violations: tuple[float, ...]
    scenario_risk_objectives: tuple[float, ...]
    scenario_mean_losses: tuple[float, ...]
    scenario_cvar_losses: tuple[float, ...]
    scenario_expected_quality: tuple[float, ...]
    scenario_expected_deflection: tuple[float, ...]

    @property
    def worst_violation_probability(self) -> float:
        return float(max(self.scenario_violations))

    @property
    def worst_risk_objective(self) -> float:
        return float(max(self.scenario_risk_objectives))


@dataclass(frozen=True)
class RobustChancePortfolioResult:
    selection: SelectionResult
    worst_violation_probability_per_target: np.ndarray
    nominal_violation_probability_per_target: np.ndarray
    worst_weighted_violation_excess: float
    worst_risk_objective: float
    scenario_count: int
    feasible: bool
    ambiguity_mode: str = "common"


_OBJECTIVE_TOLERANCE = 1e-12


def _validate_scenario_models(
    scenario_models: Sequence[TargetEvidenceModel],
) -> None:
    if not scenario_models:
        raise ValueError("at least one scenario model is required")
    base = scenario_models[0]
    for model in scenario_models[1:]:
        if model.num_uavs != base.num_uavs:
            raise ValueError("scenario models must have the same UAV count")
        if model.owner != base.owner:
            raise ValueError("scenario models must share the owner")
        if not np.array_equal(model.report_bits, base.report_bits):
            raise ValueError("scenario models must share report costs")


def enumerate_robust_target_portfolios(
    scenario_models: Sequence[TargetEvidenceModel],
    minimum_quality: float,
    target_weight: float,
    beta: float,
    tail_weight: float,
    *,
    quality_mode: str = "deflection",
    false_alarm_rate: float = 0.05,
) -> list[RobustPortfolioOption]:
    """Enumerate schedules with exact worst-case metrics over scenarios."""
    _validate_scenario_models(scenario_models)
    groups = [
        {
            option.scheduled: option
            for option in enumerate_target_portfolios(
                model,
                minimum_quality,
                target_weight,
                beta,
                tail_weight,
                quality_mode,
                false_alarm_rate,
            )
        }
        for model in scenario_models
    ]
    base = groups[0]
    for group in groups[1:]:
        if set(base) != set(group):
            raise ValueError("scenario models must admit the same schedules")
    result = []
    for schedule, nominal in base.items():
        options = [group[schedule] for group in groups]
        result.append(RobustPortfolioOption(
            target=nominal.target,
            scheduled=schedule,
            cost_bits=nominal.cost_bits,
            scenario_violations=tuple(
                float(option.violation_probability) for option in options
            ),
            scenario_risk_objectives=tuple(
                float(option.risk_objective) for option in options
            ),
            scenario_mean_losses=tuple(
                float(option.mean_loss) for option in options
            ),
            scenario_cvar_losses=tuple(
                float(option.cvar_loss) for option in options
            ),
            scenario_expected_quality=tuple(
                float(option.expected_quality) for option in options
            ),
            scenario_expected_deflection=tuple(
                float(option.expected_deflection) for option in options
            ),
        ))
    return result


def _label_dominates(
    left_excess: tuple[float, ...],
    left_risk: tuple[float, ...],
    right_excess: tuple[float, ...],
    right_risk: tuple[float, ...],
) -> bool:
    return (
        all(
            a <= b + _OBJECTIVE_TOLERANCE
            for a, b in zip(left_excess, right_excess)
        )
        and all(
            a <= b + _OBJECTIVE_TOLERANCE
            for a, b in zip(left_risk, right_risk)
        )
        and (
            any(
                a < b - _OBJECTIVE_TOLERANCE
                for a, b in zip(left_excess, right_excess)
            )
            or any(
                a < b - _OBJECTIVE_TOLERANCE
                for a, b in zip(left_risk, right_risk)
            )
        )
    )


def optimize_robust_chance_constrained_portfolio(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    budget_bits: int,
    minimum_quality: Sequence[float],
    target_weights: Sequence[float],
    violation_limits: Sequence[float],
    *,
    beta: float = 0.9,
    tail_weight: float = 1.0,
    quality_mode: str = "deflection",
    false_alarm_rate: float = 0.05,
    option_groups: Sequence[Sequence[RobustPortfolioOption]] | None = None,
) -> RobustChancePortfolioResult:
    """Exact worst-case multiple-choice knapsack over scenario vectors."""
    if budget_bits < 0:
        raise ValueError("budget_bits must be nonnegative")
    if tail_weight < 0:
        raise ValueError("tail_weight must be nonnegative")
    weights = np.asarray(target_weights, dtype=float)
    limits = np.asarray(violation_limits, dtype=float)
    if len(scenario_models_by_target) != len(minimum_quality):
        raise ValueError("one scenario group is required per target")
    if len(minimum_quality) != len(weights) or len(weights) != len(limits):
        raise ValueError("target parameter lengths must match")
    if np.any((limits < 0.0) | (limits > 1.0)):
        raise ValueError("violation limits must lie in [0, 1]")
    if option_groups is None:
        scenario_counts = {
            len(scenarios) for scenarios in scenario_models_by_target
        }
        if len(scenario_counts) != 1:
            raise ValueError(
                "all targets must have the same number of scenarios"
            )
        scenario_count = scenario_counts.pop()
        if scenario_count == 0:
            raise ValueError("each target needs at least one scenario")
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                minimum_quality[q],
                float(weights[q]),
                beta,
                tail_weight,
                quality_mode=quality_mode,
                false_alarm_rate=false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_models_by_target)
        ]
    else:
        option_groups = list(option_groups)
        if len(option_groups) != len(scenario_models_by_target):
            raise ValueError(
                "option_groups must contain one group per target"
            )
        if not option_groups or not option_groups[0]:
            raise ValueError("option groups must be nonempty")
        scenario_count = len(option_groups[0][0].scenario_violations)
        if scenario_count == 0:
            raise ValueError("each option group needs at least one scenario")
        for group in option_groups:
            if not group:
                raise ValueError("option groups must be nonempty")
            if any(
                len(option.scenario_violations) != scenario_count
                for option in group
            ):
                raise ValueError(
                    "all options in a group must use the same scenario count"
                )
    zero_excess = tuple(0.0 for _ in range(scenario_count))
    zero_risk = tuple(0.0 for _ in range(scenario_count))

    states: dict[
        int,
        list[tuple[
            tuple[float, ...],
            tuple[float, ...],
            tuple[RobustPortfolioOption, ...],
        ]],
    ] = {0: [(zero_excess, zero_risk, tuple())]}
    for q, options in enumerate(option_groups):
        next_states: dict[
            int,
            list[tuple[
                tuple[float, ...],
                tuple[float, ...],
                tuple[RobustPortfolioOption, ...],
            ]],
        ] = {}
        for prior_cost, labels in states.items():
            for prior_excess, prior_risk, chosen in labels:
                for option in options:
                    cost = prior_cost + option.cost_bits
                    if cost > budget_bits:
                        continue
                    excess = tuple(
                        prior_excess[s]
                        + weights[q]
                        * max(
                            option.scenario_violations[s] - limits[q],
                            0.0,
                        )
                        for s in range(scenario_count)
                    )
                    risk = tuple(
                        prior_risk[s] + option.scenario_risk_objectives[s]
                        for s in range(scenario_count)
                    )
                    label = (excess, risk, chosen + (option,))
                    labels_at_cost = next_states.setdefault(cost, [])
                    if labels_at_cost:
                        existing_excess = np.asarray([
                            value[0] for value in labels_at_cost
                        ])
                        existing_risk = np.asarray([
                            value[1] for value in labels_at_cost
                        ])
                        new_excess_array = np.asarray(excess, dtype=float)
                        new_risk_array = np.asarray(risk, dtype=float)
                        existing_dominates_new = (
                            np.all(
                                existing_excess
                                <= new_excess_array[None, :] + _OBJECTIVE_TOLERANCE,
                                axis=1,
                            )
                            & np.all(
                                existing_risk
                                <= new_risk_array[None, :] + _OBJECTIVE_TOLERANCE,
                                axis=1,
                            )
                            & (
                                np.any(
                                    existing_excess
                                    < new_excess_array[None, :] - _OBJECTIVE_TOLERANCE,
                                    axis=1,
                                )
                                | np.any(
                                    existing_risk
                                    < new_risk_array[None, :] - _OBJECTIVE_TOLERANCE,
                                    axis=1,
                                )
                            )
                        )
                        if bool(existing_dominates_new.any()):
                            continue
                        new_dominates_existing = (
                            np.all(
                                new_excess_array[None, :]
                                <= existing_excess + _OBJECTIVE_TOLERANCE,
                                axis=1,
                            )
                            & np.all(
                                new_risk_array[None, :]
                                <= existing_risk + _OBJECTIVE_TOLERANCE,
                                axis=1,
                            )
                            & (
                                np.any(
                                    new_excess_array[None, :]
                                    < existing_excess - _OBJECTIVE_TOLERANCE,
                                    axis=1,
                                )
                                | np.any(
                                    new_risk_array[None, :]
                                    < existing_risk - _OBJECTIVE_TOLERANCE,
                                    axis=1,
                                )
                            )
                        )
                        labels_at_cost[:] = [
                            value for value, keep in zip(
                                labels_at_cost, ~new_dominates_existing
                            ) if keep
                        ]
                    labels_at_cost.append(label)
        states = next_states

    candidates = []
    for used, labels in states.items():
        for excess, risk, chosen in labels:
            candidates.append((
                float(max(excess)),
                float(max(risk)),
                used,
                chosen,
            ))
    if not candidates:
        raise RuntimeError("no feasible budget allocation")
    best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    _, worst_risk, used, chosen = best
    worst_violation = np.asarray([
        option.worst_violation_probability for option in chosen
    ])
    nominal_violation = np.asarray([
        option.scenario_violations[0] for option in chosen
    ])
    nominal_quality = np.asarray([
        option.scenario_expected_quality[0] for option in chosen
    ])
    nominal_deflection = np.asarray([
        option.scenario_expected_deflection[0] for option in chosen
    ])
    nominal_loss = np.asarray([
        option.scenario_mean_losses[0] for option in chosen
    ])
    selection = SelectionResult(
        scheduled=tuple(option.scheduled for option in chosen),
        expected_deflection=nominal_deflection,
        used_bits=used,
        normalized_qos_gap=float(np.sum(
            weights * nominal_loss / np.maximum(
                np.asarray(minimum_quality), 1e-12
            )
        )),
        trace=tuple(),
        quality_mode=quality_mode,
        expected_quality=nominal_quality,
    )
    return RobustChancePortfolioResult(
        selection=selection,
        worst_violation_probability_per_target=worst_violation,
        nominal_violation_probability_per_target=nominal_violation,
        worst_weighted_violation_excess=best[0],
        worst_risk_objective=worst_risk,
        scenario_count=scenario_count,
        feasible=bool(best[0] <= _OBJECTIVE_TOLERANCE),
        ambiguity_mode="common",
    )


def select_scenario_options(
    option_groups: Sequence[Sequence[RobustPortfolioOption]],
    scenario_index: int,
) -> tuple[tuple[RobustPortfolioOption, ...], ...]:
    """Rebuild option groups restricted to one scenario."""
    result = []
    for group in option_groups:
        if not group:
            raise ValueError("option groups must be nonempty")
        scenario_count = len(group[0].scenario_violations)
        if not 0 <= scenario_index < scenario_count:
            raise ValueError("scenario_index is outside the option group")
        result.append(tuple(
            RobustPortfolioOption(
                target=option.target,
                scheduled=option.scheduled,
                cost_bits=option.cost_bits,
                scenario_violations=(
                    option.scenario_violations[scenario_index],
                ),
                scenario_risk_objectives=(
                    option.scenario_risk_objectives[scenario_index],
                ),
                scenario_mean_losses=(
                    option.scenario_mean_losses[scenario_index],
                ),
                scenario_cvar_losses=(
                    option.scenario_cvar_losses[scenario_index],
                ),
                scenario_expected_quality=(
                    option.scenario_expected_quality[scenario_index],
                ),
                scenario_expected_deflection=(
                    option.scenario_expected_deflection[scenario_index],
                ),
            )
            for option in group
        ))
    return tuple(result)


def optimize_independent_robust_chance_constrained_portfolio(
    scenario_models_by_target: Sequence[Sequence[TargetEvidenceModel]],
    budget_bits: int,
    minimum_quality: Sequence[float],
    target_weights: Sequence[float],
    violation_limits: Sequence[float],
    *,
    beta: float = 0.9,
    tail_weight: float = 1.0,
    quality_mode: str = "deflection",
    false_alarm_rate: float = 0.05,
) -> RobustChancePortfolioResult:
    """Exact worst-case DP for independent per-target ambiguity.

    When degradation states are independent across targets, the worst-case
    total excess is the sum of each target's worst-case excess, and the
    worst-case total risk is the sum of each target's worst-case risk.  The
    scalar DP over these worst-case option values is therefore exact without
    carrying a common scenario vector.
    """
    if budget_bits < 0:
        raise ValueError("budget_bits must be nonnegative")
    if tail_weight < 0:
        raise ValueError("tail_weight must be nonnegative")
    weights = np.asarray(target_weights, dtype=float)
    limits = np.asarray(violation_limits, dtype=float)
    if len(scenario_models_by_target) != len(minimum_quality):
        raise ValueError("one scenario group is required per target")
    if len(minimum_quality) != len(weights) or len(weights) != len(limits):
        raise ValueError("target parameter lengths must match")
    if np.any((limits < 0.0) | (limits > 1.0)):
        raise ValueError("violation limits must lie in [0, 1]")
    option_groups = [
        enumerate_robust_target_portfolios(
            scenarios,
            minimum_quality[q],
            float(weights[q]),
            beta,
            tail_weight,
            quality_mode=quality_mode,
            false_alarm_rate=false_alarm_rate,
        )
        for q, scenarios in enumerate(scenario_models_by_target)
    ]
    states: dict[int, tuple[tuple[float, float], tuple[RobustPortfolioOption, ...]]] = {
        0: ((0.0, 0.0), tuple())
    }
    for q, options in enumerate(option_groups):
        next_states: dict[int, tuple[tuple[float, float], tuple[RobustPortfolioOption, ...]]] = {}
        for prior_cost, (prior_key, chosen) in states.items():
            for option in options:
                cost = prior_cost + option.cost_bits
                if cost > budget_bits:
                    continue
                excess = weights[q] * max(
                    option.worst_violation_probability - limits[q], 0.0
                )
                risk = option.worst_risk_objective
                key = (prior_key[0] + excess, prior_key[1] + risk)
                incumbent = next_states.get(cost)
                if incumbent is None or key < incumbent[0]:
                    next_states[cost] = (key, chosen + (option,))
        states = next_states
    if not states:
        raise RuntimeError("no feasible budget allocation")
    used, (key, chosen) = min(
        states.items(), key=lambda item: (item[1][0][0], item[1][0][1], item[0])
    )
    worst_violation = np.asarray([
        option.worst_violation_probability for option in chosen
    ])
    nominal_violation = np.asarray([
        option.scenario_violations[0] for option in chosen
    ])
    nominal_quality = np.asarray([
        option.scenario_expected_quality[0] for option in chosen
    ])
    nominal_deflection = np.asarray([
        option.scenario_expected_deflection[0] for option in chosen
    ])
    nominal_loss = np.asarray([
        option.scenario_mean_losses[0] for option in chosen
    ])
    selection = SelectionResult(
        scheduled=tuple(option.scheduled for option in chosen),
        expected_deflection=nominal_deflection,
        used_bits=used,
        normalized_qos_gap=float(np.sum(
            weights * nominal_loss / np.maximum(
                np.asarray(minimum_quality), 1e-12
            )
        )),
        trace=tuple(),
        quality_mode=quality_mode,
        expected_quality=nominal_quality,
    )
    return RobustChancePortfolioResult(
        selection=selection,
        worst_violation_probability_per_target=worst_violation,
        nominal_violation_probability_per_target=nominal_violation,
        worst_weighted_violation_excess=float(key[0]),
        worst_risk_objective=float(key[1]),
        scenario_count=max(len(group) for group in scenario_models_by_target),
        feasible=bool(key[0] <= _OBJECTIVE_TOLERANCE),
        ambiguity_mode="independent",
    )
