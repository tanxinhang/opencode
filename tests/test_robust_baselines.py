import itertools

import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.robust_baselines import (
    evaluate_robust_schedule_worst_excess,
    evaluate_robust_schedule_worst_violation,
    robust_greedy_worst_case,
    worst_case_communication_top_k,
    worst_case_independent_post_top_k,
    worst_case_no_cooperation,
    worst_case_random_top_k,
    worst_case_sensing_top_k,
)
from uav_otfs_isac.robust_portfolio import enumerate_robust_target_portfolios


def scenario_groups():
    clean_a = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.6
    )
    degraded_a = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.3
    )
    clean_b = symmetric_diversity_model(
        np.array([0.6, 1.2, 0.8, 0.4]), success_probability=0.7
    )
    degraded_b = symmetric_diversity_model(
        np.array([0.6, 1.2, 0.8, 0.4]), success_probability=0.4
    )
    return [[clean_a, degraded_a], [clean_b, degraded_b]]


def test_top_k_baselines_respect_budget():
    groups = scenario_groups()
    weights = np.ones(2)
    limits = np.full(2, 0.1)
    for baseline in (
        worst_case_sensing_top_k,
        worst_case_communication_top_k,
        worst_case_independent_post_top_k,
        worst_case_random_top_k,
    ):
        result = baseline(
            groups, 2, weights, limits,
            minimum_quality=[0.17, 0.17],
            false_alarm_rate=0.05,
        )
        assert result.used_bits <= 2
        assert np.isfinite(result.worst_excess)
        assert len(result.worst_violation_probability_per_target) == 2
        assert all(
            0.0 <= value <= 1.0 + 1e-12
            for value in result.worst_violation_probability_per_target
        )

    no_coop = worst_case_no_cooperation(
        groups, weights, limits,
        minimum_quality=[0.17, 0.17],
        false_alarm_rate=0.05,
    )
    assert no_coop.used_bits == 0
    assert 0.0 <= no_coop.worst_excess


def test_robust_greedy_is_no_worse_than_first_feasible_nominal():
    groups = scenario_groups()
    option_groups = [
        enumerate_robust_target_portfolios(
            scenarios, 0.17, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )
        for scenarios in groups
    ]
    weights = np.ones(2)
    limits = np.full(2, 0.1)
    first_feasible = next(
        pair for pair in itertools.product(*option_groups)
        if sum(option.cost_bits for option in pair) <= 2
    )
    nominal_excess = evaluate_robust_schedule_worst_excess(
        option_groups,
        tuple(option.scheduled for option in first_feasible),
        weights,
        limits,
    )
    greedy = robust_greedy_worst_case(
        groups, 2, weights, limits,
        minimum_quality=[0.17, 0.17],
        false_alarm_rate=0.05,
    )
    assert greedy.worst_excess <= nominal_excess + 1e-12
    assert greedy.used_bits <= 2
    assert all(
        0.0 <= value <= 1.0 + 1e-12
        for value in greedy.worst_violation_probability_per_target
    )


def test_worst_violation_helper_uses_max_over_scenarios():
    groups = scenario_groups()
    option_groups = [
        enumerate_robust_target_portfolios(
            scenarios, 0.17, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )
        for scenarios in groups
    ]
    scheduled = tuple(
        option.scheduled for option in (
            option_groups[0][0], option_groups[1][0]
        )
    )
    violations = evaluate_robust_schedule_worst_violation(
        option_groups, scheduled
    )
    assert len(violations) == 2
    for q, group in enumerate(option_groups):
        option = next(
            option for option in group if option.scheduled == scheduled[q]
        )
        assert np.isclose(
            violations[q], max(option.scenario_violations)
        )
