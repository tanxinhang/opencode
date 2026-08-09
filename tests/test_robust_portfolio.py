import itertools

import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.robust_portfolio import (
    enumerate_robust_target_portfolios,
    optimize_independent_robust_chance_constrained_portfolio,
    optimize_robust_chance_constrained_portfolio,
)


def scenario_pair():
    clean = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.6
    )
    degraded = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.3
    )
    return clean, degraded


def test_robust_option_uses_worst_case_over_scenarios():
    clean, degraded = scenario_pair()
    options = enumerate_robust_target_portfolios(
        [clean, degraded],
        minimum_quality=0.17,
        target_weight=1.0,
        beta=0.9,
        tail_weight=1.0,
        quality_mode="gaussian_pd",
        false_alarm_rate=0.05,
    )
    all_scheduled = next(
        option for option in options
        if option.scheduled == frozenset(range(clean.num_uavs))
    )
    assert all_scheduled.worst_violation_probability == max(
        all_scheduled.scenario_violations
    )
    assert all_scheduled.worst_violation_probability > (
        all_scheduled.scenario_violations[0] + 1e-12
    )


def test_robust_dp_matches_direct_product_enumeration():
    clean_a, degraded_a = scenario_pair()
    clean_b, degraded_b = scenario_pair()
    groups = [
        enumerate_robust_target_portfolios(
            [clean_a, degraded_a],
            0.17, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        ),
        enumerate_robust_target_portfolios(
            [clean_b, degraded_b],
            0.17, 1.3, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        ),
    ]
    result = optimize_robust_chance_constrained_portfolio(
        [[clean_a, degraded_a], [clean_b, degraded_b]],
        budget_bits=2,
        minimum_quality=[0.17, 0.17],
        target_weights=[1.0, 1.3],
        violation_limits=[0.1, 0.1],
        quality_mode="gaussian_pd",
        false_alarm_rate=0.05,
    )

    def worst_excess(pair):
        return max(
            sum(
                max(option.scenario_violations[s] - limit, 0.0) * weight
                for (option, weight, limit) in zip(pair, [1.0, 1.3], [0.1, 0.1])
            )
            for s in range(2)
        )

    feasible = [
        pair for pair in itertools.product(*groups)
        if sum(option.cost_bits for option in pair) <= 2
    ]
    direct = min(
        (
            worst_excess(pair),
            max(
                sum(option.scenario_risk_objectives[s] for option in pair)
                for s in range(2)
            ),
        )
        for pair in feasible
    )
    assert np.isclose(
        result.worst_weighted_violation_excess, direct[0]
    )
    assert np.isclose(result.worst_risk_objective, direct[1])


def test_robust_allocation_is_not_worse_than_nominal_under_worst_scenario():
    clean_a, degraded_a = scenario_pair()
    clean_b, degraded_b = scenario_pair()
    groups = [
        enumerate_robust_target_portfolios(
            [clean_a, degraded_a],
            0.17, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        ),
        enumerate_robust_target_portfolios(
            [clean_b, degraded_b],
            0.17, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        ),
    ]

    def worst_excess_of_schedule(scheduled):
        excesses = []
        for s in range(2):
            total = 0.0
            for q, group in enumerate(groups):
                option = next(
                    option for option in group
                    if option.scheduled == scheduled[q]
                )
                total += max(
                    option.scenario_violations[s] - 0.1, 0.0
                )
            excesses.append(total)
        return max(excesses)

    nominal = next(
        pair for pair in itertools.product(*groups)
        if sum(option.cost_bits for option in pair) <= 2
    )
    nominal_schedule = tuple(option.scheduled for option in nominal)
    robust = optimize_robust_chance_constrained_portfolio(
        [[clean_a, degraded_a], [clean_b, degraded_b]],
        budget_bits=2,
        minimum_quality=[0.17, 0.17],
        target_weights=[1.0, 1.0],
        violation_limits=[0.1, 0.1],
        quality_mode="gaussian_pd",
        false_alarm_rate=0.05,
    )
    assert robust.worst_weighted_violation_excess <= (
        worst_excess_of_schedule(nominal_schedule) + 1e-12
    )


def test_scenario_models_must_share_report_costs():
    clean = symmetric_diversity_model(report_bits=np.ones(4, dtype=int))
    other = symmetric_diversity_model(
        report_bits=np.array([1, 2, 1, 1], dtype=int)
    )
    try:
        enumerate_robust_target_portfolios(
            [clean, other],
            0.17, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )
    except ValueError as error:
        assert "report costs" in str(error)
    else:
        raise AssertionError("mismatched report costs must be rejected")


def test_scenario_count_mismatch_is_rejected():
    clean_a = symmetric_diversity_model(success_probability=0.6)
    clean_b = symmetric_diversity_model(success_probability=0.6)
    degraded_b = symmetric_diversity_model(success_probability=0.3)
    try:
        optimize_robust_chance_constrained_portfolio(
            [[clean_a], [clean_b, degraded_b]],
            budget_bits=1,
            minimum_quality=[0.17, 0.17],
            target_weights=[1.0, 1.0],
            violation_limits=[0.1, 0.1],
            quality_mode="gaussian_pd",
            false_alarm_rate=0.05,
        )
    except ValueError as error:
        assert "same number of scenarios" in str(error)
    else:
        raise AssertionError("scenario-count mismatch must be rejected")


def test_independent_robust_dp_matches_bruteforce_with_different_scenario_counts():
    clean_a = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.6
    )
    degraded_a = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.3
    )
    clean_b = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.6
    )
    mid_b = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.45
    )
    hard_b = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.2
    )
    groups = [
        enumerate_robust_target_portfolios(
            [clean_a, degraded_a],
            0.17, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        ),
        enumerate_robust_target_portfolios(
            [clean_b, mid_b, hard_b],
            0.17, 1.2, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        ),
    ]
    result = optimize_independent_robust_chance_constrained_portfolio(
        [[clean_a, degraded_a], [clean_b, mid_b, hard_b]],
        budget_bits=2,
        minimum_quality=[0.17, 0.17],
        target_weights=[1.0, 1.2],
        violation_limits=[0.1, 0.1],
        quality_mode="gaussian_pd",
        false_alarm_rate=0.05,
    )
    assert result.ambiguity_mode == "independent"
    feasible = [
        pair for pair in itertools.product(*groups)
        if sum(option.cost_bits for option in pair) <= 2
    ]

    def worst_excess(pair):
        return sum(
            weight
            * max(
                option.worst_violation_probability - limit,
                0.0,
            )
            for (option, weight, limit) in zip(pair, [1.0, 1.2], [0.1, 0.1])
        )

    def worst_risk(pair):
        return sum(
            option.worst_risk_objective for option in pair
        )

    direct = min(
        (worst_excess(pair), worst_risk(pair)) for pair in feasible
    )
    assert np.isclose(result.worst_weighted_violation_excess, direct[0])
    assert np.isclose(result.worst_risk_objective, direct[1])
