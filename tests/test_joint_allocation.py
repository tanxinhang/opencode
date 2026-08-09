from __future__ import annotations

import itertools

import numpy as np

from uav_otfs_isac.fusion import optimal_gaussian_detection_probability
from uav_otfs_isac.joint_allocation import (
    exact_joint_maxmin,
    exact_joint_maxmin_selection,
    minimum_cost_joint_threshold,
    minimum_cost_for_threshold,
    moments,
    subset_options,
    target_options,
    vectorized_target_options,
)


def test_subset_options_frontier_is_monotone() -> None:
    options = subset_options(
        0.3,
        np.array([1.0, 1.2]),
        np.array([2, 3]),
        grid=16,
    )
    costs = [cost for cost, _ in options]
    values = [value for _, value in options]
    assert all(lo <= hi for lo, hi in zip(costs, costs[1:]))
    assert all(lo <= hi for lo, hi in zip(values, values[1:]))


def test_exact_joint_maxmin_matches_bruteforce() -> None:
    first = target_options(0.4, np.array([1.6, 1.8]), grid=16)
    second = target_options(0.3, np.array([1.2, 1.4]), grid=16)
    budget = 8
    exact = exact_joint_maxmin([first, second], budget)

    best = -1.0
    for cost_a, value_a in first:
        for cost_b, value_b in second:
            if cost_a + cost_b <= budget:
                best = max(best, min(value_a, value_b))
    assert abs(exact - best) < 1e-9


def test_exact_joint_maxmin_selection_is_feasible_and_at_threshold() -> None:
    groups = [
        target_options(0.4, np.array([1.0, 1.2, 1.4]), grid=16),
        target_options(0.3, np.array([0.8, 1.0, 1.2]), grid=16),
    ]
    value, chosen = exact_joint_maxmin_selection(groups, 6)
    assert np.isclose(value, exact_joint_maxmin(groups, 6))
    assert sum(cost for cost, _ in chosen) <= 6
    assert all(option_value >= value - 1e-9 for _, option_value in chosen)


def test_vectorized_target_options_matches_enumeration() -> None:
    deltas = np.array([1.0, 1.2, 1.4, 1.6])
    enumerated = dict(target_options(0.4, deltas, grid=64))
    vectorized = dict(vectorized_target_options(0.4, deltas, grid=64))
    assert set(enumerated) == set(vectorized)
    for cost in enumerated:
        assert abs(enumerated[cost] - vectorized[cost]) < 1e-9


def test_minimum_cost_joint_threshold_matches_enumeration() -> None:
    deltas = np.array([1.0, 1.2, 1.4, 1.6])
    options = target_options(0.4, deltas, grid=32)
    for threshold in (0.4, 0.6, 0.75, 0.85):
        brute = min(
            (cost for cost, value in options if value >= threshold - 1e-12),
            default=None,
        )
        branch = minimum_cost_joint_threshold(
            0.4, deltas, threshold, grid=32,
        )
        assert branch == brute


def test_minimum_cost_for_threshold_matches_linear_scan() -> None:
    options = target_options(0.4, np.array([1.0, 1.2, 1.4]), grid=16)
    for threshold in (0.3, 0.5, 0.7, 0.9):
        expected = min(
            (cost for cost, value in options if value >= threshold - 1e-12),
            default=None,
        )
        actual = minimum_cost_for_threshold(options, threshold)
        assert actual == expected


def test_exact_joint_maxmin_matches_bruteforce_with_more_targets() -> None:
    groups = [
        target_options(
            0.3 + 0.1 * q,
            np.array([0.8 + 0.3 * q, 1.0 + 0.4 * q]),
            grid=16,
        )
        for q in range(5)
    ]
    budget = 12
    exact = exact_joint_maxmin(groups, budget)
    best = -1.0
    for combo in itertools.product(*groups):
        if sum(cost for cost, _ in combo) <= budget:
            best = max(best, min(value for _, value in combo))
    assert abs(exact - best) < 1e-9


def _brute_limited_options(
    owner_delta, deltas, grid, max_bits, max_reports,
):
    options = [[(0, 0.0, 0.0, 1.0, 1.0)] for _ in deltas]
    for index, delta in enumerate(deltas):
        for bits in range(1, max_bits + 1):
            options[index].append((
                bits, *moments(float(delta), bits),
            ))
    out = []
    for combo in itertools.product(*options):
        selected = [item[0] > 0 for item in combo]
        if sum(selected) > max_reports:
            continue
        cost = sum(item[0] for item in combo)
        mu0 = [0.0]
        mu1 = [owner_delta]
        var0 = [1.0]
        var1 = [1.0]
        for bits, m0, m1, v0, v1 in combo:
            if bits > 0:
                mu0.append(m0)
                mu1.append(m1)
                var0.append(v0)
                var1.append(v1)
        pd = float(optimal_gaussian_detection_probability(
            np.asarray(mu0), np.asarray(mu1),
            np.diag(var0), np.diag(var1),
            set(range(len(mu0))), 0.05, grid=grid,
        ))
        out.append((cost, pd))
    out.sort(key=lambda item: (item[0], -item[1]))
    pareto = []
    best_value = -1.0
    last_cost = None
    for cost, pd in out:
        if cost == last_cost:
            continue
        last_cost = cost
        if pd > best_value + 1e-12:
            pareto.append((cost, pd))
            best_value = pd
    return pareto


def test_target_options_respects_max_reports_and_max_bits() -> None:
    deltas = np.array([1.0, 1.2, 1.4])
    for max_bits in (2, 3):
        for max_reports in (1, 2):
            limited = dict(target_options(
                0.4, deltas, grid=16,
                max_bits=max_bits, max_reports=max_reports,
            ))
            brute = dict(_brute_limited_options(
                0.4, deltas, 16, max_bits, max_reports,
            ))
            assert set(limited) == set(brute)
            for cost in limited:
                assert abs(limited[cost] - brute[cost]) < 1e-9


def test_limited_enumeration_never_improves_exact_joint() -> None:
    groups_full = [
        target_options(
            0.3 + 0.1 * q,
            np.array([1.0 + 0.3 * q, 1.2 + 0.3 * q, 1.4 + 0.3 * q]),
            grid=16,
        )
        for q in range(3)
    ]
    groups_limited = [
        target_options(
            0.3 + 0.1 * q,
            np.array([1.0 + 0.3 * q, 1.2 + 0.3 * q, 1.4 + 0.3 * q]),
            grid=16,
            max_bits=2,
            max_reports=2,
        )
        for q in range(3)
    ]
    budget = 10
    full = exact_joint_maxmin(groups_full, budget)
    limited = exact_joint_maxmin(groups_limited, budget)
    assert limited <= full + 1e-9
