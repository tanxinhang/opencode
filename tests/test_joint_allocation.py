from __future__ import annotations

import numpy as np

from uav_otfs_isac.joint_allocation import (
    exact_joint_maxmin,
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


def test_vectorized_target_options_matches_enumeration() -> None:
    deltas = np.array([1.0, 1.2, 1.4, 1.6])
    enumerated = dict(target_options(0.4, deltas, grid=64))
    vectorized = dict(vectorized_target_options(0.4, deltas, grid=64))
    assert set(enumerated) == set(vectorized)
    for cost in enumerated:
        assert abs(enumerated[cost] - vectorized[cost]) < 1e-9
