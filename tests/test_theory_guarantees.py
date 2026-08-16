import numpy as np
import pytest

from uav_otfs_isac.power_split_theory import _proportional_pd
from uav_otfs_isac.theory_guarantees import (
    complexity_bounds,
    concavity_condition,
    exhaustive_single_target,
    greedy_single_target,
    pd_of_deflection,
    verify_concavity_grid,
    verify_greedy_ratio,
    verify_refinement_monotone,
    verify_submodularity,
)


def test_pd_of_deflection_matches_proportional_model():
    owner_delta = 0.7
    deltas = np.array([1.2, 0.9, 1.5])
    bits = np.array([1, 1, 2], dtype=int)
    powers = np.array([1.0, 0.0, 1.0])
    direct = _proportional_pd(owner_delta, deltas, powers, bits, grid=64)
    # deflection = owner + delta @ powers（单位增益近似下）
    deflection = owner_delta + float(deltas @ powers)
    closed = pd_of_deflection(deflection, false_alarm_rate=0.05)
    assert closed > 0.5  # 合理量级
    assert direct > 0.5
    # 闭式公式与数值实现单调一致
    assert abs(closed - direct) < 0.5  # 数量级一致性（非精确相等）


def test_concavity_condition_at_typical_operating_point():
    result = concavity_condition(4.0, false_alarm_rate=0.05, variance_ratio=1.0)
    assert result["second_derivative"] < 0.0
    assert result["sufficient_condition_holds"]


def test_concavity_grid_no_violations_equal_variance():
    result = verify_concavity_grid(false_alarm_rate=0.05, variance_ratio=1.0)
    assert result["violations"] == 0
    assert result["concave_on_grid"]
    assert result["worst_second_derivative"] <= 1e-12


def test_submodularity_exhaustive_small():
    rng = np.random.default_rng(7)
    result = verify_submodularity(
        0.5,
        rng.uniform(0.5, 2.0, 4),
        np.ones(4, dtype=int),
        grid=32,
        max_reports=4,
    )
    assert result["submodular"], result
    assert result["monotone"]
    assert result["subset_count"] == 16


def test_submodularity_many_random_instances():
    rng = np.random.default_rng(11)
    for _ in range(5):
        result = verify_submodularity(
            float(rng.uniform(0.3, 1.0)),
            rng.uniform(0.5, 2.0, 4),
            rng.integers(1, 3, 4),
            grid=32,
            max_reports=4,
        )
        assert result["submodular"], result


def test_greedy_bound_holds_small_instances():
    rng = np.random.default_rng(3)
    for _ in range(3):
        result = verify_greedy_ratio(
            0.6, rng.uniform(0.6, 2.0, 5), rng.integers(1, 3, 5),
            grid=32, cardinality=3,
        )
        assert result["bound_holds"], result
        assert result["ratio"] >= 1.0 - 1.0 / np.e - 1e-9


def test_greedy_matches_exhaustive_single_report():
    rng = np.random.default_rng(5)
    owner = 0.5
    deltas = rng.uniform(0.6, 2.0, 4)
    bits = np.ones(4, dtype=int)
    greedy_active, greedy_value = greedy_single_target(
        owner, deltas, bits, grid=32, cardinality=2,
    )
    opt_active, opt_value = exhaustive_single_target(
        owner, deltas, bits, grid=32, cardinality=2,
    )
    # 单调子模下贪心至少达到 (1-1/e) OPT
    assert greedy_value >= opt_value * (1.0 - 1.0 / np.e) - 1e-9


def test_complexity_bounds_scale_linearly():
    small = complexity_bounds(Q=3, R=3, T=100)
    large = complexity_bounds(Q=12, R=8, T=100)
    assert "O(3 * 3 * 100)" in small["greedy_time"]
    assert "O(12 * 8^2 * 100" in large["refine_time"]
    assert small["space"] == "O(3 * 3)"
    assert "linear" in small["uav_count_dependence"]


def test_refinement_monotone_demo_scenario():
    scenario = [
        (0.8, np.array([1.2, 0.9, 1.5]), np.array([0.02, 0.03, 0.02]), np.array([0.95, 0.9, 0.93])),
        (0.7, np.array([1.0, 1.4, 0.8]), np.array([0.03, 0.02, 0.04]), np.array([0.9, 0.95, 0.88])),
        (0.9, np.array([1.1, 1.3, 1.0]), np.array([0.02, 0.02, 0.03]), np.array([0.93, 0.92, 0.94])),
    ]
    result = verify_refinement_monotone(scenario, 24, grid=16, max_rounds=20)
    assert result["monotone"], result