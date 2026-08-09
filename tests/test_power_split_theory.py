import numpy as np

from uav_otfs_isac.power_split_theory import (
    power_gain_coefficient,
    verify_winner_take_all,
    winner_take_all_allocation,
)


def test_winner_take_all_matches_exhaustive_allocations():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        deltas = rng.uniform(0.8, 2.0, 4)
        bits = np.array([2, 3, 2, 3])
        result = verify_winner_take_all(
            0.4,
            deltas,
            bits,
            flip_probability=0.1,
            success_probability=0.8,
            budget=4.0,
            power_levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            grid=16,
        )
        assert result["passed"]


def test_winner_take_all_concentrates_budget():
    coefficients = np.array([1.0, 2.0, 0.5])
    allocation = winner_take_all_allocation(coefficients, 5.0)
    assert np.argmax(allocation) == 1
    assert np.isclose(allocation.sum(), 5.0)


def test_power_gain_coefficient_is_nonnegative():
    assert power_gain_coefficient(1.5, 2, 0.1, 0.9) >= 0.0
