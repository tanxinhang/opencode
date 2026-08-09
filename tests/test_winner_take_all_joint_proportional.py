import numpy as np

from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.power_split_theory import (
    proportional_power_bit_options,
    winner_take_all_proportional_options,
)


def test_winner_take_all_joint_matches_full_proportional_frontier():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        deltas_a = rng.uniform(0.8, 2.0, 3)
        deltas_b = rng.uniform(0.8, 2.0, 3)
        budget = 4
        full_groups = [
            proportional_power_bit_options(
                0.4,
                deltas_a,
                power_levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                bit_options=np.array([0, 1, 2]),
                budget=budget,
                grid=16,
            ),
            proportional_power_bit_options(
                0.3,
                deltas_b,
                power_levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                bit_options=np.array([0, 1, 2]),
                budget=budget,
                grid=16,
            ),
        ]
        winner_groups = [
            winner_take_all_proportional_options(
                0.4,
                deltas_a,
                bit_options=np.array([0, 1, 2]),
                budget=budget,
                grid=16,
            ),
            winner_take_all_proportional_options(
                0.3,
                deltas_b,
                bit_options=np.array([0, 1, 2]),
                budget=budget,
                grid=16,
            ),
        ]
        full_value = exact_joint_power_bit_maxmin(full_groups, budget)
        winner_value = exact_joint_power_bit_maxmin(winner_groups, budget)
        assert abs(winner_value - full_value) < 1e-9
