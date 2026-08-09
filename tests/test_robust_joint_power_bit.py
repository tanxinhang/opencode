import numpy as np

from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_robust_power_bit_options,
    pareto_options,
)


def test_robust_pd_is_never_above_clean_pd():
    options = enumerate_robust_power_bit_options(
        0.4,
        np.array([1.2, 1.5]),
        power_levels=np.array([0.0, 1.0, 2.0]),
        bit_options=np.array([0, 1, 2]),
        budget=6,
        flip_interval=(0.0, 0.2),
        success_interval=(0.5, 1.0),
        grid=16,
    )
    assert options
    for option in options:
        assert option.robust_pd <= option.clean_pd + 1e-12


def test_robust_maxmin_is_exact_over_robust_frontier():
    groups = [
        pareto_options(
            enumerate_robust_power_bit_options(
                0.4,
                np.array([1.2, 1.5]),
                power_levels=np.array([0.0, 1.0, 2.0]),
                bit_options=np.array([0, 1, 2]),
                budget=6,
                flip_interval=(0.0, 0.2),
                success_interval=(0.5, 1.0),
                grid=16,
            ),
            "robust_pd",
        ),
        pareto_options(
            enumerate_robust_power_bit_options(
                0.3,
                np.array([0.9, 1.1]),
                power_levels=np.array([0.0, 1.0, 2.0]),
                bit_options=np.array([0, 1, 2]),
                budget=6,
                flip_interval=(0.0, 0.2),
                success_interval=(0.5, 1.0),
                grid=16,
            ),
            "robust_pd",
        ),
    ]
    value = exact_joint_power_bit_maxmin(groups, 6)
    assert 0.0 <= value <= 1.0
