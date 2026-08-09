import itertools

import numpy as np

from uav_otfs_isac.joint_power_bit import (
    exact_joint_power_bit_maxmin,
    power_bit_target_options,
)


def test_power_bit_options_match_bruteforce():
    deltas = np.array([1.0, 1.2])
    options = power_bit_target_options(
        0.4,
        deltas,
        power_levels=np.array([0.0, 1.0, 2.0]),
        bit_options=np.array([0, 1, 2]),
        budget=6,
        grid=16,
    )
    brute = []
    for powers in itertools.product((0.0, 1.0, 2.0), repeat=2):
        for bits in itertools.product((0, 1, 2), repeat=2):
            cost = int(sum(powers) + sum(bits))
            if cost > 6:
                continue
            from uav_otfs_isac.joint_power_bit import _target_pd
            pd = _target_pd(
                0.4, deltas,
                np.asarray(powers), np.asarray(bits), 16,
            )
            brute.append((cost, pd))
    brute.sort(key=lambda item: (item[0], -item[1]))
    expected = []
    best = -1.0
    last = None
    for cost, pd in brute:
        if cost == last:
            continue
        last = cost
        if pd > best + 1e-12:
            expected.append((cost, pd))
            best = pd
    actual = dict(options)
    reference = dict(expected)
    assert set(actual) == set(reference)
    for cost in actual:
        assert abs(actual[cost] - reference[cost]) < 1e-9


def test_joint_power_bit_maxmin_matches_bruteforce():
    groups = [
        power_bit_target_options(
            0.4,
            np.array([1.0, 1.2]),
            power_levels=np.array([0.0, 1.0, 2.0]),
            bit_options=np.array([0, 1, 2]),
            budget=6,
            grid=16,
        ),
        power_bit_target_options(
            0.3,
            np.array([0.8, 1.0]),
            power_levels=np.array([0.0, 1.0, 2.0]),
            bit_options=np.array([0, 1, 2]),
            budget=6,
            grid=16,
        ),
    ]
    exact = exact_joint_power_bit_maxmin(groups, 6)
    best = -1.0
    for combo in itertools.product(*groups):
        if sum(cost for cost, _ in combo) <= 6:
            best = max(best, min(value for _, value in combo))
    assert abs(exact - best) < 1e-9


def test_joint_dominates_sensing_only_and_communication_only():
    common = dict(
        power_levels=np.array([0.0, 1.0, 2.0]),
        bit_options=np.array([0, 1, 2]),
        budget=6,
        grid=16,
    )
    groups = [
        power_bit_target_options(0.4, np.array([1.0, 1.2]), **common),
        power_bit_target_options(0.3, np.array([0.8, 1.0]), **common),
    ]
    sensing_groups = [
        power_bit_target_options(
            0.4, np.array([1.0, 1.2]),
            power_levels=np.array([0.0, 1.0, 2.0]),
            bit_options=np.array([1]),
            budget=6, grid=16,
        ),
        power_bit_target_options(
            0.3, np.array([0.8, 1.0]),
            power_levels=np.array([0.0, 1.0, 2.0]),
            bit_options=np.array([1]),
            budget=6, grid=16,
        ),
    ]
    comm_groups = [
        power_bit_target_options(
            0.4, np.array([1.0, 1.2]),
            power_levels=np.array([1.0]),
            bit_options=np.array([0, 1, 2]),
            budget=6, grid=16,
        ),
        power_bit_target_options(
            0.3, np.array([0.8, 1.0]),
            power_levels=np.array([1.0]),
            bit_options=np.array([0, 1, 2]),
            budget=6, grid=16,
        ),
    ]
    joint = exact_joint_power_bit_maxmin(groups, 6)
    sensing = exact_joint_power_bit_maxmin(sensing_groups, 6)
    comm = exact_joint_power_bit_maxmin(comm_groups, 6)
    assert joint + 1e-9 >= sensing
    assert joint + 1e-9 >= comm
