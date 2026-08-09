import numpy as np

from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.nomp_refinement import (
    initial_min_cover,
    leximin_improves,
    nomp_wta_greedy_joint_multi,
    wta_greedy_joint_multi,
)
from uav_otfs_isac.power_split_theory import (
    winner_take_all_proportional_options,
)


def test_leximin_improves_never_lowers_worst():
    assert leximin_improves([0.1, 0.5], [0.2, 0.4])
    assert not leximin_improves([0.1, 0.5], [0.05, 0.9])
    assert not leximin_improves([0.2, 0.4], [0.2, 0.4])


def test_min_cover_activates_every_target():
    scenario = [
        np.array([0.4, 1.2, 1.8]),
        np.array([0.3, 1.5, 2.0]),
    ]
    powers, bits, used = initial_min_cover(scenario, 8)
    assert used == 4
    assert all(int(powers[q].sum()) == 1 for q in range(2))
    assert all(int(bits[q].sum()) == 1 for q in range(2))


def test_nomp_refinement_matches_exact_wta_frontier():
    scenario = [
        np.array([0.4144419269484395, 1.7589404662431949, 1.176969702820865]),
        np.array([0.30909064469102476, 1.106363940626104, 1.0502143528737093]),
    ]
    for budget in (8, 10, 12):
        result = nomp_wta_greedy_joint_multi(
            scenario, budget, max_rounds=100
        )
        groups = [
            winner_take_all_proportional_options(
                float(t[0]),
                t[1:],
                bit_options=np.arange(3, dtype=int),
                budget=budget,
                grid=16,
            )
            for t in scenario
        ]
        exact = exact_joint_power_bit_maxmin(groups, budget)
        assert abs(result["worst_pd"] - exact) < 1e-6


def test_refinement_respects_budget_and_round_cap():
    scenario = [
        np.array([0.4, 1.2, 1.8]),
        np.array([0.3, 1.5, 2.0]),
    ]
    result = nomp_wta_greedy_joint_multi(
        scenario, 10, max_rounds=5
    )
    assert result["used"] <= 10
    assert result["refine_rounds"] <= 5


def test_wta_greedy_uses_oracle_power_range_and_improves_with_budget():
    scenario = [
        np.array([0.35500712069913876, 1.3473983050328615, 1.6485513544627934]),
        np.array([0.2647764054170017, 0.8398146100530075, 2.0726389423009004]),
    ]
    values = [
        wta_greedy_joint_multi(scenario, budget)["worst_pd"]
        for budget in (8, 10, 12)
    ]
    assert values[0] <= values[1] <= values[2]
    assert values[0] > 0.9
