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
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_heterogeneous_robust_power_bit_options,
    enumerate_robust_power_bit_options,
    pareto_options,
)
from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)
from scripts.run_qos_weighted_maxmin_gate import exact_qos_worst
from scripts.run_joint_power_comparison import (
    make_scenario,
    ucb_wta_greedy_joint_multi,
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


def test_nomp_refinement_matches_exact_on_random_scenarios():
    for seed in range(5):
        scenario = make_scenario(3000 + seed, 2, 2, heterogeneous=True)
        result = nomp_wta_greedy_joint_multi(
            scenario, 8, max_rounds=100
        )
        groups = [
            winner_take_all_proportional_options(
                float(t[0]),
                t[1:],
                bit_options=np.arange(3, dtype=int),
                budget=8,
                grid=16,
            )
            for t in scenario
        ]
        exact = exact_joint_power_bit_maxmin(groups, 8)
        assert abs(result["worst_pd"] - exact) < 1e-6


def test_ucb_nomp_feedback_is_finite_and_keeps_worst_value():
    scenario = make_scenario(10000, 2, 2, heterogeneous=True)
    noisy = ucb_wta_greedy_joint_multi(
        scenario,
        8,
        noise_scale=0.2,
        seed=0,
        min_cover=True,
        refine=True,
        max_steps=100,
        max_feedback_rounds=10,
    )
    exact_value = nomp_wta_greedy_joint_multi(
        scenario, 8, max_rounds=100
    )["worst_pd"]
    assert noisy["steps_used"] <= 100
    assert noisy["feedback_rounds"] <= 10
    assert abs(noisy["worst_pd"] - exact_value) < 1e-9
    assert isinstance(noisy["stopped_by_certificate"], bool)


def test_nomp_refinement_improves_wta_under_comm_errors():
    scenario = make_scenario(10001, 2, 2, heterogeneous=True)
    wta = wta_greedy_joint_multi(
        scenario,
        8,
        min_cover=False,
        flip_probability=0.2,
        success_probability=0.7,
    )["worst_pd"]
    nomp = nomp_wta_greedy_joint_multi(
        scenario,
        8,
        flip_probability=0.2,
        success_probability=0.7,
    )["worst_pd"]
    assert nomp > wta + 1e-6


def test_nomp_refinement_matches_robust_exact_under_comm_errors():
    scenario = make_scenario(10001, 2, 2, heterogeneous=True)
    result = nomp_wta_greedy_joint_multi(
        scenario,
        8,
        flip_probability=0.2,
        success_probability=0.7,
    )
    groups = []
    for target in scenario:
        options = enumerate_robust_power_bit_options(
            float(target[0]),
            target[1:],
            power_levels=np.arange(9, dtype=float),
            bit_options=np.arange(3, dtype=int),
            budget=8,
            flip_interval=(0.0, 0.2),
            success_interval=(0.7, 1.0),
            grid=16,
        )
        groups.append(pareto_options(options, "robust_pd"))
    exact = exact_joint_power_bit_maxmin(groups, 8)
    assert abs(result["worst_pd"] - exact) < 1e-6


def test_success_probability_changes_robust_frontier():
    options_weak = enumerate_robust_power_bit_options(
        0.4,
        np.array([1.8, 2.0]),
        power_levels=np.array([0.0, 1.0, 2.0]),
        bit_options=np.array([0, 1, 2]),
        budget=4,
        flip_interval=(0.0, 0.2),
        success_interval=(0.5, 1.0),
        grid=16,
    )
    options_strong = enumerate_robust_power_bit_options(
        0.4,
        np.array([1.8, 2.0]),
        power_levels=np.array([0.0, 1.0, 2.0]),
        bit_options=np.array([0, 1, 2]),
        budget=4,
        flip_interval=(0.0, 0.2),
        success_interval=(0.99, 1.0),
        grid=16,
    )
    weak = pareto_options(options_weak, "robust_pd")
    strong = pareto_options(options_strong, "robust_pd")
    assert any(
        weak_value < strong_value
        for (_, weak_value), (_, strong_value) in zip(weak, strong)
    )


def test_nomp_refinement_improves_wta_under_per_link_channels():
    scenario = make_comm_mismatch_scenario(10001, 2, 2)
    wta = wta_greedy_joint_multi(scenario, 8, min_cover=False)["worst_pd"]
    nomp = nomp_wta_greedy_joint_multi(scenario, 8)["worst_pd"]
    assert nomp >= wta - 1e-12


def test_per_link_nomp_matches_robust_exact_at_low_budget():
    scenario = make_comm_mismatch_scenario(10001, 2, 2)
    result = nomp_wta_greedy_joint_multi(scenario, 8)
    groups = []
    for owner, deltas, flips, successes in scenario:
        options = enumerate_heterogeneous_robust_power_bit_options(
            owner,
            deltas,
            [(0.0, float(value)) for value in flips],
            [(float(value), 1.0) for value in successes],
            power_levels=np.arange(9, dtype=float),
            bit_options=np.arange(3, dtype=int),
            budget=8,
            grid=16,
        )
        groups.append(pareto_options(options, "robust_pd"))
    exact = exact_joint_power_bit_maxmin(groups, 8)
    assert abs(result["worst_pd"] - exact) < 1e-6


def test_per_link_refinement_reaches_robust_exact_on_hard_scenario():
    for seed in (10002, 10006, 10009):
        scenario = make_comm_mismatch_scenario(seed, 2, 2)
        result = nomp_wta_greedy_joint_multi(scenario, 12)
        groups = []
        for owner, deltas, flips, successes in scenario:
            options = enumerate_heterogeneous_robust_power_bit_options(
                owner,
                deltas,
                [(0.0, float(value)) for value in flips],
                [(float(value), 1.0) for value in successes],
                power_levels=np.arange(13, dtype=float),
                bit_options=np.arange(3, dtype=int),
                budget=12,
                grid=16,
            )
            groups.append(pareto_options(options, "robust_pd"))
        exact = exact_joint_power_bit_maxmin(groups, 12)
        assert abs(result["worst_pd"] - exact) < 1e-6


def test_ucb_nomp_per_link_is_finite_and_close_to_deterministic():
    scenario = make_comm_mismatch_scenario(10001, 2, 2)
    noisy = ucb_wta_greedy_joint_multi(
        scenario,
        8,
        noise_scale=0.2,
        seed=0,
        min_cover=True,
        refine=True,
        max_feedback_rounds=10,
    )
    deterministic = nomp_wta_greedy_joint_multi(scenario, 8)
    assert noisy["steps_used"] <= 100
    assert noisy["feedback_rounds"] <= 10
    assert abs(noisy["worst_pd"] - deterministic["worst_pd"]) < 1e-3


def test_qos_nomp_matches_exact_weighted_maxmin():
    scenario = make_comm_mismatch_scenario(10000, 2, 2)
    floors = [0.30, 0.45]
    weights = [1.0, 1.3]
    result = nomp_wta_greedy_joint_multi(
        scenario, 8, floors=floors, weights=weights
    )
    exact = exact_qos_worst(scenario, 8, floors, weights, 16)
    assert abs(result["qos_worst"] - exact) < 1e-6


def test_ucb_nomp_qos_matches_deterministic():
    scenario = make_comm_mismatch_scenario(10000, 2, 2)
    floors = [0.30, 0.45]
    weights = [1.0, 1.3]
    noisy = ucb_wta_greedy_joint_multi(
        scenario,
        8,
        noise_scale=0.2,
        seed=0,
        min_cover=True,
        refine=True,
        max_feedback_rounds=10,
        floors=floors,
        weights=weights,
    )
    deterministic = nomp_wta_greedy_joint_multi(
        scenario, 8, floors=floors, weights=weights
    )
    assert noisy["feedback_rounds"] <= 10
    assert abs(noisy["qos_worst"] - deterministic["qos_worst"]) < 1e-6
