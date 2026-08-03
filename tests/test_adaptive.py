import numpy as np

from uav_otfs_isac.adaptive import (
    ack_joint_distribution,
    complementary_adaptive_success,
    complementary_static_success,
    controlled_two_report_oracles,
    optimize_single_target_two_stage_oracle,
    two_stage_hidden_state_model,
    gaussian_pd_threshold_by_mask,
)
from uav_otfs_isac.controlled import symmetric_diversity_model


def test_ack_repair_has_strict_complementary_report_advantage():
    for p in (0.2, 0.5, 0.8):
        static = complementary_static_success(p)
        adaptive = complementary_adaptive_success(p)
        assert np.isclose(adaptive - static, p**2 * (1.0 - p))
        assert adaptive > static


def test_complementary_report_formulas_include_probability_boundaries():
    for p in (0.0, 1.0):
        assert np.isclose(complementary_static_success(p), p)
        assert np.isclose(complementary_adaptive_success(p), p)


def test_complementary_report_formulas_reject_invalid_probability():
    for p in (-0.1, 1.1):
        with np.testing.assert_raises(ValueError):
            complementary_static_success(p)


def test_ack_joint_posterior_probability_audit():
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    conditional = np.array([
        [0.1, 0.2],
        [0.1, 0.8],
        [0.9, 0.2],
        [0.9, 0.8],
    ])
    histories, history_probability, posterior = ack_joint_distribution(
        prior, conditional
    )
    assert histories == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert np.isclose(history_probability.sum(), 1.0)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert np.allclose(history_probability @ posterior, prior)


def test_multiple_failures_jointly_update_shared_state():
    prior = np.array([0.5, 0.5])
    conditional = np.array([[0.2, 0.2], [0.8, 0.8]])
    histories, _, posterior = ack_joint_distribution(prior, conditional)
    failure = histories.index((0, 0))
    success = histories.index((1, 1))
    assert posterior[failure, 0] > prior[0]
    assert posterior[success, 1] > prior[1]


def test_controlled_oracle_recovers_independent_closed_forms():
    p = 0.6
    result = controlled_two_report_oracles(
        [1.0], [[p, p]], [[p, p]]
    )
    assert np.isclose(result["static"].success_probability,
                      complementary_static_success(p))
    assert np.isclose(result["adaptive"].success_probability,
                      complementary_adaptive_success(p))
    assert result["adaptive"].success_probability > result["static"].success_probability
    assert result["adaptive"].expected_transmissions < 3.0


def test_persistent_hidden_state_preserves_nonanticipative_ordering():
    prior = [0.25, 0.25, 0.25, 0.25]
    stage_one = [
        [0.25, 0.35], [0.25, 0.85],
        [0.75, 0.35], [0.75, 0.85],
    ]
    repair = [
        [0.20, 0.30], [0.70, 0.80],
        [0.20, 0.30], [0.70, 0.80],
    ]
    result = controlled_two_report_oracles(prior, stage_one, repair)
    assert result["static"].success_probability < result["adaptive"].success_probability
    assert result["adaptive"].success_probability <= (
        result["clairvoyant"].success_probability + 1e-12
    )
    # One action per ACK history: no hidden-state index exists in the policy.
    assert len(result["adaptive"].policy) == 4


def test_random_controlled_oracle_never_beats_clairvoyant():
    rng = np.random.default_rng(20260803)
    for _ in range(20):
        prior = rng.dirichlet(np.ones(4))
        stage_one = rng.uniform(0.1, 0.9, size=(4, 2))
        repair = rng.uniform(0.1, 0.9, size=(4, 2))
        result = controlled_two_report_oracles(prior, stage_one, repair)
        assert result["adaptive"].success_probability + 1e-12 >= (
            result["static"].success_probability
        )
        assert result["adaptive"].success_probability <= (
            result["clairvoyant"].success_probability + 1e-12
        )
        assert 2.0 <= result["adaptive"].expected_transmissions <= 3.0


def test_controlled_oracle_allows_zero_probability_hidden_states():
    result = controlled_two_report_oracles(
        [1.0, 0.0], [[0.6, 0.6], [0.1, 0.9]],
        [[0.6, 0.6], [0.2, 0.8]],
    )
    assert np.isclose(
        result["adaptive"].success_probability,
        complementary_adaptive_success(0.6),
    )


def test_general_threshold_oracle_recovers_two_report_closed_forms():
    p = 0.6
    success_by_mask = np.array([False, False, False, True])
    result = optimize_single_target_two_stage_oracle(
        [1.0], np.full((1, 2, 2), p), [1, 1],
        np.ones((2, 2), dtype=bool), success_by_mask,
        total_budget_bits=3, first_stage_budget_bits=2,
        domain_capacities=[2, 2],
    )
    assert np.isclose(result["static"].success_probability,
                      complementary_static_success(p))
    assert np.isclose(result["adaptive"].success_probability,
                      complementary_adaptive_success(p))
    assert result["adaptive"].success_probability <= (
        result["clairvoyant"].success_probability + 1e-12
    )
    assert result["adaptive"].expected_bits < result["static"].expected_bits


def test_general_threshold_oracle_without_repair_reduces_to_static_selection():
    success_by_mask = np.array([False, True, True, True])
    result = optimize_single_target_two_stage_oracle(
        [1.0], np.array([[[0.7, 0.7], [0.5, 0.5]]]), [1, 1],
        np.ones((2, 2), dtype=bool), success_by_mask,
        total_budget_bits=2, first_stage_budget_bits=2,
        domain_capacities=[1, 1],
    )
    assert np.isclose(result["adaptive"].success_probability,
                      result["selection"].success_probability)
    assert np.isclose(result["adaptive"].expected_bits,
                      result["selection"].expected_bits)
    assert result["static"].success_probability >= (
        result["selection"].success_probability
    )


def test_general_oracle_policy_obeys_every_ack_branch_capacity():
    threshold = np.array([
        False, False, False, True,
        False, True, True, True,
    ])
    result = optimize_single_target_two_stage_oracle(
        [0.5, 0.5],
        np.array([
            [[0.3, 0.6], [0.4, 0.7], [0.5, 0.8]],
            [[0.7, 0.4], [0.8, 0.5], [0.6, 0.3]],
        ]),
        [1, 1, 1], np.ones((3, 2), dtype=bool), threshold,
        total_budget_bits=4, first_stage_budget_bits=2,
        domain_capacities=[2, 2],
    )["adaptive"]
    first_domain = [0, 0]
    for _, domain in result.first_stage_actions:
        first_domain[domain] += 1
    for plan in result.policy_by_ack:
        assert len(result.first_stage_actions) + len(plan) <= 4
        used = first_domain.copy()
        for _, domain in plan:
            used[domain] += 1
        assert used[0] <= 2 and used[1] <= 2


def test_two_stage_hidden_state_model_preserves_report_marginals():
    model = symmetric_diversity_model(success_probability=0.6)
    paths = np.array([-1, 0, 0, 1, 1])
    reporters, _, prior, success = two_stage_hidden_state_model(
        model, paths, 0.5, 0.5
    )
    assert reporters == [1, 2, 3, 4]
    assert np.allclose(prior @ success[:, :, 0], 0.6)
    assert np.allclose(prior @ success[:, :, 1], 0.6)


def test_gaussian_pd_threshold_mask_is_monotone_for_symmetric_evidence():
    model = symmetric_diversity_model(success_probability=0.6)
    reporters = [1, 2, 3, 4]
    table = gaussian_pd_threshold_by_mask(model, reporters, 0.3, 0.05)
    for mask in range(table.size):
        for bit in range(len(reporters)):
            if table[mask]:
                assert table[mask | (1 << bit)]


def test_general_oracle_uses_zero_bits_when_owner_already_meets_threshold():
    result = optimize_single_target_two_stage_oracle(
        [1.0], np.full((1, 2, 2), 0.6), [1, 1],
        np.ones((2, 2), dtype=bool), np.ones(4, dtype=bool),
        total_budget_bits=3, first_stage_budget_bits=2,
        domain_capacities=[2, 2],
    )
    for value in result.values():
        assert np.isclose(value.success_probability, 1.0)
        assert np.isclose(value.expected_bits, 0.0)
