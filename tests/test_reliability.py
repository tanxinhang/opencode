from dataclasses import replace

import numpy as np

from uav_otfs_isac.config import ExperimentConfig
from uav_otfs_isac.reliability import (
    common_state_pattern_distribution,
    mean_off_diagonal_failure_correlation,
    with_common_state_erasures,
    grouped_common_state_parameters,
    with_grouped_common_state_erasures,
)
from uav_otfs_isac.risk import gaussian_pd_loss_distribution
from uav_otfs_isac.scenario import build_models


def test_common_state_distribution_preserves_marginals_and_adds_dependence():
    marginal = np.array([1.0, 0.8, 0.6, 0.4])
    patterns, probabilities = common_state_pattern_distribution(marginal, owner=0, strength=0.7)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.allclose(probabilities @ patterns, marginal)
    failures = 1.0 - patterns[:, 1:]
    mean = probabilities @ failures
    covariance = ((failures - mean) * probabilities[:, None]).T @ (failures - mean)
    assert covariance[0, 1] > 0.0
    assert covariance[0, 2] > 0.0


def test_zero_strength_matches_independent_loss_distribution():
    cfg = ExperimentConfig(num_uavs=4, num_targets=1, owners=(0,), target_present=(True,),
                           qos_min_deflection=(3.0,), qos_weights=(1.0,), performance_weights=(1.0,))
    independent = build_models(cfg, np.random.default_rng(12))[0]
    zero_common = with_common_state_erasures([independent], 0.0)[0]
    scheduled = set(range(independent.num_uavs))
    left = gaussian_pd_loss_distribution(independent, scheduled, 0.7, 0.05)
    right = gaussian_pd_loss_distribution(zero_common, scheduled, 0.7, 0.05)
    for value in np.unique(np.concatenate([left.values, right.values])):
        left_probability = left.probabilities[np.isclose(left.values, value)].sum()
        right_probability = right.probabilities[np.isclose(right.values, value)].sum()
        assert np.isclose(left_probability, right_probability)


def test_failure_correlation_increases_with_common_state_strength():
    cfg = ExperimentConfig(num_uavs=4, num_targets=1, owners=(0,), target_present=(True,),
                           qos_min_deflection=(3.0,), qos_weights=(1.0,), performance_weights=(1.0,))
    model = build_models(cfg, np.random.default_rng(13))[0]
    weak = with_common_state_erasures([model], 0.2)[0]
    strong = with_common_state_erasures([model], 0.8)[0]
    assert mean_off_diagonal_failure_correlation(strong) > mean_off_diagonal_failure_correlation(weak) > 0


def test_compact_common_state_enumeration_matches_explicit_patterns():
    cfg = ExperimentConfig(num_uavs=4, num_targets=1, owners=(0,), target_present=(True,),
                           qos_min_deflection=(3.0,), qos_weights=(1.0,), performance_weights=(1.0,))
    model = with_common_state_erasures(
        build_models(cfg, np.random.default_rng(14)), 0.7
    )[0]
    explicit = replace(
        model,
        reception_state_probabilities=None,
        conditional_success_probabilities=None,
    )
    scheduled = {0, 1, 3}
    compact_loss = gaussian_pd_loss_distribution(model, scheduled, 0.7, 0.05)
    explicit_loss = gaussian_pd_loss_distribution(explicit, scheduled, 0.7, 0.05)
    for value in np.unique(np.concatenate([compact_loss.values, explicit_loss.values])):
        compact_probability = compact_loss.probabilities[np.isclose(compact_loss.values, value)].sum()
        explicit_probability = explicit_loss.probabilities[np.isclose(explicit_loss.values, value)].sum()
        assert np.isclose(compact_probability, explicit_probability)


def test_grouped_common_states_correlate_within_but_not_across_groups():
    marginal = np.array([1.0, 0.8, 0.7, 0.6, 0.5])
    groups = np.array([0, 0, 1, 0, 1])
    weights, conditional = grouped_common_state_parameters(
        marginal, owner=0, groups=groups, strength=0.8
    )
    assert np.allclose(weights @ conditional, marginal)
    covariance = (conditional - marginal).T @ (
        (conditional - marginal) * weights[:, None]
    )
    assert covariance[1, 3] > 0.0
    assert covariance[2, 4] > 0.0
    assert abs(covariance[1, 2]) < 1e-12


def test_grouped_model_validates_and_preserves_marginals():
    cfg = ExperimentConfig(num_uavs=5, num_targets=1, owners=(0,), target_present=(True,),
                           qos_min_deflection=(3.0,), qos_weights=(1.0,), performance_weights=(1.0,))
    base = build_models(cfg, np.random.default_rng(15))
    grouped = with_grouped_common_state_erasures(
        base, 0.7, np.array([0, 0, 1, 0, 1])
    )[0]
    assert np.allclose(
        grouped.pattern_probabilities @ grouped.reception_patterns,
        grouped.success_prob,
    )
