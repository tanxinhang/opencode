import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.expected_pd import (
    expected_gaussian_detection_probability,
    expected_pd_greedy_select,
    pd_inflection_condition,
)
from uav_otfs_isac.fusion import optimal_gaussian_detection_probability
from uav_otfs_isac.reliability import with_common_state_erasures


def test_expected_pd_reduces_to_deterministic_pd_without_erasures():
    model = symmetric_diversity_model(
        np.array([1.5, 1.2, 0.9, 0.7]), success_probability=1.0
    )
    scheduled = {0, 1, 3}
    expected = expected_gaussian_detection_probability(
        model, scheduled, 0.05, pd_mode="optimal"
    )
    deterministic = optimal_gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1,
        scheduled, 0.05,
    )
    assert np.isclose(expected, deterministic, rtol=1e-9, atol=1e-9)


def test_expected_pd_is_set_monotone_at_operating_points():
    rng = np.random.default_rng(20260805)
    for _ in range(8):
        model = symmetric_diversity_model(
            np.array([1.6, 1.3, 1.1, 0.9])
        )
        correlated = with_common_state_erasures([model], 0.7)[0]
        base = {0}
        base_value = expected_gaussian_detection_probability(
            correlated, base, 0.05
        )
        if base_value < 0.5:
            continue
        for candidate in range(1, correlated.num_uavs):
            new_value = expected_gaussian_detection_probability(
                correlated, base | {candidate}, 0.05
            )
            assert new_value >= base_value - 1e-9


def test_expected_pd_greedy_respects_budget():
    rng = np.random.default_rng(20260805)
    models = []
    for _ in range(3):
        model = symmetric_diversity_model(
            np.array([1.6, 1.3, 1.1, 0.9])
        )
        models.append(with_common_state_erasures([model], 0.6)[0])
    result = expected_pd_greedy_select(
        models, budget_bits=8, false_alarm_rate=0.05
    )
    assert result.used_bits <= 8
    for q, model in enumerate(models):
        assert model.owner in result.scheduled[q]
    assert np.all(result.expected_pd >= 0.0)
    assert np.all(result.expected_pd <= 1.0 + 1e-12)


def test_saa_matches_exact_for_small_set():
    model = symmetric_diversity_model(np.array([1.5, 1.2, 0.9, 0.7]))
    scheduled = {0, 1, 2}
    rng = np.random.default_rng(7)
    exact = expected_gaussian_detection_probability(
        model, scheduled, 0.05, max_exact_reports=14
    )
    sampled = expected_gaussian_detection_probability(
        model, scheduled, 0.05, max_exact_reports=0,
        rng=rng, samples=4096,
    )
    assert abs(sampled - exact) < 2e-3


def test_pd_inflection_condition_boundaries():
    assert pd_inflection_condition(0.0, 1.0, 0.05)
    assert pd_inflection_condition(10.0, 0.5, 0.05)
    assert not pd_inflection_condition(0.5, 0.5, 0.05)
