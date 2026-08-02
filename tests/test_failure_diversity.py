import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.reliability import with_grouped_common_state_erasures
from uav_otfs_isac.risk import gaussian_pd_loss_distribution, optimize_chance_constrained_portfolio


GROUPS = np.array([-1, 0, 0, 1, 1])
SAME = {0, 1, 2}
CROSS = {0, 1, 3}


def pd(model, received):
    return gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1, received, 0.05
    )


def violation(model, scheduled, threshold):
    return gaussian_pd_loss_distribution(model, scheduled, threshold, 0.05).violation_probability()


def test_c0_equal_quality_one_of_two_prefers_cross_group():
    independent = symmetric_diversity_model(success_probability=0.6)
    correlated = with_grouped_common_state_erasures([independent], 1.0, GROUPS)[0]
    threshold = (pd(independent, {0}) + pd(independent, {0, 1})) / 2
    assert np.isclose(violation(independent, SAME, threshold), violation(independent, CROSS, threshold))
    assert violation(correlated, CROSS, threshold) < violation(correlated, SAME, threshold)
    result = optimize_chance_constrained_portfolio(
        [correlated], 2, [threshold], [1.0], [0.0],
        quality_mode="gaussian_pd", false_alarm_rate=0.05,
    )
    selected = result.portfolio.selection.scheduled[0]
    assert len((set(selected) - {0}) & {1, 2}) == 1
    assert len((set(selected) - {0}) & {3, 4}) == 1


def test_c1_two_of_two_prefers_positive_within_group_dependence():
    independent = symmetric_diversity_model(success_probability=0.6)
    correlated = with_grouped_common_state_erasures([independent], 1.0, GROUPS)[0]
    threshold = (pd(independent, {0, 1}) + pd(independent, {0, 1, 2})) / 2
    assert violation(correlated, SAME, threshold) < violation(correlated, CROSS, threshold)


def test_c2_quality_gap_causes_diversity_to_quality_switch():
    def selected(delta):
        base = symmetric_diversity_model(
            np.array([1 + delta, 1 + delta, 1 - delta, 1 - delta]), 0.6
        )
        truth = with_grouped_common_state_erasures([base], 1.0, GROUPS)[0]
        reference = symmetric_diversity_model(success_probability=0.6)
        threshold = (pd(reference, {0}) + pd(reference, {0, 1})) / 2
        result = optimize_chance_constrained_portfolio(
            [truth], 2, [threshold], [1.0], [0.0],
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )
        return set(result.portfolio.selection.scheduled[0])
    low_gap = selected(0.3); high_gap = selected(0.4)
    assert len((low_gap - {0}) & {3, 4}) == 1
    assert (high_gap - {0}).issubset({1, 2})
