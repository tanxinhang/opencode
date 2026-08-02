import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.reliability import with_grouped_common_state_erasures
from uav_otfs_isac.risk import attribute_failure_diversity_headroom


GROUPS = np.array([-1, 0, 0, 1, 1])


def pd(model, received):
    return gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1, received, 0.05
    )


def test_attribution_identifies_diversifiable_substitute_and_headroom():
    independent = symmetric_diversity_model(success_probability=0.6)
    truth = with_grouped_common_state_erasures([independent], 1.0, GROUPS)[0]
    threshold = (pd(independent, {0}) + pd(independent, {0, 1})) / 2
    result = attribute_failure_diversity_headroom(
        independent, truth, {0, 1, 2}, {0, 1, 3}, 2, threshold, GROUPS
    )
    assert result.minimum_successful_reports == 1
    assert result.supporting_failure_domains == 2
    assert result.classification == "diversifiable_substitute"
    assert np.isclose(result.recoverable_headroom, 0.16)
    assert np.isclose(result.headroom_use_ratio, 1.0)
    assert result.oracle_all_scheduled_gap >= -1e-12


def test_attribution_marks_complementary_threshold():
    independent = symmetric_diversity_model(success_probability=0.6)
    truth = with_grouped_common_state_erasures([independent], 1.0, GROUPS)[0]
    threshold = (pd(independent, {0, 1}) + pd(independent, {0, 1, 2})) / 2
    result = attribute_failure_diversity_headroom(
        independent, truth, {0, 1, 3}, {0, 1, 2}, 2, threshold, GROUPS
    )
    assert result.minimum_successful_reports == 2
    assert result.supporting_failure_domains == 0
    assert result.classification == "complementary_evidence"
