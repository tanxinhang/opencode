import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.replication import (
    optimize_replication_chance_portfolio,
    replicated_reception_model,
)
from uav_otfs_isac.reliability import with_grouped_common_state_erasures
from uav_otfs_isac.risk import optimize_chance_constrained_portfolio


DOMAINS = np.array([-1, 0, 0, 1, 1])


def pd(model, received):
    return gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1, received, 0.05
    )


def test_cross_domain_copy_preserves_per_copy_link_model_and_improves_effective_success():
    model = symmetric_diversity_model(success_probability=0.6)
    counts = np.array([0, 2, 0, 0, 0])
    repaired = replicated_reception_model(model, counts, DOMAINS, 1.0)
    assert np.isclose(repaired.success_prob[1], 1.0 - (1.0 - 0.6) ** 2)
    assert np.allclose(repaired.success_prob[[2, 3, 4]], 0.0)


def test_c5_replication_beats_selection_when_only_same_domain_reports_clear_threshold():
    model = symmetric_diversity_model(
        np.array([1.4, 1.4, 0.6, 0.6]), success_probability=0.6
    )
    reference = symmetric_diversity_model(success_probability=0.6)
    threshold = (pd(reference, {0}) + pd(reference, {0, 1})) / 2
    truth = with_grouped_common_state_erasures([model], 1.0, DOMAINS)[0]
    selection = optimize_chance_constrained_portfolio(
        [truth], 2, [threshold], [1.0], [0.0],
        quality_mode="gaussian_pd", false_alarm_rate=0.05,
    )
    result = optimize_replication_chance_portfolio(
        [model], 2, [threshold], [1.0], [0.0], [DOMAINS], 1.0
    )
    counts = result.copy_counts[0]
    assert sum(counts) == 2
    assert max(counts[1], counts[2]) == 2
    assert np.isclose(
        selection.portfolio.violation_probability_per_target[0], 0.32
    )
    assert np.isclose(result.violation_probability_per_target[0], 0.16)
