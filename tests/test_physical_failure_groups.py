import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.reliability import (
    grouped_failure_correlation,
    physical_failure_groups,
    with_grouped_common_state_erasures,
)


def positions():
    return np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.2, 0.0],
        [-2.0, 0.0, 0.0],
        [-3.0, -0.2, 0.0],
    ])


def test_physical_groups_are_target_aware_and_geometry_only():
    model = symmetric_diversity_model()
    groups = physical_failure_groups(
        [model], positions(), "owner_angle_path", 2
    )[0]
    assert groups[model.owner] == -1
    assert len(set(groups[1:])) == 2
    assert groups[1] == groups[2]
    assert groups[3] == groups[4]


def test_midpoint_and_position_clustering_are_equivalent_for_straight_links():
    model = symmetric_diversity_model()
    formation = physical_failure_groups(
        [model], positions(), "formation_position", 2
    )[0]
    midpoint = physical_failure_groups(
        [model], positions(), "link_midpoint", 2
    )[0]
    assert np.array_equal(formation, midpoint)


def test_grouped_correlation_gate_separates_within_and_between_domains():
    model = symmetric_diversity_model()
    groups = physical_failure_groups(
        [model], positions(), "owner_angle_path", 2
    )[0]
    truth = with_grouped_common_state_erasures([model], 1.0, groups)[0]
    within, between = grouped_failure_correlation(truth, groups)
    assert within > between + 1e-6
    assert np.allclose(
        truth.pattern_probabilities @ truth.reception_patterns,
        truth.success_prob,
        atol=1e-12,
    )
