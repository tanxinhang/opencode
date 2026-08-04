import numpy as np

from scripts.run_multistatic_configuration_stepdown_gate import (
    fit_configuration_threshold, normal_component_maxima, proposed_summary,
    unlabeled_frames,
)


def test_normal_component_maximum_excludes_true_collision_components():
    frames = [[
        {"gain": 100.0, "is_collision": True},
        {"gain": 3.0, "is_collision": False},
        {"gain": 2.0, "is_collision": False},
    ], [{"gain": 4.0, "is_collision": False}]]
    assert np.array_equal(normal_component_maxima(frames), (3.0, 4.0))


def test_configuration_threshold_uses_finite_sample_rule():
    frames = [[{"gain": float(index), "is_collision": False}]
              for index in range(100)]
    threshold, count = fit_configuration_threshold(frames, 0.01)
    assert threshold == 99.0
    assert count == 100


def test_unlabeled_frames_remove_offline_truth_fields():
    frames = [[{"gain": 3.0, "is_collision": True, "target_count": 2}]]
    assert unlabeled_frames(frames) == [[(3.0, 0, 0)]]


def test_proposed_summary_reports_core_metrics():
    from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator
    calibrator = IsotonicProbabilityCalibrator(
        np.asarray([1.0]), np.asarray([0.9])
    )
    result = proposed_summary(
        "separated", 1, 7, calibrator, (float("inf"),) * 4, 4
    )
    assert result["trials"] == 1
    assert "position_set_exact_15m" in result
