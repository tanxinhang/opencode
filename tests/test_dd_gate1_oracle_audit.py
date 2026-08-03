import numpy as np

from scripts.run_dd_gate1_oracle_audit import (
    calibrate_frame_thresholds,
    evaluate_scenario,
    match_detections,
)


def test_matching_is_one_to_one_and_counts_false_peaks():
    matched, errors, false = match_detections(
        [(0, 0), (0, 1)], [(0, 0), (3, 4)], (4, 8), 1
    )
    assert matched == 1
    assert errors == [(0, 0)]
    assert false == 1


def test_matching_uses_cyclic_distance():
    matched, errors, false = match_detections(
        [(0, 0)], [(7, 15)], (8, 16), 1
    )
    assert (matched, errors, false) == (1, [(1, 1)], 0)


def test_matching_maximizes_cardinality_before_distance():
    matched, _, false = match_detections(
        [(0, 0), (0, 2)], [(0, 1), (0, 0)], (8, 16), 1
    )
    assert matched == 2
    assert false == 0


def test_oracle_audit_runs_end_to_end_for_one_trial():
    result = evaluate_scenario(
        np.array([1.2, 4.35, 7.1, 10.3]),
        np.array([0.35, 2.18, 4.42, 6.25]),
        trials=1,
        validation_trials=2,
    )
    assert 0.0 <= result["detection_oracle"]["detection_probability"] <= 1.0
    assert len(result["top_detection_assignments"]) == 5


def test_frame_threshold_calibration_controls_empirical_false_alarm():
    rng = np.random.default_rng(20260812)
    dictionary = np.eye(16, dtype=complex)[None, :, :]
    thresholds = calibrate_frame_thresholds(
        dictionary, noise_variance=0.2,
        frame_false_alarm_probability=0.01,
        trials=20_000,
    )
    noise = np.sqrt(0.2 / 2) * (
        rng.standard_normal((50_000, 16))
        + 1j * rng.standard_normal((50_000, 16))
    )
    empirical = np.mean(np.max(np.abs(noise) ** 2, axis=1)
                        >= thresholds[(0,)])
    assert abs(empirical - 0.01) < 0.002
