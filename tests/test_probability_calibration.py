import numpy as np

from uav_otfs_isac.probability_calibration import (
    ExcessPeakConformalNull,
    IsotonicProbabilityCalibrator,
    fit_isotonic_probability,
    probability_metrics,
)


def test_pav_is_monotone_and_order_invariant():
    scores = np.asarray([0.1, 0.3, 0.2, 0.4, 0.3])
    labels = np.asarray([0, 1, 0, 1, 0])
    first = fit_isotonic_probability(scores, labels)
    reverse = fit_isotonic_probability(scores[::-1], labels[::-1])
    grid = np.linspace(0.0, 0.5, 21)
    assert np.all(np.diff(first.predict(grid)) >= 0.0)
    assert np.allclose(first.predict(grid), reverse.predict(grid))


def test_pav_pools_violating_adjacent_blocks():
    model = fit_isotonic_probability([0.1, 0.2, 0.3], [0, 1, 0])
    predictions = model.predict([0.1, 0.2, 0.3])
    assert np.allclose(predictions, (1e-4, 0.5, 0.5))
    assert len(model.thresholds) == 2


def test_calibrator_round_trip_and_probability_metrics():
    model = fit_isotonic_probability([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    restored = IsotonicProbabilityCalibrator.from_dict(model.to_dict())
    predictions = restored.predict([0.1, 0.5, 0.9])
    assert np.all((predictions > 0) & (predictions < 1))
    metrics = probability_metrics(predictions, [0, 0, 1], bins=5)
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["ece"] <= 1


def test_excess_peak_conformal_pvalue_is_monotone_and_smoothed():
    model = ExcessPeakConformalNull({
        0: np.asarray([1.0, 2.0, 3.0]),
        1: np.asarray([2.0, 4.0, 6.0]),
        2: np.asarray([5.0, 10.0, 15.0]),
    })
    assert model.p_value(2.0, 8, 8) > model.p_value(4.0, 8, 8)
    assert model.p_value(100.0, 10, 8) == 0.25
    assert ExcessPeakConformalNull.stratum(12, 8) == 2
