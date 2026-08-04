import numpy as np

from uav_otfs_isac.report_channel_calibration import (
    exact_received_moments,
    relative_errors,
    simulate_received_moments,
)
from uav_otfs_isac.reporting import quantizer_from_gaussian_range


def test_exact_and_monte_carlo_moments_agree():
    mu = np.asarray([0.5, 1.0, 1.5, 2.0])
    covariance = np.eye(4)
    covariance[0, 1] = covariance[1, 0] = 0.5
    edges, values = quantizer_from_gaussian_range(
        mu, covariance, mu + 1.0, covariance, 2
    )
    success = np.full(4, 0.9)
    exact = exact_received_moments(
        mu, covariance, edges, values, 2, 0.08, success
    )
    simulated = simulate_received_moments(
        mu, covariance, edges, values, 2, 0.08, success,
        20_000, seed=11,
    )
    errors = relative_errors(exact, simulated)
    assert errors["mean_relative_error"] < 0.06
    assert errors["covariance_relative_error"] < 0.10


def test_perfect_channel_preserves_quantized_moments():
    mu = np.asarray([0.5, 1.0])
    covariance = np.eye(2)
    edges, values = quantizer_from_gaussian_range(
        mu, covariance, mu + 1.0, covariance, 2
    )
    success = np.ones(2)
    exact = exact_received_moments(
        mu, covariance, edges, values, 2, 0.0, success
    )
    simulated = simulate_received_moments(
        mu, covariance, edges, values, 2, 0.0, success,
        10_000, seed=13,
    )
    errors = relative_errors(exact, simulated)
    assert errors["mean_relative_error"] < 0.05
    assert errors["covariance_relative_error"] < 0.10
