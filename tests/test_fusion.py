import numpy as np

from uav_otfs_isac.fusion import (
    conditional_marginal_deflection,
    optimal_deflection,
    optimal_weights,
)


def test_schur_gain_matches_direct_difference():
    delta = np.array([1.0, 0.8, 0.4])
    sigma = np.array([[1.0, 0.4, 0.1], [0.4, 1.2, 0.25], [0.1, 0.25, 0.9]])
    selected = {0, 1}
    direct = optimal_deflection(delta, sigma, {0, 1, 2}) - optimal_deflection(delta, sigma, selected)
    schur = conditional_marginal_deflection(delta, sigma, selected, 2)
    assert np.isclose(direct, schur, rtol=1e-10, atol=1e-10)
    assert schur >= 0


def test_optimal_weights_have_unit_null_variance():
    delta = np.array([1.0, 0.3])
    sigma = np.array([[2.0, 0.2], [0.2, 1.0]])
    weights = optimal_weights(delta, sigma, {0, 1})
    assert np.isclose(weights @ sigma @ weights, 1.0)


def test_correlation_reduces_redundant_gain():
    delta = np.array([1.0, 1.0, 0.6])
    sigma = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]])
    redundant = conditional_marginal_deflection(delta, sigma, {0}, 1)
    independent = conditional_marginal_deflection(delta, sigma, {0}, 2)
    assert independent > redundant

