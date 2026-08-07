import numpy as np

from uav_otfs_isac.ris_optimization import (
    projected_gradient_shared_phase,
    shared_array_power_and_gradient,
    shared_phase_gain_matrix,
)
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.scenario import target_geometry


def _config():
    return RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=64,
        weak_target_id=2,
    )


def test_array_power_gradient_matches_finite_difference():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    u = 0.37
    gains, gradients = shared_array_power_and_gradient(config, targets, u)
    epsilon = 1e-6
    gains_plus, _ = shared_array_power_and_gradient(config, targets, u + epsilon)
    gains_minus, _ = shared_array_power_and_gradient(config, targets, u - epsilon)
    numeric = (gains_plus - gains_minus) / (2.0 * epsilon)
    assert np.allclose(gradients, numeric, atol=1e-4)


def test_projected_gradient_improves_worst_array_power():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    initial_gains, _ = shared_array_power_and_gradient(config, targets, 0.0)
    result = projected_gradient_shared_phase(
        config, targets, surrogate="worst", max_steps=80
    )
    optimized_gains, _ = shared_array_power_and_gradient(
        config, targets, result["steering_cosine"]
    )
    assert np.min(optimized_gains) > 1.1 * np.min(initial_gains)


def test_shared_phase_gain_matrix_shape_and_lower_bound():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    transmitters = np.array([[180.0, 0.0, 100.0], [90.0, 180.0, 100.0]])
    receiver = np.array([0.0, 0.0, 0.0])
    phase = np.zeros(config.num_elements)
    gain = shared_phase_gain_matrix(
        config, transmitters, targets, receiver,
        aperture_scale=1e-2, phase=phase,
    )
    assert gain.shape == (3, 2)
    assert np.all(gain >= 1.0)
