import numpy as np

from uav_otfs_isac.ris_null_steering import (
    array_power,
    array_power_gradient,
    optimize_null_steering_phases,
    quantized_null_steering_phases,
)
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.ris_upd import upd_ideal_phase
from uav_otfs_isac.scenario import target_geometry


def _config():
    return RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=64,
        aperture_shape=(8, 8),
        weak_target_id=2,
        phase_bits=None,
    )


def test_array_power_gradient_matches_finite_difference():
    config = _config()
    target = target_geometry(0)
    ideal = upd_ideal_phase(config, target)
    phase = ideal + 0.1 * np.sin(np.arange(config.num_elements))
    gradient = array_power_gradient(phase, ideal)
    epsilon = 1e-6
    numeric = np.zeros_like(gradient)
    for index in range(0, config.num_elements, 8):
        plus = phase.copy()
        minus = phase.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (
            array_power(plus, ideal) - array_power(minus, ideal)
        ) / (2.0 * epsilon)
    assert np.allclose(gradient[::8], numeric[::8], atol=1e-4)


def test_null_steering_reduces_interference_power():
    config = _config()
    target = target_geometry(0)
    interference = np.array([60.0, -20.0, 0.0])
    aligned = upd_ideal_phase(config, target)
    optimized = optimize_null_steering_phases(
        config, target, [interference], lambda_=1.0
    )
    aligned_interference = array_power(
        aligned, upd_ideal_phase(config, interference)
    )
    optimized_interference = array_power(
        optimized, upd_ideal_phase(config, interference)
    )
    assert optimized_interference < aligned_interference - 1e-4


def test_quantized_null_steering_reduces_interference():
    config = _config()
    config = RisConfig(
        position=config.position,
        num_elements=config.num_elements,
        aperture_shape=(8, 8),
        weak_target_id=2,
        phase_bits=3,
    )
    target = target_geometry(0)
    interference = np.array([60.0, -20.0, 0.0])
    from uav_otfs_isac.ris_scenario import quantize_phase
    from uav_otfs_isac.ris_upd import upd_ideal_phase
    aligned = quantize_phase(upd_ideal_phase(config, target), 3)
    optimized = quantized_null_steering_phases(
        config, target, [interference], lambda_=1.0
    )
    aligned_interference = array_power(
        aligned, upd_ideal_phase(config, interference)
    )
    optimized_interference = array_power(
        optimized, upd_ideal_phase(config, interference)
    )
    assert optimized_interference < aligned_interference - 1e-4
