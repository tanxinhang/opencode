import numpy as np

from uav_otfs_isac.covariance_aware_ris import (
    covariance_aware_phase,
    direction_error_std,
    expected_array_gain_squared,
)
from uav_otfs_isac.ris_scenario import RisConfig, ris_beam_phase


def _config():
    return RisConfig(
        position=np.array([0.0, 0.0, 0.0]),
        num_elements=16,
    )


def test_covariance_aware_phase_improves_expected_gain():
    config = _config()
    target = np.array([60.0, 8.0, 0.0])
    sigma_direction = 0.04
    mmse_phase = ris_beam_phase(target, config)
    robust_phase = covariance_aware_phase(
        target, config, sigma_direction, iterations=300
    )
    mmse_gain = expected_array_gain_squared(
        mmse_phase, target[0] / np.linalg.norm(target),
        sigma_direction, config,
    )
    robust_gain = expected_array_gain_squared(
        robust_phase, target[0] / np.linalg.norm(target),
        sigma_direction, config,
    )
    assert robust_gain >= mmse_gain - 1e-9
    assert not np.allclose(robust_phase, mmse_phase)


def test_direction_error_std_scales_with_covariance():
    assert np.isclose(
        direction_error_std(0.36, 2.0, 50.0),
        0.6 * 2.0 / 50.0,
    )
