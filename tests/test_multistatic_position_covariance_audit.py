import numpy as np

from scripts.run_multistatic_position_covariance_audit import (
    bistatic_position_covariance,
)
from uav_otfs_isac.multistatic_association import position_from_angle_range


def test_bistatic_covariance_is_symmetric_positive_definite():
    covariance = bistatic_position_covariance(
        (350.0, 0.0), (0.0, 0.0), 500.0, 1.0
    )
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_bistatic_covariance_grows_with_measurement_noise():
    low = bistatic_position_covariance(
        (350.0, 0.0), (0.0, 0.0), 500.0, 1.0,
        range_sigma_m=1.0, angle_sigma_rad=0.005,
    )
    high = bistatic_position_covariance(
        (350.0, 0.0), (0.0, 0.0), 500.0, 1.0,
        range_sigma_m=2.0, angle_sigma_rad=0.01,
    )
    assert np.all(np.linalg.eigvalsh(high - low) >= -1e-12)


def test_bistatic_covariance_matches_finite_difference_jacobian():
    transmitter = np.asarray((350.0, -40.0))
    receiver = np.asarray((0.0, 0.0))
    rho, theta = 510.0, 1.1
    range_sigma, angle_sigma = 1.5, np.deg2rad(0.4)
    steps = (1e-4, 1e-7)
    columns = []
    for index, step in enumerate(steps):
        plus = [rho, theta]
        minus = [rho, theta]
        plus[index] += step
        minus[index] -= step
        columns.append((
            position_from_angle_range(
                transmitter, receiver, plus[1], plus[0]
            ) - position_from_angle_range(
                transmitter, receiver, minus[1], minus[0]
            )
        ) / (2.0 * step))
    numerical_jacobian = np.column_stack(columns)
    expected = numerical_jacobian @ np.diag((
        range_sigma ** 2, angle_sigma ** 2,
    )) @ numerical_jacobian.T
    actual = bistatic_position_covariance(
        transmitter, receiver, rho, theta,
        range_sigma_m=range_sigma, angle_sigma_rad=angle_sigma,
        eigenvalue_floor_m2=0.0,
    )
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-8)
