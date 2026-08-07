"""Prediction-covariance-aware RIS phase design.

The ideal phase is the MMSE prediction of the target direction.  Under a
Gaussian direction error with standard deviation ``sigma_direction``, the
normalized expected squared array gain is

``G(phi) = (1/N^2) sum_{n,m} cos(phi_n - phi_m) exp(-0.5 (k d (n-m) sigma)^2)``,

which is a real Toeplitz quadratic form.  Starting from the MMSE phase, a
projected-gradient ascent maximizes this smooth objective locally, so the
robust phase is never below its MMSE start in expected squared gain.
"""

from __future__ import annotations

import numpy as np

from .ris_scenario import PROPAGATION_SPEED, RisConfig, ris_beam_phase


def direction_error_std(
    error_covariance_scale: float,
    target_sigma: float,
    range_to_ris: float,
) -> float:
    """Approximate angular error standard deviation for small displacements."""
    if error_covariance_scale < 0.0:
        raise ValueError("error_covariance_scale must be nonnegative")
    if target_sigma < 0.0 or range_to_ris <= 0.0:
        raise ValueError("target_sigma and range_to_ris must be positive")
    return float(
        np.sqrt(error_covariance_scale) * target_sigma / range_to_ris
    )


def _phase_gradient(
    phase: np.ndarray,
    direction_x: float,
    sigma_direction: float,
    config: RisConfig,
) -> np.ndarray:
    ideal = _ideal_phase_for_direction(direction_x, config)
    residual = (phase - ideal + np.pi) % (2.0 * np.pi) - np.pi
    wavelength = PROPAGATION_SPEED / config.carrier_hz
    wavenumber = 2.0 * np.pi / wavelength
    spacing = config.element_spacing_lambda * wavelength
    n = np.arange(config.num_elements, dtype=float)
    gradient = np.zeros_like(phase)
    for index in range(config.num_elements):
        diff = n - index
        weights = np.exp(
            -0.5 * (wavenumber * spacing * diff * sigma_direction) ** 2
        )
        gradient[index] = float(np.sum(
            np.sin(residual - residual[index]) * weights
        ))
    return 2.0 * gradient


def expected_array_gain_squared(
    phase: np.ndarray,
    direction_x: float,
    sigma_direction: float,
    config: RisConfig,
) -> float:
    """Normalized expected squared array gain under Gaussian direction error."""
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (config.num_elements,):
        raise ValueError("phase must have one entry per RIS element")
    ideal = _ideal_phase_for_direction(direction_x, config)
    residual = (phase - ideal + np.pi) % (2.0 * np.pi) - np.pi
    wavelength = PROPAGATION_SPEED / config.carrier_hz
    wavenumber = 2.0 * np.pi / wavelength
    spacing = config.element_spacing_lambda * wavelength
    n = np.arange(config.num_elements, dtype=float)
    value = 0.0
    for index in range(config.num_elements):
        diff = n - index
        weights = np.exp(
            -0.5 * (wavenumber * spacing * diff * sigma_direction) ** 2
        )
        value += float(np.sum(
            np.cos(residual - residual[index]) * weights
        ))
    return value / config.num_elements**2


def covariance_aware_phase(
    predicted_target: np.ndarray,
    config: RisConfig,
    sigma_direction: float,
    *,
    iterations: int = 400,
    step_size: float = 0.05,
) -> np.ndarray:
    """Optimize expected squared array gain from the MMSE-predicted phase."""
    target = np.asarray(predicted_target, dtype=float)
    if target.shape != config.position.shape:
        raise ValueError("target and RIS positions must share dimension")
    direction = target - config.position
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("target cannot coincide with the RIS")
    direction_x = float(direction[0] / norm)
    phase = ris_beam_phase(target, config).astype(float)
    for _ in range(iterations):
        gradient = _phase_gradient(
            phase, direction_x, sigma_direction, config
        )
        update = np.clip(gradient, -0.25, 0.25)
        phase = (phase + step_size * update) % (2.0 * np.pi)
    return phase


def _ideal_phase_for_direction(
    direction_x: float,
    config: RisConfig,
) -> np.ndarray:
    """Ideal RIS phase for a unit direction projected on the array axis."""
    if config.position.size == 2:
        synthetic = config.position + np.array([direction_x, 0.0]) * 1e3
    elif config.position.size == 3:
        synthetic = config.position + np.array([direction_x, 0.0, 0.0]) * 1e3
    else:
        raise ValueError("RIS position must be 2-D or 3-D")
    return ris_beam_phase(synthetic, config)
