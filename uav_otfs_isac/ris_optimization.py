"""Continuous shared-phase optimization for the RIS-assisted channel.

The existing G5 chain configures a different phase profile per target, which
is an upper bound that assumes per-target time multiplexing.  This module
optimizes one physical phase profile shared by all targets.  For a 1-D
uniform linear array the profile is parameterized by one steering cosine
``u``, and the squared array gain toward target ``q`` has a closed-form
analytic gradient, so the optimization is a projected gradient ascent.
Quantization is applied at evaluation time by :func:`ris_array_gain`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ris_scenario import (
    PROPAGATION_SPEED,
    RisConfig,
    ris_array_gain,
    ris_physics_gain_matrix,
)


def target_direction_cosines(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
) -> np.ndarray:
    """Project each target direction onto the ULA axis (x in the model)."""
    result = []
    for target in target_positions:
        direction = np.asarray(target, dtype=float) - config.position
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("target cannot coincide with the RIS")
        result.append(float(direction[0] / norm))
    return np.asarray(result, dtype=float)


def shared_array_power_and_gradient(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    steering_cosine: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (squared array gains, gradients wrt steering cosine)."""
    steering_cosine = float(np.clip(steering_cosine, -1.0, 1.0))
    target_cosines = target_direction_cosines(config, target_positions)
    wavelength = PROPAGATION_SPEED / config.carrier_hz
    spacing = config.element_spacing_lambda * wavelength
    wavenumber = 2.0 * np.pi / wavelength
    elements = np.arange(config.num_elements, dtype=float)
    phase_scale = wavenumber * spacing
    gains = np.zeros(target_cosines.size, dtype=float)
    gradients = np.zeros_like(gains)
    for q, target_cosine in enumerate(target_cosines):
        phase_difference = -phase_scale * elements * (
            steering_cosine - target_cosine
        )
        complex_gain = complex(np.mean(np.exp(1j * phase_difference)))
        derivative = complex(np.mean(
            -1j * phase_scale * elements * np.exp(1j * phase_difference)
        ))
        gains[q] = abs(complex_gain) ** 2
        gradients[q] = 2.0 * float(np.real(np.conj(complex_gain) * derivative))
    return gains, gradients


def projected_gradient_shared_phase(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    *,
    surrogate: str = "worst",
    initial_cosine: float = 0.0,
    max_steps: int = 120,
    initial_step: float = 0.2,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """Projected gradient ascent on the worst or mean shared array power."""
    if surrogate not in {"worst", "mean"}:
        raise ValueError("surrogate must be 'worst' or 'mean'")
    u = float(np.clip(initial_cosine, -1.0, 1.0))
    step = float(initial_step)
    best_u = u
    best_value = -np.inf
    for _ in range(max_steps):
        gains, gradients = shared_array_power_and_gradient(
            config, target_positions, u
        )
        if surrogate == "worst":
            value = float(np.min(gains))
            gradient = float(gradients[int(np.argmin(gains))])
        else:
            value = float(np.mean(gains))
            gradient = float(np.mean(gradients))
        if value > best_value:
            best_value = value
            best_u = u
        candidate = float(np.clip(u + step * gradient, -1.0, 1.0))
        candidate_gains, _ = shared_array_power_and_gradient(
            config, target_positions, candidate
        )
        candidate_value = (
            float(np.min(candidate_gains))
            if surrogate == "worst"
            else float(np.mean(candidate_gains))
        )
        if candidate_value > value + tolerance:
            improvement = candidate_value - value
            u = candidate
            step = min(step * 1.2, 0.8)
            if improvement <= tolerance:
                break
        else:
            step *= 0.5
            if step < 1e-6:
                break
    return {
        "steering_cosine": best_u,
        "surrogate": surrogate,
        "surrogate_value": best_value,
        "steps": max_steps,
    }


def shared_phase_gain_matrix(
    config: RisConfig,
    transmitter_positions: Sequence[Sequence[float]],
    target_positions: Sequence[Sequence[float]],
    receiver_position: Sequence[float],
    aperture_scale: float,
    direct_blockage: float = 0.01,
    phase: Sequence[float] | None = None,
) -> np.ndarray:
    """Gain matrix with one physical phase profile shared by all targets."""
    if phase is None:
        target_cosines = target_direction_cosines(config, target_positions)
        weak_index = (
            len(target_positions) - 1
            if config.weak_target_id is None
            else config.weak_target_id
        )
        phase = ris_beam_phase_from_cosine(
            config, float(target_cosines[weak_index])
        )
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (config.num_elements,):
        raise ValueError("phase must have one entry per RIS element")
    return ris_physics_gain_matrix(
        config,
        transmitter_positions,
        target_positions,
        receiver_position,
        aperture_scale,
        direct_blockage=direct_blockage,
        phase_per_target=[phase for _ in target_positions],
    )


def ris_beam_phase_from_cosine(
    config: RisConfig,
    steering_cosine: float,
) -> np.ndarray:
    """Ideal ULA phase vector for a steering cosine in [-1, 1]."""
    steering_cosine = float(np.clip(steering_cosine, -1.0, 1.0))
    wavelength = PROPAGATION_SPEED / config.carrier_hz
    spacing = config.element_spacing_lambda * wavelength
    wavenumber = 2.0 * np.pi / wavelength
    elements = np.arange(config.num_elements, dtype=float)
    return (-wavenumber * spacing * elements * steering_cosine) % (2.0 * np.pi)
