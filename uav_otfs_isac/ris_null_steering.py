"""UPA phase design with target enhancement and interference null-steering.

The objective for a target direction and interference directions is the
scalarized array power

``J(phi) = G_t(phi) - lambda sum_j G_j(phi)``,

where ``G_d = |mean_n exp(j(phi_n - ideal_nd))|^2``.  The gradient is
analytic, so the phase vector is optimized with L-BFGS-B.  Reflected
interference power is then evaluated with the designed array gains.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from .ris_scenario import quantize_phase
from .ris_upd import upd_array_gain, upd_ideal_phase


def array_power(phase: np.ndarray, ideal: np.ndarray) -> float:
    phase = np.asarray(phase, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    value = complex(np.mean(np.exp(1j * (phase - ideal))))
    return abs(value) ** 2


def array_power_gradient(
    phase: np.ndarray,
    ideal: np.ndarray,
) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    value = complex(np.mean(np.exp(1j * (phase - ideal))))
    gradient = (
        2.0 / phase.size
        * np.real(np.conj(value) * 1j * np.exp(1j * (phase - ideal)))
    )
    return gradient


def optimize_null_steering_phases(
    config,
    target_position: Sequence[float],
    interference_positions: Sequence[Sequence[float]],
    *,
    lambda_: float = 1.0,
) -> np.ndarray:
    """Optimize UPA phases for target gain and interference suppression."""
    target_ideal = upd_ideal_phase(config, target_position)
    interference_ideals = [
        upd_ideal_phase(config, position)
        for position in interference_positions
    ]

    def negative_objective(phase):
        phase = np.asarray(phase, dtype=float)
        target_power = array_power(phase, target_ideal)
        interference_power = sum(
            array_power(phase, ideal) for ideal in interference_ideals
        )
        return -(target_power - lambda_ * interference_power)

    def negative_gradient(phase):
        phase = np.asarray(phase, dtype=float)
        gradient = array_power_gradient(phase, target_ideal)
        for ideal in interference_ideals:
            gradient -= lambda_ * array_power_gradient(phase, ideal)
        return -gradient

    initial = target_ideal
    bounds = [(0.0, 2.0 * np.pi)] * config.num_elements
    result = minimize(
        negative_objective, initial, jac=negative_gradient,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    return (np.asarray(result.x, dtype=float) % (2.0 * np.pi))


def quantized_null_steering_phases(
    config,
    target_position: Sequence[float],
    interference_positions: Sequence[Sequence[float]],
    *,
    lambda_: float = 1.0,
    max_rounds: int = 6,
) -> np.ndarray:
    """Discrete phase coordinate ascent for quantized null-steering.

    The phase of each element is constrained to the ``2^b`` uniform
    quantization levels.  The optimization starts from the quantized
    continuous-phase solution and flips one element at a time while
    improving the scalarized array power.
    """
    if config.phase_bits is None:
        raise ValueError("quantized null-steering requires phase_bits")
    continuous = optimize_null_steering_phases(
        config, target_position, interference_positions, lambda_=lambda_
    )
    phase = quantize_phase(continuous, config.phase_bits)
    target_ideal = upd_ideal_phase(config, target_position)
    interference_ideals = [
        upd_ideal_phase(config, position)
        for position in interference_positions
    ]

    def objective(candidate):
        target_power = array_power(candidate, target_ideal)
        interference_power = sum(
            array_power(candidate, ideal) for ideal in interference_ideals
        )
        return target_power - lambda_ * interference_power

    step = 2.0 * np.pi / (2 ** config.phase_bits)
    levels = np.arange(2 ** config.phase_bits, dtype=float) * step
    for _ in range(max_rounds):
        improved = False
        for index in range(config.num_elements):
            current = objective(phase)
            best_value = current
            best_level = phase[index]
            for level in levels:
                if abs(level - phase[index]) < 1e-12:
                    continue
                trial = phase.copy()
                trial[index] = level
                value = objective(trial)
                if value > best_value + 1e-12:
                    best_value = value
                    best_level = level
            if abs(best_level - phase[index]) > 1e-12:
                phase[index] = best_level
                improved = True
        if not improved:
            break
    return phase


def reflected_interference_inr(
    config,
    phase: np.ndarray,
    sources: Sequence[Sequence[float]],
    transmitter_positions: Sequence[Sequence[float]],
    *,
    inr_ref: float = 0.1,
    reference_distance: float = 100.0,
) -> np.ndarray:
    """Per-UAV INR contributed by RIS-reflected interference paths."""
    transmitters = np.asarray(transmitter_positions, dtype=float)
    total = np.zeros(transmitters.shape[0], dtype=float)
    for source in sources:
        source = np.asarray(source, dtype=float)
        array_gain = upd_array_gain(phase, source, config)
        source_ris = float(np.linalg.norm(source - config.position))
        for i, transmitter in enumerate(transmitters):
            ris_uav = float(np.linalg.norm(config.position - transmitter))
            total[i] += (
                inr_ref
                * array_gain**2
                * (reference_distance / max(source_ris, 1e-9)) ** 2
                * (reference_distance / max(ris_uav, 1e-9)) ** 2
            )
    return total
