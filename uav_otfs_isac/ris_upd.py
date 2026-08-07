"""2-D uniform planar array (UPA) model for the RIS.

For an ``Nx x Ny`` aperture, the ideal phase of element ``(nx, ny)`` is

``ideal = -k d (nx u_x + ny u_y)``,

where ``u`` is the unit direction toward the target.  The array gain is the
normalized magnitude of the full 2-D array factor, and the physics gain
matrix follows the same cascaded path-loss model as the 1-D ULA.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ris_scenario import (
    PROPAGATION_SPEED,
    RisConfig,
    quantize_phase,
)


def upd_ideal_phase(
    config: RisConfig,
    target_position: Sequence[float],
) -> np.ndarray:
    """Ideal continuous phase vector for a 2-D UPA."""
    if config.aperture_shape is None:
        raise ValueError("aperture_shape is required for UPA")
    rows, columns = config.aperture_shape
    direction = np.asarray(target_position, dtype=float) - config.position
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("target cannot coincide with the RIS")
    direction /= norm
    wavelength = PROPAGATION_SPEED / config.carrier_hz
    spacing = config.element_spacing_lambda * wavelength
    wavenumber = 2.0 * np.pi / wavelength
    nx = np.arange(columns, dtype=float)
    ny = np.arange(rows, dtype=float)
    phases = -wavenumber * spacing * (
        nx[None, :] * direction[0] + ny[:, None] * direction[1]
    )
    return (phases % (2.0 * np.pi)).reshape(-1)


def upd_array_gain(
    phase: Sequence[float],
    target_position: Sequence[float],
    config: RisConfig,
) -> float:
    """Normalized 2-D array gain in [0, 1]."""
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (config.num_elements,):
        raise ValueError("phase must have one entry per RIS element")
    aligned = upd_ideal_phase(config, target_position)
    phase = quantize_phase(phase, config.phase_bits)
    return float(abs(np.mean(np.exp(1j * (phase - aligned)))))


def upd_physics_gain_matrix(
    config: RisConfig,
    transmitter_positions: Sequence[Sequence[float]],
    target_positions: Sequence[Sequence[float]],
    receiver_position: Sequence[float],
    aperture_scale: float,
    direct_blockage: float = 0.01,
    phase_per_target: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Physics gain matrix for the UPA, one phase vector per target."""
    transmitters = [
        np.asarray(position, dtype=float) for position in transmitter_positions
    ]
    targets = [
        np.asarray(position, dtype=float) for position in target_positions
    ]
    receiver = np.asarray(receiver_position, dtype=float)
    if phase_per_target is None:
        phases = [upd_ideal_phase(config, target) for target in targets]
    else:
        phases = [np.asarray(phase, dtype=float) for phase in phase_per_target]
    if len(phases) != len(targets):
        raise ValueError("one phase vector is required per target")
    gains = np.ones((len(targets), len(transmitters)), dtype=float)
    for q, target in enumerate(targets):
        array_gain = upd_array_gain(phases[q], target, config)
        for i, transmitter in enumerate(transmitters):
            tx_target = float(np.linalg.norm(transmitter - target))
            target_rx = float(np.linalg.norm(target - receiver))
            tx_ris = float(np.linalg.norm(transmitter - config.position))
            ris_target = float(np.linalg.norm(config.position - target))
            if min(tx_target, target_rx, tx_ris, ris_target) == 0.0:
                raise ValueError("degenerate zero-length channel path")
            direct = 1.0 / (tx_target**2 * target_rx**2)
            if config.weak_target_id == q:
                direct *= direct_blockage
            ris_power = (
                config.num_elements**2
                * array_gain**2
                * aperture_scale
                / (tx_ris**2 * ris_target**2 * target_rx**2)
            )
            gains[q, i] = 1.0 + ris_power / max(direct, 1e-30)
    return gains
