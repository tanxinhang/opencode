"""Multi-RIS non-coherent cascaded channel model.

Each RIS contributes an independent cascaded path.  Because the RIS surfaces
are not phase-coherent with each other in the audited model, their reflected
powers add:

``gain_iq = 1 + sum_r P_ris,r / P_dir_iq``,

where

``P_ris,r = N_r^2 G_rq^2 A / (R_1r^2 R_2r^2 R_3r^2)``.

The control overhead is also additive:

``B_control = sum_r N_r phase_bits / coherence_frames``.

This keeps the communication/sensing ledger exact while adding placement
diversity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ris_scenario import RisConfig, ris_array_gain, ris_beam_phase


def multi_ris_control_overhead(
    ris_configs: Sequence[RisConfig],
    coherence_frames: int,
) -> float:
    """Total amortized control bits per frame over all RIS surfaces."""
    from .ris_scenario import ris_control_overhead_bits

    return float(sum(
        ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
        for ris in ris_configs
    ))


def multi_ris_physics_gain_matrix(
    ris_configs: Sequence[RisConfig],
    transmitter_positions: Sequence[Sequence[float]],
    target_positions: Sequence[Sequence[float]],
    receiver_position: Sequence[float],
    aperture_scale: float,
    direct_blockage: float = 0.01,
    phases_per_ris: Sequence[Sequence[Sequence[float]]] | None = None,
) -> np.ndarray:
    """Gain matrix with power-summed cascaded RIS paths."""
    ris_configs = list(ris_configs)
    if not ris_configs:
        raise ValueError("at least one RIS is required")
    transmitters = [
        np.asarray(position, dtype=float) for position in transmitter_positions
    ]
    targets = [
        np.asarray(position, dtype=float) for position in target_positions
    ]
    receiver = np.asarray(receiver_position, dtype=float)
    dimension = receiver.size
    if any(position.shape != (dimension,) for position in transmitters + targets):
        raise ValueError("all positions must share one dimension")
    if phases_per_ris is None:
        phases_per_ris = [
            [ris_beam_phase(target, ris) for target in targets]
            for ris in ris_configs
        ]
    if len(phases_per_ris) != len(ris_configs):
        raise ValueError("one phase set is required per RIS")
    gains = np.ones((len(targets), len(transmitters)), dtype=float)
    for q, target in enumerate(targets):
        for i, transmitter in enumerate(transmitters):
            tx_target = float(np.linalg.norm(transmitter - target))
            target_rx = float(np.linalg.norm(target - receiver))
            direct_power = 1.0 / (tx_target**2 * target_rx**2)
            if any(
                ris.weak_target_id == q for ris in ris_configs
            ):
                direct_power *= direct_blockage
            ris_power = 0.0
            for ris, phases in zip(ris_configs, phases_per_ris):
                if len(phases) != len(targets):
                    raise ValueError("one phase vector is required per target")
                array_gain = ris_array_gain(
                    np.asarray(phases[q], dtype=float),
                    target,
                    ris,
                )
                tx_ris = float(np.linalg.norm(transmitter - ris.position))
                ris_target = float(np.linalg.norm(ris.position - target))
                if min(tx_ris, ris_target, target_rx) == 0.0:
                    raise ValueError("degenerate zero-length RIS path")
                ris_power += (
                    ris.num_elements**2
                    * array_gain**2
                    * aperture_scale
                    / (tx_ris**2 * ris_target**2 * target_rx**2)
                )
            gains[q, i] = 1.0 + ris_power / max(direct_power, 1e-30)
    return gains
