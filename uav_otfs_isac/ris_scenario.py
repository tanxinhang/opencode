"""RIS-assisted 6G UAV-OTFS-ISAC scenario.

The channel model replaces the single-hop direct view with a direct path plus
a reconfigurable-intelligent-surface (RIS) cascaded path.  The RIS phase
profile steers an array gain toward a target, so a blocked weak target can be
illuminated through a controllable NLoS path.  For UAV ``i`` and target ``q``
the controlled channel gain is

``gain_iq(theta) = 1 + (ris_strength * array_gain(theta))^2``

so the RIS adds a controllable NLoS power component that is monotone in array
alignment and never reduces the direct evidence SNR.  This additive-power
model is deliberately used instead of coherent complex combining: the
moment-matched Gaussian detector is not guaranteed monotone in coherent
amplitude, and the additive model respects the basic sensing principle that a
better RIS alignment cannot hurt a link.  The resulting gain matrix is injected into
:func:`uav_otfs_isac.scenario.build_models` before quantization, BSC, and
erasure reporting, so communication principles are unchanged and the sensing
channel is the 6G RIS-assisted cascade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ExperimentConfig
from .models import TargetEvidenceModel
from .scenario import build_models, target_geometry


PROPAGATION_SPEED = 299_792_458.0


@dataclass(frozen=True)
class RisConfig:
    """Controlled RIS geometry and illumination parameters."""

    position: np.ndarray
    num_elements: int
    aperture_shape: tuple[int, int] | None = None
    carrier_hz: float = 5.9e9
    element_spacing_lambda: float = 0.5
    weak_target_id: int | None = None
    ris_strength_weak: float = 3.0
    ris_strength_strong: float = 0.5
    phase_bits: int | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        if position.ndim != 1 or position.size not in (2, 3):
            raise ValueError("RIS position must be a finite 2-D or 3-D vector")
        if not np.all(np.isfinite(position)):
            raise ValueError("RIS position must be finite")
        if self.num_elements <= 0:
            raise ValueError("num_elements must be positive")
        if self.aperture_shape is not None:
            rows, columns = self.aperture_shape
            if rows <= 0 or columns <= 0:
                raise ValueError("aperture_shape entries must be positive")
            if rows * columns != self.num_elements:
                raise ValueError("aperture_shape product must equal num_elements")
        if self.carrier_hz <= 0.0:
            raise ValueError("carrier_hz must be positive")
        if self.element_spacing_lambda <= 0.0:
            raise ValueError("element_spacing_lambda must be positive")
        if self.ris_strength_weak < 0.0 or self.ris_strength_strong < 0.0:
            raise ValueError("RIS strengths must be nonnegative")
        if self.phase_bits is not None and self.phase_bits <= 0:
            raise ValueError("phase_bits must be positive or None")
        object.__setattr__(self, "position", position.copy())


def ris_control_overhead_bits(
    config: RisConfig,
    coherence_frames: int = 1,
) -> float:
    """Control-plane bits per frame for configuring the RIS phase profile."""
    if config.phase_bits is None:
        return 0.0
    if coherence_frames <= 0:
        raise ValueError("coherence_frames must be positive")
    return float(config.num_elements * config.phase_bits / coherence_frames)


def ris_beam_phase(
    target_position: Sequence[float],
    config: RisConfig,
) -> np.ndarray:
    """Quantized RIS phases that steer a beam toward the target."""
    target = np.asarray(target_position, dtype=float)
    if target.shape != config.position.shape:
        raise ValueError("target and RIS positions must share dimension")
    direction = target - config.position
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("target cannot coincide with the RIS")
    direction /= norm
    wavelength = PROPAGATION_SPEED / config.carrier_hz
    wavenumber = 2.0 * np.pi / wavelength
    spacing = config.element_spacing_lambda * wavelength
    elements = np.arange(config.num_elements, dtype=float)
    # 1-D uniform array along the x-axis; beam steering uses the projected
    # direction on the array axis.
    return (-wavenumber * spacing * elements * direction[0]) % (2.0 * np.pi)


def quantize_phase(
    phase: Sequence[float],
    bits: int | None,
) -> np.ndarray:
    """Quantize phase to ``bits`` uniform levels, or pass through unchanged."""
    phase = np.asarray(phase, dtype=float)
    if bits is None:
        return phase.copy()
    if bits <= 0:
        raise ValueError("bits must be positive or None")
    levels = 2**bits
    step = 2.0 * np.pi / levels
    return (np.round(phase / step) * step) % (2.0 * np.pi)


def ris_quantized_gain_loss(bits: int | None) -> float:
    """Theoretical mean array-gain factor of b-bit phase quantization.

    For uniformly distributed phase errors on ``[-pi/2^b, pi/2^b]``,
    ``E[cos(error)] = sinc(1/2^b)``, so the mean power gain scales by
    ``sinc^2(1/2^b)``.  ``None`` denotes ideal continuous phase with factor 1.
    """
    if bits is None:
        return 1.0
    if bits <= 0:
        raise ValueError("bits must be positive or None")
    return float(np.sinc(1.0 / 2**bits) ** 2)


def ris_array_gain(
    phase: Sequence[float],
    target_position: Sequence[float],
    config: RisConfig,
) -> float:
    """Normalized RIS array gain in [0, 1] toward the target direction."""
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (config.num_elements,):
        raise ValueError("phase must have one entry per RIS element")
    aligned = ris_beam_phase(target_position, config)
    phase = quantize_phase(phase, config.phase_bits)
    return float(abs(np.mean(np.exp(1j * (phase - aligned)))))


def ris_gain_matrix(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    num_uavs: int,
    phase_per_target: Sequence[Sequence[float]],
) -> np.ndarray:
    """Per-target, per-UAV evidence SNR gain induced by the RIS channel."""
    targets = [np.asarray(position, dtype=float) for position in target_positions]
    phases = [np.asarray(phase, dtype=float) for phase in phase_per_target]
    if len(targets) != len(phases):
        raise ValueError("one RIS phase vector is required per target")
    for phase in phases:
        if phase.shape != (config.num_elements,):
            raise ValueError("each phase vector must match num_elements")
    gains = np.ones((len(targets), num_uavs), dtype=float)
    for target_index, (target, phase) in enumerate(zip(targets, phases)):
        strength = (
            config.ris_strength_weak
            if config.weak_target_id == target_index
            else config.ris_strength_strong
        )
        array_gain_value = ris_array_gain(phase, target, config)
        gains[target_index, :] = 1.0 + (
            strength * array_gain_value
        ) ** 2
    return gains


def ris_physics_gain_matrix(
    config: RisConfig,
    transmitter_positions: Sequence[Sequence[float]],
    target_positions: Sequence[Sequence[float]],
    receiver_position: Sequence[float],
    aperture_scale: float,
    direct_blockage: float = 0.01,
    phase_per_target: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Physics-based direct-plus-RIS cascaded gain matrix.

    The direct bistatic path power follows the two-way radar law

    ``P_dir = 1 / (R_tx^2 R_rx^2)``

    and the RIS cascaded path (transmitter -> RIS -> target -> receiver)
    follows the product of three propagation losses with an ``N^2`` coherent
    array gain:

    ``P_ris = N^2 array_gain^2 aperture_scale / (R_1^2 R_2^2 R_3^2)``.

    The evidence SNR gain is ``1 + P_ris / P_dir`` for a clean link and
    ``direct_blockage + P_ris / P_dir`` for the weak target: the blockage
    attenuates ONLY the direct term, while the RIS boost stays referenced
    to the unblocked ``P_dir``.  The channel never amplifies a link beyond
    its clean baseline and never reduces a link below its blocked floor
    (array alignment only ever adds RIS power).
    """
    transmitters = [np.asarray(position, dtype=float) for position in transmitter_positions]
    targets = [np.asarray(position, dtype=float) for position in target_positions]
    receiver = np.asarray(receiver_position, dtype=float)
    dimension = receiver.size
    if any(position.shape != (dimension,) for position in transmitters + targets):
        raise ValueError("all positions must share one dimension")
    if aperture_scale < 0.0:
        raise ValueError("aperture_scale must be nonnegative")
    if not 0.0 <= direct_blockage <= 1.0:
        raise ValueError("direct_blockage must lie in [0, 1]")
    if phase_per_target is None:
        phases = [ris_beam_phase(target, config) for target in targets]
    else:
        phases = [np.asarray(phase, dtype=float) for phase in phase_per_target]
    if len(phases) != len(targets):
        raise ValueError("one phase vector is required per target")
    gains = np.ones((len(targets), len(transmitters)), dtype=float)
    for target_index, (target, phase) in enumerate(zip(targets, phases)):
        array_gain_value = ris_array_gain(phase, target, config)
        for transmitter_index, transmitter in enumerate(transmitters):
            tx_target = float(np.linalg.norm(transmitter - target))
            target_rx = float(np.linalg.norm(target - receiver))
            tx_ris = float(np.linalg.norm(transmitter - config.position))
            ris_target = float(np.linalg.norm(config.position - target))
            if min(tx_target, target_rx, tx_ris, ris_target) == 0.0:
                raise ValueError("degenerate zero-length channel path")
            direct_power = 1.0 / (tx_target**2 * target_rx**2)
            direct_gain = (
                direct_blockage if config.weak_target_id == target_index else 1.0
            )
            ris_power = (
                config.num_elements**2
                * array_gain_value**2
                * aperture_scale
                / (tx_ris**2 * ris_target**2 * target_rx**2)
            )
            gains[target_index, transmitter_index] = (
                direct_gain + ris_power / max(direct_power, 1e-30)
            )
    return gains


def blocked_direct_gain_matrix(
    num_targets: int,
    num_uavs: int,
    weak_target_id: int | None = None,
    clean_gain: float = 1.0,
    direct_blockage: float = 0.01,
) -> np.ndarray:
    """P4 matched-control factory (advice/011 section 5): the
    \"blocked, no RIS\" reference gain matrix.

    Clean targets keep ``clean_gain`` (the normalized direct path = 1);
    the weak target gets the ``direct_blockage`` floor and NO RIS benefit.
    This is the correct control for the RIS-recovery experiment:

    ``eta = (P_M[aligned] - P_M[blocked]) / (P_M[clean] - P_M[blocked])``,

    while ``clean_gain == 1`` (plain build_models) is only the
    unblocked upper/reference bound.
    """
    gain = np.full((int(num_targets), int(num_uavs)), float(clean_gain))
    if weak_target_id is not None:
        gain[int(weak_target_id), :] = float(direct_blockage)
    return gain


def build_ris_models(
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    config: RisConfig,
    phase_per_target: Sequence[Sequence[float]],
) -> list[TargetEvidenceModel]:
    """Build models whose evidence SNR includes the RIS channel gain."""
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    gain = ris_gain_matrix(
        config, targets, cfg.num_uavs, phase_per_target
    )
    models = build_models(cfg, rng, snr_gain=gain)
    return models
