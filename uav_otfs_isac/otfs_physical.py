"""Waveform-level OTFS primitives for collision-aware UAV-ISAC studies.

This first physical prototype uses rectangular pulses and a circular frame
model, equivalent to ideal periodic extension or a sufficient cyclic prefix.
It is a Gate-0 collision model, not yet a pulse-shaped SDR transceiver.
"""

from __future__ import annotations

import numpy as np
from math import gcd
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]


def isfft(dd_grid: ComplexArray) -> ComplexArray:
    """Map [Doppler, delay] symbols to the time-frequency grid.

    The convention is ``F_N^H X_DD F_M``: an inverse DFT along Doppler and a
    DFT along delay. Orthonormal FFT scaling makes the map unitary.
    """
    grid = np.asarray(dd_grid, dtype=complex)
    if grid.ndim != 2 or 0 in grid.shape:
        raise ValueError("dd_grid must be a nonempty two-dimensional array")
    return np.fft.fft(
        np.fft.ifft(grid, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )


def sfft(tf_grid: ComplexArray) -> ComplexArray:
    """Map a time-frequency grid back to [Doppler, delay] symbols."""
    grid = np.asarray(tf_grid, dtype=complex)
    if grid.ndim != 2 or 0 in grid.shape:
        raise ValueError("tf_grid must be a nonempty two-dimensional array")
    return np.fft.fft(
        np.fft.ifft(grid, axis=1, norm="ortho"),
        axis=0,
        norm="ortho",
    )


def heisenberg(tf_grid: ComplexArray) -> ComplexArray:
    """Generate rectangular-pulse time samples from a time-frequency grid."""
    grid = np.asarray(tf_grid, dtype=complex)
    if grid.ndim != 2 or 0 in grid.shape:
        raise ValueError("tf_grid must be a nonempty two-dimensional array")
    time_slots = np.fft.ifft(grid, axis=1, norm="ortho")
    return time_slots.reshape(-1)


def wigner(time_samples: ComplexArray, doppler_bins: int, delay_bins: int) -> ComplexArray:
    """Recover the time-frequency grid for rectangular pulses."""
    samples = np.asarray(time_samples, dtype=complex)
    if samples.ndim != 1 or samples.size != doppler_bins * delay_bins:
        raise ValueError("time_samples length must equal doppler_bins * delay_bins")
    time_slots = samples.reshape(doppler_bins, delay_bins)
    return np.fft.fft(time_slots, axis=1, norm="ortho")


def otfs_modulate(dd_grid: ComplexArray) -> ComplexArray:
    """ISFFT followed by the rectangular-pulse Heisenberg transform."""
    return heisenberg(isfft(dd_grid))


def otfs_demodulate(
    time_samples: ComplexArray, doppler_bins: int, delay_bins: int
) -> ComplexArray:
    """Rectangular-pulse Wigner transform followed by the SFFT."""
    return sfft(wigner(time_samples, doppler_bins, delay_bins))


def fractional_circular_delay(
    time_samples: ComplexArray, delay_samples: float
) -> ComplexArray:
    """Apply a unitary circular delay, including fractional sample offsets."""
    samples = np.asarray(time_samples, dtype=complex)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("time_samples must be a nonempty vector")
    if not np.isfinite(delay_samples):
        raise ValueError("delay_samples must be finite")
    frequencies = np.fft.fftfreq(samples.size)
    return np.fft.ifft(
        np.fft.fft(samples)
        * np.exp(-2j * np.pi * frequencies * float(delay_samples))
    )


def delay_doppler_path(
    time_samples: ComplexArray, delay_samples: float,
    doppler_bins: float, doppler_grid_size: int, gain: complex = 1.0,
) -> ComplexArray:
    """Apply one persistent circular delay-Doppler path in the time domain.

    ``doppler_bins`` is normalized to the OTFS Doppler resolution: an integer
    value shifts an ideal DD impulse by that many Doppler bins.
    """
    samples = np.asarray(time_samples, dtype=complex)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("time_samples must be a nonempty vector")
    if doppler_grid_size <= 0 or samples.size % doppler_grid_size != 0:
        raise ValueError("doppler_grid_size must positively divide the frame length")
    if not np.isfinite(delay_samples) or not np.isfinite(doppler_bins):
        raise ValueError("delay and Doppler offsets must be finite")
    if not np.isfinite(gain.real) or not np.isfinite(gain.imag):
        raise ValueError("gain must be finite")
    delayed = fractional_circular_delay(samples, delay_samples)
    indices = np.arange(delayed.size)
    phase = np.exp(
        2j * np.pi * float(doppler_bins) * indices
        / (doppler_grid_size * (delayed.size // doppler_grid_size))
    )
    return complex(gain) * delayed * phase


def apply_delay_doppler_channel(
    time_samples: ComplexArray,
    paths: list[tuple[complex, float, float]],
    doppler_grid_size: int,
) -> ComplexArray:
    """Superpose time-domain paths given as (gain, delay samples, Doppler bins)."""
    samples = np.asarray(time_samples, dtype=complex)
    received = np.zeros_like(samples)
    for gain, delay, doppler in paths:
        received += delay_doppler_path(
            samples, delay, doppler, doppler_grid_size, gain
        )
    return received


def cyclic_impulse_pattern(
    doppler_bins: int, delay_bins: int, doppler_index: int, delay_index: int
) -> ComplexArray:
    """Create a unit-energy DD pilot-sensing impulse at a cyclic grid index."""
    if doppler_bins <= 0 or delay_bins <= 0:
        raise ValueError("grid dimensions must be positive")
    grid = np.zeros((doppler_bins, delay_bins), dtype=complex)
    grid[doppler_index % doppler_bins, delay_index % delay_bins] = 1.0
    return grid


def qpsk_phase_pattern(
    doppler_bins: int, delay_bins: int, seed: int
) -> ComplexArray:
    """Create a reproducible unit-energy constant-modulus DD codeword."""
    if doppler_bins <= 0 or delay_bins <= 0:
        raise ValueError("grid dimensions must be positive")
    rng = np.random.default_rng(seed)
    phases = rng.integers(0, 4, size=(doppler_bins, delay_bins))
    return np.exp(0.5j * np.pi * phases) / np.sqrt(doppler_bins * delay_bins)


def separable_cazac_pattern(
    doppler_bins: int, delay_bins: int,
    doppler_root: int, delay_root: int,
) -> ComplexArray:
    """Create a unit-energy separable quadratic-phase CAZAC DD pattern."""
    if doppler_bins <= 0 or delay_bins <= 0:
        raise ValueError("grid dimensions must be positive")
    if gcd(int(doppler_root), doppler_bins) != 1:
        raise ValueError("doppler_root must be coprime with doppler_bins")
    if gcd(int(delay_root), delay_bins) != 1:
        raise ValueError("delay_root must be coprime with delay_bins")

    def sequence(length, root):
        indices = np.arange(length, dtype=float)
        parity = length % 2
        return np.exp(-1j * np.pi * root * indices * (indices + parity) / length)

    doppler_sequence = sequence(doppler_bins, int(doppler_root))
    delay_sequence = sequence(delay_bins, int(delay_root))
    pattern = np.outer(doppler_sequence, delay_sequence)
    return pattern / np.sqrt(doppler_bins * delay_bins)


def cazac_sequence(length: int, root: int) -> ComplexArray:
    """Create a unit-energy quadratic-phase CAZAC sequence."""
    if length <= 0:
        raise ValueError("length must be positive")
    if gcd(int(root), length) != 1:
        raise ValueError("root must be coprime with length")
    indices = np.arange(length, dtype=float)
    parity = length % 2
    return (
        np.exp(-1j * np.pi * int(root) * indices * (indices + parity) / length)
        / np.sqrt(length)
    )


def ula_steering_vector(
    num_antennas: int, angle_degrees: float, spacing_wavelengths: float = 0.5,
) -> ComplexArray:
    """Return a unit-norm broadside-referenced ULA steering vector."""
    if num_antennas <= 0:
        raise ValueError("num_antennas must be positive")
    if not np.isfinite(angle_degrees):
        raise ValueError("angle_degrees must be finite")
    if not np.isfinite(spacing_wavelengths) or spacing_wavelengths <= 0.0:
        raise ValueError("spacing_wavelengths must be finite and positive")
    elements = np.arange(num_antennas, dtype=float)
    phase = 2j * np.pi * spacing_wavelengths * elements * np.sin(
        np.deg2rad(angle_degrees)
    )
    return np.exp(phase) / np.sqrt(num_antennas)


def spatial_otfs_template(
    pattern: ComplexArray, delay_samples: float, doppler_bins: float,
    angle_degrees: float, num_antennas: int,
    spatial_code: ComplexArray | None = None,
) -> ComplexArray:
    """Create an [antenna, time] separable DD-spatial path template.

    ``spatial_code`` is an optional known per-element unit-norm signature.  It
    models a resolved spatial probing dimension; using it requires separate
    per-element observations or equivalent orthogonal probing in practice.
    """
    pattern_array = np.asarray(pattern, dtype=complex)
    if pattern_array.ndim != 2 or 0 in pattern_array.shape:
        raise ValueError("pattern must be a nonempty two-dimensional grid")
    steering = ula_steering_vector(num_antennas, angle_degrees)
    if spatial_code is not None:
        code = np.asarray(spatial_code, dtype=complex)
        if code.shape != (num_antennas,) or np.linalg.norm(code) <= 0.0:
            raise ValueError("spatial_code must be a nonzero antenna-length vector")
        steering = steering * code
        steering = steering / np.linalg.norm(steering)
    waveform = delay_doppler_path(
        otfs_modulate(pattern_array), delay_samples, doppler_bins,
        pattern_array.shape[0],
    )
    return np.outer(steering, waveform)


def spatial_matched_filter_map(
    received: ComplexArray, pattern: ComplexArray, angle_grid_degrees,
    spatial_code: ComplexArray | None = None,
) -> NDArray[np.float64]:
    """Return an [angle, Doppler, delay] matched-filter energy cube."""
    observations = np.asarray(received, dtype=complex)
    pattern_array = np.asarray(pattern, dtype=complex)
    if observations.ndim != 2 or observations.shape[1] != pattern_array.size:
        raise ValueError("received must have shape [antenna, pattern.size]")
    angles = np.asarray(angle_grid_degrees, dtype=float)
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("angle_grid_degrees must be a nonempty finite vector")
    cube = np.empty((angles.size,) + pattern_array.shape, dtype=float)
    for angle_index, angle in enumerate(angles):
        steering = ula_steering_vector(observations.shape[0], angle)
        if spatial_code is not None:
            code = np.asarray(spatial_code, dtype=complex)
            if code.shape != steering.shape or np.linalg.norm(code) <= 0.0:
                raise ValueError("spatial_code must be a nonzero antenna-length vector")
            steering = steering * code
            steering = steering / np.linalg.norm(steering)
        beamformed = steering.conj() @ observations
        cube[angle_index] = matched_filter_map(beamformed, pattern_array)
    return cube


def dd_cross_ambiguity(
    first_pattern: ComplexArray, second_pattern: ComplexArray,
    delay_shifts, doppler_shifts,
) -> ComplexArray:
    """Evaluate normalized cross ambiguity over continuous DD offsets."""
    first = np.asarray(first_pattern, dtype=complex)
    second = np.asarray(second_pattern, dtype=complex)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("patterns must have the same two-dimensional shape")
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    if norm <= 0.0:
        raise ValueError("patterns must have nonzero energy")
    first_samples = otfs_modulate(first)
    second_samples = otfs_modulate(second)
    delays = np.asarray(delay_shifts, dtype=float)
    dopplers = np.asarray(doppler_shifts, dtype=float)
    ambiguity = np.empty((dopplers.size, delays.size), dtype=complex)
    for k, doppler in enumerate(dopplers):
        for l, delay in enumerate(delays):
            shifted = delay_doppler_path(
                second_samples, delay, doppler, first.shape[0]
            )
            ambiguity[k, l] = np.vdot(first_samples, shifted) / norm
    return ambiguity


def superpose_uav_echoes(
    patterns: list[ComplexArray],
    paths_per_uav: list[list[tuple[complex, float, float]]],
    noise_variance: float,
    rng: np.random.Generator,
) -> ComplexArray:
    """Modulate and superpose concurrent UAV echoes in one receiver."""
    if len(patterns) != len(paths_per_uav) or not patterns:
        raise ValueError("one path list is required per nonempty pattern list")
    shape = np.asarray(patterns[0]).shape
    if any(np.asarray(pattern).shape != shape for pattern in patterns):
        raise ValueError("all patterns must share one DD grid shape")
    received = np.zeros(shape[0] * shape[1], dtype=complex)
    for pattern, paths in zip(patterns, paths_per_uav):
        received += apply_delay_doppler_channel(
            otfs_modulate(pattern), paths, shape[0]
        )
    if noise_variance < 0.0:
        raise ValueError("noise_variance must be nonnegative")
    if noise_variance > 0.0:
        received += np.sqrt(noise_variance / 2.0) * (
            rng.standard_normal(received.size)
            + 1j * rng.standard_normal(received.size)
        )
    return received


def matched_filter_map(
    received_samples: ComplexArray, pattern: ComplexArray
) -> NDArray[np.float64]:
    """Return the integer-grid normalized matched-filter energy map."""
    pattern_array = np.asarray(pattern, dtype=complex)
    reference = otfs_modulate(pattern_array)
    received = np.asarray(received_samples, dtype=complex)
    if received.shape != reference.shape:
        raise ValueError("received samples and pattern frame must match")
    denominator = max(float(np.vdot(reference, reference).real), 1e-15)
    result = np.empty(pattern_array.shape, dtype=float)
    for k in range(pattern_array.shape[0]):
        for l in range(pattern_array.shape[1]):
            template = delay_doppler_path(
                reference, float(l), float(k), pattern_array.shape[0]
            )
            result[k, l] = abs(np.vdot(template, received)) ** 2 / denominator
    return result


def cyclic_nms_peaks(
    energy_map: NDArray[np.float64], count: int, guard_radius: int = 1
) -> list[tuple[int, int]]:
    """Extract strongest distinct peaks using cyclic non-maximum suppression."""
    values = np.asarray(energy_map, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("energy_map must be a nonempty matrix")
    if count < 0 or count > values.size:
        raise ValueError("count must lie between zero and the map size")
    if guard_radius < 0:
        raise ValueError("guard_radius must be nonnegative")
    remaining = values.copy()
    peaks: list[tuple[int, int]] = []
    for _ in range(count):
        peak = tuple(int(value) for value in np.unravel_index(
            np.argmax(remaining), remaining.shape
        ))
        peaks.append(peak)
        for dk in range(-guard_radius, guard_radius + 1):
            for dl in range(-guard_radius, guard_radius + 1):
                remaining[
                    (peak[0] + dk) % values.shape[0],
                    (peak[1] + dl) % values.shape[1],
                ] = -np.inf
    return peaks


def threshold_cyclic_nms_peaks(
    energy_map: NDArray[np.float64], threshold: float, guard_radius: int = 1
) -> list[tuple[int, int]]:
    """Extract all cyclic NMS peaks above a fixed detection threshold.

    Unlike :func:`cyclic_nms_peaks`, this detector does not require the target
    count.  The threshold is expected to be calibrated under the null
    hypothesis for the desired false-alarm rate.
    """
    values = np.asarray(energy_map, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("energy_map must be a nonempty matrix")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    if guard_radius < 0:
        raise ValueError("guard_radius must be nonnegative")
    remaining = values.copy()
    peaks: list[tuple[int, int]] = []
    while float(np.max(remaining)) >= threshold:
        peak = tuple(int(value) for value in np.unravel_index(
            np.argmax(remaining), remaining.shape
        ))
        peaks.append(peak)
        for dk in range(-guard_radius, guard_radius + 1):
            for dl in range(-guard_radius, guard_radius + 1):
                remaining[
                    (peak[0] + dk) % values.shape[0],
                    (peak[1] + dl) % values.shape[1],
                ] = -np.inf
    return peaks


def matched_filter_cell_threshold(noise_variance: float, p_false_alarm: float) -> float:
    """Exact per-cell threshold for unit-energy templates in complex AWGN."""
    if not np.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    if not np.isfinite(p_false_alarm) or not 0.0 < p_false_alarm < 1.0:
        raise ValueError("p_false_alarm must lie strictly between zero and one")
    return float(-noise_variance * np.log(p_false_alarm))
