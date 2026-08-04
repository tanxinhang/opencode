"""Toy OTFS waveform front end feeding geometry-aware path candidates.

The module intentionally follows the repository's Gate-0 convention: delay
and Doppler are represented on a small DD grid with explicitly declared
resolutions.  The mapping is a lightweight waveform mechanism model, not a
bandwidth-consistent SDR transceiver.  Its role is to replace the synthetic
path-candidate oracle in G0-B with a matched-filter/CFAR front end that also
exports calibrated path-existence probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .multistatic_association import PathCandidate
from .multistatic_targets import BistaticPath
from .otfs_physical import (
    qpsk_phase_pattern,
    separable_cazac_pattern,
    spatial_otfs_template,
)
from .probability_calibration import (
    IsotonicProbabilityCalibrator,
    fit_isotonic_probability,
)
from .spatial_detection import (
    separable_detection_cube,
    spatial_dictionary,
    threshold_nms_3d,
    waveform_dictionary,
)


PROPAGATION_SPEED = 299_792_458.0


@dataclass(frozen=True)
class FrontEndConfig:
    """Declared DD-grid resolutions and receive-array settings."""

    doppler_bins: int = 16
    delay_bins: int = 32
    delay_resolution_m: float = 8.0
    doppler_resolution_hz: float = 20.0
    angle_grid_degrees: np.ndarray = field(
        default_factory=lambda: np.arange(-60.0, 60.01, 5.0)
    )
    num_antennas: int = 8
    noise_variance: float = 0.02

    def __post_init__(self) -> None:
        if self.doppler_bins <= 0 or self.delay_bins <= 0:
            raise ValueError("DD grid dimensions must be positive")
        if self.delay_resolution_m <= 0.0 or self.doppler_resolution_hz <= 0.0:
            raise ValueError("grid resolutions must be positive")
        angles = np.asarray(self.angle_grid_degrees, dtype=float)
        if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
            raise ValueError("angle_grid_degrees must be a nonempty finite vector")
        if np.any(np.diff(angles) <= 0.0):
            raise ValueError("angle grid must be strictly increasing")
        if self.num_antennas <= 0:
            raise ValueError("num_antennas must be positive")
        if self.noise_variance < 0.0:
            raise ValueError("noise_variance must be nonnegative")
        object.__setattr__(self, "angle_grid_degrees", angles.copy())

    @property
    def frame_samples(self) -> int:
        return self.doppler_bins * self.delay_bins

    def delay_bin(self, delay_s: float) -> float:
        if not np.isfinite(delay_s) or delay_s <= 0.0:
            raise ValueError("delay_s must be positive and finite")
        return delay_s * PROPAGATION_SPEED / self.delay_resolution_m

    def signed_doppler_bin(self, doppler_hz: float) -> float:
        if not np.isfinite(doppler_hz):
            raise ValueError("doppler_hz must be finite")
        return doppler_hz / self.doppler_resolution_hz

    def wrapped_doppler_bin(self, doppler_hz: float) -> float:
        return self.signed_doppler_bin(doppler_hz) % self.doppler_bins

    def estimate_from_bins(
        self, delay_bin: float, doppler_bin: float
    ) -> tuple[float, float]:
        """Convert estimated continuous bins back to SI units."""
        delay_s = delay_bin * self.delay_resolution_m / PROPAGATION_SPEED
        signed = doppler_bin % self.doppler_bins
        if signed > self.doppler_bins / 2.0:
            signed -= self.doppler_bins
        return delay_s, signed * self.doppler_resolution_hz


def identity_patterns(
    config: FrontEndConfig, count: int, *, seed: int = 20260830,
    kind: str = "qpsk",
) -> list[np.ndarray]:
    """Create unit-energy transmitter identity signatures on one DD grid."""
    if count <= 0:
        raise ValueError("count must be positive")
    if kind == "qpsk":
        return [
            qpsk_phase_pattern(
                config.doppler_bins, config.delay_bins, seed + index
            )
            for index in range(count)
        ]
    if kind == "cazac":
        roots = [
            root for root in range(1, max(
                config.doppler_bins, config.delay_bins
            ))
            if root % 2 == 1
        ]
        if count > len(roots):
            raise ValueError("not enough coprime CAZAC roots")
        return [
            separable_cazac_pattern(
                config.doppler_bins, config.delay_bins,
                doppler_root=roots[index],
                delay_root=roots[index],
            )
            for index in range(count)
        ]
    raise ValueError("unsupported identity pattern kind")


def simulate_received(
    config: FrontEndConfig,
    patterns: Sequence[np.ndarray],
    paths: Iterable[BistaticPath],
    path_gains: Iterable[complex],
    rng: np.random.Generator,
) -> np.ndarray:
    """Superpose multistatic echoes on an L-element receive array."""
    paths = tuple(paths)
    gains = tuple(path_gains)
    if len(paths) != len(gains):
        raise ValueError("one gain is required per path")
    if len(patterns) == 0 or any(
        np.asarray(pattern).shape
        != (config.doppler_bins, config.delay_bins)
        for pattern in patterns
    ):
        raise ValueError("patterns must match the configured DD grid")
    received = np.zeros(
        (config.num_antennas, config.frame_samples), dtype=complex
    )
    for path, gain in zip(paths, gains):
        if path.transmitter_id >= len(patterns):
            raise ValueError("path transmitter_id is out of range")
        received += gain * spatial_otfs_template(
            patterns[path.transmitter_id],
            config.delay_bin(path.delay_s),
            config.wrapped_doppler_bin(path.doppler_hz),
            np.rad2deg(path.receive_azimuth_rad),
            config.num_antennas,
        )
    if config.noise_variance > 0.0:
        received += np.sqrt(config.noise_variance / 2.0) * (
            rng.standard_normal(received.shape)
            + 1j * rng.standard_normal(received.shape)
        )
    return received


def _detection_cube(
    config: FrontEndConfig,
    received: np.ndarray,
    pattern: np.ndarray,
    *,
    waveform_templates: np.ndarray,
    spatial_templates: np.ndarray,
) -> np.ndarray:
    return separable_detection_cube(
        received, waveform_templates, spatial_templates
    )


def detection_cubes(
    config: FrontEndConfig,
    received: np.ndarray,
    patterns: Sequence[np.ndarray],
    *,
    templates: tuple[
        list[np.ndarray], np.ndarray
    ] | None = None,
) -> list[np.ndarray]:
    """Return per-transmitter [angle, Doppler, delay] energy cubes."""
    if templates is None:
        templates = precompute_templates(config, patterns)
    waveform_templates, spatial = templates
    cubes = []
    for pattern_index in range(len(patterns)):
        cubes.append(_detection_cube(
            config, received, patterns[pattern_index],
            waveform_templates=waveform_templates[pattern_index],
            spatial_templates=spatial,
        ))
    return cubes


def integrated_detection_cubes(
    config: FrontEndConfig,
    received_frames: np.ndarray | Sequence[np.ndarray],
    patterns: Sequence[np.ndarray],
    *,
    templates: tuple[
        list[np.ndarray], np.ndarray
    ] | None = None,
) -> list[np.ndarray]:
    """Average per-transmitter matched-filter energy over a frame burst.

    Noncoherent integration averages the per-frame energy maps, which is the
    detector statistic used by the audited Gate G0-C results.  The mean keeps
    thresholds directly comparable across integration lengths.
    """
    if isinstance(received_frames, np.ndarray):
        frames = [received_frames]
    else:
        frames = list(received_frames)
    if not frames:
        raise ValueError("at least one received frame is required")
    if templates is None:
        templates = precompute_templates(config, patterns)
    cubes = [
        np.zeros((
            config.angle_grid_degrees.size,
            config.doppler_bins * config.delay_bins,
        ), dtype=float)
        for _ in patterns
    ]
    for frame in frames:
        for code_index, cube in enumerate(
            detection_cubes(config, frame, patterns, templates=templates)
        ):
            cubes[code_index] += cube
    return [cube / len(frames) for cube in cubes]


def precompute_templates(
    config: FrontEndConfig,
    patterns: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Precompute waveform dictionaries and the shared spatial dictionary."""
    waveform_templates = [
        waveform_dictionary(pattern) for pattern in patterns
    ]
    spatial = spatial_dictionary(
        config.angle_grid_degrees, config.num_antennas
    )
    return waveform_templates, spatial


def calibrate_frame_threshold(
    config: FrontEndConfig,
    patterns: Sequence[np.ndarray],
    *,
    trials: int,
    frame_false_alarm_probability: float,
    seed: int = 20260831,
    batch_size: int = 250,
    integration_frames: int = 1,
    templates: tuple[
        list[np.ndarray], np.ndarray
    ] | None = None,
) -> float:
    """Empirical max-map threshold for one frame-level false alarm bound."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if integration_frames <= 0:
        raise ValueError("integration_frames must be positive")
    if not 0.0 < frame_false_alarm_probability < 1.0:
        raise ValueError("frame false-alarm probability must lie in (0, 1)")
    if templates is None:
        templates = precompute_templates(config, patterns)
    waveform_templates, spatial = templates
    rng = np.random.default_rng(seed)
    frame_maxima = np.empty(0, dtype=float)
    for offset in range(0, trials, batch_size):
        count = min(batch_size, trials - offset)
        noise = np.sqrt(config.noise_variance / 2.0) * (
            rng.standard_normal(
                (count, integration_frames,
                 config.num_antennas, config.frame_samples)
            )
            + 1j * rng.standard_normal(
                (count, integration_frames,
                 config.num_antennas, config.frame_samples)
            )
        )
        batch_maxima = np.full(count, -np.inf)
        for dictionary in waveform_templates:
            dd_energy = noise @ dictionary.conj().T
            cube = np.abs(np.einsum(
                "ai,tfic->tfac", spatial.conj(), dd_energy
            )) ** 2
            integrated = cube.mean(axis=1)
            batch_maxima = np.maximum(
                batch_maxima, np.max(integrated, axis=(1, 2))
            )
        frame_maxima = np.concatenate((frame_maxima, batch_maxima))
    return float(np.quantile(
        frame_maxima, 1.0 - frame_false_alarm_probability, method="higher"
    ))


def calibrate_sidelobe_aware_threshold(
    config: FrontEndConfig,
    patterns: Sequence[np.ndarray],
    frames: Iterable[tuple[np.ndarray | Sequence[np.ndarray],
                           tuple[BistaticPath, ...] | None]],
    *,
    templates: tuple[list[np.ndarray], np.ndarray] | None = None,
    frame_false_alarm_probability: float,
    angle_guard: int = 2,
    dd_guard: int = 1,
) -> float:
    """Calibrate CFAR threshold above the reference-path sidelobe floor.

    Pure-noise calibration is optimistic when strong multistatic echoes leave
    deterministic waveform and array sidelobes.  This routine computes the
    max matched-filter energy outside the true peak neighborhoods on held-out
    single-target frames, so the threshold is set above the interference floor
    that a target itself produces.
    """
    if not 0.0 < frame_false_alarm_probability < 1.0:
        raise ValueError("frame false-alarm probability must lie in (0, 1)")
    if templates is None:
        templates = precompute_templates(config, patterns)
    maxima = []
    for received, true_paths in frames:
        cubes = integrated_detection_cubes(
            config, received, patterns, templates=templates
        )
        working = [
            cube.reshape((
                config.angle_grid_degrees.size,
                config.doppler_bins,
                config.delay_bins,
            )).copy()
            for cube in cubes
        ]
        if true_paths is not None:
            for path in true_paths:
                if path.transmitter_id >= len(working):
                    continue
                angle_index = int(np.argmin(np.abs(
                    config.angle_grid_degrees
                    - np.rad2deg(path.receive_azimuth_rad)
                )))
                doppler_index = int(np.round(
                    config.wrapped_doppler_bin(path.doppler_hz)
                )) % config.doppler_bins
                delay_index = int(np.round(
                    config.delay_bin(path.delay_s)
                )) % config.delay_bins
                for da in range(-angle_guard, angle_guard + 1):
                    a = angle_index + da
                    if not 0 <= a < working[path.transmitter_id].shape[0]:
                        continue
                    for dk in range(-dd_guard, dd_guard + 1):
                        for dl in range(-dd_guard, dd_guard + 1):
                            working[path.transmitter_id][
                                a,
                                (doppler_index + dk) % config.doppler_bins,
                                (delay_index + dl) % config.delay_bins,
                            ] = -np.inf
        maxima.append(float(max(
            np.max(cube) for cube in working
        )))
    if not maxima:
        raise ValueError("at least one calibration frame is required")
    return float(np.quantile(
        np.asarray(maxima), 1.0 - frame_false_alarm_probability,
        method="higher",
    ))


def extract_peaks(
    config: FrontEndConfig,
    received: np.ndarray,
    patterns: Sequence[np.ndarray],
    threshold: float,
    *,
    templates: tuple[
        list[np.ndarray], np.ndarray
    ] | None = None,
    angle_guard: int = 2,
    dd_guard: int = 1,
) -> list[tuple[int, int, int, int, float]]:
    """Extract unknown-count NMS peaks with their raw energy scores."""
    peaks: list[tuple[int, int, int, int, float]] = []
    shape = (config.doppler_bins, config.delay_bins)
    for transmitter_id, cube in enumerate(
        detection_cubes(config, received, patterns, templates=templates)
    ):
        cube3 = cube.reshape((config.angle_grid_degrees.size,) + shape)
        for angle_index, doppler_index, delay_index in threshold_nms_3d(
            cube, threshold, shape,
            angle_guard=angle_guard, dd_guard=dd_guard,
        ):
            peaks.append((
                transmitter_id, angle_index, doppler_index, delay_index,
                float(cube3[angle_index, doppler_index, delay_index]),
            ))
    return peaks


def extract_integrated_peaks(
    config: FrontEndConfig,
    received_frames: np.ndarray | Sequence[np.ndarray],
    patterns: Sequence[np.ndarray],
    threshold: float,
    *,
    templates: tuple[
        list[np.ndarray], np.ndarray
    ] | None = None,
    angle_guard: int = 2,
    dd_guard: int = 1,
) -> list[tuple[int, int, int, int, float]]:
    """Extract unknown-count peaks from noncoherently integrated energy."""
    peaks: list[tuple[int, int, int, int, float]] = []
    shape = (config.doppler_bins, config.delay_bins)
    for transmitter_id, cube in enumerate(
        integrated_detection_cubes(
            config, received_frames, patterns, templates=templates
        )
    ):
        cube3 = cube.reshape((config.angle_grid_degrees.size,) + shape)
        for angle_index, doppler_index, delay_index in threshold_nms_3d(
            cube, threshold, shape,
            angle_guard=angle_guard, dd_guard=dd_guard,
        ):
            peaks.append((
                transmitter_id, angle_index, doppler_index, delay_index,
                float(cube3[angle_index, doppler_index, delay_index]),
            ))
    return peaks


def _parabolic_offset(
    center: float, left: float, right: float
) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) <= 1e-15:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def peak_estimate(
    config: FrontEndConfig,
    cube: np.ndarray,
    peak: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Parabolic interpolation of angle, Doppler, and delay around a peak."""
    cube = np.asarray(cube).reshape(
        (config.angle_grid_degrees.size, config.doppler_bins, config.delay_bins)
    )
    angle_index, doppler_index, delay_index = peak
    angles = config.angle_grid_degrees
    angle_scale = angles[1] - angles[0] if angles.size > 1 else 1.0
    if 0 < angle_index < angles.size - 1:
        angle_offset = _parabolic_offset(
            cube[angle_index, doppler_index, delay_index],
            cube[angle_index - 1, doppler_index, delay_index],
            cube[angle_index + 1, doppler_index, delay_index],
        )
    else:
        angle_offset = 0.0
    doppler_left = cube[
        angle_index, (doppler_index - 1) % config.doppler_bins, delay_index
    ]
    doppler_right = cube[
        angle_index, (doppler_index + 1) % config.doppler_bins, delay_index
    ]
    doppler_offset = _parabolic_offset(
        cube[angle_index, doppler_index, delay_index],
        doppler_left, doppler_right,
    )
    delay_left = cube[
        angle_index, doppler_index, (delay_index - 1) % config.delay_bins
    ]
    delay_right = cube[
        angle_index, doppler_index, (delay_index + 1) % config.delay_bins
    ]
    delay_offset = _parabolic_offset(
        cube[angle_index, doppler_index, delay_index],
        delay_left, delay_right,
    )
    angle_degrees = angles[angle_index] + angle_offset * angle_scale
    doppler_bin = float(doppler_index) + doppler_offset
    delay_bin = float(delay_index) + delay_offset
    delay_s, doppler_hz = config.estimate_from_bins(delay_bin, doppler_bin)
    return angle_degrees, delay_s, doppler_hz


def _curvature_sigma_bins(
    center: float, left: float, right: float
) -> float:
    """Local Gaussian width estimate from the discrete matched-filter peak."""
    denominator = left - 2.0 * center + right
    if denominator >= -1e-15 or center <= 1e-15:
        return 1.0
    return float(np.sqrt(center / max(-denominator, 1e-15)))


def peak_measurement_sigmas(
    config: FrontEndConfig,
    cube: np.ndarray,
    peak: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Return angle, range, and Doppler scales from local peak curvature.

    Under a locally quadratic log-likelihood, the inverse of the negative
    Hessian of the matched-filter energy surface is a Fisher-type precision
    estimate.  The returned values are therefore per-path measurement scales
    that can replace the fixed global sigmas used by the association back end.
    """
    cube = np.asarray(cube).reshape(
        (config.angle_grid_degrees.size, config.doppler_bins, config.delay_bins)
    )
    angle_index, doppler_index, delay_index = peak
    center = float(cube[peak])
    angles = config.angle_grid_degrees
    angle_scale = angles[1] - angles[0] if angles.size > 1 else 1.0
    if 0 < angle_index < angles.size - 1:
        angle_sigma = _curvature_sigma_bins(
            center,
            float(cube[angle_index - 1, doppler_index, delay_index]),
            float(cube[angle_index + 1, doppler_index, delay_index]),
        )
    else:
        angle_sigma = 1.0
    doppler_sigma = _curvature_sigma_bins(
        center,
        float(cube[angle_index, (doppler_index - 1) % config.doppler_bins,
                     delay_index]),
        float(cube[angle_index, (doppler_index + 1) % config.doppler_bins,
                     delay_index]),
    )
    delay_sigma = _curvature_sigma_bins(
        center,
        float(cube[angle_index, doppler_index,
                     (delay_index - 1) % config.delay_bins]),
        float(cube[angle_index, doppler_index,
                     (delay_index + 1) % config.delay_bins]),
    )
    range_sigma_m = delay_sigma * config.delay_resolution_m
    angle_sigma_rad = (
        angle_sigma * angle_scale * np.pi / 180.0
    )
    doppler_sigma_hz = doppler_sigma * config.doppler_resolution_hz
    return (
        float(max(angle_sigma_rad, 1e-6)),
        float(max(range_sigma_m, 1e-6)),
        float(max(doppler_sigma_hz, 1e-6)),
    )


def peaks_to_candidates(
    config: FrontEndConfig,
    peaks: Iterable[tuple[int, int, int, int, float]],
    cubes: Sequence[np.ndarray],
    calibrator: IsotonicProbabilityCalibrator,
) -> list[PathCandidate]:
    """Convert thresholded peaks into calibrated association candidates."""
    candidates = []
    for transmitter_id, angle_index, doppler_index, delay_index, energy in peaks:
        if transmitter_id >= len(cubes):
            raise ValueError("peak transmitter_id is out of range")
        angle_degrees, delay_s, doppler_hz = peak_estimate(
            config, cubes[transmitter_id],
            (angle_index, doppler_index, delay_index),
        )
        angle_sigma_rad, range_sigma_m, doppler_sigma_hz = (
            peak_measurement_sigmas(
                config, cubes[transmitter_id],
                (angle_index, doppler_index, delay_index),
            )
        )
        candidates.append(PathCandidate(
            transmitter_id=transmitter_id,
            delay_s=float(max(delay_s, 1e-9)),
            doppler_hz=float(doppler_hz),
            receive_azimuth_rad=float(np.deg2rad(angle_degrees)),
            confidence=float(calibrator(energy)),
            range_sigma_m=float(range_sigma_m),
            angle_sigma_rad=float(angle_sigma_rad),
            doppler_sigma_hz=float(doppler_sigma_hz),
        ))
    return candidates


def _cyclic_bin_distance(first: float, second: float, size: int) -> int:
    return int(min(
        (first - second) % size,
        (second - first) % size,
    ))


def peak_matches_path(
    config: FrontEndConfig,
    peak: tuple[int, int, int, int, float],
    path: BistaticPath,
    *,
    angle_tolerance_degrees: float,
    range_tolerance_m: float = 15.0,
    doppler_tolerance_hz: float = 25.0,
) -> bool:
    transmitter_id, angle_index, doppler_index, delay_index = peak[:4]
    if transmitter_id != path.transmitter_id:
        return False
    angle_error = abs(
        config.angle_grid_degrees[angle_index]
        - np.rad2deg(path.receive_azimuth_rad)
    )
    delay_s, doppler_hz = config.estimate_from_bins(
        float(delay_index), float(doppler_index)
    )
    range_error = abs(
        delay_s * PROPAGATION_SPEED - path.delay_s * PROPAGATION_SPEED
    )
    doppler_error = abs(doppler_hz - path.doppler_hz)
    return (
        angle_error <= angle_tolerance_degrees
        and range_error <= range_tolerance_m
        and doppler_error <= doppler_tolerance_hz
    )


def calibrate_confidence(
    config: FrontEndConfig,
    patterns: Sequence[np.ndarray],
    frames: Iterable[tuple[np.ndarray, tuple[BistaticPath, ...] | None]],
    *,
    collect_threshold: float,
    templates: tuple[
        list[np.ndarray], np.ndarray
    ] | None = None,
    angle_guard: int = 2,
    dd_guard: int = 1,
) -> IsotonicProbabilityCalibrator:
    """Fit isotonic path-existence probabilities from labelled front-end peaks."""
    scores = []
    labels = []
    for received, true_paths in frames:
        for transmitter_id, cube in enumerate(
            integrated_detection_cubes(
                config, received, patterns, templates=templates
            )
        ):
            shape = (config.doppler_bins, config.delay_bins)
            cube3 = cube.reshape((config.angle_grid_degrees.size,) + shape)
            for peak in threshold_nms_3d(
                cube, collect_threshold, shape,
                angle_guard=angle_guard, dd_guard=dd_guard,
            ):
                full_peak = (transmitter_id,) + peak
                scores.append(float(cube3[peak]))
                if true_paths is None:
                    labels.append(0.0)
                    continue
                matched = any(
                    peak_matches_path(
                        config, full_peak, path,
                        angle_tolerance_degrees=(
                            min(4.0, config.angle_grid_degrees[1]
                                - config.angle_grid_degrees[0])
                        ),
                    )
                    for path in true_paths
                )
                labels.append(1.0 if matched else 0.0)
    return fit_isotonic_probability(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=float),
    )
