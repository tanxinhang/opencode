"""Separable angle-delay-Doppler matched filtering and peak extraction."""

from __future__ import annotations

import numpy as np

from .otfs_physical import (
    delay_doppler_path,
    otfs_modulate,
    ula_steering_vector,
)


def waveform_dictionary(pattern):
    """Return unit-energy integer-DD templates indexed as [k*M+l, sample]."""
    pattern = np.asarray(pattern, dtype=complex)
    if pattern.ndim != 2 or 0 in pattern.shape:
        raise ValueError("pattern must be a nonempty two-dimensional grid")
    reference = otfs_modulate(pattern)
    dictionary = np.empty((pattern.size, pattern.size), dtype=complex)
    for k in range(pattern.shape[0]):
        for l in range(pattern.shape[1]):
            dictionary[k * pattern.shape[1] + l] = delay_doppler_path(
                reference, float(l), float(k), pattern.shape[0]
            )
    norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    return dictionary / np.maximum(norms, 1e-15)


def spatial_dictionary(angle_grid_degrees, num_antennas, spatial_code=None):
    """Return unit-energy array templates indexed by the angle grid."""
    angles = np.asarray(angle_grid_degrees, dtype=float)
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("angle_grid_degrees must be a nonempty finite vector")
    result = []
    for angle in angles:
        vector = ula_steering_vector(num_antennas, angle)
        if spatial_code is not None:
            code = np.asarray(spatial_code, dtype=complex)
            if code.shape != vector.shape or np.linalg.norm(code) <= 0.0:
                raise ValueError("spatial_code must be nonzero and antenna-length")
            vector = vector * code
            vector = vector / np.linalg.norm(vector)
        result.append(vector)
    return np.asarray(result)


def separable_detection_cube(received, waveform_templates, spatial_templates):
    """Compute [angle, DD-template] energy without a Kronecker dictionary."""
    observations = np.asarray(received, dtype=complex)
    waveform_templates = np.asarray(waveform_templates, dtype=complex)
    spatial_templates = np.asarray(spatial_templates, dtype=complex)
    if observations.ndim != 2:
        raise ValueError("received must have shape [antenna, sample]")
    if waveform_templates.ndim != 2 or observations.shape[1] != waveform_templates.shape[1]:
        raise ValueError("waveform template sample length must match received")
    if spatial_templates.ndim != 2 or observations.shape[0] != spatial_templates.shape[1]:
        raise ValueError("spatial template antenna length must match received")
    dd_by_antenna = observations @ waveform_templates.conj().T
    return np.abs(spatial_templates.conj() @ dd_by_antenna) ** 2


def probe_coded_spatial_template(spatial_time_template, probe_code):
    """Lift an [antenna,time] path into [probe,antenna,time] observations."""
    template = np.asarray(spatial_time_template, dtype=complex)
    code = np.asarray(probe_code, dtype=complex)
    if template.ndim != 2 or code.ndim != 1 or code.size == 0:
        raise ValueError("template must be 2D and probe_code a nonempty vector")
    norm = np.linalg.norm(code)
    if norm <= 0.0:
        raise ValueError("probe_code must have nonzero energy")
    return (code / norm)[:, None, None] * template[None, :, :]


def decode_probe_code(received, probe_code):
    """Matched-filter a resolved probe dimension into [antenna,time]."""
    observations = np.asarray(received, dtype=complex)
    code = np.asarray(probe_code, dtype=complex)
    if observations.ndim != 3 or code.shape != (observations.shape[0],):
        raise ValueError("probe code must match received probe dimension")
    norm = np.linalg.norm(code)
    if norm <= 0.0:
        raise ValueError("probe_code must have nonzero energy")
    return np.einsum("p,paf->af", (code / norm).conj(), observations)


def threshold_nms_3d(
    energy_cube, threshold, dd_shape, angle_guard=1, dd_guard=1,
):
    """Extract unknown-count peaks from [angle, flattened DD] by cyclic DD NMS."""
    cube = np.asarray(energy_cube, dtype=float)
    if cube.ndim != 2 or cube.shape[1] != int(np.prod(dd_shape)):
        raise ValueError("energy_cube must have shape [angle, prod(dd_shape)]")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    if angle_guard < 0 or dd_guard < 0:
        raise ValueError("guard radii must be nonnegative")
    working = cube.reshape((cube.shape[0],) + tuple(dd_shape)).copy()
    peaks = []
    while float(np.max(working)) >= threshold:
        peak = tuple(int(value) for value in np.unravel_index(
            np.argmax(working), working.shape
        ))
        peaks.append(peak)
        angle, doppler, delay = peak
        for da in range(-angle_guard, angle_guard + 1):
            angle_index = angle + da
            if not 0 <= angle_index < working.shape[0]:
                continue
            for dk in range(-dd_guard, dd_guard + 1):
                for dl in range(-dd_guard, dd_guard + 1):
                    working[
                        angle_index,
                        (doppler + dk) % working.shape[1],
                        (delay + dl) % working.shape[2],
                    ] = -np.inf
    return peaks
