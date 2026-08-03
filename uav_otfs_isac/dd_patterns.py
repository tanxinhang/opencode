"""Collision metrics and small-scale DD pattern assignment methods."""

from __future__ import annotations

from itertools import product
from itertools import combinations

import numpy as np

from .otfs_physical import (
    dd_cross_ambiguity,
    delay_doppler_path,
    otfs_modulate,
)


def pattern_collision_matrix(patterns, delay_shifts, doppler_shifts, weights=None):
    """Average squared cross ambiguity for every ordered pattern pair."""
    patterns = tuple(np.asarray(pattern, dtype=complex) for pattern in patterns)
    if not patterns:
        raise ValueError("at least one pattern is required")
    shape = patterns[0].shape
    if any(pattern.shape != shape for pattern in patterns):
        raise ValueError("all patterns must have the same shape")
    delays = np.asarray(delay_shifts, dtype=float)
    dopplers = np.asarray(doppler_shifts, dtype=float)
    if delays.size == 0 or dopplers.size == 0:
        raise ValueError("delay and Doppler grids must be nonempty")
    if weights is None:
        integration_weights = np.ones((dopplers.size, delays.size), dtype=float)
    else:
        integration_weights = np.asarray(weights, dtype=float)
        if integration_weights.shape != (dopplers.size, delays.size):
            raise ValueError("weights must match [doppler, delay] grid")
        if np.any(integration_weights < 0.0):
            raise ValueError("weights must be nonnegative")
    total_weight = integration_weights.sum()
    if total_weight <= 0.0:
        raise ValueError("weights must contain positive mass")
    integration_weights = integration_weights / total_weight
    matrix = np.empty((len(patterns), len(patterns)), dtype=float)
    for i, first in enumerate(patterns):
        for j, second in enumerate(patterns):
            ambiguity = dd_cross_ambiguity(
                first, second, delays, dopplers
            )
            matrix[i, j] = float(
                np.sum(integration_weights * np.abs(ambiguity) ** 2)
            )
    return matrix


def assignment_cost(assignment, uav_pair_weights, pattern_collisions) -> float:
    """Weighted sum of pairwise cross-ambiguity collision costs."""
    assignment = tuple(int(value) for value in assignment)
    pair_weights = np.asarray(uav_pair_weights, dtype=float)
    collisions = np.asarray(pattern_collisions, dtype=float)
    if pair_weights.shape != (len(assignment), len(assignment)):
        raise ValueError("uav_pair_weights must match assignment length")
    if not np.all(np.isfinite(pair_weights)) or np.any(pair_weights < 0.0):
        raise ValueError("uav_pair_weights must be finite and nonnegative")
    if not np.allclose(pair_weights, pair_weights.T):
        raise ValueError("uav_pair_weights must be symmetric")
    if not np.all(np.isfinite(collisions)) or np.any(collisions < 0.0):
        raise ValueError("pattern_collisions must be finite and nonnegative")
    if collisions.ndim == 2:
        num_patterns = collisions.shape[0]
        if collisions.shape[1] != num_patterns:
            raise ValueError("pattern_collisions must be square")
    elif collisions.ndim == 4:
        if collisions.shape[:2] != pair_weights.shape:
            raise ValueError("pair-specific collisions must match UAV dimensions")
        num_patterns = collisions.shape[2]
        if collisions.shape[3] != num_patterns:
            raise ValueError("pattern dimensions must be square")
    else:
        raise ValueError("pattern_collisions must be a matrix or four-dimensional tensor")
    if any(value < 0 or value >= num_patterns for value in assignment):
        raise ValueError("assignment contains an invalid pattern index")
    cost = 0.0
    for i in range(len(assignment)):
        for j in range(i + 1, len(assignment)):
            collision = (
                collisions[assignment[i], assignment[j]]
                if collisions.ndim == 2
                else collisions[i, j, assignment[i], assignment[j]]
            )
            cost += pair_weights[i, j] * collision
    return float(cost)


def exhaustive_pattern_assignment(uav_pair_weights, pattern_collisions):
    """Exact small-scale collision-minimizing pattern assignment."""
    weights = np.asarray(uav_pair_weights, dtype=float)
    collisions = np.asarray(pattern_collisions, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("uav_pair_weights must be square")
    num_patterns = collisions.shape[0] if collisions.ndim == 2 else collisions.shape[2]
    candidates = product(range(num_patterns), repeat=weights.shape[0])
    return min(
        candidates,
        key=lambda assignment: (
            assignment_cost(assignment, weights, collisions), assignment
        ),
    )


def greedy_pattern_assignment(uav_pair_weights, pattern_collisions):
    """Weighted-degree ordering followed by local single-UAV improvements."""
    weights = np.asarray(uav_pair_weights, dtype=float)
    collisions = np.asarray(pattern_collisions, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("uav_pair_weights must be square")
    if collisions.ndim == 2:
        num_patterns = collisions.shape[0]
    elif collisions.ndim == 4 and collisions.shape[:2] == weights.shape:
        num_patterns = collisions.shape[2]
    else:
        raise ValueError("pattern_collisions dimensions are invalid")
    num_uavs = weights.shape[0]
    order = sorted(range(num_uavs), key=lambda i: (-weights[i].sum(), i))
    assignment = [-1] * num_uavs
    for uav in order:
        assigned = [index for index in range(num_uavs) if assignment[index] >= 0]
        if not assigned:
            assignment[uav] = 0
            continue
        assignment[uav] = min(
            range(num_patterns),
            key=lambda pattern: (
                sum(
                    weights[min(uav, other), max(uav, other)] * (
                        collisions[pattern, assignment[other]]
                        if collisions.ndim == 2
                        else collisions[
                            min(uav, other), max(uav, other),
                            pattern if uav < other else assignment[other],
                            assignment[other] if uav < other else pattern,
                        ]
                    )
                    for other in assigned
                ),
                pattern,
            ),
        )
    improved = True
    while improved:
        improved = False
        current = assignment_cost(assignment, weights, collisions)
        for uav in order:
            incumbent = assignment[uav]
            best = incumbent
            best_cost = current
            for pattern in range(num_patterns):
                trial = list(assignment); trial[uav] = pattern
                cost = assignment_cost(trial, weights, collisions)
                if cost < best_cost - 1e-15:
                    best, best_cost = pattern, cost
            if best != incumbent:
                assignment[uav] = best
                current = best_cost
                improved = True
    return tuple(assignment)


def geometry_collision_tensor(
    patterns, delays, dopplers, gains=None,
):
    """Pair-specific normalized interference at nominal UAV path offsets.

    Templates are generated through the same absolute-delay/Doppler channel
    used by the waveform experiment.  This avoids assuming that fractional
    circular delay and Doppler operators can be replaced by a simple signed
    relative shift under the finite-frame model.
    """
    patterns = tuple(np.asarray(pattern, dtype=complex) for pattern in patterns)
    delays = np.asarray(delays, dtype=float)
    dopplers = np.asarray(dopplers, dtype=float)
    if delays.shape != dopplers.shape:
        raise ValueError("delays and dopplers must have the same UAV dimension")
    amplitudes = (
        np.ones(delays.size, dtype=complex)
        if gains is None else np.asarray(gains, dtype=complex)
    )
    if amplitudes.shape != delays.shape:
        raise ValueError("gains must match the UAV dimension")
    if not patterns:
        raise ValueError("at least one pattern is required")
    shape = patterns[0].shape
    if any(pattern.shape != shape for pattern in patterns):
        raise ValueError("all patterns must have the same shape")
    if len(shape) != 2:
        raise ValueError("patterns must be two-dimensional")
    unit_templates = np.empty(
        (delays.size, len(patterns), shape[0] * shape[1]), dtype=complex
    )
    for i in range(delays.size):
        for pattern_index, pattern in enumerate(patterns):
            unit_templates[i, pattern_index] = delay_doppler_path(
                otfs_modulate(pattern), delays[i], dopplers[i], shape[0]
            )
    tensor = np.zeros((delays.size, delays.size, len(patterns), len(patterns)))
    for i in range(delays.size):
        for j in range(i + 1, delays.size):
            for first_index in range(len(patterns)):
                for second_index in range(len(patterns)):
                    first_template = unit_templates[i, first_index]
                    second_template = unit_templates[j, second_index]
                    first_energy = max(
                        float(np.vdot(first_template, first_template).real), 1e-15
                    )
                    second_energy = max(
                        float(np.vdot(second_template, second_template).real), 1e-15
                    )
                    cross_power = abs(np.vdot(first_template, second_template)) ** 2
                    forward = cross_power / first_energy ** 2
                    reverse = cross_power / second_energy ** 2
                    desired_i = max(abs(amplitudes[i]) ** 2, 1e-15)
                    desired_j = max(abs(amplitudes[j]) ** 2, 1e-15)
                    value = 0.5 * (
                        abs(amplitudes[j]) ** 2 / desired_i * forward
                        + abs(amplitudes[i]) ** 2 / desired_j * reverse
                    )
                    tensor[i, j, first_index, second_index] = value
                    tensor[j, i, second_index, first_index] = value
    return tensor


def full_grid_ambiguity_metrics(patterns):
    """Return full integer-DD auto-sidelobe and cross-ambiguity metrics."""
    patterns = tuple(np.asarray(pattern, dtype=complex) for pattern in patterns)
    if not patterns:
        raise ValueError("at least one pattern is required")
    shape = patterns[0].shape
    if len(shape) != 2 or any(pattern.shape != shape for pattern in patterns):
        raise ValueError("patterns must share one two-dimensional shape")
    delays = np.arange(shape[1])
    dopplers = np.arange(shape[0])
    auto_peak_sidelobes = []
    for pattern in patterns:
        ambiguity = np.abs(dd_cross_ambiguity(
            pattern, pattern, delays, dopplers
        )) ** 2
        ambiguity[0, 0] = 0.0
        auto_peak_sidelobes.append(float(np.max(ambiguity)))
    cross_peaks = []
    for first, second in combinations(patterns, 2):
        forward = np.abs(dd_cross_ambiguity(
            first, second, delays, dopplers
        )) ** 2
        reverse = np.abs(dd_cross_ambiguity(
            second, first, delays, dopplers
        )) ** 2
        cross_peaks.append(float(max(np.max(forward), np.max(reverse))))
    return {
        "auto_peak_sidelobes": auto_peak_sidelobes,
        "worst_auto_peak_sidelobe": max(auto_peak_sidelobes),
        "cross_peaks": cross_peaks,
        "worst_cross_peak": max(cross_peaks) if cross_peaks else 0.0,
        "auto_peak_spread": max(auto_peak_sidelobes) - min(auto_peak_sidelobes),
    }


def select_balanced_ambiguity_codebook(candidates, codebook_size):
    """Exhaustively select a small low-ambiguity, quality-balanced codebook."""
    candidates = tuple(np.asarray(pattern, dtype=complex) for pattern in candidates)
    if codebook_size <= 0 or codebook_size > len(candidates):
        raise ValueError("codebook_size must be positive and available")
    shape = candidates[0].shape
    if len(shape) != 2 or any(candidate.shape != shape for candidate in candidates):
        raise ValueError("candidates must share one two-dimensional shape")
    delays = np.arange(shape[1])
    dopplers = np.arange(shape[0])
    auto_peaks = np.empty(len(candidates), dtype=float)
    for index, candidate in enumerate(candidates):
        ambiguity = np.abs(dd_cross_ambiguity(
            candidate, candidate, delays, dopplers
        )) ** 2
        ambiguity[0, 0] = 0.0
        auto_peaks[index] = np.max(ambiguity)
    cross_peaks = np.zeros((len(candidates), len(candidates)), dtype=float)
    for first, second in combinations(range(len(candidates)), 2):
        forward = np.abs(dd_cross_ambiguity(
            candidates[first], candidates[second], delays, dopplers
        )) ** 2
        reverse = np.abs(dd_cross_ambiguity(
            candidates[second], candidates[first], delays, dopplers
        )) ** 2
        value = max(np.max(forward), np.max(reverse))
        cross_peaks[first, second] = cross_peaks[second, first] = value
    best = None
    for indices in combinations(range(len(candidates)), codebook_size):
        selected_auto = [float(auto_peaks[index]) for index in indices]
        selected_cross = [
            float(cross_peaks[first, second])
            for first, second in combinations(indices, 2)
        ]
        metrics = {
            "auto_peak_sidelobes": selected_auto,
            "worst_auto_peak_sidelobe": max(selected_auto),
            "cross_peaks": selected_cross,
            "worst_cross_peak": max(selected_cross) if selected_cross else 0.0,
            "auto_peak_spread": max(selected_auto) - min(selected_auto),
        }
        key = (
            max(metrics["worst_auto_peak_sidelobe"],
                metrics["worst_cross_peak"]),
            metrics["auto_peak_spread"],
            indices,
        )
        if best is None or key < best[0]:
            best = key, indices, metrics
    return best[1], best[2]
