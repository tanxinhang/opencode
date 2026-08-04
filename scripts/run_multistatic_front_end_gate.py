"""Gate G0-C: waveform front end feeding unknown-cardinality association.

This gate replaces the synthetic candidate oracle of G0-B with a toy
matched-filter/CFAR front end.  A ULA receiver observes superposed
transmitter-specific OTFS pilot signatures, extracts unknown-count
angle--delay--Doppler peaks, calibrates their path-existence probabilities on
held-out frames, and feeds the resulting ``PathCandidate`` objects into the
physics-constrained association back end.

The DD grid uses explicitly declared delay and Doppler resolutions.  This is
the repository's lightweight Gate-0 waveform model, not a bandwidth-consistent
SDR; no equal-bandwidth or communication-rate claim is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.front_end import (
    FrontEndConfig,
    calibrate_confidence,
    calibrate_frame_threshold,
    calibrate_sidelobe_aware_threshold,
    extract_integrated_peaks,
    identity_patterns,
    integrated_detection_cubes,
    peaks_to_candidates,
    precompute_templates,
    simulate_received,
)
from uav_otfs_isac.multistatic_baselines import _dbscan_labels, _project
from uav_otfs_isac.multistatic_model_selection import bic_conflict_association
from uav_otfs_isac.multistatic_targets import (
    KinematicNode,
    PhysicalTarget,
    generate_bistatic_paths,
)


CARRIER_HZ = 5.9e9
PROPAGATION_SPEED = 299_792_458.0


def transmitter_geometry(count: int) -> tuple[KinematicNode, ...]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return tuple(KinematicNode(
        (100.0 * np.cos(angle), 100.0 * np.sin(angle)), (0.0, 0.0)
    ) for angle in angles)


def receiver_node() -> KinematicNode:
    return KinematicNode((0.0, 0.0), (0.0, 0.0))


def draw_targets(
    rng: np.random.Generator, count: int
) -> tuple[PhysicalTarget, ...]:
    """Draw separated targets within the declared DD-grid ambiguity region."""
    if count == 1:
        angles = rng.uniform(-40.0, 20.0, 1)
    elif count == 2:
        angle0 = rng.uniform(-40.0, -15.0)
        angle1 = angle0 + rng.uniform(20.0, 35.0)
        angles = np.asarray((angle0, angle1))
    else:
        raise ValueError("this gate currently uses one or two targets")
    ranges = 55.0 + rng.uniform(-8.0, 8.0, count)
    velocities = rng.uniform(-1.5, 1.5, (count, 2))
    return tuple(PhysicalTarget(
        target_id=index,
        position=ranges[index] * np.asarray((
            np.cos(np.deg2rad(angles[index])),
            np.sin(np.deg2rad(angles[index])),
        )),
        velocity=velocities[index],
    ) for index in range(count))


def random_path_gains(
    paths: tuple, amplitude: float, rng: np.random.Generator,
    rayleigh_fading: bool = False,
) -> list[complex]:
    if rayleigh_fading:
        # Unit-mean-power complex Gaussian: E[|h|^2] = 1.
        fading = (
            rng.standard_normal(len(paths))
            + 1j * rng.standard_normal(len(paths))
        ) / np.sqrt(2.0)
        return [
            amplitude * complex(value) for value in fading
        ]
    return [
        amplitude * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        for _ in paths
    ]


def scenario_frames(
    config: FrontEndConfig,
    patterns: list[np.ndarray],
    nodes: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    amplitude: float,
    rng: np.random.Generator,
    count: int,
    frames: int,
    integration_frames: int = 1,
    rayleigh_fading: bool = False,
) -> list[tuple[np.ndarray, tuple | None]]:
    if integration_frames <= 0:
        raise ValueError("integration_frames must be positive")
    output = []
    for _ in range(frames):
        targets = draw_targets(rng, count)
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        received = [
            simulate_received(
                config, patterns, paths,
                random_path_gains(
                    paths, amplitude, rng, rayleigh_fading
                ), rng,
            )
            for _ in range(integration_frames)
        ]
        received = received[0] if integration_frames == 1 else received
        output.append((received, paths))
    for _ in range(max(1, frames // 2)):
        received = [
            simulate_received(config, patterns, (), (), rng)
            for _ in range(integration_frames)
        ]
        received = received[0] if integration_frames == 1 else received
        output.append((received, None))
    return output


def match_candidate_labels(
    candidates: list,
    paths: tuple,
    config: FrontEndConfig,
) -> dict[int, int]:
    """Greedy one-to-one matching, returning candidate index to target id."""
    labels = {}
    used = set()
    for path in paths:
        best_index = None
        best_error = np.inf
        for index, candidate in enumerate(candidates):
            if index in used or candidate.transmitter_id != path.transmitter_id:
                continue
            angle_error = abs(np.angle(np.exp(
                1j * (candidate.receive_azimuth_rad - path.receive_azimuth_rad)
            )))
            range_error = abs(
                candidate.delay_s * PROPAGATION_SPEED
                - path.delay_s * PROPAGATION_SPEED
            )
            doppler_error = abs(candidate.doppler_hz - path.doppler_hz)
            error = (
                angle_error > np.deg2rad(4.0)
                or range_error > 15.0
                or doppler_error > 25.0
            )
            if not error and range_error + doppler_error < best_error:
                best_error = range_error + doppler_error
                best_index = index
        if best_index is not None:
            used.add(best_index)
            labels[best_index] = path.target_id
    return labels


def match_candidate_indices(
    candidates: list,
    paths: tuple,
    config: FrontEndConfig,
) -> set[int]:
    """Return candidate indices matched to true paths."""
    return set(match_candidate_labels(candidates, paths, config))


def candidate_truth(
    candidates: list,
    paths: tuple,
    config: FrontEndConfig,
) -> tuple[int, int]:
    """Count true-path matches and remaining false candidates."""
    matched = match_candidate_indices(candidates, paths, config)
    return len(matched), len(candidates) - len(matched)


def calibrate_view_probabilities(
    config: FrontEndConfig,
    patterns: list[np.ndarray],
    frames: list[tuple[np.ndarray, tuple | None]],
    *,
    calibrator,
    templates,
    threshold: float,
    order_confidence_threshold: float,
    angle_guard: int = 2,
    dd_guard: int = 1,
) -> tuple[list[float], list[float]]:
    """Estimate per-view false-target and false-extra probabilities.

    The unit of estimation is the single-target spatial component rather than
    the whole frame, matching the event used by the support gates: under
    ``H_{q-1}``, view ``m`` contributes one false support or one extra peak in
    a local component.  Only held-out calibration frames are used.
    """
    transmitter_count = len(patterns)
    false_target = np.zeros(transmitter_count)
    false_extra = np.zeros(transmitter_count)
    component_count = np.zeros(transmitter_count)
    for received, true_paths in frames:
        if true_paths is None:
            continue
        peaks = extract_integrated_peaks(
            config, received, patterns, threshold, templates=templates,
            angle_guard=angle_guard, dd_guard=dd_guard,
        )
        cubes = integrated_detection_cubes(
            config, received, patterns, templates=templates
        )
        candidates = peaks_to_candidates(config, peaks, cubes, calibrator)
        labels = match_candidate_labels(candidates, true_paths, config)
        high_confidence = [
            index for index, candidate in enumerate(candidates)
            if candidate.confidence >= order_confidence_threshold
        ]
        projected = _project(
            candidates,
            transmitter_geometry(transmitter_count),
            receiver_node(),
            PROPAGATION_SPEED,
        )
        if not projected:
            continue
        component_labels = _dbscan_labels(
            np.asarray([entry[1] for entry in projected]) / 12.0,
            radius=1.0, min_samples=2,
        )
        for label in range(int(component_labels.max()) + 1):
            component = [
                projected[index]
                for index in np.flatnonzero(component_labels == label)
            ]
            component_candidates = [entry[0] for entry in component]
            labeled_ids = {
                labels[candidates.index(candidate)]
                for candidate in component_candidates
                if candidates.index(candidate) in labels
            }
            if len(labeled_ids) != 1:
                continue
            component_high = [
                candidates.index(candidate)
                for candidate in component_candidates
                if candidate.confidence >= order_confidence_threshold
            ]
            for transmitter_id in range(transmitter_count):
                high_from_view = [
                    index for index in component_high
                    if candidates[index].transmitter_id == transmitter_id
                ]
                if not high_from_view:
                    continue
                component_count[transmitter_id] += 1.0
                if any(index not in labels for index in high_from_view):
                    false_target[transmitter_id] += 1.0
                if len(high_from_view) >= 2:
                    false_extra[transmitter_id] += 1.0
    p_target = false_target / np.maximum(component_count, 1.0)
    p_extra = false_extra / np.maximum(component_count, 1.0)
    return (
        [float(np.clip(value, 1e-4, 0.5)) for value in p_target],
        [float(np.clip(value, 1e-4, 0.5)) for value in p_extra],
    )


def calibrate_empirical_collision_support(
    config: FrontEndConfig,
    patterns: list[np.ndarray],
    frames: list[tuple[np.ndarray, tuple | None]],
    *,
    calibrator,
    templates,
    threshold: float,
    order_confidence_threshold: float,
    false_alarm_probability: float,
    angle_guard: int = 2,
) -> int | None:
    """Calibrate collision support from held-out single-target components.

    The null statistic is the number of distinct UAV views that provide two
    or more high-confidence candidates in one local component.  Using the
    empirical tail directly avoids the independence assumption of the
    Poisson-binomial model when false peaks are correlated across views.
    """
    transmitter_count = len(patterns)
    supports = []
    for received, true_paths in frames:
        if true_paths is None:
            continue
        peaks = extract_integrated_peaks(
            config, received, patterns, threshold, templates=templates,
            angle_guard=angle_guard,
        )
        cubes = integrated_detection_cubes(
            config, received, patterns, templates=templates
        )
        candidates = peaks_to_candidates(config, peaks, cubes, calibrator)
        labels = match_candidate_labels(candidates, true_paths, config)
        projected = _project(
            candidates,
            transmitter_geometry(transmitter_count),
            receiver_node(),
            PROPAGATION_SPEED,
        )
        if not projected:
            continue
        component_labels = _dbscan_labels(
            np.asarray([entry[1] for entry in projected]) / 12.0,
            radius=1.0, min_samples=2,
        )
        for label in range(int(component_labels.max()) + 1):
            component = [
                projected[index]
                for index in np.flatnonzero(component_labels == label)
            ]
            component_candidates = [entry[0] for entry in component]
            labeled_ids = {
                labels[candidates.index(candidate)]
                for candidate in component_candidates
                if candidates.index(candidate) in labels
            }
            if len(labeled_ids) != 1:
                continue
            high = [
                candidates.index(candidate)
                for candidate in component_candidates
                if candidate.confidence >= order_confidence_threshold
            ]
            support = 0
            for transmitter_id in range(transmitter_count):
                count = sum(
                    1 for index in high
                    if candidates[index].transmitter_id == transmitter_id
                )
                support += int(count >= 2)
            supports.append(support)
    if not supports:
        return None
    values = np.asarray(supports, dtype=int)
    for support in range(1, transmitter_count + 1):
        if float(np.mean(values >= support)) <= false_alarm_probability:
            return support
    return None


def gospa_distance(
    truth_positions: np.ndarray,
    estimate_positions: np.ndarray,
    cutoff_m: float = 15.0,
) -> float:
    truth = np.asarray(truth_positions, dtype=float)
    estimates = np.asarray(estimate_positions, dtype=float)
    matched_cost = 0.0
    matched_count = 0
    if len(truth) and len(estimates):
        distances = np.linalg.norm(
            truth[:, None, :] - estimates[None, :, :], axis=2
        )
        rows, columns = linear_sum_assignment(
            np.minimum(distances, cutoff_m) ** 2
        )
        matched_cost = float(np.sum(
            np.minimum(distances[rows, columns], cutoff_m) ** 2
        ))
        matched_count = len(rows)
    cardinality_cost = (
        cutoff_m ** 2 / 2.0
        * (len(truth) + len(estimates) - 2 * matched_count)
    )
    return float(np.sqrt(matched_cost + cardinality_cost))


def evaluate_separated_scene(
    config: FrontEndConfig,
    patterns: list[np.ndarray],
    nodes: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    threshold: float,
    calibrator,
    templates,
    amplitude: float,
    target_count: int,
    trials: int,
    rng: np.random.Generator,
    view_false_target_probability: float | list[float] | None = None,
    view_false_extra_probability: float | list[float] | None = None,
    required_collision_support_override: int | None = None,
    angle_guard: int = 2,
    rayleigh_fading: bool = False,
) -> dict:
    rows = []
    for _ in range(trials):
        targets = draw_targets(rng, target_count)
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        received = simulate_received(
            config, patterns, paths,
            random_path_gains(
                paths, amplitude, rng, rayleigh_fading
            ), rng,
        )
        start = perf_counter()
        peaks = extract_integrated_peaks(
            config, received, patterns, threshold, templates=templates,
            angle_guard=angle_guard,
        )
        cubes = integrated_detection_cubes(
            config, received, patterns, templates=templates
        )
        candidates = peaks_to_candidates(config, peaks, cubes, calibrator)
        front_end_seconds = perf_counter() - start
        matched_true, false_candidates = candidate_truth(
            candidates, paths, config
        )
        for mode, use_covariance in (
            ("fixed", False),
            ("front_end_covariance", True),
        ):
            start = perf_counter()
            groups = bic_conflict_association(
                candidates, nodes, receiver, CARRIER_HZ,
                position_tolerance_m=12.0,
                position_sigma_m=6.0,
                doppler_sigma_hz=15.0,
                clutter_doppler_span_hz=160.0,
                view_false_target_probability=(
                    view_false_target_probability
                    if view_false_target_probability is not None else 0.1
                ),
                view_false_extra_probability=(
                    view_false_extra_probability
                    if view_false_extra_probability is not None else 0.1
                ),
                required_collision_support_override=(
                    required_collision_support_override
                ),
                order_confidence_threshold=0.5,
                maximum_local_targets=2,
                final_joint_refinement_iterations=0,
                covariance_weighted_final_state=use_covariance,
            )
            association_seconds = perf_counter() - start
            truth_positions = np.asarray(
                [target.position for target in targets]
            )
            estimate_positions = np.asarray(
                [group.position for group in groups]
            )
            matched = []
            if len(groups):
                errors = np.linalg.norm(
                    truth_positions[:, None, :]
                    - estimate_positions[None, :, :],
                    axis=2,
                )
                truth_indices, estimate_indices = linear_sum_assignment(errors)
                matched = [
                    (truth_index, estimate_index)
                    for truth_index, estimate_index
                    in zip(truth_indices, estimate_indices)
                    if errors[truth_index, estimate_index] <= 15.0
                ]
            position_errors = [
                np.linalg.norm(
                    groups[estimate].position - targets[truth].position
                )
                for truth, estimate in matched
            ]
            velocity_errors = [
                np.linalg.norm(
                    groups[estimate].velocity - targets[truth].velocity
                )
                for truth, estimate in matched
            ]
            rows.append({
                "mode": mode,
                "target_count": target_count,
                "path_recall": matched_true / max(len(paths), 1),
                "candidate_count": len(candidates),
                "false_candidates": false_candidates,
                "estimated_targets": len(groups),
                "target_count_correct": float(len(groups) == target_count),
                "scene_exact_recovery": float(
                    len(groups) == target_count
                    and len(matched) == target_count
                ),
                "state_exact_recovery": float(
                    len(groups) == target_count
                    and len(matched) == target_count
                    and all(error <= 1.0 for error in velocity_errors)
                ),
                "gospa_15m": gospa_distance(
                    truth_positions, estimate_positions
                ),
                "mean_position_error": float(np.mean(position_errors))
                if position_errors else None,
                "mean_velocity_error": float(np.mean(velocity_errors))
                if velocity_errors else None,
                "front_end_seconds": front_end_seconds,
                "association_seconds": association_seconds,
            })
    return rows


def evaluate_equal_total_energy(
    *,
    config: FrontEndConfig,
    total_energy: float,
    target_count: int,
    trials: int,
    calibration_frames: int,
    seed: int,
    frame_false_alarm_probability: float,
    angle_guard: int,
    integration_frames: int,
    pattern_kind: str,
    rayleigh_fading: bool,
) -> dict:
    """Compare M=2 and M=4 under one fixed total pilot energy budget."""
    rows = {}
    for transmitter_count in (4, 2):
        amplitude = float(np.sqrt(total_energy / transmitter_count))
        patterns = identity_patterns(
            config, transmitter_count, seed=seed, kind=pattern_kind
        )
        templates = precompute_templates(config, patterns)
        rng = np.random.default_rng(seed + 20)
        nodes = transmitter_geometry(transmitter_count)
        receiver = receiver_node()
        frames = scenario_frames(
            config, patterns, nodes, receiver, amplitude, rng,
            count=target_count, frames=max(1, calibration_frames // 2),
            integration_frames=integration_frames,
            rayleigh_fading=rayleigh_fading,
        )
        noise_threshold = calibrate_frame_threshold(
            config, patterns, trials=300,
            frame_false_alarm_probability=frame_false_alarm_probability,
            templates=templates, batch_size=100, seed=seed + 10,
            integration_frames=integration_frames,
        )
        sidelobe_threshold = calibrate_sidelobe_aware_threshold(
            config, patterns, frames,
            templates=templates,
            frame_false_alarm_probability=frame_false_alarm_probability,
            angle_guard=angle_guard,
        )
        threshold = (
            max(noise_threshold, sidelobe_threshold)
            if integration_frames > 1 else noise_threshold
        )
        calibrator = calibrate_confidence(
            config, patterns, frames,
            collect_threshold=threshold * 0.5,
            templates=templates, angle_guard=angle_guard,
        )
        p_target, p_extra = calibrate_view_probabilities(
            config, patterns, frames,
            calibrator=calibrator, templates=templates, threshold=threshold,
            order_confidence_threshold=0.5,
            angle_guard=angle_guard,
        )
        collision_support = calibrate_empirical_collision_support(
            config, patterns, frames,
            calibrator=calibrator, templates=templates, threshold=threshold,
            order_confidence_threshold=0.5,
            false_alarm_probability=0.05,
            angle_guard=angle_guard,
        )
        effective_collision_support = (
            max(2, collision_support)
            if collision_support is not None else 2
        )
        rows[str(transmitter_count)] = {
            "per_uav_amplitude": amplitude,
            "view_false_target_probability": p_target,
            "view_false_extra_probability": p_extra,
            "empirical_collision_support": collision_support,
            "effective_collision_support": effective_collision_support,
            "summary": summarize(evaluate_separated_scene(
                config, patterns, nodes, receiver, threshold, calibrator,
                templates, amplitude, target_count, trials, rng,
                view_false_target_probability=p_target,
                view_false_extra_probability=p_extra,
                required_collision_support_override=(
                    effective_collision_support
                ),
                angle_guard=angle_guard,
                rayleigh_fading=rayleigh_fading,
            )),
        }
    return {
        "total_pilot_energy": total_energy,
        "target_count": target_count,
        "transmitters": rows,
    }


def evaluate_noise_only(
    config: FrontEndConfig,
    patterns: list[np.ndarray],
    threshold: float,
    templates,
    trials: int,
    rng: np.random.Generator,
    integration_frames: int = 1,
) -> dict:
    frames_with_peaks = 0
    candidate_counts = []
    for _ in range(trials):
        received = [
            simulate_received(config, patterns, (), (), rng)
            for _ in range(integration_frames)
        ]
        received = received[0] if integration_frames == 1 else received
        peaks = extract_integrated_peaks(
            config, received, patterns, threshold, templates=templates
        )
        candidate_counts.append(len(peaks))
        frames_with_peaks += int(len(peaks) > 0)
    return {
        "trials": trials,
        "frame_false_alarm_rate": frames_with_peaks / trials,
        "mean_false_candidates_per_frame": float(np.mean(candidate_counts)),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["target_count"], {}).setdefault(
            row["mode"], []
        ).append(row)
    summary = {}
    for target_count, group in grouped.items():
        summary[str(target_count)] = {}
        for mode, mode_rows in group.items():
            position_errors = [
                row["mean_position_error"]
                for row in mode_rows
                if row["mean_position_error"] is not None
            ]
            velocity_errors = [
                row["mean_velocity_error"]
                for row in mode_rows
                if row["mean_velocity_error"] is not None
            ]
            summary[str(target_count)][mode] = {
                "trials": len(mode_rows),
                "path_recall": float(np.mean([
                    row["path_recall"] for row in mode_rows
                ])),
                "false_candidates_per_frame": float(np.mean([
                    row["false_candidates"] for row in mode_rows
                ])),
                "scene_exact_recovery": float(np.mean([
                    row["scene_exact_recovery"] for row in mode_rows
                ])),
                "state_exact_recovery": float(np.mean([
                    row["state_exact_recovery"] for row in mode_rows
                ])),
                "target_count_accuracy": float(np.mean([
                    row["target_count_correct"] for row in mode_rows
                ])),
                "mean_gospa": float(np.mean([
                    row["gospa_15m"] for row in mode_rows
                ])),
                "mean_position_error_m": float(np.mean(position_errors))
                if position_errors else None,
                "mean_velocity_error_mps": float(np.mean(velocity_errors))
                if velocity_errors else None,
                "mean_front_end_seconds": float(np.mean([
                    row["front_end_seconds"] for row in mode_rows
                ])),
                "mean_association_seconds": float(np.mean([
                    row["association_seconds"] for row in mode_rows
                ])),
            }
    return summary


def run_gate(
    *,
    output: Path,
    trials: int,
    threshold_trials: int,
    calibration_frames: int,
    noise_trials: int,
    seed: int,
    equal_energy_trials: int,
    frame_false_alarm_probability: float,
    angle_guard: int,
    collision_support_override: int | None,
    integration_frames: int,
    pattern_kind: str,
    amplitude: float,
    rayleigh_fading: bool,
) -> None:
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 4, kind=pattern_kind)
    templates = precompute_templates(config, patterns)
    rng = np.random.default_rng(seed + 2)
    nodes = transmitter_geometry(4)
    receiver = receiver_node()
    frames = []
    for count in (1, 2):
        frames.extend(scenario_frames(
            config, patterns, nodes, receiver, amplitude, rng,
            count=count, frames=max(1, calibration_frames // 2),
            integration_frames=integration_frames,
            rayleigh_fading=rayleigh_fading,
        ))
    noise_threshold = calibrate_frame_threshold(
        config, patterns, trials=threshold_trials,
        frame_false_alarm_probability=frame_false_alarm_probability,
        templates=templates, batch_size=200, seed=seed + 1,
        integration_frames=integration_frames,
    )
    sidelobe_threshold = calibrate_sidelobe_aware_threshold(
        config, patterns, frames,
        templates=templates,
        frame_false_alarm_probability=frame_false_alarm_probability,
        angle_guard=angle_guard,
    )
    threshold = (
        max(noise_threshold, sidelobe_threshold)
        if integration_frames > 1 else noise_threshold
    )
    calibrator = calibrate_confidence(
        config, patterns, frames,
        collect_threshold=threshold * 0.5,
        templates=templates, angle_guard=angle_guard,
    )
    p_target, p_extra = calibrate_view_probabilities(
        config, patterns, frames,
        calibrator=calibrator, templates=templates, threshold=threshold,
        order_confidence_threshold=0.5,
        angle_guard=angle_guard,
    )
    empirical_collision_support = calibrate_empirical_collision_support(
        config, patterns, frames,
        calibrator=calibrator, templates=templates, threshold=threshold,
        order_confidence_threshold=0.5,
        false_alarm_probability=0.05,
        angle_guard=angle_guard,
    )
    if collision_support_override is not None:
        effective_collision_support = collision_support_override
    else:
        effective_collision_support = (
            max(2, empirical_collision_support)
            if empirical_collision_support is not None else 2
        )
    rows = []
    rows.extend(evaluate_separated_scene(
        config, patterns, nodes, receiver, threshold, calibrator,
        templates, amplitude, 1, trials, rng,
        view_false_target_probability=p_target,
        view_false_extra_probability=p_extra,
        required_collision_support_override=effective_collision_support,
        angle_guard=angle_guard,
        rayleigh_fading=rayleigh_fading,
    ))
    rows.extend(evaluate_separated_scene(
        config, patterns, nodes, receiver, threshold, calibrator,
        templates, amplitude, 2, trials, rng,
        view_false_target_probability=p_target,
        view_false_extra_probability=p_extra,
        required_collision_support_override=effective_collision_support,
        angle_guard=angle_guard,
        rayleigh_fading=rayleigh_fading,
    ))
    noise = evaluate_noise_only(
        config, patterns, threshold, templates, noise_trials, rng,
        integration_frames=integration_frames,
    )
    equal_total_energy = None
    if equal_energy_trials > 0:
        equal_total_energy = evaluate_equal_total_energy(
            config=config,
            total_energy=4.0 * amplitude ** 2,
            target_count=2,
            trials=equal_energy_trials,
            calibration_frames=calibration_frames,
            seed=seed + 100,
            frame_false_alarm_probability=frame_false_alarm_probability,
            angle_guard=angle_guard,
            integration_frames=integration_frames,
            pattern_kind=pattern_kind,
            rayleigh_fading=rayleigh_fading,
        )
    payload = {
        "model": (
            "8-element ULA; per-UAV unit-energy QPSK identity signatures; "
            "toy DD grid with declared delay/Doppler resolutions"
        ),
        "grid": {
            "doppler_bins": config.doppler_bins,
            "delay_bins": config.delay_bins,
            "delay_resolution_m": config.delay_resolution_m,
            "doppler_resolution_hz": config.doppler_resolution_hz,
        },
        "path_gain_amplitude": amplitude,
        "rayleigh_fading": rayleigh_fading,
        "total_pilot_energy": (
            integration_frames * len(patterns) * amplitude ** 2
        ),
        "noise_variance": config.noise_variance,
        "frame_false_alarm_probability": frame_false_alarm_probability,
        "integration_frames": integration_frames,
        "threshold": threshold,
        "calibration": {
            "confidence_calibrator_blocks": len(calibrator.thresholds),
            "confidence_calibration_frames": len(frames),
            "threshold_calibration_frames": threshold_trials,
        },
        "association_settings": {
            "position_tolerance_m": 12.0,
            "position_sigma_m": 6.0,
            "doppler_sigma_hz": 15.0,
            "order_confidence_threshold": 0.5,
            "collision_false_alarm_probability": 0.05,
        },
        "empirical_collision_support": empirical_collision_support,
        "effective_collision_support": effective_collision_support,
        "calibrated_view_probabilities": {
            "view_false_target_probability": p_target,
            "view_false_extra_probability": p_extra,
        },
        "noise_only": noise,
        "equal_total_energy": equal_total_energy,
        "summary": summarize(rows),
        "instances": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "threshold": threshold,
        "noise_only": noise,
        "summary": payload["summary"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/multistatic_front_end_gate.json",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--threshold-trials", type=int, default=1000)
    parser.add_argument("--calibration-frames", type=int, default=100)
    parser.add_argument("--noise-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--equal-energy-trials", type=int, default=30)
    parser.add_argument("--frame-pfa", type=float, default=0.002)
    parser.add_argument("--angle-guard", type=int, default=2)
    parser.add_argument("--collision-support", type=int, default=None)
    parser.add_argument("--integration-frames", type=int, default=1)
    parser.add_argument("--pattern-kind", choices=("qpsk", "cazac"),
                        default="qpsk")
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--rayleigh-fading", action="store_true")
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        trials=args.trials,
        threshold_trials=args.threshold_trials,
        calibration_frames=args.calibration_frames,
        noise_trials=args.noise_trials,
        seed=args.seed,
        equal_energy_trials=args.equal_energy_trials,
        frame_false_alarm_probability=args.frame_pfa,
        angle_guard=args.angle_guard,
        collision_support_override=args.collision_support,
        integration_frames=args.integration_frames,
        pattern_kind=args.pattern_kind,
        amplitude=args.amplitude,
        rayleigh_fading=args.rayleigh_fading,
    )


if __name__ == "__main__":
    main()
