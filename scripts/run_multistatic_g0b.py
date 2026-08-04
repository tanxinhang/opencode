"""Gate G0-B: geometry association from imperfect path candidates.

This gate isolates the target-level back end.  It perturbs geometry-generated
path truth, independently drops paths, and adds unlabelled false candidates;
it does not simulate an OTFS matched-filter/CFAR front end.  Consequently its
metrics measure association robustness conditional on the stated candidate
model, not end-to-end detection performance.
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

from uav_otfs_isac.multistatic_association import (
    PathCandidate,
    associate_path_candidates,
)
from uav_otfs_isac.multistatic_baselines import (
    conflict_aware_dbscan_association,
    dbscan_path_association,
)
from uav_otfs_isac.multistatic_model_selection import (
    bic_conflict_association, collision_support_threshold,
)
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
        (350.0 * np.cos(angle), 350.0 * np.sin(angle)), (0.0, 0.0)
    ) for angle in angles)


def nested_transmitter_geometry(
    count: int, maximum_count: int = 12
) -> tuple[KinematicNode, ...]:
    """Select a nested subset from one fixed circular mother deployment.

    The order keeps the first four views uniformly spaced and then adds
    approximately antipodal pairs. Thus changing ``count`` adds measurements
    without moving UAVs already present in the smaller system.
    """
    if maximum_count != 12:
        raise ValueError("the audited nested deployment currently uses 12 UAVs")
    if count < 1 or count > maximum_count:
        raise ValueError("count must lie between one and maximum_count")
    order = np.asarray((0, 3, 6, 9, 1, 7, 4, 10, 2, 8, 5, 11))
    angles = 2.0 * np.pi * order[:count] / maximum_count
    return tuple(KinematicNode(
        (350.0 * np.cos(angle), 350.0 * np.sin(angle)), (0.0, 0.0)
    ) for angle in angles)


def gospa_distance(
    truth_positions: np.ndarray,
    estimate_positions: np.ndarray,
    cutoff_m: float = 15.0,
    order: int = 2,
    alpha: float = 2.0,
) -> float:
    """Generalized OSPA distance with explicit miss/false-target penalties."""
    truth = np.asarray(truth_positions, dtype=float)
    estimates = np.asarray(estimate_positions, dtype=float)
    if cutoff_m <= 0 or order < 1 or alpha <= 0 or alpha > 2:
        raise ValueError("invalid GOSPA parameters")
    matched_cost = 0.0
    matched_count = 0
    if len(truth) and len(estimates):
        distances = np.linalg.norm(
            truth[:, None, :] - estimates[None, :, :], axis=2
        )
        rows, columns = linear_sum_assignment(
            np.minimum(distances, cutoff_m) ** order
        )
        matched_cost = float(np.sum(
            np.minimum(distances[rows, columns], cutoff_m) ** order
        ))
        matched_count = len(rows)
    cardinality_cost = (
        cutoff_m ** order / alpha
        * (len(truth) + len(estimates) - 2 * matched_count)
    )
    return float((matched_cost + cardinality_cost) ** (1.0 / order))


def draw_targets(
    rng: np.random.Generator, count: int, scenario: str = "separated"
) -> tuple[PhysicalTarget, ...]:
    # Separation is deliberately moderate: nearby angles remain possible, but
    # the Gate does not include the fundamentally identical angle/DD boundary.
    if scenario == "separated":
        if count == 1:
            angles = rng.uniform(-1.15, 1.15, 1)
        else:
            minimum_gap = np.deg2rad(6.0)
            available_slack = 2.3 - (count - 1) * minimum_gap
            if available_slack < 0.0:
                raise ValueError(
                    "target count cannot satisfy the separated-angle constraint"
                )
            # Randomly partition the feasible angular slack. Direct sampling
            # avoids the rejection probability collapsing as N grows.
            slack = rng.dirichlet(np.ones(count + 1)) * available_slack
            gaps = minimum_gap + slack[1:count]
            angles = -1.15 + slack[0] + np.concatenate((
                np.zeros(1), np.cumsum(gaps),
            ))
        ranges = rng.uniform(100.0, 260.0, count)
    elif scenario == "paired_collision":
        pair_count = (count + 1) // 2
        bases = np.linspace(-0.9, 0.9, pair_count)
        angles = np.asarray([
            bases[index // 2] + (-1.0 if index % 2 == 0 else 1.0)
            * np.deg2rad(1.0)
            for index in range(count)
        ])
        ranges = 180.0 + rng.normal(0.0, 1.0, count)
    elif scenario == "single_pair_collision":
        if count < 2:
            raise ValueError("single-pair collision requires at least two targets")
        pair_angles = np.asarray((-0.82, -0.82)) + np.deg2rad((-1.0, 1.0))
        separated_count = count - 2
        angles = np.concatenate((
            pair_angles, np.linspace(-0.25, 1.0, separated_count)
        ))
        pair_ranges = 180.0 + rng.normal(0.0, 1.0, 2)
        separated_ranges = np.linspace(115.0, 255.0, separated_count)
        separated_ranges += rng.normal(0.0, 2.0, separated_count)
        ranges = np.concatenate((pair_ranges, separated_ranges))
    elif scenario == "two_pair_collision":
        if count != 6:
            raise ValueError("audited two-pair collision scenario requires N=6")
        pair_bases = (-0.82, -0.15)
        angles = np.asarray([
            base + offset
            for base in pair_bases
            for offset in np.deg2rad((-1.0, 1.0))
        ] + [0.55, 1.0])
        ranges = np.asarray([
            180.0, 180.0, 215.0, 215.0, 130.0, 255.0
        ]) + rng.normal(0.0, 1.0, count)
    else:
        raise ValueError("unsupported target scenario")
    velocities = rng.uniform(-8.0, 8.0, (count, 2))
    return tuple(PhysicalTarget(
        target_id=index,
        position=ranges[index] * np.array([
            np.cos(angles[index]), np.sin(angles[index])
        ]),
        velocity=velocities[index],
    ) for index in range(count))


def imperfect_candidates(
    rng: np.random.Generator,
    paths,
    transmitter_count: int,
    *,
    miss_probability: float,
    false_mean: float,
    clutter_model: str = "diffuse",
    confidence_model: str = "separated",
) -> tuple[list[PathCandidate], dict[int, int | None], int]:
    candidates: list[PathCandidate] = []
    truth: dict[int, int | None] = {}
    retained = 0
    for path in paths:
        if rng.random() < miss_probability:
            continue
        retained += 1
        range_error = rng.normal(0.0, 1.5)
        if confidence_model == "separated":
            true_confidence = rng.uniform(0.7, 1.0)
        elif confidence_model == "overlap":
            true_confidence = rng.uniform(0.45, 0.9)
        else:
            raise ValueError("unsupported confidence model")
        candidate = PathCandidate(
            transmitter_id=path.transmitter_id,
            delay_s=max(
                path.delay_s + range_error / PROPAGATION_SPEED, 1e-12
            ),
            doppler_hz=path.doppler_hz + rng.normal(0.0, 1.5),
            receive_azimuth_rad=(
                path.receive_azimuth_rad + rng.normal(0.0, np.deg2rad(0.4))
            ),
            confidence=float(true_confidence),
        )
        candidates.append(candidate)
        truth[id(candidate)] = path.target_id

        if clutter_model == "correlated_sidelobes":
            if rng.random() < 0.35:
                sidelobe = PathCandidate(
                    transmitter_id=path.transmitter_id,
                    delay_s=max(
                        path.delay_s + rng.choice((-1.0, 1.0))
                        * rng.uniform(3.0, 8.0) / PROPAGATION_SPEED,
                        1e-12,
                    ),
                    doppler_hz=path.doppler_hz + rng.choice((-1.0, 1.0))
                    * rng.uniform(8.0, 20.0),
                    receive_azimuth_rad=path.receive_azimuth_rad
                    + rng.normal(0.0, np.deg2rad(0.5)),
                    confidence=float(
                        rng.uniform(0.15, 0.45)
                        if confidence_model == "separated"
                        else rng.uniform(0.2, 0.8)
                    ),
                )
                candidates.append(sidelobe)
                truth[id(sidelobe)] = None
        elif clutter_model != "diffuse":
            raise ValueError("unsupported clutter model")

    false_count = int(rng.poisson(false_mean))
    for _ in range(false_count):
        candidate = PathCandidate(
            transmitter_id=int(rng.integers(transmitter_count)),
            delay_s=float(rng.uniform(0.7e-6, 4.0e-6)),
            doppler_hz=float(rng.uniform(-900.0, 900.0)),
            receive_azimuth_rad=float(rng.uniform(-np.pi, np.pi)),
            confidence=float(
                rng.uniform(0.1, 0.6)
                if confidence_model == "separated"
                else rng.uniform(0.2, 0.8)
            ),
        )
        candidates.append(candidate)
        truth[id(candidate)] = None
    return candidates, truth, retained


def evaluate_trial(
    rng: np.random.Generator,
    transmitter_count: int,
    target_count: int,
    miss_probability: float,
    false_mean: float,
    use_spatial_index: bool = True,
    method: str = "geometry_doppler",
    scenario: str = "separated",
    clutter_model: str = "diffuse",
    view_false_target_probability: float = 0.1,
    geometry_mode: str = "independent_uniform",
    confidence_model: str = "separated",
    collision_gate_mode: str = "hard_null",
    score_calibrator=None,
    collision_support_calibrators=None,
    collision_statistic_thresholds=None,
    physics_collision_threshold=None,
    physics_frame_thresholds=None,
    physics_stepdown_thresholds=None,
    robust_final_velocity=False,
    physics_stepdown_activation_count=1,
    physics_glrt_refinement_iterations=0,
    physics_cascade_thresholds=None,
    final_joint_refinement_iterations=3,
    covariance_weighted_final_state=False,
    physics_conformal_null=None,
    physics_conformal_p_threshold=None,
) -> dict[str, float]:
    if geometry_mode == "independent_uniform":
        transmitters = transmitter_geometry(transmitter_count)
    elif geometry_mode == "nested_12":
        transmitters = nested_transmitter_geometry(transmitter_count)
    else:
        raise ValueError("unsupported transmitter geometry mode")
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    targets = draw_targets(rng, target_count, scenario)
    paths = generate_bistatic_paths(
        transmitters, targets, receiver, CARRIER_HZ
    )
    candidates, candidate_truth, retained = imperfect_candidates(
        rng, paths, transmitter_count,
        miss_probability=miss_probability, false_mean=false_mean,
        clutter_model=clutter_model, confidence_model=confidence_model,
    )
    if score_calibrator is not None:
        calibrated_candidates = []
        calibrated_truth = {}
        for candidate in candidates:
            calibrated = PathCandidate(
                candidate.transmitter_id, candidate.delay_s,
                candidate.doppler_hz, candidate.receive_azimuth_rad,
                float(score_calibrator(candidate.confidence)),
            )
            calibrated_candidates.append(calibrated)
            calibrated_truth[id(calibrated)] = candidate_truth[id(candidate)]
        candidates, candidate_truth = calibrated_candidates, calibrated_truth
    association_start = perf_counter()
    if method == "geometry_doppler":
        groups = associate_path_candidates(
            candidates, transmitters, receiver, CARRIER_HZ,
            angle_tolerance_rad=np.deg2rad(2.5),
            position_tolerance_m=14.0,
            doppler_tolerance_hz=7.0,
            min_transmitters=2,
            use_spatial_index=use_spatial_index,
        )
    elif method == "conflict_aware_dbscan":
        groups = conflict_aware_dbscan_association(
            candidates, transmitters, receiver, CARRIER_HZ,
            position_tolerance_m=14.0,
        )
    elif method == "bic_conflict":
        groups = bic_conflict_association(
            candidates, transmitters, receiver, CARRIER_HZ,
            position_tolerance_m=14.0,
            view_false_target_probability=view_false_target_probability,
            collision_gate_mode=collision_gate_mode,
            collision_support_calibrators=collision_support_calibrators,
            collision_statistic_thresholds=collision_statistic_thresholds,
            physics_collision_threshold=physics_collision_threshold,
            physics_frame_thresholds=physics_frame_thresholds,
            physics_stepdown_thresholds=physics_stepdown_thresholds,
            robust_final_velocity=robust_final_velocity,
            physics_stepdown_activation_count=physics_stepdown_activation_count,
            physics_glrt_refinement_iterations=physics_glrt_refinement_iterations,
            physics_cascade_thresholds=physics_cascade_thresholds,
            final_joint_refinement_iterations=final_joint_refinement_iterations,
            covariance_weighted_final_state=covariance_weighted_final_state,
            physics_conformal_null=physics_conformal_null,
            physics_conformal_p_threshold=physics_conformal_p_threshold,
        )
    elif method in {
        "position_dbscan", "angle_position_dbscan", "identity_dbscan",
        "gated_identity_dbscan",
    }:
        groups = dbscan_path_association(
            candidates, transmitters, receiver, CARRIER_HZ,
            position_tolerance_m=14.0,
            angle_tolerance_rad=(
                None if method == "position_dbscan" else np.deg2rad(2.5)
            ),
            enforce_unique_transmitter=method in {
                "identity_dbscan", "gated_identity_dbscan"
            },
        )
        if method == "gated_identity_dbscan":
            support = collision_support_threshold(
                (view_false_target_probability,) * transmitter_count, 0.05
            )
            groups = tuple(group for group in groups if support is not None and
                           len({path.transmitter_id for path in group.paths}) >= support)
    else:
        raise ValueError("unsupported association method")
    association_time_s = perf_counter() - association_start

    truth_positions = np.asarray([target.position for target in targets])
    estimate_positions = np.asarray([group.position for group in groups])
    matched: list[tuple[int, int]] = []
    if len(groups):
        errors = np.linalg.norm(
            truth_positions[:, None, :] - estimate_positions[None, :, :], axis=2
        )
        truth_indices, estimate_indices = linear_sum_assignment(errors)
        matched = [
            (int(truth_index), int(estimate_index))
            for truth_index, estimate_index in zip(truth_indices, estimate_indices)
            if errors[truth_index, estimate_index] <= 15.0
        ]

    correct_associations = 0
    grouped_true = 0
    grouped_false = 0
    for truth_index, estimate_index in matched:
        for candidate in groups[estimate_index].paths:
            label = candidate_truth[id(candidate)]
            if label is None:
                grouped_false += 1
            else:
                grouped_true += 1
                correct_associations += int(label == truth_index)
    matched_estimates = {estimate for _, estimate in matched}
    for estimate_index, group in enumerate(groups):
        if estimate_index in matched_estimates:
            continue
        for candidate in group.paths:
            grouped_false += int(candidate_truth[id(candidate)] is None)
            grouped_true += int(candidate_truth[id(candidate)] is not None)

    position_errors = [
        np.linalg.norm(groups[estimate].position - targets[truth].position)
        for truth, estimate in matched
    ]
    velocity_errors = [
        np.linalg.norm(groups[estimate].velocity - targets[truth].velocity)
        for truth, estimate in matched
    ]
    assigned_candidates = sum(len(group.paths) for group in groups)
    association_precision = correct_associations / max(assigned_candidates, 1)
    association_recall = correct_associations / max(retained, 1)
    association_f1 = (
        2.0 * association_precision * association_recall
        / max(association_precision + association_recall, 1e-15)
    )
    state_exact = bool(
        len(groups) == target_count and len(matched) == target_count
        and all(error <= 1.0 for error in velocity_errors)
    )
    return {
        "path_candidate_recall": retained / len(paths),
        "path_candidates": float(len(candidates)),
        "association_time_s": association_time_s,
        "estimated_targets": float(len(groups)),
        "target_count_correct": float(len(groups) == target_count),
        "target_recall": len(matched) / target_count,
        "scene_exact_recovery": float(
            len(matched) == target_count and len(groups) == target_count
        ),
        "position_set_exact_15m": float(
            len(matched) == target_count and len(groups) == target_count
        ),
        "position_velocity_state_exact": float(state_exact),
        "gospa_15m_p2": gospa_distance(truth_positions, estimate_positions),
        "path_association_precision": float(association_precision),
        "path_association_recall": float(association_recall),
        "path_association_f1": float(association_f1),
        "correct_identity_associations": float(correct_associations),
        "grouped_true_candidates": float(grouped_true),
        "grouped_false_candidates": float(grouped_false),
        "position_squared_error_sum": float(np.sum(
            np.asarray(position_errors) ** 2
        )),
        "velocity_error_sum": float(np.sum(velocity_errors)),
        "matched_targets": float(len(matched)),
    }


def run_study(
    trials: int = 100,
    seed: int = 20260803,
    transmitter_counts: tuple[int, ...] = (2, 4),
    target_counts: tuple[int, ...] = (1, 2, 3),
    miss_probability: float = 0.08,
    false_mean: float = 0.4,
    use_spatial_index: bool = True,
    method: str = "geometry_doppler",
    scenario: str = "separated",
    clutter_model: str = "diffuse",
    geometry_mode: str = "independent_uniform",
    confidence_model: str = "separated",
    collision_gate_mode: str = "hard_null",
) -> dict:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for transmitter_count in transmitter_counts:
        for target_count in target_counts:
            outcomes = [evaluate_trial(
                rng, transmitter_count, target_count,
                miss_probability, false_mean, use_spatial_index, method, scenario,
                clutter_model, 0.1, geometry_mode, confidence_model,
                collision_gate_mode,
            ) for _ in range(trials)]
            matched_total = sum(row["matched_targets"] for row in outcomes)
            rows.append({
                "transmitters_M": transmitter_count,
                "targets_N": target_count,
                "trials": trials,
                "path_candidate_recall": float(np.mean([
                    row["path_candidate_recall"] for row in outcomes
                ])),
                "mean_path_candidates": float(np.mean([
                    row["path_candidates"] for row in outcomes
                ])),
                "mean_association_time_ms": 1000.0 * float(np.mean([
                    row["association_time_s"] for row in outcomes
                ])),
                "p95_association_time_ms": 1000.0 * float(np.percentile([
                    row["association_time_s"] for row in outcomes
                ], 95)),
                "target_count_accuracy": float(np.mean([
                    row["target_count_correct"] for row in outcomes
                ])),
                "target_recall": float(np.mean([
                    row["target_recall"] for row in outcomes
                ])),
                "scene_exact_recovery": float(np.mean([
                    row["scene_exact_recovery"] for row in outcomes
                ])),
                "position_set_exact_15m": float(np.mean([
                    row["position_set_exact_15m"] for row in outcomes
                ])),
                "position_velocity_state_exact": float(np.mean([
                    row["position_velocity_state_exact"] for row in outcomes
                ])),
                "mean_gospa_15m_p2": float(np.mean([
                    row["gospa_15m_p2"] for row in outcomes
                ])),
                "path_association_precision": float(np.mean([
                    row["path_association_precision"] for row in outcomes
                ])),
                "path_association_recall": float(np.mean([
                    row["path_association_recall"] for row in outcomes
                ])),
                "path_association_f1": float(np.mean([
                    row["path_association_f1"] for row in outcomes
                ])),
                "identity_association_accuracy": float(
                    sum(row["correct_identity_associations"] for row in outcomes)
                    / max(sum(row["grouped_true_candidates"] for row in outcomes), 1)
                ),
                "mean_grouped_false_candidates": float(np.mean([
                    row["grouped_false_candidates"] for row in outcomes
                ])),
                "position_rmse_m": float(np.sqrt(
                    sum(row["position_squared_error_sum"] for row in outcomes)
                    / max(matched_total, 1)
                )),
                "mean_velocity_error_mps": float(
                    sum(row["velocity_error_sum"] for row in outcomes)
                    / max(matched_total, 1)
                ),
            })
    return {
        "scope": (
            "Gate G0-B target-level geometry association with unknown N, "
            "conditioned on a synthetic imperfect path-candidate front end; "
            "not end-to-end OTFS detection performance"
        ),
        "candidate_model": {
            "path_miss_probability": miss_probability,
            "poisson_false_candidates_mean": false_mean,
            "range_error_std_m": 1.5,
            "angle_error_std_deg": 0.4,
            "doppler_error_std_hz": 1.5,
        },
        "receiver": {
            "known_target_count": False,
            "minimum_distinct_transmitters_per_group": 2,
            "angle_tolerance_deg": 2.5,
            "position_tolerance_m": 14.0,
            "doppler_tolerance_hz": 7.0,
            "spatial_index": use_spatial_index,
            "method": method,
            "scenario": scenario,
            "clutter_model": clutter_model,
            "geometry_mode": geometry_mode,
            "confidence_model": confidence_model,
            "collision_gate_mode": collision_gate_mode,
        },
        "rows": rows,
        "warning": (
            "Path candidates already contain transmitter identity, angle, "
            "delay, and Doppler. Gate G0-B validates only path-to-target "
            "association and cannot establish the value of an OTFS front end "
            "or of the fixed identity codebook."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--transmitters", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--targets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--no-spatial-index", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/multistatic_g0b.json")
    )
    args = parser.parse_args()
    payload = run_study(
        trials=args.trials,
        seed=args.seed,
        transmitter_counts=tuple(args.transmitters),
        target_counts=tuple(args.targets),
        use_spatial_index=not args.no_spatial_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
