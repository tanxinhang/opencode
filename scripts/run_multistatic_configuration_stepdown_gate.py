"""Configuration-calibrated ordered GLRT candidate for mixed collision scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_baseline_comparison import run_comparison
from scripts.run_multistatic_g0b import evaluate_trial
from scripts.run_multistatic_calibrated_gate import calibration_sample
from scripts.run_multistatic_g0b import (
    CARRIER_HZ, PROPAGATION_SPEED, draw_targets, imperfect_candidates,
    nested_transmitter_geometry,
)
from scripts.run_multistatic_physics_glrt_gate import (
    finite_sample_upper_threshold, fit_ordered_frame_thresholds,
)
from uav_otfs_isac.multistatic_association import PathCandidate
from uav_otfs_isac.multistatic_baselines import _dbscan_labels, _project
from uav_otfs_isac.multistatic_model_selection import (
    collision_support_threshold, physics_order_gain,
)
from uav_otfs_isac.multistatic_targets import KinematicNode, generate_bistatic_paths
from uav_otfs_isac.probability_calibration import fit_isotonic_probability


def labeled_physics_components(seed, scenes, calibrator, scenario,
                               transmitters=8, glrt_refinement_iterations=0):
    """Return GLRT components with truth labels for offline calibration only."""
    rng = np.random.default_rng(seed)
    nodes = nested_transmitter_geometry(transmitters)
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    minimum_views = collision_support_threshold((0.05,) * transmitters, 0.05)
    clutter_ratio = 2.0 * np.log(
        (np.pi * 14.0 ** 2 * 1800.0)
        / ((2.0 * np.pi) ** 1.5 * 3.0 ** 2 * 3.0)
    )
    frames = []
    for _ in range(scenes):
        targets = draw_targets(rng, 6, scenario)
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        candidates, truth, _ = imperfect_candidates(
            rng, paths, transmitters, miss_probability=0.08, false_mean=0.4,
            clutter_model="correlated_sidelobes", confidence_model="overlap",
        )
        calibrated, calibrated_truth = [], {}
        for candidate in candidates:
            item = PathCandidate(
                candidate.transmitter_id, candidate.delay_s,
                candidate.doppler_hz, candidate.receive_azimuth_rad,
                float(calibrator(candidate.confidence)),
            )
            calibrated.append(item)
            calibrated_truth[id(item)] = truth[id(candidate)]
        projected = _project(calibrated, nodes, receiver, PROPAGATION_SPEED)
        components = []
        if projected:
            positions = np.asarray([item[1] for item in projected])
            labels = _dbscan_labels(positions / 14.0, 1.0, 2)
            for label in range(int(labels.max()) + 1):
                members = [projected[index]
                           for index in np.flatnonzero(labels == label)]
                gain = physics_order_gain(
                    members, nodes, receiver, CARRIER_HZ, 3.0, 3.0,
                    clutter_ratio, minimum_views, 100.0, 20,
                    PROPAGATION_SPEED, glrt_refinement_iterations,
                )
                if not np.isfinite(gain):
                    continue
                target_ids = {
                    calibrated_truth[id(candidate)] for candidate, _ in members
                    if calibrated_truth[id(candidate)] is not None
                }
                components.append({
                    "gain": float(gain),
                    "is_collision": len(target_ids) >= 2,
                    "target_count": len(target_ids),
                })
        frames.append(components)
    return frames


def normal_component_maxima(frames):
    return np.asarray([
        max((item["gain"] for item in frame if not item["is_collision"]),
            default=-np.inf)
        for frame in frames
    ])


def fit_configuration_threshold(frames, alpha=0.01):
    maxima = normal_component_maxima(frames)
    finite = maxima[np.isfinite(maxima)]
    return finite_sample_upper_threshold(finite, alpha), int(len(finite))


def proposed_summary(scenario, trials, seed, calibrator, thresholds,
                     transmitters=8, activation_count=1,
                     glrt_refinement_iterations=0):
    """Evaluate only the threshold-dependent proposed method on paired scenes."""
    rng = np.random.default_rng(seed)
    rows = [evaluate_trial(
        rng, transmitters, 6, 0.08, 0.4, True, "bic_conflict", scenario,
        "correlated_sidelobes", 0.05, "nested_12", "overlap",
        "physics_stepdown", calibrator, None, None, None, None,
        tuple(thresholds), True, activation_count,
        glrt_refinement_iterations,
    ) for _ in range(trials)]
    matched = sum(row["matched_targets"] for row in rows)
    return {
        "trials": trials,
        "position_set_exact_15m": float(np.mean([
            row["position_set_exact_15m"] for row in rows])),
        "position_velocity_state_exact": float(np.mean([
            row["position_velocity_state_exact"] for row in rows])),
        "over_count_rate": float(np.mean([
            row["estimated_targets"] > 6 for row in rows])),
        "under_count_rate": float(np.mean([
            row["estimated_targets"] < 6 for row in rows])),
        "mean_gospa_15m_p2": float(np.mean([
            row["gospa_15m_p2"] for row in rows])),
        "mean_path_f1": float(np.mean([
            row["path_association_f1"] for row in rows])),
        "mean_velocity_error_mps": float(
            sum(row["velocity_error_sum"] for row in rows) / max(matched, 1)),
        "mean_time_ms": float(1000.0 * np.mean([
            row["association_time_s"] for row in rows])),
    }


def unlabeled_frames(frames):
    """Project labeled offline records to the online component tuple format."""
    return [[(item["gain"], 0, 0) for item in frame] for frame in frames]


def run_configuration_gate(
    probability_scenes=500, configuration_frames=1000,
    evaluation_trials=100, transmitters=8,
    probability_seed=20261021, configuration_seed=20261022,
    evaluation_seed=20261023,
):
    scores, labels = calibration_sample(
        probability_seed, probability_scenes, transmitters
    )
    calibrator = fit_isotonic_probability(scores, labels)
    scenarios = ("separated", "single_pair_collision", "two_pair_collision")
    thresholds, counts, configuration_data = [], {}, {}
    for offset, scenario in enumerate(scenarios):
        frames = labeled_physics_components(
            configuration_seed + offset, configuration_frames, calibrator,
            scenario, transmitters,
        )
        threshold, count = fit_configuration_threshold(frames, 0.01)
        thresholds.append(threshold)
        counts[scenario] = count
        configuration_data[scenario] = frames
    # At N=6 there are at most three pair-collision components.  A fourth
    # rejection has no modeled physical use and is deliberately disabled.
    thresholds.append(float("inf"))
    weak_thresholds = fit_ordered_frame_thresholds(
        unlabeled_frames(configuration_data["separated"]), 4, 0.01
    )
    # All methods share the same frame-maximum threshold.  This removes
    # probability-calibration and finite-sample variation from the ablation.
    weak_thresholds = (thresholds[0],) + tuple(weak_thresholds[1:])
    single_thresholds = (thresholds[0],) * 4
    def comparisons(scenario):
        return {
            "single_threshold": proposed_summary(
                scenario, evaluation_trials, evaluation_seed, calibrator,
                single_thresholds, transmitters),
            "global_null_stepdown": proposed_summary(
                scenario, evaluation_trials, evaluation_seed, calibrator,
                weak_thresholds, transmitters),
            "configuration_stepdown": proposed_summary(
                scenario, evaluation_trials, evaluation_seed, calibrator,
                thresholds, transmitters),
            "density_triggered_stepdown": proposed_summary(
                scenario, evaluation_trials, evaluation_seed, calibrator,
                weak_thresholds, transmitters, activation_count=2),
        }
    return {
        "scope": (
            "configuration-calibrated ordered GLRT candidate; truth labels are "
            "used offline only to form normal-component null maxima"
        ),
        "seeds": {"probability": probability_seed,
                  "configuration_base": configuration_seed,
                  "evaluation": evaluation_seed},
        "sample_sizes": {"probability_scenes": probability_scenes,
                         "frames_per_configuration": configuration_frames,
                         "evaluation_trials_per_scenario": evaluation_trials},
        "gate": {"alpha_per_configuration": 0.01,
                 "ordered_thresholds": thresholds,
                 "shared_first_threshold": thresholds[0],
                 "single_thresholds": list(single_thresholds),
                 "global_null_stepdown_thresholds": list(weak_thresholds),
                 "finite_normal_maxima": counts,
                 "strong_fwer_claimed": False,
                 "calibrated_configurations": list(scenarios)},
        "probability_calibrator": calibrator.to_dict(),
        "separated": comparisons("separated"),
        "single_pair_collision": comparisons("single_pair_collision"),
        "two_pair_collision": comparisons("two_pair_collision"),
        "three_pair_collision": comparisons("paired_collision"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-scenes", type=int, default=500)
    parser.add_argument("--configuration-frames", type=int, default=1000)
    parser.add_argument("--evaluation-trials", type=int, default=100)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_configuration_stepdown_m8_n6.json"))
    args = parser.parse_args()
    payload = run_configuration_gate(
        args.probability_scenes, args.configuration_frames,
        args.evaluation_trials, args.transmitters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
