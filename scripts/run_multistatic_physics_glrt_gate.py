"""Frame-level empirical-null calibration of the physical order GLRT."""

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
from scripts.run_multistatic_calibrated_gate import calibration_sample
from scripts.run_multistatic_g0b import (
    CARRIER_HZ, PROPAGATION_SPEED, draw_targets, imperfect_candidates,
    nested_transmitter_geometry,
)
from uav_otfs_isac.multistatic_association import PathCandidate
from uav_otfs_isac.multistatic_baselines import _dbscan_labels, _project
from uav_otfs_isac.multistatic_model_selection import (
    collision_support_threshold, physics_order_gain,
)
from uav_otfs_isac.multistatic_targets import KinematicNode, generate_bistatic_paths
from uav_otfs_isac.probability_calibration import fit_isotonic_probability
from uav_otfs_isac.probability_calibration import ExcessPeakConformalNull


def frame_physics_components(
    seed, scenes, calibrator, transmitters=8, scenario="separated",
    glrt_refinement_iterations=0,
):
    rng = np.random.default_rng(seed)
    nodes = nested_transmitter_geometry(transmitters)
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    minimum_views = collision_support_threshold((0.05,) * transmitters, 0.05)
    clutter_volume = np.pi * 14.0 ** 2 * 1800.0
    target_volume = (2.0 * np.pi) ** 1.5 * 3.0 ** 2 * 3.0
    clutter_ratio = 2.0 * np.log(clutter_volume / target_volume)
    frames = []
    for _ in range(scenes):
        targets = draw_targets(rng, 6, scenario)
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        candidates, _, _ = imperfect_candidates(
            rng, paths, transmitters, miss_probability=0.08, false_mean=0.4,
            clutter_model="correlated_sidelobes", confidence_model="overlap",
        )
        calibrated = [PathCandidate(
            candidate.transmitter_id, candidate.delay_s,
            candidate.doppler_hz, candidate.receive_azimuth_rad,
            float(calibrator(candidate.confidence)),
        ) for candidate in candidates]
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
                if np.isfinite(gain):
                    components.append((
                        gain, len(members), len({
                            candidate.transmitter_id for candidate, _ in members
                        }),
                    ))
        frames.append(components)
    return frames


def frame_physics_statistics(
    seed, scenes, calibrator, transmitters=8, scenario="separated",
    glrt_refinement_iterations=0,
):
    frames = frame_physics_components(
        seed, scenes, calibrator, transmitters, scenario,
        glrt_refinement_iterations,
    )
    return np.asarray([
        max((component[0] for component in frame), default=-np.inf)
        for frame in frames
    ])


def fit_excess_peak_null(frames) -> ExcessPeakConformalNull:
    strata = {0: [], 1: [], 2: []}
    for frame in frames:
        for statistic, candidates, views in frame:
            key = ExcessPeakConformalNull.stratum(candidates, views)
            strata[key].append(statistic)
    return ExcessPeakConformalNull({
        key: np.asarray(values) for key, values in strata.items()
    })


def frame_minimum_pvalues(frames, null_model) -> np.ndarray:
    return np.asarray([
        min((null_model.p_value(*component) for component in frame), default=1.0)
        for frame in frames
    ])


def frame_excess_stratum(frame) -> int:
    """Pre-decision nuisance stratum for a complete candidate frame."""
    return int(any(candidates - views >= 2
                   for _, candidates, views in frame))


def finite_sample_upper_threshold(values, false_alarm_probability=0.01):
    """Split-conformal upper threshold with finite-sample false-alarm control.

    With ``n`` exchangeable calibration scores, the order statistic at
    ``ceil((n+1)(1-alpha))`` controls ``P(S_new > threshold) <= alpha``.
    If that order statistic does not exist, the requested resolution is not
    supported by the sample and the only valid non-randomized threshold is
    infinity.
    """
    values = np.sort(np.asarray(values, dtype=float))
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("calibration values must be a finite vector")
    if not 0 < false_alarm_probability < 1:
        raise ValueError("false-alarm probability must lie in (0, 1)")
    rank = int(np.ceil((len(values) + 1) * (1.0 - false_alarm_probability)))
    return float(values[rank - 1]) if rank <= len(values) else float("inf")


def fit_frame_stratified_thresholds(frames, false_alarm_probability=0.01):
    """Calibrate frame-maximum GLRT thresholds within fixed excess strata."""
    values = {0: [], 1: []}
    for frame in frames:
        maximum = max((component[0] for component in frame), default=-np.inf)
        # Empty frames cannot trigger and need not consume tail resolution.
        if np.isfinite(maximum):
            values[frame_excess_stratum(frame)].append(maximum)
    return ({key: finite_sample_upper_threshold(samples,
                                                false_alarm_probability)
             for key, samples in values.items()},
            {key: len(samples) for key, samples in values.items()})


def fit_ordered_frame_thresholds(
    frames, maximum_rank=4, false_alarm_probability=0.01
):
    """Finite-sample upper thresholds for successive frame order statistics.

    Missing components are represented by ``-inf``.  Sequential rejection
    stops at the first failure, so global-null FWER is exactly governed by the
    first (frame-maximum) threshold; later thresholds increase power without
    changing the event of at least one rejection under the global null.
    """
    if maximum_rank <= 0:
        raise ValueError("maximum rank must be positive")
    ordered = np.full((len(frames), maximum_rank), -np.inf)
    for row, frame in enumerate(frames):
        statistics = sorted((item[0] for item in frame if np.isfinite(item[0])),
                            reverse=True)[:maximum_rank]
        ordered[row, :len(statistics)] = statistics
    thresholds = []
    for rank in range(maximum_rank):
        values = ordered[:, rank]
        finite = values[np.isfinite(values)]
        thresholds.append(
            finite_sample_upper_threshold(finite, false_alarm_probability)
            if len(finite) else float("inf")
        )
    return tuple(thresholds)


def run_physics_gate(
    calibration_scenes=500, null_scenes=1000, validation_scenes=300,
    evaluation_trials=100, transmitters=8,
    calibration_seed=20260911, null_seed=20260912,
    validation_seed=20260913, evaluation_seed=20260914,
):
    if len({calibration_seed, null_seed, validation_seed, evaluation_seed}) != 4:
        raise ValueError("all data partitions require distinct seeds")
    scores, labels = calibration_sample(
        calibration_seed, calibration_scenes, transmitters
    )
    calibrator = fit_isotonic_probability(scores, labels)
    null = frame_physics_statistics(
        null_seed, null_scenes, calibrator, transmitters
    )
    threshold = float(np.quantile(null, 0.99, method="higher"))
    validation = frame_physics_statistics(
        validation_seed, validation_scenes, calibrator, transmitters
    )
    common = dict(
        trials=evaluation_trials, seed=evaluation_seed,
        transmitters=transmitters, targets=6,
        clutter_model="correlated_sidelobes",
        view_false_target_probability=0.05, geometry_mode="nested_12",
        confidence_model="overlap", collision_gate_mode="physics_glrt",
        score_calibrator=calibrator, physics_collision_threshold=threshold,
    )
    return {
        "scope": (
            "frame-maximum physical order-GLRT calibrated on independent "
            "synthetic separated N=6 frames; not end-to-end OTFS"
        ),
        "seeds": {"probability_calibration": calibration_seed,
                  "null_calibration": null_seed,
                  "null_validation": validation_seed,
                  "evaluation": evaluation_seed},
        "sample_sizes": {"probability_calibration_scenes": calibration_scenes,
                         "null_calibration_frames": null_scenes,
                         "null_validation_frames": validation_scenes,
                         "evaluation_trials_per_scenario": evaluation_trials},
        "gate": {"frame_false_trigger_target": 0.01,
                 "threshold": threshold,
                 "validation_false_trigger_rate": float(np.mean(validation > threshold)),
                 "calibrated_target_load": 6},
        "calibrator": calibrator.to_dict(),
        "separated": run_comparison(scenario="separated", **common),
        "paired_collision": run_comparison(scenario="paired_collision", **common),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-scenes", type=int, default=500)
    parser.add_argument("--null-scenes", type=int, default=1000)
    parser.add_argument("--validation-scenes", type=int, default=300)
    parser.add_argument("--evaluation-trials", type=int, default=100)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("results/multistatic_physics_glrt_gate.json"))
    args = parser.parse_args()
    payload = run_physics_gate(
        args.calibration_scenes, args.null_scenes, args.validation_scenes,
        args.evaluation_trials, args.transmitters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
