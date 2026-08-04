"""Independent probability-calibration Gate for multistatic association."""

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
from scripts.run_multistatic_g0b import (
    CARRIER_HZ, PROPAGATION_SPEED, draw_targets, imperfect_candidates,
    nested_transmitter_geometry,
)
from uav_otfs_isac.multistatic_association import PathCandidate
from uav_otfs_isac.multistatic_baselines import _dbscan_labels, _project
from uav_otfs_isac.multistatic_targets import (
    KinematicNode, generate_bistatic_paths,
)
from uav_otfs_isac.probability_calibration import (
    fit_isotonic_probability, probability_metrics,
)
from uav_otfs_isac.multistatic_model_selection import (
    collision_support_threshold, poisson_binomial_tail,
)


def calibration_sample(seed: int, scenes: int, transmitters: int = 8):
    """Draw labelled front-end candidates for calibration only."""
    rng = np.random.default_rng(seed)
    nodes = nested_transmitter_geometry(transmitters)
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    scores, labels = [], []
    for scene in range(scenes):
        scenario = "separated" if scene % 2 == 0 else "paired_collision"
        targets = draw_targets(rng, 6, scenario)
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        candidates, truth, _ = imperfect_candidates(
            rng, paths, transmitters, miss_probability=0.08,
            false_mean=0.4, clutter_model="correlated_sidelobes",
            confidence_model="overlap",
        )
        scores.extend(candidate.confidence for candidate in candidates)
        labels.extend(truth[id(candidate)] is not None for candidate in candidates)
    return np.asarray(scores), np.asarray(labels, dtype=float)


def rank_support_sample(
    seed: int, scenes: int, base_calibrator, transmitters: int = 8,
    maximum_order: int = 4,
):
    """Selection-aware samples for the event needed by order-q gating.

    For each data-selected local component and UAV, the q-th score is labelled
    one only when the top-q candidates contain q distinct true target IDs.
    Missing q-th candidates are handled as zero probability by the receiver
    and are not included in fitting the conditional score mapping.
    """
    rng = np.random.default_rng(seed)
    nodes = nested_transmitter_geometry(transmitters)
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    samples = {order: ([], []) for order in range(2, maximum_order + 1)}
    for scene in range(scenes):
        scenario = "separated" if scene % 2 == 0 else "paired_collision"
        targets = draw_targets(rng, 6, scenario)
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        candidates, truth, _ = imperfect_candidates(
            rng, paths, transmitters, miss_probability=0.08,
            false_mean=0.4, clutter_model="correlated_sidelobes",
            confidence_model="overlap",
        )
        calibrated = []
        calibrated_truth = {}
        for candidate in candidates:
            item = PathCandidate(
                candidate.transmitter_id, candidate.delay_s,
                candidate.doppler_hz, candidate.receive_azimuth_rad,
                float(base_calibrator(candidate.confidence)),
            )
            calibrated.append(item)
            calibrated_truth[id(item)] = truth[id(candidate)]
        projected = _project(
            calibrated, nodes, receiver, PROPAGATION_SPEED
        )
        if not projected:
            continue
        positions = np.asarray([item[1] for item in projected])
        labels = _dbscan_labels(positions / 14.0, radius=1.0, min_samples=2)
        for label in range(int(labels.max()) + 1):
            members = [projected[index]
                       for index in np.flatnonzero(labels == label)]
            for transmitter_id in range(transmitters):
                ranked = sorted((
                    item[0] for item in members
                    if item[0].transmitter_id == transmitter_id
                ), key=lambda candidate: candidate.confidence, reverse=True)
                for order in range(2, maximum_order + 1):
                    if len(ranked) < order:
                        continue
                    top_labels = [calibrated_truth[id(candidate)]
                                  for candidate in ranked[:order]]
                    distinct_targets = {
                        target_id for target_id in top_labels
                        if target_id is not None
                    }
                    scores, outcomes = samples[order]
                    scores.append(ranked[order - 1].confidence)
                    outcomes.append(float(len(distinct_targets) >= order))
    return {
        order: (np.asarray(scores), np.asarray(outcomes))
        for order, (scores, outcomes) in samples.items()
    }


def null_collision_statistics(
    seed: int, scenes: int, base_calibrator, rank_calibrator,
    transmitters: int = 8, separated_targets: int = 6,
):
    """Frame-maximum order-2 statistic under separated multi-target H0.

    ``separated_targets`` is part of the calibrated operating condition: it
    determines how many data-selected local components are scanned per frame.
    The resulting family-wise threshold must not be extrapolated to an
    arbitrary unknown target load without alpha spending or a load bound.
    """
    rng = np.random.default_rng(seed)
    nodes = nested_transmitter_geometry(transmitters)
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    required_support = collision_support_threshold(
        (0.1,) * transmitters, 0.05
    )
    maxima = []
    for _ in range(scenes):
        targets = draw_targets(rng, separated_targets, "separated")
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        candidates, _, _ = imperfect_candidates(
            rng, paths, transmitters, miss_probability=0.08,
            false_mean=0.4, clutter_model="correlated_sidelobes",
            confidence_model="overlap",
        )
        calibrated = [PathCandidate(
            candidate.transmitter_id, candidate.delay_s,
            candidate.doppler_hz, candidate.receive_azimuth_rad,
            float(base_calibrator(candidate.confidence)),
        ) for candidate in candidates]
        projected = _project(calibrated, nodes, receiver, PROPAGATION_SPEED)
        scene_statistics = [0.0]
        if projected:
            positions = np.asarray([item[1] for item in projected])
            labels = _dbscan_labels(positions / 14.0, radius=1.0, min_samples=2)
            for label in range(int(labels.max()) + 1):
                members = [projected[index]
                           for index in np.flatnonzero(labels == label)]
                probabilities = []
                for transmitter_id in range(transmitters):
                    scores = sorted((
                        candidate.confidence for candidate, _ in members
                        if candidate.transmitter_id == transmitter_id
                    ), reverse=True)
                    probabilities.append(
                        float(rank_calibrator(scores[1])) if len(scores) >= 2
                        else 0.0
                    )
                scene_statistics.append(poisson_binomial_tail(
                    probabilities, required_support
                ))
        maxima.append(max(scene_statistics))
    return np.asarray(maxima)


def run_calibrated_gate(
    calibration_scenes: int = 500,
    validation_scenes: int = 200,
    rank_calibration_scenes: int = 300,
    null_calibration_scenes: int = 1000,
    evaluation_trials: int = 100,
    calibration_seed: int = 20260901,
    rank_calibration_seed: int = 20260904,
    null_calibration_seed: int = 20260905,
    validation_seed: int = 20260902,
    evaluation_seed: int = 20260903,
    transmitters: int = 8,
) -> dict:
    if len({calibration_seed, rank_calibration_seed, null_calibration_seed,
            validation_seed, evaluation_seed}) != 5:
        raise ValueError("all calibration, validation, and evaluation seeds must differ")
    train_scores, train_labels = calibration_sample(
        calibration_seed, calibration_scenes, transmitters
    )
    calibrator = fit_isotonic_probability(train_scores, train_labels)
    rank_samples = rank_support_sample(
        rank_calibration_seed, rank_calibration_scenes,
        calibrator, transmitters,
    )
    rank_calibrators = {
        order: fit_isotonic_probability(scores, labels)
        for order, (scores, labels) in rank_samples.items()
        if len(scores) and len(np.unique(labels)) > 1
    }
    if 2 not in rank_calibrators:
        raise RuntimeError("rank calibration did not contain two-class order-2 data")
    null_statistics = null_collision_statistics(
        null_calibration_seed, null_calibration_scenes, calibrator,
        rank_calibrators[2], transmitters,
    )
    collision_thresholds = {
        2: float(np.quantile(null_statistics, 0.99, method="higher"))
    }
    null_validation_statistics = null_collision_statistics(
        validation_seed + 10_000, validation_scenes, calibrator,
        rank_calibrators[2], transmitters,
    )
    validation_scores, validation_labels = calibration_sample(
        validation_seed, validation_scenes, transmitters
    )
    raw_validation = probability_metrics(validation_scores, validation_labels)
    calibrated_validation = probability_metrics(
        calibrator.predict(validation_scores), validation_labels
    )
    common = dict(
        trials=evaluation_trials, seed=evaluation_seed,
        transmitters=transmitters, targets=6,
        clutter_model="correlated_sidelobes",
        view_false_target_probability=0.05,
        geometry_mode="nested_12", confidence_model="overlap",
        score_calibrator=calibrator,
        collision_support_calibrators=rank_calibrators,
        collision_statistic_thresholds=collision_thresholds,
    )
    separated = run_comparison(
        scenario="separated", collision_gate_mode="empirical_null", **common
    )
    collision = run_comparison(
        scenario="paired_collision", collision_gate_mode="empirical_null",
        **common,
    )
    return {
        "scope": (
            "score-only isotonic calibration on independent synthetic candidate "
            "scenes; not an end-to-end OTFS front-end result"
        ),
        "seeds": {
            "calibration": calibration_seed,
            "rank_calibration": rank_calibration_seed,
            "null_calibration": null_calibration_seed,
            "validation": validation_seed,
            "evaluation": evaluation_seed,
        },
        "sample_sizes": {
            "calibration_scenes": calibration_scenes,
            "calibration_candidates": len(train_scores),
            "validation_scenes": validation_scenes,
            "validation_candidates": len(validation_scores),
            "evaluation_trials_per_scenario": evaluation_trials,
            "rank_calibration_scenes": rank_calibration_scenes,
            "null_calibration_scenes": null_calibration_scenes,
            "rank_samples_by_order": {
                str(order): len(scores)
                for order, (scores, _) in rank_samples.items()
            },
        },
        "class_prevalence": {
            "calibration": float(np.mean(train_labels)),
            "validation": float(np.mean(validation_labels)),
        },
        "validation_probability_metrics": {
            "raw_score_as_probability": raw_validation,
            "isotonic_probability": calibrated_validation,
        },
        "calibrator": calibrator.to_dict(),
        "rank_support_calibrators": {
            str(order): model.to_dict()
            for order, model in rank_calibrators.items()
        },
        "empirical_collision_gate": {
            "scene_false_trigger_target": 0.01,
            "order_2_threshold": collision_thresholds[2],
            "calibrated_separated_target_load": 6,
            "null_validation_scenes": validation_scenes,
            "null_validation_false_trigger_rate": float(np.mean(
                null_validation_statistics > collision_thresholds[2]
            )),
        },
        "separated": separated,
        "paired_collision": collision,
        "warning": (
            "Calibration labels come from the synthetic candidate generator. "
            "A deployable claim requires candidates and labels from an actual "
            "OTFS MF/CFAR or sparse-recovery front end."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-scenes", type=int, default=500)
    parser.add_argument("--validation-scenes", type=int, default=200)
    parser.add_argument("--null-calibration-scenes", type=int, default=1000)
    parser.add_argument("--evaluation-trials", type=int, default=100)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/multistatic_calibrated_gate.json"),
    )
    args = parser.parse_args()
    payload = run_calibrated_gate(
        args.calibration_scenes, args.validation_scenes, 300,
        args.null_calibration_scenes, args.evaluation_trials,
        transmitters=args.transmitters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
