"""Frame-stratified finite-sample calibration of the physical order GLRT."""

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
from scripts.run_multistatic_physics_glrt_gate import (
    finite_sample_upper_threshold, fit_frame_stratified_thresholds,
    frame_excess_stratum,
    frame_physics_components,
)
from uav_otfs_isac.probability_calibration import fit_isotonic_probability


def _false_trigger_rate(frames, thresholds):
    decisions = []
    for frame in frames:
        maximum = max((item[0] for item in frame), default=-np.inf)
        decisions.append(maximum > thresholds[frame_excess_stratum(frame)])
    return float(np.mean(decisions))


def _serializable_threshold(value):
    return float(value) if np.isfinite(value) else None


def run_frame_stratified_gate(
    probability_scenes=500, null_frames=2000, validation_frames=500,
    evaluation_trials=100, transmitters=8,
    probability_seed=20260941, null_seed=20260942,
    validation_seed=20260943, evaluation_seed=20260944,
):
    if len({probability_seed, null_seed, validation_seed, evaluation_seed}) != 4:
        raise ValueError("all four data partitions require distinct seeds")
    scores, labels = calibration_sample(
        probability_seed, probability_scenes, transmitters
    )
    calibrator = fit_isotonic_probability(scores, labels)
    null = frame_physics_components(
        null_seed, null_frames, calibrator, transmitters
    )
    thresholds, counts = fit_frame_stratified_thresholds(null, 0.01)
    pooled_maxima = np.asarray([
        max((item[0] for item in frame), default=-np.inf) for frame in null
    ])
    pooled_threshold = finite_sample_upper_threshold(
        pooled_maxima[np.isfinite(pooled_maxima)], 0.01
    )
    validation = frame_physics_components(
        validation_seed, validation_frames, calibrator, transmitters
    )
    validation_counts = {
        key: sum(frame_excess_stratum(frame) == key for frame in validation)
        for key in (0, 1)
    }
    common = dict(
        trials=evaluation_trials, seed=evaluation_seed,
        transmitters=transmitters, targets=6,
        clutter_model="correlated_sidelobes",
        view_false_target_probability=0.05, geometry_mode="nested_12",
        confidence_model="overlap",
        collision_gate_mode="physics_frame_stratified",
        score_calibrator=calibrator, physics_frame_thresholds=thresholds,
    )
    pooled_common = {
        **common,
        "collision_gate_mode": "physics_glrt",
        "physics_frame_thresholds": None,
        "physics_collision_threshold": pooled_threshold,
    }
    return {
        "scope": (
            "finite-sample frame-maximum physical GLRT conditioned on a fixed "
            "binary same-UAV excess-peak stratum; separated synthetic N=6 null"
        ),
        "seeds": {"probability": probability_seed, "null": null_seed,
                  "validation": validation_seed, "evaluation": evaluation_seed},
        "sample_sizes": {"probability_scenes": probability_scenes,
                         "null_frames": null_frames,
                         "validation_frames": validation_frames,
                         "evaluation_trials_per_scenario": evaluation_trials},
        "gate": {"frame_false_trigger_target": 0.01,
                 "thresholds": {
                     str(k): _serializable_threshold(v)
                     for k, v in thresholds.items()},
                 "null_stratum_counts": {str(k): v for k, v in counts.items()},
                 "validation_stratum_counts": {
                     str(k): v for k, v in validation_counts.items()},
                 "validation_false_trigger_rate": _false_trigger_rate(
                     validation, thresholds),
                 "calibrated_target_load": 6},
        "same_partition_pooled_gate": {
            "threshold": pooled_threshold,
            "validation_false_trigger_rate": _false_trigger_rate(
                validation, {0: pooled_threshold, 1: pooled_threshold}),
        },
        "probability_calibrator": calibrator.to_dict(),
        "separated": run_comparison(scenario="separated", **common),
        "paired_collision": run_comparison(scenario="paired_collision", **common),
        "pooled_separated": run_comparison(
            scenario="separated", **pooled_common),
        "pooled_paired_collision": run_comparison(
            scenario="paired_collision", **pooled_common),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-scenes", type=int, default=500)
    parser.add_argument("--null-frames", type=int, default=2000)
    parser.add_argument("--validation-frames", type=int, default=500)
    parser.add_argument("--evaluation-trials", type=int, default=100)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_frame_stratified_glrt_gate_m8_n6.json"))
    args = parser.parse_args()
    payload = run_frame_stratified_gate(
        args.probability_scenes, args.null_frames, args.validation_frames,
        args.evaluation_trials, args.transmitters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
