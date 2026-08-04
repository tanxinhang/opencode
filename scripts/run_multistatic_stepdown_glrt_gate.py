"""Successive frame-order physical GLRT with global-null FWER control."""

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
    fit_ordered_frame_thresholds, frame_physics_components,
)
from uav_otfs_isac.probability_calibration import fit_isotonic_probability


def rejection_count(frame, thresholds):
    count = 0
    for statistic, threshold in zip(
        sorted((item[0] for item in frame), reverse=True), thresholds
    ):
        if statistic <= threshold:
            break
        count += 1
    return count


def run_stepdown_gate(
    probability_scenes=500, null_frames=2000, validation_frames=500,
    evaluation_trials=100, transmitters=8,
    probability_seed=20260961, null_seed=20260962,
    validation_seed=20260963, evaluation_seed=20260964,
    robust_final_velocity=False,
):
    if len({probability_seed, null_seed, validation_seed, evaluation_seed}) != 4:
        raise ValueError("all four data partitions require distinct seeds")
    scores, labels = calibration_sample(
        probability_seed, probability_scenes, transmitters
    )
    calibrator = fit_isotonic_probability(scores, labels)
    null = frame_physics_components(null_seed, null_frames, calibrator,
                                    transmitters)
    thresholds = fit_ordered_frame_thresholds(null, 4, 0.01)
    validation = frame_physics_components(
        validation_seed, validation_frames, calibrator, transmitters
    )
    validation_counts = np.asarray([
        rejection_count(frame, thresholds) for frame in validation
    ])
    common = dict(
        trials=evaluation_trials, seed=evaluation_seed,
        transmitters=transmitters, targets=6,
        clutter_model="correlated_sidelobes",
        view_false_target_probability=0.05, geometry_mode="nested_12",
        confidence_model="overlap", collision_gate_mode="physics_stepdown",
        score_calibrator=calibrator,
        physics_stepdown_thresholds=thresholds,
        robust_final_velocity=robust_final_velocity,
    )
    return {
        "scope": (
            "successive ordered frame-level physical GLRT; finite-sample weak "
            "FWER control under the separated N=6 global null only"
        ),
        "seeds": {"probability": probability_seed, "null": null_seed,
                  "validation": validation_seed, "evaluation": evaluation_seed},
        "sample_sizes": {"probability_scenes": probability_scenes,
                         "null_frames": null_frames,
                         "validation_frames": validation_frames,
                         "evaluation_trials_per_scenario": evaluation_trials},
        "gate": {"global_null_frame_false_trigger_target": 0.01,
                 "ordered_thresholds": list(thresholds),
                 "validation_any_rejection_rate": float(np.mean(
                     validation_counts > 0)),
                 "validation_rejection_count_distribution": {
                     str(k): int(np.sum(validation_counts == k))
                     for k in range(int(np.max(validation_counts)) + 1)
                 },
                 "strong_fwer_under_mixed_null": False,
                 "robust_final_velocity": bool(robust_final_velocity),
                 "calibrated_target_load": 6},
        "probability_calibrator": calibrator.to_dict(),
        "separated": run_comparison(scenario="separated", **common),
        "paired_collision": run_comparison(scenario="paired_collision", **common),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-scenes", type=int, default=500)
    parser.add_argument("--null-frames", type=int, default=2000)
    parser.add_argument("--validation-frames", type=int, default=500)
    parser.add_argument("--evaluation-trials", type=int, default=100)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--robust-final-velocity", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_stepdown_glrt_gate_m8_n6.json"))
    args = parser.parse_args()
    payload = run_stepdown_gate(
        args.probability_scenes, args.null_frames, args.validation_frames,
        args.evaluation_trials, args.transmitters,
        robust_final_velocity=args.robust_final_velocity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
