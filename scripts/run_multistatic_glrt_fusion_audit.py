"""Split-calibrated fusion of coarse and refined physical GLRT statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_configuration_stepdown_gate import (
    labeled_physics_components,
)
from scripts.run_multistatic_glrt_information_audit import empirical_auc
from scripts.run_multistatic_physics_glrt_gate import finite_sample_upper_threshold
from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator


def paired_labeled_frames(seed, frames, calibrator, scenario):
    coarse = labeled_physics_components(seed, frames, calibrator, scenario, 8, 0)
    refined = labeled_physics_components(seed, frames, calibrator, scenario, 8, 3)
    output = []
    for left_frame, right_frame in zip(coarse, refined):
        if len(left_frame) != len(right_frame):
            raise AssertionError("coarse/refined component count mismatch")
        frame = []
        for left, right in zip(left_frame, right_frame):
            if (left["is_collision"] != right["is_collision"] or
                    left["target_count"] != right["target_count"]):
                raise AssertionError("coarse/refined component label mismatch")
            frame.append((left["gain"], right["gain"], left["is_collision"]))
        output.append(frame)
    return output


def empirical_percentile(training, values):
    training = np.sort(np.asarray(training, dtype=float))
    values = np.asarray(values, dtype=float)
    if not len(training) or np.any(~np.isfinite(training)):
        raise ValueError("finite nonempty training statistics are required")
    return (np.searchsorted(training, values, side="right") + 1.0) / (
        len(training) + 1.0
    )


def fusion_scores(frames, coarse_training, refined_training):
    return [[
        (float(max(
            empirical_percentile(coarse_training, [coarse])[0],
            empirical_percentile(refined_training, [refined])[0],
        )), collision)
        for coarse, refined, collision in frame
    ] for frame in frames]


def run_fusion_audit(calibration_payload, training_frames=300,
                     threshold_frames=500, evaluation_frames=300,
                     training_seed=20261071, threshold_seed=20261072,
                     collision_seed=20261073, normal_seed=20261074):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    training = paired_labeled_frames(
        training_seed, training_frames, calibrator, "two_pair_collision"
    )
    coarse_null = [item[0] for frame in training for item in frame if not item[2]]
    refined_null = [item[1] for frame in training for item in frame if not item[2]]
    threshold_data = fusion_scores(paired_labeled_frames(
        threshold_seed, threshold_frames, calibrator, "two_pair_collision"
    ), coarse_null, refined_null)
    frame_maxima = np.asarray([
        max((score for score, collision in frame if not collision),
            default=-np.inf)
        for frame in threshold_data
    ])
    finite_maxima = frame_maxima[np.isfinite(frame_maxima)]
    threshold = finite_sample_upper_threshold(finite_maxima, 0.01)
    collision_frames = paired_labeled_frames(
        collision_seed, evaluation_frames, calibrator, "paired_collision"
    )
    normal_frames = paired_labeled_frames(
        normal_seed, evaluation_frames, calibrator, "separated"
    )
    fused_collision = fusion_scores(collision_frames, coarse_null, refined_null)
    fused_normal = fusion_scores(normal_frames, coarse_null, refined_null)
    collision_values = np.asarray([
        score for frame in fused_collision for score, collision in frame if collision
    ])
    normal_values = np.asarray([
        score for frame in fused_normal for score, collision in frame if not collision
    ])
    collision_counts = np.asarray([
        sum(score > threshold for score, collision in frame if collision)
        for frame in fused_collision
    ])
    normal_false = np.asarray([
        any(score > threshold for score, collision in frame if not collision)
        for frame in fused_normal
    ])
    return {
        "scope": (
            "split empirical-CDF max fusion; truth labels select offline normal "
            "calibration components only and are unavailable online"
        ),
        "seeds": {"marginal_training": training_seed,
                  "frame_threshold": threshold_seed,
                  "collision_evaluation": collision_seed,
                  "normal_evaluation": normal_seed},
        "sample_sizes": {"training_frames": training_frames,
                         "coarse_normal_components": len(coarse_null),
                         "refined_normal_components": len(refined_null),
                         "threshold_frames": threshold_frames,
                         "finite_threshold_maxima": int(len(finite_maxima)),
                         "evaluation_frames_per_scenario": evaluation_frames},
        "fusion_threshold": threshold,
        "normal_frame_false_trigger_rate": float(np.mean(normal_false)),
        "collision_component_auc": empirical_auc(normal_values, collision_values),
        "collision_component_detection_rate": float(np.mean(
            collision_values > threshold)),
        "collision_frame_all_three_rate": float(np.mean(collision_counts >= 3)),
        "collision_trigger_count_distribution": {
            str(count): int(np.sum(collision_counts == count))
            for count in range(int(np.max(collision_counts)) + 1)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--training-frames", type=int, default=300)
    parser.add_argument("--threshold-frames", type=int, default=500)
    parser.add_argument("--evaluation-frames", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_glrt_fusion_m8_n6.json"))
    args = parser.parse_args()
    payload = run_fusion_audit(
        json.loads(args.calibration.read_text(encoding="utf-8")),
        args.training_frames, args.threshold_frames, args.evaluation_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
