"""Information audit for the frame-maximum physical order-GLRT statistic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_calibrated_gate import calibration_sample
from scripts.run_multistatic_physics_glrt_gate import (
    finite_sample_upper_threshold, frame_physics_components,
)
from uav_otfs_isac.probability_calibration import fit_isotonic_probability


def empirical_auc(null_statistics, alternative_statistics):
    """Probability that an alternative score exceeds a null score.

    Ties receive half credit, which is the standard Mann--Whitney definition
    and remains valid for discrete or clipped statistics.
    """
    null = np.sort(np.asarray(null_statistics, dtype=float))
    alternative = np.asarray(alternative_statistics, dtype=float)
    if null.ndim != 1 or alternative.ndim != 1 or not len(null) or not len(alternative):
        raise ValueError("both statistic samples must be nonempty vectors")
    less = np.searchsorted(null, alternative, side="left")
    less_equal = np.searchsorted(null, alternative, side="right")
    return float(np.mean((less + less_equal) / (2.0 * len(null))))


def summarize_statistics(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
    }


def run_information_audit(
    probability_scenes=500, null_frames=2000, collision_frames=1000,
    transmitters=8, probability_seed=20260951, null_seed=20260952,
    collision_seed=20260953,
):
    if len({probability_seed, null_seed, collision_seed}) != 3:
        raise ValueError("all three data partitions require distinct seeds")
    scores, labels = calibration_sample(
        probability_seed, probability_scenes, transmitters
    )
    calibrator = fit_isotonic_probability(scores, labels)
    null_components = frame_physics_components(
        null_seed, null_frames, calibrator, transmitters, "separated"
    )
    collision_components = frame_physics_components(
        collision_seed, collision_frames, calibrator, transmitters,
        "paired_collision",
    )
    null = np.asarray([
        max((item[0] for item in frame), default=-np.inf)
        for frame in null_components
    ])
    collision = np.asarray([
        max((item[0] for item in frame), default=-np.inf)
        for frame in collision_components
    ])
    finite_null = null[np.isfinite(null)]
    finite_collision = collision[np.isfinite(collision)]
    threshold = finite_sample_upper_threshold(finite_null, 0.01)
    collision_trigger_counts = np.asarray([
        sum(item[0] > threshold for item in frame)
        for frame in collision_components
    ])
    return {
        "scope": (
            "candidate-level separability audit of the frame-maximum physical "
            "order GLRT; this is not an end-to-end OTFS result"
        ),
        "seeds": {"probability": probability_seed, "null": null_seed,
                  "collision": collision_seed},
        "sample_sizes": {"probability_scenes": probability_scenes,
                         "null_frames": null_frames,
                         "collision_frames": collision_frames},
        "frame_false_trigger_target": 0.01,
        "finite_sample_threshold": threshold,
        "calibration_false_trigger_rate": float(np.mean(null > threshold)),
        "collision_trigger_rate_at_one_percent": float(np.mean(
            collision > threshold)),
        "collision_component_trigger_count": {
            "mean": float(np.mean(collision_trigger_counts)),
            "distribution": {
                str(count): int(np.sum(collision_trigger_counts == count))
                for count in range(int(np.max(collision_trigger_counts)) + 1)
            },
            "at_least_three_rate": float(np.mean(
                collision_trigger_counts >= 3)),
        },
        "empirical_auc": empirical_auc(finite_null, finite_collision),
        "null_statistics": summarize_statistics(finite_null),
        "collision_statistics": summarize_statistics(finite_collision),
        "nonfinite_frame_counts": {
            "null": int(np.sum(~np.isfinite(null))),
            "collision": int(np.sum(~np.isfinite(collision))),
        },
        "probability_calibrator": calibrator.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-scenes", type=int, default=500)
    parser.add_argument("--null-frames", type=int, default=2000)
    parser.add_argument("--collision-frames", type=int, default=1000)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_glrt_information_audit_m8_n6.json"))
    args = parser.parse_args()
    payload = run_information_audit(
        args.probability_scenes, args.null_frames, args.collision_frames,
        args.transmitters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
