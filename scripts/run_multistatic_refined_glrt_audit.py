"""Paired coarse-versus-refined physical GLRT audit at equal null calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_configuration_stepdown_gate import proposed_summary
from scripts.run_multistatic_physics_glrt_gate import (
    fit_ordered_frame_thresholds, frame_physics_components,
)
from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator


def run_refined_glrt_audit(calibration_payload, null_frames=500, trials=50,
                           null_seed=20261031, evaluation_seed=20261032):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    thresholds = {}
    for name, iterations in (("coarse", 0), ("refined", 3)):
        frames = frame_physics_components(
            null_seed, null_frames, calibrator, 8, "separated", iterations
        )
        thresholds[name] = fit_ordered_frame_thresholds(frames, 4, 0.01)
    scenarios = {
        "separated": "separated",
        "single_pair_collision": "single_pair_collision",
        "two_pair_collision": "two_pair_collision",
        "three_pair_collision": "paired_collision",
    }
    results = {}
    for label, scenario in scenarios.items():
        results[label] = {}
        for name, iterations in (("coarse", 0), ("refined", 3)):
            results[label][name] = proposed_summary(
                scenario, trials, evaluation_seed, calibrator,
                thresholds[name], 8, activation_count=1,
                glrt_refinement_iterations=iterations,
            )
    return {
        "scope": (
            "paired coarse versus three-iteration physical GLRT; each statistic "
            "uses its own finite-sample ordered global-null thresholds"
        ),
        "seeds": {"null": null_seed, "evaluation": evaluation_seed},
        "sample_sizes": {"null_frames_per_method": null_frames,
                         "trials_per_scenario": trials},
        "joint_refinement_iterations": {"coarse": 0, "refined": 3},
        "ordered_thresholds": {
            key: list(value) for key, value in thresholds.items()
        },
        "strong_fwer_claimed": False,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--null-frames", type=int, default=500)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_refined_glrt_m8_n6.json"))
    args = parser.parse_args()
    payload = run_refined_glrt_audit(
        json.loads(args.calibration.read_text(encoding="utf-8")),
        args.null_frames, args.trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
