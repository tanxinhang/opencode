"""Coarse-to-refined cascade GLRT for a weak third collision component."""

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
    labeled_physics_components, normal_component_maxima, proposed_summary,
)
from scripts.run_multistatic_g0b import evaluate_trial
from scripts.run_multistatic_physics_glrt_gate import (
    finite_sample_upper_threshold, fit_ordered_frame_thresholds,
    frame_physics_components,
)
from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator


def cascade_summary(scenario, trials, seed, calibrator, thresholds,
                    final_refinement_iterations=3,
                    covariance_weighted_final_state=False):
    rng = np.random.default_rng(seed)
    rows = [evaluate_trial(
        rng, 8, 6, 0.08, 0.4, True, "bic_conflict", scenario,
        "correlated_sidelobes", 0.05, "nested_12", "overlap",
        "physics_cascade", calibrator, None, None, None, None, None,
        True, 1, 0, tuple(thresholds), final_refinement_iterations,
        covariance_weighted_final_state,
    ) for _ in range(trials)]
    estimates = np.asarray([row["estimated_targets"] for row in rows])
    matched = sum(row["matched_targets"] for row in rows)
    return {
        "trials": trials,
        "position_set_exact_15m": float(np.mean([
            row["position_set_exact_15m"] for row in rows])),
        "position_velocity_state_exact": float(np.mean([
            row["position_velocity_state_exact"] for row in rows])),
        "over_count_rate": float(np.mean(estimates > 6)),
        "under_count_rate": float(np.mean(estimates < 6)),
        "mean_gospa_15m_p2": float(np.mean([
            row["gospa_15m_p2"] for row in rows])),
        "mean_path_f1": float(np.mean([
            row["path_association_f1"] for row in rows])),
        "mean_velocity_error_mps": float(
            sum(row["velocity_error_sum"] for row in rows) / max(matched, 1)),
        "mean_time_ms": float(1000.0 * np.mean([
            row["association_time_s"] for row in rows])),
    }


def run_cascade_gate(calibration_payload, null_frames=500,
                     mixed_null_frames=500, trials=50,
                     null_seed=20261041, single_pair_seed=20261042,
                     two_pair_seed=20261043, evaluation_seed=20261044):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    coarse_null = frame_physics_components(
        null_seed, null_frames, calibrator, 8, "separated", 0
    )
    coarse_thresholds = fit_ordered_frame_thresholds(
        coarse_null, 4, 0.01
    )
    single_pair_coarse = labeled_physics_components(
        single_pair_seed, mixed_null_frames, calibrator,
        "single_pair_collision", 8, 0
    )
    second_normal_maxima = normal_component_maxima(single_pair_coarse)
    finite_second = second_normal_maxima[np.isfinite(second_normal_maxima)]
    second_threshold = finite_sample_upper_threshold(finite_second, 0.01)
    two_pair_refined = labeled_physics_components(
        two_pair_seed, mixed_null_frames, calibrator, "two_pair_collision", 8, 3
    )
    normal_maxima = normal_component_maxima(two_pair_refined)
    finite_normal = normal_maxima[np.isfinite(normal_maxima)]
    third_threshold = finite_sample_upper_threshold(finite_normal, 0.01)
    cascade_thresholds = (
        coarse_thresholds[0], second_threshold, third_threshold
    )
    scenarios = {
        "separated": "separated",
        "single_pair_collision": "single_pair_collision",
        "two_pair_collision": "two_pair_collision",
        "three_pair_collision": "paired_collision",
    }
    results = {}
    for label, scenario in scenarios.items():
        results[label] = {
            "single_threshold": proposed_summary(
                scenario, trials, evaluation_seed, calibrator,
                (coarse_thresholds[0],) * 4, 8),
            "coarse_stepdown": proposed_summary(
                scenario, trials, evaluation_seed, calibrator,
                coarse_thresholds, 8),
            "cascade": cascade_summary(
                scenario, trials, evaluation_seed, calibrator,
                cascade_thresholds),
        }
    return {
        "scope": (
            "coarse first-two collision tests followed by a refined third test; "
            "third null calibrated on normal components in two-pair frames"
        ),
        "seeds": {"global_null": null_seed,
                  "single_pair_null": single_pair_seed,
                  "two_pair_null": two_pair_seed,
                  "evaluation": evaluation_seed},
        "sample_sizes": {"global_null_frames": null_frames,
                         "two_pair_null_frames": mixed_null_frames,
                         "finite_second_null_maxima": int(len(finite_second)),
                         "finite_third_null_maxima": int(len(finite_normal)),
                         "trials_per_scenario": trials},
        "coarse_ordered_thresholds": list(coarse_thresholds),
        "cascade_thresholds": list(cascade_thresholds),
        "third_test_refinement_iterations": 3,
        "strong_fwer_claimed": False,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--null-frames", type=int, default=500)
    parser.add_argument("--mixed-null-frames", type=int, default=500)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_cascade_glrt_m8_n6.json"))
    args = parser.parse_args()
    payload = run_cascade_gate(
        json.loads(args.calibration.read_text(encoding="utf-8")),
        args.null_frames, args.mixed_null_frames, args.trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
