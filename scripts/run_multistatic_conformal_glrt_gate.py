"""Excess-peak-stratified split-conformal physical collision Gate."""

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
    fit_excess_peak_null, frame_minimum_pvalues, frame_physics_components,
)
from uav_otfs_isac.probability_calibration import fit_isotonic_probability


def run_conformal_gate(
    probability_scenes=500, component_null_frames=1000,
    frame_threshold_frames=1000, validation_frames=300,
    evaluation_trials=100, transmitters=8,
    probability_seed=20260931, component_seed=20260932,
    threshold_seed=20260933, validation_seed=20260934,
    evaluation_seed=20260935,
):
    seeds = {probability_seed, component_seed, threshold_seed,
             validation_seed, evaluation_seed}
    if len(seeds) != 5:
        raise ValueError("all five data partitions require distinct seeds")
    scores, labels = calibration_sample(
        probability_seed, probability_scenes, transmitters
    )
    calibrator = fit_isotonic_probability(scores, labels)
    component_frames = frame_physics_components(
        component_seed, component_null_frames, calibrator, transmitters
    )
    null_model = fit_excess_peak_null(component_frames)
    threshold_frames = frame_physics_components(
        threshold_seed, frame_threshold_frames, calibrator, transmitters
    )
    minimum_pvalues = frame_minimum_pvalues(threshold_frames, null_model)
    p_threshold = float(np.quantile(minimum_pvalues, 0.01, method="lower"))
    validation = frame_physics_components(
        validation_seed, validation_frames, calibrator, transmitters
    )
    validation_minimum = frame_minimum_pvalues(validation, null_model)
    common = dict(
        trials=evaluation_trials, seed=evaluation_seed,
        transmitters=transmitters, targets=6,
        clutter_model="correlated_sidelobes",
        view_false_target_probability=0.05, geometry_mode="nested_12",
        confidence_model="overlap", collision_gate_mode="physics_conformal",
        score_calibrator=calibrator, physics_conformal_null=null_model,
        physics_conformal_p_threshold=p_threshold,
    )
    return {
        "scope": (
            "split empirical-conformal physical GLRT, stratified by same-UAV "
            "excess-peak count and calibrated for separated N=6 frames"
        ),
        "seeds": {"probability": probability_seed,
                  "component_null": component_seed,
                  "frame_threshold": threshold_seed,
                  "validation": validation_seed,
                  "evaluation": evaluation_seed},
        "sample_sizes": {"probability_scenes": probability_scenes,
                         "component_null_frames": component_null_frames,
                         "frame_threshold_frames": frame_threshold_frames,
                         "validation_frames": validation_frames,
                         "evaluation_trials_per_scenario": evaluation_trials},
        "gate": {"frame_false_trigger_target": 0.01,
                 "minimum_p_threshold": p_threshold,
                 "validation_false_trigger_rate": float(np.mean(
                     validation_minimum < p_threshold)),
                 "stratum_component_counts": {
                     str(key): len(values) for key, values in null_model.strata.items()
                 },
                 "calibrated_target_load": 6},
        "probability_calibrator": calibrator.to_dict(),
        "component_null": null_model.to_dict(),
        "separated": run_comparison(scenario="separated", **common),
        "paired_collision": run_comparison(scenario="paired_collision", **common),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-scenes", type=int, default=500)
    parser.add_argument("--component-null-frames", type=int, default=1000)
    parser.add_argument("--frame-threshold-frames", type=int, default=1000)
    parser.add_argument("--validation-frames", type=int, default=300)
    parser.add_argument("--evaluation-trials", type=int, default=100)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("results/multistatic_conformal_glrt_gate.json"))
    args = parser.parse_args()
    payload = run_conformal_gate(
        args.probability_scenes, args.component_null_frames,
        args.frame_threshold_frames, args.validation_frames,
        args.evaluation_trials, args.transmitters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
