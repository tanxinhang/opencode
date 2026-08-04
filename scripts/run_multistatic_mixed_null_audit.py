"""Audit weak-FWER step-down GLRT in a mixed collision/null scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_g0b import evaluate_trial
from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator


def _rows(seed, trials, calibrator, mode, thresholds):
    rng = np.random.default_rng(seed)
    common = dict(
        rng=rng, transmitter_count=8, target_count=6,
        miss_probability=0.08, false_mean=0.4, use_spatial_index=True,
        method="bic_conflict", scenario="single_pair_collision",
        clutter_model="correlated_sidelobes",
        view_false_target_probability=0.05, geometry_mode="nested_12",
        confidence_model="overlap", collision_gate_mode=mode,
        score_calibrator=calibrator, robust_final_velocity=True,
    )
    if mode == "physics_glrt":
        common["physics_collision_threshold"] = thresholds[0]
    else:
        common["physics_stepdown_thresholds"] = thresholds
    return [evaluate_trial(**common) for _ in range(trials)]


def _summary(rows):
    estimates = np.asarray([row["estimated_targets"] for row in rows])
    return {
        "position_recovery": float(np.mean([
            row["position_set_exact_15m"] for row in rows])),
        "state_recovery": float(np.mean([
            row["position_velocity_state_exact"] for row in rows])),
        "over_count_rate": float(np.mean(estimates > 6)),
        "under_count_rate": float(np.mean(estimates < 6)),
        "mean_estimated_targets": float(np.mean(estimates)),
        "mean_gospa_15m_p2": float(np.mean([
            row["gospa_15m_p2"] for row in rows])),
        "mean_path_f1": float(np.mean([
            row["path_association_f1"] for row in rows])),
    }


def run_mixed_null_audit(calibration_payload, trials=100,
                         seeds=(20261011, 20261012, 20261013)):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    thresholds = tuple(calibration_payload["gate"]["ordered_thresholds"])
    records = []
    for seed in seeds:
        records.append({
            "seed": seed, "trials": trials,
            "single_threshold": _summary(_rows(
                seed, trials, calibrator, "physics_glrt", thresholds
            )),
            "weak_fwer_stepdown": _summary(_rows(
                seed, trials, calibrator, "physics_stepdown", thresholds
            )),
        })
    return {
        "scope": (
            "candidate-level mixed-null audit with exactly one true collision "
            "pair and four separated targets; paired candidate scenes"
        ),
        "ordered_thresholds": list(thresholds),
        "strong_fwer_claimed": False,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=Path(
        "results/multistatic_stepdown_glrt_gate_m8_n6.json"))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_mixed_null_audit_m8_n6.json"))
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    payload = run_mixed_null_audit(calibration, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
