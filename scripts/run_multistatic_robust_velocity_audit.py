"""Multi-seed paired audit of robust post-association velocity refinement."""

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


def _evaluate(seed, trials, scenario, calibrator, thresholds, robust):
    rng = np.random.default_rng(seed)
    return [evaluate_trial(
        rng, 8, 6, 0.08, 0.4, True, "bic_conflict", scenario,
        "correlated_sidelobes", 0.05, "nested_12", "overlap",
        "physics_stepdown", calibrator, None, None, None, None,
        thresholds, robust,
    ) for _ in range(trials)]


def run_robust_velocity_audit(calibration_payload, trials=100,
                              seeds=(20261001, 20261002, 20261003)):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    thresholds = tuple(calibration_payload["gate"]["ordered_thresholds"])
    records = []
    for scenario in ("separated", "paired_collision"):
        for seed in seeds:
            ordinary = _evaluate(
                seed, trials, scenario, calibrator, thresholds, False
            )
            robust = _evaluate(
                seed, trials, scenario, calibrator, thresholds, True
            )
            if [row["position_set_exact_15m"] for row in ordinary] != [
                row["position_set_exact_15m"] for row in robust
            ]:
                raise AssertionError("velocity-only refinement changed positions")
            ordinary_state = np.asarray([
                row["position_velocity_state_exact"] for row in ordinary
            ])
            robust_state = np.asarray([
                row["position_velocity_state_exact"] for row in robust
            ])
            ordinary_velocity = sum(row["velocity_error_sum"] for row in ordinary)
            robust_velocity = sum(row["velocity_error_sum"] for row in robust)
            matches = sum(row["matched_targets"] for row in ordinary)
            records.append({
                "scenario": scenario, "seed": seed, "trials": trials,
                "position_recovery": float(np.mean([
                    row["position_set_exact_15m"] for row in ordinary])),
                "ordinary_state_recovery": float(np.mean(ordinary_state)),
                "robust_state_recovery": float(np.mean(robust_state)),
                "robust_minus_ordinary_state_pp": float(
                    100.0 * np.mean(robust_state - ordinary_state)),
                "robust_wins": int(np.sum(robust_state > ordinary_state)),
                "ordinary_wins": int(np.sum(ordinary_state > robust_state)),
                "ordinary_mean_velocity_error_mps": float(
                    ordinary_velocity / max(matches, 1)),
                "robust_mean_velocity_error_mps": float(
                    robust_velocity / max(matches, 1)),
            })
    return {
        "scope": (
            "paired multi-seed candidate-level audit; robust Huber velocity "
            "refinement uses identical target count, paths, and positions"
        ),
        "fixed_gate_thresholds": list(thresholds),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=Path(
        "results/multistatic_stepdown_glrt_gate_m8_n6.json"))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_robust_velocity_multiseed_m8_n6.json"))
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    payload = run_robust_velocity_audit(calibration, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
