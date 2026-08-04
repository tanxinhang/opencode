"""Evaluate two-strong-component activation for the calibrated step-down GLRT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_configuration_stepdown_gate import proposed_summary
from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator


def run_density_audit(calibration_payload, trials=100, seed=20261023,
                      activation_count=2):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    thresholds = tuple(
        calibration_payload["gate"]["global_null_stepdown_thresholds"]
    )
    scenarios = {
        "separated": "separated",
        "single_pair_collision": "single_pair_collision",
        "two_pair_collision": "two_pair_collision",
        "three_pair_collision": "paired_collision",
    }
    return {
        "scope": (
            "paired candidate-level audit; low step-down thresholds activate "
            "only after at least two components exceed the shared first threshold"
        ),
        "seed": seed,
        "trials_per_scenario": trials,
        "activation_count": activation_count,
        "thresholds": list(thresholds),
        "strong_fwer_claimed": False,
        "results": {
            label: proposed_summary(
                scenario, trials, seed, calibrator, thresholds, 8,
                activation_count=activation_count,
            ) for label, scenario in scenarios.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--activation-count", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_density_stepdown_m8_n6.json"))
    args = parser.parse_args()
    payload = run_density_audit(
        json.loads(args.calibration.read_text(encoding="utf-8")), args.trials,
        activation_count=args.activation_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
