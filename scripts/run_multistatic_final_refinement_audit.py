"""Paired audit of final post-decision joint-refinement iteration limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_cascade_glrt_gate import cascade_summary
from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator


def run_final_refinement_audit(calibration_payload, gate_payload, trials=100,
                               seed=20261081):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    thresholds = tuple(gate_payload["cascade_thresholds"])
    scenarios = {
        "separated": "separated",
        "single_pair_collision": "single_pair_collision",
        "two_pair_collision": "two_pair_collision",
        "three_pair_collision": "paired_collision",
    }
    return {
        "scope": "paired post-decision refinement limit audit",
        "seed": seed,
        "trials_per_scenario": trials,
        "thresholds": list(thresholds),
        "results": {
            label: {
                str(limit): cascade_summary(
                    scenario, trials, seed, calibrator, thresholds, limit
                ) for limit in (3, 10)
            } for label, scenario in scenarios.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_final_refinement_m8_n6.json"))
    args = parser.parse_args()
    payload = run_final_refinement_audit(
        json.loads(args.calibration.read_text(encoding="utf-8")),
        json.loads(args.gate.read_text(encoding="utf-8")), args.trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
