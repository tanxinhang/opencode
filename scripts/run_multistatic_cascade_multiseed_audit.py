"""Independent multi-seed validation of fixed cascade GLRT thresholds."""

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


def run_multiseed_audit(calibration_payload, gate_payload, trials=50,
                        seeds=(20261051, 20261052, 20261053)):
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
    records = []
    for seed in seeds:
        for label, scenario in scenarios.items():
            records.append({
                "seed": seed, "scenario": label,
                **cascade_summary(
                    scenario, trials, seed, calibrator, thresholds
                ),
            })
    return {
        "scope": "fixed-threshold independent multi-seed cascade validation",
        "thresholds": list(thresholds),
        "trials_per_seed_scenario": trials,
        "seeds": list(seeds),
        "strong_fwer_claimed": False,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_cascade_multiseed_m8_n6.json"))
    args = parser.parse_args()
    payload = run_multiseed_audit(
        json.loads(args.calibration.read_text(encoding="utf-8")),
        json.loads(args.gate.read_text(encoding="utf-8")), args.trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
