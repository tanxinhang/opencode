"""Truth-labeled offline audit of coarse/refined GLRT complementarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_configuration_stepdown_gate import (
    labeled_physics_components,
)
from scripts.run_multistatic_glrt_information_audit import empirical_auc
from uav_otfs_isac.probability_calibration import IsotonicProbabilityCalibrator


def paired_component_statistics(seed, frames, calibrator, scenario):
    coarse = labeled_physics_components(seed, frames, calibrator, scenario, 8, 0)
    refined = labeled_physics_components(seed, frames, calibrator, scenario, 8, 3)
    pairs = []
    if len(coarse) != len(refined):
        raise AssertionError("coarse/refined frame count mismatch")
    for coarse_frame, refined_frame in zip(coarse, refined):
        if len(coarse_frame) != len(refined_frame):
            raise AssertionError("coarse/refined component count mismatch")
        for left, right in zip(coarse_frame, refined_frame):
            if (left["is_collision"] != right["is_collision"] or
                    left["target_count"] != right["target_count"]):
                raise AssertionError("coarse/refined truth label mismatch")
            pairs.append((left["gain"], right["gain"], left["is_collision"]))
    return pairs


def run_complementarity_audit(calibration_payload, frames=300,
                              coarse_threshold=64.92287911375547,
                              refined_threshold=55.95774430087211,
                              collision_seed=20261061,
                              normal_seed=20261062):
    calibrator = IsotonicProbabilityCalibrator.from_dict(
        calibration_payload["probability_calibrator"]
    )
    pairs = paired_component_statistics(
        collision_seed, frames, calibrator, "paired_collision"
    ) + paired_component_statistics(
        normal_seed, frames, calibrator, "separated"
    )
    coarse = np.asarray([item[0] for item in pairs])
    refined = np.asarray([item[1] for item in pairs])
    collision = np.asarray([item[2] for item in pairs], dtype=bool)
    coarse_hit = coarse > coarse_threshold
    refined_hit = refined > refined_threshold
    return {
        "scope": "offline truth-labeled component complementarity audit",
        "seeds": {"collision": collision_seed, "normal": normal_seed},
        "frames": frames,
        "component_counts": {"collision": int(np.sum(collision)),
                             "normal": int(np.sum(~collision))},
        "thresholds": {"coarse": coarse_threshold,
                       "refined": refined_threshold},
        "spearman_rank_correlation": float(spearmanr(coarse, refined).statistic),
        "auc": {
            "coarse": empirical_auc(coarse[~collision], coarse[collision]),
            "refined": empirical_auc(refined[~collision], refined[collision]),
        },
        "collision_detection": {
            "coarse_only": int(np.sum(collision & coarse_hit & ~refined_hit)),
            "refined_only": int(np.sum(collision & ~coarse_hit & refined_hit)),
            "both": int(np.sum(collision & coarse_hit & refined_hit)),
            "neither": int(np.sum(collision & ~coarse_hit & ~refined_hit)),
        },
        "normal_false_trigger": {
            "coarse_only": int(np.sum(~collision & coarse_hit & ~refined_hit)),
            "refined_only": int(np.sum(~collision & ~coarse_hit & refined_hit)),
            "both": int(np.sum(~collision & coarse_hit & refined_hit)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_glrt_complementarity_m8_n6.json"))
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    thresholds = gate["cascade_thresholds"]
    payload = run_complementarity_audit(
        calibration, args.frames, thresholds[1], thresholds[2]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
