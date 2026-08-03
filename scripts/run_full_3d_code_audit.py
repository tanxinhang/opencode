from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_3d_gate import evaluate_full_search


def random_code_pair(seed, length=8):
    rng = np.random.default_rng(seed)
    return [
        np.exp(0.5j * np.pi * rng.integers(0, 4, length)) / np.sqrt(length)
        for _ in range(2)
    ]


def main():
    screening = []
    for seed in range(200, 220):
        result = evaluate_full_search(
            "cazac_codes", 5.0, trials=200,
            supplied_codes=random_code_pair(seed), calibration_trials=2_000,
        )
        screening.append({
            "seed": seed,
            "joint_position": result["joint_position_resolution_probability"],
            "joint_identity": result["joint_identity_correct_probability"],
        })
    ranked = sorted(screening, key=lambda row: row["joint_identity"])
    selected = {
        "worst": ranked[0]["seed"],
        "median": ranked[len(ranked) // 2]["seed"],
        "best": ranked[-1]["seed"],
    }
    validation = {
        label: evaluate_full_search(
            "cazac_codes", 5.0, trials=1_000,
            supplied_codes=random_code_pair(seed), calibration_trials=10_000,
        )
        for label, seed in selected.items()
    }
    cazac = evaluate_full_search(
        "cazac_codes", 5.0, trials=1_000, calibration_trials=10_000
    )
    payload = {
        "screening": screening,
        "selected_random_seeds": selected,
        "independent_validation": validation,
        "cazac_reference": cazac,
        "warning": (
            "Best/median/worst labels are selected on a small screening set; "
            "only independent validation values are used for comparison."
        ),
    }
    output = Path("results/full_3d_code_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
