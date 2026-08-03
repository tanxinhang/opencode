from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_confirmation_mismatch_gate import evaluate_case


def main():
    totals = (1.3, 1.5, 1.6, 1.7, 1.8, 2.0)
    cases = {
        "ideal": (1.0, 0.0, 0.0),
        "mild": (0.99, 5.0, 0.01),
        "moderate": (0.95, 15.0, 0.05),
        "severe": (0.9, 30.0, 0.1),
    }
    rows = []
    seed = 20261101
    for case_name, (correlation, phase_std, cfo) in cases.items():
        for total in totals:
            difference_energy = total - 1.0
            # Split the difference observation into two real incremental blocks.
            first = min(0.3, difference_energy / 2.0)
            second = difference_energy - first
            import scripts.run_confirmation_mismatch_gate as mismatch
            original_equal = mismatch.EQUAL_DIFFERENCE_ENERGIES
            original_full = mismatch.FULL_DIFFERENCE_ENERGIES
            try:
                mismatch.EQUAL_DIFFERENCE_ENERGIES = (first, second)
                mismatch.FULL_DIFFERENCE_ENERGIES = (0.3, 0.7)
                result = evaluate_case(
                    correlation, phase_std, cfo, trials=300, seed=seed
                )
            finally:
                mismatch.EQUAL_DIFFERENCE_ENERGIES = original_equal
                mismatch.FULL_DIFFERENCE_ENERGIES = original_full
            seed += 1
            rows.append({
                "case": case_name,
                "total_energy": total,
                "energy_saving_vs_full": 1.0 - total / 2.0,
                "exact_support_probability": result[
                    "exact_support_probability"
                ]["equal"],
                "full_energy_probability": result[
                    "exact_support_probability"
                ]["full"],
                "loss_vs_full": {
                    decoder: -value
                    for decoder, value in result["equal_minus_full"].items()
                },
                "mean_probe_condition_number": result[
                    "mean_actual_probe_condition_number"
                ]["equal"],
            })
    minimum_energy = {}
    for case_name in cases:
        selected = [row for row in rows if row["case"] == case_name]
        minimum_energy[case_name] = {}
        for decoder in ("oracle", "oracle_ridge", "cfo_compensated", "nominal"):
            eligible = [
                row for row in selected
                if row["loss_vs_full"][decoder] <= 0.02
            ]
            minimum_energy[case_name][decoder] = (
                min(eligible, key=lambda row: row["total_energy"])["total_energy"]
                if eligible else None
            )
    payload = {
        "scope": (
            "non-saturated 5-degree same-DD, 6-dB near-far energy curve; "
            "known two-source coarse cluster"
        ),
        "trials_per_point": 300,
        "rows": rows,
        "minimum_total_energy_for_loss_at_most_2pp": minimum_energy,
        "gate_requires_energy_at_most_1.7": True,
        "warning": (
            "Energy points and their full-energy references use independent "
            "noise draws with 300 trials, so nonmonotone minimum-energy entries "
            "are exploratory and must not be treated as precise optima."
        ),
    }
    output = Path("results/confirmation_energy_curve.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "scope": payload["scope"],
        "minimum_total_energy_for_loss_at_most_2pp": minimum_energy,
    }, indent=2))


if __name__ == "__main__":
    main()
