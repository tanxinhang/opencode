from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dd_gate1_oracle_audit import evaluate_scenario


def main():
    cases = {}
    for delay_gap, doppler_gap in (
        (0.25, 0.15), (0.50, 0.30), (1.00, 0.60), (2.00, 1.20)
    ):
        name = f"dtau_{delay_gap:.2f}_dnu_{doppler_gap:.2f}"
        delays = np.array([4.20, 4.20 + delay_gap, 8.10, 11.30])
        dopplers = np.array([2.15, 2.15 + doppler_gap, 5.42, 6.25])
        result = evaluate_scenario(
            delays, dopplers, trials=120, validation_trials=1_000
        )
        validation = result["independent_validation"]
        cases[name] = {
            "delay_gap_bins": delay_gap,
            "doppler_gap_bins": doppler_gap,
            "selected_assignment": validation[
                "detection_oracle_selected_on_training"
            ]["assignment"],
            "uniform_pd": validation["uniform"]["detection_probability"],
            "selected_pd": validation[
                "detection_oracle_selected_on_training"
            ]["detection_probability"],
            "selected_gain_vs_uniform": validation[
                "detection_oracle_selected_on_training"
            ]["pd_difference_vs_uniform"],
            "selected_gain_95ci": validation[
                "detection_oracle_selected_on_training"
            ]["pd_difference_vs_uniform_95ci"],
            "surrogate_gain_vs_uniform": validation[
                "collision_surrogate_oracle"
            ]["pd_difference_vs_uniform"],
            "best_balanced_assignment": validation[
                "best_balanced_selected_on_training"
            ]["assignment"],
            "best_balanced_pd": validation[
                "best_balanced_selected_on_training"
            ]["detection_probability"],
            "selected_gain_vs_best_balanced": validation[
                "detection_oracle_selected_on_training"
            ]["pd_difference_vs_best_balanced"],
            "selected_gain_vs_best_balanced_95ci": validation[
                "detection_oracle_selected_on_training"
            ]["pd_difference_vs_best_balanced_95ci"],
            "best_same_code_assignment": validation[
                "best_same_code_selected_on_training"
            ]["assignment"],
            "best_same_code_pd": validation[
                "best_same_code_selected_on_training"
            ]["detection_probability"],
            "rank_correlation": result[
                "collision_detection_rank_correlation"
            ],
        }
    payload = {"cases": cases}
    output = Path("results/dd_gate1_spacing_sweep.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
