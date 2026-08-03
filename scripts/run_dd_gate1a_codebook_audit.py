from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dd_gate1_oracle_audit import evaluate_scenario
from uav_otfs_isac.dd_patterns import (
    full_grid_ambiguity_metrics,
    select_balanced_ambiguity_codebook,
)
from uav_otfs_isac.otfs_physical import (
    qpsk_phase_pattern,
    separable_cazac_pattern,
)


def rank_correlation(first, second):
    first_ranks = np.argsort(np.argsort(np.asarray(first)))
    second_ranks = np.argsort(np.argsort(np.asarray(second)))
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def compact_validation(result):
    validation = result["independent_validation"]
    return {
        "uniform": validation["uniform"],
        "best_balanced_selected_on_training": validation[
            "best_balanced_selected_on_training"
        ],
        "detection_oracle_selected_on_training": validation[
            "detection_oracle_selected_on_training"
        ],
        "rank_correlation": result["collision_detection_rank_correlation"],
    }


def main():
    n_doppler, n_delay = 8, 16
    original_seeds = (11, 29, 47)
    candidate_seeds = tuple(range(100, 118))
    original = [
        qpsk_phase_pattern(n_doppler, n_delay, seed)
        for seed in original_seeds
    ]
    candidates = [
        qpsk_phase_pattern(n_doppler, n_delay, seed)
        for seed in candidate_seeds
    ]
    selected_indices, selected_metrics = select_balanced_ambiguity_codebook(
        candidates, 3
    )
    selected_seeds = tuple(candidate_seeds[index] for index in selected_indices)
    selected = [candidates[index] for index in selected_indices]
    cazac_roots = ((1, 1), (3, 5), (5, 7))
    cazac = [
        separable_cazac_pattern(n_doppler, n_delay, doppler_root, delay_root)
        for doppler_root, delay_root in cazac_roots
    ]
    delays = np.array([4.20, 4.70, 8.10, 11.30])
    dopplers = np.array([2.15, 2.45, 5.42, 6.25])
    original_result = evaluate_scenario(
        delays, dopplers, trials=120, validation_trials=2_000,
        patterns=original,
    )
    selected_result = evaluate_scenario(
        delays, dopplers, trials=120, validation_trials=2_000,
        patterns=selected,
    )
    cazac_result = evaluate_scenario(
        delays, dopplers, trials=120, validation_trials=2_000,
        patterns=cazac,
    )
    single_code_audit = []
    for seed, pattern in zip(candidate_seeds, candidates):
        metrics = full_grid_ambiguity_metrics([pattern])
        result = evaluate_scenario(
            delays, dopplers, trials=40, validation_trials=400,
            patterns=[pattern, pattern, pattern],
        )
        validation = result["independent_validation"]["same"]
        single_code_audit.append({
            "seed": seed,
            "auto_peak_sidelobe": metrics["worst_auto_peak_sidelobe"],
            "same_code_pd": validation["detection_probability"],
        })
    auto_values = [row["auto_peak_sidelobe"] for row in single_code_audit]
    pd_values = [row["same_code_pd"] for row in single_code_audit]
    original_validation = original_result["independent_validation"]["uniform"]
    selected_validation = selected_result["independent_validation"]["uniform"]
    payload = {
        "candidate_count": len(candidates),
        "original_seeds": list(original_seeds),
        "selected_seeds": list(selected_seeds),
        "original_ambiguity_metrics": full_grid_ambiguity_metrics(original),
        "selected_ambiguity_metrics": selected_metrics,
        "fixed_balanced_assignment": [0, 1, 2, 0],
        "original_fixed_balanced_pd": original_validation[
            "detection_probability"
        ],
        "selected_fixed_balanced_pd": selected_validation[
            "detection_probability"
        ],
        "fixed_balanced_pd_difference": (
            selected_validation["detection_probability"]
            - original_validation["detection_probability"]
        ),
        "original": compact_validation(original_result),
        "selected": compact_validation(selected_result),
        "cazac_roots": [list(value) for value in cazac_roots],
        "cazac_ambiguity_metrics": full_grid_ambiguity_metrics(cazac),
        "cazac": compact_validation(cazac_result),
        "single_code_audit": single_code_audit,
        "auto_sidelobe_vs_same_code_pd_rank_correlation": rank_correlation(
            auto_values, [-value for value in pd_values]
        ),
    }
    output = Path("results/dd_gate1a_codebook_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
