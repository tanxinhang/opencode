from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_joint_gram_gate import (
    conditional_resolution_probability,
    progressive_probe_pair,
)
from uav_otfs_isac.identifiability import (
    factorized_joint_gram,
    gram_identifiability_metrics,
)
from uav_otfs_isac.otfs_physical import (
    delay_doppler_path,
    otfs_modulate,
    qpsk_phase_pattern,
    ula_steering_vector,
)


def mismatched_probe_pair(length, phase_step, cfo, phase_noise_std, rng):
    """DFT-prefix probes with relative phase evolution and residual noise."""
    indices = np.arange(length)
    first = np.ones(length, dtype=complex) / np.sqrt(length)
    nominal_second = np.exp(2j * np.pi * indices / 4) / np.sqrt(length)
    residual_phase = (
        phase_step * indices + 2.0 * np.pi * cfo * indices
        + rng.normal(0.0, phase_noise_std, length)
    )
    actual_second = nominal_second * np.exp(1j * residual_phase)
    return [first, actual_second]


def main():
    pattern = qpsk_phase_pattern(8, 16, 11)
    reference = otfs_modulate(pattern)
    steering = [
        ula_steering_vector(8, -2.5), ula_steering_vector(8, 2.5)
    ]
    waveforms = [
        delay_doppler_path(reference, 3.2, 1.15, 8),
        delay_doppler_path(reference, 3.3, 1.25, 8),
    ]
    rows = []
    rng = np.random.default_rng(20260825)
    seed = 20262000
    for phase_step in (0.0, 0.05 * np.pi, 0.1 * np.pi, 0.25 * np.pi, 0.5 * np.pi):
        for cfo in (0.0, 0.01, 0.05, 0.1):
            for phase_noise_std in (0.0, 0.05, 0.15):
                for length in (1, 2, 3, 4):
                    probes = mismatched_probe_pair(
                        length, phase_step, cfo, phase_noise_std, rng
                    )
                    actual_gram = factorized_joint_gram(
                        probes, steering, waveforms
                    )
                    nominal_gram = factorized_joint_gram(
                        progressive_probe_pair(length), steering, waveforms
                    )
                    metrics = gram_identifiability_metrics(actual_gram)
                    nominal_metrics = gram_identifiability_metrics(nominal_gram)
                    probability = conditional_resolution_probability(
                        actual_gram, 0.05, 4_000, seed
                    )
                    seed += 1
                    rows.append({
                        "phase_step_radians": phase_step,
                        "normalized_cfo": cfo,
                        "phase_noise_std": phase_noise_std,
                        "probe_length": length,
                        **metrics,
                        "nominal_lambda_min": nominal_metrics["lambda_min"],
                        "conditional_resolution_probability": probability,
                    })
    lambda_values = [row["lambda_min"] for row in rows]
    nominal_lambda_values = [row["nominal_lambda_min"] for row in rows]
    probabilities = [row["conditional_resolution_probability"] for row in rows]
    groups = {}
    for row in rows:
        key = (
            row["phase_step_radians"], row["normalized_cfo"],
            row["phase_noise_std"],
        )
        groups.setdefault(key, []).append(row)
    nonmonotone_groups = 0
    best_length_counts = {str(length): 0 for length in (1, 2, 3, 4)}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["probe_length"])
        values = [row["conditional_resolution_probability"] for row in ordered]
        if any(first > second + 0.02 for first, second in zip(values, values[1:])):
            nonmonotone_groups += 1
        best = max(ordered, key=lambda row: (
            row["conditional_resolution_probability"], -row["probe_length"]
        ))
        best_length_counts[str(best["probe_length"])] += 1
    payload = {
        "scope": "actual-signature mismatch mechanism audit on one hard DD-angle cluster",
        "rows": rows,
        "lambda_vs_resolution_spearman": float(spearmanr(
            lambda_values, probabilities
        ).statistic),
        "nominal_lambda_vs_resolution_spearman": float(spearmanr(
            nominal_lambda_values, probabilities
        ).statistic),
        "mismatch_group_count": len(groups),
        "nonmonotone_resolution_groups": nonmonotone_groups,
        "best_probe_length_counts": best_length_counts,
    }
    output = Path("results/probe_mismatch_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
