from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def progressive_probe_pair(length, design_length=4):
    """Return normalized prefixes of two columns of a design-length DFT."""
    if not 1 <= length <= design_length:
        raise ValueError("length must lie within the design length")
    indices = np.arange(length)
    first = np.ones(length, dtype=complex) / np.sqrt(length)
    second = np.exp(2j * np.pi * indices / design_length) / np.sqrt(length)
    return [first, second]


def conditional_resolution_probability(gram, noise_variance, trials, seed):
    """Joint-LS resolution on a known two-candidate coarse support."""
    metrics = gram_identifiability_metrics(gram)
    if metrics["lambda_min"] <= 1e-10:
        return 0.0
    inverse = np.linalg.inv(gram)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, (trials, 2))
    amplitudes = np.exp(1j * phases)
    covariance = noise_variance * gram
    factor = np.linalg.cholesky(covariance + 1e-14 * np.eye(2))
    noise = np.sqrt(0.5) * (
        rng.standard_normal((trials, 2))
        + 1j * rng.standard_normal((trials, 2))
    ) @ factor.T
    matched_statistics = amplitudes @ gram.T + noise
    estimates = matched_statistics @ inverse.T
    estimate_variances = noise_variance * np.real(np.diag(inverse))
    normalized_power = np.abs(estimates) ** 2 / estimate_variances[None, :]
    threshold = -np.log(0.01)
    return float(np.mean(np.all(normalized_power >= threshold, axis=1)))


def best_trigger_threshold(values, hard_labels):
    """Select a lambda-min threshold maximizing balanced classification rate."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(hard_labels, dtype=bool)
    candidates = np.unique(np.concatenate(([-np.inf], values, [np.inf])))
    best = None
    for threshold in candidates:
        triggered = values < threshold
        true_positive = np.mean(triggered[labels]) if np.any(labels) else 0.0
        false_positive = np.mean(triggered[~labels]) if np.any(~labels) else 0.0
        balanced = 0.5 * (true_positive + 1.0 - false_positive)
        key = (balanced, true_positive - false_positive, -float(threshold))
        if best is None or key > best[0]:
            best = key, threshold, true_positive, false_positive
    return {
        "threshold": float(best[1]),
        "hard_trigger_rate": float(best[2]),
        "easy_false_trigger_rate": float(best[3]),
        "balanced_accuracy": float(best[0][0]),
    }


def wilson_interval(successes, total, confidence_z=1.96):
    """Wilson score interval for a binomial probability."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes and total must define a nonempty binomial sample")
    probability = successes / total
    denominator = 1.0 + confidence_z ** 2 / total
    center = (probability + confidence_z ** 2 / (2.0 * total)) / denominator
    radius = confidence_z / denominator * np.sqrt(
        probability * (1.0 - probability) / total
        + confidence_z ** 2 / (4.0 * total ** 2)
    )
    return [float(center - radius), float(center + radius)]


def main():
    pattern = qpsk_phase_pattern(8, 16, 11)
    reference = otfs_modulate(pattern)
    angle_gaps = (0.0, 2.5, 5.0, 10.0, 20.0)
    dd_offsets = (0.0, 0.1, 0.25, 0.5)
    rows = []
    seed = 20260822
    for angle_gap in angle_gaps:
        steering = [
            ula_steering_vector(8, -angle_gap / 2.0),
            ula_steering_vector(8, angle_gap / 2.0),
        ]
        for delay_gap in dd_offsets:
            for doppler_gap in dd_offsets:
                waveforms = [
                    delay_doppler_path(reference, 3.2, 1.15, 8),
                    delay_doppler_path(
                        reference, 3.2 + delay_gap,
                        1.15 + doppler_gap, 8,
                    ),
                ]
                for probe_length in (1, 2, 3, 4):
                    probes = progressive_probe_pair(probe_length)
                    gram = factorized_joint_gram(probes, steering, waveforms)
                    metrics = gram_identifiability_metrics(gram)
                    probability = conditional_resolution_probability(
                        gram, noise_variance=0.05, trials=2_000, seed=seed
                    )
                    seed += 1
                    rows.append({
                        "angle_gap_degrees": angle_gap,
                        "delay_gap_bins": delay_gap,
                        "doppler_gap_bins": doppler_gap,
                        "probe_length": probe_length,
                        **metrics,
                        "conditional_resolution_probability": probability,
                    })
    lambda_values = np.array([row["lambda_min"] for row in rows])
    condition_values = np.array([
        min(row["condition_number"], 1e12) for row in rows
    ])
    coherence_values = np.array([
        row["max_effective_coherence"] for row in rows
    ])
    probabilities = np.array([
        row["conditional_resolution_probability"] for row in rows
    ])
    hard = probabilities < 0.8
    trained_trigger = best_trigger_threshold(lambda_values, hard)
    validation_rng = np.random.default_rng(20260824)
    validation_rows = []
    for case in range(1_200):
        angle_gap = validation_rng.uniform(0.0, 20.0)
        delay_gap = validation_rng.uniform(0.0, 0.5)
        doppler_gap = validation_rng.uniform(0.0, 0.5)
        probe_length = int(validation_rng.integers(1, 5))
        probes = progressive_probe_pair(probe_length)
        steering = [
            ula_steering_vector(8, -angle_gap / 2.0),
            ula_steering_vector(8, angle_gap / 2.0),
        ]
        waveforms = [
            delay_doppler_path(reference, 3.2, 1.15, 8),
            delay_doppler_path(
                reference, 3.2 + delay_gap, 1.15 + doppler_gap, 8
            ),
        ]
        gram = factorized_joint_gram(probes, steering, waveforms)
        metrics = gram_identifiability_metrics(gram)
        probability = conditional_resolution_probability(
            gram, noise_variance=0.05, trials=4_000,
            seed=20261000 + case,
        )
        validation_rows.append({
            "lambda_min": metrics["lambda_min"],
            "resolution_probability": probability,
        })
    validation_lambda = np.array([
        row["lambda_min"] for row in validation_rows
    ])
    validation_probability = np.array([
        row["resolution_probability"] for row in validation_rows
    ])
    validation_hard = validation_probability < 0.8
    validation_trigger = validation_lambda < trained_trigger["threshold"]
    validation_hard_rate = float(np.mean(
        validation_trigger[validation_hard]
    )) if np.any(validation_hard) else 0.0
    validation_false_rate = float(np.mean(
        validation_trigger[~validation_hard]
    )) if np.any(~validation_hard) else 0.0
    hard_total = int(np.sum(validation_hard))
    easy_total = int(np.sum(~validation_hard))
    hard_triggered = int(np.sum(validation_trigger & validation_hard))
    easy_triggered = int(np.sum(validation_trigger & ~validation_hard))

    scenario_groups = {}
    for row in rows:
        key = (
            row["angle_gap_degrees"], row["delay_gap_bins"],
            row["doppler_gap_bins"],
        )
        scenario_groups.setdefault(key, []).append(row)
    minimum_lengths = []
    for key, group in scenario_groups.items():
        eligible = [
            row["probe_length"] for row in group
            if row["lambda_min"] >= trained_trigger["threshold"]
        ]
        minimum_lengths.append(min(eligible) if eligible else None)
    minimum_length_counts = {
        str(length): sum(value == length for value in minimum_lengths)
        for length in (1, 2, 3, 4)
    }
    minimum_length_counts["unresolved"] = sum(
        value is None for value in minimum_lengths
    )
    payload = {
        "scope": (
            "known two-candidate coarse support; joint-LS mechanism audit, "
            "not full unknown-parameter detection"
        ),
        "noise_variance": 0.05,
        "per_source_false_alarm_probability": 0.01,
        "rows": rows,
        "spearman": {
            "lambda_min_vs_resolution": float(spearmanr(
                lambda_values, probabilities
            ).statistic),
            "negative_condition_vs_resolution": float(spearmanr(
                -condition_values, probabilities
            ).statistic),
            "negative_coherence_vs_resolution": float(spearmanr(
                -coherence_values, probabilities
            ).statistic),
        },
        "lambda_trigger_for_resolution_below_0.8": trained_trigger,
        "hard_case_fraction": float(np.mean(hard)),
        "independent_continuous_validation": {
            "cases": len(validation_rows),
            "hard_case_fraction": float(np.mean(validation_hard)),
            "lambda_vs_resolution_spearman": float(spearmanr(
                validation_lambda, validation_probability
            ).statistic),
            "hard_trigger_rate": validation_hard_rate,
            "hard_trigger_rate_95ci": wilson_interval(
                hard_triggered, hard_total
            ),
            "easy_false_trigger_rate": validation_false_rate,
            "easy_false_trigger_rate_95ci": wilson_interval(
                easy_triggered, easy_total
            ),
            "hard_cases": hard_total,
            "easy_cases": easy_total,
        },
        "minimum_probe_length_by_training_gram_threshold": {
            "scenario_count": len(minimum_lengths),
            "counts": minimum_length_counts,
        },
    }
    output = Path("results/joint_gram_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
