from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_joint_gram_gate import (
    conditional_resolution_probability,
    progressive_probe_pair,
)
from scripts.run_probe_mismatch_gate import mismatched_probe_pair
from uav_otfs_isac.identifiability import (
    factorized_joint_gram,
    gram_identifiability_metrics,
    minimum_probe_length,
)
from uav_otfs_isac.otfs_physical import (
    delay_doppler_path,
    otfs_modulate,
    qpsk_phase_pattern,
    ula_steering_vector,
)


LAMBDA_THRESHOLD = 0.26409570321722925


def factor_gram(
    reference, angle_gap, delay_gap, doppler_gap, probe_length,
    phase_step=0.0, cfo=0.0,
):
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
    probes = mismatched_probe_pair(
        probe_length, phase_step, cfo, 0.0, np.random.default_rng(0)
    )
    return factorized_joint_gram(probes, steering, waveforms)


def uncertainty_grams_by_length(reference, estimate, radii):
    """Corner-sample a deterministic box uncertainty set for every length."""
    values = {}
    signs = (-1.0, 1.0)
    for length in (1, 2, 3, 4):
        grams = []
        for angle_sign in signs:
            for delay_sign in signs:
                for doppler_sign in signs:
                    for phase_sign in signs:
                        for cfo_sign in signs:
                            grams.append(factor_gram(
                                reference,
                                max(0.0, estimate["angle_gap"]
                                    + angle_sign * radii["angle"]),
                                max(0.0, estimate["delay_gap"]
                                    + delay_sign * radii["delay"]),
                                max(0.0, estimate["doppler_gap"]
                                    + doppler_sign * radii["doppler"]),
                                length,
                                estimate["phase_step"]
                                + phase_sign * radii["phase"],
                                max(0.0, estimate["cfo"]
                                    + cfo_sign * radii["cfo"]),
                            ))
        values[length] = grams
    return values


def nominal_minimum_length(reference, estimate):
    for length in (1, 2, 3, 4):
        gram = factor_gram(
            reference, estimate["angle_gap"], estimate["delay_gap"],
            estimate["doppler_gap"], length,
            estimate["phase_step"], estimate["cfo"],
        )
        if gram_identifiability_metrics(gram)["lambda_min"] >= LAMBDA_THRESHOLD:
            return length
    return 4


def summarize_policy(rows, policy):
    probabilities = np.array([
        row["probability_by_length"][str(row[policy])] for row in rows
    ])
    lengths = np.array([row[policy] for row in rows])
    return {
        "mean_resolution_probability": float(np.mean(probabilities)),
        "scenario_reliability_rate_at_0.8": float(np.mean(probabilities >= 0.8)),
        "mean_probe_length": float(np.mean(lengths)),
        "probe_length_counts": {
            str(length): int(np.sum(lengths == length))
            for length in (1, 2, 3, 4)
        },
        "by_stratum": {
            stratum: {
                "scenarios": len(selected),
                "mean_resolution_probability": float(np.mean([
                    row["probability_by_length"][str(row[policy])]
                    for row in selected
                ])),
                "reliability_rate_at_0.8": float(np.mean([
                    row["probability_by_length"][str(row[policy])] >= 0.8
                    for row in selected
                ])),
                "mean_probe_length": float(np.mean([
                    row[policy] for row in selected
                ])),
            }
            for stratum in sorted({row["stratum"] for row in rows})
            for selected in [[row for row in rows if row["stratum"] == stratum]]
        },
    }


def paired_policy_comparison(rows, first_policy, second_policy):
    """Compare first minus second with paired normal confidence intervals."""
    first_probability = np.array([
        row["probability_by_length"][str(row[first_policy])] for row in rows
    ])
    second_probability = np.array([
        row["probability_by_length"][str(row[second_policy])] for row in rows
    ])
    probability_difference = first_probability - second_probability
    length_difference = np.array([
        row[first_policy] - row[second_policy] for row in rows
    ], dtype=float)

    def summary(values):
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        return {
            "mean": mean,
            "95ci": [mean - 1.96 * standard_error,
                      mean + 1.96 * standard_error],
        }

    return {
        "resolution_probability_difference": summary(probability_difference),
        "probe_length_difference": summary(length_difference),
        "first_has_lower_resolution_scenarios": int(np.sum(
            probability_difference < -0.01
        )),
        "first_fails_0.8_while_second_passes": int(np.sum(
            (first_probability < 0.8) & (second_probability >= 0.8)
        )),
    }


def main():
    pattern = qpsk_phase_pattern(8, 16, 11)
    reference = otfs_modulate(pattern)
    rng = np.random.default_rng(20260826)
    radii = {
        "angle": 1.5,
        "delay": 0.075,
        "doppler": 0.075,
        "phase": 0.1 * np.pi,
        "cfo": 0.025,
    }
    rows = []
    for scenario in range(180):
        stratum = ("easy", "medium", "hard")[scenario % 3]
        if stratum == "easy":
            angle_range, dd_range = (10.0, 20.0), (0.2, 0.5)
        elif stratum == "medium":
            angle_range, dd_range = (4.0, 10.0), (0.1, 0.3)
        else:
            angle_range, dd_range = (0.0, 6.0), (0.0, 0.15)
        estimate = {
            "angle_gap": rng.uniform(*angle_range),
            "delay_gap": rng.uniform(*dd_range),
            "doppler_gap": rng.uniform(*dd_range),
            "phase_step": rng.uniform(0.0, 0.5 * np.pi),
            "cfo": rng.uniform(0.0, 0.1),
        }
        uncertainty = uncertainty_grams_by_length(reference, estimate, radii)
        robust_length, _ = minimum_probe_length(
            uncertainty, LAMBDA_THRESHOLD
        )
        robust_length = 4 if robust_length is None else robust_length
        nominal_length = nominal_minimum_length(reference, estimate)
        actual = {
            key: max(0.0, estimate[key] + rng.uniform(-radii[error], radii[error]))
            for key, error in (
                ("angle_gap", "angle"), ("delay_gap", "delay"),
                ("doppler_gap", "doppler"), ("phase_step", "phase"),
                ("cfo", "cfo"),
            )
        }
        probabilities = {}
        for length in (1, 2, 3, 4):
            gram = factor_gram(
                reference, actual["angle_gap"], actual["delay_gap"],
                actual["doppler_gap"], length,
                actual["phase_step"], actual["cfo"],
            )
            probabilities[str(length)] = conditional_resolution_probability(
                gram, 0.05, 4_000, 20263000 + 10 * scenario + length
            )
        rows.append({
            "scenario": scenario,
            "stratum": stratum,
            "estimate": estimate,
            "actual": actual,
            "nominal_length": nominal_length,
            "robust_length": robust_length,
            "fixed_2": 2,
            "fixed_4": 4,
            "probability_by_length": probabilities,
        })
    payload = {
        "scope": (
            "two-source known-coarse-support policy audit; 32 sampled box "
            "corners per length are used for selection, and the actual "
            "realization is sampled independently"
        ),
        "lambda_threshold": LAMBDA_THRESHOLD,
        "uncertainty_radii": radii,
        "scenarios": len(rows),
        "policies": {
            policy: summarize_policy(rows, policy)
            for policy in ("nominal_length", "robust_length", "fixed_2", "fixed_4")
        },
        "paired_comparisons": {
            "robust_minus_fixed_2": paired_policy_comparison(
                rows, "robust_length", "fixed_2"
            ),
            "robust_minus_nominal": paired_policy_comparison(
                rows, "robust_length", "nominal_length"
            ),
        },
        "rows": rows,
    }
    output = Path("results/robust_minimal_probe_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
