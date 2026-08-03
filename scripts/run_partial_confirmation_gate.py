from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_end_to_end_adaptive_probe_gate import (
    ANTENNAS,
    NOISE_VARIANCE,
    SHAPE,
    complex_noise,
    draw_truth,
    fine_local_dictionary,
    supports_match,
)
from uav_otfs_isac.otfs_physical import (
    qpsk_phase_pattern,
    spatial_otfs_template,
)


def decoded_sum_difference(plus, partial_minus, energy_fraction):
    """Decode two sources from a full sum and partial-energy difference probe."""
    if not np.isfinite(energy_fraction) or not 0.0 < energy_fraction <= 1.0:
        raise ValueError("energy_fraction must lie in (0, 1]")
    scaled_difference = np.asarray(partial_minus) / np.sqrt(energy_fraction)
    plus = np.asarray(plus)
    if plus.shape != scaled_difference.shape:
        raise ValueError("sum and difference observations must have equal shape")
    return (0.5 * (plus + scaled_difference),
            0.5 * (plus - scaled_difference))


def detect_decoded_pair(decoded, dictionary, parameters, threshold):
    estimates = []
    peak_ratios = []
    detected = []
    for observation in decoded:
        powers = np.abs(dictionary.conj().T @ observation.reshape(-1)) ** 2
        index = int(np.argmax(powers))
        estimates.append(parameters[index])
        peak_ratios.append(float(powers[index] / threshold))
        detected.append(bool(powers[index] >= threshold))
    return np.asarray(estimates), min(peak_ratios), all(detected)


def calibrate_decoded_threshold(dictionary, energy_fraction, trials=10_000,
                                seed=20260901):
    """Calibrate maximum over both decoded channels at frame PFA 1%."""
    rng = np.random.default_rng(seed + int(round(100 * energy_fraction)))
    maxima = []
    feature_count = dictionary.shape[0]
    for offset in range(0, trials, 100):
        count = min(100, trials - offset)
        plus_noise = complex_noise(rng, (count, feature_count))
        minus_noise = complex_noise(rng, (count, feature_count))
        first, second = decoded_sum_difference(
            plus_noise, minus_noise, energy_fraction
        )
        powers = np.abs(
            np.stack((first, second), axis=1) @ dictionary.conj()
        ) ** 2
        maxima.extend(np.max(powers, axis=(1, 2)))
    return float(np.quantile(maxima, 0.99, method="higher"))


def wilson_interval(successes, total, z=1.96):
    probability = successes / total
    denominator = 1.0 + z ** 2 / total
    center = (probability + z ** 2 / (2.0 * total)) / denominator
    radius = z / denominator * np.sqrt(
        probability * (1.0 - probability) / total
        + z ** 2 / (4.0 * total ** 2)
    )
    return [float(center - radius), float(center + radius)]


def combine_incremental_difference(partial, supplemental, energy_fraction):
    """Combine independent partial and supplemental energy observations."""
    if not 0.0 < energy_fraction < 1.0:
        raise ValueError("energy_fraction must lie strictly between zero and one")
    return (
        np.sqrt(energy_fraction) * np.asarray(partial)
        + np.sqrt(1.0 - energy_fraction) * np.asarray(supplemental)
    )


def evaluate_three_stage_policy(dictionary, parameters, pattern,
                                partial_threshold, full_threshold,
                                energy_fraction=0.3, scenarios=1_500,
                                fallback_quantile=0.4, seed=20260921):
    """Train an unlabeled confidence quantile and validate incremental fallback."""
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in range(scenarios):
        stratum = ("easy", "medium", "hard")[scenario % 3]
        truth = draw_truth(rng, stratum)
        templates = [
            spatial_otfs_template(pattern, delay, doppler, angle, ANTENNAS)
            for angle, delay, doppler in truth
        ]
        phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, 2))
        signals = [phases[index] * templates[index] for index in range(2)]
        plus = signals[0] + signals[1] + complex_noise(
            rng, (ANTENNAS, pattern.size)
        )
        difference = signals[0] - signals[1]
        partial = np.sqrt(energy_fraction) * difference + complex_noise(
            rng, (ANTENNAS, pattern.size)
        )
        supplemental = np.sqrt(1.0 - energy_fraction) * difference + complex_noise(
            rng, (ANTENNAS, pattern.size)
        )
        partial_decoded = decoded_sum_difference(plus, partial, energy_fraction)
        partial_estimates, confidence, partial_detected = detect_decoded_pair(
            partial_decoded, dictionary, parameters, partial_threshold
        )
        full_difference = combine_incremental_difference(
            partial, supplemental, energy_fraction
        )
        full_decoded = decoded_sum_difference(plus, full_difference, 1.0)
        full_estimates, _, full_detected = detect_decoded_pair(
            full_decoded, dictionary, parameters, full_threshold
        )
        rows.append({
            "scenario": scenario,
            "stratum": stratum,
            "confidence": confidence,
            "partial_success": bool(
                partial_detected and supports_match(partial_estimates, truth)
            ),
            "full_success": bool(
                full_detected and supports_match(full_estimates, truth)
            ),
        })
    training = [row for row in rows if row["scenario"] % 2 == 0]
    validation = [row for row in rows if row["scenario"] % 2 == 1]
    confidence_threshold = float(np.quantile(
        [row["confidence"] for row in training], fallback_quantile,
        method="higher",
    ))
    for row in validation:
        row["fallback"] = row["confidence"] < confidence_threshold
        row["success"] = (
            row["full_success"] if row["fallback"] else row["partial_success"]
        )
        row["normalized_probe_energy"] = (
            2.0 if row["fallback"] else 1.0 + energy_fraction
        )
    successes = sum(row["success"] for row in validation)
    full_successes = sum(row["full_success"] for row in validation)
    mean_energy = float(np.mean([
        row["normalized_probe_energy"] for row in validation
    ]))
    return {
        "energy_fraction": energy_fraction,
        "fallback_quantile_selected_without_labels": fallback_quantile,
        "confidence_threshold_from_training_half": confidence_threshold,
        "validation_scenarios": len(validation),
        "fallback_rate": float(np.mean([
            row["fallback"] for row in validation
        ])),
        "mean_normalized_probe_energy": mean_energy,
        "energy_saving_vs_full_two_snapshot_probe": 1.0 - mean_energy / 2.0,
        "joint_detection_probability": successes / len(validation),
        "joint_detection_95ci": wilson_interval(successes, len(validation)),
        "fixed_full_energy_probability": full_successes / len(validation),
        "detection_loss_vs_fixed_full_energy": (
            full_successes - successes
        ) / len(validation),
        "passes_joint_gate": bool(
            1.0 - mean_energy / 2.0 >= 0.20
            and (full_successes - successes) / len(validation) <= 0.01
        ),
    }


def audit_sequential_false_alarm(dictionary, parameters, pattern,
                                 partial_threshold, full_threshold,
                                 confidence_threshold, energy_fraction=0.3,
                                 trials=10_000, seed=20260925):
    """Replay the data-dependent stopping rule under noise-only frames."""
    rng = np.random.default_rng(seed)
    alarms = 0
    fallbacks = 0
    for _ in range(trials):
        plus = complex_noise(rng, (ANTENNAS, pattern.size))
        partial = complex_noise(rng, (ANTENNAS, pattern.size))
        supplemental = complex_noise(rng, (ANTENNAS, pattern.size))
        partial_decoded = decoded_sum_difference(
            plus, partial, energy_fraction
        )
        _, confidence, detected = detect_decoded_pair(
            partial_decoded, dictionary, parameters, partial_threshold
        )
        fallback = confidence < confidence_threshold
        if fallback:
            fallbacks += 1
            full_difference = combine_incremental_difference(
                partial, supplemental, energy_fraction
            )
            _, _, detected = detect_decoded_pair(
                decoded_sum_difference(plus, full_difference, 1.0),
                dictionary, parameters, full_threshold,
            )
        alarms += int(detected)
    return {
        "trials": trials,
        "alarms": alarms,
        "empirical_joint_false_alarm_probability": alarms / trials,
        "fallback_rate_under_h0": fallbacks / trials,
        "joint_false_alarm_95ci": wilson_interval(alarms, trials),
    }


def evaluate_fraction(energy_fraction, dictionary, parameters, pattern,
                      threshold, scenarios=1_500, seed=20260911):
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in range(scenarios):
        stratum = ("easy", "medium", "hard")[scenario % 3]
        truth = draw_truth(rng, stratum)
        templates = [
            spatial_otfs_template(pattern, delay, doppler, angle, ANTENNAS)
            for angle, delay, doppler in truth
        ]
        phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, 2))
        signals = [phases[index] * templates[index] for index in range(2)]
        plus = signals[0] + signals[1] + complex_noise(
            rng, (ANTENNAS, pattern.size)
        )
        partial_minus = np.sqrt(energy_fraction) * (
            signals[0] - signals[1]
        ) + complex_noise(rng, (ANTENNAS, pattern.size))
        decoded = decoded_sum_difference(plus, partial_minus, energy_fraction)
        estimates, minimum_peak_ratio, detected = detect_decoded_pair(
            decoded, dictionary, parameters, threshold
        )
        success = detected and supports_match(estimates, truth)
        rows.append({
            "scenario": scenario,
            "stratum": stratum,
            "success": success,
            "minimum_peak_to_threshold_ratio": minimum_peak_ratio,
        })
    successes = sum(row["success"] for row in rows)
    probability = successes / scenarios
    return {
        "energy_fraction": energy_fraction,
        "normalized_probe_energy": 1.0 + energy_fraction,
        "energy_saving_vs_full_two_snapshot_probe": (
            1.0 - energy_fraction
        ) / 2.0,
        "joint_detection_probability": probability,
        "joint_detection_95ci": wilson_interval(successes, scenarios),
        "by_stratum": {
            stratum: float(np.mean([
                row["success"] for row in rows if row["stratum"] == stratum
            ]))
            for stratum in ("easy", "medium", "hard")
        },
        "rows": rows,
    }


def main():
    pattern = qpsk_phase_pattern(*SHAPE, 11)
    dictionary, parameters = fine_local_dictionary(pattern)
    fractions = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
    results = {}
    for fraction in fractions:
        threshold = calibrate_decoded_threshold(dictionary, fraction)
        result = evaluate_fraction(
            fraction, dictionary, parameters, pattern, threshold
        )
        result["frame_pfa_threshold"] = threshold
        results[str(fraction)] = result
    fixed = results["1.0"]["joint_detection_probability"]
    for result in results.values():
        result["detection_loss_vs_fixed_p2"] = (
            fixed - result["joint_detection_probability"]
        )
        result["passes_joint_gate"] = bool(
            result["energy_saving_vs_full_two_snapshot_probe"] >= 0.20
            and result["detection_loss_vs_fixed_p2"] <= 0.01
        )
    three_stage = evaluate_three_stage_policy(
        dictionary, parameters, pattern,
        results["0.3"]["frame_pfa_threshold"],
        results["1.0"]["frame_pfa_threshold"],
    )
    sequential_h0 = audit_sequential_false_alarm(
        dictionary, parameters, pattern,
        results["0.3"]["frame_pfa_threshold"],
        results["1.0"]["frame_pfa_threshold"],
        three_stage["confidence_threshold_from_training_half"],
    )
    payload = {
        "scope": (
            "independent validation of a full-energy sum snapshot followed by "
            "a partial-energy sign-reversed confirmation snapshot"
        ),
        "noise_model": (
            "confirmation is divided by sqrt(delta) during decoding, so its "
            "noise amplification is included"
        ),
        "familywise_false_alarm_upper_bound": 0.01,
        "scenarios_per_fraction": 1_500,
        "fractions": results,
        "three_stage_incremental_policy": three_stage,
        "sequential_h0_audit": sequential_h0,
        "recommended_fraction": min(
            (value for value in results.values() if value["passes_joint_gate"]),
            key=lambda value: value["normalized_probe_energy"],
            default=None,
        ),
    }
    output = Path("results/partial_confirmation_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    printable = {
        **{key: value for key, value in payload.items() if key != "fractions"},
        "fractions": {
            key: {field: value[field] for field in (
                "normalized_probe_energy",
                "energy_saving_vs_full_two_snapshot_probe",
                "joint_detection_probability", "joint_detection_95ci",
                "detection_loss_vs_fixed_p2", "passes_joint_gate",
            )}
            for key, value in results.items()
        },
    }
    if printable["recommended_fraction"] is not None:
        printable["recommended_fraction"].pop("rows", None)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
