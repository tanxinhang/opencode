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
    SHAPE,
    complex_noise,
    draw_truth,
    fine_local_dictionary,
    supports_match,
)
from scripts.run_partial_confirmation_gate import (
    combine_incremental_difference,
    decoded_sum_difference,
    wilson_interval,
)
from uav_otfs_isac.otfs_physical import (
    qpsk_phase_pattern,
    spatial_otfs_template,
)


PARTIAL_ENERGY = 0.3
TARGET_FALLBACK_RATE = 0.404
EQUAL_ENERGY = 1.0 + PARTIAL_ENERGY + (
    1.0 - PARTIAL_ENERGY
) * TARGET_FALLBACK_RATE


def decoded_score_and_support(decoded, dictionary, parameters,
                              return_diagnostics=False):
    """Return minimum identity-channel peak power and two support estimates."""
    scores = []
    margins = []
    residual_ratios = []
    peak_indices = []
    estimates = []
    for observation in decoded:
        powers = np.abs(dictionary.conj().T @ observation.reshape(-1)) ** 2
        index = int(np.argmax(powers))
        top = np.partition(powers, -2)[-2:]
        scores.append(float(powers[index]))
        margins.append(float(top[-1] / max(top[-2], 1e-15)))
        residual = observation.reshape(-1) - dictionary[:, index] * (
            np.vdot(dictionary[:, index], observation.reshape(-1))
        )
        residual_ratios.append(float(
            np.vdot(residual, residual).real
            / max(np.vdot(observation, observation).real, 1e-15)
        ))
        peak_indices.append(index)
        if parameters is not None:
            estimates.append(parameters[index])
    support = None if parameters is None else np.asarray(estimates)
    if not return_diagnostics:
        return min(scores), support
    coherence = float(abs(np.vdot(
        dictionary[:, peak_indices[0]], dictionary[:, peak_indices[1]]
    )))
    return min(scores), support, {
        "minimum_peak_margin": min(margins),
        "maximum_residual_ratio": max(residual_ratios),
        "estimated_support_lambda_min": 1.0 - coherence,
    }


def combine_energy_blocks(blocks, energies):
    """Coherently combine independent observations with listed signal energies."""
    energies = np.asarray(energies, dtype=float)
    if len(blocks) != energies.size or np.any(energies <= 0.0):
        raise ValueError("blocks and positive energies must have matching length")
    total = float(np.sum(energies))
    return sum(
        np.sqrt(energy) * np.asarray(block)
        for block, energy in zip(blocks, energies)
    ) / np.sqrt(total)


def h0_policy_scores(dictionary, confidence_cutoff, fallback_rate,
                     trials=10_000, seed=20260931):
    """Generate final detector scores for every complete policy under H0."""
    rng = np.random.default_rng(seed)
    scores = {name: [] for name in (
        "fixed_partial", "fixed_equal_energy", "random_fallback",
        "confidence_fallback", "full_energy",
    )}
    equal_extra = EQUAL_ENERGY - 1.0 - PARTIAL_ENERGY
    final_extra = 1.0 - PARTIAL_ENERGY - equal_extra
    features = ANTENNAS * SHAPE[0] * SHAPE[1]
    for offset in range(0, trials, 100):
        count = min(100, trials - offset)
        plus = complex_noise(rng, (count, features))
        first = complex_noise(rng, plus.shape)
        second = complex_noise(rng, plus.shape)
        third = complex_noise(rng, plus.shape)
        equal = combine_energy_blocks(
            (first, second), (PARTIAL_ENERGY, equal_extra)
        )
        full = combine_energy_blocks(
            (first, second, third),
            (PARTIAL_ENERGY, equal_extra, final_extra),
        )

        def batch_scores(difference, energy):
            scaled = difference / np.sqrt(energy)
            decoded = np.stack((
                0.5 * (plus + scaled), 0.5 * (plus - scaled)
            ), axis=1)
            powers = np.abs(decoded @ dictionary.conj()) ** 2
            return np.min(np.max(powers, axis=2), axis=1)

        partial_score = batch_scores(first, PARTIAL_ENERGY)
        equal_score = batch_scores(equal, EQUAL_ENERGY - 1.0)
        full_score = batch_scores(full, 1.0)
        random_fallback = rng.random(count) < fallback_rate
        scores["fixed_partial"].extend(partial_score)
        scores["fixed_equal_energy"].extend(equal_score)
        scores["confidence_fallback"].extend(np.where(
            partial_score < confidence_cutoff, full_score, partial_score
        ))
        scores["random_fallback"].extend(np.where(
            random_fallback, full_score, partial_score
        ))
        scores["full_energy"].extend(full_score)
    return {name: np.asarray(values) for name, values in scores.items()}


def calibrate_policy_thresholds(scores, false_alarm_probability=0.01):
    """Calibrate each final policy score to the same empirical H0 tail."""
    return {
        name: float(np.quantile(
            values, 1.0 - false_alarm_probability, method="higher"
        ))
        for name, values in scores.items()
    }


def paired_difference(first, second):
    values = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return {
        "mean": mean,
        "95ci": [mean - 1.96 * standard_error,
                  mean + 1.96 * standard_error],
    }


def main():
    pattern = qpsk_phase_pattern(*SHAPE, 11)
    dictionary, parameters = fine_local_dictionary(pattern)
    rng = np.random.default_rng(20260932)
    cases = []
    equal_extra = EQUAL_ENERGY - 1.0 - PARTIAL_ENERGY
    final_extra = 1.0 - PARTIAL_ENERGY - equal_extra
    for scenario in range(3_000):
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
        blocks = [
            np.sqrt(energy) * difference + complex_noise(
                rng, (ANTENNAS, pattern.size)
            )
            for energy in (PARTIAL_ENERGY, equal_extra, final_extra)
        ]
        combined = {
            "partial": blocks[0],
            "equal": combine_energy_blocks(
                blocks[:2], (PARTIAL_ENERGY, equal_extra)
            ),
            "full": combine_energy_blocks(
                blocks, (PARTIAL_ENERGY, equal_extra, final_extra)
            ),
        }
        stages = {}
        for name, difference_observation, energy in (
            ("partial", combined["partial"], PARTIAL_ENERGY),
            ("equal", combined["equal"], EQUAL_ENERGY - 1.0),
            ("full", combined["full"], 1.0),
        ):
            score, estimate, diagnostics = decoded_score_and_support(
                decoded_sum_difference(plus, difference_observation, energy),
                dictionary, parameters, return_diagnostics=True,
            )
            stages[name] = {
                "score": score,
                "support_correct": supports_match(estimate, truth),
                **diagnostics,
            }
        cases.append({
            "scenario": scenario,
            "stratum": stratum,
            "stages": stages,
        })

    training = [case for case in cases if case["scenario"] % 2 == 0]
    validation = [case for case in cases if case["scenario"] % 2 == 1]
    confidence_cutoff = float(np.quantile(
        [case["stages"]["partial"]["score"] for case in training],
        TARGET_FALLBACK_RATE, method="higher",
    ))
    fallback_count = int(round(TARGET_FALLBACK_RATE * len(validation)))
    random_rng = np.random.default_rng(20260933)
    random_indices = set(random_rng.choice(
        len(validation), fallback_count, replace=False
    ).tolist())
    partial_failures = [
        index for index, case in enumerate(validation)
        if not case["stages"]["partial"]["support_correct"]
        and case["stages"]["full"]["support_correct"]
    ]
    oracle_indices = set(partial_failures[:fallback_count])
    if len(oracle_indices) < fallback_count:
        oracle_indices.update(
            index for index in range(len(validation))
            if index not in oracle_indices
            and len(oracle_indices) < fallback_count
        )

    h0_scores = h0_policy_scores(
        dictionary, confidence_cutoff, TARGET_FALLBACK_RATE
    )
    thresholds = calibrate_policy_thresholds(h0_scores)
    outcomes = {name: [] for name in (
        "fixed_partial", "fixed_equal_energy", "random_fallback",
        "confidence_fallback", "oracle_fallback", "full_energy",
    )}
    confidence_fallbacks = []
    for index, case in enumerate(validation):
        stages = case["stages"]
        confidence_fallback = stages["partial"]["score"] < confidence_cutoff
        confidence_fallbacks.append(confidence_fallback)
        choices = {
            "fixed_partial": "partial",
            "fixed_equal_energy": "equal",
            "random_fallback": "full" if index in random_indices else "partial",
            "confidence_fallback": "full" if confidence_fallback else "partial",
            "oracle_fallback": "full" if index in oracle_indices else "partial",
            "full_energy": "full",
        }
        for policy, stage in choices.items():
            threshold_policy = (
                "random_fallback" if policy == "oracle_fallback" else policy
            )
            outcomes[policy].append(bool(
                stages[stage]["score"] >= thresholds[threshold_policy]
                and stages[stage]["support_correct"]
            ))

    confidence_rate = float(np.mean(confidence_fallbacks))
    energies = {
        "fixed_partial": 1.0 + PARTIAL_ENERGY,
        "fixed_equal_energy": EQUAL_ENERGY,
        "random_fallback": EQUAL_ENERGY,
        "confidence_fallback": 1.0 + PARTIAL_ENERGY
        + (1.0 - PARTIAL_ENERGY) * confidence_rate,
        "oracle_fallback": EQUAL_ENERGY,
        "full_energy": 2.0,
    }
    policies = {}
    for name, values in outcomes.items():
        successes = int(np.sum(values))
        h0_name = "random_fallback" if name == "oracle_fallback" else name
        empirical_pfa = float(np.mean(h0_scores[h0_name] >= thresholds[h0_name]))
        policies[name] = {
            "mean_normalized_probe_energy": energies[name],
            "conditional_exact_support_probability": successes / len(values),
            "conditional_exact_support_95ci": wilson_interval(
                successes, len(values)
            ),
            "calibrated_policy_threshold": thresholds[h0_name],
            "empirical_policy_false_alarm_probability": empirical_pfa,
        }
    payload = {
        "scope": (
            "paired two-source coarse-cluster fairness audit under ideal "
            "coherent incremental snapshots"
        ),
        "policy_false_alarm_target": 0.01,
        "h0_calibration_trials": 10_000,
        "h1_training_scenarios": len(training),
        "h1_validation_scenarios": len(validation),
        "partial_energy": PARTIAL_ENERGY,
        "target_fallback_rate": TARGET_FALLBACK_RATE,
        "confidence_cutoff_from_unlabeled_training_scores": confidence_cutoff,
        "validation_confidence_fallback_rate": confidence_rate,
        "policies": policies,
        "paired_advantages": {
            "confidence_minus_random": paired_difference(
                outcomes["confidence_fallback"], outcomes["random_fallback"]
            ),
            "confidence_minus_fixed_equal_energy": paired_difference(
                outcomes["confidence_fallback"], outcomes["fixed_equal_energy"]
            ),
            "oracle_minus_confidence": paired_difference(
                outcomes["oracle_fallback"], outcomes["confidence_fallback"]
            ),
        },
        "gate": {
            "confidence_beats_fixed_equal_energy": bool(
                np.mean(outcomes["confidence_fallback"])
                > np.mean(outcomes["fixed_equal_energy"])
            ),
            "confidence_significantly_beats_random": bool(
                paired_difference(
                    outcomes["confidence_fallback"],
                    outcomes["random_fallback"],
                )["95ci"][0] > 0.0
            ),
        },
        "partial_stage_diagnostic_separation": {
            feature: {
                "correct_median": float(np.median([
                    case["stages"]["partial"][feature]
                    for case in validation
                    if case["stages"]["partial"]["support_correct"]
                ])),
                "incorrect_median": float(np.median([
                    case["stages"]["partial"][feature]
                    for case in validation
                    if not case["stages"]["partial"]["support_correct"]
                ])),
            }
            for feature in (
                "score", "minimum_peak_margin", "maximum_residual_ratio",
                "estimated_support_lambda_min",
            )
        },
    }
    output = Path("results/confirmation_fairness_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
