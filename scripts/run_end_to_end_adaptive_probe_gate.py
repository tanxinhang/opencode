from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.otfs_physical import (
    qpsk_phase_pattern,
    spatial_otfs_template,
)


SHAPE = (4, 8)
ANTENNAS = 8
NOISE_VARIANCE = 0.02


def fine_local_dictionary(pattern):
    """Build a unit-energy off-grid refinement dictionary for one coarse cluster."""
    parameters = []
    columns = []
    for angle in np.arange(-20.0, 20.01, 2.5):
        for delay in np.arange(2.9, 3.76, 0.1):
            for doppler in np.arange(0.85, 1.76, 0.1):
                column = spatial_otfs_template(
                    pattern, delay, doppler, angle, ANTENNAS
                ).reshape(-1)
                columns.append(column / np.linalg.norm(column))
                parameters.append((angle, delay, doppler))
    return np.asarray(columns).T, np.asarray(parameters)


def two_source_residual_statistic(observation, dictionary, parameters):
    """Return the best separated two-atom joint-LS gain and its support."""
    vector = np.asarray(observation, dtype=complex).reshape(-1)
    correlations = dictionary.conj().T @ vector
    powers = np.abs(correlations) ** 2
    first = int(np.argmax(powers))
    energy = float(np.vdot(vector, vector).real)
    one_source_residual = energy - float(powers[first])
    shortlist_size = min(64, dictionary.shape[1])
    shortlist = np.argpartition(powers, -shortlist_size)[-shortlist_size:]
    best_residual = np.inf
    second_best_residual = np.inf
    best_support = None
    best_coefficients = None
    best_coherence = None
    scales = np.array([5.0, 0.2, 0.2])
    for position, first_index in enumerate(shortlist):
        for second_index in shortlist[position + 1:]:
            distance = np.max(np.abs(
                (parameters[first_index] - parameters[second_index]) / scales
            ))
            if distance < 1.0:
                continue
            coherence = np.vdot(
                dictionary[:, first_index], dictionary[:, second_index]
            )
            gram = np.array([[1.0, coherence], [coherence.conjugate(), 1.0]])
            if 1.0 - abs(coherence) ** 2 <= 1e-8:
                continue
            selected_correlations = correlations[[first_index, second_index]]
            coefficients = np.linalg.solve(gram, selected_correlations)
            residual = energy - float(np.real(np.vdot(
                selected_correlations, coefficients
            )))
            if residual < best_residual:
                second_best_residual = best_residual
                best_residual = residual
                best_support = (int(first_index), int(second_index))
                best_coefficients = coefficients
                best_coherence = coherence
            elif residual < second_best_residual:
                second_best_residual = residual
    if best_support is None:
        return 0.0, (first, first), {
            "fit_margin": 0.0, "lambda_min": 0.0,
            "minimum_coefficient_power": 0.0,
        }
    gain = max(0.0, (
        one_source_residual - best_residual
    ) / max(one_source_residual, 1e-15))
    diagnostics = {
        "fit_margin": max(0.0, (
            second_best_residual - best_residual
        ) / max(one_source_residual, 1e-15)),
        "lambda_min": float(1.0 - abs(best_coherence)),
        "minimum_coefficient_power": float(np.min(
            np.abs(best_coefficients) ** 2
        )),
    }
    return gain, best_support, diagnostics


def supports_match(estimated, truth, angle_tolerance=5.0,
                   delay_tolerance=0.2, doppler_tolerance=0.2):
    """One-to-one matching of two continuous angle-delay-Doppler supports."""
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimated.shape != (2, 3) or truth.shape != (2, 3):
        raise ValueError("estimated and truth must both have shape (2, 3)")
    scales = np.array([angle_tolerance, delay_tolerance, doppler_tolerance])
    errors = np.max(np.abs(
        estimated[:, None, :] - truth[None, :, :]
    ) / scales, axis=2)
    rows, columns = linear_sum_assignment(errors)
    return bool(np.all(errors[rows, columns] <= 1.0))


def single_atom_detect(observation, dictionary, parameters, threshold):
    vector = np.asarray(observation, dtype=complex).reshape(-1)
    powers = np.abs(dictionary.conj().T @ vector) ** 2
    index = int(np.argmax(powers))
    return float(powers[index]) >= threshold, parameters[index]


def complex_noise(rng, shape):
    return np.sqrt(NOISE_VARIANCE / 2.0) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


def calibrate_h0_threshold(dictionary, trials=5_000, seed=20260831):
    """Calibrate the maximum over two decoded channels at frame PFA 1%."""
    rng = np.random.default_rng(seed)
    maxima = []
    for offset in range(0, trials, 100):
        count = min(100, trials - offset)
        noise = complex_noise(rng, (count, 2, dictionary.shape[0]))
        powers = np.abs(noise @ dictionary.conj()) ** 2
        maxima.extend(np.max(powers, axis=(1, 2)))
    return float(np.quantile(maxima, 0.99, method="higher"))


def draw_truth(rng, stratum):
    ranges = {
        "easy": ((10.0, 20.0), (0.2, 0.35)),
        "medium": ((4.0, 10.0), (0.1, 0.3)),
        "hard": ((0.0, 6.0), (0.0, 0.15)),
    }
    angle_range, dd_range = ranges[stratum]
    gap = rng.uniform(*angle_range)
    dd_gap = rng.uniform(*dd_range)
    center_angle = rng.uniform(-5.0, 5.0)
    delay = rng.uniform(3.0, 3.4)
    doppler = rng.uniform(0.95, 1.35)
    return np.asarray([
        (center_angle - gap / 2.0, delay, doppler),
        (center_angle + gap / 2.0, delay + dd_gap, doppler + dd_gap),
    ])


def calibrate_collision_threshold(pattern, dictionary, parameters,
                                  trials=1_000, seed=20260832):
    """Set a 1% false collision trigger under off-grid single-source H1."""
    rng = np.random.default_rng(seed)
    statistics = []
    for _ in range(trials):
        truth = (
            rng.uniform(-5.0, 5.0), rng.uniform(3.0, 3.4),
            rng.uniform(0.95, 1.35),
        )
        observation = (
            np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
            * spatial_otfs_template(pattern, truth[1], truth[2], truth[0], ANTENNAS)
            + complex_noise(rng, (ANTENNAS, pattern.size))
        )
        statistic, _, _ = two_source_residual_statistic(
            observation, dictionary, parameters
        )
        statistics.append(statistic)
    return float(np.quantile(statistics, 0.99, method="higher")), statistics


def main():
    pattern = qpsk_phase_pattern(*SHAPE, 11)
    dictionary, parameters = fine_local_dictionary(pattern)
    h0_threshold = calibrate_h0_threshold(dictionary)
    collision_threshold, null_statistics = calibrate_collision_threshold(
        pattern, dictionary, parameters
    )
    rng = np.random.default_rng(20260833)
    rows = []
    for scenario in range(600):
        stratum = ("easy", "medium", "hard")[scenario % 3]
        truth = draw_truth(rng, stratum)
        templates = [
            spatial_otfs_template(pattern, delay, doppler, angle, ANTENNAS)
            for angle, delay, doppler in truth
        ]
        phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, 2))
        first_observation = (
            phases[0] * templates[0] + phases[1] * templates[1]
            + complex_noise(rng, (ANTENNAS, pattern.size))
        )
        statistic, support_indices, diagnostics = two_source_residual_statistic(
            first_observation, dictionary, parameters
        )
        stop_after_one = statistic >= collision_threshold
        one_snapshot_success = supports_match(
            parameters[list(support_indices)], truth
        )

        # The two-snapshot benchmark uses orthogonal DFT probe signatures.  After
        # matched decoding each source has one unit-energy observation and AWGN.
        decoded_success = []
        for target in range(2):
            decoded = phases[target] * templates[target] + complex_noise(
                rng, (ANTENNAS, pattern.size)
            )
            detected, estimate = single_atom_detect(
                decoded, dictionary, parameters, h0_threshold
            )
            error = np.abs(estimate - truth[target])
            decoded_success.append(bool(
                detected and error[0] <= 5.0
                and error[1] <= 0.2 and error[2] <= 0.2
            ))
        two_snapshot_success = all(decoded_success)
        adaptive_success = (
            one_snapshot_success if stop_after_one else two_snapshot_success
        )
        rows.append({
            "scenario": scenario,
            "stratum": stratum,
            "collision_statistic": statistic,
            **diagnostics,
            "stop_after_one": stop_after_one,
            "one_snapshot_success": one_snapshot_success,
            "two_snapshot_success": two_snapshot_success,
            "adaptive_success": adaptive_success,
            "probe_length": 1 if stop_after_one else 2,
        })

    def summarize(selected):
        return {
            "scenarios": len(selected),
            "early_stop_rate": float(np.mean([
                row["stop_after_one"] for row in selected
            ])),
            "wrong_early_stop_rate": float(np.mean([
                row["stop_after_one"] and not row["one_snapshot_success"]
                for row in selected
            ])),
            "adaptive_joint_detection_probability": float(np.mean([
                row["adaptive_success"] for row in selected
            ])),
            "fixed_p2_joint_detection_probability": float(np.mean([
                row["two_snapshot_success"] for row in selected
            ])),
            "mean_probe_length": float(np.mean([
                row["probe_length"] for row in selected
            ])),
        }

    overall = summarize(rows)
    fixed_probability = overall["fixed_p2_joint_detection_probability"]
    adaptive_probability = overall["adaptive_joint_detection_probability"]
    statistics = np.asarray([row["collision_statistic"] for row in rows])
    one_success = np.asarray([row["one_snapshot_success"] for row in rows])
    two_success = np.asarray([row["two_snapshot_success"] for row in rows])
    threshold_sweep = []
    for quantile in np.linspace(0.0, 1.0, 101):
        threshold = float(np.quantile(statistics, quantile, method="higher"))
        early = statistics >= threshold
        probability = float(np.mean(np.where(early, one_success, two_success)))
        threshold_sweep.append({
            "quantile": float(quantile),
            "threshold": threshold,
            "early_stop_rate": float(np.mean(early)),
            "detection_loss_vs_fixed_p2": fixed_probability - probability,
        })
    resource_eligible = [
        row for row in threshold_sweep if row["early_stop_rate"] >= 0.40
    ]
    loss_eligible = [
        row for row in threshold_sweep
        if row["detection_loss_vs_fixed_p2"] <= 0.01 + 1e-12
    ]
    payload = {
        "scope": (
            "local fine-grid two-source collision discovery followed by "
            "orthogonal two-snapshot probing; the coarse cluster is supplied"
        ),
        "frame_false_alarm_probability": 0.01,
        "collision_false_trigger_probability": float(np.mean(
            np.asarray(null_statistics) >= collision_threshold
        )),
        "h0_threshold": h0_threshold,
        "collision_threshold": collision_threshold,
        "overall": overall,
        "by_stratum": {
            stratum: summarize([
                row for row in rows if row["stratum"] == stratum
            ])
            for stratum in ("easy", "medium", "hard")
        },
        "gate": {
            "resource_saving_vs_fixed_p2": 1.0 - overall["mean_probe_length"] / 2.0,
            "detection_loss_vs_fixed_p2": fixed_probability - adaptive_probability,
            "passes_resource_saving_20_percent": (
                1.0 - overall["mean_probe_length"] / 2.0 >= 0.20
            ),
            "passes_detection_loss_1pp": (
                fixed_probability - adaptive_probability <= 0.01
            ),
        },
        "label_aided_raw_statistic_reachability_audit": {
            "warning": (
                "This sweep uses evaluation outcomes and is only an upper-bound "
                "diagnostic; it is not a deployable threshold-selection method."
            ),
            "minimum_detection_loss_at_40_percent_early_stop": float(min(
                row["detection_loss_vs_fixed_p2"] for row in resource_eligible
            )),
            "maximum_early_stop_at_1pp_detection_loss": float(max(
                row["early_stop_rate"] for row in loss_eligible
            )),
            "has_jointly_feasible_threshold": any(
                row["early_stop_rate"] >= 0.40
                and row["detection_loss_vs_fixed_p2"] <= 0.01 + 1e-12
                for row in threshold_sweep
            ),
            "threshold_sweep": threshold_sweep,
        },
        "rows": rows,
    }
    output = Path("results/end_to_end_adaptive_probe_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
