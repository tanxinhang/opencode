from __future__ import annotations

from itertools import product
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.dd_patterns import assignment_cost, geometry_collision_tensor
from uav_otfs_isac.otfs_physical import (
    delay_doppler_path,
    otfs_modulate,
    qpsk_phase_pattern,
    threshold_cyclic_nms_peaks,
)


def cyclic_distance(first, second, shape):
    dk = min((first[0] - second[0]) % shape[0],
             (second[0] - first[0]) % shape[0])
    dl = min((first[1] - second[1]) % shape[1],
             (second[1] - first[1]) % shape[1])
    return int(dk), int(dl)


def match_detections(truth, detections, shape, tolerance=1):
    """Maximum-cardinality, minimum-error cyclic one-to-one matching."""
    if not truth:
        return 0, [], len(detections)
    dummy_cost = 1000.0
    invalid_cost = 1_000_000.0
    cost = np.full((len(truth), len(detections) + len(truth)), invalid_cost)
    errors_by_pair = {}
    for ti, target in enumerate(truth):
        for di, detection in enumerate(detections):
            error = cyclic_distance(target, detection, shape)
            if max(error) <= tolerance:
                cost[ti, di] = error[0] ** 2 + error[1] ** 2
                errors_by_pair[ti, di] = error
        cost[ti, len(detections) + ti] = dummy_cost
    truth_indices, detection_indices = linear_sum_assignment(cost)
    errors = []
    matched_detections = set()
    for ti, di in zip(truth_indices, detection_indices):
        if di < len(detections) and cost[ti, di] < dummy_cost:
            errors.append(errors_by_pair[ti, di])
            matched_detections.add(di)
    return len(errors), errors, len(detections) - len(matched_detections)


def calibrate_frame_thresholds(
    dictionaries, noise_variance, frame_false_alarm_probability,
    trials=20_000, batch_size=1_000,
):
    """Empirically calibrate max-map thresholds for a fixed frame P_FA."""
    rng = np.random.default_rng(20260810)
    num_codes = dictionaries.shape[0]
    maxima = np.empty((trials, num_codes), dtype=float)
    offset = 0
    while offset < trials:
        count = min(batch_size, trials - offset)
        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal((count, dictionaries.shape[-1]))
            + 1j * rng.standard_normal((count, dictionaries.shape[-1]))
        )
        maps = np.abs(np.einsum(
            "cmf,tf->tcm", dictionaries.conj(), noise
        )) ** 2
        maxima[offset:offset + count] = np.max(maps, axis=2)
        offset += count
    thresholds = {}
    for active_count in range(1, num_codes + 1):
        for active_codes in combinations(range(num_codes), active_count):
            frame_maximum = np.max(maxima[:, active_codes], axis=1)
            thresholds[active_codes] = float(np.quantile(
                frame_maximum, 1.0 - frame_false_alarm_probability,
                method="higher",
            ))
    return thresholds


def evaluate_scenario(
    delays, dopplers, trials=120, validation_trials=2_000, gains=None,
    patterns=None,
):
    n_doppler, n_delay = 8, 16
    shape = (n_doppler, n_delay)
    patterns = (
        [qpsk_phase_pattern(n_doppler, n_delay, seed)
         for seed in (11, 29, 47)]
        if patterns is None else [np.asarray(pattern, dtype=complex)
                                  for pattern in patterns]
    )
    if len(patterns) != 3 or any(pattern.shape != shape for pattern in patterns):
        raise ValueError("patterns must contain three grids of the scenario shape")
    gains = (
        np.array([1.0, 0.85, 0.75, 0.65])
        if gains is None else np.asarray(gains, dtype=float)
    )
    if gains.shape != (4,) or np.any(gains <= 0.0):
        raise ValueError("gains must contain four positive amplitudes")
    noise_variance = 0.02
    frame_pfa = 0.01
    true_bins = [
        (int(np.rint(doppler)) % n_doppler,
         int(np.rint(delay)) % n_delay)
        for delay, doppler in zip(delays, dopplers)
    ]
    path_templates = np.empty((4, 3, n_doppler * n_delay), complex)
    dictionaries = np.empty((3, n_doppler * n_delay,
                             n_doppler * n_delay), complex)
    for code, pattern in enumerate(patterns):
        reference = otfs_modulate(pattern)
        for k in range(n_doppler):
            for l in range(n_delay):
                dictionaries[code, k * n_delay + l] = delay_doppler_path(
                    reference, l, k, n_doppler
                )
        for uav in range(4):
            path_templates[uav, code] = delay_doppler_path(
                reference, delays[uav], dopplers[uav], n_doppler
            )
    thresholds = calibrate_frame_thresholds(
        dictionaries, noise_variance, frame_pfa
    )
    rng = np.random.default_rng(20260809)
    phases = rng.uniform(0, 2 * np.pi, (trials, 4))
    trial_gains = gains[None, :] * np.exp(1j * phases)
    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal((trials, n_doppler * n_delay))
        + 1j * rng.standard_normal((trials, n_doppler * n_delay))
    )
    collisions = geometry_collision_tensor(patterns, delays, dopplers, gains)
    weights = np.ones((4, 4)) - np.eye(4)
    def evaluate_assignment(assignment, gains_by_trial, noise_by_trial):
        active_codes = sorted(set(assignment))
        threshold = thresholds[tuple(active_codes)]
        selected_templates = path_templates[
            np.arange(4), np.asarray(assignment)
        ]
        signal = np.einsum(
            "tu,uf->tf", gains_by_trial, selected_templates
        ) + noise_by_trial
        maps = np.abs(np.einsum(
            "cmf,tf->tcm", dictionaries.conj(), signal
        )) ** 2
        hits = false_alarms = 0
        doppler_squared_error = 0.0
        trial_hit_rates = []
        for trial in range(gains_by_trial.shape[0]):
            trial_hits = 0
            for code in active_codes:
                truth = [true_bins[uav] for uav in range(4)
                         if assignment[uav] == code]
                detected = threshold_cyclic_nms_peaks(
                    maps[trial, code].reshape(shape), threshold, 1
                )
                matched, errors, false = match_detections(
                    truth, detected, shape, 1
                )
                hits += matched
                trial_hits += matched
                false_alarms += false
                doppler_squared_error += sum(error[0] ** 2 for error in errors)
            trial_hit_rates.append(trial_hits / 4.0)
        total_targets = gains_by_trial.shape[0] * 4
        hit_rate = hits / total_targets
        return {
            "assignment": list(assignment),
            "collision_cost": assignment_cost(assignment, weights, collisions),
            "detection_probability": hit_rate,
            "miss_or_merge_rate": 1.0 - hit_rate,
            "false_peaks_per_frame": false_alarms / gains_by_trial.shape[0],
            "doppler_rmse_on_matches": (
                float(np.sqrt(doppler_squared_error / hits)) if hits else None
            ),
            "trial_hit_rates": trial_hit_rates,
        }

    rows = [
        evaluate_assignment(assignment, trial_gains, noise)
        for assignment in product(range(3), repeat=4)
    ]
    by_detection = sorted(
        rows, key=lambda row: (-row["detection_probability"],
                               row["false_peaks_per_frame"], row["assignment"])
    )
    by_collision = sorted(rows, key=lambda row: (row["collision_cost"],
                                                  row["assignment"]))
    same_code_candidates = [
        row for row in by_detection if len(set(row["assignment"])) == 1
    ]
    balanced_candidates = [
        row for row in by_detection
        if sorted(np.bincount(row["assignment"], minlength=3)) == [1, 1, 2]
    ]
    costs = np.array([row["collision_cost"] for row in rows])
    detection = np.array([row["detection_probability"] for row in rows])
    cost_ranks = np.argsort(np.argsort(costs))
    detection_ranks = np.argsort(np.argsort(-detection))
    rank_correlation = float(np.corrcoef(cost_ranks, detection_ranks)[0, 1])
    lookup = {tuple(row["assignment"]): row for row in rows}
    validation_rng = np.random.default_rng(20260811)
    validation_phases = validation_rng.uniform(
        0, 2 * np.pi, (validation_trials, 4)
    )
    validation_gains = gains[None, :] * np.exp(1j * validation_phases)
    validation_noise = np.sqrt(noise_variance / 2) * (
        validation_rng.standard_normal((validation_trials, n_doppler * n_delay))
        + 1j * validation_rng.standard_normal(
            (validation_trials, n_doppler * n_delay)
        )
    )
    selected_assignments = {
        "same": (0, 0, 0, 0),
        "uniform": (0, 1, 2, 0),
        "detection_oracle_selected_on_training": tuple(
            by_detection[0]["assignment"]
        ),
        "best_same_code_selected_on_training": tuple(
            same_code_candidates[0]["assignment"]
        ),
        "best_balanced_selected_on_training": tuple(
            balanced_candidates[0]["assignment"]
        ),
        "collision_surrogate_oracle": tuple(by_collision[0]["assignment"]),
    }
    validation = {
        name: evaluate_assignment(
            assignment, validation_gains, validation_noise
        )
        for name, assignment in selected_assignments.items()
    }
    baseline_rates = np.asarray(validation["uniform"]["trial_hit_rates"])
    balanced_rates = np.asarray(
        validation["best_balanced_selected_on_training"]["trial_hit_rates"]
    )
    for result in validation.values():
        rates = np.asarray(result.pop("trial_hit_rates"))
        difference = rates - baseline_rates
        standard_error = np.std(difference, ddof=1) / np.sqrt(validation_trials)
        mean_difference = float(np.mean(difference))
        result["pd_difference_vs_uniform"] = mean_difference
        result["pd_difference_vs_uniform_95ci"] = [
            mean_difference - 1.96 * standard_error,
            mean_difference + 1.96 * standard_error,
        ]
        balanced_difference = rates - balanced_rates
        balanced_standard_error = (
            np.std(balanced_difference, ddof=1) / np.sqrt(validation_trials)
        )
        balanced_mean = float(np.mean(balanced_difference))
        result["pd_difference_vs_best_balanced"] = balanced_mean
        result["pd_difference_vs_best_balanced_95ci"] = [
            balanced_mean - 1.96 * balanced_standard_error,
            balanced_mean + 1.96 * balanced_standard_error,
        ]
    for row in rows:
        row.pop("trial_hit_rates")
    return {
        "true_bins": [list(value) for value in true_bins],
        "trials": trials,
        "noise_variance": noise_variance,
        "frame_false_alarm_probability": frame_pfa,
        "thresholds_by_active_code_set": {
            ",".join(map(str, key)): value for key, value in thresholds.items()
        },
        "same": lookup[(0, 0, 0, 0)],
        "uniform": lookup[(0, 1, 2, 0)],
        "detection_oracle": by_detection[0],
        "three_code_balanced_oracle": balanced_candidates[0],
        "collision_surrogate_oracle": by_collision[0],
        "collision_detection_rank_correlation": rank_correlation,
        "top_detection_assignments": by_detection[:5],
        "independent_validation_trials": validation_trials,
        "independent_validation": validation,
    }


def main():
    scenarios = {
        "separated": (
            np.array([1.20, 4.35, 7.10, 10.30]),
            np.array([0.35, 2.18, 4.42, 6.25]),
        ),
        "close_collision": (
            np.array([4.20, 4.65, 8.10, 11.30]),
            np.array([2.15, 2.45, 5.42, 6.25]),
        ),
    }
    payload = {
        name: evaluate_scenario(delays, dopplers)
        for name, (delays, dopplers) in scenarios.items()
    }
    output = Path("results/dd_gate1_oracle_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
