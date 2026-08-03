from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.dd_patterns import (
    assignment_cost,
    exhaustive_pattern_assignment,
    geometry_collision_tensor,
    greedy_pattern_assignment,
    pattern_collision_matrix,
)
from uav_otfs_isac.otfs_physical import (
    cyclic_nms_peaks,
    matched_filter_map,
    qpsk_phase_pattern,
    superpose_uav_echoes,
)


def cyclic_peak_errors(true_bins, detected_bins, shape):
    """Minimum-total-error one-to-one matching of detected and true peaks."""
    if len(true_bins) != len(detected_bins):
        raise ValueError("true and detected peak counts must match")
    best = None
    for candidate_order in permutations(detected_bins):
        errors = []
        for truth, detected in zip(true_bins, candidate_order):
            dk = min(
                (detected[0] - truth[0]) % shape[0],
                (truth[0] - detected[0]) % shape[0],
            )
            dl = min(
                (detected[1] - truth[1]) % shape[1],
                (truth[1] - detected[1]) % shape[1],
            )
            errors.append((dk, dl))
        score = sum(dk ** 2 + dl ** 2 for dk, dl in errors)
        candidate = (score, errors)
        if best is None or candidate < best:
            best = candidate
    return best[1]


def main() -> None:
    n_doppler, n_delay = 8, 16
    # Three patterns for four UAVs forces at least one reuse collision.
    patterns = [
        qpsk_phase_pattern(n_doppler, n_delay, seed)
        for seed in (11, 29, 47)
    ]
    offsets = np.linspace(-0.45, 0.45, 7)
    collisions = pattern_collision_matrix(patterns, offsets, offsets)
    pair_weights = np.ones((4, 4)) - np.eye(4)
    delays = np.array([1.20, 4.35, 7.10, 10.30])
    dopplers = np.array([0.35, 2.18, 4.42, 6.25])
    gains = np.array([1.0, 0.85, 0.75, 0.65])
    geometry_collisions = geometry_collision_tensor(
        patterns, delays, dopplers, gains
    )
    assignments = {
        "same": (0, 0, 0, 0),
        "uniform": (0, 1, 2, 0),
        "random": tuple(int(x) for x in np.random.default_rng(20260803).integers(
            0, len(patterns), size=4
        )),
        "greedy": greedy_pattern_assignment(pair_weights, geometry_collisions),
        "oracle": exhaustive_pattern_assignment(pair_weights, geometry_collisions),
    }
    costs = {
        name: assignment_cost(value, pair_weights, geometry_collisions)
        for name, value in assignments.items()
    }

    true_bins = [
        (int(np.rint(doppler)) % n_doppler, int(np.rint(delay)) % n_delay)
        for delay, doppler in zip(delays, dopplers)
    ]
    noise_variance = 0.02
    trials = 200
    rng = np.random.default_rng(20260804)
    trial_phases = rng.uniform(0.0, 2.0 * np.pi, size=(trials, len(gains)))
    trial_noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal((trials, n_doppler * n_delay))
        + 1j * rng.standard_normal((trials, n_doppler * n_delay))
    )
    detection = {}
    for name, assignment in assignments.items():
        selected = [patterns[index] for index in assignment]
        true_to_false = []
        localization_hits = []
        squared_bin_errors = []
        trial_hit_rates = []
        for trial in range(trials):
            trial_hits = []
            trial_gains = gains * np.exp(1j * trial_phases[trial])
            paths = [
                [(trial_gains[i], delays[i], dopplers[i])]
                for i in range(len(gains))
            ]
            received = superpose_uav_echoes(
                selected, paths, 0.0, np.random.default_rng(0)
            )
            received += trial_noise[trial]
            maps = [matched_filter_map(received, pattern) for pattern in patterns]
            detected_by_pattern = {
                pattern_index: cyclic_nms_peaks(
                    maps[pattern_index], assignment.count(pattern_index), 1
                )
                for pattern_index in set(assignment)
            }
            errors_by_target = {}
            for pattern_index in set(assignment):
                target_indices = [
                    index for index, value in enumerate(assignment)
                    if value == pattern_index
                ]
                matched_errors = cyclic_peak_errors(
                    [true_bins[index] for index in target_indices],
                    detected_by_pattern[pattern_index],
                    (n_doppler, n_delay),
                )
                errors_by_target.update(zip(target_indices, matched_errors))
            for target, (pattern_index, index) in enumerate(zip(assignment, true_bins)):
                grid = maps[pattern_index]
                k, l = index
                false_map = grid.copy()
                neighborhood = []
                for dk in (-1, 0, 1):
                    for dl in (-1, 0, 1):
                        neighborhood.append(
                            grid[(k + dk) % n_doppler, (l + dl) % n_delay]
                        )
                        false_map[(k + dk) % n_doppler, (l + dl) % n_delay] = 0.0
                true_to_false.append(float(
                    np.max(neighborhood) / max(np.max(false_map), 1e-15)
                ))
                doppler_error, delay_error = errors_by_target[target]
                hit = doppler_error <= 1 and delay_error <= 1
                localization_hits.append(hit)
                trial_hits.append(hit)
                squared_bin_errors.append(doppler_error ** 2 + delay_error ** 2)
            trial_hit_rates.append(float(np.mean(trial_hits)))
        hit_rate = float(np.mean(localization_hits))
        hit_standard_error = float(
            np.std(trial_hit_rates, ddof=1) / np.sqrt(trials)
        )
        detection[name] = {
            "mean_true_to_strongest_false_ratio": float(np.mean(true_to_false)),
            "localization_hit_rate": hit_rate,
            "localization_hit_rate_95ci": [
                max(0.0, hit_rate - 1.96 * hit_standard_error),
                min(1.0, hit_rate + 1.96 * hit_standard_error),
            ],
            "localization_bin_rmse": float(np.sqrt(np.mean(squared_bin_errors))),
        }
    payload = {
        "grid": [n_doppler, n_delay],
        "fractional_offset_support": [-0.45, 0.45],
        "monte_carlo_trials": trials,
        "noise_variance": noise_variance,
        "detector_assumption": "known targets per assigned code; cyclic NMS",
        "max_same_pattern_collision": float(np.max(np.diag(collisions))),
        "max_distinct_pattern_collision": float(np.max(
            collisions[~np.eye(len(patterns), dtype=bool)]
        )),
        "assignments": {name: list(value) for name, value in assignments.items()},
        "weighted_collision_cost": costs,
        "waveform_detection_map": detection,
    }
    output = Path("results/dd_collision_gate0_smoke.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
