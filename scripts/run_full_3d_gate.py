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
    cazac_sequence,
    qpsk_phase_pattern,
    spatial_otfs_template,
)
from uav_otfs_isac.spatial_detection import (
    separable_detection_cube,
    spatial_dictionary,
    threshold_nms_3d,
    waveform_dictionary,
)


def calibrate_full_cube_threshold(
    waveform_templates, spatial_template_sets, noise_variance,
    frame_pfa=0.01, trials=20_000, seed=20260816,
):
    """Calibrate the maximum over all searched codes, angles, and DD cells."""
    rng = np.random.default_rng(seed)
    maxima = np.empty(trials)
    antennas = spatial_template_sets[0].shape[1]
    samples = waveform_templates.shape[1]
    for trial in range(trials):
        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal((antennas, samples))
            + 1j * rng.standard_normal((antennas, samples))
        )
        maxima[trial] = max(
            np.max(separable_detection_cube(
                noise, waveform_templates, spatial_templates
            ))
            for spatial_templates in spatial_template_sets
        )
    return float(np.quantile(maxima, 1.0 - frame_pfa, method="higher"))


def cyclic_dd_distance(first, second, shape):
    dk = min((first[0] - second[0]) % shape[0],
             (second[0] - first[0]) % shape[0])
    dl = min((first[1] - second[1]) % shape[1],
             (second[1] - first[1]) % shape[1])
    return dk, dl


def match_full_3d_detections(
    detections, true_angles, true_dd, angle_grid, shape, require_identity,
):
    """Maximum-cardinality minimum-error target/detection assignment."""
    if not detections:
        return 0, 0, set()
    dummy_cost = 1_000.0
    invalid_cost = 1_000_000.0
    cost = np.full((len(true_angles), len(detections) + len(true_angles)), invalid_cost)
    identity = np.zeros((len(true_angles), len(detections)), dtype=bool)
    for target in range(len(true_angles)):
        for detection_index, detection in enumerate(detections):
            code_index, angle_index, k, l = detection
            angle_error = abs(angle_grid[angle_index] - true_angles[target])
            dd_error = cyclic_dd_distance((k, l), true_dd[target], shape)
            if angle_error <= 5.0 and max(dd_error) <= 1:
                cost[target, detection_index] = angle_error + sum(dd_error)
                identity[target, detection_index] = code_index == target
        cost[target, len(detections) + target] = dummy_cost
    targets, assigned = linear_sum_assignment(cost)
    matched = []
    for target, detection_index in zip(targets, assigned):
        if detection_index < len(detections) and cost[target, detection_index] < dummy_cost:
            matched.append((target, detection_index))
    identity_matches = (
        sum(identity[target, detection] for target, detection in matched)
        if require_identity else 0
    )
    return len(matched), identity_matches, {detection for _, detection in matched}


def evaluate_full_search(
    mode, angle_gap, trials=1_000, supplied_codes=None,
    calibration_trials=10_000,
):
    shape = (4, 8)
    antennas = 8
    pattern = qpsk_phase_pattern(*shape, 11)
    waveform_templates = waveform_dictionary(pattern)
    angle_grid = np.arange(-60.0, 60.01, 5.0)
    true_angles = (-angle_gap / 2.0, angle_gap / 2.0)
    delays = (3.20, 3.45)
    dopplers = (1.15, 1.35)
    true_dd = [
        (int(np.rint(doppler)) % shape[0], int(np.rint(delay)) % shape[1])
        for delay, doppler in zip(delays, dopplers)
    ]
    if mode == "array_only":
        codes = [None, None]
        spatial_sets = [spatial_dictionary(angle_grid, antennas)]
    elif mode == "cazac_codes":
        codes = (
            [cazac_sequence(antennas, root) for root in (1, 3)]
            if supplied_codes is None else supplied_codes
        )
        spatial_sets = [
            spatial_dictionary(angle_grid, antennas, code) for code in codes
        ]
    else:
        raise ValueError("unsupported mode")
    noise_variance = 0.02
    threshold = calibrate_full_cube_threshold(
        waveform_templates, spatial_sets, noise_variance,
        trials=calibration_trials,
    )
    h0_rng = np.random.default_rng(20260818)
    h0_alarm = []
    for _ in range(2_000):
        h0_noise = np.sqrt(noise_variance / 2) * (
            h0_rng.standard_normal((antennas, pattern.size))
            + 1j * h0_rng.standard_normal((antennas, pattern.size))
        )
        h0_alarm.append(any(
            np.max(separable_detection_cube(
                h0_noise, waveform_templates, spatial_templates
            )) >= threshold
            for spatial_templates in spatial_sets
        ))
    templates = [
        spatial_otfs_template(
            pattern, delays[target], dopplers[target], true_angles[target],
            antennas, codes[target],
        )
        for target in range(2)
    ]
    rng = np.random.default_rng(20260817)
    phases = rng.uniform(0.0, 2 * np.pi, (trials, 2))
    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal((trials, antennas, pattern.size))
        + 1j * rng.standard_normal((trials, antennas, pattern.size))
    )
    joint_hits = []
    identity_hits = []
    extra_candidate_peaks = []
    for trial in range(trials):
        received = (
            np.exp(1j * phases[trial, 0]) * templates[0]
            + np.exp(1j * phases[trial, 1]) * templates[1]
            + noise[trial]
        )
        detections = []
        for code_index, spatial_templates in enumerate(spatial_sets):
            cube = separable_detection_cube(
                received, waveform_templates, spatial_templates
            )
            peaks = threshold_nms_3d(cube, threshold, shape, 1, 1)
            detections.extend((code_index,) + peak for peak in peaks)
        matched, identity_matched, matched_detection_indices = (
            match_full_3d_detections(
                detections, true_angles, true_dd, angle_grid, shape,
                require_identity=mode != "array_only",
            )
        )
        joint_hits.append(matched == 2)
        identity_hits.append(identity_matched == 2)
        extra_candidate_peaks.append(
            len(detections) - len(matched_detection_indices)
        )
    return {
        "mode": mode,
        "angle_gap_degrees": angle_gap,
        "true_dd_bins": [list(value) for value in true_dd],
        "threshold": threshold,
        "frame_false_alarm_probability": 0.01,
        "empirical_h0_frame_false_alarm_probability": float(np.mean(h0_alarm)),
        "joint_position_resolution_probability": float(np.mean(joint_hits)),
        "joint_identity_correct_probability": (
            float(np.mean(identity_hits)) if mode != "array_only" else None
        ),
        "extra_candidate_peaks_per_h1_frame": float(
            np.mean(extra_candidate_peaks)
        ),
        "trials": trials,
    }


def main():
    payload = {
        "scope": "unknown angle and full integer DD search; unknown target count",
        "grid": [4, 8],
        "angle_grid_step_degrees": 5.0,
        "cases": {
            str(gap): {
                mode: evaluate_full_search(mode, gap)
                for mode in ("array_only", "cazac_codes")
            }
            for gap in (5.0, 10.0, 20.0)
        },
    }
    output = Path("results/full_3d_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
