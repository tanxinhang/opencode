from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.otfs_physical import (
    cazac_sequence,
    qpsk_phase_pattern,
    spatial_otfs_template,
    ula_steering_vector,
)


def spatial_dictionary(angle_grid, num_antennas, code=None):
    vectors = []
    for angle in angle_grid:
        vector = ula_steering_vector(num_antennas, angle)
        if code is not None:
            vector = vector * code
            vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    return np.asarray(vectors)


def calibrate_threshold(dictionaries, noise_variance, frame_pfa, trials=20_000):
    rng = np.random.default_rng(20260814)
    maxima = []
    for offset in range(0, trials, 1_000):
        count = min(1_000, trials - offset)
        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal((count, dictionaries.shape[-1]))
            + 1j * rng.standard_normal((count, dictionaries.shape[-1]))
        )
        maps = np.abs(np.einsum("caf,tf->tca", dictionaries.conj(), noise)) ** 2
        maxima.append(np.max(maps, axis=(1, 2)))
    return float(np.quantile(
        np.concatenate(maxima), 1.0 - frame_pfa, method="higher"
    ))


def evaluate_mode(codes, angle_gap, trials=2_000):
    num_antennas = 8
    noise_variance = 0.02
    frame_pfa = 0.01
    angle_grid = np.arange(-60.0, 60.01, 0.5)
    true_angles = (-angle_gap / 2.0, angle_gap / 2.0)
    pattern = qpsk_phase_pattern(8, 16, 11)
    if codes is None:
        dictionaries = spatial_dictionary(
            angle_grid, num_antennas
        )[None, :, :]
        target_codes = [None, None]
    else:
        dictionaries = np.asarray([
            spatial_dictionary(angle_grid, num_antennas, code)
            for code in codes
        ])
        target_codes = codes
    threshold = calibrate_threshold(
        dictionaries, noise_variance, frame_pfa
    )
    spatial_signatures = []
    for target, angle in enumerate(true_angles):
        template = spatial_otfs_template(
            pattern, 4.0, 2.0, angle, num_antennas,
            target_codes[target],
        )
        spatial_signatures.append(template[:, 0] / np.linalg.norm(template[:, 0]))
    rng = np.random.default_rng(20260815)
    phases = rng.uniform(0.0, 2.0 * np.pi, (trials, 2))
    gains = np.exp(1j * phases)
    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal((trials, num_antennas))
        + 1j * rng.standard_normal((trials, num_antennas))
    )
    observations = np.einsum(
        "tu,uf->tf", gains, np.asarray(spatial_signatures)
    ) + noise
    maps = np.abs(np.einsum(
        "caf,tf->tca", dictionaries.conj(), observations
    )) ** 2
    resolved = []
    for trial in range(trials):
        if codes is not None:
            hits = []
            for target, truth in enumerate(true_angles):
                peak = int(np.argmax(maps[trial, target]))
                estimate = angle_grid[peak]
                hits.append(
                    maps[trial, target, peak] >= threshold
                    and abs(estimate - truth) <= 3.0
                )
            resolved.append(all(hits))
        else:
            spectrum = maps[trial, 0]
            candidates = [
                index for index in range(1, len(angle_grid) - 1)
                if spectrum[index] >= threshold
                and spectrum[index] > spectrum[index - 1]
                and spectrum[index] >= spectrum[index + 1]
            ]
            estimates = [angle_grid[index] for index in candidates]
            left = any(abs(estimate - true_angles[0]) <= 3.0 for estimate in estimates)
            right = any(abs(estimate - true_angles[1]) <= 3.0 for estimate in estimates)
            resolved.append(left and right and angle_gap > 6.0)
    probability = float(np.mean(resolved))
    standard_error = np.sqrt(probability * (1.0 - probability) / trials)
    return {
        "angle_gap_degrees": angle_gap,
        "joint_resolution_probability": probability,
        "joint_resolution_95ci": [
            max(0.0, probability - 1.96 * standard_error),
            min(1.0, probability + 1.96 * standard_error),
        ],
        "frame_false_alarm_probability": frame_pfa,
        "threshold": threshold,
    }


def main():
    rng = np.random.default_rng(20260813)
    random_codes = [
        np.exp(0.5j * np.pi * rng.integers(0, 4, 8)) / np.sqrt(8)
        for _ in range(2)
    ]
    cazac_codes = [cazac_sequence(8, root) for root in (1, 3)]
    payload = {
        "scope": (
            "conditional spatial gate at a known DD cell; not yet a full "
            "unknown-delay/Doppler/angle detector"
        ),
        "gaps": {
            str(gap): {
                "array_only": evaluate_mode(None, gap),
                "cazac_codes": evaluate_mode(cazac_codes, gap),
                "random_codes": evaluate_mode(random_codes, gap),
            }
            for gap in (2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
        },
    }
    output = Path("results/dd_spatial_monte_carlo.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
