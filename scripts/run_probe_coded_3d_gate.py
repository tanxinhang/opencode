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
)
from uav_otfs_isac.spatial_detection import (
    decode_probe_code,
    probe_coded_spatial_template,
    separable_detection_cube,
    spatial_dictionary,
    threshold_nms_3d,
    waveform_dictionary,
)


def make_probe_codes(length, kind, seed=20260820):
    if kind == "cazac":
        base = cazac_sequence(length, 1)
        return [base, np.roll(base, 1)]
    if kind == "dft":
        indices = np.arange(length)
        return [
            np.ones(length, dtype=complex) / np.sqrt(length),
            np.exp(2j * np.pi * indices / length) / np.sqrt(length),
        ]
    if kind == "random_qpsk":
        rng = np.random.default_rng(seed)
        return [
            np.exp(0.5j * np.pi * rng.integers(0, 4, length)) / np.sqrt(length)
            for _ in range(2)
        ]
    raise ValueError("unsupported probe code kind")


def evaluate_probe_coded(
    angle_gap, probe_length=8, code_kind="cazac", trials=1_000,
):
    shape = (4, 8)
    antennas = 8
    pattern = qpsk_phase_pattern(*shape, 11)
    waveforms = waveform_dictionary(pattern)
    angle_grid = np.arange(-60.0, 60.01, 5.0)
    spatial = spatial_dictionary(angle_grid, antennas)
    true_angles = (-angle_gap / 2.0, angle_gap / 2.0)
    codes = make_probe_codes(probe_length, code_kind)
    paths = [
        spatial_otfs_template(pattern, 3.20, 1.15, angle, antennas)
        for angle in true_angles
    ]
    templates = [
        probe_coded_spatial_template(path, code)
        for path, code in zip(paths, codes)
    ]
    noise_variance = 0.02
    rng = np.random.default_rng(20260819)
    h0_maxima = []
    for _ in range(10_000):
        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal((probe_length, antennas, pattern.size))
            + 1j * rng.standard_normal((probe_length, antennas, pattern.size))
        )
        h0_maxima.append(max(
            np.max(separable_detection_cube(
                decode_probe_code(noise, code), waveforms, spatial
            ))
            for code in codes
        ))
    threshold = float(np.quantile(h0_maxima, 0.99, method="higher"))
    phases = rng.uniform(0.0, 2 * np.pi, (trials, 2))
    joint_identity = []
    extra_candidates = []
    for trial in range(trials):
        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal((probe_length, antennas, pattern.size))
            + 1j * rng.standard_normal((probe_length, antennas, pattern.size))
        )
        received = (
            np.exp(1j * phases[trial, 0]) * templates[0]
            + np.exp(1j * phases[trial, 1]) * templates[1]
            + noise
        )
        hits = []
        candidate_count = 0
        for target, code in enumerate(codes):
            decoded = decode_probe_code(received, code)
            cube = separable_detection_cube(decoded, waveforms, spatial)
            peaks = threshold_nms_3d(cube, threshold, shape, 1, 1)
            candidate_count += len(peaks)
            hit = any(
                abs(angle_grid[angle] - true_angles[target]) <= 5.0
                and min((k - 1) % shape[0], (1 - k) % shape[0]) <= 1
                and min((l - 3) % shape[1], (3 - l) % shape[1]) <= 1
                for angle, k, l in peaks
            )
            hits.append(hit)
        joint_identity.append(all(hits))
        extra_candidates.append(candidate_count - sum(hits))
    return {
        "angle_gap_degrees": angle_gap,
        "probe_length": probe_length,
        "code_kind": code_kind,
        "probe_code_squared_coherence": float(abs(np.vdot(codes[0], codes[1])) ** 2),
        "threshold": threshold,
        "joint_identity_correct_probability": float(np.mean(joint_identity)),
        "extra_candidate_peaks_per_h1_frame": float(np.mean(extra_candidates)),
        "trials": trials,
        "resource_warning": (
            f"Uses {probe_length} resolved probing snapshots at fixed total "
            "template energy."
        ),
    }


def main():
    payload = {
        "model": "Codes on an independent resolved probe-snapshot dimension",
        "cases": {
            f"{kind}_p{length}": evaluate_probe_coded(
                5.0, length, kind, trials=1_000
            )
            for length in (2, 4, 8)
            for kind in ("cazac", "dft", "random_qpsk")
        },
    }
    output = Path("results/probe_coded_3d_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
