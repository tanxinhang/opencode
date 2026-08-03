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
    matched_filter_map,
    qpsk_phase_pattern,
    spatial_matched_filter_map,
    spatial_otfs_template,
)


def local_maxima(values):
    return [
        index for index in range(1, len(values) - 1)
        if values[index] > values[index - 1]
        and values[index] >= values[index + 1]
    ]


def spatial_resolution_case(angle_gap, spatial_codes=None):
    num_antennas = 8
    pattern = qpsk_phase_pattern(8, 16, 11)
    angles = (-angle_gap / 2.0, angle_gap / 2.0)
    templates = [
        spatial_otfs_template(
            pattern, 4.0, 2.0, angle, num_antennas,
            None if spatial_codes is None else spatial_codes[index],
        )
        for index, angle in enumerate(angles)
    ]
    received = templates[0] + templates[1]
    two_dimensional = matched_filter_map(received[0], pattern)
    angle_grid = np.arange(-60.0, 60.01, 0.5)
    cube = spatial_matched_filter_map(received, pattern, angle_grid)
    spatial_slice = cube[:, 2, 4]
    if spatial_codes is None:
        peaks = local_maxima(spatial_slice)
        strongest = sorted(
            peaks, key=lambda index: spatial_slice[index], reverse=True
        )[:2]
        strongest = sorted(strongest)
        estimated_angles = [float(angle_grid[index]) for index in strongest]
        if len(strongest) == 2:
            valley = float(np.min(spatial_slice[strongest[0]:strongest[1] + 1]))
            weaker_peak = float(min(spatial_slice[index] for index in strongest))
            valley_ratio = valley / max(weaker_peak, 1e-15)
        else:
            valley_ratio = 1.0
        resolved = (
            len(estimated_angles) == 2
            and all(abs(estimate - truth) <= 3.0
                    for estimate, truth in zip(estimated_angles, angles))
            and valley_ratio <= 0.8
        )
    else:
        estimated_angles = []
        desired_to_leakage = []
        for target, code in enumerate(spatial_codes):
            coded_cube = spatial_matched_filter_map(
                received, pattern, angle_grid, code
            )
            coded_slice = coded_cube[:, 2, 4]
            peak = int(np.argmax(coded_slice))
            estimated_angles.append(float(angle_grid[peak]))
            desired_index = int(np.argmin(np.abs(angle_grid - angles[target])))
            other_index = int(np.argmin(np.abs(angle_grid - angles[1 - target])))
            desired_to_leakage.append(float(
                coded_slice[desired_index]
                / max(coded_slice[other_index], 1e-15)
            ))
        estimated_angles = sorted(estimated_angles)
        valley_ratio = None
        resolved = (
            min(desired_to_leakage) >= 2.0
            and all(
            abs(estimate - truth) <= 3.0
            for estimate, truth in zip(estimated_angles, angles)
            )
        )
    return {
        "angle_gap_degrees": angle_gap,
        "two_dimensional_peak_count": int(
            np.sum(two_dimensional >= 0.5 * np.max(two_dimensional))
        ),
        "estimated_angles": estimated_angles,
        "resolved": resolved,
        "valley_to_weaker_peak_ratio": valley_ratio,
        "mean_desired_to_other_angle_ratio": (
            float(np.mean(desired_to_leakage))
            if spatial_codes is not None else None
        ),
        "spatial_peak_to_midpoint_ratio": float(
            max(spatial_slice) / max(
                spatial_slice[np.argmin(np.abs(angle_grid))], 1e-15
            )
        ),
    }


def main():
    gaps = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    array_only = [spatial_resolution_case(gap) for gap in gaps]
    cazac_codes = [cazac_sequence(8, root) for root in (1, 3)]
    cazac_coded = [
        spatial_resolution_case(gap, cazac_codes) for gap in gaps
    ]
    random_rng = np.random.default_rng(20260813)
    random_codes = [
        np.exp(0.5j * np.pi * random_rng.integers(0, 4, 8)) / np.sqrt(8)
        for _ in range(2)
    ]
    random_coded = [
        spatial_resolution_case(gap, random_codes) for gap in gaps
    ]
    payload = {
        "model": "8-element half-wavelength ULA; same DD cell; noiseless",
        "array_only": array_only,
        "cazac_spatial_codes": cazac_coded,
        "random_spatial_codes": random_coded,
        "interpretation_warning": (
            "Distinct spatial codes require resolved per-element or orthogonal "
            "probing observations; elementwise coding changes the effective "
            "array manifold and is not free array gain."
        ),
    }
    output = Path("results/dd_spatial_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
