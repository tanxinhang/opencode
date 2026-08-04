"""Audit geometry-derived covariance of bistatic range/angle position inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_g0b import (
    CARRIER_HZ, PROPAGATION_SPEED, draw_targets, nested_transmitter_geometry,
)
from uav_otfs_isac.multistatic_association import (
    bistatic_position_covariance, position_from_angle_range,
)
from uav_otfs_isac.multistatic_targets import KinematicNode, generate_bistatic_paths


def run_covariance_audit(scenes=300, seed=20261091, transmitters=8,
                         monte_carlo_draws=20000):
    rng = np.random.default_rng(seed)
    nodes = nested_transmitter_geometry(transmitters)
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    major, minor, condition, target_range = [], [], [], []
    reference = None
    for _ in range(scenes):
        targets = draw_targets(rng, 6, "separated")
        paths = generate_bistatic_paths(nodes, targets, receiver, CARRIER_HZ)
        target_by_id = {target.target_id: target for target in targets}
        for path in paths:
            covariance = bistatic_position_covariance(
                nodes[path.transmitter_id].position, receiver.position,
                path.delay_s * PROPAGATION_SPEED,
                path.receive_azimuth_rad,
            )
            eigenvalues = np.linalg.eigvalsh(covariance)
            minor.append(np.sqrt(eigenvalues[0]))
            major.append(np.sqrt(eigenvalues[1]))
            condition.append(eigenvalues[1] / eigenvalues[0])
            target_range.append(np.linalg.norm(
                target_by_id[path.target_id].position - receiver.position
            ))
            if reference is None:
                reference = path
    reference_node = nodes[reference.transmitter_id]
    rho = reference.delay_s * PROPAGATION_SPEED
    theta = reference.receive_azimuth_rad
    predicted = bistatic_position_covariance(
        reference_node.position, receiver.position, rho, theta
    )
    nominal = position_from_angle_range(
        reference_node.position, receiver.position, theta, rho
    )
    samples = []
    for _ in range(monte_carlo_draws):
        noisy_rho = rho + rng.normal(0.0, 1.5)
        noisy_theta = theta + rng.normal(0.0, np.deg2rad(0.4))
        try:
            samples.append(position_from_angle_range(
                reference_node.position, receiver.position,
                noisy_theta, noisy_rho,
            ) - nominal)
        except ValueError:
            pass
    empirical = np.cov(np.asarray(samples), rowvar=False)
    relative_error = np.linalg.norm(empirical - predicted) / np.linalg.norm(predicted)
    def quantiles(values):
        return {str(q): float(np.quantile(values, q))
                for q in (0.01, 0.1, 0.5, 0.9, 0.99)}
    return {
        "scope": (
            "first-order covariance from the exact synthetic front-end range "
            "and bearing noise; not inferred from confidence"
        ),
        "seed": seed,
        "sample_count": len(major),
        "assumed_noise": {"bistatic_range_sigma_m": 1.5,
                          "receive_angle_sigma_deg": 0.4},
        "minor_axis_sigma_m": quantiles(minor),
        "major_axis_sigma_m": quantiles(major),
        "covariance_condition_number": quantiles(condition),
        "major_sigma_range_spearman": float(spearmanr(
            major, target_range).statistic),
        "uniform_3m_area_ratio_quantiles": quantiles(
            9.0 / (np.asarray(major) * np.asarray(minor))
        ),
        "monte_carlo_validation": {
            "draws_retained": len(samples),
            "predicted_covariance_m2": predicted.tolist(),
            "empirical_covariance_m2": empirical.tolist(),
            "relative_frobenius_error": float(relative_error),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multistatic_position_covariance_audit_m8_n6.json"))
    args = parser.parse_args()
    payload = run_covariance_audit(args.scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
