from __future__ import annotations

import json
from pathlib import Path
import sys
from functools import lru_cache

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_end_to_end_adaptive_probe_gate import (
    ANTENNAS,
    NOISE_VARIANCE,
    SHAPE,
    fine_local_dictionary,
    supports_match,
)
from uav_otfs_isac.otfs_physical import (
    qpsk_phase_pattern,
    spatial_otfs_template,
)


EQUAL_DIFFERENCE_ENERGIES = (0.3, 0.2828)
FULL_DIFFERENCE_ENERGIES = (0.3, 0.7)


@lru_cache(maxsize=1)
def fixed_collision_problem():
    """Cache the common fine dictionary and fixed 5-degree collision templates."""
    pattern = qpsk_phase_pattern(*SHAPE, 11)
    dictionary, parameters = fine_local_dictionary(pattern)
    truth = np.asarray([
        (-2.5, 3.2, 1.15), (2.5, 3.2, 1.15)
    ])
    templates = np.asarray([
        spatial_otfs_template(
            pattern, delay, doppler, angle, ANTENNAS
        ).reshape(-1)
        for angle, delay, doppler in truth
    ])
    return dictionary, parameters, truth, templates


def nominal_probe_matrix(difference_energies):
    """Return sum then sign-reversed incremental probe coefficients."""
    rows = [[1.0, 1.0]]
    rows.extend([
        [np.sqrt(energy), -np.sqrt(energy)]
        for energy in difference_energies
    ])
    return np.asarray(rows, dtype=complex)


def actual_probe_matrix(base, fading, cfo, phase_noise):
    """Apply target- and snapshot-dependent complex channel coefficients."""
    matrix = np.asarray(base, dtype=complex).copy()
    snapshots = np.arange(matrix.shape[0])[:, None]
    phase = 2.0 * np.pi * snapshots * np.asarray(cfo)[None, :] + phase_noise
    return matrix * fading * np.exp(1j * phase)


def decode_with_probe_matrix(observations, assumed_matrix):
    """Jointly demix two identity channels with an assumed probe matrix."""
    observations = np.asarray(observations, dtype=complex)
    assumed = np.asarray(assumed_matrix, dtype=complex)
    if observations.ndim != 2 or assumed.shape != (observations.shape[0], 2):
        raise ValueError("observations and assumed probe matrix are incompatible")
    return np.linalg.pinv(assumed) @ observations


def regularized_decode(observations, assumed_matrix, regularization):
    """Ridge-stabilized two-source demixing for ill-conditioned probes."""
    assumed = np.asarray(assumed_matrix, dtype=complex)
    gram = assumed.conj().T @ assumed
    return np.linalg.solve(
        gram + regularization * np.eye(gram.shape[0]),
        assumed.conj().T @ np.asarray(observations),
    )


def support_success(decoded, dictionary, parameters, truth):
    estimates = []
    for channel in decoded:
        powers = np.abs(dictionary.conj().T @ channel) ** 2
        estimates.append(parameters[int(np.argmax(powers))])
    return supports_match(np.asarray(estimates), truth)


def correlated_fading(rng, correlation, snapshots, targets):
    """Stationary complex Gauss-Markov coefficients anchored at unit gain."""
    fading = np.ones((snapshots, targets), dtype=complex)
    innovation_scale = np.sqrt(max(0.0, 1.0 - correlation ** 2))
    for snapshot in range(1, snapshots):
        innovation = (
            rng.standard_normal(targets) + 1j * rng.standard_normal(targets)
        ) / np.sqrt(2.0)
        fading[snapshot] = (
            correlation * fading[snapshot - 1]
            + innovation_scale * innovation
        )
    return fading


def evaluate_case(correlation, phase_std_degrees, residual_cfo,
                  trials=200, seed=20261001, signal_scale=0.95,
                  near_far_db=6.0):
    dictionary, parameters, truth, templates = fixed_collision_problem()
    rng = np.random.default_rng(seed)
    phase_std = np.deg2rad(phase_std_degrees)
    results = {
        energy: {decoder: [] for decoder in (
            "oracle", "oracle_ridge", "cfo_compensated", "nominal"
        )}
        for energy in ("equal", "full")
    }
    condition_numbers = {energy: [] for energy in ("equal", "full")}
    for _ in range(trials):
        initial_phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, 2))
        target_scales = signal_scale * np.asarray([
            1.0, 10.0 ** (-near_far_db / 20.0)
        ])
        signals = initial_phase[:, None] * target_scales[:, None] * templates
        fading = correlated_fading(rng, correlation, 3, 2)
        cfo = np.asarray([residual_cfo, -residual_cfo])
        phase_noise = np.zeros((3, 2))
        phase_noise[1:] = rng.normal(0.0, phase_std, (2, 2))
        for energy_name, difference_energies in (
            ("equal", EQUAL_DIFFERENCE_ENERGIES),
            ("full", FULL_DIFFERENCE_ENERGIES),
        ):
            nominal = nominal_probe_matrix(difference_energies)
            actual = actual_probe_matrix(
                nominal, fading, cfo, phase_noise
            )
            condition_numbers[energy_name].append(float(np.linalg.cond(actual)))
            observations = actual @ signals
            observations += np.sqrt(NOISE_VARIANCE / 2.0) * (
                rng.standard_normal(observations.shape)
                + 1j * rng.standard_normal(observations.shape)
            )
            snapshots = np.arange(actual.shape[0])[:, None]
            compensated = nominal * np.exp(
                2j * np.pi * snapshots * cfo[None, :]
            )
            for decoder, assumed in (
                ("oracle", actual),
                ("cfo_compensated", compensated),
                ("nominal", nominal),
            ):
                decoded = decode_with_probe_matrix(observations, assumed)
                results[energy_name][decoder].append(
                    support_success(decoded, dictionary, parameters, truth)
                )
            ridge = regularized_decode(
                observations, actual, regularization=NOISE_VARIANCE
            )
            results[energy_name]["oracle_ridge"].append(
                support_success(ridge, dictionary, parameters, truth)
            )
    summary = {
        energy: {
            decoder: float(np.mean(values))
            for decoder, values in decoders.items()
        }
        for energy, decoders in results.items()
    }
    return {
        "snapshot_correlation": correlation,
        "phase_noise_std_degrees": phase_std_degrees,
        "residual_cfo": residual_cfo,
        "signal_scale": signal_scale,
        "near_far_db": near_far_db,
        "trials": trials,
        "exact_support_probability": summary,
        "equal_minus_full": {
            decoder: summary["equal"][decoder] - summary["full"][decoder]
            for decoder in summary["equal"]
        },
        "mean_actual_probe_condition_number": {
            energy: float(np.mean(values))
            for energy, values in condition_numbers.items()
        },
    }


def main():
    rows = []
    seed = 20261001
    for correlation in (1.0, 0.99, 0.95, 0.9):
        for phase_std in (0.0, 5.0, 15.0, 30.0):
            for cfo in (0.0, 0.01, 0.05, 0.1):
                rows.append(evaluate_case(
                    correlation, phase_std, cfo, trials=200, seed=seed
                ))
                seed += 1
    acceptable = {
        decoder: [
            row for row in rows
            if row["exact_support_probability"]["full"][decoder] > 0.0
            and row["equal_minus_full"][decoder] >= -0.02
        ]
        for decoder in (
            "oracle", "oracle_ridge", "cfo_compensated", "nominal"
        )
    }
    payload = {
        "scope": (
            "known two-source 5-degree same-DD cluster with three physical "
            "incremental snapshots; known target count"
        ),
        "equal_normalized_energy": 1.5828,
        "full_normalized_energy": 2.0,
        "energy_saving": 0.2086,
        "signal_scale": 0.95,
        "near_far_db": 6.0,
        "rows": rows,
        "fraction_of_mismatch_grid_with_loss_at_most_2pp": {
            decoder: len(values) / len(rows)
            for decoder, values in acceptable.items()
        },
        "warning": (
            "Noncoherent energy combining cannot recover identity from sign-only "
            "probe codes and is therefore not treated as exact-support recovery."
        ),
    }
    output = Path("results/confirmation_mismatch_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
