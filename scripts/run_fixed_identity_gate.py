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
    NOISE_VARIANCE,
    SHAPE,
    draw_truth,
    fine_local_dictionary,
    supports_match,
)
from uav_otfs_isac.otfs_physical import (
    qpsk_phase_pattern,
    spatial_otfs_template,
)


def identity_codebook(mode, identities=4):
    """Return equal-energy two-feature identity signatures."""
    first = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    if mode == "shared":
        return np.tile(first, (identities, 1))
    elif mode == "fixed_nonorthogonal":
        phases = 2.0 * np.pi * np.arange(identities) / identities
        return np.column_stack((
            np.ones(identities), np.exp(1j * phases)
        )) / np.sqrt(2.0)
    elif mode == "ideal_orthogonal":
        return np.asarray([
            first, np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
        ])
    else:
        raise ValueError("unsupported identity-code mode")


def joint_pair_ls(observation, physical_dictionary, parameters, codes,
                  shortlist=24):
    """Two-source joint LS with the same shortlist size for every code mode."""
    observation = np.asarray(observation, dtype=complex)
    code_count = codes.shape[0]
    correlations = np.asarray([
        physical_dictionary.conj().T @ (codes[identity].conj() @ observation)
        for identity in range(code_count)
    ])
    if np.allclose(codes[0], codes[1]):
        powers = np.abs(correlations[0]) ** 2
        count = min(2 * shortlist, powers.size)
        candidates = np.argpartition(powers, -count)[-count:]
        pairs = (
            ((0, int(first)), (1, int(second)))
            for position, first in enumerate(candidates)
            for second in candidates[position + 1:]
        )
    else:
        candidate_sets = []
        for identity in range(2):
            powers = np.abs(correlations[identity]) ** 2
            count = min(shortlist, powers.size)
            candidate_sets.append(np.argpartition(powers, -count)[-count:])
        pairs = (
            ((0, int(first)), (1, int(second)))
            for first in candidate_sets[0] for second in candidate_sets[1]
        )
    energy = float(np.vdot(observation, observation).real)
    best = None
    for first, second in pairs:
        first_identity, first_index = first
        second_identity, second_index = second
        coherence = (
            np.vdot(codes[first_identity], codes[second_identity])
            * np.vdot(
                physical_dictionary[:, first_index],
                physical_dictionary[:, second_index],
            )
        )
        gram = np.array([
            [1.0, coherence], [coherence.conjugate(), 1.0]
        ])
        determinant = 1.0 - abs(coherence) ** 2
        if determinant <= 1e-8:
            continue
        selected = np.array([
            correlations[first_identity, first_index],
            correlations[second_identity, second_index],
        ])
        coefficients = np.linalg.solve(gram, selected)
        residual = energy - float(np.real(np.vdot(selected, coefficients)))
        if best is None or residual < best[0]:
            best = residual, (first_index, second_index)
    if best is None:
        return None
    return parameters[list(best[1])]


def draw_difficult_truth(rng, case):
    center = rng.uniform(-4.0, 4.0)
    delay = rng.uniform(3.05, 3.35)
    doppler = rng.uniform(1.0, 1.3)
    if case == "same_dd_near_angle":
        angle_gap, delay_gap, doppler_gap, near_far = 5.0, 0.0, 0.0, 6.0
    elif case == "fractional_leakage":
        angle_gap, delay_gap, doppler_gap, near_far = 5.0, 0.25, 0.25, 6.0
    elif case == "strong_near_far":
        angle_gap, delay_gap, doppler_gap, near_far = 2.5, 0.1, 0.1, 10.0
    else:
        raise ValueError("unsupported difficult case")
    truth = np.asarray([
        (center - angle_gap / 2.0, delay, doppler),
        (center + angle_gap / 2.0, delay + delay_gap, doppler + doppler_gap),
    ])
    gains = np.asarray([1.0, 10.0 ** (-near_far / 20.0)])
    return truth, gains


def evaluate_mode(mode, cases, trials_per_case=300, seed=20262001):
    pattern = qpsk_phase_pattern(*SHAPE, 11)
    dictionary, parameters = fine_local_dictionary(pattern)
    full_codebook = identity_codebook(mode)
    rng = np.random.default_rng(seed)
    rows = []
    for case in cases:
        for trial in range(trials_per_case):
            identity_pair = rng.choice(4, 2, replace=False)
            codes = (
                full_codebook[:2]
                if mode == "ideal_orthogonal"
                else full_codebook[identity_pair]
            )
            truth, gains = draw_difficult_truth(rng, case)
            phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, 2))
            physical = [
                spatial_otfs_template(
                    pattern, delay, doppler, angle, ANTENNAS
                ).reshape(-1)
                for angle, delay, doppler in truth
            ]
            observation = sum(
                gains[source] * phases[source]
                * np.outer(codes[source], physical[source])
                for source in range(2)
            )
            observation += np.sqrt(NOISE_VARIANCE / 2.0) * (
                rng.standard_normal(observation.shape)
                + 1j * rng.standard_normal(observation.shape)
            )
            estimate = joint_pair_ls(
                observation, dictionary, parameters, codes
            )
            position_success = bool(
                estimate is not None and supports_match(estimate, truth)
            )
            identity_success = None
            if mode != "shared":
                errors = np.max(np.abs(
                    estimate - truth
                ) / np.array([5.0, 0.2, 0.2]), axis=1) if estimate is not None else [np.inf]
                identity_success = bool(np.all(np.asarray(errors) <= 1.0))
            rows.append({
                "case": case,
                "trial": trial,
                "position_success": position_success,
                "identity_success": identity_success,
            })
    return {
        "mode": mode,
        "maximum_pilot_coherence": float(max(
            abs(np.vdot(full_codebook[first], full_codebook[second]))
            for first in range(len(full_codebook))
            for second in range(first + 1, len(full_codebook))
        )),
        "position_set_recovery": float(np.mean([
            row["position_success"] for row in rows
        ])),
        "identity_exact_recovery": (
            None if mode == "shared" else float(np.mean([
                row["identity_success"] for row in rows
            ]))
        ),
        "by_case": {
            case: {
                "position_set_recovery": float(np.mean([
                    row["position_success"] for row in rows if row["case"] == case
                ])),
            }
            for case in cases
        },
        "position_outcomes": [bool(row["position_success"]) for row in rows],
    }


def paired_binary_difference(first, second):
    values = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return {
        "mean": mean,
        "95ci": [mean - 1.96 * standard_error,
                  mean + 1.96 * standard_error],
    }


def main():
    cases = (
        "same_dd_near_angle", "fractional_leakage", "strong_near_far"
    )
    results = {
        mode: evaluate_mode(mode, cases, seed=20262001)
        for mode in ("shared", "fixed_nonorthogonal", "ideal_orthogonal")
    }
    shared = results["shared"]
    fixed = results["fixed_nonorthogonal"]
    paired_gain = paired_binary_difference(
        fixed["position_outcomes"], shared["position_outcomes"]
    )
    for result in results.values():
        result.pop("position_outcomes")
    payload = {
        "scope": (
            "controlled M=4, N=2 known-path-count single-frame "
            "identity-angle-continuous-DD joint-LS mechanism gate with equal "
            "two-feature pilot energy; not the general unknown-target receiver"
        ),
        "trials_per_case": 300,
        "cases": list(cases),
        "results": results,
        "gate": {
            "fixed_position_gain_over_shared_pp": 100.0 * (
                fixed["position_set_recovery"] - shared["position_set_recovery"]
            ),
            "paired_fixed_minus_shared": paired_gain,
            "passes_10pp_position_gain": bool(
                fixed["position_set_recovery"] - shared["position_set_recovery"] >= 0.10
            ),
        },
        "warning": (
            "Shared pilots do not identify transmitter labels, so their identity "
            "exact-recovery entry is intentionally null; position-set recovery "
            "provides the fair physical-separation comparison. A merge metric "
            "is omitted because known-Q joint LS always returns two atoms and "
            "the true sources may legitimately lie within one tolerance cell."
            " Fixed identity signatures separate overlapping returns from "
            "different transmitters; they cannot resolve the fundamental "
            "same-transmitter, same-angle, same-DD ambiguity."
        ),
    }
    output = Path("results/fixed_identity_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
