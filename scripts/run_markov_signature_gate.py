from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_joint_gram_gate import conditional_resolution_probability
from uav_otfs_isac.identifiability import gram_identifiability_metrics
from uav_otfs_isac.otfs_physical import (
    delay_doppler_path,
    otfs_modulate,
    qpsk_phase_pattern,
    ula_steering_vector,
)
from uav_otfs_isac.temporal_signatures import (
    deterministic_cycle_path,
    multiframe_joint_gram,
    sample_markov_path,
)


def qpsk_pilot_codebook(states=3, length=2):
    """Fixed nonorthogonal state codebook; the transition law is the only design."""
    indices = np.arange(length)
    return np.asarray([
        np.exp(2j * np.pi * state * indices / states) / np.sqrt(length)
        for state in range(states)
    ])


def permutation_transition(permutation):
    """Represent a deterministic state map as a row-stochastic matrix."""
    permutation = np.asarray(permutation, dtype=int)
    matrix = np.zeros((permutation.size, permutation.size))
    matrix[np.arange(permutation.size), permutation] = 1.0
    return matrix


def sticky_transition(states, stay_probability, preferred_step=1):
    """Randomize between staying and one preferred cyclic successor."""
    matrix = np.zeros((states, states))
    for state in range(states):
        matrix[state, state] = stay_probability
        matrix[state, (state + preferred_step) % states] += 1.0 - stay_probability
    return matrix


def source_physics():
    pattern = qpsk_phase_pattern(4, 8, 11)
    reference = otfs_modulate(pattern)
    steerings = [
        ula_steering_vector(8, -2.5), ula_steering_vector(8, 2.5)
    ]
    waveforms = [
        delay_doppler_path(reference, 3.2, 1.15, 4),
        delay_doppler_path(reference, 3.3, 1.25, 4),
    ]
    return steerings, waveforms


def evaluate_paths(paths, frames, seed):
    codebook = qpsk_pilot_codebook()
    steerings, waveforms = source_physics()
    gram = multiframe_joint_gram(
        [codebook, codebook], paths, steerings, waveforms,
        doppler_phase_steps=(0.08 * np.pi, 0.12 * np.pi),
    )
    metrics = gram_identifiability_metrics(gram)
    probability = conditional_resolution_probability(
        gram, noise_variance=0.08, trials=3_000, seed=seed
    )
    switches = np.mean([
        np.mean(np.diff(path) != 0) if frames > 1 else 0.0
        for path in paths
    ])
    return {
        **metrics,
        "conditional_exact_resolution_probability": probability,
        "mean_switch_rate": float(switches),
        "paths": [np.asarray(path, dtype=int).tolist() for path in paths],
    }


def evaluate_random_policy(transitions, initial_states, frames,
                           realizations, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for realization in range(realizations):
        paths = [
            sample_markov_path(
                transitions[source], initial_states[source], frames, rng
            )
            for source in range(2)
        ]
        rows.append(evaluate_paths(
            paths, frames, seed + 10_000 + realization
        ))
    return {
        "mean_resolution_probability": float(np.mean([
            row["conditional_exact_resolution_probability"] for row in rows
        ])),
        "mean_lambda_min": float(np.mean([
            row["lambda_min"] for row in rows
        ])),
        "mean_switch_rate": float(np.mean([
            row["mean_switch_rate"] for row in rows
        ])),
        "realizations": realizations,
    }


def exhaustive_deterministic_schedules(frames, switch_penalty=0.05):
    """Find the best known state paths under an identifiability/switch objective."""
    states = 3
    best = None
    seed = 20261201
    for first_tail in itertools.product(range(states), repeat=frames - 1):
        first = np.asarray((0,) + first_tail)
        for second_tail in itertools.product(range(states), repeat=frames - 1):
            second = np.asarray((1,) + second_tail)
            result = evaluate_paths((first, second), frames, seed)
            seed += 1
            objective = result["lambda_min"] - switch_penalty * result["mean_switch_rate"]
            if best is None or objective > best[0]:
                best = objective, result
    return {
        "objective": float(best[0]),
        **best[1],
    }


def best_fixed_state_pair(frames):
    """Use the strongest fixed-state assignment as the stationary baseline."""
    candidates = []
    for first in range(3):
        for second in range(3):
            result = evaluate_paths((
                np.full(frames, first, dtype=int),
                np.full(frames, second, dtype=int),
            ), frames, 20261220 + 3 * first + second)
            candidates.append(result)
    return max(
        candidates,
        key=lambda row: row["conditional_exact_resolution_probability"],
    )


def main():
    frames = 3
    states = 3
    fixed = best_fixed_state_pair(frames)
    cycle = evaluate_paths((
        deterministic_cycle_path(0, states, frames, 1),
        deterministic_cycle_path(1, states, frames, 1),
    ), frames, 20261211)
    iid = evaluate_random_policy(
        (np.full((states, states), 1.0 / states),) * 2,
        (0, 1), frames, 80, 20261212,
    )
    sticky_candidates = []
    for stay_first in (0.0, 0.25, 0.5, 0.75):
        for stay_second in (0.0, 0.25, 0.5, 0.75):
            result = evaluate_random_policy(
                (
                    sticky_transition(states, stay_first, 1),
                    sticky_transition(states, stay_second, 1),
                ),
                (0, 1), frames, 80,
                20261300 + int(10 * stay_first + 40 * stay_second),
            )
            objective = (
                result["mean_lambda_min"]
                - 0.05 * result["mean_switch_rate"]
            )
            sticky_candidates.append({
                "stay_probabilities": [stay_first, stay_second],
                "objective": objective,
                **result,
            })
    best_stochastic = max(
        sticky_candidates, key=lambda row: row["objective"]
    )
    selected_stays = best_stochastic["stay_probabilities"]
    independent_markov_validation = evaluate_random_policy(
        (
            sticky_transition(states, selected_stays[0], 1),
            sticky_transition(states, selected_stays[1], 1),
        ),
        (0, 1), frames, 400, 20261999,
    )
    optimized_schedule = exhaustive_deterministic_schedules(frames)
    strongest_simple = max(
        cycle["conditional_exact_resolution_probability"],
        iid["mean_resolution_probability"],
    )
    optimized_probability = optimized_schedule[
        "conditional_exact_resolution_probability"
    ]
    payload = {
        "scope": (
            "known-state T=3 normal-frame multi-source joint-LS gate; fixed "
            "pilot codebook, energy, receiver, and short-window geometry"
        ),
        "frames": frames,
        "pilot_states": states,
        "fixed_signature": fixed,
        "simple_cycle": cycle,
        "iid_switching": iid,
        "best_sticky_markov": best_stochastic,
        "independent_markov_validation": independent_markov_validation,
        "optimized_known_schedule": optimized_schedule,
        "gate": {
            "optimized_gain_over_strongest_simple_pp": 100.0 * (
                optimized_probability - strongest_simple
            ),
            "passes_5pp_gain": bool(
                optimized_probability - strongest_simple >= 0.05
            ),
            "best_markov_is_deterministic_boundary": bool(
                any(value == 0.0 for value in best_stochastic["stay_probabilities"])
            ),
            "validated_markov_gain_over_cycle_pp": 100.0 * (
                independent_markov_validation["mean_resolution_probability"]
                - cycle["conditional_exact_resolution_probability"]
            ),
        },
        "interpretation": (
            "Deterministic schedules are permutation-transition Markov chains. "
            "If optimization lands on that boundary or fails to beat the simple "
            "cycle by 5 pp, Markov randomization is not retained as an innovation."
        ),
    }
    output = Path("results/markov_signature_gate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
