import numpy as np

from scripts.run_markov_signature_gate import (
    best_fixed_state_pair,
    permutation_transition,
    qpsk_pilot_codebook,
    sticky_transition,
)


def test_pilot_codebook_has_unit_energy_states():
    codebook = qpsk_pilot_codebook(3, 2)
    assert codebook.shape == (3, 2)
    assert np.allclose(np.linalg.norm(codebook, axis=1), 1.0)


def test_deterministic_transition_is_a_valid_markov_matrix():
    matrix = permutation_transition([1, 2, 0])
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(matrix, axis=1), [1, 2, 0])


def test_sticky_transition_interpolates_stay_and_cycle():
    matrix = sticky_transition(3, 0.25)
    assert np.allclose(np.diag(matrix), 0.25)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_fixed_baseline_uses_constant_paths():
    result = best_fixed_state_pair(3)
    assert all(len(set(path)) == 1 for path in result["paths"])
