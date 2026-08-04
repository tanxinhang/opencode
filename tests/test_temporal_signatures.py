import numpy as np
import pytest

from uav_otfs_isac.temporal_signatures import (
    deterministic_cycle_path,
    gauss_markov_gains,
    multiframe_joint_gram,
    sample_markov_path,
    stationary_switch_probability,
    validate_transition_matrix,
)


def test_transition_validation_and_reproducible_markov_path():
    transition = np.array([[0.8, 0.2], [0.1, 0.9]])
    assert np.array_equal(validate_transition_matrix(transition), transition)
    first = sample_markov_path(transition, 0, 8, np.random.default_rng(7))
    second = sample_markov_path(transition, 0, 8, np.random.default_rng(7))
    assert np.array_equal(first, second)
    with pytest.raises(ValueError):
        validate_transition_matrix([[0.8, 0.3], [0.1, 0.9]])


def test_cycle_and_switch_probability_have_expected_values():
    assert np.array_equal(
        deterministic_cycle_path(1, 3, 5), [1, 2, 0, 1, 2]
    )
    transition = np.full((3, 3), 0.1)
    np.fill_diagonal(transition, 0.8)
    assert np.isclose(stationary_switch_probability(transition), 0.2)


def test_different_temporal_paths_improve_collocated_source_gram():
    codebook = np.eye(3, dtype=complex)
    physical = [np.ones(2), np.ones(2)]
    waveform = [np.ones(3), np.ones(3)]
    collapsed = multiframe_joint_gram(
        [codebook, codebook], [[0, 0, 0], [0, 0, 0]],
        physical, waveform,
    )
    separated = multiframe_joint_gram(
        [codebook, codebook], [[0, 1, 2], [1, 2, 0]],
        physical, waveform,
    )
    assert np.isclose(abs(collapsed[0, 1]), 1.0)
    assert np.isclose(abs(separated[0, 1]), 0.0)


def test_gauss_markov_gain_is_constant_at_unit_correlation():
    gains = gauss_markov_gains(4, 1.0, np.random.default_rng(5))
    assert np.allclose(gains, 1.0)
