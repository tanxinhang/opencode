import numpy as np

from scripts.run_confirmation_fairness_gate import (
    combine_energy_blocks,
    decoded_score_and_support,
)


def test_energy_block_combination_preserves_common_signal():
    signal = np.array([1.0 + 2.0j, -0.5j])
    energies = (0.3, 0.2, 0.5)
    blocks = [np.sqrt(energy) * signal for energy in energies]
    assert np.allclose(combine_energy_blocks(blocks, energies), signal)


def test_decoded_score_uses_weaker_identity_channel():
    dictionary = np.eye(2, dtype=complex)
    parameters = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    score, support = decoded_score_and_support(
        (np.array([2.0, 0.0]), np.array([0.0, 1.0])),
        dictionary, parameters,
    )
    assert score == 1.0
    assert np.array_equal(support, parameters)
