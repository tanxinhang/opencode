import numpy as np

from scripts.run_confirmation_mismatch_gate import (
    actual_probe_matrix,
    decode_with_probe_matrix,
    nominal_probe_matrix,
    regularized_decode,
)


def test_oracle_probe_demixing_recovers_two_noiseless_sources():
    nominal = nominal_probe_matrix((0.3, 0.7))
    fading = np.ones((3, 2), dtype=complex)
    phase_noise = np.zeros((3, 2))
    actual = actual_probe_matrix(nominal, fading, (0.02, -0.01), phase_noise)
    sources = np.array([[1.0 + 1.0j, 2.0], [-1.0j, 0.5]])
    assert np.allclose(
        decode_with_probe_matrix(actual @ sources, actual), sources
    )


def test_nominal_probe_matrix_has_sum_then_difference_rows():
    matrix = nominal_probe_matrix((0.25, 0.5))
    assert np.allclose(matrix[0], [1.0, 1.0])
    assert np.allclose(matrix[1], [0.5, -0.5])


def test_regularized_decode_is_finite_for_rank_deficient_probe():
    matrix = np.ones((2, 2), dtype=complex)
    decoded = regularized_decode(np.ones((2, 3)), matrix, 0.1)
    assert decoded.shape == (2, 3)
    assert np.all(np.isfinite(decoded))
