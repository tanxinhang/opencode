import numpy as np

from scripts.run_end_to_end_adaptive_probe_gate import (
    supports_match,
    two_source_residual_statistic,
)


def test_support_matching_is_one_to_one_and_order_invariant():
    truth = np.array([[-5.0, 3.1, 1.0], [5.0, 3.3, 1.2]])
    estimated = np.array([[4.0, 3.25, 1.15], [-4.0, 3.05, 0.95]])
    assert supports_match(estimated, truth)
    assert not supports_match(estimated + np.array([0.0, 0.3, 0.0]), truth)


def test_second_atom_improves_residual_for_separated_dictionary_columns():
    dictionary = np.eye(3, dtype=complex)
    parameters = np.array([
        [0.0, 0.0, 0.0], [10.0, 1.0, 1.0], [20.0, 2.0, 2.0]
    ])
    statistic, support, diagnostics = two_source_residual_statistic(
        np.array([1.0, 0.8, 0.0]), dictionary, parameters
    )
    assert statistic > 0.99
    assert set(support) == {0, 1}
    assert diagnostics["lambda_min"] == 1.0
