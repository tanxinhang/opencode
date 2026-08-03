import numpy as np

from scripts.run_probe_mismatch_gate import mismatched_probe_pair


def test_zero_mismatch_reaches_nominal_dft_orthogonality_at_length_four():
    first, second = mismatched_probe_pair(
        4, 0.0, 0.0, 0.0, np.random.default_rng(1)
    )
    assert abs(np.vdot(first, second)) < 1e-12


def test_cfo_breaks_nominal_probe_orthogonality():
    first, second = mismatched_probe_pair(
        4, 0.0, 0.1, 0.0, np.random.default_rng(1)
    )
    assert abs(np.vdot(first, second)) > 0.1
