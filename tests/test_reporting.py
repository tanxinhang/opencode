import numpy as np

from uav_otfs_isac.reporting import bsc_transition, post_bsc_moments


def test_bsc_transition_rows_sum_to_one():
    transition = bsc_transition(bits=3, bit_flip_probability=0.08)
    assert np.allclose(transition.sum(axis=1), 1.0)


def test_zero_ber_preserves_quantized_distribution_moments():
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    values = np.array([-1.5, -0.5, 0.5, 1.5])
    mean, variance = post_bsc_moments(0.0, 1.0, edges, values, 2, 0.0)
    assert abs(mean) < 1e-12
    assert variance > 0

