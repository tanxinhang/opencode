import numpy as np

from scripts.run_dd_spatial_monte_carlo import (
    calibrate_threshold,
    spatial_dictionary,
)


def test_spatial_dictionary_is_unit_norm():
    dictionary = spatial_dictionary([-20.0, 0.0, 20.0], 8)
    assert dictionary.shape == (3, 8)
    assert np.allclose(np.sum(np.abs(dictionary) ** 2, axis=1), 1.0)


def test_spatial_frame_threshold_is_positive():
    dictionary = spatial_dictionary([-20.0, 0.0, 20.0], 4)[None, :, :]
    assert calibrate_threshold(dictionary, 0.2, 0.01, trials=2_000) > 0.0
