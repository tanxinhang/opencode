import numpy as np

from uav_otfs_isac.linalg import nearest_psd, regularize_covariance


def test_nearest_psd_repairs_indefinite_matrix():
    matrix = np.array([[1.0, 1.5], [1.5, 1.0]])
    repaired = nearest_psd(matrix, epsilon=1e-8)
    assert np.allclose(repaired, repaired.T)
    assert np.linalg.eigvalsh(repaired).min() >= 1e-8 - 1e-12


def test_regularization_is_positive_definite():
    matrix = np.ones((3, 3))
    repaired = regularize_covariance(matrix, shrinkage=0.1, epsilon=1e-6)
    assert np.linalg.eigvalsh(repaired).min() > 0

