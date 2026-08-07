import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.hybrid_fusion import hybrid_gaussian_hard_pd


def test_hybrid_reduces_to_soft_only():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=1.0
    )
    soft = {0, 1, 2}
    hybrid = hybrid_gaussian_hard_pd(model, soft, [], 0.05, grid=256)
    expected = gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1,
        soft, 0.05,
    )
    assert abs(hybrid["pd"] - expected) < 1e-6
    assert hybrid["pfa"] <= 0.05 + 1e-9


def test_hybrid_with_hard_reports_respects_pfa():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
    )
    result = hybrid_gaussian_hard_pd(
        model, {0}, {1, 2, 3}, 0.05, grid=256
    )
    assert result["pfa"] <= 0.05 + 1e-9
    assert result["pd"] > 0.0
