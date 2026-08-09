import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.erasure_dominance import (
    erasure_cascade_factor,
    verify_expected_pd_monotonicity,
    verify_monotone_coupling,
)


def test_erasure_cascade_factor_preserves_marginal():
    factor = erasure_cascade_factor(0.8, 0.6)
    assert np.isclose(0.8 * factor, 0.6)
    assert np.isclose(erasure_cascade_factor(0.0, 0.0), 0.0)


def test_monotone_coupling_always_keeps_degraded_set_as_subset():
    high = np.array([0.9, 0.8, 0.7, 0.6])
    low = np.array([0.6, 0.5, 0.4, 0.3])
    result = verify_monotone_coupling(high, low, samples=50_000)
    assert result["passed"]
    assert result["coupling_violations"] == 0


def test_expected_pd_is_nondecreasing_in_success_probability():
    clean = symmetric_diversity_model(
        np.full(4, 1.4), success_probability=0.9
    )
    degraded = symmetric_diversity_model(
        np.full(4, 1.4), success_probability=0.5
    )
    result = verify_expected_pd_monotonicity(
        clean, degraded, grid=128
    )
    assert result["passed"]
    assert result["gap"] >= -1e-9
    assert result["in_theorem_scope"]


def test_invalid_success_ordering_is_rejected():
    try:
        erasure_cascade_factor(0.4, 0.6)
    except ValueError as error:
        assert "0 <= low <= high" in str(error)
    else:
        raise AssertionError("invalid ordering must be rejected")
