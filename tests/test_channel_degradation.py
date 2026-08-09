import numpy as np

from uav_otfs_isac.channel_degradation import (
    bsc_cascade_flip,
    bsc_cascade_transition,
    bsc_lrt_roc_point,
    exact_lrt_roc_point,
    verify_bsc_roc_dominance,
)
from uav_otfs_isac.reporting import bsc_transition


def test_bsc_cascade_transition_matches_direct_high_flip():
    for bits in (1, 2, 3, 4):
        for lo, hi in ((0.0, 0.3), (0.1, 0.4), (0.2, 0.45)):
            cascade = bsc_cascade_transition(bits, lo, hi)
            direct = bsc_transition(bits, hi)
            assert np.allclose(cascade, direct, atol=1e-12)


def test_bsc_cascade_flip_composition_identity():
    assert np.isclose(bsc_cascade_flip(0.1, 0.4), 0.3 / 0.8)
    assert np.isclose(bsc_cascade_flip(0.0, 0.3), 0.3)
    assert np.isclose(bsc_cascade_flip(0.5, 0.5), 0.5)


def test_exact_lrt_roc_point_randomizes_boundary_atom():
    null = np.array([0.8, 0.2])
    alternative = np.array([0.2, 0.8])
    pfa, achieved = exact_lrt_roc_point(null, alternative, 0.05)
    assert np.isclose(achieved, 0.05)
    assert 0.0 <= pfa <= 1.0


def test_bsc_roc_dominance_holds_on_exact_lrt_grid():
    result = verify_bsc_roc_dominance(
        bits_options=(1, 2, 3),
        mu1_options=(1.0, 1.5, 2.0),
        lo_options=(0.0, 0.1, 0.2),
        hi_options=(0.3, 0.4, 0.45),
        false_alarm_grid=(0.01, 0.05, 0.1, 0.2),
    )
    assert result["passed"]
    assert result["cells"] > 0
    assert result["minimum_pd_gap_clean_minus_degraded"] >= -1e-10


def test_bsc_lrt_roc_point_is_between_zero_and_one():
    pd, pfa = bsc_lrt_roc_point(
        0.0, 1.0, 1.5, 1.0, bits=2,
        bit_flip_probability=0.1, false_alarm_rate=0.05,
    )
    assert np.isclose(pfa, 0.05)
    assert 0.0 <= pd <= 1.0


def test_bsc_roc_dominance_rejects_empty_grid():
    try:
        verify_bsc_roc_dominance(
            bits_options=[],
            mu1_options=[1.0],
            lo_options=[0.0],
            hi_options=[0.3],
            false_alarm_grid=[0.05],
        )
    except ValueError as error:
        assert "nonempty" in str(error)
    else:
        raise AssertionError("empty option grids must be rejected")
