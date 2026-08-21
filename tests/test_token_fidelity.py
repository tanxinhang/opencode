"""Token-fidelity (Lloyd-Max codebook) tests (F0-D follow-up)."""

import numpy as np
import pytest

from uav_otfs_isac.distributed_audit import (
    build_distributed_scenario,
    build_token_quantizer,
    lloyd_max_quantizer,
    mu_law_quantizer,
    quantize_llr,
    quantize_with,
    quantized_kernels,
    uniform_quantizer,
)


def _mixture_source():
    """Small-atom-heavy mixture mimicking weak-target LLR atoms."""
    rng = np.random.default_rng(3)
    small = rng.normal(0.0, 0.25, 600)
    large = rng.normal(0.0, 1.5, 120)
    return np.concatenate([small, large])


def test_lloyd_max_reduces_mse_vs_uniform():
    x = _mixture_source()
    w = np.ones_like(x)
    lm = lloyd_max_quantizer(x, w, bits=4, llr_range=4.0, iters=300)
    mse_lm = float(np.mean((x - np.array([quantize_with(lm, v)
                                          for v in x])) ** 2))
    mse_unif = float(np.mean((x - np.array([quantize_llr(v, 4, 4.0)
                                            for v in x])) ** 2))
    assert mse_lm <= mse_unif


def test_lloyd_max_centroid_condition_unbiased():
    """The centroid condition makes the per-bin reconstruction error
    zero-mean (preserves the H1 drift of the belief)."""
    x = _mixture_source()
    w = np.ones_like(x)
    lm = lloyd_max_quantizer(x, w, bits=4, llr_range=4.0, iters=300)
    edges = lm["edges"]
    centroids = lm["centroids"]
    for k in range(len(centroids)):
        mask = (x >= edges[k]) & (x < edges[k + 1])
        if mask.sum() >= 3:
            err = float(np.mean(x[mask] - centroids[k]))
            assert abs(err) < 1e-6


def test_quantize_with_bin_mapping():
    lm = lloyd_max_quantizer(np.array([-3.0, -1.0, 0.0, 1.0, 3.0]),
                             np.ones(5), bits=3, llr_range=4.0)
    for v in (-3.5, -1.2, 0.1, 2.3, 3.9):
        q = quantize_with(lm, v)
        assert -4.0 <= q <= 4.0
    # value at a centroid maps to itself
    c = float(lm["centroids"][len(lm["centroids"]) // 2])
    assert quantize_with(lm, c) == pytest.approx(c, abs=1e-9)


def test_build_token_quantizer_on_scenario():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    lm = build_token_quantizer(sc, weight="h1")
    assert lm["bits"] == 5
    assert len(lm["centroids"]) == 32
    assert len(lm["edges"]) == 33
    # codebook covers the atom span
    all_atoms = np.concatenate([act["llr"]
                                for qq in range(3)
                                for act in sc["links"][qq]])
    assert np.min(all_atoms) >= lm["edges"][0] - 1e-9
    assert np.max(all_atoms) <= lm["edges"][-1] + 1e-9


def test_quantized_kernels_with_codebook():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    lm = build_token_quantizer(sc, weight="h1")
    qk = quantized_kernels(sc["links"][0], quantizer=lm)
    assert len(qk) == len(sc["links"][0])
    for orig, act in zip(sc["links"][0], qk):
        for v in act["llr"]:
            assert v in lm["centroids"]
        # masses unchanged, only the atoms are re-encoded
        assert np.array_equal(act["p0"], orig["p0"])
        assert np.array_equal(act["p1"], orig["p1"])


def test_uniform_quantizer_codebook_matches_quantize_llr():
    u = uniform_quantizer(bits=5, llr_range=6.0)
    assert len(u["centroids"]) == 32
    for v in (-5.9, -2.3, 0.05, 1.7, 5.9):
        assert quantize_with(u, v) == pytest.approx(
            quantize_llr(v, 5, 6.0), abs=1e-12)


def test_mu_law_fine_near_zero_and_monotone():
    m = mu_law_quantizer(bits=5, llr_range=4.0, mu=100.0)
    assert len(m["centroids"]) == 32
    # fine resolution near zero by construction
    near = np.abs(m["centroids"]) < 0.25
    assert near.sum() >= 8
    # monotone codebook, symmetric-ish, bounded by the range
    assert np.all(np.diff(m["centroids"]) > 0)
    assert np.min(m["centroids"]) >= -4.0
    assert np.max(m["centroids"]) <= 4.0
    # expansion of the companding law: small inputs stay small
    assert abs(quantize_with(m, 0.0)) < 0.1
    assert abs(quantize_with(m, 0.05)) < 0.2


def test_token_reallocation_layout_inside_budget():
    from uav_otfs_isac.distributed_audit import token_bits
    frozen = token_bits()["total"]
    reallocated = 2 + 10 + 2 + 4  # q + Lhat + intent + stamp
    assert reallocated <= frozen
    assert frozen == 19
