"""Tests for the post-communication detection-information layer.

Covers the KL-drift identities, the erasure scaling identity, the
data-processing contraction chain ``I_post <= I_quant <= I_sensing``,
the Chernoff bound relation, and the exact FFT-convolution sequential
detector against brute-force enumeration on small alphabets (i.i.d. and
heterogeneous sequences).
"""

import numpy as np
import pytest

from uav_otfs_isac.detection_information import (
    chernoff_information,
    contraction_chain,
    cycles_to_pd,
    discrete_kl,
    gaussian_kl,
    llr_pmf,
    post_communication_likelihoods,
    predicted_remaining_cycles,
    sequential_pd,
    sequential_pd_sequence,
    threshold_curve,
)
from uav_otfs_isac.reporting import quantizer_from_gaussian_range


def _random_link(rng, snr_db_range=(-3.0, 8.0)):
    l_acc = 4
    snr_db = float(rng.uniform(*snr_db_range))
    noncentrality = l_acc * 10 ** (snr_db / 10.0)
    mu0, var0 = float(l_acc), float(l_acc)
    mu1, var1 = mu0 + noncentrality, var0 + 2.0 * noncentrality
    bits = int(rng.integers(1, 6))
    flip = float(rng.uniform(0.01, 0.15))
    success = float(rng.uniform(0.5, 0.98))
    edges, values = quantizer_from_gaussian_range(
        np.array([mu0]), np.array([[var0]]),
        np.array([mu1]), np.array([[var1]]),
        bits,
    )
    return (mu0, var0, mu1, var1, edges, values, bits, flip, success)


def test_llr_drift_identities_hold_exactly():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu0, var0, mu1, var1, edges, values, bits, flip, success = (
            _random_link(rng)
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values, bits, flip, success,
        )
        llr, pmf1, pmf0 = llr_pmf(info["p1_y"], info["p0_y"])
        drift_h1 = float(pmf1 @ llr)
        drift_h0 = float(pmf0 @ llr)
        assert drift_h1 == pytest.approx(info["kl_plus"], abs=1e-9)
        assert drift_h0 == pytest.approx(-info["kl_minus"], abs=1e-9)


def test_erasure_scaling_identity():
    rng = np.random.default_rng(1)
    for _ in range(20):
        mu0, var0, mu1, var1, edges, values, bits, flip, success = (
            _random_link(rng)
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values, bits, flip, success,
        )
        assert info["kl_plus"] == pytest.approx(
            success * info["kl_rec_plus"], abs=1e-12
        )
        assert info["p0_y"][-1] == pytest.approx(1.0 - success, abs=1e-12)
        assert info["p1_y"][-1] == pytest.approx(1.0 - success, abs=1e-12)


def test_data_processing_contraction_chain():
    rng = np.random.default_rng(2)
    for _ in range(30):
        mu0, var0, mu1, var1, edges, values, bits, flip, success = (
            _random_link(rng)
        )
        chain = contraction_chain(
            mu0, var0, mu1, var1, edges, values, bits, flip, success,
        )
        assert chain["contraction_holds"]
        assert chain["i_quant"] >= 0.0
        assert chain["i_sensing"] >= 0.0


def test_chernoff_information_bounds_kldivergence():
    rng = np.random.default_rng(3)
    for _ in range(20):
        mu0, var0, mu1, var1, edges, values, bits, flip, success = (
            _random_link(rng)
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values, bits, flip, success,
        )
        c = info["chernoff"]
        assert 0.0 <= c <= min(info["kl_plus"], info["kl_minus"]) + 1e-12


def test_gaussian_kl_closed_form():
    assert gaussian_kl(1.0, 1.0, 0.0, 1.0) == pytest.approx(0.5, abs=1e-12)
    assert gaussian_kl(0.0, 4.0, 0.0, 1.0) == pytest.approx(
        0.5 * (np.log(0.25) + 3.0), abs=1e-12
    )


def test_discrete_kl_inf_when_support_missing():
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    assert discrete_kl(p, q) == float("inf")
    assert discrete_kl(q, p) == pytest.approx(np.log(2.0), abs=1e-12)


def test_llr_pmf_erasure_atom_is_zero_llr():
    rng = np.random.default_rng(3)
    p1, p0 = _two_symbol_pmfs(rng, 0.1)
    llr, pmf1, pmf0 = llr_pmf(p1, p0)
    assert np.any((p1 == p0) & (pmf1 > 0.0))
    assert np.all(llr[(pmf1 > 0.0) & (p1 == p0)] == 0.0)


def test_llr_pmf_rejects_one_sided_zero_atoms():
    # a symbol with mass under exactly one hypothesis has infinite LLR;
    # the uniform-grid machinery cannot represent it and must refuse
    # instead of silently zeroing it
    p1 = np.array([0.5, 0.5])
    p0 = np.array([0.25, 0.75])
    with pytest.raises(ValueError):
        llr_pmf(p1, np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        llr_pmf(np.array([0.0, 1.0]), p0)


def _bruteforce_sequential(pmfs, alpha):
    """Exact P_FA/P_D by full enumeration over a small alphabet.

    The threshold is the *smallest* accumulated LLR whose H0 upper tail
    is at most ``alpha`` (the same rule as the grid implementation).
    """
    outcomes = [(0.0, 1.0, 1.0)]  # (llr, prob under H0, prob under H1)
    for p1, p0 in pmfs:
        llr, pmf1, pmf0 = llr_pmf(p1, p0)
        new = []
        for l_acc, w0, w1 in outcomes:
            for l, a0, a1 in zip(llr, pmf0, pmf1):
                new.append((round(l_acc + l, 9), w0 * a0, w1 * a1))
        outcomes = new
    eta = None
    for l in sorted({o[0] for o in outcomes}):
        pfa = sum(w0 for ll, w0, _ in outcomes if ll > l)
        if pfa <= alpha:
            eta = l
            break
    if eta is None:
        eta = sorted({o[0] for o in outcomes})[0]
    pfa = sum(w0 for ll, w0, _ in outcomes if ll > eta)
    pd = sum(w1 for ll, _, w1 in outcomes if ll > eta)
    return eta, pfa, pd


def _two_symbol_pmfs(rng, noise):
    """Small-alphabet PMFs with two discriminative levels plus erasure."""
    p0 = np.array([1.0 - noise, noise / 2.0, noise / 2.0])
    p1 = np.array([1.0 - noise, noise / 4.0, 3.0 * noise / 4.0])
    p0 = p0 / p0.sum()
    p1 = p1 / p1.sum()
    return p1, p0


def test_sequential_pd_matches_bruteforce_iid():
    rng = np.random.default_rng(4)
    for _ in range(5):
        p1, p0 = _two_symbol_pmfs(rng, 0.3)
        alpha = 0.1
        for n in (1, 2, 3):
            eta_bf, pfa_bf, pd_bf = _bruteforce_sequential([(p1, p0)] * n, alpha)
            result = sequential_pd(p1, p0, n, alpha, grid_step=0.005)
            assert result["pfa"] == pytest.approx(pfa_bf, abs=0.02)
            assert result["pd"] == pytest.approx(pd_bf, abs=0.02)
            assert abs(result["threshold"] - eta_bf) <= 0.01


def test_sequential_pd_heterogeneous_matches_bruteforce():
    rng = np.random.default_rng(5)
    for _ in range(5):
        p1a, p0a = _two_symbol_pmfs(rng, 0.3)
        p1b, p0b = _two_symbol_pmfs(rng, 0.2)
        sequence = [(p1a, p0a), (p1b, p0b), (p1a, p0a)]
        alpha = 0.1
        eta_bf, pfa_bf, pd_bf = _bruteforce_sequential(sequence, alpha)
        result = sequential_pd_sequence(sequence, alpha, grid_step=0.005)
        assert result["pfa"] == pytest.approx(pfa_bf, abs=0.02)
        assert result["pd"] == pytest.approx(pd_bf, abs=0.02)
        assert abs(result["threshold"] - eta_bf) <= 0.01


def test_pd_grows_with_n():
    rng = np.random.default_rng(6)
    for _ in range(10):
        mu0, var0, mu1, var1, edges, values, bits, flip, success = (
            _random_link(rng)
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values, bits, flip, success,
        )
        pds = [
            float(sequential_pd(info["p1_y"], info["p0_y"], n, 0.05)["pd"])
            for n in range(1, 7)
        ]
        # Per-n adaptive thresholds are not pointwise monotone for
        # discrete PMFs; the power must grow over the horizon.
        assert pds[-1] >= pds[0] + 0.05
        assert all(a >= 0.0 and a <= 1.0 for a in pds)


def test_pfa_constraint_holds():
    rng = np.random.default_rng(7)
    for _ in range(15):
        mu0, var0, mu1, var1, edges, values, bits, flip, success = (
            _random_link(rng)
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values, bits, flip, success,
        )
        for n in (1, 4, 8):
            result = sequential_pd(info["p1_y"], info["p0_y"], n, 0.05)
            assert result["pfa"] <= 0.05 + 1e-12
            assert 0.0 <= result["pd"] <= 1.0


def test_cycles_to_pd_consistency():
    rng = np.random.default_rng(8)
    checked = 0
    for _ in range(30):
        mu0, var0, mu1, var1, edges, values, bits, flip, success = (
            _random_link(rng)
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values, bits, flip, success,
        )
        nstar = cycles_to_pd(info["p1_y"], info["p0_y"], 0.05, 0.9, max_n=10)
        if nstar is None:
            continue
        checked += 1
        pd_star = float(sequential_pd(
            info["p1_y"], info["p0_y"], nstar, 0.05,
        )["pd"])
        assert pd_star >= 0.9 - 1e-12
        if nstar > 1:
            pd_before = float(sequential_pd(
                info["p1_y"], info["p0_y"], nstar - 1, 0.05,
            )["pd"])
            assert pd_before < 0.9
    assert checked >= 5


def test_threshold_curve_shape_and_usage():
    rng = np.random.default_rng(9)
    mu0, var0, mu1, var1, edges, values, bits, flip, success = (
        _random_link(rng)
    )
    info = post_communication_likelihoods(
        mu0, var0, mu1, var1, edges, values, bits, flip, success,
    )
    curve = threshold_curve(info["p1_y"], info["p0_y"], 0.05, 6)
    assert curve.shape == (6,)
    assert np.all(np.isfinite(curve))
    tau = predicted_remaining_cycles(2, info["kl_plus"], curve[2])
    assert np.isfinite(tau)
    assert tau > 0.0 or np.isclose(
        tau, (curve[2] - 2.0 * info["kl_plus"]) / info["kl_plus"], atol=1e-9
    )


def test_predicted_remaining_cycles_inf_for_zero_information():
    assert predicted_remaining_cycles(3, 0.0, 1.0) == float("inf")


def test_llr_erasure_atom_contributes_zero():
    p0 = np.array([0.5, 0.5])
    p1 = np.array([0.7, 0.3])
    p0_y = np.array([0.4, 0.4, 0.2])
    p1_y = np.array([0.56, 0.24, 0.2])
    llr, pmf1, pmf0 = llr_pmf(p1_y, p0_y)
    erased = llr[np.argmin(np.abs(llr - 0.0))]
    assert erased == pytest.approx(0.0, abs=1e-12)
    assert np.sum(pmf1[llr == 0.0]) == pytest.approx(0.2, abs=1e-12)


def test_chernoff_zero_for_identical_distributions():
    p = np.array([0.5, 0.5])
    assert chernoff_information(p, p) == pytest.approx(0.0, abs=1e-12)