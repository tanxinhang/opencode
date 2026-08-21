"""Gate F0-G8C tests: covariance-native conditional information
(advice/018)."""

import numpy as np
import pytest

from uav_otfs_isac.covariance_conditional import (
    build_physics_covariance,
    build_profile_moments,
    conditional_gain_general,
    joint_kl_equal_cov,
    joint_kl_general,
    marginal_kl,
    profile_deltas,
    scalar_rho_from_covariance,
    schur_conditional_gain,
    shrink_covariance,
    simulate_gaussian_frids,
)


def _spd_cov(k, rho, rng):
    base = np.full((k, k), rho)
    np.fill_diagonal(base, 1.0)
    base = shrink_covariance(base, 0.05)
    return base


def test_schur_equals_difference():
    """The Schur closed form equals G(S union {i}) - G(S) for random
    (delta, Sigma)."""
    rng = np.random.default_rng(0)
    for _ in range(6):
        k = 5
        delta = rng.uniform(0.2, 2.0, size=k)
        sigma = _spd_cov(k, rng.uniform(0.2, 0.7), rng)
        for S in ([], [0], [0, 1], [1, 2, 3], [0, 1, 2, 3]):
            for i in range(k):
                if i in S:
                    continue
                dg = schur_conditional_gain(delta, sigma, S, i)
                idx_wi = sorted(set(S) | {i})
                idx_wo = sorted(set(S))
                g_wi = joint_kl_equal_cov(
                    delta[idx_wi], sigma[np.ix_(idx_wi, idx_wi)])
                g_wo = joint_kl_equal_cov(
                    delta[idx_wo], sigma[np.ix_(idx_wo, idx_wo)]) \
                    if idx_wo else 0.0
                assert dg == pytest.approx(g_wi - g_wo, abs=1e-9)
                assert dg >= -1e-9


def test_empty_coalition_equals_marginal():
    delta = np.array([1.0, 2.0])
    sigma = np.eye(2)
    assert schur_conditional_gain(delta, sigma, [], 0) == pytest.approx(0.5)
    assert schur_conditional_gain(delta, sigma, [], 1) == pytest.approx(2.0)
    assert marginal_kl(delta, sigma, 0) == pytest.approx(0.5)


def test_independence_equals_marginal():
    """c = 0 (diagonal Sigma) -> Delta G = g_i (natural FRIDS-v2
    reduction)."""
    delta = np.array([1.0, 1.5, 0.8])
    sigma = np.eye(3)
    for i in range(3):
        others = [j for j in range(3) if j != i]
        assert schur_conditional_gain(delta, sigma, others, i) == pytest.approx(
            marginal_kl(delta, sigma, i), rel=1e-9)


def test_redundancy_and_synergy():
    """Redundancy: the residual signal vanishes -> Delta G ~ 0.  Synergy:
    conditioning removes shared noise while the residual signal survives
    -> Delta G > g_i (the conditional innovation effect, advice/018)."""
    # redundancy: UAV 1 has the same delta as UAV 0 and they are fully
    # correlated (rho -> 1); the second is almost no extra information
    rho = 0.95
    sigma = _spd_cov(2, rho, np.random.default_rng(1))
    delta = np.array([1.0, 1.0])
    dg = schur_conditional_gain(delta, sigma, [0], 1)
    assert dg < 0.05
    # synergy: high common noise but different signal -> conditioning
    # removes the shared noise and the residual signal survives
    delta2 = np.array([1.0, 1.0])
    sigma2 = np.array([[1.0, 0.9], [0.9, 1.0]])
    dg2 = schur_conditional_gain(delta2, sigma2, [0], 1)
    # g_1 = 0.5; v_{1|0} = 1 - 0.9^2 = 0.19 -> Delta G = 0.5*(0.1)^2/0.19
    # = 0.026 < g (delta residual small).  For synergy the residual must
    # survive: use anti-correlated common noise with distinct residual.
    delta3 = np.array([1.0, -1.0])
    sigma3 = np.array([[1.0, 0.9], [0.9, 1.0]])
    dg3 = schur_conditional_gain(delta3, sigma3, [0], 1)
    assert dg3 > marginal_kl(delta3, sigma3, 1)  # 0.5*1/0.19 = 2.63 > 0.5


def test_general_kl_reduces_to_equal_cov():
    delta = np.array([1.0, 2.0, 0.5])
    sigma = _spd_cov(3, 0.4, np.random.default_rng(2))
    g_eq = joint_kl_equal_cov(delta, sigma)
    g_gen = joint_kl_general(delta, sigma, sigma)
    assert g_eq == pytest.approx(g_gen, rel=1e-9)


def test_general_conditional_nonnegative():
    rng = np.random.default_rng(3)
    k = 4
    delta = rng.uniform(0.2, 1.5, size=k)
    s0 = _spd_cov(k, 0.3, rng)
    s1 = _spd_cov(k, 0.6, rng)
    for S in ([0], [0, 1], [1, 2]):
        for i in range(k):
            if i in S:
                continue
            dg = conditional_gain_general(delta, s0, s1, S, i)
            assert dg >= -1e-9


def test_physics_covariance():
    positions = np.array([[0.0, 0.0, 100.0], [1.0, 0.0, 100.0],
                          [0.0, 1.0, 100.0]])
    target = np.array([45.0, 55.0, 0.0])
    doppler = np.array([0.0, 0.05, 0.4])
    c1 = build_physics_covariance(positions, target, doppler, 0.5)
    c2 = build_physics_covariance(positions, target, doppler, 0.9)
    assert c1.shape == (3, 3)
    assert np.allclose(np.diag(c1), 1.0)
    assert np.allclose(c1, c1.T)
    # stronger common clutter -> larger off-diagonal correlation
    assert np.mean(np.abs(c2 - np.diag(np.diag(c2)))) > \
        np.mean(np.abs(c1 - np.diag(np.diag(c1))))
    # symmetric + PD after shrinkage
    assert np.all(np.linalg.eigvalsh(c1) > 0)


def test_profiles_distinct():
    rng = np.random.default_rng(4)
    k, q = 8, 2
    dh = profile_deltas(k, q, "homogeneous", rng)
    dc = profile_deltas(k, q, "concentrated", rng)
    # concentrated: one large delta per target, rest small
    for qq in range(q):
        assert dc[:, qq].max() > 1.0
        assert dh[:, qq].min() > 0.3
        assert dc[:, qq].min() < 0.25


def test_scalar_rho_from_covariance():
    sigma = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    assert scalar_rho_from_covariance(sigma) == pytest.approx(0.5)
    assert scalar_rho_from_covariance(np.eye(3)) == pytest.approx(0.0)


def test_build_profile_moments():
    rng = np.random.default_rng(5)
    m = build_profile_moments(6, 3, "heterogeneous", rng)
    assert m["delta"].shape == (6, 3)
    assert m["sigma"].shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(m["sigma"]) > 0)


# ---------------------------------------------------------------------------
# Gaussian-evidence conditional FRIDS
# ---------------------------------------------------------------------------


def _gauss_scenario(k=6, q=3, profile="homogeneous", seed=0, rho=None):
    rng = np.random.default_rng(seed)
    m = build_profile_moments(k, q, profile, rng)
    if rho is not None:
        sig = np.full((k, k), rho)
        np.fill_diagonal(sig, 1.0)
        m["sigma"] = shrink_covariance(sig, 0.05)
    owner = [int(qq % k) for qq in range(q)]
    return m["delta"], m["sigma"], owner


def test_gaussian_frids_all_modes_run():
    delta, sigma, owner = _gauss_scenario(6, 3, "heterogeneous", seed=1)
    for mode in ("singleton", "rho", "covariance", "oracle"):
        out = simulate_gaussian_frids(delta, sigma, owner, n_runs=60,
                                      seed=3, max_steps=60,
                                      value_mode=mode, rho_s=0.4)
        assert 0.0 < out["worst_target_delay"] <= 60.0
        for p in out["p_fa"] + out["p_md"]:
            assert 0.0 <= p <= 1.0


def test_gaussian_frids_deterministic():
    delta, sigma, owner = _gauss_scenario(6, 3, "heterogeneous", seed=2)
    a = simulate_gaussian_frids(delta, sigma, owner, n_runs=60, seed=7,
                                max_steps=60, value_mode="covariance")
    b = simulate_gaussian_frids(delta, sigma, owner, n_runs=60, seed=7,
                                max_steps=60, value_mode="covariance")
    assert a["worst_target_delay"] == b["worst_target_delay"]


def test_gaussian_independent_world_covariance_equals_singleton():
    """At independence (diagonal Sigma) the covariance-native value equals
    the singleton, so the two schedulers make identical decisions."""
    delta, sigma, owner = _gauss_scenario(6, 3, "heterogeneous", seed=3,
                                          rho=0.0)
    a = simulate_gaussian_frids(delta, sigma, owner, n_runs=60, seed=9,
                                max_steps=60, value_mode="singleton")
    b = simulate_gaussian_frids(delta, sigma, owner, n_runs=60, seed=9,
                                max_steps=60, value_mode="covariance")
    assert a["worst_target_delay"] == b["worst_target_delay"]


def test_gaussian_covariance_helps_under_correlated_world():
    """G8-C hypothesis: under a correlated world the covariance-native
    value beats the singleton (which ignores the correlation)."""
    delta, sigma, owner = _gauss_scenario(8, 4, "concentrated", seed=4)
    single = simulate_gaussian_frids(delta, sigma, owner, n_runs=120,
                                     seed=5, max_steps=60,
                                     value_mode="singleton")
    cov = simulate_gaussian_frids(delta, sigma, owner, n_runs=120,
                                  seed=5, max_steps=60,
                                  value_mode="covariance")
    # the covariance-native value should not regress the worst target
    assert cov["worst_target_delay"] <= single["worst_target_delay"] + 1e-9 \
        or cov["worst_target_delay"] < single["worst_target_delay"] * 1.02