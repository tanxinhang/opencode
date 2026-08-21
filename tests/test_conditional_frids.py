"""Gate F0-G8B tests: conditional-reliable-information FRIDS (advice/013
section 7, advice/015 section 4)."""

import numpy as np
import pytest

from uav_otfs_isac.conditional_frids import (
    coalition_from_intents,
    conditional_gain,
    observation_delta_matrix,
    reliable_delta_matrix,
    simulate_frids_v2_cond,
)
from uav_otfs_isac.distributed_audit import (
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.evidence_correlation import singleton_kl
from uav_otfs_isac.frids import simulate_frids_v2


@pytest.fixture(scope="module")
def scenario():
    return build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=6, q_targets=3)


@pytest.fixture(scope="module")
def bounds(scenario):
    return calibrate_target_bounds(scenario, n_runs=60, seed=100,
                                   verify_runs=0)


def test_conditional_gain_rho0_equals_singleton():
    delta = np.array([1.0, 2.0, 1.5, 0.8])
    for i in range(4):
        others = [j for j in range(4) if j != i]
        dg = conditional_gain(delta, others, i, 0.0)
        assert dg == pytest.approx(singleton_kl(delta[i]), rel=1e-9)


def test_conditional_gain_reduces_with_correlation():
    """For a homogeneous coalition the conditional gain given the others
    is below the singleton when rho_s > 0 (the redundancy discount)."""
    delta = np.ones(4)
    others = [1, 2, 3]
    dg0 = conditional_gain(delta, others, 0, 0.0)
    dg05 = conditional_gain(delta, others, 0, 0.5)
    assert dg0 == pytest.approx(0.5)
    assert dg05 < dg0


def test_conditional_gain_empty_coalition_equals_singleton():
    delta = np.array([1.0, 2.0])
    dg = conditional_gain(delta, [], 0, 0.5)
    assert dg == pytest.approx(singleton_kl(delta[0]), rel=1e-9)


def test_coalition_from_intents():
    intents = np.full((4, 4), -1, dtype=int)
    intents[0, 1] = 2     # UAV0 received UAV1's intent for target 2
    intents[0, 2] = 2     # and UAV2's
    intents[0, 3] = 0
    c = coalition_from_intents(intents, 0, 2)
    assert set(c) == {0, 1, 2}
    assert 3 not in c
    assert 0 in c        # itself is always in the coalition


def test_delta_matrices():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    dr = reliable_delta_matrix(sc, sc["owner_of"])
    do_ = observation_delta_matrix(sc)
    assert dr.shape == (6, 3) and do_.shape == (6, 3)
    # the owner's reliable delta = its raw delta (delivery success 1)
    owner = sc["owner_of"][1]
    assert dr[owner, 1] == pytest.approx(do_[owner, 1], rel=1e-9)
    # reliable <= raw for non-owner links (s <= 1)
    assert np.all(dr <= do_ + 1e-12)


def test_cond_rho0_matches_frids_v2(scenario, bounds):
    """At rho_s = 0, world_rho = 0 the conditional scheduler must make
    IDENTICAL decisions to the frozen FRIDS-v2 (the sanity check)."""
    a = simulate_frids_v2(scenario, bounds, n_runs=80, seed=7,
                          max_steps=40)
    b = simulate_frids_v2_cond(scenario, bounds, n_runs=80, seed=7,
                               max_steps=40, rho_s=0.0, world_rho=0.0)
    assert a["worst_target_delay"] == b["worst_target_delay"]
    assert a["e1_delays"] == b["e1_delays"]
    assert a["p_fa"] == b["p_fa"]
    assert a["p_md"] == b["p_md"]


def test_cond_deterministic(scenario, bounds):
    a = simulate_frids_v2_cond(scenario, bounds, n_runs=80, seed=29,
                               max_steps=40, rho_s=0.5, world_rho=0.0)
    b = simulate_frids_v2_cond(scenario, bounds, n_runs=80, seed=29,
                               max_steps=40, rho_s=0.5, world_rho=0.0)
    assert a["worst_target_delay"] == b["worst_target_delay"]


def test_cond_runs_all_configs(scenario, bounds):
    for rho_s in (0.2, 0.5):
        for world_rho in (0.0, 0.2, 0.5):
            out = simulate_frids_v2_cond(scenario, bounds, n_runs=40,
                                         seed=3, max_steps=40,
                                         rho_s=rho_s, world_rho=world_rho)
            assert 0.0 < out["worst_target_delay"] <= 40.0
            for p in out["p_fa"] + out["p_md"]:
                assert 0.0 <= p <= 1.0


def test_cond_covariance_identity_equals_v2(scenario, bounds):
    """G8-C: with a diagonal (identity) covariance the Schur conditional
    gain equals the singleton, so the covariance-native scheduler reduces
    exactly to FRIDS-v2 (the independence reduction of Theorem 4.108)."""
    k = scenario["k"]
    cov = {qq: np.eye(k) for qq in range(scenario["q"])}
    a = simulate_frids_v2(scenario, bounds, n_runs=60, seed=13,
                          max_steps=40)
    b = simulate_frids_v2_cond(scenario, bounds, n_runs=60, seed=13,
                               max_steps=40, covariance=cov)
    assert a["worst_target_delay"] == b["worst_target_delay"]
    assert a["e1_delays"] == b["e1_delays"]


def test_cond_covariance_runs(scenario, bounds):
    """G8-C: covariance-native value + covariance-native world run and
    are deterministic."""
    from uav_otfs_isac.covariance_conditional import (
        build_profile_moments,
    )
    rng = np.random.default_rng(0)
    m = build_profile_moments(scenario["k"], scenario["q"],
                              "heterogeneous", rng)
    cov = {qq: m["sigma"] for qq in range(scenario["q"])}
    a = simulate_frids_v2_cond(scenario, bounds, n_runs=40, seed=3,
                               max_steps=40, covariance=cov,
                               world_covariance=cov)
    b = simulate_frids_v2_cond(scenario, bounds, n_runs=40, seed=3,
                               max_steps=40, covariance=cov,
                               world_covariance=cov)
    assert a["worst_target_delay"] == b["worst_target_delay"]
    assert 0.0 < a["worst_target_delay"] <= 40.0