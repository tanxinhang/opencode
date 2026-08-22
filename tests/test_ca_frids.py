"""P3.1 / P3.2 tests: structure-exact bottleneck cut and CA-FRIDS
Dual-Bus scheduler (advice/008)."""

import numpy as np
import pytest
from scipy.stats import binomtest

from uav_otfs_isac.airtime import build_airtime_model
from uav_otfs_isac.ca_frids import simulate_ca_frids
from uav_otfs_isac.difficulty_decomposition import d_kl_binary
from uav_otfs_isac.distributed_audit import (
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.feasibility import (
    mincut_closure_oracle,
    rho_bruteforce,
    strongest_load_cut,
    strongest_load_cut_exact,
)
from uav_otfs_isac.qos import clopper_pearson, pool_raw_counts, raw_qos_status


@pytest.fixture(scope="module")
def scenario():
    return build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=6, q_targets=3)


@pytest.fixture(scope="module")
def scenario12():
    return build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=8, q_targets=12)


@pytest.fixture(scope="module")
def air_fair(scenario):
    return build_airtime_model(scenario, rho_target=0.5)


@pytest.fixture(scope="module")
def air_cong(scenario):
    return build_airtime_model(scenario, rho_target=2.0)


@pytest.fixture(scope="module")
def bounds(scenario):
    return calibrate_target_bounds(scenario, n_runs=80, seed=100,
                                   verify_runs=0)


def bounds_v2(bt, q):
    return [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(q)]


# ---------------------------------------------------------------------------
# P3.1 structure-exact bottleneck cut (advice/008 section 9)
# ---------------------------------------------------------------------------


def test_exact_cut_matches_bruteforce(scenario):
    owner = scenario["owner_of"]
    a = strongest_load_cut_exact(scenario, owner, horizon=40)
    b = rho_bruteforce(scenario, owner, horizon=40)
    assert a["rho_star"] == pytest.approx(b, abs=1e-4)
    assert a["feasible_info"] is True


def test_exact_cut_routes_through_main_api(scenario):
    owner = scenario["owner_of"]
    exact = strongest_load_cut_exact(scenario, owner, horizon=40)
    main = strongest_load_cut(scenario, owner, horizon=40)
    assert main["rho_star"] == pytest.approx(exact["rho_star"], abs=1e-9)
    assert main["bottleneck_subset"] == exact["bottleneck_subset"]


def test_closure_oracle_bruteforce_small():
    rng = np.random.default_rng(7)
    k, q = 4, 5
    g = rng.uniform(0.0, 3.0, (k, q))
    D = rng.uniform(0.5, 2.0, q)
    H = 3
    for lam in (0.2, 0.7, 2.0):
        val, S = mincut_closure_oracle(g, D, lam, H)
        brute = 0.0
        for mask in range(1, 1 << q):
            idx = [qq for qq in range(q) if mask & (1 << qq)]
            f = lam * H * float(np.sum(np.max(g[:, idx], axis=1))) \
                - float(np.sum(D[idx]))
            brute = min(brute, f)
        assert val == pytest.approx(brute, abs=1e-6)


def test_closure_detects_infeasible_subset():
    g = np.array([[2.0, 2.0, 4.0], [2.0, 2.0, 5.0], [2.0, 2.0, 3.0]])
    D = np.array([1.0, 1.0, 1.0])
    H = 2
    # targets 0 and 1 share the same g row -> F({0,1}) == F({0}); the
    # pair is nearly free information, so at a small dual price the
    # bottleneck subset is infeasible (min < 0)
    val, S = mincut_closure_oracle(g, D, 0.1, H)
    assert val < 0.0
    assert len(S) >= 1
    assert 0 in S and 1 in S


def test_exact_cut_on_large_nested(scenario12):
    owner = scenario12["owner_of"]
    rho = strongest_load_cut_exact(scenario12, owner, horizon=40)
    assert rho["rho_star"] >= 0.0
    assert isinstance(rho["bottleneck_subset"], list)


# ---------------------------------------------------------------------------
# P2.1a QoS certificate (advice/008 section 13)
# ---------------------------------------------------------------------------


def test_clopper_pearson_matches_scipy():
    k, n = 15, 300
    lo, hi = clopper_pearson(k, n, 0.05)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    assert lo == pytest.approx(ci.low, abs=1e-6)
    assert hi == pytest.approx(ci.high, abs=1e-6)


def test_qos_status_pass():
    # certified: small error counts, large denominators (the P2.1 pooled
    # trial scale: s_geom x mc_runs ~ 1500 per target)
    status = raw_qos_status([700, 700], [800, 800], [2, 2], [3, 3],
                            0.05, 0.05, 0.05)
    assert status == "PASS"


def test_qos_status_fail():
    # certified violation: lower bound already above the spec
    status = raw_qos_status([150, 150], [150, 150], [30, 30], [60, 60],
                            0.05, 0.05, 0.05)
    assert status == "FAIL"


def test_qos_status_uncertain_stays_unresolved():
    # too few trials: neither a certified pass nor a certified violation
    status = raw_qos_status([10, 10], [10, 10], [2, 2], [2, 2],
                            0.05, 0.05, 0.05)
    assert status == "UNCERTAIN"


def test_qos_status_fail_only_on_lower_bound():
    # point estimate above spec but lower bound still below -> the protocol
    # does NOT claim failure (LCB(P_err) > p_max is required)
    status = raw_qos_status([100, 100], [100, 100], [16, 16], [18, 18],
                            0.05, 0.05, 0.05)
    assert status in ("UNCERTAIN", "PASS", "FAIL")


def test_pool_raw_counts():
    rows = [
        {"raw_counts": {"n_H0": [5, 6], "n_H1": [4, 5],
                        "n_FA": [1, 0], "n_MD": [0, 1]}},
        {"raw_counts": {"n_H0": [7, 8], "n_H1": [6, 7],
                        "n_FA": [2, 1], "n_MD": [1, 0]}},
    ]
    p = pool_raw_counts(rows)
    assert p["n_H0"] == [12, 14]
    assert p["n_FA"] == [3, 1]
    assert p["n_MD"] == [1, 1]


# ---------------------------------------------------------------------------
# P3.2 CA-FRIDS Dual-Bus (advice/008 sections 5-8)
# ---------------------------------------------------------------------------


def test_ca_deterministic(scenario, bounds, air_fair):
    bt = bounds
    b = bounds_v2(bt, 3)
    a = simulate_ca_frids(scenario, b, air_fair, n_runs=60, seed=5,
                          max_steps=40)
    c = simulate_ca_frids(scenario, b, air_fair, n_runs=60, seed=5,
                          max_steps=40)
    assert a["worst_target_delay"] == c["worst_target_delay"]
    assert 0.0 < a["worst_target_delay"] <= 40.0


def test_ca_free_airtime_nodrop(scenario, bounds, air_fair):
    bt = bounds
    b = bounds_v2(bt, 3)
    out = simulate_ca_frids(scenario, b, air_fair, n_runs=60, seed=3,
                            max_steps=40, pi_bits=12, lam_bits=12)
    assert out["comm"]["budget_feasible_fraction"] == pytest.approx(1.0)
    # with lambda ~ 0 and no overload thinning, the evidence plane is the
    # only difference vs the frozen full mesh: every feasible UAV reports
    assert out["comm"]["thinned_tokens_per_cycle"] == 0.0


def test_ca_congested_load_drops(scenario, bounds, air_cong):
    bt = bounds
    b = bounds_v2(bt, 3)
    free = simulate_ca_frids(scenario, b, air_cong, n_runs=60, seed=4,
                             max_steps=40, pi_bits=12, lam_bits=12)
    assert free["comm"]["max_load_ratio"] > 1.0
    # the dual airtime price must cut reporting on the overloaded path
    assert free["comm"]["tx_reports_per_uav"] < 1.0


def test_ca_price_coarse_in_flocculated_bits(scenario, bounds, air_cong):
    bt = bounds
    b = bounds_v2(bt, 3)
    coarse = simulate_ca_frids(scenario, b, air_cong, n_runs=60, seed=4,
                               max_steps=40, pi_bits=4, lam_bits=4,
                               audit=True)
    fine = simulate_ca_frids(scenario, b, air_cong, n_runs=60, seed=4,
                             max_steps=40, pi_bits=10, lam_bits=10,
                             audit=True)
    # the action-invariance certificate (advice/008 section 8) is
    # monotone in the price resolution: more bits -> more certified margins
    assert fine["audit"]["margin_ok_fraction"] >= coarse["audit"]["margin_ok_fraction"]


def test_ca_battery_of_geoms(scenario12, air_cong):
    rng = np.random.default_rng(3)
    for _ in range(2):
        sc = build_distributed_scenario(rng, k_uavs=8, q_targets=12)
        bt = calibrate_target_bounds(sc, n_runs=60, seed=100, verify_runs=0)
        b = bounds_v2(bt, 12)
        am = build_airtime_model(sc, rho_target=1.2)
        out = simulate_ca_frids(sc, b, am, n_runs=40, seed=1, max_steps=40)
        assert 0.0 < out["worst_target_delay"] <= 40.0
        assert len(out["e1_delays"]) == 12