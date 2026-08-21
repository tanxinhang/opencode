"""Gate F0-G7 tests: physical airtime reporting (advice/013)."""

import numpy as np
import pytest

from uav_otfs_isac.airtime import (
    build_airtime_model,
    link_rate_from_snr,
    oracle_admission,
    overflow_survival,
    receive_load,
    report_score,
    simulate_frids_v2_air,
    snr_from_outage_success,
    token_airtime,
    update_airtime_price,
)
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)


@pytest.fixture(scope="module")
def scenario():
    return build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=6, q_targets=3)


@pytest.fixture(scope="module")
def bounds(scenario):
    return calibrate_target_bounds(scenario, n_runs=60, seed=100,
                                   verify_runs=0)


# ---------------------------------------------------------------------------
# Physical airtime math
# ---------------------------------------------------------------------------


def test_link_rate_is_capacity_upper_bound():
    r0 = link_rate_from_snr(0.0, 1e6)
    r1 = link_rate_from_snr(10.0, 1e6)
    r2 = link_rate_from_snr(20.0, 1e6)
    assert r0 < r1 < r2
    # Shannon capacity: W log2(1 + gamma)
    assert r1 == pytest.approx(1e6 * np.log2(11.0), rel=1e-9)
    assert r0 == pytest.approx(1e6, rel=1e-9)


def test_token_airtime_decreases_with_rate():
    a1 = token_airtime(17, 1e6)
    a2 = token_airtime(17, 4e6)
    assert a2 == pytest.approx(a1 / 4.0, rel=1e-9)
    assert a1 > 0.0


def test_snr_coupling_monotone_in_success():
    s_lo = snr_from_outage_success(0.2)
    s_mid = snr_from_outage_success(0.5)
    s_hi = snr_from_outage_success(0.9)
    assert s_lo < s_mid < s_hi
    # success 0.5 -> SNR = threshold (Phi(0) = 0.5)
    assert s_mid == pytest.approx(5.0, abs=1e-9)


def test_build_airtime_model_rho_target(scenario):
    am = build_airtime_model(scenario, rho_target=0.5)
    assert am["rho_full"] == pytest.approx(0.5, rel=1e-3)
    am2 = build_airtime_model(scenario, rho_target=1.5)
    assert am2["rho_full"] == pytest.approx(1.5, rel=1e-3)
    # tighter budget -> smaller T_air
    assert am2["t_air"] < am["t_air"]
    # c_air = tau / T_air
    assert np.allclose(am["c_air"], am["tau"] / am["t_air"])
    # self links carry no airtime
    assert np.all(np.diag(am["tau"]) == 0.0)


def test_receive_load_and_overflow_survival():
    tau = np.array([[0.0, 1.0, 2.0],
                    [3.0, 0.0, 4.0],
                    [5.0, 6.0, 0.0]])
    load = receive_load([0, 1], tau, 3)
    assert load == pytest.approx([3.0, 1.0, 6.0])   # receivers 0,1,2
    surv = overflow_survival(load, t_air=3.0)
    assert surv == pytest.approx([1.0, 1.0, 0.5])
    assert np.all(surv <= 1.0)


def test_price_dual_ascent():
    lam = np.array([0.0, 0.0])
    # over-budget load -> price rises
    lam = update_airtime_price(lam, np.array([1.5, 0.5]) * 10.0, 10.0, 0.2)
    assert lam[0] == pytest.approx(0.2 * 0.5, abs=1e-12)
    assert lam[1] == pytest.approx(0.0, abs=1e-12)
    # under-budget load -> price decays to 0 (clipped)
    lam = update_airtime_price(lam, np.array([0.0, 0.0]), 10.0, 0.2)
    assert lam[0] == pytest.approx(0.0, abs=1e-12)
    # cap
    lam2 = update_airtime_price(np.array([1.9, 0.0]), np.array([30.0, 0.0]),
                                10.0, 0.2, lam_cap=2.0)
    assert lam2[0] == pytest.approx(2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The no-op Lemma (advice/013 section 2)
# ---------------------------------------------------------------------------


def test_common_price_without_idle_is_noop():
    """A common additive price c_i over the target choices never changes
    the argmax (the F0-G5 scale-invariance phenomenon restated for the
    communication price)."""
    J = np.array([0.5, 0.3, 0.2])
    c = 0.1
    for lam in (0.1, 0.5, 2.0, 10.0):
        assert np.argmax(J - lam * c) == np.argmax(J)


def test_price_with_idle_changes_decision():
    """With the no-report (idle) option the price decides report vs
    silence: z = 1{max_q (J_q - lam * c_q) > 0}."""
    J = np.array([0.5, 0.3, 0.2])
    c = np.array([0.1, 0.1, 0.1])
    assert report_score(1.0, 0.5, 1.0, 0.1, 0.0, c[0]) > 0.0
    assert report_score(1.0, 0.5, 1.0, 0.1, 5.0, c[0]) < 0.0
    # heterogeneous c_q can also reorder the sensing target
    assert np.argmax(J - 2.0 * c) == np.argmax(J - 2.0 * np.array([0.0, 0.4, 0.0]))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def test_air_sim_deterministic(scenario, bounds):
    am = build_airtime_model(scenario, rho_target=1.0)
    a = simulate_frids_v2_air(scenario, bounds, am, n_runs=60, seed=5,
                              max_steps=40)
    b = simulate_frids_v2_air(scenario, bounds, am, n_runs=60, seed=5,
                              max_steps=40)
    assert a["worst_target_delay"] == b["worst_target_delay"]
    assert 0.0 < a["worst_target_delay"] <= 40.0
    for p in a["p_fa"] + a["p_md"]:
        assert 0.0 <= p <= 1.0


def test_air_sim_all_modes_run(scenario, bounds):
    am = build_airtime_model(scenario, rho_target=1.0)
    for mode in ("always", "value", "random", "periodic", "oracle"):
        out = simulate_frids_v2_air(scenario, bounds, am, n_runs=40,
                                    seed=1, max_steps=40,
                                    report_mode=mode, p=0.7, period=2)
        assert 0.0 < out["worst_target_delay"] <= 40.0
        assert 0.0 <= out["comm"]["tx_reports_per_uav"] <= 1.0
        assert out["comm"]["airtime_per_cycle"] >= 0.0


def test_value_with_zero_price_equals_always_in_non_congested(scenario, bounds):
    """With lambda_base = 0 and the budget not binding, the dual stays 0
    and the value-triggered gate admits every UAV -- identical to the
    frozen always-report mainline (the gate is a no-op when airtime is
    free, which is the honest non-congested behaviour)."""
    am = build_airtime_model(scenario, rho_target=0.5)
    always = simulate_frids_v2_air(scenario, bounds, am, n_runs=60, seed=3,
                                   max_steps=40, report_mode="always")
    value = simulate_frids_v2_air(scenario, bounds, am, n_runs=60, seed=3,
                                  max_steps=40, report_mode="value",
                                  lambda_base=0.0)
    assert value["worst_target_delay"] == always["worst_target_delay"]
    assert value["comm"]["tx_reports_per_uav"] == pytest.approx(
        always["comm"]["tx_reports_per_uav"])


def test_congested_value_reduces_load(scenario, bounds):
    """In the congested regime (rho_full > 1) the value-triggered gate
    must cut the committed load and the report rate below the always-
    report mainline."""
    am = build_airtime_model(scenario, rho_target=1.5)
    always = simulate_frids_v2_air(scenario, bounds, am, n_runs=60, seed=4,
                                   max_steps=40, report_mode="always")
    value = simulate_frids_v2_air(scenario, bounds, am, n_runs=60, seed=4,
                                  max_steps=40, report_mode="value",
                                  lambda_base=0.05)
    assert am["rho_full"] > 1.0
    assert value["comm"]["rx_load_per_uav"] < always["comm"]["rx_load_per_uav"]
    assert value["comm"]["tx_reports_per_uav"] < always["comm"]["tx_reports_per_uav"]
    # the value-triggered frame is budget-feasible a positive fraction of
    # cycles, while the always-report frame is never feasible
    assert value["comm"]["budget_feasible_fraction"] > 0.0
    assert always["comm"]["budget_feasible_fraction"] == pytest.approx(0.0)


def test_oracle_admission_respects_budget():
    tau = np.array([[0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]])
    values = np.array([10.0, 1.0, 1.0])
    z = oracle_admission(values, tau, t_air=2.0)
    # receiver 0 would see 1+1 = 2 if both 1 and 2 admitted -> budget binds
    assert bool(z[0])
    assert np.sum(z) >= 1
    load = receive_load([i for i in range(3) if z[i]], tau, 3)
    assert np.max(load) <= 2.0 + 1e-9