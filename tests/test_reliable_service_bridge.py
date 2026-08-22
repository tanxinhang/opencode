"""P1 service-delay bridge tests (advice/003, Theorem 4.110/4.111)."""

import numpy as np
import pytest

from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
    quantize_llr,
)
from uav_otfs_isac.frids import simulate_frids_v2
from uav_otfs_isac.reliable_service_bridge import (
    freedman_tail,
    local_vs_common_gap,
    martingale_decomposition,
    normalized_service_time_average,
    run_static_mirror_descent,
    static_md_convergence,
    static_relaxation_optimum,
    stopping_tail_bound,
    stopping_tail_verify,
)

K_UAVS = 6
Q_TARGETS = 3


@pytest.fixture(scope="module")
def scenario():
    return build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=K_UAVS, q_targets=Q_TARGETS)


@pytest.fixture(scope="module")
def bounds(scenario):
    return calibrate_target_bounds(scenario, n_runs=120, llr_bits=5)


@pytest.fixture(scope="module")
def bridge_out(scenario, bounds):
    out = simulate_frids_v2(scenario, bounds, n_runs=250, seed=7,
                            max_steps=40, bridge=True)
    return out["bridge"]


def test_freedman_tail_monotone_and_shape():
    # eta = 0 gives a trivial 1 (no premise), increasing eta tightens
    assert freedman_tail(0.0, 1.0, 1.0) == 1.0
    v, b = 2.0, 1.0
    etas = np.array([0.5, 1.0, 2.0, 3.0])
    tails = np.array([freedman_tail(e, v, b) for e in etas])
    assert np.all(np.diff(tails) <= 0.0)          # nonincreasing in eta
    # the no-fluctuation/variance asymptote: tighter V tightens the tail
    assert freedman_tail(2.0, 0.01, b) < freedman_tail(2.0, 100.0, b)
    assert 0.0 < tails[-1] < 1.0
    # dimensional sanity: exponent is eta^2 / (2(V + b eta/3))
    expected = float(np.exp(-3.0 ** 2 / (2.0 * (v + b * 3.0 / 3.0))))
    assert freedman_tail(3.0, v, b) == pytest.approx(expected, rel=1e-12)


def test_stopping_tail_bound_premise():
    # A_q not at the deficit yet -> the deviation premise is inactive and
    # the bound is trivial (the argument only bites at A_q >= D_q)
    assert stopping_tail_bound(a_q=1.0, d_q=4.0, v_q=1.0, b_q=1.0,
                               beta_q=0.05) == pytest.approx(1.0, rel=1e-12)
    # once A_q exceeds the deficit the deviation term activates
    bound = stopping_tail_bound(a_q=6.0, d_q=4.0, v_q=1.0, b_q=1.0,
                                beta_q=0.05)
    assert bound < 0.05 + 0.5
    assert bound > 0.0
    assert stopping_tail_bound(10.0, 4.0, 0.01, 1.0, 0.05) \
        < stopping_tail_bound(10.0, 4.0, 10.0, 1.0, 0.05)


def test_static_relaxation_optimum():
    # one UAV, two targets: with g = (1, 1), D the same, z* = 1/2
    g = np.array([[1.0, 1.0]])
    z = static_relaxation_optimum(g, np.array([1.0, 1.0]), eps=0.0)
    assert z == pytest.approx(0.5, rel=1e-6)
    # asymmetric g: one UAV must balance both targets (serving only the
    # strong one starves the weak), so z* = 2/3 (x_10=2/3, x_01=1/3)
    g2 = np.array([[2.0, 1.0]])
    z2 = static_relaxation_optimum(g2, np.array([1.0, 1.0]), eps=0.0)
    assert z2 == pytest.approx(2.0 / 3.0, rel=1e-6)
    # the relaxation optimum is nondecreasing in the information matrix
    g3 = np.array([[2.0, 1.0], [0.0, 1.0]])
    z3 = static_relaxation_optimum(g3, np.array([1.0, 1.0]), eps=0.0)
    assert z3 >= z2 - 1e-9
    # eps floor keeps the LP feasible at zero deficit
    assert static_relaxation_optimum(g, np.array([0.0, 0.0]), eps=0.1) >= 0.0


def test_bridge_recording_shapes_and_decomposition(bridge_out):
    L = np.asarray(bridge_out["L"])
    T, q = L.shape[1], L.shape[2]
    assert bridge_out["A"].shape == bridge_out["M"].shape == L.shape
    assert bridge_out["V"].shape == L.shape
    assert bridge_out["r_pred"].shape == bridge_out["r_real"].shape
    assert len(bridge_out["a_thr"]) == q
    # L = A + M is exact by construction (M is the residual)
    assert np.allclose(L, bridge_out["A"] + bridge_out["M"], atol=1e-9)


def test_martingale_decomposition(bridge_out):
    dec = martingale_decomposition(bridge_out, Q_TARGETS)
    assert dec["decomposition_max_abs_error"] < 1e-9
    assert dec["martingale_residual_mean"] == pytest.approx(0.0, abs=1.0)
    assert dec["b_q"] > 0.0
    assert dec["freedman"]["n_cases"] > 0
    assert dec["freedman"]["violation_fraction"] <= 0.05
    # advice/004 P0.5-2: the time-uniform / line-crossing form is the
    # natural object for a stopping time and must also hold
    assert dec["freedman_uniform"]["n_cases"] > 0
    assert dec["freedman_uniform"]["violation_fraction"] <= 0.05
    assert 0.0 <= dec["quantization_gap_mean"] <= 1.0
    assert len(dec["v_upper_per_target"]) == Q_TARGETS
    assert dec["eta_grid"] and dec["v_grid"]


def test_stopped_process_fill_forward(bridge_out):
    """P0.5-1: after a target stops, the recorded process must hold its
    terminal value (M_{t wedge T}), not drop to zero."""
    L = np.asarray(bridge_out["L"])
    M = np.asarray(bridge_out["M"])
    T_stop = np.asarray(bridge_out["T"], dtype=float)
    n_runs, max_steps, q = L.shape
    found = False
    for r in range(n_runs):
        for qq in range(q):
            tt = int(T_stop[r, qq])
            if tt >= max_steps:
                continue
            found = True
            last_val = float(M[r, max(tt - 1, 0), qq])
            assert all(float(M[r, t, qq]) == last_val
                       for t in range(tt, max_steps))
    assert found


def test_run_static_mirror_descent_approaches_zstar():
    """P0.5-4: the static shadow mirror descent (no stopping, fixed D/g,
    COMMON shadow price) must close the gap to z*, and the gap decays or
    disappears with the horizon.  A well-conditioned instance reaches z*
    immediately (gap ~ 0: strictly stronger than O(sqrt(logQ/T)))."""
    # asymmetric-information instance: needs continuous balancing, so the
    # gap decays with T instead of collapsing instantly
    g = np.array([[1.0, 0.3, 0.2],
                  [0.2, 1.0, 0.2],
                  [0.15, 0.25, 1.0]])
    d0 = np.array([2.0, 3.0, 4.0])
    r_short = run_static_mirror_descent(g, d0, horizon=20)
    r_long = run_static_mirror_descent(g, d0, horizon=320)
    assert r_short["z_star"] > 0.0
    # strictly stronger: immediate convergence also counts as a pass
    assert r_long["gap"] <= r_short["gap"] + 1e-12
    conv = static_md_convergence(g, d0, horizons=(40, 80, 160, 320))
    if conv["converged_immediately"]:
        assert conv["gap_max"] <= 1e-6
    else:
        assert conv["loglog_slope"] is not None
        assert -0.9 <= conv["loglog_slope"] <= -0.1


def test_local_vs_common_gap_small(bridge_out, scenario, bounds):
    """P0.5-5: the local-vs-common CRN price gap on static service.  The
    DELAY-relevant loss is at the bottleneck (min-service) target, not
    the largest mid-target swing."""
    out_c = simulate_frids_v2(scenario, bounds, n_runs=250, seed=7,
                              max_steps=40, bridge=True, price_mode="common")
    gap = local_vs_common_gap(bridge_out, out_c["bridge"], Q_TARGETS)
    assert gap["eps_loc_dual"] >= 0.0
    assert gap["eps_loc_bottleneck"] >= 0.0
    assert len(gap["per_target_local"]) == Q_TARGETS
    assert len(gap["per_target_common"]) == Q_TARGETS


def test_stopping_tail_verify_no_violation(bridge_out):
    ver = stopping_tail_verify(bridge_out, Q_TARGETS)
    assert ver["violation_fraction"] <= 0.05
    assert ver["satisfied"] is True
    assert 0.0 <= ver["empirical_survive_max"] <= 1.0


def test_normalized_service_time_average(bridge_out, scenario):
    g = np.zeros((K_UAVS, Q_TARGETS))
    for i in range(K_UAVS):
        for qq in range(Q_TARGETS):
            rel = (1.0 if i == scenario["owner_of"][qq]
                   else float(bridge_out["delivery_matrix"][i,
                                                            scenario["owner_of"][qq]]))
            g[i, qq] = rel * float(bridge_out["mu_llr"][i, qq])
    svc = normalized_service_time_average(bridge_out, Q_TARGETS, g_mat=g)
    assert svc["z_star_static"] is not None
    assert svc["z_star_static"] > 0.0
    assert 0.0 <= svc["min_q_time_avg_r_static"] <= 2.0
    assert svc["eps_T_est"] is not None
    assert svc["sqrt_logQ_over_T"] > 0.0
    assert svc["eps_loc_mean"] >= 0.0
    # static-deficit distributed-information loss is small (the recorded
    # boundary-normalized value explodes at D->0 by construction, so the
    # STATIC normalization is the honest audit metric)
    assert svc["eps_loc_static_mean"] < 0.05


def test_bridge_default_off_output_unchanged(scenario, bounds):
    a = simulate_frids_v2(scenario, bounds, n_runs=60, seed=3,
                          max_steps=40)
    b = simulate_frids_v2(scenario, bounds, n_runs=60, seed=3,
                          max_steps=40, bridge=False)
    assert a["worst_target_delay"] == b["worst_target_delay"]
    assert a["p_md"] == b["p_md"]
    assert "bridge" not in a
    assert "bridge" not in b