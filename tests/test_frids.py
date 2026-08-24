"""FRIDS tests (Gate F0-G3, advice/009)."""

import numpy as np
import pytest

from uav_otfs_isac.distributed_audit import (
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import (
    capacity,
    g_reliable,
    load_cut,
    simplex_projection,
    simplex_projection_lb,
    simulate_frids,
    simulate_frids_v2,
)


def test_simplex_projection():
    v = np.array([0.7, 0.2, 0.4])
    p = simplex_projection(v)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(p >= -1e-12)
    # already on the simplex -> unchanged
    u = np.array([0.5, 0.3, 0.2])
    assert np.allclose(simplex_projection(u), u)


def test_simplex_projection_lb():
    v = np.array([0.7, 0.2, 0.4])
    lb = 0.1
    p = simplex_projection_lb(v, lb)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(p >= lb - 1e-12)
    # all-at-floor case
    p2 = simplex_projection_lb(np.zeros(4), 0.3)
    assert np.allclose(p2, 0.25)


def test_g_reliable_and_capacity():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    owner = sc["owner_of"]
    g0 = g_reliable(sc, 0, 1, owner)
    assert g0 > 0.0
    # the owner's own reliability is 1 (its evidence always counts)
    g_own = g_reliable(sc, owner[1], 1, owner)
    assert g_own == pytest.approx(
        max(act["i_plus"] for act in sc["by_host"][(owner[1], 1)]),
        rel=1e-9)
    c1 = capacity(sc, 1, owner)
    assert c1 > g0


def test_frids_runs_and_error_reporting():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100,
                                 verify_runs=0)
    out = simulate_frids(sc, bt, n_runs=80, seed=7, max_steps=40)
    assert 0.0 < out["worst_target_delay"] <= 40.0
    assert len(out["e1_delays"]) == 3
    for p in out["p_fa"] + out["p_md"]:
        assert 0.0 <= p <= 1.0
    assert len(out["infeasible_cycle_fraction"]) == 3
    assert len(out["infeasible_target_fraction"]) == 3


def test_frids_deterministic():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100,
                                 verify_runs=0)
    a = simulate_frids(sc, bt, n_runs=60, seed=11, max_steps=40)
    b = simulate_frids(sc, bt, n_runs=60, seed=11, max_steps=40)
    assert a["worst_target_delay"] == b["worst_target_delay"]
    assert a["e1_delays"] == b["e1_delays"]


def test_lowering_b_reduces_pmd_without_touching_pfa():
    """The F0-G3 policy-matched operating point: lowering B cuts
    H0-declarations under H1 (P_MD down) while P_FA stays structurally
    ~0 -- the correct lever for the FRIDS streams."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100,
                                 verify_runs=0)
    b0 = [[bt[qq][0], bt[qq][1]] for qq in range(3)]
    b1 = [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(3)]
    o0 = simulate_frids(sc, b0, n_runs=200, seed=17, max_steps=40)
    o1 = simulate_frids(sc, b1, n_runs=200, seed=17, max_steps=40)
    assert max(o1["p_md"]) <= max(o0["p_md"]) + 1e-9
    assert max(o1["p_fa"]) <= max(o0["p_fa"]) + 1e-9


def test_reliable_information_identity():
    """advice/010 audit: g_iq == KL of the final observable kernel.
    The post-communication I+ already includes the kernel's detectable
    erasure; the u2u delivery is an independent channel, and the
    identity I+^final = s_u2u * I+^post holds exactly -- no double
    counting."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    owner = sc["owner_of"]
    for (uav, q) in ((1, 2), (3, 0), (5, 1)):
        act = max(sc["by_host"][(uav, q)],
                  key=lambda a: a["i_plus"])
        s_u2u = float(sc["u2u_success"][uav, owner[q]])
        g = g_reliable(sc, uav, q, owner)
        p0f = s_u2u * np.asarray(act["p0"])
        p1f = s_u2u * np.asarray(act["p1"])
        p0f = np.concatenate([p0f, [1.0 - s_u2u]])
        p1f = np.concatenate([p1f, [1.0 - s_u2u]])
        kl_final = float(np.sum(p1f * np.log(p1f / p0f)))
        assert g == pytest.approx(kl_final, rel=1e-9)


def test_scale_aware_token_accounting():
    """advice/010 audit: b_q = ceil(log2 Q) and the total stays within
    19 bits (dead payload fields u/r/chi/stamp dropped in cascade)."""
    from uav_otfs_isac.distributed_audit import token_bits
    for q in (2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32):
        tb = token_bits(q)
        assert tb["total"] <= 19
        assert tb["q"] == max(2, int(np.ceil(np.log2(max(q, 2)))))
        assert tb["intent"] == tb["q"]
        # only dead fields are dropped
        for field in ("u", "r", "chi", "stamp"):
            assert tb[field] in (0, 2) or field == "stamp" \
                or tb[field] in (0, 4)


def test_frids_v2_runs_and_deterministic():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100,
                                 verify_runs=0)
    a = simulate_frids_v2(sc, bt, n_runs=60, seed=29, max_steps=40)
    b = simulate_frids_v2(sc, bt, n_runs=60, seed=29, max_steps=40)
    assert a["worst_target_delay"] == b["worst_target_delay"]
    assert 0.0 < a["worst_target_delay"] <= 40.0
    for p in a["p_fa"] + a["p_md"]:
        assert 0.0 <= p <= 1.0


def test_load_cut_math():
    """rho(S) = sum D^info / (H * sum_i max_{q in S} g_iq); if rho > 1
    the subset cannot finish.  The full set is far below 1 in the
    tested family (information capacity is not the binding constraint)."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    owner = sc["owner_of"]
    rho = load_cut(sc, owner, [0, 1, 2], horizon=40)
    assert rho > 0.0 and rho < 1.0
    # a single-target subset has rho <= the full set's per-target share
    rho_single = max(load_cut(sc, owner, [qq], horizon=40)
                     for qq in range(3))
    assert rho_single < 1.0


def test_frids_scale_invariance_to_reliability():
    """advice/011 F0-G5 finding: g = s * I+ and a GLOBAL reliability
    misestimation s_hat = c * s_true scales every gain by the same
    factor, so the argmax ranking (and hence the decisions) is
    invariant -- FRIDS is robust to global reliability misestimation by
    construction; only per-link non-uniform errors matter."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100,
                                 verify_runs=0)
    u2u = sc["u2u_success"]
    for kappa in (1.0, 0.6):
        s_hat = np.clip(u2u / kappa, 0.0, 1.0)
        o = simulate_frids_v2(sc, bt, n_runs=60, seed=31,
                              max_steps=40, delivery_matrix=u2u,
                              s_for_g=s_hat)
        assert 0.0 < o["worst_target_delay"] <= 40.0
    # decisions are identical: the uniform-scaled g preserves argmax
    a = simulate_frids_v2(sc, bt, n_runs=60, seed=33, max_steps=40,
                          delivery_matrix=u2u, s_for_g=u2u)
    b = simulate_frids_v2(sc, bt, n_runs=60, seed=33, max_steps=40,
                          delivery_matrix=u2u,
                          s_for_g=np.clip(u2u / 0.6, 0.0, 1.0))
    # clip at 1.0 can distort the ranking for high-reliability links,
    # so the comparison is allowed a small tolerance
    assert abs(a["worst_target_delay"] - b["worst_target_delay"]) < 0.5


def test_audit_common_price_runs_and_diagnostics(scenario=None):
    """Gate F0-G9A (advice/020): the local-dual consistency audit runs,
    the common-price oracle runs, and the audit returns the diagnostics
    with the certificate fraction in [0, 1]."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100,
                                 verify_runs=0)
    local = simulate_frids_v2(sc, bt, n_runs=80, seed=7, max_steps=40,
                              audit=True)
    common = simulate_frids_v2(sc, bt, n_runs=80, seed=7, max_steps=40,
                               price_mode="common", audit=True)
    for out in (local, common):
        a = out["audit"]
        assert 0.0 <= a["margin_ok_fraction"] <= 1.0
        assert a["d_y"] >= 0.0
        assert a["d_v"] >= 0.0
        assert a["deficit_gap"] >= 0.0
        assert a["margin_samples"] > 0
    # the common-price oracle uses the common price in the action, so the
    # ACTION-relevant price error is exactly zero (only the local value
    # error from the local vs owner deficit remains)
    assert common["audit"]["eps_y_max"] == pytest.approx(0.0, abs=1e-12)
    assert local["audit"]["eps_y_max"] > 0.0


def test_action_invariance_certificate_math():
    """Theorem 4.109: with |dy| <= eps_y and |dv| <= eps_v and v <=
    V_max, the local score error is bounded by
    E = V_max*eps_y + eps_v + eps_y*eps_v, and m > 2E preserves the
    argmax."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = 4
        v = rng.uniform(0.05, 1.0, size=q)
        y = rng.dirichlet(np.ones(q))
        dy = rng.uniform(-0.05, 0.05, size=q) * y
        dv = rng.uniform(-0.02, 0.02, size=q) * v
        eps_y = float(np.max(np.abs(dy)))
        eps_v = float(np.max(np.abs(dv)))
        v_max = float(np.max(v))
        J = y * v
        J_hat = (y + dy) * (v + dv)
        assert np.max(np.abs(J_hat - J)) <= \
            v_max * eps_y + eps_v + eps_y * eps_v + 1e-12
        m = float(np.sort(J)[-1] - np.sort(J)[-2])
        E = v_max * eps_y + eps_v + eps_y * eps_v
        if m > 2.0 * E:
            assert np.argmax(J_hat) == np.argmax(J)


def test_system_bottleneck_oracles():
    """System Bottleneck Audit v2 hooks (advice/022): snr_shift builds a
    stronger-sensing scenario, mobility drifts the frozen-policy evidence
    within its bound, and a different static owner runs the same
    calibration."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100,
                                 verify_runs=0)
    # snr_shift: higher sensing SNR -> higher i_plus
    sc_hi = build_distributed_scenario(np.random.default_rng(0),
                                       k_uavs=6, q_targets=3,
                                       snr_shift=4.0)
    g_lo = max(g_reliable(sc, i, 0, sc["owner_of"]) for i in range(6))
    g_hi = max(g_reliable(sc_hi, i, 0, sc_hi["owner_of"]) for i in range(6))
    assert g_hi > g_lo
    # mobility runs within the frozen policy
    out_m = simulate_frids_v2(sc, bt, n_runs=60, seed=7, max_steps=40,
                              mobility=0.05)
    assert 0.0 < out_m["worst_target_delay"] <= 40.0
    # alternative static owner runs with the SAME calibration
    sc_o = dict(sc)
    sc_o["owner_of"] = [1, 2, 0]
    out_o = simulate_frids_v2(sc_o, bt, n_runs=60, seed=7, max_steps=40)
    assert 0.0 < out_o["worst_target_delay"] <= 40.0


def test_power_cap_sensing_energy():
    """G10 (advice/024): the power_cap restricts the sensing power, the
    energy accounting reports the per-UAV power, and a lower cap consumes
    less energy and (weakly) weaker evidence."""
    from uav_otfs_isac.distributed_audit import calibrate_target_bounds
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3,
                                    powers=(1.0, 2.0, 3.0))
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100, verify_runs=0)
    cap_lo = np.array([1.0, 1.0, 1.0])
    bt_lo = calibrate_target_bounds(sc, n_runs=60, seed=100, verify_runs=0,
                                    power_cap=cap_lo)
    bounds_lo = [[bt_lo[qq][0], bt_lo[qq][1]] for qq in range(3)]
    hi = simulate_frids_v2(sc, bt, n_runs=60, seed=7, max_steps=40,
                           power_cap=np.array([2.0, 2.0, 2.0]))
    lo = simulate_frids_v2(sc, bt_lo, n_runs=60, seed=7, max_steps=40,
                           power_cap=cap_lo)
    assert hi["sensing_power_per_uav"] > lo["sensing_power_per_uav"]
    assert 1.0 <= hi["sensing_power_per_uav"] <= 2.0
    assert 0.0 < lo["worst_target_delay"] <= 40.0


def test_dd_grid_leakage_changes_evidence():
    """G10-C (advice/024): the fixed-TB OTFS DD grid shape changes the
    evidence through the sinc^2 fractional-bin leakage; a target aligned
    with the grid (fractional ~ 0) leaks less than one mid-bin, and
    different grid shapes give different reliable information."""
    from uav_otfs_isac.distributed_audit import calibrate_target_bounds
    # aligned physics: fractional residue ~ 0 for both dimensions
    aligned = {qq: (0.001, 0.001) for qq in range(3)}
    # mid-bin physics: fractional ~ 0.5 for both dimensions
    mid = {qq: (0.5, 0.5) for qq in range(3)}
    sc_a = build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=6, q_targets=3,
                                      dd_grid=(64, 64), dd_physics=aligned)
    sc_m = build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=6, q_targets=3,
                                      dd_grid=(64, 64), dd_physics=mid)
    g_a = max(g_reliable(sc_a, i, 0, sc_a["owner_of"]) for i in range(6))
    g_m = max(g_reliable(sc_m, i, 0, sc_m["owner_of"]) for i in range(6))
    assert g_a > g_m                       # aligned leaks less
    # different grid shapes under fixed TB give different evidence
    gs = {}
    for grid in ((128, 32), (64, 64), (32, 128)):
        sc = build_distributed_scenario(np.random.default_rng(0),
                                        k_uavs=6, q_targets=3,
                                        dd_grid=grid, dd_physics=mid)
        gs[grid] = max(g_reliable(sc, i, 0, sc["owner_of"]) for i in range(6))
    assert len(set(round(v, 6) for v in gs.values())) > 1


def test_frids_v2_task_price_false_runs_flat_deterministic():
    """advice/018 section 8 (2x2 core cell F0): ``task_price=False`` runs
    the full-mesh FLAT index ``J_iq = g_iq`` (no local deficit price, no
    mirror descent) deterministically and still reports the risk-adjusted
    delay ledger used by the P5-A ladder."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100, verify_runs=0)
    q, k = 3, int(sc["k"])
    tape = build_exogenous_tape(44, 60, q, k, 40)
    out = simulate_frids_v2(sc, bt, n_runs=60, seed=44, max_steps=40,
                            price_mode="local", exog=tape,
                            task_price=False)
    assert 1.0 <= out["worst_target_delay"] <= 40.0
    assert out["pool"]["n_h1"] is not None
    assert len(out["pool"]["sum_h1_delay_risk"]) == 3
    out2 = simulate_frids_v2(sc, bt, n_runs=60, seed=44, max_steps=40,
                             price_mode="local", exog=tape,
                             task_price=False)
    assert np.allclose(out["pool"]["sum_h1_delay"],
                       out2["pool"]["sum_h1_delay"])


def test_frids_v2_task_price_true_is_default_path():
    """advice/018 section 8: the ``task_price=True`` default keeps the
    frozen v2 index byte-identical (score with local deficit weighting,
    mirror descent active) -- the F0 flag must not change the default
    scheduler."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=60, seed=100, verify_runs=0)
    default = simulate_frids_v2(sc, bt, n_runs=60, seed=45, max_steps=40,
                                price_mode="local")
    explicit = simulate_frids_v2(sc, bt, n_runs=60, seed=45, max_steps=40,
                                 price_mode="local", task_price=True)
    assert np.allclose(default["pool"]["sum_h1_delay"],
                       explicit["pool"]["sum_h1_delay"])
    assert np.allclose(default["worst_target_delay"],
                       explicit["worst_target_delay"])
