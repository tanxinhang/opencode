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
    # point estimate above spec but LOWER bound still below -> the
    # protocol does NOT claim failure (LCB(P_err) > p_max is required)
    lo, _ = clopper_pearson(8, 100, 0.05 / 4)
    md_lo, _ = clopper_pearson(8, 100, 0.05 / 4)
    assert lo <= 0.05
    status = raw_qos_status([100, 100], [100, 100], [8, 8], [8, 8],
                            0.05, 0.05, 0.05)
    # point estimates at 0.08 exceed the spec but the simultaneous LCB
    # clears it -- the protocol must NOT claim FAIL on the point estimate
    assert status in ("UNCERTAIN", "PASS")
    # the reverse: LCB > p_max IS a certified FAIL even when the point
    # estimate alone looks "not too bad"
    status2 = raw_qos_status([100, 100], [100, 100], [18, 18], [18, 18],
                             0.05, 0.05, 0.05)
    assert status2 == "FAIL"


def test_qos_uncertain_before_fail_must_return_fail():
    """P0-1 regression (advice/009 section 1): an EARLIER UNCERTAIN target
    must NOT mask a LATER certified FAIL.  The certificate scans every
    target for a certified violation FIRST and only returns UNCERTAIN if
    no violation exists anywhere."""
    # target 0: too few trials (UNCERTAIN on its own)
    # target 1: LCB already above spec (certified FAIL)
    status = raw_qos_status([10, 100], [10, 100], [2, 60], [2, 60],
                            0.05, 0.05, 0.05)
    assert status == "FAIL"


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
    # P3.5-B pathwise invariant: the hard owner admission keeps
    # ``sum c <= 1`` on EVERY cycle, so the overload ratio never exceeds
    # 1 and the budget-feasible fraction is 1 -- in the free regime no
    # capacity drops occur at all (advice/009 sections 5, 11)
    assert out["comm"]["budget_feasible_fraction"] == pytest.approx(1.0)
    assert out["comm"]["max_load_ratio"] <= 1.0 + 1e-9
    assert out["comm"]["rx_capacity_dropped_per_uav"] == 0.0
    assert out["comm"]["rx_delivered_per_uav"] > 0.0


def test_ca_congested_load_drops(scenario, bounds, air_cong):
    bt = bounds
    b = bounds_v2(bt, 3)
    out = simulate_ca_frids(scenario, b, air_cong, n_runs=60, seed=4,
                            max_steps=40, pi_bits=12, lam_bits=12)
    # P3.5-B (advice/009 sections 5, 11): the receiver HARD-admits a
    # density-ranked ``J/c`` subset under the PATHWISE budget ``sum c <=
    # 1`` -- so the overload ratio NEVER exceeds 1 and every cycle is
    # budget-feasible (lambda is the steering price; hard admission is
    # the MAC feasibility fuse).  Congestion now shows up as a
    # capacity-dropped ledger, not as overloaded cycles.
    assert out["comm"]["max_load_ratio"] <= 1.0 + 1e-9
    assert out["comm"]["budget_feasible_fraction"] == pytest.approx(1.0)
    assert out["comm"]["rx_capacity_dropped_per_uav"] > 0.0
    # the dual airtime price must cut the ATTEMPT rate in overload
    assert out["comm"]["tx_attempts_per_uav"] < 1.0
    # ledger identity (advice/009 section 15): attempts = delivered +
    # link-dropped + capacity-dropped
    a = out["comm"]["tx_attempts_per_uav"]
    d = out["comm"]["rx_delivered_per_uav"]
    ld = out["comm"]["rx_link_dropped_per_uav"]
    cd = out["comm"]["rx_capacity_dropped_per_uav"]
    assert a == pytest.approx(d + ld + cd, abs=1e-9)


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


def test_ca_price_mode_global_simplex_runs(scenario, bounds, air_fair):
    """P3.6 (advice/009 sections 6-8): the global-simplex task price
    variant runs end-to-end with deterministic output."""
    bt = bounds
    b = bounds_v2(bt, 3)
    g = simulate_ca_frids(scenario, b, air_fair, n_runs=60, seed=5,
                          max_steps=40, pi_bits=12, lam_bits=12,
                          price_mode="global_simplex")
    assert 0.0 < g["worst_target_delay"] <= 40.0
    for p in g["p_fa"] + g["p_md"]:
        assert 0.0 <= p <= 1.0


def test_single_target_owner_price_can_still_change_globally():
    """P3.6 regression (advice/009 section 8): with one target per owner,
    the owner-local simplex freezes ``y_q = 1`` forever (the v0 failure),
    while the global-simplex mirror descent keeps ``sum y = 1`` and lets
    every price move."""
    from uav_otfs_isac.ca_frids import _global_simplex
    y = np.array([1.0 / 3.0] * 3)
    ratios = np.array([2.0, 0.5, 0.5])
    for _ in range(20):
        y = _global_simplex(0.5, ratios, y, [0, 1, 2])
    assert y.sum() == pytest.approx(1.0, abs=1e-9)
    # the well-served target's price drops below the uniform price
    assert y[0] < 1.0 / 3.0
    assert y[1] > 0.0 and y[2] > 0.0


def test_crn_tape_deterministic_and_reproducible(scenario, bounds, air_fair):
    """P0-2 (advice/010): the exogenous tape is deterministic and
    reproducible, so a scheduler run with a tape is byte-identical across
    two builds of the same tape."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    q = scenario["q"]
    k = int(scenario["k"])
    b = bounds_v2(bounds, 3)
    a1 = simulate_ca_frids(scenario, b, air_fair, n_runs=60, seed=5,
                           max_steps=40,
                           exog=build_exogenous_tape(7, 60, q, k, 40))
    a2 = simulate_ca_frids(scenario, b, air_fair, n_runs=60, seed=5,
                           max_steps=40,
                           exog=build_exogenous_tape(7, 60, q, k, 40))
    assert a1["worst_target_delay"] == a2["worst_target_delay"]
    assert np.allclose(a1["pool"]["sum_h1_delay"],
                       a2["pool"]["sum_h1_delay"])
    assert np.allclose(a1["pool"]["sum2_h1_delay"],
                       a2["pool"]["sum2_h1_delay"])


def test_crn_tape_shares_target_presence_between_v2_and_ca(
        scenario, bounds, air_fair):
    """P0-2 (advice/010): FRIDS-v2 and CA-FRIDS read the SAME tape H, so
    the per-target H1 denominators agree and the paired comparison is over
    identical exogenous presence draws."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    from uav_otfs_isac.frids import simulate_frids_v2
    q = scenario["q"]
    k = int(scenario["k"])
    n_runs = 80
    tape = build_exogenous_tape(9, n_runs, q, k, 40)
    b = bounds_v2(bounds, q)
    out_v2 = simulate_frids_v2(scenario, b, n_runs=n_runs, seed=9,
                               max_steps=40, exog=tape)
    out_ca = simulate_ca_frids(scenario, b, air_fair, n_runs=n_runs, seed=9,
                               max_steps=40, exog=tape)
    expected = [int(np.sum(tape.U_H[:, qq] < 0.5)) for qq in range(q)]
    assert out_v2["pool"]["n_h1"] == expected
    assert out_ca["pool"]["n_h1"] == expected
    assert out_v2["pool"]["n_h1"] == out_ca["pool"]["n_h1"]


def test_pool_ledger_is_per_target_pooled():
    """P0-5 (advice/010 section 3): the ``pool`` block carries raw
    per-target H1 counts and delay sums so a geometry can compute
    J_g = max_q sum_h1_delay[q]/n_h1[q] without the per-cell
    worst-then-average selection bias."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    rng = np.random.default_rng(11)
    sc = build_distributed_scenario(rng, k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=40, seed=100, verify_runs=0)
    b = bounds_v2(bt, 3)
    am = build_airtime_model(sc, rho_target=0.5)
    q = sc["q"]
    k = int(sc["k"])
    tape = build_exogenous_tape(13, 50, q, k, 40)
    out = simulate_ca_frids(sc, b, am, n_runs=50, seed=13, max_steps=40,
                            exog=tape)
    pool = out["pool"]
    assert len(pool["n_h1"]) == q
    assert all(int(v) >= 0 for v in pool["n_h1"])
    assert all(float(v) >= 0.0 for v in pool["sum_h1_delay"])
    # every H1 run has a positive delay, so sum/n in [1, max_steps]
    for count, total in zip(pool["n_h1"], pool["sum_h1_delay"]):
        if count > 0:
            mean = total / count
            assert 1.0 <= mean <= 40.0


def test_density_admission_is_label_equivariant():
    """P0-B (advice/011 section 6): the density admission tie-break uses
    independent policy keys that live on the UAV indices, so a permutation
    of UAV labels permutes the keys equally and the admitted order is
    label-equivariant (no fixed low-index bias)."""
    from uav_otfs_isac.ca_frids import _density_sorted_offers
    offers = [
        (0, 0, 0.5, 2.0, None),
        (1, 0, 0.5, 2.0, None),   # identical density: tie decided below
        (2, 0, 0.4, 1.0, None),
    ]
    keys = np.array([0.3, 0.9, 0.5])
    order = [o[0] for o in _density_sorted_offers(offers, keys)]
    assert order[0] == 0          # denser pairs keep their rank
    assert order[2] == 2          # lowest density last
    # permute LABELS and permute the keys by the INVERSE permutation: the
    # key attached to an offer instance must follow that instance through
    # the relabeling (keys_p[new] = keys[old], old = inv[new]).
    perm = [2, 0, 1]
    inv = [perm.index(i) for i in range(len(offers))]
    offers_p = [(perm[uav], qq, c, score, None)
                for uav, qq, c, score, _ in offers]
    order_p = [o[0] for o in _density_sorted_offers(offers_p, keys[inv])]
    assert order_p == [perm[uav] for uav in order]


def test_p41_airtime_v2_matches_legacy_uncongested(
        scenario, bounds, air_fair):
    """P4.1 (advice/012): under the CRN tape the airtime-neutral v2 path
    is EXACTLY the legacy independent-delivery path when the budget is
    non-binding (rho<1): the delivered set is decided by the SAME
    positional U_link cells, so delays are unchanged and only the split
    ledger is added."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    from uav_otfs_isac.frids import simulate_frids_v2
    q = scenario["q"]
    k = int(scenario["k"])
    n_runs = 80
    tape = build_exogenous_tape(23, n_runs, q, k, 40)
    b = bounds_v2(bounds, 3)
    legacy = simulate_frids_v2(
        scenario, b, n_runs=n_runs, seed=23, max_steps=40, exog=tape)
    airtime = simulate_frids_v2(
        scenario, b, n_runs=n_runs, seed=23, max_steps=40, exog=tape,
        airtime=air_fair)
    assert np.allclose(legacy["e1_delays"], airtime["e1_delays"])
    assert np.allclose(legacy["pool"]["sum_h1_delay"],
                       airtime["pool"]["sum_h1_delay"])
    # the non-binding budget admits every offer, so no capacity drops
    assert airtime["comm"]["capacity_dropped_tx_per_uav"] == 0.0
    assert np.isclose(airtime["comm"]["offer_attempts_per_uav"],
                      airtime["comm"]["admitted_tx_per_uav"], atol=1e-9)


def test_p41_congested_split_ledger_identities(
        scenario, bounds, air_cong):
    """P4.1 (advice/012 section 5): under the shared airtime capacity the
    split ledger identities hold for BOTH schedulers on the congested
    regime: offers = admitted + capacity-dropped and
    admitted = delivered + link-dropped."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    from uav_otfs_isac.ca_frids import simulate_ca_frids
    from uav_otfs_isac.frids import simulate_frids_v2
    q = scenario["q"]
    k = int(scenario["k"])
    n_runs = 80
    tape = build_exogenous_tape(31, n_runs, q, k, 40)
    b = bounds_v2(bounds, 3)
    for out in (
        simulate_frids_v2(
            scenario, b, n_runs=n_runs, seed=31, max_steps=40, exog=tape,
            airtime=air_cong),
        simulate_ca_frids(
            scenario, b, air_cong, n_runs=n_runs, seed=31, max_steps=40,
            exog=tape),
    ):
        cm = out["comm"]
        assert np.isclose(
            cm["offer_attempts_per_uav"],
            cm["admitted_tx_per_uav"]
            + cm["capacity_dropped_tx_per_uav"], atol=1e-9)
        assert np.isclose(
            cm["admitted_tx_per_uav"],
            cm["delivered_tx_per_uav"]
            + cm["link_dropped_tx_per_uav"], atol=1e-9)



def test_global_simplex_only_scalar_normalizer_is_networked():
    """P3.6 (advice/009 section 8): the only global quantity is the scalar
    ``Z = sum_p w_p`` -- the update has NO global ``rbar`` factor (the
    public ``exp(mu*rbar)`` cancels in the simplex normalization), so the
    reduction is a single-scalar spanning-tree/gossip, not a full-mesh
    belief exchange."""
    from uav_otfs_isac.ca_frids import _global_simplex
    y = np.array([0.5, 0.3, 0.2])
    ratios = np.array([1.0, 2.0, 3.0])
    y1 = _global_simplex(0.5, ratios, y, [0, 1, 2])
    # adding a constant to every ratio scales all ``w`` uniformly, which
    # the normalization removes -- ``rbar`` never needs to be known
    y2 = _global_simplex(0.5, ratios + 100.0, y, [0, 1, 2])
    assert np.allclose(y1, y2, atol=1e-9)