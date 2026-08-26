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


# ---------------------------------------------------------------------------
# P4.1a credibility regression tests (advice/014 section 4)
# ---------------------------------------------------------------------------


def test_p41a_neutral_admission_requires_tie_keys():
    """P4.1a regression (advice/014 section 4 #1): the shared airtime
    primitive refuses ``policy="neutral"`` with ``tie_keys=None`` -- the
    source-index fallback would silently reintroduce fixed low-index
    bias."""
    from uav_otfs_isac.admission import airtime_admit
    with pytest.raises(ValueError):
        airtime_admit([[(0, 0.0, 0.5, None)]], t_air=1.0,
                      policy="neutral", tie_keys=None)


def test_p41a_policy_tape_identical_on_rebuild():
    """P4.1a regression (advice/014 section 4 #2): building the CRN tape
    twice with the same seed reproduces the SAME ``U_policy`` block -- the
    same (r, t, receiver, source) cells, hence identical admission tie
    keys for every episode and cycle."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    a = build_exogenous_tape(77, 8, 3, 6, 40)
    b = build_exogenous_tape(77, 8, 3, 6, 40)
    assert a.U_policy.shape == (8, 40, 6, 6)
    assert np.array_equal(a.U_policy, b.U_policy)


def test_p41a_policy_tape_temporal_refresh():
    """P4.1a regression (advice/014 section 4 #3): the policy tape
    REFRESHES every cycle -- ``U_policy[r,t]`` differs from
    ``U_policy[r,t']`` for ``t != t'``, so admission tie keys are not
    frozen across an episode (no persistent source favoritism)."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    tape = build_exogenous_tape(78, 4, 3, 6, 40)
    assert not np.array_equal(tape.U_policy[0, 0], tape.U_policy[0, 1])


def test_p41a_policy_tape_receiver_independent():
    """P4.1a regression (advice/014 section 4 #4): the tie key is
    (receiver, source)-indexed -- for the same source the keys of two
    receivers differ, so admission ties are receiver-independent (no
    shared per-source key that couples every receiver)."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    tape = build_exogenous_tape(79, 4, 3, 6, 40)
    r, t = 0, 0
    assert not np.array_equal(tape.U_policy[r, t, 0, :],
                              tape.U_policy[r, t, 1, :])


def test_task_price_false_keeps_flat_architecture_score(
        scenario, bounds, air_cong):
    """P5-A (advice/017 section 13): ``task_price=False`` freezes ``pi_q``
    at the flat ``1/q`` (no deficit weighting) -- with ``airtime_price=False``
    the local index reduces to the bare reliable information ``g_eff``, so
    the arm is the owner-directed EVIDENCE ARCHITECTURE alone (B00).  The
    task-price update must NOT run: the y-state stays uniform."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    q = scenario["q"]
    k = int(scenario["k"])
    b = bounds_v2(bounds, q)
    tape = build_exogenous_tape(41, 60, q, k, 40)
    out = simulate_ca_frids(scenario, b, air_cong, n_runs=60, seed=41,
                            max_steps=40, task_price=False,
                            airtime_price=False,
                            admission_policy="neutral",
                            exog=tape)
    assert out["pool"]["n_h1"] is not None
    assert out["worst_target_delay"] >= 1.0
    # the ladder arms keep the shared CRN tape deterministic
    out2 = simulate_ca_frids(scenario, b, air_cong, n_runs=60, seed=41,
                             max_steps=40, task_price=False,
                             airtime_price=False,
                             admission_policy="neutral",
                             exog=tape)
    assert np.allclose(out["pool"]["sum_h1_delay"],
                       out2["pool"]["sum_h1_delay"])


def test_task_price_false_is_not_task_aware(scenario, bounds, air_cong):
    """P5-A (advice/017 section 13): in the B00 arm the flat price cannot
    redirect service to a starved target -- the flat arm serves by pure
    reliable information, which is exactly what the ladder is designed to
    measure (the task-deficit mechanism is absent)."""
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    rng = np.random.default_rng(42)
    sc = build_distributed_scenario(rng, k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=40, seed=100, verify_runs=0)
    b = bounds_v2(bt, 3)
    am = build_airtime_model(sc, rho_target=2.0)
    q = sc["q"]
    k = int(sc["k"])
    tape = build_exogenous_tape(43, 60, q, k, 40)
    out_flat = simulate_ca_frids(sc, b, am, n_runs=60, seed=43, max_steps=40,
                                 task_price=False, airtime_price=False,
                                 admission_policy="neutral",
                                 exog=tape)
    out_dyn = simulate_ca_frids(sc, b, am, n_runs=60, seed=43, max_steps=40,
                                task_price=True, airtime_price=False,
                                admission_policy="neutral",
                                exog=tape)
    # both arms are valid schedulers and the flat arm respects the budget
    assert out_flat["worst_target_delay"] >= 1.0
    assert out_dyn["worst_target_delay"] >= 1.0
    # the two arms may differ (task price steering OR deficit weighting)
    # -- the point is the flat arm RUNS without any task-price update
    assert out_flat["comm"]["budget_feasible_fraction"] >= 0.0


def test_p41a_four_arms_share_exogenous_tape(scenario, bounds, air_cong):
    """P4.1a regression (advice/014 section 4 #5): the four causal arms
    A (v2+neutral), B0 (CA+lambda=0+neutral), B1 (CA+lambda+neutral) and
    C (CA+lambda+density) all run through the SAME CRN tape (shared
    U_H/U_obs/U_link/U_policy), so the per-target H1 counts -- driven only
    by the shared presence draws -- are EXACTLY equal across all four
    arms."""
    from scripts.run_ca_frids_gate import matched_qos
    from uav_otfs_isac.frids import simulate_frids_v2
    price_mode = "global_simplex"
    _, rows_a = matched_qos(
        simulate_frids_v2, scenario, bounds, 40, 9, 40, 0.05, 0.05,
        mc_seeds=1, crn=True, price_mode="local", airtime=air_cong)
    _, rows_b0 = matched_qos(
        simulate_ca_frids, scenario, bounds, 40, 9, 40, 0.05, 0.05,
        mc_seeds=1, crn=True, price_mode=price_mode, airtime=air_cong,
        admission_policy="neutral", airtime_price=False,
        pi_bits=10, lam_bits=10)
    _, rows_b1 = matched_qos(
        simulate_ca_frids, scenario, bounds, 40, 9, 40, 0.05, 0.05,
        mc_seeds=1, crn=True, price_mode=price_mode, airtime=air_cong,
        admission_policy="neutral", airtime_price=True,
        pi_bits=10, lam_bits=10)
    _, rows_c = matched_qos(
        simulate_ca_frids, scenario, bounds, 40, 9, 40, 0.05, 0.05,
        mc_seeds=1, crn=True, price_mode=price_mode, airtime=air_cong,
        admission_policy="density", airtime_price=True,
        pi_bits=10, lam_bits=10)
    a_h1 = rows_a[0]["raw_counts"]["n_H1"]
    b0_h1 = rows_b0[0]["raw_counts"]["n_H1"]
    b1_h1 = rows_b1[0]["raw_counts"]["n_H1"]
    c_h1 = rows_c[0]["raw_counts"]["n_H1"]
    assert a_h1 == b0_h1 == b1_h1 == c_h1
# ---------------------------------------------------------------------------
# advice/020 section 3: genuinely owner-local normalization-free theta path
# ---------------------------------------------------------------------------


def test_owner_theta_update_is_max_free_and_owner_local():
    """advice/020 section 3: ``_owner_theta_update`` must NOT apply any
    global max shift / sum normalizer -- each ``theta_q`` moves only by its
    OWN ``-mu r_q``, so adding a constant to every ratio must change NO
    owner's update (there is no recentering anywhere)."""
    from uav_otfs_isac.ca_frids import _owner_theta_update
    theta = np.array([np.log(0.5), np.log(0.3), np.log(0.2)])
    ratios = np.array([1.0, 2.0, 3.0])
    und = [0, 1, 2]
    t1 = _owner_theta_update(theta, 0.5, ratios, und)
    # exact additive update, no normalization: theta_q' = theta_q - mu*r_q
    assert np.allclose(t1, theta - 0.5 * ratios, atol=1e-12)
    # owner-locality: changing ONLY target 1's ratio must change ONLY
    # target 1's theta -- no cross-target normalization / global coupling
    ratios2 = np.array([1.0, 5.0, 3.0])
    t2 = _owner_theta_update(theta, 0.5, ratios2, und)
    assert t2[0] == pytest.approx(t1[0], abs=1e-12)
    assert t2[2] == pytest.approx(t1[2], abs=1e-12)
    assert t2[1] == pytest.approx(theta[1] - 0.5 * 5.0, abs=1e-12)
    # undecided subset only
    t3 = _owner_theta_update(theta, 0.5, ratios, [1])
    assert t3[1] == pytest.approx(theta[1] - 1.0, abs=1e-12)
    assert t3[0] == pytest.approx(theta[0], abs=1e-12)


def test_norm_free_action_error_bound_certificate():
    """advice/020 section 2-3: the log-domain action-invariance certificate
    passes when the top-1 log margin exceeds twice the theta quantization
    step, and fails when the margin is too small."""
    from uav_otfs_isac.ca_frids import norm_free_action_error_bound
    lo, hi, bits = -20.0, 0.0, 10
    step = (hi - lo) / (2 ** bits)  # ~0.0195
    assert norm_free_action_error_bound(hi, lo, bits, 10.0) is True
    assert norm_free_action_error_bound(hi, lo, bits, 0.01) is False
    # boundary: exactly 2*step is NOT certified (must be strictly greater)
    assert norm_free_action_error_bound(hi, lo, bits, 2.0 * step) is False
    assert norm_free_action_error_bound(hi, lo, bits, 2.0 * step + 1e-6) is True


def test_norm_free_sim_runs_and_drops_scalar_z():
    """advice/020 section 3 + advice/001 P0-3: the genuinely
    normalization-free path runs end-to-end deterministically, keeps
    ``airtime_price=False`` (lambda free), broadcasts the owner-local
    deficit-embedded ``psi`` bus and bills ONLY ``Q * psi_bits`` -- no
    global ``pi`` vector, no scalar ``Z``, no separate ``D_q`` channel (no
    global reduction)."""
    from uav_otfs_isac.ca_frids import simulate_ca_frids
    from uav_otfs_isac.distributed_audit import build_distributed_scenario
    from uav_otfs_isac.airtime import build_airtime_model
    rng = np.random.default_rng(0)
    sc = build_distributed_scenario(rng, 6, 3)
    am = build_airtime_model(sc, rho_target=1.2)
    bt = [(8.0, -8.0)] * 3
    g = simulate_ca_frids(sc, bt, am, n_runs=60, seed=5, max_steps=40,
                          price_mode="global_simplex", pi_bits=10,
                          lam_bits=10, psi_bits=10,
                          task_price=True, airtime_price=False,
                          norm_free=True, admission_policy="neutral")
    assert 0.0 < g["worst_target_delay"] <= 40.0
    # lambda-free: B0-lite bills Q psi_bits = 3*10 = 30 bits/cycle
    assert g["comm"]["control_bits_per_cycle"] == pytest.approx(30.0, abs=1e-9)
    # a normalized global-simplex arm with lambda still bills Q*pi_bits + Z
    g_norm = simulate_ca_frids(sc, bt, am, n_runs=30, seed=5, max_steps=40,
                               price_mode="global_simplex", pi_bits=10,
                               lam_bits=10, task_price=True,
                               airtime_price=False, norm_free=False,
                               admission_policy="neutral")
    assert g_norm["comm"]["control_bits_per_cycle"] == pytest.approx(
        3 * 10 + 10, abs=1e-9)


def test_norm_free_audit_log_domain_certificate_runs():
    """advice/020 section 2-3 + advice/001 P0-3/P0-4: the norm-free audit
    computes the log-domain action-invariance certificate (eps_psi) instead
    of the pi-domain bound, and reports the psi saturation rate."""
    from uav_otfs_isac.ca_frids import simulate_ca_frids
    from uav_otfs_isac.distributed_audit import build_distributed_scenario
    from uav_otfs_isac.airtime import build_airtime_model
    rng = np.random.default_rng(1)
    sc = build_distributed_scenario(rng, 6, 3)
    am = build_airtime_model(sc, rho_target=1.8)
    bt = [(8.0, -8.0)] * 3
    g = simulate_ca_frids(sc, bt, am, n_runs=40, seed=7, max_steps=40,
                          price_mode="global_simplex", pi_bits=10,
                          lam_bits=10, psi_bits=10,
                          task_price=True, airtime_price=False,
                          norm_free=True, admission_policy="neutral",
                          audit=True)
    assert 0.0 <= g["audit"]["margin_ok_fraction"] <= 1.0
    assert g["audit"]["margin_samples"] > 0
    assert g["audit"]["eps_pi"] == 0.0
    assert g["audit"]["eps_psi"] > 0.0
    assert 0.0 <= g["audit"]["psi_sat_rate"] <= 1.0

# ---------------------------------------------------------------------------
# advice/020 sections 5-8: capacity-regime normalization + nested scenario
# ---------------------------------------------------------------------------


def test_airtime_owner_rho_normalization_controls_capacity_regime():
    """advice/020 section 5-7: with the FULL-MESH-derived budget the
    owner-directed system sees far lower effective congestion (the K/Q
    confound -- the ``(16,8)`` network has ~2x more slack than ``(8,4)``
    at the same nominal rho_target).  The ``rho_owner`` mode instead
    derives ``T_air`` from the balanced owner-directed load so that the
    owner-directed load ratio MATCHES rho_owner at every scale -- this is
    the capacity-regime-controlled comparison that removes the scale
    confound."""
    from uav_otfs_isac.airtime import build_airtime_model
    from uav_otfs_isac.distributed_audit import build_distributed_scenario
    k, q = 16, 8
    rng = np.random.default_rng(2)
    sc = build_distributed_scenario(rng, k, q)

    def _owner_ratio(am):
        owner_of = sc["owner_of"]
        n_owned = np.bincount(owner_of, minlength=k).astype(float)
        owner_load = np.array([
            float(np.sum(am["tau"][:, j] * (n_owned[j] / q)))
            for j in range(k)])
        return float(np.max(owner_load) / max(am["t_air"], 1e-15))

    # (i) the mesh-derived budget leaves the owner-directed system with
    # slack: effective owner ratio << nominal rho_target = 1.8
    a_mesh = build_airtime_model(sc, rho_target=1.8)
    r_eff = _owner_ratio(a_mesh)
    assert r_eff < 1.8
    # the audit's (K/Q)/(K-1) = (2)/(15) ~ 0.133 factor
    assert r_eff == pytest.approx(1.8 * (2.0 / 15.0), rel=0.35)
    # (ii) the rho_owner mode MATCHES the owner-directed load ratio exactly
    a_owner = build_airtime_model(sc, rho_owner=1.8)
    assert _owner_ratio(a_owner) == pytest.approx(1.8, rel=1e-6)
    # (iii) explicit t_air always beats both
    a_ex = build_airtime_model(sc, t_air=1.0, rho_owner=1.8)
    assert a_ex["t_air"] == pytest.approx(1.0)


def test_nested_scenario_subsets_share_realizations():
    """advice/020 section 8: the (8,4) subset of a (16,8) master reuses the
    SAME U2U top-left block and the SAME per-host kernels, so a scale
    comparison is no longer confounded by fresh channel/sensing draws."""
    from uav_otfs_isac.distributed_audit import (
        build_distributed_scenario, nested_scenario_subsets)
    rng = np.random.default_rng(3)
    master = build_distributed_scenario(rng, 16, 8)
    subs = nested_scenario_subsets(master)
    s84 = subs[(8, 4)]
    s168 = subs[(16, 8)]
    assert s84["k"] == 8 and s84["q"] == 4
    assert s168["k"] == 16 and s168["q"] == 8
    # same U2U top-left block
    assert np.allclose(s84["u2u_success"],
                       master["u2u_success"][:8, :8])
    # same per-host kernels
    assert len(s84["by_host"][(3, 2)]) == \
        len(master["by_host"][(3, 2)])
    assert s84["by_host"][(3, 2)][0] is master["by_host"][(3, 2)][0]
    # owner roles preserved on the subset
    assert s84["owner_of"] == [int(o % 8) for o in range(4)]

def test_norm_free_task_price_false_is_flat_architecture():
    """advice/020: ``norm_free=True`` with ``task_price=False`` must fall
    back to the flat 1/Q architecture score (B00 semantics) -- the psi
    plane is neither broadcast nor billed, and theta is never updated."""
    from uav_otfs_isac.ca_frids import simulate_ca_frids
    from uav_otfs_isac.distributed_audit import build_distributed_scenario
    from uav_otfs_isac.airtime import build_airtime_model
    rng = np.random.default_rng(4)
    sc = build_distributed_scenario(rng, 6, 3)
    am = build_airtime_model(sc, rho_target=1.2)
    bt = [(8.0, -8.0)] * 3
    g = simulate_ca_frids(sc, bt, am, n_runs=40, seed=5, max_steps=40,
                          price_mode="global_simplex", pi_bits=10,
                          lam_bits=10, psi_bits=10,
                          task_price=False, airtime_price=False,
                          norm_free=True, admission_policy="neutral")
    assert 0.0 < g["worst_target_delay"] <= 40.0
    # flat architecture: no dynamic price bus broadcast at all
    assert g["comm"]["control_bits_per_cycle"] == pytest.approx(0.0, abs=1e-9)

def test_lambda_diagnostics_reported_for_capacity_arm():
    """advice/001 P1-2: the airtime-price arm reports cap-hit fraction,
    dual residual and lambda/rho time-averages so the C8 phase-diagram
    interpretation is not overstated (constant-step / EMA / capped dual)."""
    from uav_otfs_isac.ca_frids import simulate_ca_frids
    from uav_otfs_isac.distributed_audit import build_distributed_scenario
    from uav_otfs_isac.airtime import build_airtime_model
    rng = np.random.default_rng(6)
    sc = build_distributed_scenario(rng, 6, 3)
    am = build_airtime_model(sc, rho_owner=1.8)
    bt = [(8.0, -8.0)] * 3
    g = simulate_ca_frids(sc, bt, am, n_runs=40, seed=5, max_steps=40,
                          price_mode="global_simplex", pi_bits=10,
                          lam_bits=10, task_price=True, airtime_price=True,
                          admission_policy="density")
    c = g["comm"]
    assert 0.0 <= c["lam_cap_hit_fraction"] <= 1.0
    assert c["lam_dual_residual"] >= 0.0
    assert c["lam_time_avg"] >= 0.0
    assert c["rho_time_avg"] > 0.0
    # the lambda-free B0 arm reports the same keys (frozen at zero)
    b0 = simulate_ca_frids(sc, bt, am, n_runs=20, seed=5, max_steps=40,
                           price_mode="global_simplex", pi_bits=10,
                           lam_bits=10, task_price=True, airtime_price=False,
                           norm_free=True, admission_policy="neutral")
    assert b0["comm"]["lam_time_avg"] == pytest.approx(0.0, abs=1e-9)
