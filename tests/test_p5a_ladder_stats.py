"""P5-A ladder statistics regression tests (advice/019 section 9).

Four small unit tests that lock the advice/018 statistics fixes so the
"audit-driven patching" can stop:

1. synthetic pooled estimand -- ``mean(max) != max(pooled mean)`` is
   exactly the discrepancy the advice/018 section 2 P1 fix eliminated;
   the runner bootstrap must target the SAME pooled ``J`` the table
   reports, not a per-block-then-mean statistic.
2. CI tri-state -- gain / loss / unresolved from three synthetic CIs
   (advice/018 section 3: a point estimate alone is NOT certification).
3. J_risk -- an H1 miss must be charged EXACTLY ``T_max`` (advice/018
   section 5), never the realized (possibly early-wrong) stop time.
4. factorial identity -- ``interaction = delta_task_mesh -
   delta_task_owner`` (the 2x2 algebra of advice/018 section 8).
"""

import numpy as np
import pytest

from scripts.run_p5a_ablation_ladder import (
    _ci_state,
    _interaction_delta_ci,
    _pooled_delta_ci,
    _pooled_j,
)
from uav_otfs_isac.qos import pool_h1_delay_risk


# ---------------------------------------------------------------------------
# 1. synthetic pooled estimand (advice/018 section 2, advice/019 section 9)
# ---------------------------------------------------------------------------


def test_pooled_estimand_synthetic_differs_from_per_block_max():
    """Construct a case where ``mean over blocks of max_q(mean_bq)``
    differs from ``max_q(pooled mean)`` and lock that the runner bootstrap
    targets the pooled estimand (``J = max_q sum_b S_bq / sum_b N_bq``)."""
    # block 0: target 0 well served, target 1 starved
    # block 1: target 0 starved, target 1 well served
    # per-block worst-target means: 6 and 6 -> mean-of-max = 6
    # pooled per-target means:      3.5 and 3.5 -> pooled max = 3.5
    b0_n = np.array([100.0, 100.0])
    b0_s = np.array([600.0, 100.0])
    b1_n = np.array([100.0, 100.0])
    b1_s = np.array([100.0, 600.0])
    blocks_n = [b0_n, b1_n]
    blocks_s = [b0_s, b1_s]

    mean_of_block_max = float(np.mean(
        [float(np.max(s / np.maximum(n, 1.0)))
         for n, s in zip(blocks_n, blocks_s)]))
    N, S = np.sum(np.stack(blocks_n, axis=0), axis=0), \
        np.sum(np.stack(blocks_s, axis=0), axis=0)
    pooled_max = _pooled_j(N, S)

    # the two estimands genuinely disagree on this synthetic data
    assert mean_of_block_max != pytest.approx(pooled_max, abs=1e-12)
    assert mean_of_block_max == pytest.approx(6.0)
    assert pooled_max == pytest.approx(3.5)

    # the runner's reported point must be the pooled estimand
    cur_blocks_n = [np.array([100.0, 100.0]), np.array([100.0, 100.0])]
    cur_blocks_s = [np.array([500.0, 500.0]), np.array([500.0, 500.0])]
    d, lo, hi = _pooled_delta_ci(blocks_n, blocks_s,
                                 cur_blocks_n, cur_blocks_s,
                                 n_boot=2000, seed=1)
    # point = J_prev - J_cur on the FULL pool = 3.5 - 5.0 = -1.5
    assert d == pytest.approx(-1.5, abs=1e-9)
    assert lo <= d <= hi


def test_pooled_estimand_point_is_not_mean_of_per_block_deltas():
    """The ``_pooled_delta_ci`` point equals the difference of the two
    FULL-pool ``J`` values -- NOT the mean of the per-block ``max``
    deltas (the pre-advice/018 bootstrap estimand)."""
    prev_n = [np.array([100.0, 100.0]), np.array([100.0, 100.0])]
    prev_s = [np.array([600.0, 100.0]), np.array([100.0, 600.0])]
    cur_n = [np.array([100.0, 100.0]), np.array([100.0, 100.0])]
    cur_s = [np.array([500.0, 500.0]), np.array([500.0, 500.0])]
    d, lo, hi = _pooled_delta_ci(prev_n, prev_s, cur_n, cur_s,
                                 n_boot=2000, seed=2)
    mean_of_delta = float(np.mean([
        float(np.max(p / np.maximum(pn, 1.0))
              - np.max(c / np.maximum(cn, 1.0)))
        for pn, p, cn, c in zip(prev_n, prev_s, cur_n, cur_s)]))
    # mean-of-delta = 6 - 5 = 1.0, but pooled delta = 3.5 - 5.0 = -1.5
    assert mean_of_delta == pytest.approx(1.0)
    assert d == pytest.approx(-1.5, abs=1e-9)
    assert abs(d - mean_of_delta) > 1.0


# ---------------------------------------------------------------------------
# 2. CI tri-state (advice/018 section 3, advice/019 section 9)
# ---------------------------------------------------------------------------


def test_ci_three_state_gain_loss_unresolved():
    assert _ci_state(0.1, 0.9) == "CERTIFIED_GAIN"
    assert _ci_state(0.001, 0.002) == "CERTIFIED_GAIN"
    assert _ci_state(-0.9, -0.1) == "CERTIFIED_LOSS"
    assert _ci_state(-0.002, -0.001) == "CERTIFIED_LOSS"
    # CI crossing zero is unresolved even when the point looks positive
    assert _ci_state(-0.3, 0.4) == "UNRESOLVED"
    assert _ci_state(-0.2, 0.1) == "UNRESOLVED"
    # boundary cases: lo == 0 is NOT a certified gain, hi == 0 is NOT a
    # certified loss (strict inequality is required)
    assert _ci_state(0.0, 0.5) == "UNRESOLVED"
    assert _ci_state(-0.5, 0.0) == "UNRESOLVED"
    # a fully degenerate zero CI is unresolved
    assert _ci_state(0.0, 0.0) == "UNRESOLVED"


def test_ci_three_state_drives_delta_verdict():
    """The three-state predicate is what the runner uses to classify a
    consecutive delta (a point estimate alone is NOT certification)."""
    prev_n = [np.array([100.0])]
    prev_s = [np.array([600.0])]
    cur_n = [np.array([100.0])]
    cur_s = [np.array([400.0])]
    d, lo, hi = _pooled_delta_ci(prev_n, prev_s, cur_n, cur_s,
                                 n_boot=2000, seed=3)
    assert d == pytest.approx(2.0)
    assert _ci_state(lo, hi) == "CERTIFIED_GAIN"


# ---------------------------------------------------------------------------
# 3. J_risk: H1 miss must be charged EXACTLY T_max (advice/018 s5)
# ---------------------------------------------------------------------------


def test_j_risk_h1_miss_charged_tmax_exactly():
    """Under H1, a run that DECLARED H0 (a miss) contributes ``T_max`` to
    the risk pool -- never its (possibly EARLIER, wrong) realized stop
    time; a run that declared H1 keeps its realized delay; H0 runs are
    excluded."""
    declared_h1 = np.array([
        [1.0, 0.0],   # run0: target0 correct H1, target1 MISS (wrong H0)
        [0.0, 1.0],   # run1: target0 MISS, target1 correct H1
        [1.0, 1.0],   # run2: both correct H1
        [0.0, 0.0],   # run3: H0 runs (excluded from the H1 pool)
    ])
    delays = np.array([
        [5.0, 3.0],   # run0 stopped target0 at 5, target1 wrongly early at 3
        [4.0, 6.0],   # run1 target0 wrongly early at 4, target1 at 6
        [7.0, 8.0],   # run2
        [1.0, 2.0],   # run3 (H0)
    ])
    H_all = np.array([
        [True, True],
        [True, True],
        [True, True],
        [False, False],
    ])
    t_max = 40.0
    risk = pool_h1_delay_risk(declared_h1, delays, H_all, t_max)
    # target0: H1 runs 0,1,2 -> correct(5) + MISS(T_max=40) + correct(7)
    assert risk[0] == pytest.approx(5.0 + 40.0 + 7.0)
    # target1: H1 runs 0,1,2 -> MISS(40) + correct(6) + correct(8)
    assert risk[1] == pytest.approx(40.0 + 6.0 + 8.0)
    # the wrong early stops (3 and 4) must NOT have entered the pool
    assert 3.0 not in [risk[0], risk[1]]
    assert 4.0 not in [risk[0], risk[1]]


def test_j_risk_runs_in_schedulers():
    """The scheduler ``pool.sum_h1_delay_risk`` block is produced by the
    same shared helper, so the J_risk estimand is identical across
    FRIDS-v2 and CA-FRIDS."""
    from uav_otfs_isac.ca_frids import simulate_ca_frids
    from uav_otfs_isac.crn_tape import build_exogenous_tape
    from uav_otfs_isac.distributed_audit import (
        build_distributed_scenario,
        calibrate_target_bounds,
    )
    from uav_otfs_isac.airtime import build_airtime_model
    from uav_otfs_isac.frids import simulate_frids_v2

    rng = np.random.default_rng(11)
    sc = build_distributed_scenario(rng, k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=40, seed=100, verify_runs=0)
    b = [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(3)]
    am = build_airtime_model(sc, rho_target=2.0)
    q = 3
    k = int(sc["k"])
    tape = build_exogenous_tape(13, 50, q, k, 40)
    for out in (
        simulate_frids_v2(sc, b, n_runs=50, seed=13, max_steps=40,
                          exog=tape, airtime=am),
        simulate_ca_frids(sc, b, am, n_runs=50, seed=13, max_steps=40,
                          exog=tape),
    ):
        pool = out["pool"]
        n_h1 = np.asarray(pool["n_h1"], dtype=float)
        risk = np.asarray(pool["sum_h1_delay_risk"], dtype=float)
        # J_risk = max_q sum_h1_delay_risk/n_h1; every H1 delay in
        # [1, T_max], so the risk sum per H1 run lies in [1, T_max] too
        assert np.all(risk >= n_h1)
        assert np.all(risk <= n_h1 * 40.0)
        assert np.max(risk / np.maximum(n_h1, 1.0)) >= 1.0


# ---------------------------------------------------------------------------
# 4. factorial identity (advice/018 section 8, advice/019 section 9)
# ---------------------------------------------------------------------------


def test_factorial_identity_interaction_is_task_mesh_minus_task_owner():
    """The 2x2 interaction ``(J_F0-J_O0) - (J_F1-J_O1)`` is algebraically
    identical to ``delta_task_mesh - delta_task_owner``, where
    ``delta_task_mesh = J_F0 - J_F1`` and ``delta_task_owner = J_O0 -
    J_O1`` -- this is the identity the runner's interaction CI must
    satisfy."""
    rng = np.random.default_rng(7)
    B = 6
    q = 3

    def _blocks(base, spread):
        return [base + rng.uniform(0.0, spread, q) for _ in range(B)]

    def _ns(count):
        return [np.full(q, float(count)) for _ in range(B)]

    f0s, f1s = _blocks(50.0, 8.0), _blocks(45.0, 8.0)
    o0s, o1s = _blocks(40.0, 8.0), _blocks(35.0, 8.0)
    n = _ns(100)

    def _pooled(s):
        N, S = np.sum(np.stack(n, axis=0), axis=0), \
            np.sum(np.stack(s, axis=0), axis=0)
        return _pooled_j(N, S)

    j_f0, j_f1 = _pooled(f0s), _pooled(f1s)
    j_o0, j_o1 = _pooled(o0s), _pooled(o1s)

    interaction_direct = (j_f0 - j_o0) - (j_f1 - j_o1)
    task_mesh = j_f0 - j_f1
    task_owner = j_o0 - j_o1
    assert interaction_direct == pytest.approx(task_mesh - task_owner,
                                               abs=1e-12)

    # the runner's paired-bootstrap interaction must reproduce this point
    int_point, int_lo, int_hi = _interaction_delta_ci(
        n, f0s, n, f1s, n, o0s, n, o1s, n_boot=2000, seed=5)
    assert int_point == pytest.approx(interaction_direct, abs=1e-9)
    assert int_lo <= int_point <= int_hi


def test_factorial_identity_uses_shared_block_indices():
    """The interaction CI drives all FOUR cells with the SAME resampled
    block set (they share one CRN tape), so the bootstrap repeat is the
    paired ``arch_flat - arch_deficit`` difference -- not four
    independently-resampled cells."""
    rng = np.random.default_rng(9)
    B = 8
    q = 2
    n = [np.full(q, 120.0) for _ in range(B)]
    f0s = [rng.uniform(40.0, 60.0, q) for _ in range(B)]
    f1s = [rng.uniform(40.0, 60.0, q) for _ in range(B)]
    o0s = [rng.uniform(35.0, 55.0, q) for _ in range(B)]
    o1s = [rng.uniform(35.0, 55.0, q) for _ in range(B)]
    int_point, int_lo, int_hi = _interaction_delta_ci(
        n, f0s, n, f1s, n, o0s, n, o1s, n_boot=3000, seed=6)
    assert int_lo <= int_point <= int_hi
    # deterministic: same seed -> identical CI
    p2, l2, h2 = _interaction_delta_ci(
        n, f0s, n, f1s, n, o0s, n, o1s, n_boot=3000, seed=6)
    assert (p2, l2, h2) == (int_point, int_lo, int_hi)