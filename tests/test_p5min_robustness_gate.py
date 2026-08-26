"""P5-MIN cross-seed gate regression tests (advice/019 sections 5, 7, 8).

Locks the multi-seed aggregation primitives of the minimality gate so the
cross-seed claims are reproducible:

1. ``_pooled_blocks`` concatenation: pooling per-block arrays ACROSS test
   seeds equals pooling all runs directly (the bootstrap resampling unit
   is the block, advice/019 section 8 -- concatenating seeds widens the
   block population without changing the estimand).
2. ``_verdict`` minimality criteria: delay tolerance, norm-free ~ B0
   equivalence, control-bit reduction, airtime feasibility.
3. ``pool_h1_delay_risk`` charge rule is reused by the gate's J_risk.
"""

import numpy as np
import pytest

from scripts.run_p5a_ablation_ladder import _pooled_j
from scripts.run_p5min_robustness_gate import (
    _budget_feasible,
    _pooled_blocks,
    _verdict,
)
from uav_otfs_isac.qos import pool_h1_delay_risk


def _block_like(n, s):
    """Return a fake arm result dict shaped like ``_run_arm`` output."""
    return {"block_n": [np.asarray(x, dtype=float) for x in n],
            "block_s": [np.asarray(x, dtype=float) for x in s],
            "block_risk": [np.asarray(x, dtype=float) for x in s],
            "ctrl_bits_per_cycle": 80.0,
            "qos": "UNCERTAIN",
            "budget_feasible": 1.0}


def _pooled_j_from_seed_results(seed_results):
    N = np.sum(np.stack(_pooled_blocks(seed_results, "block_n"), axis=0),
               axis=0)
    S = np.sum(np.stack(_pooled_blocks(seed_results, "block_s"), axis=0),
               axis=0)
    return _pooled_j(N, S)


def test_pooled_blocks_across_seeds_matches_direct_pool():
    """Concatenating per-block arrays across test seeds and pooling must
    give the SAME J as pooling every run directly (the estimand is
    invariant to how the 12000 runs are grouped into blocks)."""
    rng = np.random.default_rng(3)
    # two test seeds, each 2 blocks of 3 targets
    seed_results = []
    direct_n = []
    direct_s = []
    for s in range(2):
        blocks_n, blocks_s = [], []
        for _ in range(2):
            n = np.array([100.0, 120.0, 90.0]) + rng.integers(0, 5, 3)
            mu = np.array([3.0, 4.0, 5.0])
            s_ = n * mu * rng.uniform(0.9, 1.1, 3)
            blocks_n.append(n)
            blocks_s.append(s_)
            direct_n.append(n)
            direct_s.append(s_)
        seed_results.append({"arms": {"B0-lite": _block_like(
            blocks_n, blocks_s)}})
    # via cross-seed block concatenation
    j_cross = _pooled_j_from_seed_results(
        [r["arms"]["B0-lite"] for r in seed_results])
    # via direct all-run pool
    N = np.sum(np.stack(direct_n, axis=0), axis=0)
    S = np.sum(np.stack(direct_s, axis=0), axis=0)
    j_direct = _pooled_j(N, S)
    assert j_cross == pytest.approx(j_direct, abs=1e-12)
    # and the concatenation has 4 blocks (2 seeds x 2 blocks)
    assert len(_pooled_blocks([r["arms"]["B0-lite"] for r in seed_results],
                              "block_n")) == 4


def test_verdict_delay_minimality_within_tolerance():
    """B0-lite within delta_J of C passes the delay criterion even when
    slightly slower."""
    cell = {
        "arms": {
            "B0-lite": {"J": 3.0, "control_bits_per_cycle": 80.0,
                        "budget_feasible": 1.0},
            "C": {"J": 2.98, "control_bits_per_cycle": 250.0,
                  "budget_feasible": 1.0},
            "B0": {"J": 3.0, "control_bits_per_cycle": 80.0,
                   "budget_feasible": 1.0},
        },
        "deltas": {
            # J_C - J_lite = -0.02 (within delta_J=0.05 -> OK)
            "D_C_minus_lite": {"point": -0.02, "state": "UNRESOLVED"},
            # norm-free ~ normalized: J_B0 - J_lite = 0.0
            "D_B0_minus_lite": {"point": 0.0, "state": "UNRESOLVED"},
        },
        "minimality_frac": 1.0,
    }
    v = _verdict(cell, delta_j=0.05)
    assert v["pass"] is True


def test_verdict_fails_on_certified_normfree_loss():
    """If B0-lite is CERTIFIED slower than normalized B0, the norm-free
    reparameterization is broken -> verdict must fail."""
    cell = {
        "arms": {
            "B0-lite": {"J": 3.5, "control_bits_per_cycle": 80.0,
                        "budget_feasible": 1.0},
            "C": {"J": 3.0, "control_bits_per_cycle": 250.0,
                  "budget_feasible": 1.0},
            "B0": {"J": 3.1, "control_bits_per_cycle": 80.0,
                   "budget_feasible": 1.0},
        },
        "deltas": {
            "D_C_minus_lite": {"point": -0.5, "state": "CERTIFIED_LOSS"},
            "D_B0_minus_lite": {"point": -0.4, "state": "CERTIFIED_LOSS"},
        },
        "minimality_frac": 0.0,
    }
    v = _verdict(cell, delta_j=0.05)
    assert v["pass"] is False
    assert "CERTIFIED slower" in v["reason"]


def test_verdict_fails_when_control_bits_not_reduced():
    """B0-lite must have strictly fewer control bits than C (the whole
    point of dropping the global normalizer Z)."""
    cell = {
        "arms": {
            "B0-lite": {"J": 3.0, "control_bits_per_cycle": 250.0,
                        "budget_feasible": 1.0},
            "C": {"J": 3.0, "control_bits_per_cycle": 250.0,
                  "budget_feasible": 1.0},
            "B0": {"J": 3.0, "control_bits_per_cycle": 80.0,
                   "budget_feasible": 1.0},
        },
        "deltas": {
            "D_C_minus_lite": {"point": 0.0, "state": "UNRESOLVED"},
            "D_B0_minus_lite": {"point": 0.0, "state": "UNRESOLVED"},
        },
        "minimality_frac": 1.0,
    }
    v = _verdict(cell, delta_j=0.05)
    assert v["pass"] is False
    assert "control bits" in v["reason"]


def test_budget_feasible_reads_shared_field():
    assert _budget_feasible({"budget_feasible": 0.999}) == pytest.approx(0.999)
    assert _budget_feasible({}) == pytest.approx(1.0)


def test_gate_j_risk_uses_shared_h1_risk_pool():
    """The gate's J_risk reuses ``pool_h1_delay_risk`` (an H1 miss is
    charged T_max exactly), so the cross-seed J_risk is the same estimand
    the schedulers report."""
    declared = np.array([[1.0, 0.0], [0.0, 1.0]])
    delays = np.array([[5.0, 3.0], [4.0, 6.0]])
    H = np.array([[True, True], [True, True]])
    risk = pool_h1_delay_risk(declared, delays, H, 40.0)
    assert risk[0] == pytest.approx(5.0 + 40.0)
    assert risk[1] == pytest.approx(40.0 + 6.0)
def test_verdict_fails_when_action_distortion_not_certified():
    """advice/020 section 2-3: when --audit is on, the verdict must FAIL
    if B0-lite's finite-bit broadcast actually flips any action
    (action_change_rate > max_action_change) -- the deployed norm-free form
    is an approximation whose action distortion must be certified."""
    cell = {
        "arms": {
            "B0-lite": {"J": 3.0, "control_bits_per_cycle": 80.0,
                        "budget_feasible": 1.0,
                        "audit": {"margin_ok_fraction": 0.90,
                                  "action_change_rate": 0.005,
                                  "margin_samples": 100}},
            "C": {"J": 2.98, "control_bits_per_cycle": 250.0,
                  "budget_feasible": 1.0},
            "B0": {"J": 3.0, "control_bits_per_cycle": 80.0,
                   "budget_feasible": 1.0},
        },
        "deltas": {
            "D_C_minus_lite": {"point": -0.02, "state": "UNRESOLVED"},
            "D_B0_minus_lite": {"point": 0.0, "state": "UNRESOLVED"},
        },
        "minimality_frac": 1.0,
    }
    # action_change_rate 0.005 > max_action_change 0.0 -> FAIL
    v = _verdict(cell, delta_j=0.05)
    assert v["pass"] is False
    assert "action_change_rate" in v["reason"]
    # within the tolerance it passes
    v2 = _verdict(cell, delta_j=0.05, max_action_change=0.01)
    assert v2["pass"] is True
    # optional margin gate fails below the floor
    v3 = _verdict(cell, delta_j=0.05, max_action_change=0.01,
                  min_margin_ok=0.95)
    assert v3["pass"] is False
    assert "margin certificate" in v3["reason"]


def test_verdict_passes_when_no_audit_collected():
    """Without --audit the B0-lite arm has no audit data, so criterion (v)
    is skipped and a correct cell passes."""
    cell = {
        "arms": {
            "B0-lite": {"J": 3.0, "control_bits_per_cycle": 80.0,
                        "budget_feasible": 1.0},
            "C": {"J": 2.98, "control_bits_per_cycle": 250.0,
                  "budget_feasible": 1.0},
            "B0": {"J": 3.0, "control_bits_per_cycle": 80.0,
                   "budget_feasible": 1.0},
        },
        "deltas": {
            "D_C_minus_lite": {"point": -0.02, "state": "UNRESOLVED"},
            "D_B0_minus_lite": {"point": 0.0, "state": "UNRESOLVED"},
        },
        "minimality_frac": 1.0,
    }
    v = _verdict(cell, delta_j=0.05)
    assert v["pass"] is True


def test_pooled_audit_aggregates_and_skips_missing():
    """_pooled_audit pools margin_ok across all block audits (weighted by
    sample count, using the real per-block audit schema) and returns None
    when nothing was collected (--audit off)."""
    from scripts.run_p5min_robustness_gate import _pooled_audit
    audits = [
        {"margin_samples": 50, "margin_ok_fraction": 0.96,
         "action_change_rate": 0.04, "n_cycles": 5, "eps_pi": 0.0,
         "eps_theta": 0.02},
        {"margin_samples": 50, "margin_ok_fraction": 1.0,
         "action_change_rate": 0.0, "n_cycles": 5, "eps_pi": 0.0,
         "eps_theta": 0.02},
    ]
    pooled = _pooled_audit(audits)
    assert pooled["margin_samples"] == 100
    assert pooled["margin_ok_fraction"] == pytest.approx(0.98)
    assert pooled["eps_theta"] == pytest.approx(0.02)
    assert _pooled_audit([None, None]) is None
    assert _pooled_audit([{}]) is None
    # empty-sample audits are ignored (not collected)
    assert _pooled_audit([{"margin_samples": 0}]) is None

# ---------------------------------------------------------------------------
# advice/020 section 12: hierarchical bootstrap + cell sign consistency
# ---------------------------------------------------------------------------


def test_hierarchical_bootstrap_runs_and_bounds_point():
    """advice/020 section 12: the hierarchical (cell -> seed -> block)
    bootstrap returns a point equal to the full-pool delta and a finite
    CI that widens with cell/seed variation (no crash on a synthetic
    two-cell grid)."""
    from scripts.run_p5a_ablation_ladder import hierarchical_block_bootstrap
    # prev (B0-lite) delay sums vs cur (C) -- construct two cells, two
    # seeds, two blocks each; cur slightly smaller (a gain) so the delta
    # is positive.
    def _cell(base):
        seeds = []
        for s in range(2):
            # prev J = max(40,44)/4 = 11 on both blocks
            pn = [np.array([4.0, 4.0])] * 2
            pv = [np.array([40.0, 44.0])] * 2
            # cur J differs across blocks so the block DELTA varies:
            #   block0 cur J = 10  -> delta 1.0
            #   block1 cur J = 10.25 -> delta 0.75
            cn = [np.array([4.0, 4.0])] * 2
            cv = [np.array([36.0, 40.0]), np.array([38.0, 41.0])]
            seeds.append((pn, pv, cn, cv))
        return {"seed_blocks": seeds}
    cells = [_cell(0), _cell(1)]
    point, lo, hi = hierarchical_block_bootstrap(cells, n_boot=300, seed=1)
    assert lo <= point <= hi
    assert lo < hi
    assert point == pytest.approx(0.875, abs=1e-9)


def test_cell_sign_consistency_primary_summary():
    """advice/020 section 12: cell sign consistency is the recommended
    primary cross-scenario summary -- a pooled CI alone is NOT a
    cross-geometry generalization claim."""
    from scripts.run_p5a_ablation_ladder import cell_sign_consistency
    deltas = [
        {"point": 0.5, "state": "CERTIFIED_GAIN"},
        {"point": 0.3, "state": "CERTIFIED_GAIN"},
        {"point": -0.1, "state": "UNRESOLVED"},
    ]
    s = cell_sign_consistency(deltas)
    assert s["n_cells"] == 3
    assert s["certified_gain_cells"] == 2
    assert s["certified_loss_cells"] == 0
    assert s["sign_consistency"] == pytest.approx(2.0 / 3.0)
    assert s["cell_points"] == [0.5, 0.3, -0.1]
    empty = cell_sign_consistency([])
    assert empty["n_cells"] == 0

def test_hierarchical_delta_over_cells_sign_matches_pooled():
    """advice/020 section 12 regression: hierarchical_D_C_minus_lite must
    have the SAME sign as the pooled D_C_minus_lite = J_C - J_B0-lite.  A
    cell where B0-lite is faster (J_lite < J_C) must yield a POSITIVE
    hierarchical point, matching the pooled convention."""
    from scripts.run_p5min_robustness_gate import (
        hierarchical_delta_over_cells, _pooled_delta_ci)
    # Build a cell: 1 seed, 2 blocks.  C (cur) worse than B0-lite (prev in
    # the gate's pooled convention) so D_C_minus_lite > 0.
    def _blocks():
        # J_lite = max(30,34)/4 = 8.5 ; J_C = max(40,44)/4 = 11
        lite_n = [np.array([4.0, 4.0])] * 2
        lite_s = [np.array([30.0, 34.0])] * 2
        c_n = [np.array([4.0, 4.0])] * 2
        c_s = [np.array([40.0, 44.0])] * 2
        return lite_n, lite_s, c_n, c_s
    lite_n, lite_s, c_n, c_s = _blocks()
    cell = {"seed_results": [{"arms": {
        "B0-lite": {"block_n": lite_n, "block_s": lite_s},
        "C": {"block_n": c_n, "block_s": c_s},
    }}]}
    hier = hierarchical_delta_over_cells([cell], n_boot=200, seed=3)
    # pooled reference: D_C_minus_lite = J_C - J_lite = 11 - 8.5 = 2.5
    pooled_point, _, _ = _pooled_delta_ci(c_n, c_s, lite_n, lite_s)
    assert hier["point"] == pytest.approx(pooled_point, abs=1e-9)
    assert hier["point"] > 0.0
