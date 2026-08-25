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