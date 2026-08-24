"""P4.2b three-state frontier tests (advice/017 section 12.1).

Tests the module-level frontier classifier ``classify_frontier_state`` and
the paired held-out reduction bootstrap ``paired_reduction_bootstrap`` of
``scripts/run_p42b_qos_frontier_gate.py``:

- certified feasible needs every target with a certified-feasible m_star;
- certified infeasible needs a certified violation persisting at the
  FA-most-favorable (largest A_q) or MD-most-favorable (smallest A_q)
  extreme -- the A_q lever cannot clear the spec;
- unresolved otherwise: "not certified feasible" only, infeasibility NOT
  claimed;
- the scheduler state is certified-infeasible if any target is certified
  infeasible, certified feasible if all targets are; unresolved otherwise.
"""

import numpy as np
import pytest

from scripts.run_p42b_qos_frontier_gate import (
    classify_frontier_state,
    paired_reduction_bootstrap,
)

SPEC = 0.05


def _f(m_star, lc_lo, lc_hi):
    return classify_frontier_state(m_star, lc_lo, lc_hi, SPEC)


def test_all_targets_feasible_is_certified_feasible():
    m_star = [1.0, 1.5, 1.0]
    lc_fa = [0.02, 0.03, 0.01]
    lc_md = [0.02, 0.03, 0.02]
    target, sched = _f(m_star, lc_fa, lc_md)
    assert target == ["CERTIFIED FEASIBLE"] * 3
    assert sched == "CERTIFIED FEASIBLE"


def test_infeasible_target_dominates_scheduler_state():
    # target 1 has no m_star AND an FA LCB above spec at the largest A_q
    # (most-FA-favorable extreme) -> certified infeasible; the scheduler
    # state must be CERTIFIED INFEASIBLE even though other targets are
    # certified feasible.
    m_star = [1.0, None, 1.0]
    lc_fa = [0.02, 0.051, 0.01]   # LCB_FA(max) > 0.05
    lc_md = [0.02, 0.03, 0.02]
    target, sched = _f(m_star, lc_fa, lc_md)
    assert target[1] == "CERTIFIED INFEASIBLE"
    assert sched == "CERTIFIED INFEASIBLE"


def test_md_infeasibility_at_min_extreme():
    # no m_star AND MD LCB above spec at the smallest A_q (most-MD-favorable
    # extreme) -> certified infeasible.
    m_star = [None, 1.0, None]
    lc_fa = [0.04, 0.02, 0.04]
    lc_md = [0.052, 0.02, 0.03]   # LCB_MD(min) > 0.05 for target 0
    target, sched = _f(m_star, lc_fa, lc_md)
    assert target[0] == "CERTIFIED INFEASIBLE"
    assert sched == "CERTIFIED INFEASIBLE"


def test_no_m_star_without_lcb_violation_is_unresolved():
    # target 1 has no certified-feasible m_star, but neither extreme LCB
    # exceeds spec -> UNRESOLVED (NOT "infeasible").
    m_star = [1.0, None, 1.0]
    lc_fa = [0.02, 0.049, 0.01]
    lc_md = [0.02, 0.049, 0.02]
    target, sched = _f(m_star, lc_fa, lc_md)
    assert target[1] == "UNRESOLVED"
    assert sched == "UNRESOLVED"


def test_unresolved_not_relabelled_infeasible():
    # the entire wording discipline: a target that merely has no swept
    # certified-feasible point must never be classified infeasible unless an
    # LCB violation certifies it.
    m_star = [None, None, None]
    lc_fa = [0.04, 0.04, 0.04]
    lc_md = [0.04, 0.04, 0.04]
    target, sched = _f(m_star, lc_fa, lc_md)
    assert "CERTIFIED INFEASIBLE" not in target
    assert sched == "UNRESOLVED"


def test_boundary_lcb_exactly_at_spec_is_not_infeasible():
    # the strict comparison is LCB > spec (strictly), so an LCB exactly at
    # spec is NOT a certified violation.
    m_star = [None]
    lc_fa = [0.05]
    lc_md = [0.02]
    target, sched = _f(m_star, lc_fa, lc_md)
    assert target == ["UNRESOLVED"]
    assert sched == "UNRESOLVED"


def test_paired_reduction_bootstrap_reports_ci():
    # held-out blocks share the same CRN stream for v2 and CA; the paired
    # per-block bootstrap CI of the observed reduction must bracket the
    # point estimate and the observed scalar must equal the block-mean.
    v2_j = [7.4, 7.5, 7.3, 7.6, 7.4, 7.5, 7.2, 7.5]
    ca_j = [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.4]
    obs, lo, hi = paired_reduction_bootstrap(v2_j, ca_j, n_boot=500, seed=7)
    r_blocks = [(a - b) / a for a, b in zip(v2_j, ca_j)]
    assert obs == pytest.approx(np.mean(r_blocks), rel=1e-12)
    assert lo <= obs <= hi
    assert 0.0 <= lo <= hi <= 1.0