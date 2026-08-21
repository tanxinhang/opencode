"""Gate F0-G8A tests: evidence-dependence audit (advice/013)."""

import numpy as np
import pytest

from uav_otfs_isac.evidence_correlation import (
    alignment,
    build_delta_from_scenario,
    conditional_marginal,
    joint_kl,
    redundancy,
    sample_joint_kl,
    sequential_correlation_check,
    singleton_kl,
)
from uav_otfs_isac.distributed_audit import build_distributed_scenario


def test_singleton_kl():
    assert singleton_kl(0.0) == pytest.approx(0.0)
    assert singleton_kl(1.0) == pytest.approx(0.5)
    assert singleton_kl(2.0) == pytest.approx(2.0)


def test_joint_kl_reduces_to_sum_for_rho_zero():
    delta = np.array([1.0, 2.0, 0.5])
    G = joint_kl(delta, 0.0)
    assert G == pytest.approx(np.sum(delta ** 2) / 2.0)
    assert redundancy(delta, [0, 1, 2], 0.0) == pytest.approx(0.0)


def test_joint_kl_singleton_equals_marginal():
    delta = np.array([1.0, 2.0, 0.5])
    for i in range(3):
        assert joint_kl(delta[[i]], 0.5) == pytest.approx(singleton_kl(delta[i]))


def test_closed_form_matches_sampling():
    """The Sherman-Morrison closed form equals the Monte-Carlo estimate
    of the joint KL for several correlation levels."""
    rng = np.random.default_rng(0)
    for _ in range(4):
        delta = rng.uniform(0.2, 2.0, size=5)
        for rho in (0.0, 0.2, 0.5, 0.8):
            closed = joint_kl(delta, rho)
            sampled = sample_joint_kl(delta, rho, n=300000, seed=1)
            assert closed == pytest.approx(sampled, abs=5e-3)


def test_redundancy_grows_with_correlation_and_size():
    """R_q(S) = 0 at rho=0; increases with rho_s and (for equal deltas)
    with the coalition size; the full-redundancy ceiling appears as
    rho -> 1 for a large equal set."""
    delta = np.ones(4)
    assert redundancy(delta, [0, 1, 2, 3], 0.0) == pytest.approx(0.0)
    r02 = redundancy(delta, [0, 1, 2, 3], 0.2)
    r05 = redundancy(delta, [0, 1, 2, 3], 0.5)
    r08 = redundancy(delta, [0, 1, 2, 3], 0.8)
    assert r02 < r05 < r08
    # same rho, larger set -> more redundancy (equal deltas)
    assert redundancy(delta, [0, 1], 0.5) < redundancy(delta, [0, 1, 2, 3], 0.5)


def test_redundancy_closed_form_values():
    # two equal UAVs, rho = 0.5: R = (0.5/0.5)*(2/(1+0.5) - 1) = 1/3
    delta = np.array([1.0, 1.0])
    assert redundancy(delta, [0, 1], 0.5) == pytest.approx(1.0 / 3.0, abs=1e-9)
    # rho = 0.8, two equal: R = (0.8/0.2)*(2/1.8 - 1) = 4*0.11111 = 0.4444
    assert redundancy(delta, [0, 1], 0.8) == pytest.approx(4.0 / 9.0, abs=1e-9)


def test_alignment():
    assert alignment(np.array([1.0, 1.0, 1.0])) == pytest.approx(3.0)
    assert alignment(np.array([3.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert 1.0 <= alignment(np.array([1.0, 2.0, 3.0])) <= 3.0


def test_conditional_marginal_nonnegative():
    """Delta G_{i|S,q} >= 0 (KL chain rule) and equals the singleton at
    rho = 0 (independent).  It is NOT bounded above by the singleton in
    general -- at high correlation the conditioning can make UAV i MORE
    discriminable than its marginal (G is not submodular; the advice/013
    warning against assuming diminishing returns)."""
    delta = np.array([1.0, 2.0, 1.5])
    for rho in (0.0, 0.5, 0.8):
        for i in range(3):
            other = [j for j in range(3) if j != i]
            dm = conditional_marginal(delta, other, i, rho)
            assert dm >= -1e-12
    # rho = 0: the conditional marginal equals the singleton (independent)
    for i in range(3):
        other = [j for j in range(3) if j != i]
        assert conditional_marginal(delta, other, i, 0.0) == pytest.approx(
            singleton_kl(delta[i]), abs=1e-9)
    # non-submodularity witness: at rho = 0.8 the marginal of the
    # smallest-delta UAV can exceed its singleton
    dm = conditional_marginal(delta, [1, 2], 0, 0.8)
    assert dm > singleton_kl(delta[0])


def test_scenario_delta_profile():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    delta = build_delta_from_scenario(sc, sc["owner_of"])
    assert delta.shape == (6, 3)
    assert np.all(delta >= 0.0)
    assert np.all(np.isfinite(delta))


def test_sequential_check_reproduces_wald_delay():
    """The correlated stream is slower than the independent baseline; the
    measured delay ratio is consistent with the analytic 1/(1-R) scaling
    up to the Wald overshoot correction, and both streams meet the
    error constraints at the Wald thresholds."""
    delta = np.array([1.0, 1.0, 1.0])     # equal, R at rho=0.5 = 1/3
    out = sequential_correlation_check(delta, 0.5, n_runs=2000,
                                       max_steps=300, seed=0)
    assert out["delay_ratio_measured"] > 1.2
    assert out["E1_T_correlated"] > out["E1_T_independent"]
    assert out["p_fa_independent"] <= 0.06
    assert out["p_fa_correlated"] <= 0.06
    assert out["p_md_independent"] <= 0.06
    assert out["p_md_correlated"] <= 0.06