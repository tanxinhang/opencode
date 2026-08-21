"""Gate F0 tests: distributed information audit (advice/005.md)."""

import numpy as np
import pytest

from uav_otfs_isac.distributed_audit import (
    MODES,
    TOKEN_LLR_BITS,
    action_gain,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
    quantize_llr,
    quantized_kernels,
    run_audit,
    simulate_system,
    token_bits,
)

K_UAVS = 6
Q_TARGETS = 3


@pytest.fixture(scope="module")
def scenario():
    rng = np.random.default_rng(0)
    return build_distributed_scenario(rng, k_uavs=K_UAVS,
                                      q_targets=Q_TARGETS)


@pytest.fixture(scope="module")
def bounds(scenario):
    return calibrate_target_bounds(scenario, n_runs=120)


@pytest.fixture(scope="module")
def bounds_token(scenario):
    return calibrate_target_bounds(scenario, n_runs=120,
                                   llr_bits=TOKEN_LLR_BITS)


@pytest.fixture(scope="module")
def singles(scenario, bounds):
    return build_target_values(
        scenario, bounds, horizon=40,
        nu=tuple([1.0 / Q_TARGETS] * Q_TARGETS),
    )


def test_build_scenario_deterministic_and_shapes(scenario):
    again = build_distributed_scenario(np.random.default_rng(0),
                                       k_uavs=K_UAVS,
                                       q_targets=Q_TARGETS)
    assert scenario["k"] == K_UAVS and scenario["q"] == Q_TARGETS
    assert sorted(scenario["links"].keys()) == list(range(Q_TARGETS))
    for qq in range(Q_TARGETS):
        assert len(scenario["links"][qq]) == K_UAVS * 2  # two powers
        for kernel in scenario["links"][qq]:
            for key in ("p0", "p1", "llr", "i_plus", "cost", "host",
                        "target"):
                assert key in kernel
            assert len(kernel["p0"]) == len(kernel["p1"]) == \
                len(kernel["llr"])
    assert len(again["links"][0][0]["p0"]) == \
        len(scenario["links"][0][0]["p0"])
    assert scenario["owner_of"] == list(range(Q_TARGETS))
    u2u = scenario["u2u_success"]
    assert np.allclose(u2u, u2u.T)
    assert np.allclose(np.diag(u2u), 1.0)
    assert 0.6 <= u2u.min() and u2u.max() <= 1.0


def test_quantize_llr_midrise_bounded_error():
    # 2 bits over [-4, 4]: levels 4, step 2, midpoints -3,-1,1,3
    assert quantize_llr(0.0, bits=2, llr_range=4.0) == 1.0
    assert quantize_llr(3.9, bits=2, llr_range=4.0) == 3.0
    assert quantize_llr(-2.1, bits=2, llr_range=4.0) == -3.0
    rng = np.random.default_rng(1)
    step = 12.0 / 2 ** TOKEN_LLR_BITS
    for _ in range(200):
        x = float(rng.uniform(-5.0, 5.0))
        q = quantize_llr(x)
        assert abs(q - x) <= step / 2.0 + 1e-9
    assert quantize_llr(10.0) <= 6.0          # clipped at the range
    assert quantize_llr(-10.0) >= -6.0


def test_quantized_kernels_preserve_masses(scenario):
    for qq in range(Q_TARGETS):
        for kernel in scenario["links"][qq][:2]:
            qk = quantized_kernels([kernel])[0]
            assert np.array_equal(qk["p0"], kernel["p0"])
            assert np.array_equal(qk["p1"], kernel["p1"])
            assert np.all(np.abs(qk["llr"] - kernel["llr"])
                          <= 6.0 / 2 ** TOKEN_LLR_BITS)
            assert qk["cost"] == kernel["cost"]


def test_calibrate_target_bounds_sane(bounds, bounds_token):
    assert len(bounds) == Q_TARGETS
    assert len(bounds_token) == Q_TARGETS
    for (a, b), (at, bt) in zip(bounds, bounds_token):
        assert a > b                     # two-threshold ordering
        assert a > 0.0 and b < 0.0
        assert at > bt and at > 0.0 and bt < 0.0


def test_token_layout_counts():
    layout = token_bits()
    assert layout["q"] == 2
    assert layout["llr"] == TOKEN_LLR_BITS
    assert layout["total"] == sum(
        layout[k] for k in ("q", "llr", "u", "r", "chi", "intent", "stamp"))
    assert layout["full_message_total"] > layout["total"]


def test_action_gain_linear_in_nu(bounds, singles):
    kernel = singles[0]["actions"][0]
    l = 0.5
    g1 = action_gain(singles[0], kernel, l, 0, 8.0, nu_q=0.5, lam=1.0)
    g2 = action_gain(singles[0], kernel, l, 0, 8.0, nu_q=1.0, lam=1.0)
    # the delay values are 1e9-scaled inside the stopping band, so the
    # float cancellation noise is ~1e-7 relative
    assert g2 == pytest.approx(2.0 * g1, rel=1e-5)
    # a zero-cost-penalty target with nu -> 0 gives ~0 gain
    g0 = action_gain(singles[0], kernel, l, 0, 8.0, nu_q=0.0, lam=1.0)
    assert g0 == pytest.approx(-float(kernel["cost"]), rel=1e-9)


def test_simulate_four_modes_metric_ranges(scenario, bounds, bounds_token,
                                           singles):
    for mode in MODES:
        bnd = bounds_token if mode == "compact_token" else bounds
        out = simulate_system(mode, scenario, bnd, singles,
                              n_runs=80, seed=5, nu=(1 / 3, 1 / 3, 1 / 3))
        assert out["mode"] == mode
        assert 0.0 <= out["worst_target_delay"] <= 40.0
        assert len(out["e1_delays"]) == Q_TARGETS
        for p in out["p_fa"] + out["p_md"]:
            assert 0.0 <= p <= 1.0
        assert 0.0 <= out["conflict_rate"] <= 1.0
        assert 0.0 <= out["duplicate_sensing_rate"] <= 1.0
        assert out["role_switch_rate"] == 0.0
        assert out["mean_u2u_bits_per_cycle"] >= 0.0
        if mode == "centralized":
            assert out["belief_disagreement"] is None
        else:
            assert out["belief_disagreement"] is not None


def test_information_ordering_structure(scenario, bounds, bounds_token,
                                        singles):
    """Cooperation must matter (local-only far worse), and the
    zero-communication baseline must violate the error constraint where
    the token modes meet it (cooperation is necessary)."""
    nu = (1 / 3, 1 / 3, 1 / 3)
    rows = {}
    for mode in MODES:
        bnd = bounds_token if mode == "compact_token" else bounds
        rows[mode] = simulate_system(mode, scenario, bnd, singles,
                                     n_runs=120, seed=11, nu=nu)
    assert rows["centralized"]["worst_target_delay"] \
        < rows["local_only"]["worst_target_delay"]
    assert rows["full_message"]["worst_target_delay"] \
        <= rows["local_only"]["worst_target_delay"]
    # the token modes keep P_MD near the beta constraint; local-only blows
    # past it at the same horizon (censored runs)
    assert max(rows["compact_token"]["p_md"]) <= 0.05 + 0.03
    assert max(rows["full_message"]["p_md"]) <= 0.05 + 0.03
    assert max(rows["local_only"]["p_md"]) > 0.15


def test_token_4_bits_infeasible(scenario):
    """The token evidence with 4 bits cannot meet alpha=beta=0.05 -- the
    infeasible region finding of the audit."""
    with pytest.raises(ValueError):
        calibrate_target_bounds(scenario, n_runs=80, llr_bits=4)
    bounds5 = calibrate_target_bounds(scenario, n_runs=80, llr_bits=5)
    assert len(bounds5) == Q_TARGETS


def test_run_audit_smoke(scenario):
    audit = run_audit(scenario, n_runs=60, seeds=2, calib_n_runs=80,
                      calib_verify_runs=0)
    assert set(audit["modes"]) == set(MODES)
    assert set(audit["gaps"]) == {
        "decentralization", "communication", "cooperation"}
    assert audit["gaps"]["cooperation"]["value"] > 0.0
    assert audit["questions"]["cooperation_value"]["answer"] is True
    assert isinstance(audit["passed"], bool)
    for key in ("bounds_exact", "bounds_token", "stability",
                "token_bits", "ordering_holds", "error_constraints_met",
                "local_only_finding"):
        assert key in audit


def test_stabilized_calibrate_deterministic_and_feasible(scenario):
    from uav_otfs_isac.distributed_audit import stabilized_calibrate
    cal1 = stabilized_calibrate(scenario["links"][0], scan_runs=80,
                                verify_runs=200, seed=100)
    cal2 = stabilized_calibrate(scenario["links"][0], scan_runs=80,
                                verify_runs=200, seed=100)
    assert cal1["a_bound"] == cal2["a_bound"]
    assert cal1["b_bound"] == cal2["b_bound"]
    assert cal1["p_fa"] <= 0.05 + 1e-9
    assert cal1["p_md"] <= 0.05 + 1e-9
    assert cal1["a_bound"] > cal1["b_bound"]
