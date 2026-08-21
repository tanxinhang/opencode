"""Tests for the posterior-state Bellman active detection line.

Covers the exact log-odds update, the alpha-vector vs grid-value mutual
validation, Blackwell dominance (LP), the Bellman policy dominating the
myopic heuristics (tau_pred / Chernoff / one-step lookahead / static) in
expected cost, the sequential information lower bounds, and the
multi-target exact joint Bellman vs the dual-decomposed scheduler.
"""

import numpy as np
import pytest




from uav_otfs_isac.active_detection_bellman import (
    action_kernels,
    bellman_action_policy,
    belief_from_log_odds,
    blackwell_dominates,
    budget_bellman_policy,
    budget_bellman_value,
    calibrate_delay_prices,
    calibrate_sprt_boundaries,
    chernoff_policy,
    decomposed_scheduler,
    delay_action_selector,
    delay_policy,
    delay_value_iteration,
    dpd_policy,
    dual_decomposed_value,
    exact_alpha_vectors,
    grid_bellman_value,
    information_lower_bounds,
    _cost_tokens,
    joint_delay_policy,
    joint_delay_value,
    joint_bellman_policy,
    joint_bellman_value,
    residual_adaptive_policy,
    rollout,
    rollout_budget,
    rollout_delay_multi,
    rollout_mismatch,
    rollout_multi,
    make_deployable_controllers,
    sprt_boundary_policy,
    static_policy,
    tau_pred_policy,
    _evaluate_single,
)


def _kernel(p1, p0, cost=1.0):
    p1 = np.asarray(p1, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    p1 = p1 / p1.sum()
    p0 = p0 / p0.sum()
    llr = np.log(p1 / p0)
    with np.errstate(divide="ignore"):
        i_plus = float(np.sum(p1 * np.log(p1 / np.maximum(p0, 1e-300))))
        i_minus = float(np.sum(p0 * np.log(p0 / np.maximum(p1, 1e-300))))
    return {"p1": p1, "p0": p0, "llr": llr, "i_plus": i_plus,
            "i_minus": i_minus, "cost": cost, "chernoff": 0.5 * i_plus}


def _gauss_kernel(mu0=4.0, var0=4.0, mu1=8.0, var1=12.0, bits=1,
                  flip=0.05, success=0.95, span_std=5.0, cost=None):
    k = action_kernels(mu0, var0, mu1, var1, bits, flip, success, span_std)
    k["cost"] = float(cost if cost is not None else bits)
    return k


def _wald_boundaries(alpha=0.05, beta=0.05):
    return np.log((1.0 - beta) / alpha), np.log(beta / (1.0 - alpha))


def test_llr_update_is_exact():
    # L' = L + log(p1(y)/p0(y)); pi' = pi p1(y) / (pi p1(y) + (1-pi) p0(y))
    k = _kernel([0.9, 0.1], [0.1, 0.9])
    for l0 in (-3.0, -0.5, 0.0, 0.5, 3.0):
        for y in range(2):
            pi = belief_from_log_odds(l0)
            pi1 = pi * k["p1"][y] / (pi * k["p1"][y] + (1.0 - pi) * k["p0"][y])
            l1 = l0 + float(k["llr"][y])
            assert belief_from_log_odds(l1) == pytest.approx(pi1, abs=1e-9)


def test_blackwell_dominance_lp():
    # a perfect bit dominates its BSC-garbling and the erasure-only kernel
    perfect = _kernel([0.9, 0.1], [0.1, 0.9])
    garbled = _kernel([0.8, 0.2], [0.2, 0.8])
    erasure = _kernel([0.45, 0.45, 0.1], [0.45, 0.45, 0.1])
    assert blackwell_dominates(perfect, garbled)
    assert blackwell_dominates(perfect, erasure)
    assert not blackwell_dominates(garbled, perfect)
    # duplicated symbols are Blackwell-equivalent to the coarse bit: each
    # 4-symbol copy maps deterministically onto the 2-symbol alphabet
    dup4 = _kernel([0.45, 0.45, 0.05, 0.05], [0.05, 0.05, 0.45, 0.45])
    assert blackwell_dominates(perfect, dup4)
    assert blackwell_dominates(dup4, perfect)
    # genuinely incomparable pair (verified by LP in both directions)
    a = _kernel([0.9, 0.1], [0.1, 0.9])
    b = _kernel([0.76, 0.04, 0.2], [0.04, 0.76, 0.2])
    assert not blackwell_dominates(a, b)
    assert not blackwell_dominates(b, a)
    # identical kernels dominate each other
    assert blackwell_dominates(perfect, perfect)


def test_blackwell_pruning_matches_data_processing_facts():
    # the LP must certify the refinement/degradation chains that the
    # quantization facts already prove: more bits refines, lower BSC flip
    # refines, higher success refines (extra erasure is a garbling)
    base = (4.0, 4.0, 8.0, 12.0)
    k1 = _gauss_kernel(*base, bits=1, flip=0.05, success=0.95)
    k2 = _gauss_kernel(*base, bits=2, flip=0.05, success=0.95)
    assert blackwell_dominates(k2, k1)
    assert not blackwell_dominates(k1, k2)
    k_flip_lo = _gauss_kernel(*base, bits=1, flip=0.05, success=0.95)
    k_flip_hi = _gauss_kernel(*base, bits=1, flip=0.2, success=0.95)
    assert blackwell_dominates(k_flip_lo, k_flip_hi)
    k_s_hi = _gauss_kernel(*base, bits=1, flip=0.05, success=0.95)
    k_s_lo = _gauss_kernel(*base, bits=1, flip=0.05, success=0.6)
    assert blackwell_dominates(k_s_hi, k_s_lo)


def test_alpha_vectors_match_grid_value():
    # exact alpha-vector recursion vs dense-grid value iteration on a tiny
    # instance (2-atom kernels so the vector count stays small)
    actions = [
        _kernel([0.85, 0.15], [0.15, 0.85], cost=1.0),
        _kernel([0.7, 0.3], [0.3, 0.7], cost=1.5),
    ]
    horizon = 3
    c10 = c01 = 1.0
    exact = exact_alpha_vectors(actions, horizon, c10, c01)
    grid = grid_bellman_value(actions, horizon, c10, c01, grid=501, l_max=6.0)
    l_grid = np.linspace(-6.0, 6.0, 501)
    diffs = []
    for l in l_grid[::25]:
        pi = belief_from_log_odds(float(l))
        v_exact = min(a * pi + b for a, b, _, _ in exact["vectors"])
        v_grid = float(np.interp(l, grid["ls"], grid["v"]))
        diffs.append(abs(v_exact - v_grid))
    assert max(diffs) <= 0.1
    assert float(np.mean(diffs)) <= 0.03


def test_bellman_policy_beats_myopic_heuristics():
    # Bellman minimizes the Bayesian expected cost; the heuristics
    # (tau_pred / Chernoff / one-step lookahead / static) must not beat it
    actions = [
        _gauss_kernel(bits=1, flip=0.05, success=0.9, span_std=4.0),
        _gauss_kernel(bits=1, flip=0.05, success=0.95, span_std=6.0),
        _gauss_kernel(bits=2, flip=0.1, success=0.9, span_std=5.0),
    ]
    c10 = c01 = 1.0
    horizon = 5
    grid = grid_bellman_value(actions, horizon, c10, c01, grid=301, l_max=8.0)
    a, b = _wald_boundaries()
    policies = {
        "bellman": bellman_action_policy(grid, actions, c10, c01),
        "tau": tau_pred_policy(actions, a, b),
        "chernoff": chernoff_policy(actions, a, b),
        "dpd": dpd_policy(actions, c10, c01),
        "static": static_policy(actions, a, b),
    }
    costs = {}
    for name, pol in policies.items():
        r0 = rollout(pol, actions, horizon, 0, n_runs=2000, seed=1)
        r1 = rollout(pol, actions, horizon, 1, n_runs=2000, seed=2)
        costs[name] = 0.5 * (r0["mean_cost"] + r1["mean_cost"]) \
            + 0.5 * (c01 * r0["p_fa"] + c10 * r1["p_md"])
    for name, cost in costs.items():
        if name == "bellman":
            continue
        assert costs["bellman"] <= cost * 1.05


def test_information_lower_bounds_hold_on_rollout():
    # E_1[sum I+] >= d(1-p_md || p_fa) and E_1[T] >= d / I_max+ for the
    # realized error probabilities of a Wald-boundary detector
    actions = [
        _gauss_kernel(bits=1, flip=0.05, success=0.95, span_std=5.0),
        _gauss_kernel(bits=2, flip=0.05, success=0.95, span_std=5.0),
    ]
    a, b = _wald_boundaries()
    pol = chernoff_policy(actions, a, b)
    r0 = rollout(pol, actions, 5, 0, n_runs=4000, seed=3)
    r1 = rollout(pol, actions, 5, 1, n_runs=4000, seed=4)
    bounds = information_lower_bounds(r0["p_fa"], r1["p_md"], actions)
    assert r1["mean_info"] >= bounds["d_1"] - 0.05
    assert r1["mean_delay"] >= bounds["t1_lower"] - 0.5
    assert r1["mean_cost"] >= bounds["c1_lower"] - 0.5


def test_rollout_reports_errors_and_delay():
    actions = [_gauss_kernel(bits=1, flip=0.05, success=0.95)]
    a, b = _wald_boundaries()
    pol = static_policy(actions, a, b)
    r0 = rollout(pol, actions, 5, 0, n_runs=3000, seed=5)
    r1 = rollout(pol, actions, 5, 1, n_runs=3000, seed=6)
    assert 0.005 <= r0["p_fa"] <= 0.2
    assert 0.005 <= r1["p_md"] <= 0.2
    assert np.isnan(r0["p_md"]) and np.isnan(r1["p_fa"])
    assert 0.0 < r0["mean_delay"] < 64.0
    assert 0.0 < r1["mean_delay"] < 64.0
    assert r1["mean_cost"] > 0.0 and r0["mean_cost"] > 0.0
    assert r1["mean_info"] > 0.0


def test_dual_scheduler_stops_all_targets():
    actions_q0 = [_gauss_kernel(bits=1, flip=0.05, success=0.95)]
    actions_q1 = [_gauss_kernel(mu1=6.0, var1=10.0, bits=1, flip=0.05,
                                success=0.9)]
    c10 = c01 = 1.0
    v_q = [
        dual_decomposed_value(actions_q0, 5, c10, c01, nu=1.0, lam=0.5,
                              grid=101),
        dual_decomposed_value(actions_q1, 5, c10, c01, nu=1.0, lam=0.5,
                              grid=101),
    ]
    pol = decomposed_scheduler(v_q, [actions_q0, actions_q1], 1,
                               c10, c01)
    out = rollout_multi(pol, [actions_q0, actions_q1], [0, 1], n_runs=500,
                        seed=7, max_steps=30)
    assert out["mean_worst_delay"] < 30
    assert all(d < 30 for d in out["mean_delays"])
    assert 0.0 <= out["p_fa"][0] <= 1.0
    assert 0.0 <= out["p_md"][1] <= 1.0


def test_joint_bellman_beats_dual_scheduler_in_worst_delay():
    # the exact joint policy is optimal for the move set; the dual
    # decomposition is an implementable policy, so its worst-target mean
    # delay must not be better (MC tolerance)
    actions_q0 = [_gauss_kernel(bits=1, flip=0.05, success=0.95),
                  _gauss_kernel(bits=1, flip=0.15, success=0.95)]
    actions_q1 = [_gauss_kernel(mu1=6.0, var1=10.0, bits=1, flip=0.05,
                                success=0.9),
                  _gauss_kernel(mu1=6.0, var1=10.0, bits=1, flip=0.15,
                                success=0.9)]
    c10 = c01 = 1.0
    horizon = 3
    joint = joint_bellman_value([actions_q0, actions_q1], horizon, c10, c01,
                                grid=41, l_max=8.0)
    jpol = joint_bellman_policy(joint, [actions_q0, actions_q1], c10, c01)
    jout = rollout_multi(jpol, [actions_q0, actions_q1], [0, 1], n_runs=400,
                         seed=8, max_steps=20)
    v_q = [
        dual_decomposed_value(actions_q0, horizon, c10, c01, nu=1.0, lam=0.5,
                              grid=101),
        dual_decomposed_value(actions_q1, horizon, c10, c01, nu=1.0, lam=0.5,
                              grid=101),
    ]
    dpol = decomposed_scheduler(v_q, [actions_q0, actions_q1], 1,
                                c10, c01)
    dout = rollout_multi(dpol, [actions_q0, actions_q1], [0, 1], n_runs=400,
                         seed=9, max_steps=20)
    assert jout["mean_worst_delay"] <= dout["mean_worst_delay"] * 1.1 + 0.5


def test_joint_value_weak_duality_holds():
    # the dual bound max_{nu,lam} [sum_q value_q - lam * B_total] <= V*
    # with B_total = horizon (per-cycle budget 1)
    actions_q0 = [_gauss_kernel(bits=1, flip=0.05, success=0.95)]
    actions_q1 = [_gauss_kernel(mu1=6.0, var1=10.0, bits=1, flip=0.05,
                                success=0.9)]
    c10 = c01 = 1.0
    horizon = 3
    joint = joint_bellman_value([actions_q0, actions_q1], horizon, c10, c01,
                                grid=41, l_max=8.0)
    mid = int((len(joint["ls"]) - 1) / 2)
    v_star = float(joint["v"][mid, mid])
    best_bound = -np.inf
    for nu1 in (0.3, 0.5, 0.7):
        for lam in (0.2, 0.5, 1.0):
            bound = 0.0
            for nu_q, acts in ((nu1, actions_q0), (1.0 - nu1, actions_q1)):
                vq = dual_decomposed_value(acts, horizon, c10, c01, nu=nu_q,
                                           lam=lam, grid=101)
                bound += float(vq["v"][mid])
            bound -= lam * horizon
            best_bound = max(best_bound, bound)
    assert best_bound <= v_star + 0.05


def test_budget_bellman_matches_unconstrained_grid_when_budget_is_loose():
    # with a budget that pays every action, V_h(l, B) must coincide with the
    # unconstrained grid value (the budget constraint is inactive)
    actions = [
        _gauss_kernel(bits=1, flip=0.05, success=0.95, span_std=4.0),
        _gauss_kernel(bits=2, flip=0.05, success=0.95, span_std=6.0),
    ]
    c10 = c01 = 3.0
    horizon = 3
    grid_val = grid_bellman_value(actions, horizon, c10, c01,
                                  grid=101, l_max=6.0)
    budget_val = budget_bellman_value(actions, horizon, 8, c10, c01,
                                      grid=101, l_max=6.0)
    ls = grid_val["ls"]
    for l in ls[::20]:
        i = int(np.clip(int(np.searchsorted(ls, l)), 0, len(ls) - 1))
        v_free = float(np.interp(l, ls, grid_val["v"]))
        v_budget = float(budget_val["values"][horizon, 8][i])
        assert abs(v_free - v_budget) <= 1e-9


def test_budget_bellman_value_is_monotone_and_bounded_in_budget():
    # more budget never hurts: V_h(l, b + 1) <= V_h(l, b); with zero budget
    # the value is exactly the terminal stop cost
    actions = [
        _gauss_kernel(bits=1, flip=0.05, success=0.95),
        _gauss_kernel(bits=2, flip=0.05, success=0.95),
    ]
    c10 = c01 = 2.0
    horizon = 3
    budget_val = budget_bellman_value(actions, horizon, 5, c10, c01,
                                      grid=101, l_max=6.0)
    ls = budget_val["ls"]
    for b in range(5):
        v_b = budget_val["values"][horizon, b]
        v_b1 = budget_val["values"][horizon, b + 1]
        assert float(np.max(v_b1 - v_b)) <= 1e-9
    v0 = budget_val["values"][horizon, 0]
    terminal = np.minimum(
        c01 * (1.0 / (1.0 + np.exp(-ls))),
        c10 * (1.0 / (1.0 + np.exp(ls))),
    )
    np.testing.assert_allclose(v0, terminal, atol=1e-12)


def test_budget_bellman_policy_beats_myopic_under_tight_budget():
    # with a budget too small for the Wald myopic schedule, the budget-aware
    # Bellman policy must not be beaten in Bayesian expected cost
    actions = [
        _gauss_kernel(bits=1, flip=0.05, success=0.95),
        _gauss_kernel(bits=1, flip=0.05, success=0.9, span_std=4.0),
        _gauss_kernel(bits=2, flip=0.1, success=0.9, span_std=5.0),
    ]
    c10 = c01 = 3.0
    horizon = 4
    budget = 5
    v = budget_bellman_value(actions, horizon, budget, c10, c01,
                             grid=201, l_max=8.0)
    a, b = _wald_boundaries()
    policies = {
        "bellman": budget_bellman_policy(v, actions, c10, c01),
        "tau": tau_pred_policy(actions, a, b),
        "chernoff": chernoff_policy(actions, a, b),
        "dpd": dpd_policy(actions, c10, c01),
        "static": static_policy(actions, a, b),
    }
    # the myopic policies are budget-blind; rollout_budget calls them with
    # the remaining budget, which they ignore and may overspend (the rollout
    # then forces a terminal decision when the budget runs out)
    wrapped = {
        name: (pol if name == "bellman"
               else (lambda l, step, b, _pol=pol: _pol(l, step)))
        for name, pol in policies.items()
    }
    costs = {}
    for name, pol in wrapped.items():
        r0 = rollout_budget(pol, actions, budget, 0, n_runs=2000, seed=11)
        r1 = rollout_budget(pol, actions, budget, 1, n_runs=2000, seed=12)
        costs[name] = 0.5 * (r0["mean_cost"] + r1["mean_cost"]) \
            + 0.5 * (c01 * r0["p_fa"] + c10 * r1["p_md"])
    for name, cost in costs.items():
        if name == "bellman":
            continue
        assert costs["bellman"] <= cost * 1.1


def _mismatch_kernel(base, flip_new=None, success_new=None):
    """Copy of a Gaussian kernel with a different BSC flip / success."""
    from uav_otfs_isac.detection_quantization import quantizer_edges
    from uav_otfs_isac.detection_information import (
        post_communication_likelihoods,
    )
    mu0, var0 = 4.0, 4.0
    mu1, var1 = base["_geom"]
    bits = base["_bits"]
    edges, values = quantizer_edges(mu0, var0, mu1, var1, bits, 4.0)
    info = post_communication_likelihoods(
        mu0, var0, mu1, var1, edges, values, bits,
        flip_new if flip_new is not None else 0.05,
        success_new if success_new is not None else 0.95,
    )
    return {
        "p0": np.asarray(info["p0_y"], dtype=float),
        "p1": np.asarray(info["p1_y"], dtype=float),
        "llr": np.log(np.asarray(info["p1_y"], dtype=float)
                      / np.asarray(info["p0_y"], dtype=float)),
        "i_plus": float(info["kl_plus"]),
        "i_minus": float(info["kl_minus"]),
        "cost": base["cost"],
        "chernoff": float(info["chernoff"]),
    }


def _geom_kernel(mu1=8.0, var1=12.0, bits=1, flip=0.05, success=0.95,
                 cost=None):
    k = _gauss_kernel(mu1=mu1, var1=var1, bits=bits, flip=flip,
                      success=success, span_std=4.0)
    k["cost"] = float(cost if cost is not None else bits)
    k["_geom"] = (mu1, var1)
    k["_bits"] = bits
    return k


def test_residual_tau_stays_small_under_correct_model():
    # under the correct model the standardized residual of the active
    # hypothesis is zero-mean, so the running-mean trigger statistic
    # tau = min(|mean_0|, |mean_1|) converges to zero and never fires
    actions = [_geom_kernel(bits=1, flip=0.05, cost=1.0),
               _geom_kernel(mu1=10.0, var1=16.0, bits=2, flip=0.1, cost=2.0)]
    c10 = c01 = 20.0
    horizon = 4
    v = grid_bellman_value(actions, horizon, c10, c01, grid=301, l_max=8.0)
    pol, monitor = residual_adaptive_policy(v, actions, c10, c01, horizon,
                                            residual_margin=0.25,
                                            explore_rounds=2, warmup=20)
    rollout_budget(pol, actions, 8, 1, n_runs=3000, seed=21)
    assert not monitor["triggered"]
    assert monitor["tau"] < 0.1
    assert set(monitor["modes"]) <= {"bellman"}


def test_residual_trigger_fires_under_kernel_mismatch():
    # modeled success 0.95 but the environment succeeds with 0.5 (erasure
    # halves the evidence): the realized residual leaves both model
    # conditionals, tau grows past the margin and the controller switches
    # to robust/explore mode
    model = [_geom_kernel(bits=1, flip=0.05, cost=1.0)]
    truth = [_mismatch_kernel(model[0], success_new=0.5)]
    c10 = c01 = 20.0
    horizon = 4
    v = grid_bellman_value(model, horizon, c10, c01, grid=301, l_max=8.0)
    pol, monitor = residual_adaptive_policy(v, model, c10, c01, horizon,
                                            residual_margin=0.25,
                                            explore_rounds=2, warmup=20)
    rollout_mismatch(pol, model, truth, 1, n_runs=3000, seed=22, budget=8)
    assert monitor["triggered"]
    assert monitor["tau"] > 0.3
    assert monitor["mode"] in ("robust", "explore")
    assert ("robust" in monitor["modes"]) or ("explore" in monitor["modes"])


def test_residual_adaptive_cost_never_worse_than_naive_under_mismatch():
    # realized (true-kernel) Bayesian cost under mismatch: the adaptive
    # controller stops trusting the contaminated multi-step value, so its
    # realized cost must not exceed the naive Bellman's (MC tolerance)
    model = [_geom_kernel(bits=1, flip=0.05, cost=1.0),
             _geom_kernel(mu1=10.0, var1=16.0, bits=2, flip=0.05, cost=2.0)]
    truth = [_mismatch_kernel(model[0], success_new=0.5),
             _mismatch_kernel(model[1], success_new=0.5)]
    c10 = c01 = 20.0
    horizon = 4
    v = grid_bellman_value(model, horizon, c10, c01, grid=301, l_max=8.0)
    nominal = bellman_action_policy(v, model, c10, c01)
    adaptive, monitor = residual_adaptive_policy(
        v, model, c10, c01, horizon,
        residual_margin=0.25, explore_rounds=2, warmup=20)

    def realized(pol):
        r0 = rollout_mismatch(pol, model, truth, 0, n_runs=2500, seed=31, budget=8)
        r1 = rollout_mismatch(pol, model, truth, 1, n_runs=2500, seed=32, budget=8)
        return 0.5 * (r0["mean_cost"] + r1["mean_cost"]) \
            + 0.5 * (c01 * r0["p_fa"] + c10 * r1["p_md"])

    cost_naive = realized(lambda l, step, b: nominal(l, step))
    cost_adaptive = realized(adaptive)
    assert monitor["triggered"]
    assert cost_adaptive <= cost_naive * 1.02 + 0.15


def test_value_bound_prune_is_exact_and_active():
    # level-2 pruning must not change the value function (bit-identical)
    # and must eliminate actions that are never useful against the current
    # value (a zero-information, expensive kernel dominated by stopping)
    noise = _kernel([0.5, 0.5], [0.5, 0.5], cost=5.0)
    strong = _geom_kernel(mu1=10.0, var1=16.0, bits=2, flip=0.05, cost=2.0)
    actions = [strong, noise]
    c10 = c01 = 10.0
    horizon = 3
    v_pruned = grid_bellman_value(actions, horizon, c10, c01,
                                  grid=401, l_max=8.0)
    v_plain = grid_bellman_value([strong, noise], horizon, c10, c01,
                                 grid=401, l_max=8.0)
    for h in range(horizon + 1):
        np.testing.assert_allclose(v_pruned["values"][h],
                                   v_plain["values"][h], atol=1e-12)
    assert v_pruned["prune_stats"] == v_plain["prune_stats"]
    # the noise kernel is eliminated at every step (its continuation costs
    # 5 + V(l) which never beats the terminal stop cost <= 5)
    assert all(count >= 1 for count in v_pruned["prune_stats"])
    # the pruned action is never chosen by the resulting policy
    pol = bellman_action_policy(v_pruned, actions, c10, c01)
    chosen = set()
    rng = np.random.default_rng(41)
    for _ in range(200):
        l = 0.0
        t = 0
        while t < 8:
            dec = pol(l, t)
            if dec < 0:
                break
            chosen.add(dec)
            act = actions[dec]
            p = act["p1"]
            y = int(rng.choice(len(p), p=p))
            l += float(act["llr"][y])
            t += 1
    assert 1 not in chosen


def _d2_kernels():
    """Three-action Gaussian library with integer costs (D2 tests)."""
    return [
        _geom_kernel(mu1=8.0, var1=12.0, bits=1, flip=0.05, cost=1.0),
        _geom_kernel(mu1=10.0, var1=16.0, bits=2, flip=0.1, cost=2.0),
        _geom_kernel(mu1=6.0, var1=10.0, bits=3, flip=0.05, cost=3.0),
    ]


def test_delay_value_terminal_and_monotone():
    # objective-aligned delay value (advice/004 eq. 1): the continuation
    # branch costs exactly one cycle; the terminal is the priced
    # declaration, and more horizon/budget never hurts
    actions = _d2_kernels()
    xi = zeta = 32.0
    v = delay_value_iteration(actions, 5, 6, xi, zeta,
                              grid=301, l_max=8.0)
    ls = v["ls"]
    pi = 1.0 / (1.0 + np.exp(-ls))
    np.testing.assert_allclose(
        v["values"][0, 3], np.minimum(zeta * pi, xi * (1.0 - pi)),
        atol=1e-12)
    mid = len(ls) // 2
    for b in range(6):
        for h in range(5):
            assert v["values"][h + 1, b][mid] <= v["values"][h, b][mid] + 1e-12
    for h in range(6):
        assert v["values"][h, 5][mid] <= v["values"][h, 2][mid] + 1e-12


def test_sprt_calibration_meets_error_constraints():
    # numerically calibrated two thresholds must meet P_FA <= alpha and
    # P_MD <= beta with a finite H1 detection delay
    actions = _d2_kernels()
    cal = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                    n_runs=800, seed=10, margin=0.6)
    assert cal["p_fa"] <= 0.05 + 1e-9
    assert cal["p_md"] <= 0.05 + 1e-9
    assert 1.0 < cal["e1_delay"] < 12.0
    assert cal["a_bound"] > 0.0 > cal["b_bound"]


def test_calibrated_sprt_beats_wald_approximation():
    # Wald's continuous approximation is not exact for the quantized+BSC+
    # erasure kernels: the calibrated boundaries are at least as fast under
    # the same error constraints
    actions = _d2_kernels()
    cal = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                    n_runs=800, seed=10, margin=0.6)
    wald = sprt_boundary_policy(actions, float(np.log(19.0)),
                                float(-np.log(19.0)))
    row = _evaluate_single(wald, actions, 12, 0.05, 0.05, 2000, 30)
    assert cal["e1_delay"] <= row["e1_delay"] + 0.05


def test_delay_selector_sprt_beats_old_cost_bellman():
    # under the objective-aligned metric (E_1[T] at fixed errors) the
    # delay-value action selector must not be slower than the information
    # myopic selector (single target: both are near-optimal)
    actions = _d2_kernels()
    v = delay_value_iteration(actions, 8, 12, 32.0, 32.0,
                              grid=201, l_max=8.0)
    sel = delay_action_selector(v, actions, 32.0, 32.0)
    cal_delay = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                          n_runs=600, seed=10, margin=0.6,
                                          selector=sel)
    cal_ic = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                       n_runs=600, seed=10, margin=0.6)
    assert cal_delay["p_fa"] <= 0.05 + 1e-9
    assert cal_delay["p_md"] <= 0.05 + 1e-9
    assert cal_delay["e1_delay"] <= cal_ic["e1_delay"] + 0.05


def test_joint_delay_policy_respects_budget():
    actions0 = [_geom_kernel(mu1=9.0, var1=14.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=11.0, var1=18.0, bits=2, flip=0.1, cost=2.0)]
    actions1 = [_geom_kernel(mu1=5.5, var1=9.5, bits=1, flip=0.15, cost=1.0),
                _geom_kernel(mu1=7.0, var1=12.0, bits=2, flip=0.1, cost=2.0)]
    xi = zeta = 64.0
    vj = joint_delay_value([actions0, actions1], 6, 8, xi, zeta,
                           grid=41, l_max=8.0)
    jpol = joint_delay_policy(vj, [actions0, actions1], xi, zeta)
    out = rollout_delay_multi(jpol, [actions0, actions1], [1, 1], 8,
                              n_runs=300, seed=5, max_steps=20)
    total_cost = sum(out["mean_costs"])
    assert total_cost <= 8.0 + 1e-9
    assert out["mean_worst_delay"] <= 20.0
    assert 0.0 <= out["p_md"][0] <= 1.0


def test_joint_delay_policy_respects_budget():
    actions0 = [_geom_kernel(mu1=9.0, var1=14.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=11.0, var1=18.0, bits=2, flip=0.1, cost=2.0)]
    actions1 = [_geom_kernel(mu1=5.5, var1=9.5, bits=1, flip=0.15, cost=1.0),
                _geom_kernel(mu1=7.0, var1=12.0, bits=2, flip=0.1, cost=2.0)]
    xi = zeta = 64.0
    vj = joint_delay_value([actions0, actions1], 6, 8, xi, zeta,
                           grid=41, l_max=8.0)
    jpol = joint_delay_policy(vj, [actions0, actions1], xi, zeta)
    out = rollout_delay_multi(jpol, [actions0, actions1], [1, 1], 8,
                              n_runs=300, seed=5, max_steps=20)
    total_cost = sum(out["mean_costs"])
    assert total_cost <= 8.0 + 1e-9
    assert out["mean_worst_delay"] <= 20.0
    assert 0.0 <= out["p_md"][0] <= 1.0


def test_joint_delay_policy_respects_budget():
    actions0 = [_geom_kernel(mu1=9.0, var1=14.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=11.0, var1=18.0, bits=2, flip=0.1, cost=2.0)]
    actions1 = [_geom_kernel(mu1=5.5, var1=9.5, bits=1, flip=0.15, cost=1.0),
                _geom_kernel(mu1=7.0, var1=12.0, bits=2, flip=0.1, cost=2.0)]
    xi = zeta = 64.0
    vj = joint_delay_value([actions0, actions1], 6, 8, xi, zeta,
                           grid=41, l_max=8.0)
    jpol = joint_delay_policy(vj, [actions0, actions1], xi, zeta)
    out = rollout_delay_multi(jpol, [actions0, actions1], [1, 1], 8,
                              n_runs=300, seed=5, max_steps=20)
    total_cost = sum(out["mean_costs"])
    assert total_cost <= 8.0 + 1e-9
    assert out["mean_worst_delay"] <= 20.0
    assert 0.0 <= out["p_md"][0] <= 1.0


def _d2_kernels():
    """Three-action Gaussian library with integer costs (D2 tests)."""
    return [
        _geom_kernel(mu1=8.0, var1=12.0, bits=1, flip=0.05, cost=1.0),
        _geom_kernel(mu1=10.0, var1=16.0, bits=2, flip=0.1, cost=2.0),
        _geom_kernel(mu1=6.0, var1=10.0, bits=3, flip=0.05, cost=3.0),
    ]


def test_delay_value_terminal_and_monotone():
    # objective-aligned delay value (advice/004 eq. 1): the continuation
    # branch costs exactly one cycle; the terminal is the priced
    # declaration, and more horizon/budget never hurts
    actions = _d2_kernels()
    xi = zeta = 32.0
    v = delay_value_iteration(actions, 5, 6, xi, zeta,
                              grid=301, l_max=8.0)
    ls = v["ls"]
    pi = 1.0 / (1.0 + np.exp(-ls))
    np.testing.assert_allclose(
        v["values"][0, 3], np.minimum(zeta * pi, xi * (1.0 - pi)),
        atol=1e-12)
    mid = len(ls) // 2
    for b in range(6):
        for h in range(5):
            assert v["values"][h + 1, b][mid] <= v["values"][h, b][mid] + 1e-12
    for h in range(6):
        assert v["values"][h, 5][mid] <= v["values"][h, 2][mid] + 1e-12


def test_sprt_calibration_meets_error_constraints():
    # numerically calibrated two thresholds must meet P_FA <= alpha and
    # P_MD <= beta with a finite H1 detection delay
    actions = _d2_kernels()
    cal = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                    n_runs=800, seed=10, margin=0.6)
    assert cal["p_fa"] <= 0.05 + 1e-9
    assert cal["p_md"] <= 0.05 + 1e-9
    assert 1.0 < cal["e1_delay"] < 12.0
    assert cal["a_bound"] > 0.0 > cal["b_bound"]


def test_calibrated_sprt_beats_wald_approximation():
    # Wald's continuous approximation is not exact for the quantized+BSC+
    # erasure kernels: the calibrated boundaries are at least as fast under
    # the same error constraints
    actions = _d2_kernels()
    cal = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                    n_runs=800, seed=10, margin=0.6)
    wald = sprt_boundary_policy(actions, float(np.log(19.0)),
                                float(-np.log(19.0)))
    row = _evaluate_single(wald, actions, 12, 0.05, 0.05, 2000, 30)
    assert cal["e1_delay"] <= row["e1_delay"] + 0.05


def test_delay_selector_sprt_beats_old_cost_bellman():
    # under the objective-aligned metric (E_1[T] at fixed errors) the
    # delay-value action selector must not be slower than the information
    # myopic selector (single target: both are near-optimal)
    actions = _d2_kernels()
    v = delay_value_iteration(actions, 8, 12, 32.0, 32.0,
                              grid=201, l_max=8.0)
    sel = delay_action_selector(v, actions, 32.0, 32.0)
    cal_delay = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                          n_runs=600, seed=10, margin=0.6,
                                          selector=sel)
    cal_ic = calibrate_sprt_boundaries(actions, 0.05, 0.05, 12,
                                       n_runs=600, seed=10, margin=0.6)
    assert cal_delay["p_fa"] <= 0.05 + 1e-9
    assert cal_delay["p_md"] <= 0.05 + 1e-9
    assert cal_delay["e1_delay"] <= cal_ic["e1_delay"] + 0.05


def test_joint_delay_policy_respects_budget():
    actions0 = [_geom_kernel(mu1=9.0, var1=14.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=11.0, var1=18.0, bits=2, flip=0.1, cost=2.0)]
    actions1 = [_geom_kernel(mu1=5.5, var1=9.5, bits=1, flip=0.15, cost=1.0),
                _geom_kernel(mu1=7.0, var1=12.0, bits=2, flip=0.1, cost=2.0)]
    xi = zeta = 64.0
    vj = joint_delay_value([actions0, actions1], 6, 8, xi, zeta,
                           grid=41, l_max=8.0)
    jpol = joint_delay_policy(vj, [actions0, actions1], xi, zeta)
    out = rollout_delay_multi(jpol, [actions0, actions1], [1, 1], 8,
                              n_runs=300, seed=5, max_steps=20)
    total_cost = sum(out["mean_costs"])
    assert total_cost <= 8.0 + 1e-9
    assert out["mean_worst_delay"] <= 20.0
    assert 0.0 <= out["p_md"][0] <= 1.0


def test_joint_delay_not_worse_than_independent():
    # on the min-max metric (the system objective max_q E[T_q]) the joint
    # planner balances both targets and never falls behind a sequential
    # independent schedule that shares the same budget (MC tolerance):
    # resource competition is the source of planning value
    actions0 = [_geom_kernel(mu1=9.0, var1=14.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=11.0, var1=18.0, bits=2, flip=0.1, cost=2.0)]
    actions1 = [_geom_kernel(mu1=7.0, var1=11.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=8.5, var1=14.0, bits=2, flip=0.1, cost=2.0)]
    xi = zeta = 64.0
    budget = 8
    vj = joint_delay_value([actions0, actions1], 6, budget, xi, zeta,
                           grid=33, l_max=8.0)
    jpol = joint_delay_policy(vj, [actions0, actions1], xi, zeta)
    joint = rollout_delay_multi(jpol, [actions0, actions1], [1, 1], budget,
                                n_runs=300, seed=6, max_steps=20)
    v0 = delay_value_iteration(actions0, 6, budget, xi, zeta,
                               grid=201, l_max=8.0)
    v1 = delay_value_iteration(actions1, 6, budget, xi, zeta,
                               grid=201, l_max=8.0)
    pol0 = delay_policy(v0, actions0, xi, zeta)
    pol1 = delay_policy(v1, actions1, xi, zeta)

    def sequential(l_vec, step, b):
        # serve target 0 with its independent policy until it stops, then
        # target 1; both draw on the same budget
        dec0 = pol0(l_vec[0], step, b)
        if dec0 in (-1, -2):
            return [dec0, pol1(l_vec[1], step, b)]
        return [dec0, -3]

    seq = rollout_delay_multi(sequential, [actions0, actions1], [1, 1],
                              budget, n_runs=300, seed=9, max_steps=20)
    assert joint["mean_worst_delay"] <= seq["mean_worst_delay"] * 1.1 + 0.5




def _myopic_multi_test(actions_per_target, bounds):
    """Local one-step crossing-probability myopic policy (test-only)."""

    def policy(l_vec, step, b_remaining):
        decisions = [-3] * len(l_vec)
        for qq in range(len(l_vec)):
            if l_vec[qq] >= bounds[qq][0]:
                decisions[qq] = -2
            elif l_vec[qq] <= bounds[qq][1]:
                decisions[qq] = -1
        active = [qq for qq in range(len(l_vec)) if decisions[qq] == -3]
        best_q, best_a, best_score = None, None, -np.inf
        for qq in active:
            for ai, act in enumerate(actions_per_target[qq]):
                c = _cost_tokens(act)
                if c > b_remaining + 1e-12:
                    continue
                p1 = np.asarray(act["p1"], dtype=float)
                llr = np.asarray(act["llr"], dtype=float)
                p_cross = float(np.sum(
                    p1 * (l_vec[qq] + llr >= bounds[qq][0])))
                score = p_cross / max(c, 1e-12)
                if score > best_score:
                    best_score, best_q, best_a = score, qq, ai
        if best_q is not None:
            decisions[best_q] = best_a
        return decisions

    return policy


def test_deployable_controllers_respect_budget_and_bounds():
    # the deployable controllers share the singles + calibrated bounds and
    # must respect the budget and stop exactly at the thresholds
    actions0 = [_geom_kernel(mu1=9.0, var1=14.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=11.0, var1=18.0, bits=2, flip=0.1, cost=2.0)]
    actions1 = [_geom_kernel(mu1=7.0, var1=11.0, bits=1, flip=0.05, cost=1.0),
                _geom_kernel(mu1=8.5, var1=14.0, bits=2, flip=0.1, cost=2.0)]
    A = [actions0, actions1]
    budget = 20
    cal0 = calibrate_sprt_boundaries(actions0, 0.05, 0.05, budget,
                                     n_runs=200, seed=3, margin=0.8,
                                     points=5)
    cal1 = calibrate_sprt_boundaries(actions1, 0.05, 0.05, budget,
                                     n_runs=200, seed=4, margin=0.8,
                                     points=5)
    bounds = [(float(cal0["a_bound"]), float(cal0["b_bound"])),
              (float(cal1["a_bound"]), float(cal1["b_bound"]))]
    nu = (0.5, 0.5)
    singles = [delay_value_iteration(acts, 6, budget, 64.0, 64.0,
                                     grid=101, l_max=8.0,
                                     cycle_cost=float(nu[q]),
                                     bounds=bounds[q])
               for q, acts in enumerate(A)]
    ctrls = make_deployable_controllers(A, singles, bounds, nu=nu)
    for name, pol in ctrls.items():
        out = rollout_delay_multi(pol, A, [1, 1], budget, n_runs=200,
                                  seed=7, max_steps=20)
        assert sum(out["mean_costs"]) <= budget + 1e-9, name
        assert out["mean_worst_delay"] <= 20.0, name
        for q in range(2):
            assert 0.0 <= out["p_md"][q] <= 1.0, name


def test_deployable_rollout_not_worse_than_myopic():
    # the deployable rollout controller is the reference deployment; on the
    # strong + weak scenario it must not fall behind the myopic Delta-P_D
    # heuristic in the worst-target H1 delay (MC tolerance)
    strong = [
        _geom_kernel(mu1=10.0, var1=16.0, bits=1, flip=0.02, cost=1.0),
        _geom_kernel(mu1=10.0, var1=16.0, bits=2, flip=0.02, cost=2.0),
        _geom_kernel(mu1=10.0, var1=16.0, bits=3, flip=0.02, cost=3.0),
        _geom_kernel(mu1=10.0, var1=16.0, bits=1, flip=0.02, cost=2.0),
    ]
    weak = [
        _geom_kernel(mu1=7.0, var1=11.0, bits=1, flip=0.08, cost=1.0),
        _geom_kernel(mu1=7.0, var1=11.0, bits=2, flip=0.08, cost=2.0),
        _geom_kernel(mu1=6.0, var1=10.0, bits=2, flip=0.08, cost=2.0),
        _geom_kernel(mu1=7.0, var1=11.0, bits=2, flip=0.08, cost=3.0),
    ]
    A = [strong, weak]
    budget = 20
    bounds = []
    for acts in A:
        cal = calibrate_sprt_boundaries(acts, 0.05, 0.05, budget,
                                        n_runs=200, seed=5, margin=0.8,
                                        points=5)
        bounds.append((float(cal["a_bound"]), float(cal["b_bound"])))
    nu = (0.5, 0.5)
    singles = [delay_value_iteration(acts, 6, budget, 64.0, 64.0,
                                     grid=101, l_max=8.0,
                                     cycle_cost=float(nu[q]),
                                     bounds=bounds[q])
               for q, acts in enumerate(A)]
    ctrls = make_deployable_controllers(A, singles, bounds, nu=nu)
    rollout = rollout_delay_multi(ctrls["rollout_1step"], A, [1, 1],
                                  budget, n_runs=400, seed=9, max_steps=20)
    myopic = rollout_delay_multi(_myopic_multi_test(A, bounds), A, [1, 1],
                                 budget, n_runs=400, seed=9, max_steps=20)
    assert rollout["mean_worst_delay"] <= myopic["mean_worst_delay"] * 1.05 \
        + 0.5
