"""Gate: posterior-state Bellman active detection (advice/002.md).

The three-layer static pipeline (tau_pred / floor-cover / span design) is
restructured into a single optimal-stopping + experiment-design problem on
the posterior log-odds ``L_{t+1} = L_t + log(p1^a(Y)/p0^a(Y))``.

Part D1 (exact oracle, Q = 1): grid Bellman value iteration over a small
report/quantizer action library, evaluated by Monte Carlo against the
``tau_pred``, Chernoff, one-step-lookahead and static Wald policies in the
Bayesian expected total cost (observation cost + error costs); the
alpha-vector recursion is cross-checked against the grid on a tiny
instance.

Part D1b (advice/003 Gate D1, Q = 1, R = 3, bits 1..4, P = 2 power levels,
H = B = 6): the exact finite-horizon Bellman with an explicit budget state
``V_t(pi, B)`` is compared against the myopic family (tau_pred / Chernoff /
one-step lookahead / static floor-cover) in detection cycles and resource
efficiency.  The gate reports the honest verdict: whether the Bellman
oracle materially improves over the best myopic strategy (>= 5% cost or
cycles); if not, the belief-state value-approximation step of the agent
roadmap is not worth it at this scale.

Part D2 (dynamic quantizer): the Bellman policy's action choice across the
belief axis is compared with the fixed Chernoff-optimal quantizer; the
gate reports the adaptive gain (the belief-dependent quantizer exists iff
the Bellman policy leaves the Chernoff-best action somewhere).

Part D3 (multi-target dual decomposition, Q = 2): exact joint Bellman on
the product grid vs the Lagrangian decomposition (nu, lam) with the
per-cycle knapsack scheduler; the gate reports the worst-target mean delay
gap and verifies weak duality of the dual bound.

Part D4 (fundamental limits): the sequential-testing information bounds
``E_1[sum I+] >= d(1-beta || alpha)`` and the derived ``E_1[T] >= d/I_max+``
are checked against the rolled-out Bellman policy, and the delay ratios
``T_proposed / T_lower`` are reported for Bellman and ``tau_pred``.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.active_detection_bellman import (
    action_kernels,
    belief_from_log_odds,
    bellman_action_policy,
    blackwell_dominates,
    budget_bellman_policy,
    budget_bellman_value,
    chernoff_policy,
    decomposed_scheduler,
    dpd_policy,
    dual_decomposed_value,
    exact_alpha_vectors,
    grid_bellman_value,
    information_lower_bounds,
    joint_bellman_policy,
    joint_bellman_value,
    residual_adaptive_policy,
    rollout,
    rollout_budget,
    rollout_mismatch,
    rollout_multi,
    static_policy,
    tau_pred_policy,
)


def make_library(mu0, var0, links, bits_list, spans, flips, successes,
                 cost_of):
    """Report/quantizer action library (post-communication kernels) over
    one or more link geometries ``(mu1, var1)``."""
    actions = []
    for (mu1, var1), bits, span, flip, success in product(
            links, bits_list, spans, flips, successes):
        kernel = action_kernels(mu0, var0, mu1, var1, bits, flip, success,
                                span)
        kernel["cost"] = float(cost_of(bits, flip, success))
        actions.append(kernel)
    return actions


def prune_dominated(actions):
    """Blackwell pruning: drop ``b`` if some ``a`` with ``c(a) <= c(b)``
    strictly dominates it (``b`` can never be Bellman-optimal).  Strict
    means the dominance is not reciprocated: Blackwell-equivalent kernels
    (e.g. the same quantizer under two spans) are kept."""
    kept = []
    for b in actions:
        dominated = False
        for a in actions:
            if a is b:
                continue
            if a["cost"] <= b["cost"] + 1e-12 \
                    and blackwell_dominates(a, b) \
                    and not blackwell_dominates(b, a):
                dominated = True
                break
        if not dominated:
            kept.append(b)
    return kept


def bayesian_cost(roll0, roll1, c10, c01):
    return 0.5 * (roll0["mean_cost"] + roll1["mean_cost"]) \
        + 0.5 * (c10 * roll0["p_fa"] + c01 * roll1["p_md"])


def _budget_wrapped(policy):
    """Adapt a myopic ``policy(l, step)`` to the budget-aware signature
    ``(l, step, b_remaining)`` (the myopic policy is budget-blind and may
    overspend; rollout_budget forces the terminal decision on exhaustion)."""
    return lambda l, step, b: policy(l, step)


def evaluate(policy, actions, horizon, c10, c01, n_runs, seeds):
    roll0 = rollout(policy, actions, horizon, 0, n_runs=n_runs, seed=seeds[0])
    roll1 = rollout(policy, actions, horizon, 1, n_runs=n_runs, seed=seeds[1])
    return {
        "delay0": roll0["mean_delay"],
        "delay1": roll1["mean_delay"],
        "p_fa": roll0["p_fa"],
        "p_md": roll1["p_md"],
        "cost0": roll0["mean_cost"],
        "cost1": roll1["mean_cost"],
        "info1": roll1["mean_info"],
        "total_cost": bayesian_cost(roll0, roll1, c10, c01),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/bellman_detection_gate.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=4000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--budget", type=int, default=None,
                        help="observation budget (defaults to the horizon)")
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument("--l-max", type=float, default=10.0)
    parser.add_argument("--c10", type=float, default=20.0)
    parser.add_argument("--c01", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    c10, c01 = args.c10, args.c01
    a_log_odds = float(np.log((1.0 - args.beta) / args.alpha))
    b_log_odds = float(np.log(args.beta / (1.0 - args.alpha)))

    # ---------------- Part D1: exact Bellman oracle (Q = 1) ----------------
    mu0, var0 = 4.0, 4.0
    links = [(10.0, 16.0), (7.0, 12.0)]
    library = make_library(
        mu0, var0, links,
        bits_list=[1, 2], spans=[3.0, 6.0],
        flips=[0.05, 0.15], successes=[0.85, 0.95],
        cost_of=lambda bits, flip, success: float(bits),
    )
    n_raw = len(library)
    pruned = prune_dominated(library)
    n_pruned = n_raw - len(pruned)

    grid = grid_bellman_value(pruned, args.horizon, c10, c01,
                              grid=args.grid, l_max=args.l_max)
    policies = {
        "bellman": bellman_action_policy(grid, pruned, c10, c01),
        "tau_pred": tau_pred_policy(pruned, a_log_odds, b_log_odds),
        "chernoff": chernoff_policy(pruned, a_log_odds, b_log_odds),
        "dpd": dpd_policy(pruned, c10, c01),
        "static": static_policy(pruned, a_log_odds, b_log_odds),
    }
    d1 = {}
    for name, pol in policies.items():
        d1[name] = evaluate(pol, pruned, args.horizon, c10, c01,
                            args.n_runs, (100 + 2 * len(d1), 200 + 2 * len(d1)))

    # alpha-vector oracle cross-check on a tiny instance (single 1-bit
    # action so the vector count stays tractable)
    tiny = [a for a in pruned if len(a["p0"]) == 3][:1]
    exact = exact_alpha_vectors(tiny, 3, c10, c01)
    tiny_grid = grid_bellman_value(tiny, 3, c10, c01, grid=501, l_max=6.0)
    pi0 = 0.5
    v_exact = min(a * pi0 + b for a, b, _, _ in exact["vectors"])
    v_grid = float(np.interp(0.0, tiny_grid["ls"], tiny_grid["v"]))
    oracle_agreement = abs(v_exact - v_grid)

    bellman_cost = d1["bellman"]["total_cost"]
    d1_gap = {name: (d1[name]["total_cost"] - bellman_cost) / bellman_cost
              for name in policies}
    # Bellman minimizes the Bayesian cost; the myopic family must not beat
    # it (the best myopic within 3% MC tolerance of Bellman)
    best_myopic_d1 = min(
        d1[name]["total_cost"] for name in policies if name != "bellman"
    )
    d1_passed = bellman_cost <= best_myopic_d1 * 1.03

    # ---------------- Part D2: dynamic quantizer ---------------------------
    # the quantizer design knob is the span (and bits); the Bellman policy
    # chooses per-belief actions, so the belief-dependent quantizer exists
    # iff it leaves the Chernoff-best fixed action somewhere on the axis
    chernoff_best = int(max(range(len(pruned)),
                            key=lambda i: pruned[i]["chernoff"]
                            / max(pruned[i]["cost"], 1e-12)))
    ls = grid["ls"]
    chosen = np.full(len(ls), -1)
    for i, l in enumerate(ls):
        pi = belief_from_log_odds(float(l))
        best_v = min(c01 * pi, c10 * (1.0 - pi))
        best = -1 if c01 * pi <= c10 * (1.0 - pi) else -2
        rem = args.horizon
        v = grid["values"][rem]
        for ai, act in enumerate(pruned):
            val = 0.0
            for k in range(len(act["p0"])):
                target = float(np.clip(l + act["llr"][k], ls[0], ls[-1]))
                j = int(np.clip(int(np.searchsorted(ls, target)), 1,
                                len(ls) - 1))
                w = (target - ls[j - 1]) / max(ls[j] - ls[j - 1], 1e-300)
                val += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) \
                    * (v[j - 1] * (1.0 - w) + v[j] * w)
            val += act["cost"]
            if val < best_v - 1e-12:
                best_v = val
                best = ai
        chosen[i] = best
    cont_region = chosen >= 0
    switching = np.mean(chosen[cont_region] != chernoff_best) \
        if cont_region.any() else 0.0
    fixed_chernoff_pol = static_policy(pruned, a_log_odds, b_log_odds)
    d2_fixed = evaluate(fixed_chernoff_pol, pruned, args.horizon, c10, c01,
                        args.n_runs, (300, 400))
    d2 = {
        "n_actions_library": n_raw,
        "n_pruned_blackwell": n_pruned,
        "chernoff_best_action": chernoff_best,
        "fraction_continuation_states": float(cont_region.mean()),
        "fraction_states_leaving_chernoff_action": float(switching),
        "adaptive_gain_vs_fixed_chernoff":
            (d2_fixed["total_cost"] - bellman_cost) / d2_fixed["total_cost"],
        "fixed_chernoff_cost": d2_fixed["total_cost"],
    }

    # ------- Part D1b: advice/003 Gate D1 -- budget-state Bellman ----------
    # Q = 1, R = 3 links, bits 1..4, P = 2 power levels, horizon 6, budget 6.
    # The exact finite-horizon value V_t(pi, B) (explicit budget state) is
    # compared against the myopic strategies (tau_pred / Chernoff / one-step
    # Delta-P_D lookahead / static floor-cover) in detection cycles and
    # resource efficiency.  The verdict states whether the Bellman oracle
    # meaningfully improves over the myopic family: if not, "agent-ifying"
    # the system is not worth it (advice/003, the gate decision rule of section 14).
    b10 = args.horizon
    b_budget = args.budget if args.budget is not None else b10
    links3 = [(10.0, 16.0), (7.0, 12.0), (5.5, 9.5)]
    powers = [1.0, 2.0]
    library3 = []
    for (mu1_0, var1_0), bits, p in product(links3, [1, 2, 3, 4], powers):
        # power scales the deflection: mu1/var1 move toward the base values
        # by the power factor; cost = bits + 1 premium for high power
        mu1 = mu0 + (mu1_0 - mu0) * p
        var1 = var0 + (var1_0 - var0) * p
        kernel = action_kernels(mu0, var0, mu1, var1, bits, 0.05, 0.95, 4.0)
        kernel["cost"] = float(bits + (1 if p > 1.0 else 0))
        library3.append(kernel)
    pruned3 = prune_dominated(library3)
    bval = budget_bellman_value(pruned3, b10, b_budget, c10, c01,
                                grid=args.grid, l_max=args.l_max)
    bpol = budget_bellman_policy(bval, pruned3, c10, c01)
    b10_used = b_budget
    db1b = {}
    for name, pol in {
        "bellman_budget": bpol,
        "tau_pred": _budget_wrapped(tau_pred_policy(
            pruned3, a_log_odds, b_log_odds)),
        "chernoff": _budget_wrapped(chernoff_policy(
            pruned3, a_log_odds, b_log_odds)),
        "dpd": _budget_wrapped(dpd_policy(pruned3, c10, c01)),
        "static_floor_cover": _budget_wrapped(static_policy(
            pruned3, a_log_odds, b_log_odds)),
    }.items():
        db1b[name] = {
            "delay0": rollout_budget(
                pol, pruned3, b_budget, 0, n_runs=args.n_runs,
                seed=700 + 2 * len(db1b))["mean_delay"],
            "delay1": rollout_budget(
                pol, pruned3, b_budget, 1, n_runs=args.n_runs,
                seed=800 + 2 * len(db1b))["mean_delay"],
        }
        r0 = rollout_budget(pol, pruned3, b_budget, 0, n_runs=args.n_runs,
                            seed=900 + 2 * len(db1b))
        r1 = rollout_budget(pol, pruned3, b_budget, 1, n_runs=args.n_runs,
                            seed=1000 + 2 * len(db1b))
        db1b[name].update({
            "p_fa": r0["p_fa"],
            "p_md": r1["p_md"],
            "cost0": r0["mean_cost"],
            "cost1": r1["mean_cost"],
            "info1": r1["mean_info"],
            "total_cost": bayesian_cost(r0, r1, c10, c01),
        })
    bellman_cost = db1b["bellman_budget"]["total_cost"]
    myopic = {k: v for k, v in db1b.items() if k != "bellman_budget"}
    best_myopic_cost = min(v["total_cost"] for v in myopic.values())
    best_myopic_name = min(myopic, key=lambda k: myopic[k]["total_cost"])
    best_myopic_delay = min(v["delay1"] for v in myopic.values())
    bellman_delay = db1b["bellman_budget"]["delay1"]
    cost_gain = (best_myopic_cost - bellman_cost) / best_myopic_cost
    cycles_gain = (best_myopic_delay - bellman_delay) / best_myopic_delay
    # honest verdict: the Bellman oracle "matters" only if it improves the
    # myopic family by a material margin (>= 5% in cost or cycles) without
    # a worse error profile (Bellman P_FA/P_MD within +0.02 of the myopic)
    errors_ok = (db1b["bellman_budget"]["p_fa"]
                 <= max(v["p_fa"] for v in myopic.values()) + 0.02
                 and db1b["bellman_budget"]["p_md"]
                 <= max(v["p_md"] for v in myopic.values()) + 0.02)
    meaningful = errors_ok and (cost_gain >= 0.05 or cycles_gain >= 0.05)
    d1b = {
        "scale": {"q": 1, "links": len(links3), "bits": [1, 2, 3, 4],
                  "powers": powers, "horizon": b10, "budget": b_budget},
        "n_actions_raw": len(library3),
        "n_actions_after_blackwell": len(pruned3),
        "policies": db1b,
        "best_myopic": best_myopic_name,
        "cost_gain_vs_best_myopic": cost_gain,
        "cycles_gain_vs_best_myopic": cycles_gain,
        "errors_within_myopic": errors_ok,
        "meaningful_improvement": meaningful,
        "verdict": (
            "Bellman improves the myopic family materially; the "
            "belief-state value-approximation step (advice step 2) is "
            "justified" if meaningful else
            "Bellman does not materially beat the myopic strategies; "
            "agent-ifying the static pipeline is not worth it at this "
            "scale"
        ),
    }
    d1b_passed = bellman_cost <= best_myopic_cost * 1.02 + 1e-9

    # ------ Part D1c: residual-triggered adaptation (advice/003 §4) -------
    # the mathematical Reflexion: the controller standardizes the realized
    # Bellman residual against both model-conditional distributions and
    # triggers robust/explore modes when the running-mean statistic
    # tau = min(|mean_0|, |mean_1|) leaves the correct-model range.  The
    # gate reports tau and the realized (true-kernel) Bayesian cost of the
    # adaptive policy vs the naive Bellman under success/erasure mismatch.
    model_c = [
        action_kernels(4.0, 4.0, 8.0, 12.0, 1, 0.05, 0.95, 4.0),
        action_kernels(4.0, 4.0, 10.0, 16.0, 2, 0.05, 0.95, 4.0),
    ]
    model_c[0]["cost"] = 1.0
    model_c[1]["cost"] = 2.0
    v_c = grid_bellman_value(model_c, args.horizon, c10, c01,
                             grid=args.grid, l_max=args.l_max)

    def _mismatch_library(mu1, var1, bits, flip_new=None, success_new=None):
        from uav_otfs_isac.detection_quantization import quantizer_edges
        from uav_otfs_isac.detection_information import (
            post_communication_likelihoods,
        )
        edges, values = quantizer_edges(4.0, 4.0, mu1, var1, bits, 4.0)
        info = post_communication_likelihoods(
            4.0, 4.0, mu1, var1, edges, values, bits,
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
            "cost": 1.0 if bits == 1 else 2.0,
            "chernoff": float(info["chernoff"]),
        }

    def _realized_cost(pol, model_lib, truth_lib):
        r0 = rollout_mismatch(pol, model_lib, truth_lib, 0,
                              n_runs=args.n_runs, seed=2000, budget=8)
        r1 = rollout_mismatch(pol, model_lib, truth_lib, 1,
                              n_runs=args.n_runs, seed=2100, budget=8)
        return bayesian_cost(r0, r1, c10, c01)

    nominal_c = bellman_action_policy(v_c, model_c, c10, c01)
    d1c = {"scenarios": {}}
    for name, spec in {
        "correct_model": {},
        "success_0.5": {"success_new": 0.5},
        "flip_0.45": {"flip_new": 0.45},
        "flip_0.35_success_0.7": {"flip_new": 0.35, "success_new": 0.7},
    }.items():
        truth_c = [
            _mismatch_library(8.0, 12.0, 1, **spec),
            _mismatch_library(10.0, 16.0, 2, **spec),
        ]
        adaptive, monitor = residual_adaptive_policy(
            v_c, model_c, c10, c01, args.horizon,
            residual_margin=0.25, explore_rounds=2, warmup=20)
        if spec:
            rollout_mismatch(adaptive, model_c, truth_c, 1,
                             n_runs=args.n_runs, seed=2200, budget=8)
        else:
            rollout_budget(adaptive, model_c, 8, 1,
                           n_runs=args.n_runs, seed=2200)
        cost_naive = _realized_cost(
            lambda l, step, b: nominal_c(l, step), model_c, truth_c)
        cost_adaptive = _realized_cost(adaptive, model_c, truth_c)
        d1c["scenarios"][name] = {
            "tau": float(monitor["tau"]),
            "triggered": bool(monitor["triggered"]),
            "mode": monitor["mode"],
            "modes_seen": sorted(set(monitor["modes"])),
            "naive_bayes_cost": cost_naive,
            "adaptive_bayes_cost": cost_adaptive,
            "adaptive_gain": (cost_naive - cost_adaptive) / max(cost_naive, 1e-12),
        }
    # honest gate claims: (1) the correct model never triggers; (2) the
    # designed erasure mismatch triggers and the adapted controller is not
    # worse than the naive one; (3) adaptation never hurts in any scenario.
    # Flip-only mismatches leave the LLR atom structure unchanged and are
    # not detected by this statistic -- reported as a documented limitation.
    d1c_passed = True
    d1c_passed = d1c_passed and not d1c["scenarios"]["correct_model"]["triggered"]
    d1c_passed = d1c_passed \
        and d1c["scenarios"]["correct_model"]["tau"] < 0.1
    d1c_passed = d1c_passed \
        and d1c["scenarios"]["success_0.5"]["triggered"]
    d1c_passed = d1c_passed \
        and d1c["scenarios"]["success_0.5"]["adaptive_gain"] >= -0.02
    for row in d1c["scenarios"].values():
        d1c_passed = d1c_passed and row["adaptive_gain"] >= -0.02
    d1c["limitation"] = (
        "flip-only mismatches preserve the LLR atom magnitudes and are not "
        "detected by the residual mean statistic (documented limitation)"
    )

    # ---------------- Part D3: multi-target dual decomposition --------------
    actions_q0 = make_library(
        4.0, 4.0, [(10.0, 16.0)], bits_list=[1, 2], spans=[3.0, 6.0],
        flips=[0.05, 0.15], successes=[0.85, 0.95],
        cost_of=lambda bits, flip, success: float(bits),
    )
    actions_q1 = make_library(
        4.0, 4.0, [(7.0, 12.0)], bits_list=[1, 2], spans=[3.0, 6.0],
        flips=[0.05, 0.15], successes=[0.85, 0.95],
        cost_of=lambda bits, flip, success: float(bits),
    )
    actions_q0 = prune_dominated(actions_q0)
    actions_q1 = prune_dominated(actions_q1)
    h3 = 4
    joint = joint_bellman_value([actions_q0, actions_q1], h3, c10, c01,
                                grid=61, l_max=8.0)
    mid = int((len(joint["ls"]) - 1) / 2)
    v_star = float(joint["v"][mid, mid])
    jpol = joint_bellman_policy(joint, [actions_q0, actions_q1], c10, c01)
    jout = rollout_multi(jpol, [actions_q0, actions_q1], [0, 1],
                         n_runs=args.n_runs // 2, seed=500, max_steps=24)

    # weak-duality bound: Lagrangian relaxation of the per-cycle budget on
    # the cost objective; per-target inner problems charge (1+lam)*c(a) per
    # observation (nu = 0), the relaxation subtracts lam per cycle, lam >= 0
    best_bound = -np.inf
    for lam in (0.0, 0.5, 1.0):
        v_q = [
            dual_decomposed_value(actions_q0, h3, c10, c01, nu=0.0,
                                  lam=1.0 + lam, grid=121),
            dual_decomposed_value(actions_q1, h3, c10, c01, nu=0.0,
                                  lam=1.0 + lam, grid=121),
        ]
        bound = v_q[0]["v"][mid] + v_q[1]["v"][mid] - lam * h3
        best_bound = max(best_bound, float(bound))

    best_dual = None
    for nu1, lam in product((0.3, 0.5, 0.7), (0.2, 0.5, 1.0)):
        v_q = [
            dual_decomposed_value(actions_q0, h3, c10, c01, nu=nu1, lam=lam,
                                  grid=121),
            dual_decomposed_value(actions_q1, h3, c10, c01, nu=1.0 - nu1,
                                  lam=lam, grid=121),
        ]
        dpol = decomposed_scheduler(v_q, [actions_q0, actions_q1], 1, c10, c01)
        dout = rollout_multi(dpol, [actions_q0, actions_q1], [0, 1],
                             n_runs=args.n_runs // 2, seed=600 + int(nu1 * 100),
                             max_steps=24)
        row = {"nu1": nu1, "lam": lam,
               "worst_delay": float(dout["mean_worst_delay"]),
               "mean_delays": [float(x) for x in dout["mean_delays"]],
               "mean_costs": [float(x) for x in dout["mean_costs"]],
               "p_fa": [float(x) for x in dout["p_fa"]],
               "p_md": [float(x) for x in dout["p_md"]],
               "bayes_cost": bayesian_cost(
                   {"mean_cost": dout["mean_costs"][0], "p_fa": dout["p_fa"][0]},
                   {"mean_cost": dout["mean_costs"][1], "p_md": dout["p_md"][1]},
                   c10, c01)}
        # a row that never observes (zero worst delay) is not a detection
        # policy: it declares instantly and only looks cheap through the
        # error terms; keep the comparison among policies that actually
        # sense
        if dout["mean_worst_delay"] < 0.5:
            continue
        if best_dual is None or dout["mean_worst_delay"] < best_dual["worst_delay"]:
            best_dual = row
    joint_bayes = bayesian_cost(
        {"mean_cost": jout["mean_costs"][0], "p_fa": jout["p_fa"][0]},
        {"mean_cost": jout["mean_costs"][1], "p_md": jout["p_md"][1]},
        c10, c01)
    d3 = {
        "joint_value_star": v_star,
        "joint_worst_delay": float(jout["mean_worst_delay"]),
        "joint_mean_delays": [float(x) for x in jout["mean_delays"]],
        "joint_bayes_cost": joint_bayes,
        "best_dual_bound": float(best_bound),
        "weak_duality_gap": v_star - best_bound,
        "dual_best_row": best_dual,
        "dual_bayes_cost": best_dual["bayes_cost"],
        "worst_delay_gap_vs_joint":
            best_dual["worst_delay"] - float(jout["mean_worst_delay"]),
    }
    d3_passed = d3["weak_duality_gap"] >= -0.05

    # ---------------- Part D4: fundamental-limit gap ------------------------
    bellman_roll = d1["bellman"]
    bounds = information_lower_bounds(bellman_roll["p_fa"], bellman_roll["p_md"],
                                      pruned)
    info_ok = bellman_roll["info1"] >= bounds["d_1"] - 0.05
    t_lower = bounds["t1_lower"]
    c_lower = bounds["c1_lower"]
    d4 = {
        "p_fa": bellman_roll["p_fa"],
        "p_md": bellman_roll["p_md"],
        "d_1": bounds["d_1"],
        "i_max_plus": bounds["i_max_plus"],
        "rho_plus": bounds["rho_plus"],
        "e1_sum_i_plus": bellman_roll["info1"],
        "info_bound_holds": info_ok,
        "t_lower": t_lower,
        "t_bellman": bellman_roll["delay1"],
        "t_tau_pred": d1["tau_pred"]["delay1"],
        "bellman_over_lower": bellman_roll["delay1"] / max(t_lower, 1e-12),
        "tau_over_lower": d1["tau_pred"]["delay1"] / max(t_lower, 1e-12),
        "c_lower": c_lower,
        "c_bellman": bellman_roll["cost1"],
    }
    d4_passed = (info_ok
                 and t_lower <= bellman_roll["delay1"] + 0.5
                 and bellman_roll["delay1"] <= d1["tau_pred"]["delay1"] + 0.5)

    passed = d1_passed and d1b_passed and d1c_passed and d3_passed \
        and d4_passed
    payload = {
        "gate": "posterior-state-bellman-active-detection",
        "params": {
            "horizon": args.horizon, "grid": args.grid, "l_max": args.l_max,
            "c10": c10, "c01": c01, "alpha": args.alpha, "beta": args.beta,
            "n_runs": args.n_runs,
        },
        "part_d1": {
            "link": {"mu0": mu0, "var0": var0, "links": links},
            "n_actions": n_raw,
            "oracle_agreement_alpha_vs_grid": oracle_agreement,
            "policies": d1,
            "cost_gap_vs_bellman": d1_gap,
        },
        "part_d1b": d1b,
        "part_d1c": d1c,
        "part_d2": d2,
        "part_d3": d3,
        "part_d4": d4,
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Part D1: exact Bellman oracle (Q=1)")
    print(f"  actions: {n_raw} raw, {n_pruned} pruned by Blackwell dominance")
    print(f"  alpha-vector vs grid agreement at pi=0.5: {oracle_agreement:.5f}")
    for name in policies:
        row = d1[name]
        print(f"  {name:9s}: total cost {row['total_cost']:.4f}  "
              f"delay H1 {row['delay1']:.2f} / H0 {row['delay0']:.2f}  "
              f"P_FA {row['p_fa']:.4f}  P_MD {row['p_md']:.4f}")
    print(f"  cost gaps vs Bellman: { {k: round(v, 4) for k, v in d1_gap.items()} }")
    print("Part D2: dynamic quantizer")
    print(f"  leaves Chernoff-best action in {d2['fraction_states_leaving_chernoff_action']:.3f} "
          f"of continuation states; adaptive gain vs fixed: "
          f"{d2['adaptive_gain_vs_fixed_chernoff']:.4f}")
    print("Part D1b: advice/003 Gate D1 -- budget-state Bellman oracle")
    print(f"  scale Q=1 R={len(links3)} b=1..4 P=2 H={b10} B={b_budget}; "
          f"actions {len(library3)} raw -> {len(pruned3)} after Blackwell")
    for name, row in db1b.items():
        print(f"  {name:16s}: total cost {row['total_cost']:.4f}  "
              f"cycles H1 {row['delay1']:.2f}  cost {row['cost1']:.2f}  "
              f"P_FA {row['p_fa']:.4f}  P_MD {row['p_md']:.4f}")
    print(f"  cost gain vs best myopic ({best_myopic_name}): {cost_gain:.4f}; "
          f"cycles gain: {cycles_gain:.4f}; errors within myopic: {errors_ok}")
    print(f"  verdict: {d1b['verdict']}")
    print("Part D1c: residual-triggered adaptation (advice/003 section 4)")
    for name, row in d1c["scenarios"].items():
        print(f"  {name:24s}: tau {row['tau']:.3f}  triggered {row['triggered']}  "
              f"mode {row['mode']:8s}  naive {row['naive_bayes_cost']:.3f}  "
              f"adaptive {row['adaptive_bayes_cost']:.3f}  "
              f"gain {row['adaptive_gain']:+.3f}")
    print("Part D3: multi-target dual decomposition (Q=2)")
    print(f"  joint exact value {v_star:.4f}; best dual bound {best_bound:.4f} "
          f"(weak-duality gap {v_star - best_bound:.4f})")
    print(f"  worst-target mean delay: joint {jout['mean_worst_delay']:.2f} vs "
          f"dual {best_dual['worst_delay']:.2f} at nu1={best_dual['nu1']}, "
          f"lam={best_dual['lam']}")
    print("Part D4: fundamental-limit gap")
    print(f"  info bound holds: {info_ok}  (E1[sum I+] = {bellman_roll['info1']:.3f} "
          f">= d(1-beta||alpha) = {bounds['d_1']:.3f})")
    print(f"  T_lower = {t_lower:.2f} < T_bellman = {bellman_roll['delay1']:.2f} "
          f"(ratio {d4['bellman_over_lower']:.2f}) <= T_tau = {d1['tau_pred']['delay1']:.2f} "
          f"(ratio {d4['tau_over_lower']:.2f})")
    print(f"passed={passed}")


if __name__ == "__main__":
    main()
