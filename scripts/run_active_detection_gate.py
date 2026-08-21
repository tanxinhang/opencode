"""Gate B: active vs static vs myopic sequential detection.

The gate compares three budget-constrained sequential detection policies
over a set of targets with heterogeneous reporting links (per-report
quantizer bits, BSC flip probability, erasure probability, and a power
level action that scales the sensing noncentrality):

1. ``static``    -- winner-take-all allocation of actions by I^+/cost
                   computed once, then applied in fixed order.
2. ``myopic``    -- every cycle pick the (target, action) with the largest
                   exact one-step P_D gain per cost.
3. ``active``    -- every cycle pick the target with the largest predicted
                   remaining cycles tau = (eta(n+1) - n*I^+)/I^+ (Wald
                   first-order estimate), then spend its best I^+/cost
                   action on it.
4. ``bellman``   -- the allocation-time closed loop of advice/003 section
                   12: per-target finite-horizon budget Bellman values
                   ``V_h(l, b)`` are precomputed, and every cycle the
                   controller reallocates the *remaining* budget against
                   the *current* posteriors by playing the (target, action)
                   with the largest value gain
                   ``G = V(l, b) - [c(a) + E V(l + llr, b - c(a))]``.

Decisions are exact sequential NP tests: each target accumulates its
received observations (heterogeneous sequences allowed) and decides when
the exact ``P_D(n) >= 1 - beta`` at ``P_FA <= alpha``.  The gate reports
worst-target mean decision time, mean decision time, terminal worst-target
P_D, and budget usage; it passes when the active policy is at least as
good as both baselines on the worst-target metric without degrading the
terminal P_D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.active_detection_bellman import (
    belief_from_log_odds,
    budget_bellman_value,
)
from uav_otfs_isac.detection_information import (
    cycles_to_pd,
    post_communication_likelihoods,
    predicted_remaining_cycles,
    sequential_pd_sequence,
    threshold_curve,
)
from uav_otfs_isac.reporting import quantizer_from_gaussian_range


class Action:
    __slots__ = ("key", "cost", "i_plus", "p0_y", "p1_y", "llr", "curve")

    def __init__(self, key, cost, i_plus, p0_y, p1_y):
        self.key = key
        self.cost = cost
        self.i_plus = i_plus
        self.p0_y = p0_y
        self.p1_y = p1_y
        self.llr = np.log(np.asarray(p1_y, dtype=float)
                          / np.asarray(p0_y, dtype=float))
        self.curve = None

    def threshold(self, alpha, max_n, grid_step):
        if self.curve is None:
            self.curve = threshold_curve(
                self.p1_y, self.p0_y, alpha, max_n, grid_step,
            )
        return self.curve


def build_target(rng: np.random.Generator, l_acc: int, reports: int) -> dict:
    """One target with ``reports`` randomized reporting links and power
    level actions ``k in {1, 2}`` scaling the sensing noncentrality."""
    actions = []
    for i in range(reports):
        snr_db = float(rng.uniform(-4.0, 3.0))
        noncentrality = l_acc * 10 ** (snr_db / 10.0)
        mu0 = float(l_acc)
        var0 = float(l_acc)
        mu1 = mu0 + noncentrality
        var1 = var0 + 2.0 * noncentrality
        bits = int(rng.integers(1, 4))
        flip = float(rng.uniform(0.02, 0.15))
        success = float(rng.uniform(0.5, 0.9))
        edges, values = quantizer_from_gaussian_range(
            np.array([mu0]), np.array([[var0]]),
            np.array([mu1]), np.array([[var1]]),
            bits,
        )
        for power in (1, 2):
            mu1_p = mu0 + power * noncentrality
            var1_p = var0 + 2.0 * power * noncentrality
            info = post_communication_likelihoods(
                mu0, var0, mu1_p, var1_p, edges, values,
                bits, flip, success,
            )
            cost = bits + (power - 1)
            actions.append(Action(
                (i, power), cost, float(info["kl_plus"]),
                info["p0_y"], info["p1_y"],
            ))
    return {"actions": actions}


def run_policy(targets, policy, alpha, pd_star, max_n, budget, grid_step,
               bellman_vals=None):
    """Serial policy simulation.  Returns per-target outcomes."""
    n = np.zeros(len(targets), dtype=int)
    decided = np.zeros(len(targets), dtype=bool)
    pd_now = np.full(len(targets), float(alpha))
    observations = [[] for _ in targets]
    log_odds = np.zeros(len(targets), dtype=float)
    remaining = float(budget)

    def pd_after(q, action):
        return float(sequential_pd_sequence(
            observations[q] + [(action.p1_y, action.p0_y)],
            alpha, grid_step,
        )["pd"])

    def value_gain(q, action, step):
        """``G = V_h(l, b) - [c + E_a V_{h-1}(l + llr, b - c)]`` from the
        precomputed budget value of the target (allocation-time closed
        loop: remaining budget + current posterior, advice/003 section 12).
        The belief proxy is the H1-expected drift ``l = n * I+`` (the same
        first-order model the ``active`` tau-policy uses), keeping the
        comparison within the gate's scheduling model.
        """
        vb = bellman_vals[q]
        ls = vb["ls"]
        values = vb["values"]
        horizon = vb["horizon"]
        rem = int(np.clip(horizon - step, 0, horizon))
        b_here = int(np.clip(int(round(remaining)), 0, vb["budget"]))
        b_next = int(np.clip(b_here - action.cost, 0, vb["budget"]))
        l = float(np.clip(log_odds[q], ls[0], ls[-1]))
        pi = belief_from_log_odds(l)
        v_now = float(np.interp(l, ls, values[rem, b_here]))
        exp = 0.0
        for k in range(len(action.p0_y)):
            target = float(np.clip(l + action.llr[k], ls[0], ls[-1]))
            v_next = float(np.interp(target, ls, values[rem - 1, b_next]))
            exp += (pi * action.p1_y[k] + (1.0 - pi) * action.p0_y[k]) \
                * v_next
        return v_now - (float(action.cost) + exp)

    static_order = None
    if policy == "static":
        order = sorted(
            ((q, a) for q, t in enumerate(targets) for a in t["actions"]),
            key=lambda qa: -qa[1].i_plus / qa[1].cost,
        )
        static_order = order

    budget_used = 0.0
    cycles = 0
    while cycles < 200:
        undecided = [q for q in range(len(targets)) if not decided[q]]
        if not undecided:
            break
        if policy == "static":
            chosen = None
            for q, a in static_order:
                if not decided[q] and a.cost <= remaining + 1e-12:
                    chosen = (q, a)
                    break
            if chosen is None:
                break
        elif policy == "myopic":
            best_score = -np.inf
            chosen = None
            for q in undecided:
                for a in targets[q]["actions"]:
                    if a.cost > remaining + 1e-12:
                        continue
                    gain = pd_after(q, a) - pd_now[q]
                    score = gain / a.cost
                    if score > best_score:
                        best_score = score
                        chosen = (q, a)
            if chosen is None:
                break
        elif policy == "active":
            best_tau = -np.inf
            chosen = None
            for q in undecided:
                best_action = max(
                    targets[q]["actions"],
                    key=lambda a: a.i_plus / a.cost,
                )
                if best_action.cost > remaining + 1e-12:
                    continue
                curve = best_action.threshold(alpha, max_n, grid_step)
                tau = predicted_remaining_cycles(
                    n[q], best_action.i_plus, curve[n[q]],
                )
                if tau > best_tau:
                    best_tau = tau
                    chosen = (q, best_action)
            if chosen is None:
                break
        else:  # bellman: budget-aware value-gain reallocation every cycle
            best_g = -np.inf
            chosen = None
            for q in undecided:
                for a in targets[q]["actions"]:
                    if a.cost > remaining + 1e-12:
                        continue
                    g = value_gain(q, a, cycles)
                    if g > best_g:
                        best_g = g
                        chosen = (q, a)
            if chosen is None:
                break
        q, action = chosen
        observations[q].append((action.p1_y, action.p0_y))
        n[q] += 1
        remaining -= action.cost
        budget_used += action.cost
        log_odds[q] += float(action.i_plus)
        pd_now[q] = pd_after(q, action)
        if pd_now[q] >= pd_star:
            decided[q] = True
        cycles += 1

    T = np.where(decided, n, n)
    final_min_pd = float(np.min(np.where(decided, 1.0, pd_now)))
    return {
        "worst_T": float(np.max(T)),
        "mean_T": float(np.mean(T)),
        "final_min_pd": final_min_pd,
        "decided_fraction": float(np.mean(decided)),
        "budget_used": budget_used,
    }


def oracle_cycles(targets, alpha, pd_star, max_n, grid_step):
    """Reference: per-target best-action i.i.d. cycle count (no budget)."""
    best = []
    for t in targets:
        counts = [
            cycles_to_pd(a.p1_y, a.p0_y, alpha, pd_star, max_n, grid_step)
            for a in t["actions"]
        ]
        finite = [c for c in counts if c is not None]
        best.append(min(finite) if finite else max_n)
    return float(np.max(best))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/active_detection_gate.json")
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--reports", type=int, default=3)
    parser.add_argument("--l-acc", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pd-star", type=float, default=0.9)
    parser.add_argument("--max-n", type=int, default=14)
    parser.add_argument("--budget", type=float, default=9.0)
    parser.add_argument("--grid-step", type=float, default=0.02)
    args = parser.parse_args()

    policies = ("static", "myopic", "active", "bellman")
    aggregated = {p: {k: [] for k in (
        "worst_T", "mean_T", "final_min_pd", "decided_fraction",
        "budget_used",
    )} for p in policies}
    oracle_list = []
    bellman_costs = (20.0, 20.0)
    for seed in range(args.seed_base, args.seed_base + args.seeds):
        rng = np.random.default_rng(seed)
        targets = [
            build_target(rng, args.l_acc, args.reports)
            for _ in range(args.targets)
        ]
        bellman_vals = [
            budget_bellman_value(
                [
                    {
                        "p0": a.p0_y, "p1": a.p1_y, "llr": a.llr,
                        "cost": float(a.cost), "i_plus": a.i_plus,
                    }
                    for a in t["actions"]
                ],
                args.max_n, int(round(args.budget)),
                bellman_costs[0], bellman_costs[1],
                grid=201, l_max=10.0,
            )
            for t in targets
        ]
        for policy in policies:
            outcome = run_policy(
                targets, policy, args.alpha, args.pd_star,
                args.max_n, args.budget, args.grid_step,
                bellman_vals=bellman_vals if policy == "bellman" else None,
            )
            for key, value in outcome.items():
                aggregated[policy][key].append(value)
        oracle_list.append(oracle_cycles(
            targets, args.alpha, args.pd_star, args.max_n, args.grid_step,
        ))

    summary = {}
    for policy in policies:
        summary[policy] = {
            key: float(np.mean(values))
            for key, values in aggregated[policy].items()
        }
    summary["oracle_worst_T"] = float(np.mean(oracle_list))

    aw = summary["active"]["worst_T"]
    passed = (
        aw <= min(summary["static"]["worst_T"], summary["myopic"]["worst_T"]) + 1e-6
        and summary["active"]["final_min_pd"]
        >= min(summary["static"]["final_min_pd"],
               summary["myopic"]["final_min_pd"]) - 0.005
    )
    # the closed-loop Bellman policy must not be worse than the best of the
    # baselines on the worst-target metric, without degrading terminal P_D
    bellman_passed = (
        summary["bellman"]["worst_T"]
        <= min(summary[p]["worst_T"] for p in policies if p != "bellman")
        + 0.05
        and summary["bellman"]["final_min_pd"]
        >= min(summary[p]["final_min_pd"] for p in policies if p != "bellman")
        - 0.005
    )
    closed_loop_gain = (
        min(summary[p]["worst_T"] for p in policies if p != "bellman")
        - summary["bellman"]["worst_T"]
    ) / max(summary["bellman"]["worst_T"], 1e-12)

    payload = {
        "gate": "active-vs-static-vs-myopic-sequential-detection",
        "seeds": args.seeds,
        "targets": args.targets,
        "reports": args.reports,
        "alpha": args.alpha,
        "pd_star": args.pd_star,
        "max_n": args.max_n,
        "budget": args.budget,
        "summary": summary,
        "closed_loop": {
            "bellman_worst_T": summary["bellman"]["worst_T"],
            "best_baseline_worst_T": min(
                summary[p]["worst_T"] for p in policies if p != "bellman"
            ),
            "closed_loop_gain": closed_loop_gain,
            "bellman_passed": bellman_passed,
        },
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'policy':<10}{'worst_T':>10}{'mean_T':>10}{'min_pd':>10}"
          f"{'decided':>10}{'budget':>10}")
    for policy in policies:
        s = summary[policy]
        print(f"{policy:<10}{s['worst_T']:>10.2f}{s['mean_T']:>10.2f}"
              f"{s['final_min_pd']:>10.3f}{s['decided_fraction']:>10.3f}"
              f"{s['budget_used']:>10.1f}")
    print(f"oracle worst_T={summary['oracle_worst_T']:.2f}")
    print(f"closed-loop Bellman gain vs best baseline: {closed_loop_gain:+.3f} "
          f"(bellman_passed={bellman_passed})")
    print(f"passed={passed}")


if __name__ == "__main__":
    main()