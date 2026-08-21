"""Gate D2-D: deployable controllers (advice/004 sections 6, Case 3).

The exact joint oracle (joint_delay_value) has an exponential state space
and is not deployable beyond Q = 2.  This gate evaluates the deployable
controller family -- dual G-value, Whittle index, one/two-step rollout --
that share the per-target delay values (built once, linear in Q) and the
calibrated two-threshold stopping rule, against the oracle (Q = 2) and
the per-cycle heuristics (Q = 2, 3):

- deployment gap vs the oracle (Q = 2): how much planning value is lost
  by going from exact to deployable;
- deployment gain vs the heuristics (Q = 2, 3): the deployable family
  must keep the multi-target planning value;
- per-cycle decision cost (scalability, linear in Q).

Verdict: the best deployable controller is the reference deployment;
the oracle is the (unreachable) performance bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.active_detection_bellman import (
    calibrate_sprt_boundaries,
    delay_value_iteration,
    joint_delay_policy,
    joint_delay_value,
    make_deployable_controllers,
    rollout_delay_multi,
)
from scripts.run_d2_objective_gate import make_library, make_multi_heuristic


def build_scenario(kind):
    """Q = 2 (strong + weak) or Q = 3 (strong + medium + weak)."""
    strong = make_library(
        [(10.0, 16.0), (8.5, 14.0)], [1, 2, 3], [1.0, 2.0],
        [0.02], [0.98],
        cost_of=lambda bits, power: bits + (1 if power > 1.0 else 0))
    weak = make_library(
        [(7.0, 11.0), (6.0, 10.0)], [1, 2, 3], [1.0, 2.0],
        [0.08], [0.9],
        cost_of=lambda bits, power: bits + (1 if power > 1.0 else 0))
    if kind == "q2":
        return [strong, weak]
    medium = make_library(
        [(8.0, 13.0), (7.0, 11.5)], [1, 2, 3], [1.0, 2.0],
        [0.05], [0.93],
        cost_of=lambda bits, power: bits + (1 if power > 1.0 else 0))
    return [strong, medium, weak]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/d2_deployment_gate.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=800)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    args = parser.parse_args()

    payload = {
        "gate": "d2-deployable-controllers",
        "params": {
            "horizon": args.horizon, "budget": args.budget,
            "alpha": args.alpha, "beta": args.beta,
            "n_runs": args.n_runs,
        },
        "scenarios": {},
    }
    for scenario, A in (("q2", build_scenario("q2")),
                        ("q3", build_scenario("q3"))):
        q = len(A)
        bounds = []
        for acts in A:
            cal = calibrate_sprt_boundaries(
                acts, args.alpha, args.beta, args.budget,
                n_runs=min(args.n_runs, 200), seed=args.seed + 20,
                margin=1.0, points=7)
            bounds.append((float(cal["a_bound"]), float(cal["b_bound"])))
        nu = tuple([1.0 / q] * q)
        t0 = time.time()
        singles = [
            delay_value_iteration(acts, args.horizon, args.budget,
                                  64.0, 64.0, grid=101, l_max=8.0,
                                  cycle_cost=float(nu[i]),
                                  bounds=bounds[i])
            for i, acts in enumerate(A)
        ]
        singles_time = time.time() - t0
        ctrls = make_deployable_controllers(A, singles, bounds, nu=nu,
                                            lam=1.0)

        policies = dict(ctrls)
        policies["myopic_dpd"] = make_multi_heuristic(
            "myopic", A, bounds, args.alpha, args.beta)
        policies["static_floor_cover"] = make_multi_heuristic(
            "static", A, bounds, args.alpha, args.beta)
        if scenario == "q2":
            vj = joint_delay_value(A, args.horizon, args.budget,
                                   64.0, 64.0, grid=33, l_max=8.0,
                                   nu=nu, bounds=bounds)
            policies["joint_oracle"] = joint_delay_policy(vj, A, 64.0, 64.0)

        # per-cycle decision cost: time one policy call after warmup
        l0 = np.zeros(q)
        t0 = time.time()
        for _ in range(100):
            for pol in policies.values():
                pol(l0, 0, float(args.budget))
        per_cycle_us = (time.time() - t0) / (100 * len(policies)) * 1e6

        rows = {}
        for name, pol in policies.items():
            out = rollout_delay_multi(pol, A, [1] * q, args.budget,
                                      n_runs=args.n_runs,
                                      seed=args.seed + 40, max_steps=40)
            rows[name] = {
                "worst_target_delay": float(out["mean_worst_delay"]),
                "e1_delays": [float(x) for x in out["e1_delays"]],
                "p_md": [float(x) for x in out["p_md"]],
                "mean_costs": [float(x) for x in out["mean_costs"]],
            }
        best_deploy = min(
            (k for k in rows if k in ctrls),
            key=lambda k: rows[k]["worst_target_delay"])
        best_deploy_worst = rows[best_deploy]["worst_target_delay"]
        myopic_worst = rows["myopic_dpd"]["worst_target_delay"]
        deploy_gain = (myopic_worst - best_deploy_worst) / myopic_worst
        summary = {
            "best_deployable": best_deploy,
            "deployable_gain_vs_myopic": float(deploy_gain),
            "singles_build_s": float(singles_time),
            "per_cycle_decision_us": float(per_cycle_us),
        }
        if scenario == "q2":
            oracle_worst = rows["joint_oracle"]["worst_target_delay"]
            summary["oracle_worst_delay"] = float(oracle_worst)
            summary["deployment_gap_vs_oracle"] = float(
                (best_deploy_worst - oracle_worst) / oracle_worst)
        payload["scenarios"][scenario] = {
            "bounds": [[round(float(b[0]), 3), round(float(b[1]), 3)]
                       for b in bounds],
            "policies": rows,
            "summary": summary,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for scenario in ("q2", "q3"):
        s = payload["scenarios"][scenario]
        print(f"Scenario {scenario}:")
        print(f"  {'policy':<20}{'worst E1[T]':>12}"
              f"{'P_MD':>18}")
        for name, row in s["policies"].items():
            print(f"  {name:<20}{row['worst_target_delay']:>12.2f}"
                  f"{str([round(float(x), 3) for x in row['p_md']]):>18}")
        print(f"  best deployable: {s['summary']['best_deployable']} "
              f"(gain vs myopic {s['summary']['deployable_gain_vs_myopic']:+.3f})")
        if "deployment_gap_vs_oracle" in s["summary"]:
            print(f"  deployment gap vs oracle: "
                  f"{s['summary']['deployment_gap_vs_oracle']:+.3f}")
        print(f"  singles build {s['summary']['singles_build_s']:.2f}s, "
              f"per-cycle decision {s['summary']['per_cycle_decision_us']:.1f}us")


if __name__ == "__main__":
    main()
