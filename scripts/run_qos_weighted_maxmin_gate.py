"""QoS-weighted max-min gate for multi-target NOMP allocation.

Targets carry different detection floors and priorities.  The objective is
the worst normalized slack ``w_q (v_q - l_q) / l_q``; the gate compares
plain NOMP, QoS-aware NOMP, and an exact brute-force over the per-target
robust frontiers.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.nomp_refinement import (
    nomp_wta_greedy_joint_multi,
    qos_scores,
    target_scores,
)
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_heterogeneous_robust_power_bit_options,
    pareto_options,
)
from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)


def exact_qos_worst(scenario, budget, floors, weights, grid):
    frontiers = []
    for owner, deltas, flips, successes in scenario:
        options = enumerate_heterogeneous_robust_power_bit_options(
            owner,
            deltas,
            [(0.0, float(value)) for value in flips],
            [(float(value), 1.0) for value in successes],
            power_levels=np.arange(budget + 1, dtype=float),
            bit_options=np.arange(3, dtype=int),
            budget=budget,
            grid=grid,
        )
        frontiers.append(pareto_options(options, "robust_pd"))
    best = -np.inf
    for combo in itertools.product(*frontiers):
        costs = [item[0] for item in combo]
        values = [item[1] for item in combo]
        if sum(costs) > budget:
            continue
        best = max(best, float(np.min(qos_scores(values, floors, weights))))
    return float(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/qos_weighted_maxmin_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--floor-a", type=float, default=0.30)
    parser.add_argument("--floor-b", type=float, default=0.45)
    parser.add_argument("--weight-a", type=float, default=1.0)
    parser.add_argument("--weight-b", type=float, default=1.3)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    floors = [args.floor_a, args.floor_b]
    weights = [args.weight_a, args.weight_b]
    summary = []
    for budget in args.budgets:
        plain_qos = []
        qos_worst = []
        exact = []
        qos_raw_pd = []
        for seed in range(args.seeds):
            scenario = make_comm_mismatch_scenario(
                10000 + seed, 2, 2
            )
            plain = nomp_wta_greedy_joint_multi(scenario, budget)
            plain_raw = target_scores(
                scenario,
                plain["powers"],
                plain["bits"],
                args.grid,
            )
            raw_plain = qos_scores(
                plain_raw,
                floors,
                weights,
            )
            plain_qos.append(float(np.min(raw_plain)))
            qos_result = nomp_wta_greedy_joint_multi(
                scenario,
                budget,
                floors=floors,
                weights=weights,
            )
            qos_worst.append(float(qos_result["qos_worst"]))
            qos_raw_pd.append(float(qos_result["worst_pd"]))
            exact.append(exact_qos_worst(
                scenario, budget, floors, weights, args.grid
            ))
        summary.append({
            "budget": budget,
            "plain_nomp_qos_worst_mean": float(np.mean(plain_qos)),
            "qos_nomp_qos_worst_mean": float(np.mean(qos_worst)),
            "qos_nomp_worst_pd_mean": float(np.mean(qos_raw_pd)),
            "exact_qos_worst_mean": float(np.mean(exact)),
            "qos_improvement": float(np.mean(qos_worst) - np.mean(plain_qos)),
            "qos_nomp_gap_to_exact": float(
                np.mean(exact) - np.mean(qos_worst)
            ),
        })

    payload = {
        "gate": "qos-weighted-maxmin",
        "floors": floors,
        "weights": weights,
        "seeds": args.seeds,
        "targets": 2,
        "reports": 2,
        "per_link_channels": True,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
