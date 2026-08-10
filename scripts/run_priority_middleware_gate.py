"""Priority middleware gate: MAPPO-guided weights vs plain NOMP.

The middleware policy picks which target to prioritize, NOMP solves the
weighted max-min, and the unweighted worst P_D is fed back as reward.  This
is not a 0-1 switch: the weights continuously change NOMP's objective, which
lets the collaboration escape local optima that plain NOMP cannot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.mappo_nomp_adapter import (
    NompRequirement,
    PriorityNompAdapter,
)
from uav_otfs_isac.nomp_refinement import nomp_wta_greedy_joint_multi
from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)
from scripts.run_joint_power_comparison import robust_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/priority_middleware_gate.json")
    parser.add_argument("--figure", default="paper_figures/priority_middleware.png")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--candidate-budget", type=int, default=8)
    args = parser.parse_args()

    cases = [
        (10000, 4, 32),
        (10001, 4, 24),
        (10002, 4, 32),
        (10003, 4, 24),
        (10004, 4, 32),
    ]
    rows = []
    for seed, reports, budget in cases:
        scenario = make_comm_mismatch_scenario(seed, reports, 2)
        requirement = NompRequirement(
            modes="auto",
            budget=budget,
            max_refine_rounds=50,
            samples=args.samples,
            candidate_budget=args.candidate_budget,
        )
        plain = nomp_wta_greedy_joint_multi(
            scenario,
            budget,
            samples=args.samples,
            candidate_budget=args.candidate_budget,
        )["worst_pd"]
        state_dim = len(robust_state(scenario[0], budget))
        adapter = PriorityNompAdapter(
            state_dim,
            state_builder=robust_state,
        )
        guided = adapter.propose_and_allocate(
            scenario,
            requirement,
            episodes=args.episodes,
            seed=seed,
        )
        rows.append({
            "seed": seed,
            "reports": reports,
            "budget": budget,
            "plain_nomp_worst": float(plain),
            "priority_nomp_worst": float(guided["worst_pd"]),
            "gain": float(guided["worst_pd"] - plain),
            "best_priority": guided["trace"][-1]["priority_target"],
        })
        print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "priority-middleware",
        "episodes": args.episodes,
        "samples": args.samples,
        "candidate_budget": args.candidate_budget,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = [
        f"R={row['reports']} B={row['budget']}" for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, [row["plain_nomp_worst"] for row in rows],
            width, label="Plain NOMP")
    plt.bar(x + width / 2, [row["priority_nomp_worst"] for row in rows],
            width, label="Priority-NOMP")
    plt.xticks(x, labels)
    plt.ylabel("Mean worst P_D")
    plt.title("MAPPO-guided priority middleware")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
