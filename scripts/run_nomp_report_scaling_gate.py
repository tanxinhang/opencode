"""NOMP performance versus report count under per-link channels.

The per-target budget grows linearly with the report count, and each report
keeps its own BSC flip and erasure.  Expected P_D is marginalized over
erasures exactly up to ``max_exact_reports`` and by Monte Carlo above it, so
the online allocator remains usable as R grows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.nomp_refinement import (
    nomp_wta_greedy_joint_multi,
    wta_greedy_joint_multi,
)
from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/nomp_report_scaling_gate.json")
    parser.add_argument("--figure", default="paper_figures/nomp_report_scaling.png")
    parser.add_argument("--reports", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget-multiplier", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for reports in args.reports:
        budget = args.budget_multiplier * reports * 2
        wta_worsts = []
        nomp_worsts = []
        nomp_rounds = []
        nomp_seconds = []
        for seed in range(args.seeds):
            scenario = make_comm_mismatch_scenario(
                10000 + seed, reports, 2
            )
            wta_worsts.append(float(wta_greedy_joint_multi(
                scenario, budget, min_cover=False
            )["worst_pd"]))
            start = time.perf_counter()
            result = nomp_wta_greedy_joint_multi(scenario, budget)
            nomp_seconds.append(time.perf_counter() - start)
            nomp_worsts.append(float(result["worst_pd"]))
            nomp_rounds.append(int(result["refine_rounds"]))
        rows.append({
            "reports": reports,
            "budget": budget,
            "wta_greedy_worst_mean": float(np.mean(wta_worsts)),
            "nomp_greedy_worst_mean": float(np.mean(nomp_worsts)),
            "nomp_mean_refine_rounds": float(np.mean(nomp_rounds)),
            "nomp_mean_seconds": float(np.mean(nomp_seconds)),
        })
        print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "nomp-report-scaling",
        "seeds": args.seeds,
        "budget_multiplier": args.budget_multiplier,
        "targets": 2,
        "per_link_channels": True,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rs = [row["reports"] for row in rows]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(rs, [row["wta_greedy_worst_mean"] for row in rows],
             "v-", color="#3182bd", label="WTA-Greedy")
    plt.plot(rs, [row["nomp_greedy_worst_mean"] for row in rows],
             "D-", color="#31a354", label="NOMP-Greedy")
    plt.xlabel("Reports per target R")
    plt.ylabel("Mean worst P_D")
    plt.title("Per-link channel, Q=2, budget 6R")
    plt.grid(alpha=0.3)
    plt.legend()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
