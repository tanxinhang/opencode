"""Q x R scenario comparison for joint power-bit allocation.

Covers clean homogeneous, clean heterogeneous, and per-link communication
mismatch scenarios across target count Q and report count R.  Exact is the
winner-take-all frontier for clean models and the robust per-link frontier
when R=2 under communication mismatch.
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

from uav_otfs_isac.joint_power_bit import exact_joint_power_bit_maxmin
from uav_otfs_isac.nomp_refinement import (
    nomp_wta_greedy_joint_multi,
    wta_greedy_joint_multi,
)
from uav_otfs_isac.power_split_theory import (
    winner_take_all_proportional_options,
)
from uav_otfs_isac.robust_joint_power_bit import (
    enumerate_heterogeneous_robust_power_bit_options,
    pareto_options,
)
from scripts.run_joint_power_comm_mismatch_gate import (
    make_comm_mismatch_scenario,
)
from scripts.run_joint_power_comparison import (
    greedy_joint_multi,
    make_scenario,
    ucb_wta_greedy_joint_multi,
)


def exact_clean(scenario, budget, grid):
    groups = [
        winner_take_all_proportional_options(
            float(target[0]),
            target[1:],
            bit_options=np.arange(3, dtype=int),
            budget=budget,
            grid=grid,
        )
        for target in scenario
    ]
    return float(exact_joint_power_bit_maxmin(groups, budget))


def exact_comm(scenario, budget, grid):
    groups = []
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
        groups.append(pareto_options(options, "robust_pd"))
    return float(exact_joint_power_bit_maxmin(groups, budget))


def make_scenario_for(mode, seed, reports, targets):
    if mode == "comm_mismatch":
        return make_comm_mismatch_scenario(seed, reports, targets)
    return make_scenario(
        seed, reports, targets, heterogeneous=(mode == "heterogeneous")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/qr_scenario_comparison.json")
    parser.add_argument("--figure", default="paper_figures/qr_scenario_comparison.png")
    parser.add_argument("--modes", nargs="+", default=[
        "homogeneous", "heterogeneous", "comm_mismatch",
    ])
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--reports", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget-multiplier", type=int, default=4)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for targets in args.targets:
            for reports in args.reports:
                budget = args.budget_multiplier * targets
                greedy_worsts = []
                wta_worsts = []
                ucb_worsts = []
                nomp_worsts = []
                exact_worsts = []
                nomp_seconds = []
                for seed in range(args.seeds):
                    scenario = make_scenario_for(
                        mode, 10000 + seed, reports, targets
                    )
                    if mode != "comm_mismatch":
                        greedy_worsts.append(greedy_joint_multi(
                            scenario, budget
                        ))
                    wta_worsts.append(float(wta_greedy_joint_multi(
                        scenario, budget, min_cover=False
                    )["worst_pd"]))
                    ucb = ucb_wta_greedy_joint_multi(
                        scenario,
                        budget,
                        noise_scale=0.2,
                        seed=seed,
                        min_cover=True,
                        refine=True,
                    )
                    ucb_worsts.append(float(ucb["worst_pd"]))
                    start = time.perf_counter()
                    nomp = nomp_wta_greedy_joint_multi(scenario, budget)
                    nomp_seconds.append(time.perf_counter() - start)
                    nomp_worsts.append(float(nomp["worst_pd"]))
                    if mode == "comm_mismatch":
                        if reports <= 2:
                            exact_worsts.append(exact_comm(
                                scenario, budget, args.grid
                            ))
                    else:
                        exact_worsts.append(exact_clean(
                            scenario, budget, args.grid
                        ))
                rows.append({
                    "mode": mode,
                    "targets": targets,
                    "reports": reports,
                    "budget": budget,
                    "greedy_worst_mean": (
                        float(np.mean(greedy_worsts))
                        if greedy_worsts else None
                    ),
                    "wta_greedy_worst_mean": float(np.mean(wta_worsts)),
                    "ucb_nomp_worst_mean": float(np.mean(ucb_worsts)),
                    "nomp_greedy_worst_mean": float(np.mean(nomp_worsts)),
                    "exact_worst_mean": (
                        float(np.mean(exact_worsts))
                        if exact_worsts else None
                    ),
                    "nomp_mean_seconds": float(np.mean(nomp_seconds)),
                })
                print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "qr-scenario-comparison",
        "seeds": args.seeds,
        "budget_multiplier": args.budget_multiplier,
        "grid": args.grid,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    colors = {2: "#3182bd", 3: "#e6550d", 4: "#756bb1"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, mode in zip(axes, args.modes):
        mode_rows = [row for row in rows if row["mode"] == mode]
        qs = sorted({row["targets"] for row in mode_rows})
        for reports in args.reports:
            series = [
                next(
                    row for row in mode_rows
                    if row["targets"] == q and row["reports"] == reports
                )
                for q in qs
            ]
            ax.plot(
                qs,
                [row["wta_greedy_worst_mean"] for row in series],
                "--",
                color=colors[reports],
                label=f"WTA R={reports}",
            )
            ax.plot(
                qs,
                [row["nomp_greedy_worst_mean"] for row in series],
                "-",
                color=colors[reports],
                marker="o",
                label=f"NOMP R={reports}",
            )
        title = {
            "homogeneous": "Clean homogeneous",
            "heterogeneous": "Clean heterogeneous",
            "comm_mismatch": "Per-link comm mismatch",
        }[mode]
        ax.set_title(title)
        ax.set_xlabel("Targets Q")
        ax.set_ylabel("Mean worst P_D")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
