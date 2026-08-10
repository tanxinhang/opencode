"""Physical hard channel gate with increasing UAV count.

Channel difficulty is derived from link SNR through the BPSK bit-flip
formula and log-normal outage, instead of arbitrary flip/success values.
UAV count R is swept under baseline, hard, and extreme physical channels.
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

from uav_otfs_isac.nomp_refinement import (
    nomp_wta_greedy_joint_multi,
    wta_greedy_joint_multi,
)
from uav_otfs_isac.physical_link_model import (
    bpsk_bit_flip_probability,
    lognormal_outage_success,
    report_link_snr_db,
)
from scripts.run_joint_power_comparison import ucb_wta_greedy_joint_multi
from scripts.run_unknown_environment_gate import exact_comm


def make_physical_scenario(
    seed,
    reports,
    targets,
    *,
    reference_snr_db,
    threshold_db,
    shadowing_db,
):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(targets):
        positions = rng.uniform(0.0, 200.0, (reports + 1, 2))
        snr_db = report_link_snr_db(
            positions,
            0,
            reference_snr_db=reference_snr_db,
        )
        flips = np.asarray([
            bpsk_bit_flip_probability(float(value))
            for value in snr_db[1:]
        ])
        successes = np.asarray([
            lognormal_outage_success(
                float(value), threshold_db, shadowing_db
            )
            for value in snr_db[1:]
        ])
        owner = rng.uniform(0.2, 0.5)
        deltas = rng.uniform(0.5, 2.5, reports)
        out.append((float(owner), deltas, flips, successes))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/physical_hard_channel_gate.json")
    parser.add_argument("--figure", default="paper_figures/physical_hard_channel.png")
    parser.add_argument("--reports", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget-multiplier", type=int, default=4)
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    environments = {
        "baseline": dict(reference_snr_db=25.0, threshold_db=5.0, shadowing_db=3.0),
        "hard": dict(reference_snr_db=8.0, threshold_db=10.0, shadowing_db=4.0),
        "extreme": dict(reference_snr_db=3.0, threshold_db=12.0, shadowing_db=5.0),
    }
    rows = []
    for difficulty, params in environments.items():
        for reports in args.reports:
            budget = args.budget_multiplier * reports * 2
            wta = []
            ucb = []
            nomp = []
            exact = []
            for seed in range(args.seeds):
                scenario = make_physical_scenario(
                    10000 + seed,
                    reports,
                    2,
                    **params,
                )
                wta.append(float(wta_greedy_joint_multi(
                    scenario, budget, min_cover=False
                )["worst_pd"]))
                ucb.append(float(ucb_wta_greedy_joint_multi(
                    scenario,
                    budget,
                    noise_scale=0.2,
                    seed=0,
                    min_cover=True,
                    refine=True,
                )["worst_pd"]))
                nomp.append(float(nomp_wta_greedy_joint_multi(
                    scenario, budget
                )["worst_pd"]))
                if reports == 2:
                    exact.append(exact_comm(scenario, budget, args.grid))
            rows.append({
                "difficulty": difficulty,
                "reports": reports,
                "budget": budget,
                "wta_worst_mean": float(np.mean(wta)),
                "ucb_nomp_worst_mean": float(np.mean(ucb)),
                "nomp_worst_mean": float(np.mean(nomp)),
                "exact_worst_mean": (
                    float(np.mean(exact)) if exact else None
                ),
            })
            print(json.dumps(rows[-1], indent=2))

    payload = {
        "gate": "physical-hard-channel",
        "seeds": args.seeds,
        "budget_multiplier": args.budget_multiplier,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    colors = {
        "baseline": "#31a354",
        "hard": "#e6550d",
        "extreme": "#756bb1",
    }
    plt.figure(figsize=(7.5, 5))
    for difficulty in environments:
        for field, style in (
            ("wta_worst_mean", "--"),
            ("nomp_worst_mean", "-"),
        ):
            series = [
                next(
                    row for row in rows
                    if row["difficulty"] == difficulty
                    and row["reports"] == reports
                )
                for reports in args.reports
            ]
            plt.plot(
                args.reports,
                [row[field] for row in series],
                style,
                color=colors[difficulty],
                marker="o",
                label=f"{difficulty} {field.split('_')[0]}",
            )
    plt.xlabel("UAV count R")
    plt.ylabel("Mean worst P_D")
    plt.title("Physical channel difficulty across UAV count")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
