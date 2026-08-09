"""Joint sensing-communication greedy gate.

The proposed algorithm chooses reports by expected-P_D gain per effective
communication bit.  The effective cost uses the erasure survival and the
BSC fidelity ``(1 - 2 epsilon)^2``, so the greedy explicitly trades sensing
value against node-to-node communication reliability under the same bit
budget.
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

from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.physical_link_model import build_physical_link_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_sensing_communication_gate.json")
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--grid", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = "physical_report_links"
    rows = []
    for budget in args.budgets:
        for offset in range(args.seeds):
            seed = cfg.seed + offset
            models = build_physical_link_models(
                cfg,
                seed,
                reference_snr_db=15.0,
                threshold_db=5.0,
                shadowing_db=3.0,
            )
            bits = expected_pd_greedy_select(
                models,
                budget,
                cfg.false_alarm_rate,
                grid=args.grid,
                cost_mode="bits",
            )
            reliable = expected_pd_greedy_select(
                models,
                budget,
                cfg.false_alarm_rate,
                grid=args.grid,
                cost_mode="reliability_weighted",
            )
            capacity = expected_pd_greedy_select(
                models,
                budget,
                cfg.false_alarm_rate,
                grid=args.grid,
                cost_mode="capacity_weighted",
            )
            rows.append({
                "budget_bits": budget,
                "seed": seed,
                "bits_worst_pd": float(np.min(bits.expected_pd)),
                "reliable_worst_pd": float(np.min(reliable.expected_pd)),
                "bits_used": bits.used_bits,
                "reliable_used": reliable.used_bits,
                "capacity_worst_pd": float(np.min(capacity.expected_pd)),
                "capacity_used": capacity.used_bits,
                "reliable_improvement_pp": float(
                    (np.min(reliable.expected_pd) - np.min(bits.expected_pd))
                    * 100.0
                ),
                "capacity_improvement_pp": float(
                    (np.min(capacity.expected_pd) - np.min(bits.expected_pd))
                    * 100.0
                ),
            })
    summary = []
    for budget in args.budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        summary.append({
            "budget_bits": budget,
            "bits_worst_pd_mean": float(np.mean([
                row["bits_worst_pd"] for row in group
            ])),
            "reliable_worst_pd_mean": float(np.mean([
                row["reliable_worst_pd"] for row in group
            ])),
            "reliable_improvement_mean_pp": float(np.mean([
                row["reliable_improvement_pp"] for row in group
            ])),
            "reliable_better_rate": float(np.mean([
                row["reliable_improvement_pp"] > 1e-9 for row in group
            ])),
            "capacity_worst_pd_mean": float(np.mean([
                row["capacity_worst_pd"] for row in group
            ])),
            "capacity_improvement_mean_pp": float(np.mean([
                row["capacity_improvement_pp"] for row in group
            ])),
            "capacity_better_rate": float(np.mean([
                row["capacity_improvement_pp"] > 1e-9 for row in group
            ])),
        })
    payload = {
        "gate": "joint-sensing-communication-greedy",
        "profile": profile,
        "budgets": args.budgets,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
