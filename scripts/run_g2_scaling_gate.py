"""G2 scaling gate: performance and runtime as M (UAVs) and Q (targets) grow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select
from scripts.run_g2_system_sweep import (
    _pd_vector,
    _score_topk,
    owners_for,
)
from scripts.run_g2_algorithm_negative_gates import _exact_pd_greedy


def run_gate(*, output: Path, seeds: int, budgets_per_target: int) -> None:
    base = load_config("config/demo.yaml")
    configurations = ((8, 3), (12, 3), (12, 5), (16, 5), (16, 8))
    rows = []
    for num_uavs, num_targets in configurations:
        cfg = replace(
            base,
            num_uavs=num_uavs,
            num_targets=num_targets,
            owners=owners_for(num_uavs, num_targets),
        )
        budget = budgets_per_target * num_targets
        for offset in range(seeds):
            models = build_models(
                cfg, np.random.default_rng(cfg.seed + offset)
            )
            qos = np.zeros(num_targets)
            weights = np.ones(num_targets)
            perf = np.ones(num_targets)
            start = perf_counter()
            conditional = greedy_select(
                models, budget, qos, weights, perf, qos_first=False
            )
            conditional_seconds = perf_counter() - start
            conditional_pd = _pd_vector(
                models, conditional.scheduled, cfg.false_alarm_rate
            )
            start = perf_counter()
            exact = _exact_pd_greedy(
                models, budget, cfg.false_alarm_rate
            )
            exact_seconds = perf_counter() - start
            exact_pd = _pd_vector(
                models, exact, cfg.false_alarm_rate
            )
            independent = _score_topk(
                models, budget, "independent_deflection"
            )
            independent_pd = _pd_vector(
                models, independent, cfg.false_alarm_rate
            )
            all_scheduled = [
                set(range(model.num_uavs)) for model in models
            ]
            all_pd = _pd_vector(
                models, all_scheduled, cfg.false_alarm_rate
            )
            rows.append({
                "num_uavs": num_uavs,
                "num_targets": num_targets,
                "budget_bits": budget,
                "seed_offset": offset,
                "conditional_mean_pd": float(np.mean(conditional_pd)),
                "conditional_worst_pd": float(np.min(conditional_pd)),
                "conditional_seconds": conditional_seconds,
                "exact_pd_mean_pd": float(np.mean(exact_pd)),
                "exact_pd_worst_pd": float(np.min(exact_pd)),
                "exact_pd_seconds": exact_seconds,
                "independent_mean_pd": float(np.mean(independent_pd)),
                "independent_worst_pd": float(np.min(independent_pd)),
                "all_scheduled_mean_pd": float(np.mean(all_pd)),
                "conditional_vs_independent": float(
                    np.mean(conditional_pd) - np.mean(independent_pd)
                ),
                "exact_vs_independent": float(
                    np.mean(exact_pd) - np.mean(independent_pd)
                ),
            })
    summary = []
    for num_uavs, num_targets in configurations:
        group = [
            row for row in rows
            if row["num_uavs"] == num_uavs
            and row["num_targets"] == num_targets
        ]
        summary.append({
            "num_uavs": num_uavs,
            "num_targets": num_targets,
            "budget_bits": budgets_per_target * num_targets,
            "conditional_mean_pd": float(np.mean([
                row["conditional_mean_pd"] for row in group
            ])),
            "exact_pd_mean_pd": float(np.mean([
                row["exact_pd_mean_pd"] for row in group
            ])),
            "independent_mean_pd": float(np.mean([
                row["independent_mean_pd"] for row in group
            ])),
            "all_scheduled_mean_pd": float(np.mean([
                row["all_scheduled_mean_pd"] for row in group
            ])),
            "conditional_vs_independent": float(np.mean([
                row["conditional_vs_independent"] for row in group
            ])),
            "exact_vs_independent": float(np.mean([
                row["exact_vs_independent"] for row in group
            ])),
            "conditional_worst_pd": float(np.mean([
                row["conditional_worst_pd"] for row in group
            ])),
            "independent_worst_pd": float(np.mean([
                row["independent_worst_pd"] for row in group
            ])),
            "mean_conditional_seconds": float(np.mean([
                row["conditional_seconds"] for row in group
            ])),
            "mean_exact_pd_seconds": float(np.mean([
                row["exact_pd_seconds"] for row in group
            ])),
        })
    payload = {
        "gate": "G2-scaling",
        "seeds": seeds,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/g2_scaling_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budgets-per-target", type=int, default=10)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets_per_target=args.budgets_per_target,
    )


if __name__ == "__main__":
    main()
