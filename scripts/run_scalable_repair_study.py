from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.reliability import physical_failure_groups
from uav_otfs_isac.replication import (
    optimize_beam_dual_layer_repair,
    optimize_dual_layer_chance_portfolio,
    optimize_greedy_dual_layer_repair,
)
from uav_otfs_isac.scenario import build_models, uav_geometry
from scripts.run_replication_realism_study import bootstrap_ci, native_resource_labels


def timed(function):
    start = perf_counter(); result = function(); return result, perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/scalable_repair_study.json")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--path-fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--beam-width", type=int, default=8)
    args = parser.parse_args(); cfg = load_config(args.config)
    limits = np.full(cfg.num_targets, args.epsilon); positions = uav_geometry(cfg.num_uavs)
    rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset; models = build_models(cfg, np.random.default_rng(seed))
        paths = physical_failure_groups(models, positions, "owner_angle_path", 2)
        resources = native_resource_labels(models)
        for budget in args.budgets:
            common = dict(
                models=models, budget_bits=budget, minimum_pd=args.minimum_pd,
                target_weights=cfg.qos_weights, violation_limits=limits,
                path_groups=paths, native_resources=resources,
                strength=args.strength, path_failure_fraction=args.path_fraction,
                domain_capacities=[budget // 2, budget - budget // 2],
                false_alarm_rate=cfg.false_alarm_rate,
            )
            selection, selection_seconds = timed(lambda: optimize_dual_layer_chance_portfolio(
                **common, replication_mode="cross_domain", maximum_copies=1,
                objective_mode="fair"
            ))
            oracle, oracle_seconds = timed(lambda: optimize_dual_layer_chance_portfolio(
                **common, replication_mode="cross_domain", maximum_copies=2,
                objective_mode="fair"
            ))
            greedy, greedy_seconds = timed(lambda: optimize_beam_dual_layer_repair(
                **common, beam_width=args.beam_width
            ))
            selection_worst = float(np.max(selection.violation_probability_per_target))
            oracle_worst = float(np.max(oracle.violation_probability_per_target))
            greedy_worst = float(np.max(greedy.violation_probability_per_target))
            oracle_gain = selection_worst - oracle_worst
            rows.append({
                "seed": seed, "budget_bits": budget,
                "selection_worst_violation": selection_worst,
                "oracle_worst_violation": oracle_worst,
                "greedy_worst_violation": greedy_worst,
                "oracle_gain": oracle_gain,
                "greedy_gain": selection_worst - greedy_worst,
                "oracle_gain_capture": (
                    (selection_worst - greedy_worst) / oracle_gain if oracle_gain > 1e-12 else None
                ),
                "selection_feasible": selection.feasible,
                "oracle_feasible": oracle.feasible, "greedy_feasible": greedy.feasible,
                "oracle_seconds": oracle_seconds, "greedy_seconds": greedy_seconds,
                "selection_seconds": selection_seconds,
                "greedy_used_bits": greedy.used_bits,
            })
    summary = []
    for budget in args.budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        capture = [row["oracle_gain_capture"] for row in group if row["oracle_gain_capture"] is not None]
        summary.append({
            "budget_bits": budget,
            "mean_selection_worst_violation": float(np.mean([row["selection_worst_violation"] for row in group])),
            "mean_oracle_worst_violation": float(np.mean([row["oracle_worst_violation"] for row in group])),
            "mean_greedy_worst_violation": float(np.mean([row["greedy_worst_violation"] for row in group])),
            "ratio_of_aggregate_gains": float(np.sum([row["greedy_gain"] for row in group]) / max(np.sum([row["oracle_gain"] for row in group]), 1e-12)),
            "mean_instance_gain_capture_when_defined": float(np.mean(capture)),
            "gain_capture_bootstrap_ci95": bootstrap_ci(capture),
            "oracle_system_feasible_rate": float(np.mean([row["oracle_feasible"] for row in group])),
            "greedy_system_feasible_rate": float(np.mean([row["greedy_feasible"] for row in group])),
            "mean_oracle_seconds": float(np.mean([row["oracle_seconds"] for row in group])),
            "mean_greedy_seconds": float(np.mean([row["greedy_seconds"] for row in group])),
            "mean_speedup": float(np.mean([row["oracle_seconds"] / max(row["greedy_seconds"], 1e-12) for row in group])),
        })
    payload = {"seeds": args.seeds, "beam_width": args.beam_width,
               "summary": summary, "instances": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
