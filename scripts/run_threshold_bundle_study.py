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
    optimize_dual_layer_chance_portfolio,
    _experimental_optimize_threshold_bundle_portfolio,
)
from uav_otfs_isac.scenario import build_models, uav_geometry
from scripts.run_replication_realism_study import native_resource_labels


def timed(function):
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/threshold_bundle_smoke.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--bundle-depths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--path-fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    positions = uav_geometry(cfg.num_uavs)
    limits = np.full(cfg.num_targets, args.epsilon)
    rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        models = build_models(cfg, np.random.default_rng(seed))
        paths = physical_failure_groups(models, positions, "owner_angle_path", 2)
        resources = native_resource_labels(models)
        for budget in args.budgets:
            common = dict(
                models=models, budget_bits=budget, minimum_pd=args.minimum_pd,
                target_weights=cfg.qos_weights, violation_limits=limits,
                path_groups=paths, native_resources=resources,
                strength=args.strength,
                path_failure_fraction=args.path_fraction,
                domain_capacities=[budget // 2, budget - budget // 2],
                false_alarm_rate=cfg.false_alarm_rate,
            )
            selection, _ = timed(lambda: optimize_dual_layer_chance_portfolio(
                **common, replication_mode="cross_domain", maximum_copies=1
            ))
            oracle, oracle_seconds = timed(
                lambda: optimize_dual_layer_chance_portfolio(
                    **common, replication_mode="cross_domain", maximum_copies=2
                )
            )
            selection_worst = float(np.max(
                selection.violation_probability_per_target
            ))
            oracle_worst = float(np.max(oracle.violation_probability_per_target))
            oracle_gain = selection_worst - oracle_worst
            for depth in args.bundle_depths:
                bundle, bundle_seconds = timed(
                    lambda depth=depth: _experimental_optimize_threshold_bundle_portfolio(
                        **common, maximum_bundle_actions=depth
                    )
                )
                bundle_worst = float(np.max(
                    bundle.violation_probability_per_target
                ))
                rows.append({
                    "seed": seed, "budget_bits": budget,
                    "bundle_depth": depth,
                    "selection_worst_violation": selection_worst,
                    "oracle_worst_violation": oracle_worst,
                    "bundle_worst_violation": bundle_worst,
                    "oracle_gain": oracle_gain,
                    "bundle_gain": selection_worst - bundle_worst,
                    "oracle_feasible": oracle.feasible,
                    "bundle_feasible": bundle.feasible,
                    "oracle_seconds": oracle_seconds,
                    "bundle_seconds": bundle_seconds,
                })
    summary = []
    for budget in args.budgets:
        for depth in args.bundle_depths:
            group = [row for row in rows if
                     row["budget_bits"] == budget
                     and row["bundle_depth"] == depth]
            oracle_gain = sum(row["oracle_gain"] for row in group)
            bundle_gain = sum(row["bundle_gain"] for row in group)
            summary.append({
                "budget_bits": budget, "bundle_depth": depth,
                "ratio_of_aggregate_gains": (
                    bundle_gain / oracle_gain if oracle_gain > 1e-12 else None
                ),
                "oracle_system_feasible_rate": float(np.mean([
                    row["oracle_feasible"] for row in group
                ])),
                "bundle_system_feasible_rate": float(np.mean([
                    row["bundle_feasible"] for row in group
                ])),
                "mean_oracle_seconds": float(np.mean([
                    row["oracle_seconds"] for row in group
                ])),
                "mean_bundle_seconds": float(np.mean([
                    row["bundle_seconds"] for row in group
                ])),
                "mean_speedup": float(np.mean([
                    row["oracle_seconds"] / max(row["bundle_seconds"], 1e-12)
                    for row in group
                ])),
            })
    payload = {
        "seeds": args.seeds,
        "path_risk_allocation_factor": args.path_fraction,
        "summary": summary,
        "instances": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
