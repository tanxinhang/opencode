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
from uav_otfs_isac.reliability import physical_failure_groups
from uav_otfs_isac.replication import optimize_dual_layer_chance_portfolio
from uav_otfs_isac.scenario import build_models, uav_geometry
from scripts.run_replication_realism_study import bootstrap_ci, native_resource_labels


def nested_access_masks(models, native_resources, scores, dual_fraction):
    masks = []
    for q, model in enumerate(models):
        mask = np.zeros((model.num_uavs, 2), dtype=bool)
        reporters = [i for i in range(model.num_uavs) if i != model.owner]
        order = sorted(reporters, key=lambda i: (scores[q][i], i))
        dual_count = int(np.floor(dual_fraction * len(reporters) + 1e-12))
        dual = set(order[:dual_count])
        for i in reporters:
            native = int(native_resources[q][i])
            mask[i, native] = True
            if i in dual:
                mask[i, 1 - native] = True
        masks.append(mask)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/replication_access_study.json")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--dual-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--path-fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    cfg = load_config(args.config); positions = uav_geometry(cfg.num_uavs)
    limits = np.full(cfg.num_targets, args.epsilon); rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        models = build_models(cfg, np.random.default_rng(seed))
        path_groups = physical_failure_groups(models, positions, "owner_angle_path", 2)
        resources = native_resource_labels(models)
        access_rng = np.random.default_rng(seed + 91_003)
        scores = [access_rng.random(model.num_uavs) for model in models]
        for budget in args.budgets:
            capacities = [budget // 2, budget - budget // 2]
            common = dict(
                models=models, budget_bits=budget, minimum_pd=args.minimum_pd,
                target_weights=cfg.qos_weights, violation_limits=limits,
                path_groups=path_groups, native_resources=resources,
                strength=args.strength, path_failure_fraction=args.path_fraction,
                domain_capacities=capacities, false_alarm_rate=cfg.false_alarm_rate,
            )
            selection = optimize_dual_layer_chance_portfolio(
                **common, replication_mode="cross_domain", maximum_copies=1
            )
            baseline_worst = float(np.max(selection.violation_probability_per_target))
            for dual_fraction in args.dual_fractions:
                access = nested_access_masks(models, resources, scores, dual_fraction)
                repair = optimize_dual_layer_chance_portfolio(
                    **common, replication_mode="cross_domain", maximum_copies=2,
                    resource_access=access,
                )
                worst = float(np.max(repair.violation_probability_per_target))
                accessible = sum(
                    int(mask[i].all()) for q, mask in enumerate(access)
                    for i in range(models[q].num_uavs) if i != models[q].owner
                )
                total = sum(model.num_uavs - 1 for model in models)
                rows.append({
                    "seed": seed, "budget_bits": budget,
                    "requested_dual_fraction": dual_fraction,
                    "realized_dual_fraction": accessible / total,
                    "selection_worst_violation": baseline_worst,
                    "repair_worst_violation": worst,
                    "worst_improvement": baseline_worst - worst,
                    "relative_reduction": (
                        (baseline_worst - worst) / baseline_worst if baseline_worst > 1e-12 else 0.0
                    ),
                    "selection_feasible": selection.feasible,
                    "repair_feasible": repair.feasible,
                    "feasibility_gain": int(repair.feasible) - int(selection.feasible),
                    "replicated_reports": int(sum(
                        max(count - 1, 0) for target in repair.copy_counts for count in target
                    )),
                })
    summary = []
    for budget in args.budgets:
        for fraction in args.dual_fractions:
            group = [row for row in rows if row["budget_bits"] == budget and row["requested_dual_fraction"] == fraction]
            improvement = [row["worst_improvement"] for row in group]
            relative = [row["relative_reduction"] for row in group]
            feasible = [row["feasibility_gain"] for row in group]
            summary.append({
                "budget_bits": budget, "requested_dual_fraction": fraction,
                "realized_dual_fraction": float(np.mean([row["realized_dual_fraction"] for row in group])),
                "mean_worst_improvement": float(np.mean(improvement)),
                "improvement_paired_bootstrap_ci95": bootstrap_ci(improvement),
                "mean_seed_relative_reduction": float(np.mean(relative)),
                "relative_reduction_paired_bootstrap_ci95": bootstrap_ci(relative),
                "selection_system_feasible_rate": float(np.mean([row["selection_feasible"] for row in group])),
                "repair_system_feasible_rate": float(np.mean([row["repair_feasible"] for row in group])),
                "feasibility_rate_gain": float(np.mean(feasible)),
                "feasibility_gain_paired_bootstrap_ci95": bootstrap_ci(feasible),
                "mean_replicated_reports": float(np.mean([row["replicated_reports"] for row in group])),
            })
    payload = {
        "seeds": args.seeds, "path_risk_allocation_factor": args.path_fraction,
        "strength": args.strength, "access_assignment": "nested seed-paired random masks independent of sensing and optimization",
        "summary": summary, "instances": rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
