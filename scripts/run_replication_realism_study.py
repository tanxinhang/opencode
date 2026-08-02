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


def bootstrap_ci(values, seed=92026, samples=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def native_resource_labels(models):
    labels = []
    for model in models:
        value = np.full(model.num_uavs, -1, dtype=int)
        reporters = [i for i in range(model.num_uavs) if i != model.owner]
        for order, reporter in enumerate(reporters):
            value[reporter] = order % 2
        labels.append(value)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/replication_realism_study.json")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--path-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
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
        for budget in args.budgets:
            capacities = [budget // 2, budget - budget // 2]
            for path_fraction in args.path_fractions:
                common = dict(
                    models=models, budget_bits=budget, minimum_pd=args.minimum_pd,
                    target_weights=cfg.qos_weights, violation_limits=limits,
                    path_groups=path_groups, native_resources=resources,
                    strength=args.strength, path_failure_fraction=path_fraction,
                    domain_capacities=capacities,
                    false_alarm_rate=cfg.false_alarm_rate,
                )
                methods = {
                    "selection_only": optimize_dual_layer_chance_portfolio(
                        **common, replication_mode="cross_domain", maximum_copies=1
                    ),
                    "same_domain_replication": optimize_dual_layer_chance_portfolio(
                        **common, replication_mode="same_domain", maximum_copies=2
                    ),
                    "cross_domain_replication": optimize_dual_layer_chance_portfolio(
                        **common, replication_mode="cross_domain", maximum_copies=2
                    ),
                }
                baseline = methods["selection_only"]
                for name, result in methods.items():
                    worst = float(np.max(result.violation_probability_per_target))
                    baseline_worst = float(np.max(baseline.violation_probability_per_target))
                    rows.append({
                        "seed": seed, "budget_bits": budget,
                        "path_failure_fraction": path_fraction, "method": name,
                        "worst_violation": worst,
                        "worst_improvement_vs_selection": baseline_worst - worst,
                        "relative_reduction_vs_selection": (
                            (baseline_worst - worst) / baseline_worst if baseline_worst > 1e-12 else 0.0
                        ),
                        "system_feasible": result.feasible,
                        "feasibility_gain_vs_selection": int(result.feasible) - int(baseline.feasible),
                        "used_bits": result.used_bits,
                        "replicated_reports": int(sum(
                            max(count - 1, 0) for target in result.copy_counts for count in target
                        )),
                    })
    summary = []
    for budget in args.budgets:
        for fraction in args.path_fractions:
            for method in ("selection_only", "same_domain_replication", "cross_domain_replication"):
                group = [row for row in rows if row["budget_bits"] == budget and row["path_failure_fraction"] == fraction and row["method"] == method]
                improvement = [row["worst_improvement_vs_selection"] for row in group]
                relative = [row["relative_reduction_vs_selection"] for row in group]
                feasibility = [row["feasibility_gain_vs_selection"] for row in group]
                mean_improvement = float(np.mean(improvement)); mean_relative = float(np.mean(relative)); feasibility_gain = float(np.mean(feasibility))
                summary.append({
                    "budget_bits": budget, "path_failure_fraction": fraction, "method": method,
                    "mean_worst_violation": float(np.mean([row["worst_violation"] for row in group])),
                    "mean_worst_improvement_vs_selection": mean_improvement,
                    "improvement_paired_bootstrap_ci95": bootstrap_ci(improvement),
                    "mean_seed_relative_reduction": mean_relative,
                    "relative_reduction_of_means": mean_improvement / float(np.mean([
                        next(x["worst_violation"] for x in rows if x["seed"] == row["seed"] and x["budget_bits"] == budget and x["path_failure_fraction"] == fraction and x["method"] == "selection_only") for row in group
                    ])),
                    "relative_reduction_paired_bootstrap_ci95": bootstrap_ci(relative),
                    "system_feasible_rate": float(np.mean([row["system_feasible"] for row in group])),
                    "feasibility_rate_gain": feasibility_gain,
                    "feasibility_gain_paired_bootstrap_ci95": bootstrap_ci(feasibility),
                    "mean_replicated_reports": float(np.mean([row["replicated_reports"] for row in group])),
                    "equal_bit_and_domain_capacity_respected": all(row["used_bits"] <= budget for row in group),
                    "realism_gate_passed": bool(method == "cross_domain_replication" and mean_improvement >= 0.04 and mean_relative >= 0.20 and feasibility_gain >= 0.10),
                })
    payload = {
        "seeds": args.seeds, "strength": args.strength,
        "path_failure_fractions": args.path_fractions,
        "resource_model": "two equal-capacity schedulable resource domains; all reporters dual-connectivity capable",
        "summary": summary, "instances": rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
