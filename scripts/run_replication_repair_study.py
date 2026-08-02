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
from uav_otfs_isac.reliability import (
    physical_failure_groups,
    with_grouped_common_state_erasures,
)
from uav_otfs_isac.replication import optimize_replication_chance_portfolio
from uav_otfs_isac.risk import optimize_chance_constrained_portfolio
from uav_otfs_isac.scenario import build_models, uav_geometry


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, None, None
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/replication_repair_study.json")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--schemes", nargs="+",
        default=["owner_angle_path", "formation_position"],
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    limits = np.full(cfg.num_targets, args.epsilon)
    positions = uav_geometry(cfg.num_uavs)
    rows = []
    for scheme in args.schemes:
        for offset in range(args.seeds):
            seed = cfg.seed + offset
            independent = build_models(cfg, np.random.default_rng(seed))
            domains = physical_failure_groups(independent, positions, scheme, 2)
            truth = with_grouped_common_state_erasures(
                independent, args.strength, domains
            )
            for budget in args.budgets:
                baseline = optimize_chance_constrained_portfolio(
                    truth, budget, args.minimum_pd, cfg.qos_weights, limits,
                    quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate,
                )
                repair = optimize_replication_chance_portfolio(
                    independent, budget, args.minimum_pd, cfg.qos_weights, limits,
                    domains, args.strength, false_alarm_rate=cfg.false_alarm_rate,
                )
                baseline_violation = baseline.portfolio.violation_probability_per_target
                repair_violation = repair.violation_probability_per_target
                baseline_worst = float(np.max(baseline_violation))
                repair_worst = float(np.max(repair_violation))
                rows.append({
                    "scheme": scheme, "seed": seed, "budget_bits": budget,
                    "baseline_worst_violation": baseline_worst,
                    "repair_worst_violation": repair_worst,
                    "worst_violation_improvement": baseline_worst - repair_worst,
                    "relative_worst_violation_reduction": (
                        (baseline_worst - repair_worst) / baseline_worst
                        if baseline_worst > 1e-12 else 0.0
                    ),
                    "baseline_mean_violation": float(np.mean(baseline_violation)),
                    "repair_mean_violation": float(np.mean(repair_violation)),
                    "baseline_feasible": bool(baseline.feasible),
                    "repair_feasible": bool(repair.feasible),
                    "baseline_used_bits": baseline.portfolio.selection.used_bits,
                    "repair_used_bits": repair.used_bits,
                    "replicated_reports": int(sum(
                        max(count - 1, 0)
                        for target in repair.copy_counts for count in target
                    )),
                    "copy_counts": [list(target) for target in repair.copy_counts],
                })
    summary = []
    for scheme in args.schemes:
        for budget in args.budgets:
            group = [
                row for row in rows
                if row["scheme"] == scheme and row["budget_bits"] == budget
            ]
            improvement = mean_ci95([
                row["worst_violation_improvement"] for row in group
            ])
            relative = mean_ci95([
                row["relative_worst_violation_reduction"] for row in group
            ])
            baseline_feasible = float(np.mean([
                row["baseline_feasible"] for row in group
            ]))
            repair_feasible = float(np.mean([
                row["repair_feasible"] for row in group
            ]))
            summary.append({
                "scheme": scheme, "budget_bits": budget,
                "mean_baseline_worst_violation": float(np.mean([
                    row["baseline_worst_violation"] for row in group
                ])),
                "mean_repair_worst_violation": float(np.mean([
                    row["repair_worst_violation"] for row in group
                ])),
                "mean_worst_violation_improvement": improvement[0],
                "improvement_ci95": [improvement[1], improvement[2]],
                "mean_relative_worst_violation_reduction": relative[0],
                "relative_reduction_ci95": [relative[1], relative[2]],
                "baseline_system_feasible_rate": baseline_feasible,
                "repair_system_feasible_rate": repair_feasible,
                "system_feasible_rate_gain": repair_feasible - baseline_feasible,
                "mean_replicated_reports": float(np.mean([
                    row["replicated_reports"] for row in group
                ])),
                "equal_budget_respected": bool(all(
                    row["baseline_used_bits"] <= budget
                    and row["repair_used_bits"] <= budget for row in group
                )),
                "absolute_gate_passed": bool(improvement[0] >= 0.03 and improvement[1] is not None and improvement[1] > 0.0),
                "relative_gate_passed": bool(relative[0] >= 0.20 and relative[1] is not None and relative[1] > 0.0),
                "feasibility_gate_passed": bool(repair_feasible - baseline_feasible >= 0.05),
            })
    payload = {
        "seeds": args.seeds, "schemes": args.schemes,
        "strength": args.strength, "minimum_pd": args.minimum_pd,
        "epsilon": args.epsilon, "summary": summary, "instances": rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
