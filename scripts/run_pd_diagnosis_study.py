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
from uav_otfs_isac.risk import (
    diagnose_target_reachability,
    optimize_chance_constrained_portfolio,
    optimize_fair_chance_constrained_portfolio,
)
from uav_otfs_isac.scenario import build_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/pd_diagnosis_study.json")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    cfg = load_config(args.config); limits = np.full(cfg.num_targets, args.epsilon)
    rows = []; diagnosis_rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        models = build_models(cfg, np.random.default_rng(seed))
        for budget in args.budgets:
            weighted = optimize_chance_constrained_portfolio(
                models, budget, args.minimum_pd, cfg.qos_weights, limits,
                quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate)
            fair = optimize_fair_chance_constrained_portfolio(
                models, budget, args.minimum_pd, cfg.qos_weights, limits,
                quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate)
            weighted_relative = float(np.max(weighted.violation_excess_per_target / limits))
            rows.append({
                "seed": seed, "budget_bits": budget,
                "weighted_max_relative_excess": weighted_relative,
                "fair_max_relative_excess": fair.maximum_relative_violation_excess,
                "weighted_sum_excess": weighted.weighted_violation_excess,
                "fair_sum_excess": fair.chance.weighted_violation_excess,
                "weighted_violation": weighted.portfolio.violation_probability_per_target.tolist(),
                "fair_violation": fair.chance.portfolio.violation_probability_per_target.tolist(),
            })
            for q, model in enumerate(models):
                diagnosis = diagnose_target_reachability(
                    model, budget, args.minimum_pd[q], limits[q],
                    quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate)
                diagnosis_rows.append({
                    "seed": seed, "budget_bits": budget, "target": q,
                    "classification": diagnosis.classification,
                    "maximum_deterministic_pd": diagnosis.maximum_deterministic_quality,
                    "minimum_unlimited_violation_probability": diagnosis.minimum_unlimited_violation_probability,
                    "minimum_budgeted_violation_probability": diagnosis.minimum_budgeted_violation_probability,
                })
    summary = []
    for budget in args.budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        diagnosis_group = [row for row in diagnosis_rows if row["budget_bits"] == budget]
        counts = {name: 0 for name in ("sensing_limited", "reliability_limited", "budget_limited", "feasible")}
        for row in diagnosis_group:
            counts[row["classification"]] += 1
        summary.append({
            "budget_bits": budget,
            "mean_weighted_max_relative_excess": float(np.mean([x["weighted_max_relative_excess"] for x in group])),
            "mean_fair_max_relative_excess": float(np.mean([x["fair_max_relative_excess"] for x in group])),
            "mean_weighted_sum_excess": float(np.mean([x["weighted_sum_excess"] for x in group])),
            "mean_fair_sum_excess": float(np.mean([x["fair_sum_excess"] for x in group])),
            "diagnosis_fraction": {name: count / len(diagnosis_group) for name, count in counts.items()},
        })
    payload = {"minimum_pd": args.minimum_pd, "epsilon": args.epsilon,
               "summary": summary, "instances": rows, "diagnoses": diagnosis_rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
