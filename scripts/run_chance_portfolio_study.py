from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.risk import optimize_chance_constrained_portfolio, optimize_risk_portfolio
from uav_otfs_isac.scenario import build_models


def regimes(cfg):
    strict = (15.0, 10.0, 5.0)
    return {
        "default": cfg,
        "strict_qos": replace(cfg, qos_min_deflection=strict),
        "unreliable_strict": replace(
            cfg, qos_min_deflection=strict,
            reporting=replace(cfg.reporting, success_probability_range=(0.35, 0.90),
                              bit_flip_probability_range=(0.01, 0.15)),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/chance_portfolio_study.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30, 40, 60, 80, 105])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    args = parser.parse_args()
    cfg0 = load_config(args.config); rows = []
    for regime_name, cfg in regimes(cfg0).items():
        limits = np.full(cfg.num_targets, args.epsilon)
        for offset in range(args.seeds):
            seed = cfg.seed + offset
            models = build_models(cfg, np.random.default_rng(seed))
            for budget in args.budgets:
                chance = optimize_chance_constrained_portfolio(
                    models, budget, cfg.qos_min_deflection, cfg.qos_weights, limits,
                    beta=args.beta, tail_weight=args.tail_weight)
                unconstrained = optimize_risk_portfolio(
                    models, budget, cfg.qos_min_deflection, cfg.qos_weights,
                    beta=args.beta, tail_weight=args.tail_weight)
                rows.append({
                    "regime": regime_name, "seed": seed, "budget_bits": budget,
                    "chance_feasible": chance.feasible,
                    "weighted_violation_excess": chance.weighted_violation_excess,
                    "chance_used_bits": chance.portfolio.selection.used_bits,
                    "chance_risk_objective": chance.portfolio.objective,
                    "chance_worst_violation_probability": float(np.max(chance.portfolio.violation_probability_per_target)),
                    "unconstrained_risk_objective": unconstrained.objective,
                    "unconstrained_worst_violation_probability": float(np.max(unconstrained.violation_probability_per_target)),
                })
    summary = []
    for regime_name in regimes(cfg0):
        for budget in args.budgets:
            group = [row for row in rows if row["regime"] == regime_name and row["budget_bits"] == budget]
            summary.append({
                "regime": regime_name, "budget_bits": budget,
                "feasible_rate": float(np.mean([row["chance_feasible"] for row in group])),
                "mean_weighted_violation_excess": float(np.mean([row["weighted_violation_excess"] for row in group])),
                "mean_chance_used_bits": float(np.mean([row["chance_used_bits"] for row in group])),
                "mean_chance_worst_violation_probability": float(np.mean([row["chance_worst_violation_probability"] for row in group])),
                "mean_unconstrained_worst_violation_probability": float(np.mean([row["unconstrained_worst_violation_probability"] for row in group])),
                "mean_risk_cost_of_reliability": float(np.mean([row["chance_risk_objective"] - row["unconstrained_risk_objective"] for row in group])),
            })
    payload = {"epsilon": args.epsilon, "beta": args.beta, "tail_weight": args.tail_weight,
               "seeds_per_cell": args.seeds, "summary": summary, "instances": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
