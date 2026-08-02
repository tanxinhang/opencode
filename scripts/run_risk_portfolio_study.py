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
from uav_otfs_isac.risk import evaluate_portfolio_schedule, optimize_risk_portfolio
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select


def regimes(cfg):
    strict = (15.0, 10.0, 5.0)
    return {
        "default": cfg,
        "unreliable_links": replace(
            cfg,
            qos_min_deflection=strict,
            reporting=replace(
                cfg.reporting,
                success_probability_range=(0.35, 0.90),
                bit_flip_probability_range=(0.01, 0.15),
            ),
        ),
        "strict_qos": replace(cfg, qos_min_deflection=strict),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/risk_portfolio_study.json")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    args = parser.parse_args()
    base = load_config(args.config)
    rows = []
    for regime_name, cfg in regimes(base).items():
        for offset in range(args.seeds):
            seed = cfg.seed + offset
            models = build_models(cfg, np.random.default_rng(seed))
            for budget in args.budgets:
                neutral = optimize_risk_portfolio(
                    models, budget, cfg.qos_min_deflection, cfg.qos_weights,
                    beta=args.beta, tail_weight=0.0,
                )
                sensitive = optimize_risk_portfolio(
                    models, budget, cfg.qos_min_deflection, cfg.qos_weights,
                    beta=args.beta, tail_weight=args.tail_weight,
                )
                greedy = greedy_select(
                    models, budget, cfg.qos_min_deflection, cfg.qos_weights,
                    cfg.performance_weights, mode="exact",
                    max_exact_reports=cfg.max_exact_reports,
                )
                schedules = {
                    "risk_neutral": neutral.selection,
                    "mean_cvar": sensitive.selection,
                    "marginal_greedy": greedy,
                }
                for method, selection in schedules.items():
                    metrics = evaluate_portfolio_schedule(
                        models, selection.scheduled, cfg.qos_min_deflection,
                        cfg.qos_weights, beta=args.beta,
                        tail_weight=args.tail_weight,
                    )
                    rows.append({
                        "regime": regime_name,
                        "seed": seed,
                        "budget_bits": budget,
                        "method": method,
                        "used_bits": selection.used_bits,
                        "weighted_mean_loss": metrics["weighted_mean_loss"],
                        "weighted_cvar_loss": metrics["weighted_cvar_loss"],
                        "risk_objective": metrics["risk_objective"],
                        "mean_violation_probability": float(np.mean(
                            metrics["violation_probability_per_target"]
                        )),
                        "worst_target_violation_probability": float(np.max(
                            metrics["violation_probability_per_target"]
                        )),
                        "scheduled": [sorted(group) for group in selection.scheduled],
                    })
    summary = []
    for regime_name in regimes(base):
        for budget in args.budgets:
            for method in ("risk_neutral", "mean_cvar", "marginal_greedy"):
                group = [row for row in rows if row["regime"] == regime_name
                         and row["budget_bits"] == budget and row["method"] == method]
                summary.append({
                    "regime": regime_name,
                    "budget_bits": budget,
                    "method": method,
                    "mean_used_bits": float(np.mean([x["used_bits"] for x in group])),
                    "mean_weighted_loss": float(np.mean([x["weighted_mean_loss"] for x in group])),
                    "mean_weighted_cvar": float(np.mean([x["weighted_cvar_loss"] for x in group])),
                    "mean_risk_objective": float(np.mean([x["risk_objective"] for x in group])),
                    "mean_qos_violation_probability": float(np.mean([
                        x["mean_violation_probability"] for x in group
                    ])),
                    "mean_worst_target_violation_probability": float(np.mean([
                        x["worst_target_violation_probability"] for x in group
                    ])),
                })
    payload = {
        "beta": args.beta,
        "tail_weight": args.tail_weight,
        "seeds_per_cell": args.seeds,
        "summary": summary,
        "instances": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
