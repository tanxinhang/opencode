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
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.risk import (
    gaussian_pd_loss_distribution,
    optimize_chance_constrained_portfolio,
)
from uav_otfs_isac.scenario import build_models


def quality_by_mask(model, alpha):
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    values = {}
    for mask in range(1 << len(candidates)):
        received = {model.owner}
        received.update(candidates[j] for j in range(len(candidates)) if mask & (1 << j))
        values[mask] = gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1, received, alpha
        )
    return candidates, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/pd_proxy_study.json")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if len(args.minimum_pd) != cfg.num_targets:
        raise ValueError("minimum-pd must have num_targets entries")
    monotonic_edges = 0; decreasing_edges = 0; maximum_drop = 0.0
    rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        models = build_models(cfg, np.random.default_rng(seed))
        for model in models:
            candidates, values = quality_by_mask(model, cfg.false_alarm_rate)
            for mask, quality in values.items():
                for j in range(len(candidates)):
                    if mask & (1 << j):
                        continue
                    monotonic_edges += 1
                    drop = quality - values[mask | (1 << j)]
                    if drop > 1e-12:
                        decreasing_edges += 1
                        maximum_drop = max(maximum_drop, drop)
        for budget in args.budgets:
            deflection = optimize_chance_constrained_portfolio(
                models, budget, cfg.qos_min_deflection, cfg.qos_weights,
                np.full(cfg.num_targets, args.epsilon), beta=0.9, tail_weight=1.0,
                quality_mode="deflection", false_alarm_rate=cfg.false_alarm_rate,
            )
            pd_result = optimize_chance_constrained_portfolio(
                models, budget, args.minimum_pd, cfg.qos_weights,
                np.full(cfg.num_targets, args.epsilon), beta=0.9, tail_weight=1.0,
                quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate,
            )
            def_pd_violation = np.asarray([
                gaussian_pd_loss_distribution(
                    model, deflection.portfolio.selection.scheduled[q],
                    args.minimum_pd[q], cfg.false_alarm_rate,
                ).violation_probability()
                for q, model in enumerate(models)
            ])
            pd_pd_violation = pd_result.portfolio.violation_probability_per_target
            rows.append({
                "seed": seed, "budget_bits": budget,
                "same_schedule": deflection.portfolio.selection.scheduled == pd_result.portfolio.selection.scheduled,
                "deflection_schedule_pd_violation": def_pd_violation.tolist(),
                "pd_schedule_pd_violation": pd_pd_violation.tolist(),
                "deflection_schedule_worst_pd_violation": float(def_pd_violation.max()),
                "pd_schedule_worst_pd_violation": float(pd_pd_violation.max()),
                "pd_chance_feasible": pd_result.feasible,
            })
    summary = []
    for budget in args.budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        summary.append({
            "budget_bits": budget,
            "schedule_match_rate": float(np.mean([x["same_schedule"] for x in group])),
            "pd_chance_feasible_rate": float(np.mean([x["pd_chance_feasible"] for x in group])),
            "mean_worst_pd_violation_deflection_proxy": float(np.mean([
                x["deflection_schedule_worst_pd_violation"] for x in group
            ])),
            "mean_worst_pd_violation_pd_objective": float(np.mean([
                x["pd_schedule_worst_pd_violation"] for x in group
            ])),
        })
    payload = {
        "minimum_pd": args.minimum_pd, "epsilon": args.epsilon,
        "monotonicity": {
            "tested_addition_edges": monotonic_edges,
            "decreasing_edges": decreasing_edges,
            "decreasing_edge_rate": decreasing_edges / max(monotonic_edges, 1),
            "maximum_pd_drop": maximum_drop,
        },
        "summary": summary, "instances": rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"monotonicity": payload["monotonicity"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
