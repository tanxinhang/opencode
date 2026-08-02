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
    drop_thresholds = (1e-8, 1e-6, 1e-5, 1e-4)
    drop_counts = {str(value): 0 for value in drop_thresholds}
    decreasing_conditions = []
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
                        for threshold in drop_thresholds:
                            drop_counts[str(threshold)] += int(drop > threshold)
                        left = sorted({model.owner} | {candidates[k] for k in range(len(candidates)) if mask & (1 << k)})
                        right = sorted(set(left) | {candidates[j]})
                        for received in (left, right):
                            idx = np.asarray(received, dtype=int)
                            cov0 = model.sigma0[np.ix_(idx, idx)]
                            cov1 = model.sigma1[np.ix_(idx, idx)]
                            decreasing_conditions.append({
                                "drop": drop,
                                "condition_sigma0": float(np.linalg.cond(cov0)),
                                "condition_sigma1": float(np.linalg.cond(cov1)),
                                "minimum_eigenvalue_sigma0": float(np.linalg.eigvalsh(cov0).min()),
                            })
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
            def_pairs = {(q, i) for q, group in enumerate(deflection.portfolio.selection.scheduled)
                         for i in group if i != models[q].owner}
            pd_pairs = {(q, i) for q, group in enumerate(pd_result.portfolio.selection.scheduled)
                        for i in group if i != models[q].owner}
            union = def_pairs | pd_pairs
            global_jaccard = len(def_pairs & pd_pairs) / len(union) if union else 1.0
            target_jaccards = []
            target_differences = []
            for q in range(cfg.num_targets):
                left = set(deflection.portfolio.selection.scheduled[q]) - {models[q].owner}
                right = set(pd_result.portfolio.selection.scheduled[q]) - {models[q].owner}
                target_union = left | right
                target_jaccards.append(len(left & right) / len(target_union) if target_union else 1.0)
                target_differences.append(len(left ^ right))
            rows.append({
                "seed": seed, "budget_bits": budget,
                "exact_schedule_match": deflection.portfolio.selection.scheduled == pd_result.portfolio.selection.scheduled,
                "global_pair_jaccard": global_jaccard,
                "mean_target_jaccard": float(np.mean(target_jaccards)),
                "mean_target_different_reports": float(np.mean(target_differences)),
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
            "exact_schedule_match_rate": float(np.mean([x["exact_schedule_match"] for x in group])),
            "mean_global_pair_jaccard": float(np.mean([x["global_pair_jaccard"] for x in group])),
            "mean_target_jaccard": float(np.mean([x["mean_target_jaccard"] for x in group])),
            "mean_target_different_reports": float(np.mean([x["mean_target_different_reports"] for x in group])),
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
            "drop_counts_by_threshold": drop_counts,
            "maximum_sigma0_condition_on_decreasing_edges": max(
                (x["condition_sigma0"] for x in decreasing_conditions), default=0.0),
            "maximum_sigma1_condition_on_decreasing_edges": max(
                (x["condition_sigma1"] for x in decreasing_conditions), default=0.0),
            "minimum_sigma0_eigenvalue_on_decreasing_edges": min(
                (x["minimum_eigenvalue_sigma0"] for x in decreasing_conditions), default=0.0),
        },
        "summary": summary, "instances": rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"monotonicity": payload["monotonicity"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
