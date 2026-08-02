from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.reliability import (
    alternating_failure_groups,
    with_grouped_common_state_erasures,
)
from uav_otfs_isac.risk import (
    attribute_failure_diversity_headroom,
    optimize_chance_constrained_portfolio,
)
from uav_otfs_isac.scenario import build_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/real_network_headroom_study.json")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--num-groups", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if len(args.minimum_pd) != cfg.num_targets:
        raise ValueError("minimum-pd must provide one threshold per target")
    limits = np.full(cfg.num_targets, args.epsilon)
    rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        independent = build_models(cfg, np.random.default_rng(seed))
        groups = alternating_failure_groups(independent, args.num_groups)
        truth = with_grouped_common_state_erasures(
            independent, args.strength, groups
        )
        for budget in args.budgets:
            independent_solution = optimize_chance_constrained_portfolio(
                independent, budget, args.minimum_pd, cfg.qos_weights, limits,
                quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate,
            )
            aware_solution = optimize_chance_constrained_portfolio(
                truth, budget, args.minimum_pd, cfg.qos_weights, limits,
                quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate,
            )
            independent_sets = independent_solution.portfolio.selection.scheduled
            aware_sets = aware_solution.portfolio.selection.scheduled
            for q, model in enumerate(truth):
                attribution = attribute_failure_diversity_headroom(
                    independent[q], model, independent_sets[q], aware_sets[q],
                    budget, args.minimum_pd[q], groups[q],
                    false_alarm_rate=cfg.false_alarm_rate,
                )
                rows.append({
                    "seed": seed,
                    "budget_bits": budget,
                    "target": q,
                    "target_weight": float(cfg.qos_weights[q]),
                    "minimum_successful_reports": attribution.minimum_successful_reports,
                    "supporting_failure_domains": attribution.supporting_failure_domains,
                    "classification": attribution.classification,
                    "independent_violation": attribution.independent_violation,
                    "aware_violation": attribution.aware_violation,
                    "oracle_violation": attribution.oracle_violation,
                    "all_scheduled_violation": attribution.all_scheduled_violation,
                    "oracle_all_scheduled_gap": attribution.oracle_all_scheduled_gap,
                    "near_all_scheduled_boundary": attribution.near_all_scheduled_boundary,
                    "recoverable_headroom": attribution.recoverable_headroom,
                    "headroom_use_ratio": attribution.headroom_use_ratio,
                    "independent_scheduled": sorted(independent_sets[q]),
                    "aware_scheduled": sorted(aware_sets[q]),
                    "oracle_scheduled": sorted(attribution.oracle_scheduled),
                })

    summary = []
    for budget in args.budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        counts = Counter(row["classification"] for row in group)
        positive = [row for row in group if row["recoverable_headroom"] > 1e-12]
        ratios = [
            row["headroom_use_ratio"] for row in positive
            if row["headroom_use_ratio"] is not None
        ]
        by_classification = {}
        total_headroom = sum(row["recoverable_headroom"] for row in group)
        weighted_headroom = sum(
            row["target_weight"] * row["recoverable_headroom"] for row in group
        )
        weighted_net_gain = sum(
            row["target_weight"] * (
                row["independent_violation"] - row["aware_violation"]
            ) for row in group
        )
        weighted_positive_gain = sum(
            row["target_weight"] * max(
                row["independent_violation"] - row["aware_violation"], 0.0
            ) for row in group
        )
        for name in sorted(counts):
            classified = [row for row in group if row["classification"] == name]
            classified_positive = [
                row for row in classified if row["recoverable_headroom"] > 1e-12
            ]
            by_classification[name] = {
                "targets": len(classified),
                "positive_headroom_fraction": (
                    len(classified_positive) / len(classified)
                ),
                "mean_recoverable_headroom": float(np.mean([
                    row["recoverable_headroom"] for row in classified
                ])),
                "total_headroom_share": (
                    sum(row["recoverable_headroom"] for row in classified)
                    / total_headroom if total_headroom > 1e-12 else None
                ),
                "mean_headroom_use_ratio_when_defined": (
                    float(np.mean([
                        row["headroom_use_ratio"] for row in classified_positive
                        if row["headroom_use_ratio"] is not None
                    ])) if classified_positive else None
                ),
            }
        summary.append({
            "budget_bits": budget,
            "targets": len(group),
            "classification_fraction": {
                name: count / len(group) for name, count in sorted(counts.items())
            },
            "headroom_by_classification": by_classification,
            "near_all_scheduled_boundary_fraction": float(np.mean([
                row["near_all_scheduled_boundary"] for row in group
            ])),
            "positive_headroom_fraction": len(positive) / len(group),
            "mean_recoverable_headroom": float(np.mean([
                row["recoverable_headroom"] for row in group
            ])),
            "mean_positive_headroom": (
                float(np.mean([row["recoverable_headroom"] for row in positive]))
                if positive else None
            ),
            "mean_headroom_use_ratio_when_defined": (
                float(np.mean(ratios)) if ratios else None
            ),
            "median_headroom_use_ratio_when_defined": (
                float(np.median(ratios)) if ratios else None
            ),
            "aggregate_headroom_capture_rate": (
                weighted_net_gain / weighted_headroom
                if weighted_headroom > 1e-12 else None
            ),
            "positive_gain_capture_rate": (
                weighted_positive_gain / weighted_headroom
                if weighted_headroom > 1e-12 else None
            ),
            "fraction_aware_worse_than_independent": float(np.mean([
                row["aware_violation"] > row["independent_violation"] + 1e-12
                for row in group
            ])),
            "mean_independent_violation": float(np.mean([
                row["independent_violation"] for row in group
            ])),
            "mean_aware_violation": float(np.mean([
                row["aware_violation"] for row in group
            ])),
            "mean_oracle_violation": float(np.mean([
                row["oracle_violation"] for row in group
            ])),
        })
    payload = {
        "seeds": args.seeds,
        "strength": args.strength,
        "num_groups": args.num_groups,
        "minimum_pd": args.minimum_pd,
        "epsilon": args.epsilon,
        "summary": summary,
        "instances": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
