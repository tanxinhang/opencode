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

from uav_otfs_isac.ablations import (
    deterministic_link_models,
    diagonal_covariance_models,
    evaluate_schedule_on_truth,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select


def _regimes(cfg):
    return {
        "default": cfg,
        "high_correlation": replace(
            cfg,
            covariance_shrinkage=0.03,
            otfs=replace(cfg.otfs, common_factor_strength=0.75),
        ),
        "unreliable_links": replace(
            cfg,
            reporting=replace(
                cfg.reporting,
                success_probability_range=(0.35, 0.90),
                bit_flip_probability_range=(0.01, 0.15),
            ),
        ),
        "strict_qos": replace(cfg, qos_min_deflection=(15.0, 10.0, 5.0)),
        "combined_stress": replace(
            cfg,
            qos_min_deflection=(15.0, 10.0, 5.0),
            covariance_shrinkage=0.03,
            otfs=replace(cfg.otfs, common_factor_strength=0.75),
            reporting=replace(
                cfg.reporting,
                success_probability_range=(0.35, 0.90),
                bit_flip_probability_range=(0.01, 0.15),
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/sensitivity_study.json")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30, 40])
    args = parser.parse_args()
    base = load_config(args.config)
    methods = ("full", "no_correlation", "deterministic_links", "first_order", "single_stage")
    rows = []
    for regime_name, cfg in _regimes(base).items():
        for offset in range(args.seeds):
            seed = cfg.seed + offset
            truth = build_models(cfg, np.random.default_rng(seed))
            assumed = {
                "full": truth,
                "no_correlation": diagonal_covariance_models(truth),
                "deterministic_links": deterministic_link_models(truth),
                "first_order": truth,
                "single_stage": truth,
            }
            for budget in args.budgets:
                selections = {}
                for method in methods:
                    selected = greedy_select(
                        assumed[method], budget, cfg.qos_min_deflection, cfg.qos_weights,
                        cfg.performance_weights, mode=cfg.expected_mode,
                        max_exact_reports=cfg.max_exact_reports,
                        rng=np.random.default_rng(seed + 10000),
                        gain_mode="first_order" if method == "first_order" else "exact",
                        qos_first=method != "single_stage",
                    )
                    selections[method] = evaluate_schedule_on_truth(
                        truth, selected, cfg.qos_min_deflection, cfg.qos_weights,
                        mode=cfg.expected_mode, max_exact_reports=cfg.max_exact_reports,
                        rng=np.random.default_rng(seed + 20000),
                    )
                full = selections["full"]
                full_utility = float(np.asarray(cfg.performance_weights) @ full.expected_deflection)
                for method, result in selections.items():
                    utility = float(np.asarray(cfg.performance_weights) @ result.expected_deflection)
                    gap_tolerance = 1e-10
                    full_lexicographically_better = bool(
                        full.normalized_qos_gap < result.normalized_qos_gap - gap_tolerance
                        or (
                            abs(full.normalized_qos_gap - result.normalized_qos_gap) <= gap_tolerance
                            and full_utility >= utility - gap_tolerance
                        )
                    )
                    rows.append({
                        "regime": regime_name,
                        "seed": seed,
                        "budget_bits": budget,
                        "method": method,
                        "truth_utility": utility,
                        "utility_loss_vs_full": full_utility - utility,
                        "qos_gap": result.normalized_qos_gap,
                        "qos_gap_increase_vs_full": result.normalized_qos_gap - full.normalized_qos_gap,
                        "full_lexicographically_better_or_equal": full_lexicographically_better,
                        "schedule_changed": result.scheduled != full.scheduled,
                    })
    summary = []
    for regime in _regimes(base):
        for budget in args.budgets:
            for method in methods:
                group = [r for r in rows if r["regime"] == regime and
                         r["budget_bits"] == budget and r["method"] == method]
                summary.append({
                    "regime": regime,
                    "budget_bits": budget,
                    "method": method,
                    "mean_truth_utility": float(np.mean([r["truth_utility"] for r in group])),
                    "mean_utility_loss_vs_full": float(np.mean([r["utility_loss_vs_full"] for r in group])),
                    "mean_qos_gap": float(np.mean([r["qos_gap"] for r in group])),
                    "mean_qos_gap_increase_vs_full": float(np.mean([r["qos_gap_increase_vs_full"] for r in group])),
                    "full_lexicographic_win_or_tie_rate": float(np.mean([
                        r["full_lexicographically_better_or_equal"] for r in group
                    ])),
                    "schedule_change_rate": float(np.mean([r["schedule_changed"] for r in group])),
                })
    payload = {"seeds_per_cell": args.seeds, "summary": summary, "instances": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    compact = [row for row in summary if row["method"] != "full" and (
        row["mean_utility_loss_vs_full"] > 0.05 or
        row["mean_qos_gap_increase_vs_full"] > 0.01
    )]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
