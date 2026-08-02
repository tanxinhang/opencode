from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.baselines import (
    all_scheduled,
    communication_score,
    independent_post_report_score,
    no_cooperation,
    random_score_factory,
    ranked_baseline,
    sensing_quality_score,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select
from uav_otfs_isac.simulation import evaluate_detection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/benchmark_summary.json")
    args = parser.parse_args()
    cfg = load_config(args.config); rng = np.random.default_rng(cfg.seed)
    models = build_models(cfg, rng)
    common = dict(mode=cfg.expected_mode, max_exact_reports=cfg.max_exact_reports, rng=rng)
    results = {
        "proposed": greedy_select(models, cfg.report_budget_bits, cfg.qos_min_deflection, cfg.qos_weights, cfg.performance_weights, **common),
        "no_cooperation": no_cooperation(models, cfg.qos_min_deflection, cfg.qos_weights, **common),
        "sensing_top": ranked_baseline(models, cfg.report_budget_bits, cfg.qos_min_deflection, cfg.qos_weights, sensing_quality_score, **common),
        "communication_top": ranked_baseline(models, cfg.report_budget_bits, cfg.qos_min_deflection, cfg.qos_weights, communication_score, **common),
        "independent_post": ranked_baseline(models, cfg.report_budget_bits, cfg.qos_min_deflection, cfg.qos_weights, independent_post_report_score, **common),
        "random": ranked_baseline(models, cfg.report_budget_bits, cfg.qos_min_deflection, cfg.qos_weights, random_score_factory(rng), **common),
        "all_scheduled": all_scheduled(models, cfg.qos_min_deflection, cfg.qos_weights, **common),
    }
    summary = {}
    for name, selection in results.items():
        # Common random numbers remove artificial metric differences when two
        # methods happen to select the same reporting sets.
        evaluation_rng = np.random.default_rng(cfg.seed + 1)
        metrics = evaluate_detection(
            models, selection, cfg.false_alarm_rate, cfg.monte_carlo_trials, evaluation_rng
        )
        summary[name] = {
            "scheduled": [sorted(group) for group in selection.scheduled],
            "expected_deflection": selection.expected_deflection.tolist(),
            "weighted_expected_deflection": float(
                np.asarray(cfg.performance_weights) @ selection.expected_deflection
            ),
            "qos_gap": selection.normalized_qos_gap,
            "used_bits": selection.used_bits,
            "selected_reports": metrics.selected_reports,
            "mean_pd": metrics.mean_pd,
            "worst_pd": metrics.worst_pd,
            "pd_per_target": metrics.pd_per_target.tolist(),
            "mean_pfa": float(metrics.pfa_per_target.mean()),
        }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
