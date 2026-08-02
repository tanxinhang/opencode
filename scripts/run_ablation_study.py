from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/ablation_study.json")
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    names = ("full", "no_correlation", "deterministic_links", "first_order", "single_stage")
    rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        truth = build_models(cfg, np.random.default_rng(seed))
        selector_models = {
            "full": truth,
            "no_correlation": diagonal_covariance_models(truth),
            "deterministic_links": deterministic_link_models(truth),
            "first_order": truth,
            "single_stage": truth,
        }
        selections = {}
        for name in names:
            selection = greedy_select(
                selector_models[name],
                cfg.report_budget_bits,
                cfg.qos_min_deflection,
                cfg.qos_weights,
                cfg.performance_weights,
                mode=cfg.expected_mode,
                max_exact_reports=cfg.max_exact_reports,
                rng=np.random.default_rng(seed + 10000),
                gain_mode="first_order" if name == "first_order" else "exact",
                qos_first=name != "single_stage",
            )
            selections[name] = evaluate_schedule_on_truth(
                truth,
                selection,
                cfg.qos_min_deflection,
                cfg.qos_weights,
                mode=cfg.expected_mode,
                max_exact_reports=cfg.max_exact_reports,
                rng=np.random.default_rng(seed + 20000),
            )
        full = selections["full"]
        full_utility = float(np.asarray(cfg.performance_weights) @ full.expected_deflection)
        for name, result in selections.items():
            utility = float(np.asarray(cfg.performance_weights) @ result.expected_deflection)
            rows.append({
                "seed": seed,
                "method": name,
                "utility": utility,
                "utility_loss_vs_full": full_utility - utility,
                "qos_gap": result.normalized_qos_gap,
                "qos_violated": result.normalized_qos_gap > 1e-10,
                "schedule_changed": result.scheduled != full.scheduled,
                "scheduled": [sorted(x) for x in result.scheduled],
            })
    summary = {}
    for name in names:
        group = [row for row in rows if row["method"] == name]
        summary[name] = {
            "mean_truth_utility": float(np.mean([row["utility"] for row in group])),
            "mean_utility_loss_vs_full": float(np.mean([row["utility_loss_vs_full"] for row in group])),
            "worst_utility_loss_vs_full": float(np.max([row["utility_loss_vs_full"] for row in group])),
            "qos_violation_rate": float(np.mean([row["qos_violated"] for row in group])),
            "schedule_change_rate": float(np.mean([row["schedule_changed"] for row in group])),
        }
    payload = {"seeds": args.seeds, "summary": summary, "instances": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
