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
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select
from uav_otfs_isac.simulation import evaluate_detection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/demo_summary.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)
    models = build_models(cfg, rng)
    selection = greedy_select(
        models,
        cfg.report_budget_bits,
        cfg.qos_min_deflection,
        cfg.qos_weights,
        cfg.performance_weights,
        mode=cfg.expected_mode,
        max_exact_reports=cfg.max_exact_reports,
        rng=rng,
    )
    metrics = evaluate_detection(
        models, selection, cfg.false_alarm_rate, cfg.monte_carlo_trials, rng
    )
    summary = {
        "scheduled": [sorted(x) for x in selection.scheduled],
        "expected_deflection": selection.expected_deflection.tolist(),
        "normalized_qos_gap": selection.normalized_qos_gap,
        "used_bits": selection.used_bits,
        "selected_reports": metrics.selected_reports,
        "pd_per_target": metrics.pd_per_target.tolist(),
        "pfa_per_target": metrics.pfa_per_target.tolist(),
        "mean_pd": metrics.mean_pd,
        "worst_pd": metrics.worst_pd,
        "trace": list(selection.trace),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
