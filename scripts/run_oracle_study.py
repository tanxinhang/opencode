from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.oracle import exhaustive_oracle
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select


def _timed(call):
    start = perf_counter()
    result = call()
    return result, perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/oracle_small.yaml")
    parser.add_argument("--output", default="results/oracle_study.json")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 15, 20])
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        models = build_models(cfg, np.random.default_rng(seed))
        for budget in args.budgets:
            common = dict(
                models=models,
                budget_bits=budget,
                qos_min=cfg.qos_min_deflection,
                qos_weights=cfg.qos_weights,
                performance_weights=cfg.performance_weights,
                mode="exact",
                max_exact_reports=cfg.max_exact_reports,
            )
            greedy, greedy_seconds = _timed(lambda: greedy_select(**common))
            oracle, oracle_seconds = _timed(
                lambda: exhaustive_oracle(**common, max_candidates=12)
            )
            greedy_utility = float(
                np.asarray(cfg.performance_weights) @ greedy.expected_deflection
            )
            oracle_utility = float(
                np.asarray(cfg.performance_weights) @ oracle.expected_deflection
            )
            same_min_gap = bool(
                np.isclose(greedy.normalized_qos_gap, oracle.normalized_qos_gap, atol=1e-10)
            )
            utility_gap = (
                max(oracle_utility - greedy_utility, 0.0) / max(abs(oracle_utility), 1e-12)
                if same_min_gap else None
            )
            rows.append({
                "seed": seed,
                "budget_bits": budget,
                "greedy_qos_gap": greedy.normalized_qos_gap,
                "oracle_qos_gap": oracle.normalized_qos_gap,
                "same_minimum_qos_gap": same_min_gap,
                "greedy_utility": greedy_utility,
                "oracle_utility": oracle_utility,
                "relative_utility_gap": utility_gap,
                "greedy_seconds": greedy_seconds,
                "oracle_seconds": oracle_seconds,
                "greedy_scheduled": [sorted(x) for x in greedy.scheduled],
                "oracle_scheduled": [sorted(x) for x in oracle.scheduled],
            })
    comparable = [r["relative_utility_gap"] for r in rows if r["relative_utility_gap"] is not None]
    summary = {
        "instances": len(rows),
        "qos_optimal_fraction": float(np.mean([r["same_minimum_qos_gap"] for r in rows])),
        "exact_set_match_fraction": float(np.mean([
            r["greedy_scheduled"] == r["oracle_scheduled"] for r in rows
        ])),
        "mean_relative_utility_gap_when_qos_optimal": float(np.mean(comparable)),
        "max_relative_utility_gap_when_qos_optimal": float(np.max(comparable)),
        "mean_greedy_seconds": float(np.mean([r["greedy_seconds"] for r in rows])),
        "mean_oracle_seconds": float(np.mean([r["oracle_seconds"] for r in rows])),
    }
    payload = {"summary": summary, "instances": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
