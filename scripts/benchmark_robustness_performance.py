"""Reproducible performance benchmark for the robustness stack.

Smoke mode times the core operations used by the robustness gates.  Formal
mode additionally runs the stress suite and the robust-allocation script at
the submitted scale.  Results are written as JSON so future changes can be
compared on the same machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.channel_degradation import verify_bsc_roc_dominance
from uav_otfs_isac.config import load_config
from uav_otfs_isac.robust_portfolio import (
    enumerate_robust_target_portfolios,
    optimize_independent_robust_chance_constrained_portfolio,
    optimize_robust_chance_constrained_portfolio,
)
from uav_otfs_isac.robustness_stress import (
    StressProfile,
    build_stress_models,
    survival_envelope,
)
from uav_otfs_isac.physical_link_model import build_physical_link_models


def timed(label: str, fn) -> dict:
    start = time.perf_counter()
    fn()
    seconds = time.perf_counter() - start
    return {"label": label, "seconds": seconds}


def smoke(cfg) -> list[dict]:
    weights = np.asarray(cfg.qos_weights, dtype=float)
    clean = build_stress_models(cfg, cfg.seed, StressProfile("clean"))
    combined = build_stress_models(
        cfg,
        cfg.seed,
        StressProfile(
            "combined",
            interference_to_noise=10.0,
            bit_flip_probability=0.10,
            success_probability_scale=0.8,
            mobility_std=4.0,
        ),
    )
    groups = [[clean[q], combined[q]] for q in range(cfg.num_targets)]

    def enumerate_options():
        return enumerate_robust_target_portfolios(
            groups[0], 0.7, 1.0, 0.9, 1.0,
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )

    def common_dp():
        return optimize_robust_chance_constrained_portfolio(
            groups, 20, [0.7] * cfg.num_targets, weights,
            np.full(cfg.num_targets, 0.2),
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )

    def independent_dp():
        return optimize_independent_robust_chance_constrained_portfolio(
            groups, 20, [0.7] * cfg.num_targets, weights,
            np.full(cfg.num_targets, 0.2),
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )

    def envelope():
        return survival_envelope(
            cfg,
            [
                StressProfile("clean"),
                StressProfile("spatial", interference_to_noise=3.0),
                StressProfile("channel", bit_flip_probability=0.10),
                StressProfile("mobility", mobility_std=4.0),
            ],
            seeds=2,
            budget_bits=20,
            grid=64,
            qos_target=0.7,
        )

    def bsc_gate():
        return verify_bsc_roc_dominance(
            bits_options=(1, 2, 3),
            mu1_options=(1.0, 1.5, 2.0),
            lo_options=(0.0, 0.1, 0.2),
            hi_options=(0.3, 0.4, 0.45),
            false_alarm_grid=(0.01, 0.05, 0.1, 0.2),
        )

    def physical_links():
        return build_physical_link_models(
            cfg,
            cfg.seed,
            reference_snr_db=20.0,
            threshold_db=5.0,
            shadowing_db=3.0,
        )

    return [
        timed("enumerate_robust_options_S2_R7", enumerate_options),
        timed("common_dp_S2_B20", common_dp),
        timed("independent_dp_S2_B20", independent_dp),
        timed("survival_envelope_4x2x1_g64", envelope),
        timed("bsc_roc_gate_default", bsc_gate),
        timed("build_physical_link_models", physical_links),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robustness_performance_benchmark.json")
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = smoke(cfg)
    payload = {
        "gate": "robustness-performance-benchmark",
        "config": args.config,
        "smoke": rows,
        "smoke_max_seconds": max(row["seconds"] for row in rows),
        "formal": None,
    }
    if args.formal:
        formal = {}
        for name, command in [
            (
                "stress_suite",
                [
                    sys.executable,
                    "scripts/run_robustness_stress_suite.py",
                    "--seeds", "5", "--grid", "64",
                    "--budgets", "20", "30", "40",
                    "--output", "results/formal_stress_suite_benchmark.json",
                ],
            ),
            (
                "stress_allocation",
                [
                    sys.executable,
                    "scripts/run_robust_stress_allocation.py",
                    "--seeds", "5", "--budgets", "16", "20", "24",
                    "--output", "results/formal_stress_allocation_benchmark.json",
                ],
            ),
        ]:
            start = time.perf_counter()
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            formal[name] = time.perf_counter() - start
        payload["formal"] = formal
        payload["formal_max_seconds"] = max(formal.values())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "smoke_max_seconds": payload["smoke_max_seconds"],
        "smoke": [
            {"label": row["label"], "seconds": row["seconds"]}
            for row in rows
        ],
        "formal_max_seconds": payload.get("formal_max_seconds"),
    }, indent=2))


if __name__ == "__main__":
    main()
