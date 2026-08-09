"""Stage-1 robustness stress suite.

Currently sweeps interference, BSC flip probability, link success scaling,
and target mobility.  Later stages should add correlated failure groups,
null-steering, and model-parameter uncertainty without changing the envelope
format.
"""

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
from uav_otfs_isac.robustness_stress import (
    StressProfile,
    build_stress_models,
    evaluate_stress_profile,
)


def default_profiles() -> list[StressProfile]:
    return [
        StressProfile("clean"),
        StressProfile(
            "spatial_interference_low",
            interference_sources=((60.0, -20.0, 0.0),),
            interference_reference_inr=(0.01,),
        ),
        StressProfile(
            "spatial_interference_high",
            interference_sources=((60.0, -20.0, 0.0),),
            interference_reference_inr=(0.1,),
        ),
        StressProfile("channel_mild", bit_flip_probability=0.05),
        StressProfile("channel_hard", bit_flip_probability=0.15),
        StressProfile(
            "link_unreliable",
            bit_flip_probability=0.10,
            success_probability_scale=0.75,
        ),
        StressProfile("mobility", mobility_std=4.0),
        StressProfile(
            "combined",
            interference_sources=((60.0, -20.0, 0.0),),
            interference_reference_inr=(0.01,),
            bit_flip_probability=0.10,
            success_probability_scale=0.8,
            mobility_std=4.0,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robustness_stress_suite.json")
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--qos-target", type=float, default=0.7)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profiles = default_profiles()
    model_cache = {}

    def cached_models(profile: StressProfile, seed: int):
        key = (profile.label, seed)
        if key not in model_cache:
            model_cache[key] = build_stress_models(cfg, seed, profile)
        return model_cache[key]

    summaries = []
    for budget in args.budgets:
        for profile in profiles:
            cells = []
            for offset in range(args.seeds):
                seed = cfg.seed + offset
                cells.append(evaluate_stress_profile(
                    cfg,
                    seed,
                    profile,
                    budget_bits=budget,
                    grid=args.grid,
                    models=cached_models(profile, seed),
                ))
            worst = np.asarray([cell["worst_pd"] for cell in cells])
            row = {
                "label": profile.label,
                "worst_pd_mean": float(np.mean(worst)),
                "worst_pd_min": float(np.min(worst)),
                "qos_rate": float(np.mean(
                    worst >= args.qos_target - 1e-9
                )),
            }
            summaries.append({
                "budget_bits": budget,
                **row,
            })

    payload = {
        "gate": "robustness-stress-suite",
        "stage": 1,
        "config": args.config,
        "seeds_per_cell": args.seeds,
        "grid": args.grid,
        "qos_target": args.qos_target,
        "axes": [
            "interference_to_noise",
            "interference_sources",
            "interference_reference_inr",
            "bit_flip_probability",
            "success_probability_scale",
            "mobility_std",
            "max_displacement_std",
        ],
        "summary": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
