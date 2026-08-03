from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.adaptive import (
    gaussian_pd_threshold_by_mask,
    optimize_single_target_two_stage_oracle,
    two_stage_hidden_state_model,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.reliability import physical_failure_groups
from uav_otfs_isac.scenario import build_models, uav_geometry


def main() -> None:
    cfg = load_config("config/demo.yaml")
    positions = uav_geometry(cfg.num_uavs)
    rows = []
    for offset in range(3):
        seed = cfg.seed + offset
        models = build_models(cfg, np.random.default_rng(seed))
        paths = physical_failure_groups(
            models, positions, "owner_angle_path", 2
        )
        q = 2
        model = models[q]
        reporters, _, prior, success = two_stage_hidden_state_model(
            model, paths[q], 0.5, 0.5
        )
        threshold = gaussian_pd_threshold_by_mask(
            model, reporters, 0.70, cfg.false_alarm_rate
        )
        for first_budget in (10, 15):
            start = perf_counter()
            result = optimize_single_target_two_stage_oracle(
                prior, success,
                [int(model.report_bits[i]) for i in reporters],
                np.ones((len(reporters), 2), dtype=bool), threshold,
                total_budget_bits=20,
                first_stage_budget_bits=first_budget,
                domain_capacities=[10, 10],
            )
            rows.append({
                "seed": seed,
                "target": q,
                "first_stage_fraction": first_budget / 20,
                "seconds": perf_counter() - start,
                **{
                    f"{name}_{metric}": float(getattr(value, metric))
                    for name, value in result.items()
                    for metric in ("success_probability", "expected_bits")
                },
            })
    summary = []
    for fraction in (0.5, 0.75):
        group = [row for row in rows if row["first_stage_fraction"] == fraction]
        summary.append({
            "first_stage_fraction": fraction,
            "mean_static_success": float(np.mean([
                row["static_success_probability"] for row in group
            ])),
            "mean_adaptive_success": float(np.mean([
                row["adaptive_success_probability"] for row in group
            ])),
            "mean_clairvoyant_success": float(np.mean([
                row["clairvoyant_success_probability"] for row in group
            ])),
            "mean_adaptive_expected_bits": float(np.mean([
                row["adaptive_expected_bits"] for row in group
            ])),
            "mean_relative_violation_reduction": float(np.mean([
                ((1.0 - row["static_success_probability"])
                 - (1.0 - row["adaptive_success_probability"]))
                / max(1.0 - row["static_success_probability"], 1e-12)
                for row in group
            ])),
            "mean_seconds": float(np.mean([row["seconds"] for row in group])),
        })
    payload = {"seeds": 3, "target": 2, "summary": summary, "instances": rows}
    output = Path("results/real_target_adaptive_smoke.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
