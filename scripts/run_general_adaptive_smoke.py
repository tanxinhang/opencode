from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.adaptive import optimize_single_target_two_stage_oracle


def at_least_two_reports(num_reports):
    return np.array([
        int(mask).bit_count() >= 2 for mask in range(1 << num_reports)
    ])


def main() -> None:
    rows = []
    total_budget = 6
    for seed in range(10):
        rng = np.random.default_rng(20260803 + seed)
        prior = rng.dirichlet(np.ones(4))
        base = rng.uniform(0.35, 0.8, size=(4, 2))
        success = np.stack([
            np.clip(base + rng.normal(0.0, 0.05, size=(4, 2)))
            for _ in range(3)
        ], axis=1)
        for first_budget in (3, 4, 5):
            result = optimize_single_target_two_stage_oracle(
                prior, success, [1, 1, 1], np.ones((3, 2), dtype=bool),
                at_least_two_reports(3), total_budget, first_budget, [3, 3],
            )
            rows.append({
                "seed": 20260803 + seed,
                "first_stage_fraction": first_budget / total_budget,
                **{
                    f"{name}_{metric}": float(getattr(value, metric))
                    for name, value in result.items()
                    for metric in ("success_probability", "expected_bits")
                },
            })
    summary = []
    for fraction in (0.5, 2 / 3, 5 / 6):
        group = [row for row in rows if np.isclose(
            row["first_stage_fraction"], fraction
        )]
        summary.append({
            "first_stage_fraction": fraction,
            "mean_selection_success": float(np.mean([
                row["selection_success_probability"] for row in group
            ])),
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
            "adaptive_beats_static_rate": float(np.mean([
                row["adaptive_success_probability"]
                > row["static_success_probability"] + 1e-12 for row in group
            ])),
        })
    payload = {"seeds": 10, "summary": summary, "instances": rows}
    output = Path("results/general_adaptive_smoke.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
