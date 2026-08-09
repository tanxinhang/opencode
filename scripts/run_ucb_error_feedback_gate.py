"""UCB WTA error-feedback gate with finite stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.error_feedback import (
    one_shot_wta,
    ucb_wta_feedback_allocator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ucb_error_feedback_gate.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.2)
    args = parser.parse_args()

    true = np.array([1.0, 2.0, 1.5, 0.8])
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        noisy = true + args.noise * rng.standard_normal(true.size)
        one = one_shot_wta(true, noisy, budget=4.0)
        feedback = ucb_wta_feedback_allocator(
            true,
            noisy,
            budget=4.0,
            observation_noise_scale=args.noise,
            max_rounds=args.max_rounds,
            confidence=0.05,
            explore=2,
            seed=seed,
        )
        rows.append({
            "seed": seed,
            "one_shot_deflection": one["true_deflection"],
            "feedback_deflection": feedback["true_deflection"],
            "improvement": feedback["true_deflection"] - one["true_deflection"],
            "rounds_used": feedback["rounds_used"],
            "stopped_by_certificate": feedback["stopped_by_certificate"],
        })
    summary = {
        "mean_rounds_used": float(np.mean([
            row["rounds_used"] for row in rows
        ])),
        "max_rounds_used": int(np.max([
            row["rounds_used"] for row in rows
        ])),
        "certificate_stop_rate": float(np.mean([
            row["stopped_by_certificate"] for row in rows
        ])),
        "mean_improvement": float(np.mean([
            row["improvement"] for row in rows
        ])),
    }
    payload = {
        "gate": "ucb-error-feedback",
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
