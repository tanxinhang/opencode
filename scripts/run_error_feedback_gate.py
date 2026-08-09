"""Error-feedback WTA allocation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.error_feedback import evaluate_feedback_gain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/error_feedback_gate.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 3, 10, 30])
    parser.add_argument("--noise", type=float, default=0.8)
    args = parser.parse_args()

    true = np.array([1.0, 2.0, 1.5, 0.8])
    rows = []
    for seed in range(args.seeds):
        for rounds in args.rounds:
            row = evaluate_feedback_gain(
                true,
                args.noise,
                budget=4.0,
                rounds=rounds,
                learning_rate=0.5,
                explore=2,
                seed=seed,
            )
            rows.append({"seed": seed, "rounds": rounds, **row})
    summary = []
    for rounds in args.rounds:
        group = [row for row in rows if row["rounds"] == rounds]
        summary.append({
            "rounds": rounds,
            "mean_improvement": float(np.mean([
                row["feedback_improvement"] for row in group
            ])),
            "improvement_rate": float(np.mean([
                row["feedback_improvement"] > 1e-9 for row in group
            ])),
        })
    payload = {
        "gate": "error-feedback-wta",
        "noise": args.noise,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
