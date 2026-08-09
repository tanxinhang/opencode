"""Robust communication-aware sensing score gate.

The score is computed on the worst communication endpoint model.  Under the
diagonal/proportional regime, this maximizes the expected received
deflection at the endpoint, so it certifies the worst-case surrogate over
the communication ambiguity rectangle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.communication_aware import (
    communication_aware_top_k,
    expected_received_deflection,
)
from uav_otfs_isac.joint_allocation import model_from_bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robust_communication_aware_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 6, 8])
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        deltas = np.concatenate(([0.4], rng.uniform(0.8, 2.0, 4)))
        bits = np.array([0, 2, 2, 2, 2])
        clean = model_from_bits(deltas, bits, bit_flip_probability=0.0)
        clean = replace(
            clean,
            success_prob=np.array([1.0, 0.9, 0.9, 0.9, 0.9]),
            sigma1=clean.sigma0,
        )
        robust = model_from_bits(deltas, bits, bit_flip_probability=0.2)
        robust = replace(
            robust,
            success_prob=np.array([1.0, 0.5, 0.5, 0.5, 0.5]),
            sigma1=robust.sigma0,
        )
        for budget in args.budgets:
            nominal = communication_aware_top_k(clean, budget)
            robust_top = communication_aware_top_k(robust, budget)
            nominal_endpoint_deflection = expected_received_deflection(
                robust, nominal
            )
            robust_endpoint_deflection = expected_received_deflection(
                robust, robust_top
            )
            rows.append({
                "seed": seed,
                "budget_bits": budget,
                "nominal_endpoint_deflection": (
                    float(nominal_endpoint_deflection)
                ),
                "robust_endpoint_deflection": (
                    float(robust_endpoint_deflection)
                ),
                "robust_improvement": float(
                    robust_endpoint_deflection
                    - nominal_endpoint_deflection
                ),
            })
    summary = []
    for budget in args.budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        summary.append({
            "budget_bits": budget,
            "mean_robust_improvement": float(np.mean([
                row["robust_improvement"] for row in group
            ])),
            "robust_never_worse_rate": float(np.mean([
                row["robust_improvement"] >= -1e-12 for row in group
            ])),
        })
    payload = {
        "gate": "robust-communication-aware",
        "passed": all(row["robust_improvement"] >= -1e-12 for row in rows),
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
