"""When does robust CAS differ from nominal CAS?

The clean and robust top-K orders differ exactly when there are two reports
whose clean score order is reversed by the endpoint degradation.  This gate
sweeps communication severity and reports the divergence rate and the
endpoint-deflection improvement of the robust schedule.
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
    parser.add_argument("--output", default="results/robust_cas_divergence_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        deltas = np.concatenate(([0.4], rng.uniform(0.8, 2.0, 4)))
        bits = np.array([0, 2, 3, 2, 3])
        clean_success = np.array([1.0, 0.9, 0.7, 0.8, 0.6])
        clean = model_from_bits(deltas, bits, bit_flip_probability=0.0)
        clean = replace(
            clean,
            success_prob=clean_success,
            sigma1=clean.sigma0,
        )
        for flip in (0.1, 0.2, 0.3, 0.4):
            for scale in (0.3, 0.5, 0.7):
                robust = model_from_bits(
                    deltas, bits, bit_flip_probability=flip
                )
                robust = replace(
                    robust,
                    success_prob=np.array(
                        [1.0] + list(clean_success[1:] * scale)
                    ),
                    sigma1=robust.sigma0,
                )
                nominal = communication_aware_top_k(clean, args.budget)
                robust_top = communication_aware_top_k(robust, args.budget)
                nominal_deflection = expected_received_deflection(
                    robust, nominal
                )
                robust_deflection = expected_received_deflection(
                    robust, robust_top
                )
                rows.append({
                    "seed": seed,
                    "flip": flip,
                    "success_scale": scale,
                    "divergent": nominal != robust_top,
                    "nominal_endpoint_deflection": float(nominal_deflection),
                    "robust_endpoint_deflection": float(robust_deflection),
                    "improvement": float(
                        robust_deflection - nominal_deflection
                    ),
                })
    divergent = [row for row in rows if row["divergent"]]
    summary = {
        "total_cells": len(rows),
        "divergence_rate": float(len(divergent) / len(rows)),
        "mean_improvement_when_divergent": float(np.mean([
            row["improvement"] for row in divergent
        ])) if divergent else 0.0,
        "max_improvement_when_divergent": float(np.max([
            row["improvement"] for row in divergent
        ])) if divergent else 0.0,
    }
    payload = {
        "gate": "robust-cas-divergence",
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
