"""Communication-aware sensing score gate.

The gate compares the provably optimal CAS score with sensing-only,
communication-only, and exhaustive subset selection on controlled diagonal
models.  Under equal costs, CAS should match the exhaustive optimum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.communication_aware import (
    communication_aware_sensing_score,
    communication_aware_top_k,
    expected_received_deflection,
)
from uav_otfs_isac.expected_pd import expected_gaussian_detection_probability
from uav_otfs_isac.joint_allocation import model_from_bits


def diagonal_model(seed: int):
    rng = np.random.default_rng(seed)
    deltas = np.concatenate(([0.4], rng.uniform(0.8, 2.0, 4)))
    bits = np.array([0, 2, 2, 2, 2])
    model = model_from_bits(deltas, bits, bit_flip_probability=0.05)
    success = np.array([1.0, 0.9, 0.7, 0.8, 0.6])
    return replace(model, success_prob=success, sigma1=model.sigma0)


def exhaustive_best_pd(model, budget):
    candidates = [
        i for i in range(model.num_uavs)
        if i != model.owner and int(model.report_bits[i]) > 0
    ]
    best = -1.0
    for count in range(len(candidates) + 1):
        for subset in combinations(candidates, count):
            scheduled = {model.owner, *subset}
            cost = sum(int(model.report_bits[i]) for i in subset)
            if cost > budget:
                continue
            best = max(best, expected_gaussian_detection_probability(
                model, scheduled, 0.05,
                pd_mode="optimal", grid=32,
            ))
    return best


def exhaustive_best_deflection(model, budget):
    candidates = [
        i for i in range(model.num_uavs)
        if i != model.owner and int(model.report_bits[i]) > 0
    ]
    best = -1.0
    for count in range(len(candidates) + 1):
        for subset in combinations(candidates, count):
            scheduled = {model.owner, *subset}
            cost = sum(int(model.report_bits[i]) for i in subset)
            if cost > budget:
                continue
            best = max(best, expected_received_deflection(model, scheduled))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/communication_aware_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8])
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        model = diagonal_model(seed)
        for budget in args.budgets:
            cas = communication_aware_top_k(model, budget)
            cas_deflection = expected_received_deflection(model, cas)
            exact_deflection = exhaustive_best_deflection(model, budget)
            cas_pd = expected_gaussian_detection_probability(
                model, cas, 0.05, pd_mode="optimal", grid=32,
            )
            exact_pd = exhaustive_best_pd(model, budget)
            rows.append({
                "seed": seed,
                "budget_bits": budget,
                "cas_deflection": float(cas_deflection),
                "exact_deflection": float(exact_deflection),
                "cas_pd": float(cas_pd),
                "exhaustive_pd": float(exact_pd),
                "cas_pd_gap_pp": float((exact_pd - cas_pd) * 100.0),
            })
    summary = []
    for budget in args.budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        gaps = [row["cas_pd_gap_pp"] for row in group]
        summary.append({
            "budget_bits": budget,
            "mean_cas_pd_gap_pp": float(np.mean(gaps)),
            "max_cas_pd_gap_pp": float(np.max(gaps)),
            "expected_deflection_match_rate": float(np.mean([
                abs(row["cas_deflection"] - row["exact_deflection"]) <= 1e-9
                for row in group
            ])),
        })
    payload = {
        "gate": "communication-aware-sensing-score",
        "seeds": args.seeds,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
