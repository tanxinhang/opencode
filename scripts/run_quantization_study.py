"""Quantization study: variable-rate vs fixed-rate under a tight bit budget.

For a fixed total budget, the receiver can spend bits on fewer, finely
quantized reports or on more, coarsely quantized reports.  Under a scalar
quantizer whose distortion decreases with bit count, the better choice
depends on the marginal evidence of each report and on how tight the budget
is.  The budgets are set from the all-report costs: variable 1-4 bit
reporting needs 20 bits for all eight reports, while fixed 3-bit reporting
needs 24 bits.  Budgets 18/20/24 therefore cover the regimes where the
fixed-rate arm is short of bits, just short of its all-report cost, and has
its full budget.
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

from uav_otfs_isac.exact_quota_selection import exact_maxmin_select
from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.reporting import post_bsc_moments, quantizer_from_gaussian_range


def _quantized_model(
    deltas: np.ndarray,
    bits: np.ndarray,
) -> TargetEvidenceModel:
    n = len(deltas)
    post_mu0 = np.zeros(n)
    post_mu1 = np.zeros(n)
    var0 = np.ones(n)
    var1 = np.ones(n)
    costs = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0:
            post_mu1[i] = float(deltas[i])
            continue
        edges, values = quantizer_from_gaussian_range(
            [0.0], [1.0], [deltas[i]], [1.0], int(bits[i]),
        )
        post_mu0[i], var0[i] = post_bsc_moments(
            0.0, 1.0, edges, values, int(bits[i]), 0.0,
        )
        post_mu1[i], var1[i] = post_bsc_moments(
            float(deltas[i]), 1.0, edges, values, int(bits[i]), 0.0,
        )
        costs[i] = int(bits[i])
    return TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=post_mu0,
        mu1=post_mu1,
        sigma0=np.diag(var0),
        sigma1=np.diag(var1),
        success_prob=np.ones(n),
        report_bits=costs,
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def run_gate(*, output: Path, seeds: int, budgets, grid: int) -> None:
    rows = []
    for budget in budgets:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            deltas = np.concatenate(([0.4], rng.uniform(1.0, 1.4, 8)))
            variable_bits = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4])
            fixed_bits = np.array([0, 3, 3, 3, 3, 3, 3, 3, 3])
            variable = exact_maxmin_select(
                [_quantized_model(deltas, variable_bits)],
                budget, 0.05, grid=grid, max_exhaustive_reports=10,
            )
            fixed = exact_maxmin_select(
                [_quantized_model(deltas, fixed_bits)],
                budget, 0.05, grid=grid, max_exhaustive_reports=10,
            )
            rows.append({
                "budget_bits": budget,
                "seed": seed,
                "variable_worst": float(np.min(variable.expected_pd)),
                "fixed_worst": float(np.min(fixed.expected_pd)),
                "variable_used_bits": int(variable.used_bits),
                "fixed_used_bits": int(fixed.used_bits),
                "variable_gain_pp": float(
                    (np.min(variable.expected_pd) - np.min(fixed.expected_pd)) * 100.0
                ),
            })

    summary = []
    for budget in budgets:
        cell = [r for r in rows if r["budget_bits"] == budget]
        gains = [r["variable_gain_pp"] for r in cell]
        summary.append({
            "budget_bits": budget,
            "n_seeds": len(cell),
            "variable_worst_mean": float(np.mean([r["variable_worst"] for r in cell])),
            "fixed_worst_mean": float(np.mean([r["fixed_worst"] for r in cell])),
            "variable_gain_mean_pp": float(np.mean(gains)),
            "variable_gain_std_pp": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
            "variable_used_mean": float(np.mean([r["variable_used_bits"] for r in cell])),
            "fixed_used_mean": float(np.mean([r["fixed_used_bits"] for r in cell])),
        })

    payload = {
        "gate": "quantization-variable-vs-fixed",
        "seeds": seeds,
        "grid": grid,
        "rows": rows,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/quantization_study.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[18, 20, 24])
    parser.add_argument("--grid", type=int, default=64)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        grid=args.grid,
    )


if __name__ == "__main__":
    main()
