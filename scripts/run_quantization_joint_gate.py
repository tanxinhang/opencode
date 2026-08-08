"""Joint bit-allocation and selection gate.

For a single target, every non-owner report has five options: not selected,
or selected with 1-4 quantization bits.  Enumerating all combinations with
total cost at most the budget gives the exact joint optimum of bit allocation
and report selection; this is a small-scale oracle.  The gate compares the
oracle with a fixed 1-4 pattern and with the water-filling-inspired greedy,
so the gap between the heuristic and the global optimum is explicit.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.exact_quota_selection import exact_maxmin_select
from uav_otfs_isac.expected_pd import expected_gaussian_detection_probability
from uav_otfs_isac.fusion import optimal_gaussian_detection_probability
from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.reporting import post_bsc_moments, quantizer_from_gaussian_range


def _moments(delta: float, bits: int):
    edges, values = quantizer_from_gaussian_range(
        [0.0], [1.0], [delta], [1.0], bits,
    )
    m0, v0 = post_bsc_moments(0.0, 1.0, edges, values, bits, 0.0)
    m1, v1 = post_bsc_moments(float(delta), 1.0, edges, values, bits, 0.0)
    return m0, m1, v0, v1


def _one_report_model(delta: float, bits: int) -> TargetEvidenceModel:
    m0, m1, v0, v1 = _moments(delta, bits)
    return TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.array([0.0, m0]),
        mu1=np.array([delta, m1]),
        sigma0=np.diag([1.0, v0]),
        sigma1=np.diag([1.0, v1]),
        success_prob=np.ones(2),
        report_bits=np.array([0, bits]),
        bit_flip_prob=np.zeros(2),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def _greedy_bits(deltas, budget_bits, grid, max_bits=4) -> np.ndarray:
    bits = np.ones(len(deltas), dtype=int)
    gains = {}
    for index, delta in enumerate(deltas):
        previous = float(expected_gaussian_detection_probability(
            _one_report_model(float(delta), 1), {0, 1}, 0.05,
            pd_mode="optimal", grid=grid,
        ))
        row = []
        for candidate in range(2, max_bits + 1):
            current = float(expected_gaussian_detection_probability(
                _one_report_model(float(delta), candidate), {0, 1}, 0.05,
                pd_mode="optimal", grid=grid,
            ))
            row.append(current - previous)
            previous = current
        gains[index] = row
    while bits.sum() < budget_bits:
        best_index = None
        best_gain = 0.0
        for index, row in gains.items():
            if bits[index] >= max_bits:
                continue
            gain = row[bits[index] - 1]
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_index = index
        if best_index is None:
            break
        bits[best_index] += 1
    return bits


def _pattern_bits(n_reports: int) -> np.ndarray:
    pattern = np.array([1, 2, 3, 4], dtype=int)
    return np.resize(pattern, n_reports)


def _model_from_options(deltas, bits) -> TargetEvidenceModel:
    post_mu0 = np.zeros(len(deltas) + 1)
    post_mu1 = np.zeros(len(deltas) + 1)
    var0 = np.ones(len(deltas) + 1)
    var1 = np.ones(len(deltas) + 1)
    costs = np.zeros(len(deltas) + 1, dtype=int)
    for index, (delta, bit_count) in enumerate(zip(deltas, bits), start=1):
        m0, m1, v0, v1 = _moments(delta, int(bit_count))
        post_mu0[index], post_mu1[index] = m0, m1
        var0[index], var1[index] = v0, v1
        costs[index] = int(bit_count)
    return TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=post_mu0,
        mu1=post_mu1,
        sigma0=np.diag(var0),
        sigma1=np.diag(var1),
        success_prob=np.ones(len(deltas) + 1),
        report_bits=costs,
        bit_flip_prob=np.zeros(len(deltas) + 1),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def _exact_joint_pd(
    deltas: np.ndarray,
    budget_bits: int,
    grid: int,
    max_bits: int = 4,
) -> tuple[float, int]:
    options = [[(0, 0.0, 0.0, 1.0, 1.0)]]
    for delta in deltas:
        row = [(0, 0.0, 0.0, 1.0, 1.0)]
        for bits in range(1, max_bits + 1):
            row.append((bits, *_moments(float(delta), bits)))
        options.append(row)
    best_pd = 0.0
    best_used = 0
    for combo in itertools.product(*options):
        cost = sum(item[0] for item in combo)
        if cost > budget_bits:
            continue
        mu0 = [0.0]
        mu1 = [0.0]
        var0 = [1.0]
        var1 = [1.0]
        for (bits, m0, m1, v0, v1) in combo[1:]:
            if bits > 0:
                mu0.append(m0)
                mu1.append(m1)
                var0.append(v0)
                var1.append(v1)
        if len(mu0) == 1:
            continue
        pd = float(optimal_gaussian_detection_probability(
            np.asarray(mu0), np.asarray(mu1),
            np.diag(var0), np.diag(var1),
            set(range(len(mu0))), 0.05, grid=grid,
        ))
        if pd > best_pd:
            best_pd = pd
            best_used = cost
    return best_pd, best_used


def run_gate(*, output: Path, seeds: int, budgets, grid: int) -> None:
    rows = []
    for budget in budgets:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            deltas = rng.uniform(1.0, 1.4, 6)
            pattern = _pattern_bits(6)
            greedy = _greedy_bits(deltas, budget, grid)
            pattern_pd = float(np.min(exact_maxmin_select(
                [_model_from_options(deltas, pattern)],
                budget, 0.05, grid=grid, max_exhaustive_reports=10,
            ).expected_pd))
            greedy_pd = float(np.min(exact_maxmin_select(
                [_model_from_options(deltas, greedy)],
                budget, 0.05, grid=grid, max_exhaustive_reports=10,
            ).expected_pd))
            exact_pd, exact_used = _exact_joint_pd(deltas, budget, grid)
            rows.append({
                "budget_bits": budget,
                "seed": seed,
                "pattern_pd": pattern_pd,
                "greedy_pd": greedy_pd,
                "exact_joint_pd": exact_pd,
                "exact_joint_used_bits": exact_used,
                "exact_over_greedy_pp": float((exact_pd - greedy_pd) * 100.0),
                "exact_over_pattern_pp": float((exact_pd - pattern_pd) * 100.0),
            })

    summary = []
    for budget in budgets:
        cell = [r for r in rows if r["budget_bits"] == budget]
        summary.append({
            "budget_bits": budget,
            "n_seeds": len(cell),
            "pattern_pd_mean": float(np.mean([r["pattern_pd"] for r in cell])),
            "greedy_pd_mean": float(np.mean([r["greedy_pd"] for r in cell])),
            "exact_joint_pd_mean": float(np.mean([r["exact_joint_pd"] for r in cell])),
            "exact_over_greedy_mean_pp": float(
                np.mean([r["exact_over_greedy_pp"] for r in cell])
            ),
            "exact_over_pattern_mean_pp": float(
                np.mean([r["exact_over_pattern_pp"] for r in cell])
            ),
            "exact_joint_used_mean": float(
                np.mean([r["exact_joint_used_bits"] for r in cell])
            ),
        })

    payload = {
        "gate": "quantization-joint-bit-allocation-oracle",
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
    parser.add_argument("--output", default="results/quantization_joint_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
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
