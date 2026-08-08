"""Multi-target exact joint bit-allocation and selection gate.

With one strong and one weak target competing for the same report budget,
the greedy per-report bit allocation can spend too many bits on the strong
target and starve the weak one.  The exact joint arm enumerates, per target,
every combination of report selection and 1-4 bit allocation, keeps the
Pareto frontier of (cost, worst-target value), and solves the resulting
target-separable multiple-choice knapsack exactly.  This is the joint
extension of Theorem 1/2 to bit allocation, and requires no diminishing-
returns assumption.
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


def _model(deltas: np.ndarray, bits: np.ndarray) -> TargetEvidenceModel:
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
        m0, m1, v0, v1 = _moments(float(deltas[i]), int(bits[i]))
        post_mu0[i], post_mu1[i] = m0, m1
        var0[i], var1[i] = v0, v1
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


def _target_options(
    owner_delta: float,
    deltas: np.ndarray,
    grid: int,
    max_bits: int = 4,
) -> list[tuple[int, float]]:
    options = [[(0, 0.0, 0.0, 1.0, 1.0)] for _ in deltas]
    for index, delta in enumerate(deltas):
        for bits in range(1, max_bits + 1):
            options[index].append((bits, *_moments(float(delta), bits)))
    out: list[tuple[int, float]] = []
    for combo in itertools.product(*options):
        cost = sum(item[0] for item in combo)
        mu0 = [0.0]
        mu1 = [owner_delta]
        var0 = [1.0]
        var1 = [1.0]
        for (bits, m0, m1, v0, v1) in combo:
            if bits > 0:
                mu0.append(m0)
                mu1.append(m1)
                var0.append(v0)
                var1.append(v1)
        pd = float(optimal_gaussian_detection_probability(
            np.asarray(mu0), np.asarray(mu1),
            np.diag(var0), np.diag(var1),
            set(range(len(mu0))), 0.05, grid=grid,
        ))
        out.append((cost, pd))
    out.sort()
    pareto: list[tuple[int, float]] = []
    best_value = -1.0
    for cost, pd in out:
        if pd > best_value + 1e-12:
            pareto.append((cost, pd))
            best_value = pd
    return pareto


def _exact_joint_maxmin(
    target_options_list: list[list[tuple[int, float]]],
    budget_bits: int,
) -> float:
    values = sorted({value for options in target_options_list for _, value in options})

    def feasible(threshold: float) -> bool:
        total = 0
        for options in target_options_list:
            best = None
            for cost, value in options:
                if value >= threshold - 1e-12 and (best is None or cost < best):
                    best = cost
            if best is None:
                return False
            total += best
            if total > budget_bits:
                return False
        return True

    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(values[mid]):
            lo = mid
        else:
            hi = mid - 1
    return float(values[lo])


def run_gate(*, output: Path, seeds: int, budgets, grid: int) -> None:
    pattern = np.array([0, 1, 2, 3, 4])
    rows = []
    for budget in budgets:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            strong = np.concatenate(([0.4], rng.uniform(1.8, 2.2, 4)))
            weak = np.concatenate(([0.3], rng.uniform(1.2, 1.6, 4)))

            fixed = float(np.min(exact_maxmin_select(
                [_model(strong, pattern), _model(weak, pattern)],
                budget, 0.05, grid=grid, max_exhaustive_reports=10,
            ).expected_pd))

            strong_bits = np.concatenate((
                [0], _greedy_bits(strong[1:], budget, grid),
            ))
            weak_bits = np.concatenate((
                [0], _greedy_bits(weak[1:], budget, grid),
            ))
            greedy = float(np.min(exact_maxmin_select(
                [_model(strong, strong_bits), _model(weak, weak_bits)],
                budget, 0.05, grid=grid, max_exhaustive_reports=10,
            ).expected_pd))

            joint = _exact_joint_maxmin(
                [
                    _target_options(0.4, strong[1:], grid),
                    _target_options(0.3, weak[1:], grid),
                ],
                budget,
            )
            rows.append({
                "budget_bits": budget,
                "seed": seed,
                "fixed_pd": fixed,
                "greedy_pd": greedy,
                "exact_joint_pd": joint,
                "joint_over_greedy_pp": float((joint - greedy) * 100.0),
                "joint_over_fixed_pp": float((joint - fixed) * 100.0),
            })

    summary = []
    for budget in budgets:
        cell = [r for r in rows if r["budget_bits"] == budget]
        joint_gain = [r["joint_over_greedy_pp"] for r in cell]
        summary.append({
            "budget_bits": budget,
            "n_seeds": len(cell),
            "fixed_pd_mean": float(np.mean([r["fixed_pd"] for r in cell])),
            "greedy_pd_mean": float(np.mean([r["greedy_pd"] for r in cell])),
            "exact_joint_pd_mean": float(np.mean([r["exact_joint_pd"] for r in cell])),
            "joint_over_greedy_mean_pp": float(np.mean(joint_gain)),
            "joint_over_greedy_median_pp": float(np.median(joint_gain)),
            "joint_over_greedy_min_pp": float(np.min(joint_gain)),
            "joint_over_greedy_max_pp": float(np.max(joint_gain)),
            "joint_over_fixed_mean_pp": float(
                np.mean([r["joint_over_fixed_pp"] for r in cell])
            ),
        })

    payload = {
        "gate": "multi-target-exact-joint-bit-allocation",
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
    parser.add_argument("--output", default="results/joint_multi_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[14, 16, 18])
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
