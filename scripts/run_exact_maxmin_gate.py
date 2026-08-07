"""G8-M gate: exact max-min selection under heterogeneous report costs.

The system-level objective is worst-target expected P_D, not the
lexicographic weighted-sum objective.  This gate adds the exact max-min
selector: for a threshold `t`, feasibility is a multiple-choice knapsack
problem over enumerated report subsets with value at least `t`, and the
optimal threshold is found by binary search over the finite set of candidate
values.  The gate verifies the selector against an exhaustive global max-min
oracle, then compares it with forward greedy in the tight-budget
variable-rate demo scenario.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.exact_quota_selection import (
    exact_maxmin_select,
    subset_expected_pd_map,
)
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.scenario import build_models


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


def _paired_t_test(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if std <= 1e-15:
        p = 0.5 if mean <= 1e-12 else 0.0
        t = 0.0
    else:
        t = float(mean / (std / np.sqrt(n)))
        p = float(stats.t.sf(t, df=n - 1))
    return {"mean": mean, "std": std, "n": n, "t": t, "p_one_sided": p}


def _paired_bootstrap_ci(
    values, seed: int = 20260807, n_resamples: int = 10000
):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        means[index] = float(np.mean(sample))
    return {
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "n_resamples": n_resamples,
    }


def _aggregate_by_budget(
    rows, budget_key, numeric_fields, bool_fields=(), paired_fields=()
):
    budgets = sorted({row[budget_key] for row in rows})
    out = []
    for budget in budgets:
        cell = [row for row in rows if row[budget_key] == budget]
        entry = {"budget_bits": budget, "n_seeds": len(cell)}
        for field in numeric_fields:
            mean, std = _mean_std([row[field] for row in cell])
            entry[f"{field}_mean"] = mean
            entry[f"{field}_std"] = std
        for field in bool_fields:
            entry[f"{field}_rate"] = float(np.mean([
                1.0 if row[field] else 0.0 for row in cell
            ]))
        for field in paired_fields:
            values = [row[field] for row in cell]
            entry[f"{field}_paired_t"] = _paired_t_test(values)
            entry[f"{field}_bootstrap_ci"] = _paired_bootstrap_ci(values)
        out.append(entry)
    return out


def _exhaustive_maxmin_oracle(models, budget_bits, false_alarm_rate, grid):
    subsets = [
        subset_expected_pd_map(model, false_alarm_rate, grid=grid)
        for model in models
    ]
    best = None

    def recurse(index, groups, used):
        nonlocal best
        if index == len(models):
            values = np.asarray([
                subsets[q][groups[q]] for q in range(len(models))
            ], dtype=float)
            score = float(np.min(values))
            if best is None or score > best[0]:
                best = (score, tuple(groups), values, used)
            return
        model = models[index]
        for scheduled, _ in subsets[index].items():
            cost = sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            if used + cost <= budget_bits:
                recurse(index + 1, groups + [scheduled], used + cost)

    recurse(0, [], 0)
    assert best is not None
    return best


def _controlled_audit(seeds, budgets, grid):
    rows = []
    for seed_offset in range(seeds):
        rng = np.random.default_rng(20260807 + seed_offset)
        for budget in budgets:
            models = [
                symmetric_diversity_model(
                    np.sort(rng.uniform(0.6, 1.8, size=4))[::-1],
                    success_probability=float(rng.uniform(0.5, 0.95)),
                    report_bits=np.array([1, 2, 3, 5]),
                )
                for _ in range(3)
            ]
            greedy = expected_pd_greedy_select(
                models, budget, 0.05, grid=grid
            )
            exact = exact_maxmin_select(
                models, budget, 0.05, grid=grid
            )
            oracle = _exhaustive_maxmin_oracle(
                models, budget, 0.05, grid=grid
            )
            rows.append({
                "seed_offset": seed_offset,
                "budget_bits": budget,
                "exact_matches_oracle": bool(
                    np.isclose(np.min(exact.expected_pd), oracle[0])
                ),
                "greedy_worst": float(np.min(greedy.expected_pd)),
                "exact_worst": float(np.min(exact.expected_pd)),
                "gain_worst": float(
                    np.min(exact.expected_pd) - np.min(greedy.expected_pd)
                ),
            })
    return {
        "rows": rows,
        "summary": {
            "oracle_match_rate": float(np.mean([
                row["exact_matches_oracle"] for row in rows
            ])),
            "never_worse_than_greedy": float(np.mean([
                row["gain_worst"] >= -1e-12 for row in rows
            ])),
        },
    }


def _variable_rate_system_audit(seeds, budgets, grid):
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    profile = [1, 2, 3, 4, 5, 2, 3, 1]
    if len(profile) != cfg.num_uavs:
        raise ValueError("variable-rate profile must match num_uavs")
    rows = []
    for seed_offset in range(seeds):
        seed = cfg.seed + seed_offset
        models = build_models(
            cfg,
            np.random.default_rng(seed),
            quantizer_bits_per_uav=profile,
        )
        for budget in budgets:
            greedy = expected_pd_greedy_select(
                models, budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            exact = exact_maxmin_select(
                models, budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            rows.append({
                "seed_offset": seed_offset,
                "report_budget_bits": budget,
                "greedy_mean": float(np.mean(greedy.expected_pd)),
                "exact_mean": float(np.mean(exact.expected_pd)),
                "greedy_worst": float(np.min(greedy.expected_pd)),
                "exact_worst": float(np.min(exact.expected_pd)),
                "gain_worst": float(
                    np.min(exact.expected_pd) - np.min(greedy.expected_pd)
                ),
                "greedy_used_bits": int(greedy.used_bits),
                "exact_used_bits": int(exact.used_bits),
            })
    return {
        "rows": rows,
        "summary": {
            "never_worse_than_greedy": float(np.mean([
                row["gain_worst"] >= -1e-12 for row in rows
            ])),
        },
    }


def run_gate(*, output: Path, seeds: int, budgets, grid: int) -> None:
    controlled = _controlled_audit(seeds, budgets, grid)
    system = _variable_rate_system_audit(seeds, budgets, grid)
    controlled["by_budget"] = _aggregate_by_budget(
        controlled["rows"], "budget_bits",
        ["greedy_worst", "exact_worst", "gain_worst"],
        ["exact_matches_oracle"],
        ["gain_worst"],
    )
    system["by_budget"] = _aggregate_by_budget(
        system["rows"], "report_budget_bits",
        ["greedy_mean", "exact_mean", "greedy_worst", "exact_worst",
         "gain_worst", "greedy_used_bits", "exact_used_bits"],
        paired_fields=["gain_worst"],
    )
    payload = {
        "gate": "G8-M-exact-maxmin-selection",
        "seeds": seeds,
        "grid": grid,
        "controlled": controlled,
        "variable_rate_system": system,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "controlled_summary": controlled["summary"],
        "system_summary": system["summary"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/exact_maxmin_gate.json")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 5, 7, 9, 11])
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
