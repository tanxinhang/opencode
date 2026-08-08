"""Hard-scenario audit: exact max-min vs greedy vs lexicographic selection.

The demo geometry is easy for all selectors, so the max-min gain is small
(about 0.1 pp).  This gate constructs a hard two-target scenario: one strong
target whose owner report dominates, and one weak target whose reports are
similar and heterogeneously priced.  At tight budgets, exact max-min must
trade expensive strong-target reports for cheap weak-target evidence, while
forward greedy and lexicographic budget selection do not.
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

from uav_otfs_isac.exact_quota_selection import exact_budget_select, exact_maxmin_select
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.models import TargetEvidenceModel


def _model(deltas: np.ndarray, costs: np.ndarray) -> TargetEvidenceModel:
    n = len(deltas)
    return TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(n),
        mu1=np.asarray(deltas, dtype=float),
        sigma0=np.eye(n),
        sigma1=np.eye(n),
        success_prob=np.ones(n),
        report_bits=np.asarray(costs, dtype=int),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def _scenario(seed: int) -> list[TargetEvidenceModel]:
    rng = np.random.default_rng(seed)
    strong = _model(
        np.array([0.3, 2.0, 0.7, 0.8, 0.9, 1.0]),
        np.array([0, 1, 2, 3, 4, 5]),
    )
    weak_deltas = np.concatenate(([0.2], rng.uniform(0.45, 0.55, 6)))
    weak_costs = np.concatenate((
        [0],
        rng.choice([1, 2, 3], size=6, p=[0.6, 0.3, 0.1]),
    ))
    weak = _model(weak_deltas, weak_costs)
    return [strong, weak]


def run_gate(*, output: Path, seeds: int, budgets, grid: int) -> None:
    rows = []
    for budget in budgets:
        for seed in range(seeds):
            models = _scenario(seed)
            maxmin = exact_maxmin_select(
                models, budget, 0.05, grid=grid, max_exhaustive_reports=10,
            )
            greedy = expected_pd_greedy_select(
                models, budget, 0.05, grid=grid,
            )
            lex = exact_budget_select(
                models, budget, 0.05, grid=grid, max_exhaustive_reports=10,
            )
            rows.append({
                "budget_bits": budget,
                "seed": seed,
                "maxmin_worst": float(np.min(maxmin.expected_pd)),
                "greedy_worst": float(np.min(greedy.expected_pd)),
                "lex_worst": float(np.min(lex.expected_pd)),
                "maxmin_used_bits": int(maxmin.used_bits),
                "greedy_used_bits": int(greedy.used_bits),
                "lex_used_bits": int(lex.used_bits),
            })

    summary = []
    for budget in budgets:
        cell = [r for r in rows if r["budget_bits"] == budget]
        mm = np.mean([r["maxmin_worst"] for r in cell])
        greedy = np.mean([r["greedy_worst"] for r in cell])
        lex = np.mean([r["lex_worst"] for r in cell])
        summary.append({
            "budget_bits": budget,
            "n_seeds": len(cell),
            "maxmin_worst_mean": float(mm),
            "greedy_worst_mean": float(greedy),
            "lex_worst_mean": float(lex),
            "maxmin_gain_over_greedy_pp": float((mm - greedy) * 100.0),
            "maxmin_gain_over_lex_pp": float((mm - lex) * 100.0),
            "qos_feasible_rate": float(
                sum(r["maxmin_worst"] >= 0.85 - 1e-9 for r in cell)
                / len(cell)
            ),
        })

    payload = {
        "gate": "hard-maxmin-vs-greedy-vs-lex",
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
    parser.add_argument("--output", default="results/hard_maxmin_scenario.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10])
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
