"""G3 gate: P_D-optimal linear fusion is set-monotone at operating points.

The deflection-optimal linear score maximizes H0-null deflection, but it can
lower P_D when H1 covariance is not proportional to H0 covariance.  This gate
evaluates a one-parameter family of linear scores whose KKT optimum contains
the P_D-optimal direction.  It verifies (a) exact agreement with the closed
form in the proportional-covariance regime, (b) set monotonicity at operating
points with P_D >= 0.5, and (c) P_D gains over the deflection-optimal score
under unequal H1 covariance, including at the greedy scheduling level.
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

from uav_otfs_isac.fusion import (
    gaussian_detection_probability,
    gaussian_pd_closed_form,
    optimal_deflection,
    optimal_gaussian_detection_probability,
)
from uav_otfs_isac.models import TargetEvidenceModel


OPERATING_PD = 0.5


def stress_model(rng: np.random.Generator, n: int = 5) -> TargetEvidenceModel:
    """Unequal-covariance controlled model with a strong enough signal."""
    raw0 = rng.normal(size=(n, n))
    sigma0 = raw0 @ raw0.T + 0.4 * np.eye(n)
    raw1 = rng.normal(size=(n, n))
    scale = rng.uniform(0.4, 2.0, n)
    sigma1 = (raw1 @ raw1.T + 0.4 * np.eye(n)) * (
        scale[:, None] * scale[None, :]
    )
    mu0 = rng.normal(size=n) * 0.1
    mu1 = mu0 + rng.normal(size=n) * 1.5
    return TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=mu0,
        mu1=mu1,
        sigma0=sigma0,
        sigma1=sigma1,
        success_prob=np.ones(n),
        report_bits=np.array([0] + [1] * (n - 1), dtype=int),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def _pd_greedy(
    models,
    budget_bits: int,
    false_alarm_rate: float,
    *,
    optimal: bool,
    grid: int,
):
    scheduled = [{model.owner} for model in models]
    used = 0

    def evaluate(model, indices):
        if optimal:
            return optimal_gaussian_detection_probability(
                model.mu0, model.mu1, model.sigma0, model.sigma1,
                indices, false_alarm_rate, grid=grid,
            )
        return gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            indices, false_alarm_rate,
        )

    while True:
        best = None
        for q, model in enumerate(models):
            current = evaluate(model, scheduled[q])
            for i in range(model.num_uavs):
                if i == model.owner or i in scheduled[q]:
                    continue
                cost = int(model.report_bits[i])
                if used + cost > budget_bits:
                    continue
                new = evaluate(model, scheduled[q] | {i})
                gain = max(new - current, 0.0) / max(cost, 1)
                if best is None or gain > best[0]:
                    best = (gain, q, i, cost)
        if best is None or best[0] <= 1e-14:
            break
        _, q, i, cost = best
        scheduled[q].add(i)
        used += cost
    return scheduled


def run_gate(
    *, output: Path, seeds: int, grid: int, greedy_instances: int, seed: int,
) -> None:
    false_alarm_rate = 0.05
    rng = np.random.default_rng(seed)
    operating_edges = 0
    operating_opt_decreasing = 0
    operating_def_decreasing = 0
    maximum_def_drop = 0.0
    all_edges_gains = []
    proportional_checks = []

    for _ in range(seeds):
        model = stress_model(rng)
        n = model.num_uavs
        for mask in range(1 << n):
            base = {i for i in range(n) if mask & (1 << i)}
            if not base:
                continue
            base_optimal = optimal_gaussian_detection_probability(
                model.mu0, model.mu1, model.sigma0, model.sigma1,
                base, false_alarm_rate, grid=grid,
            )
            for candidate in range(n):
                if candidate in base:
                    continue
                new_set = base | {candidate}
                base_deflection = gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1,
                    base, false_alarm_rate,
                )
                new_deflection = gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1,
                    new_set, false_alarm_rate,
                )
                new_optimal = optimal_gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1,
                    new_set, false_alarm_rate, grid=grid,
                )
                all_edges_gains.append(new_optimal - new_deflection)
                if base_optimal < OPERATING_PD:
                    continue
                operating_edges += 1
                if new_optimal < base_optimal - 1e-9:
                    operating_opt_decreasing += 1
                if new_deflection < base_deflection - 1e-10:
                    operating_def_decreasing += 1
                    maximum_def_drop = max(
                        maximum_def_drop, base_deflection - new_deflection
                    )

        for ratio in (0.3, 0.7, 1.0, 1.8):
            sigma1 = ratio * model.sigma0
            for indices in ({0}, {0, 2}, {0, 1, 2, 3, 4}):
                deflection = optimal_deflection(
                    model.mu1 - model.mu0, model.sigma0, indices
                )
                closed = gaussian_pd_closed_form(
                    deflection, ratio, false_alarm_rate
                )
                optimal = optimal_gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, sigma1,
                    indices, false_alarm_rate, grid=grid,
                )
                proportional_checks.append(float(abs(optimal - closed)))

    greedy_rows = []
    for _ in range(greedy_instances):
        models = [stress_model(rng) for _ in range(3)]
        budget_bits = 6
        deflection_schedule = _pd_greedy(
            models, budget_bits, false_alarm_rate, optimal=False, grid=grid
        )
        optimal_schedule = _pd_greedy(
            models, budget_bits, false_alarm_rate, optimal=True, grid=grid
        )
        all_scheduled = [
            set(range(model.num_uavs)) for model in models
        ]

        def mean_pd(scheduled, optimal):
            values = []
            for q, model in enumerate(models):
                if optimal:
                    values.append(optimal_gaussian_detection_probability(
                        model.mu0, model.mu1, model.sigma0, model.sigma1,
                        scheduled[q], false_alarm_rate, grid=grid,
                    ))
                else:
                    values.append(gaussian_detection_probability(
                        model.mu0, model.mu1, model.sigma0, model.sigma1,
                        scheduled[q], false_alarm_rate,
                    ))
            return float(np.mean(values))

        greedy_rows.append({
            "deflection_greedy_mean_pd_deflection_rule": mean_pd(
                deflection_schedule, optimal=False
            ),
            "deflection_greedy_mean_pd_optimal_rule": mean_pd(
                deflection_schedule, optimal=True
            ),
            "optimal_greedy_mean_pd_deflection_rule": mean_pd(
                optimal_schedule, optimal=False
            ),
            "optimal_greedy_mean_pd_optimal_rule": mean_pd(
                optimal_schedule, optimal=True
            ),
            "all_scheduled_mean_pd_optimal_rule": mean_pd(
                all_scheduled, optimal=True
            ),
            "deflection_greedy_reports": int(sum(
                len(group) - 1 for group in deflection_schedule
            )),
            "optimal_greedy_reports": int(sum(
                len(group) - 1 for group in optimal_schedule
            )),
        })

    all_gains = np.asarray(all_edges_gains, dtype=float)
    greedy_summary = {}
    if greedy_rows:
        greedy_summary = {
            "mean_fusion_gain_on_deflection_schedule": float(np.mean([
                row["deflection_greedy_mean_pd_optimal_rule"]
                - row["deflection_greedy_mean_pd_deflection_rule"]
                for row in greedy_rows
            ])),
            "mean_scheduling_gain_under_optimal_rule": float(np.mean([
                row["optimal_greedy_mean_pd_optimal_rule"]
                - row["deflection_greedy_mean_pd_optimal_rule"]
                for row in greedy_rows
            ])),
            "mean_total_gain": float(np.mean([
                row["optimal_greedy_mean_pd_optimal_rule"]
                - row["deflection_greedy_mean_pd_deflection_rule"]
                for row in greedy_rows
            ])),
            "mean_optimal_rule_pd": float(np.mean([
                row["optimal_greedy_mean_pd_optimal_rule"]
                for row in greedy_rows
            ])),
            "mean_all_scheduled_pd": float(np.mean([
                row["all_scheduled_mean_pd_optimal_rule"]
                for row in greedy_rows
            ])),
            "scheduling_gain_positive_rate": float(np.mean([
                row["optimal_greedy_mean_pd_optimal_rule"]
                > row["deflection_greedy_mean_pd_optimal_rule"]
                + 1e-8
                for row in greedy_rows
            ])),
        }

    payload = {
        "gate": "G3-pd-optimal-fusion",
        "operating_pd": OPERATING_PD,
        "seed": seed,
        "grid": grid,
        "monotonicity": {
            "operating_edges": operating_edges,
            "optimal_decreasing_edges": operating_opt_decreasing,
            "deflection_decreasing_edges": operating_def_decreasing,
            "deflection_decreasing_edge_rate": (
                operating_def_decreasing / max(operating_edges, 1)
            ),
            "maximum_deflection_pd_drop": maximum_def_drop,
        },
        "gains": {
            "mean_pd_gain_over_deflection": float(np.mean(all_gains)),
            "maximum_pd_gain_over_deflection": float(np.max(all_gains)),
        },
        "proportional_regime": {
            "checks": len(proportional_checks),
            "maximum_absolute_error_vs_closed_form": float(max(
                proportional_checks, default=0.0
            )),
        },
        "greedy": greedy_summary,
        "rows": {
            "monotonicity_instances": seeds,
            "greedy_instances": greedy_instances,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "monotonicity": payload["monotonicity"],
        "gains": payload["gains"],
        "proportional_regime": payload["proportional_regime"],
        "greedy": payload["greedy"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/pd_optimal_fusion_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--grid", type=int, default=2048)
    parser.add_argument("--greedy-instances", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        grid=args.grid,
        greedy_instances=args.greedy_instances,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
