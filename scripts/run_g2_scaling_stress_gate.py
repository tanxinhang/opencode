"""G2 scaling stress gate: large gains persist and grow with Q.

Unlike the saturated default G2, this controlled non-saturated model gives the
conditional re-ranking room to act.  Gains over Static ID Top-K remain large
as the number of targets grows, and worst-target gains increase with Q.
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

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.selection import greedy_select
from scripts.run_g2_system_sweep import _pd_vector, _score_topk


def run_gate(*, output: Path, seeds: int, strength: float) -> None:
    rows = []
    for target_count in (3, 5, 8):
        reports_per_target = 4
        budget = 3 * target_count
        diffs = []
        for offset in range(seeds):
            models = _build_models(
                target_count, np.random.default_rng(20260805 + offset),
                strength,
            )
            false_alarm_rate = 0.05
            qos = np.zeros(target_count)
            weights = np.ones(target_count)
            perf = np.ones(target_count)
            conditional = greedy_select(
                models, budget, qos, weights, perf, qos_first=False
            )
            conditional_pd = _pd_vector(
                models, conditional.scheduled, false_alarm_rate
            )
            independent = _score_topk(
                models, budget, "independent_deflection"
            )
            independent_pd = _pd_vector(
                models, independent, false_alarm_rate
            )
            exact = _exact_pd_greedy(
                models, budget, false_alarm_rate
            )
            exact_pd = _pd_vector(models, exact, false_alarm_rate)
            diff = float(np.mean(conditional_pd) - np.mean(independent_pd))
            diffs.append(diff)
            rows.append({
                "target_count": target_count,
                "budget_bits": budget,
                "seed_offset": offset,
                "conditional_mean_pd": float(np.mean(conditional_pd)),
                "conditional_worst_pd": float(np.min(conditional_pd)),
                "exact_pd_mean_pd": float(np.mean(exact_pd)),
                "exact_pd_worst_pd": float(np.min(exact_pd)),
                "independent_mean_pd": float(np.mean(independent_pd)),
                "independent_worst_pd": float(np.min(independent_pd)),
                "paired_diff": diff,
            })
    summary = []
    for target_count in (3, 5, 8):
        group = [
            row for row in rows if row["target_count"] == target_count
        ]
        summary.append({
            "target_count": target_count,
            "conditional_mean_pd": float(np.mean([
                row["conditional_mean_pd"] for row in group
            ])),
            "exact_pd_mean_pd": float(np.mean([
                row["exact_pd_mean_pd"] for row in group
            ])),
            "independent_mean_pd": float(np.mean([
                row["independent_mean_pd"] for row in group
            ])),
            "conditional_vs_independent": float(np.mean([
                row["paired_diff"] for row in group
            ])),
            "exact_vs_independent": float(
                np.mean([row["exact_pd_mean_pd"] for row in group])
                - np.mean([row["independent_mean_pd"] for row in group])
            ),
            "conditional_worst_pd": float(np.mean([
                row["conditional_worst_pd"] for row in group
            ])),
            "independent_worst_pd": float(np.mean([
                row["independent_worst_pd"] for row in group
            ])),
        })
    payload = {
        "gate": "G2-scaling-stress",
        "strength": strength,
        "reports_per_target": 4,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _build_models(target_count, rng, strength):
    models = []
    for _ in range(target_count):
        delta = np.asarray([2.0, 1.9, 1.4, 1.35]) * strength * (
            1.0 + 0.08 * rng.normal(size=4)
        )
        model = symmetric_diversity_model(delta)
        sigma0 = np.eye(5)
        sigma0[1, 2] = sigma0[2, 1] = 0.9
        models.append(replace(
            model,
            mu1=np.concatenate(([0.2], delta)),
            sigma0=sigma0,
            sigma1=sigma0 * 0.5,
            success_prob=np.asarray([1.0, 0.95, 0.95, 0.9, 0.9]),
            report_bits=np.asarray([0, 1, 1, 1, 1]),
        ))
    return models


def _exact_pd_greedy(models, budget_bits, false_alarm_rate):
    scheduled = [{model.owner} for model in models]
    used = 0
    while True:
        best = None
        for q, model in enumerate(models):
            current = gaussian_detection_probability(
                model.mu0, model.mu1, model.sigma0, model.sigma1,
                scheduled[q], false_alarm_rate,
            )
            for i in range(1, model.num_uavs):
                if i in scheduled[q]:
                    continue
                if used + int(model.report_bits[i]) > budget_bits:
                    continue
                new = gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1,
                    scheduled[q] | {i}, false_alarm_rate,
                )
                gain = max(new - current, 0.0)
                if best is None or gain > best[0]:
                    best = (gain, q, i)
        if best is None or best[0] <= 1e-8:
            break
        _, q, i = best
        scheduled[q].add(i)
        used += int(models[q].report_bits[i])
    return scheduled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g2_scaling_stress_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.8)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        strength=args.strength,
    )


if __name__ == "__main__":
    main()
