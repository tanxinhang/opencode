"""G2 non-saturated stress gate: large correlation-penetration gains.

The default G2 model saturates P_D and compresses absolute gains.  This gate
uses a controlled three-target model at non-saturated operating points so the
conditional set-dependent ranking has room to produce large, significant
absolute P_D gains over Static Independent-Deflection Top-K.
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


def build_targets(rng: np.random.Generator, strength: float = 0.8):
    models = []
    for _ in range(3):
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


def _bootstrap_ci(values, seed, replicates=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(values))
    samples = []
    for _ in range(replicates):
        sample = rng.choice(indices, size=len(indices), replace=True)
        samples.append(float(np.mean(values[sample])))
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def run_gate(*, output: Path, seeds: int, strength: float) -> None:
    rows = []
    for budget_bits in (6, 9, 12):
        diffs = []
        for offset in range(seeds):
            models = build_targets(
                np.random.default_rng(20260805 + offset), strength
            )
            qos = np.zeros(3)
            weights = np.ones(3)
            perf = np.ones(3)
            false_alarm_rate = 0.05
            conditional = greedy_select(
                models, budget_bits, qos, weights, perf, qos_first=False
            )
            conditional_pd = _pd_vector(
                models, conditional.scheduled, false_alarm_rate
            )
            independent = _score_topk(
                models, budget_bits, "independent_deflection"
            )
            independent_pd = _pd_vector(
                models, independent, false_alarm_rate
            )
            exact = _exact_pd_greedy(
                models, budget_bits, false_alarm_rate
            )
            exact_pd = _pd_vector(models, exact, false_alarm_rate)
            all_scheduled = [
                set(range(model.num_uavs)) for model in models
            ]
            all_pd = _pd_vector(models, all_scheduled, false_alarm_rate)
            diff = float(np.mean(conditional_pd) - np.mean(independent_pd))
            diffs.append(diff)
            rows.append({
                "budget_bits": budget_bits,
                "seed_offset": offset,
                "conditional_mean_pd": float(np.mean(conditional_pd)),
                "conditional_worst_pd": float(np.min(conditional_pd)),
                "exact_pd_mean_pd": float(np.mean(exact_pd)),
                "exact_pd_worst_pd": float(np.min(exact_pd)),
                "independent_mean_pd": float(np.mean(independent_pd)),
                "independent_worst_pd": float(np.min(independent_pd)),
                "all_scheduled_mean_pd": float(np.mean(all_pd)),
                "paired_diff": diff,
                "conditional_win": bool(diff > 0.0),
            })
    summary = []
    for budget_bits in (6, 9, 12):
        group = [
            row for row in rows if row["budget_bits"] == budget_bits
        ]
        summary.append({
            "budget_bits": budget_bits,
            "conditional_mean_pd": float(np.mean([
                row["conditional_mean_pd"] for row in group
            ])),
            "exact_pd_mean_pd": float(np.mean([
                row["exact_pd_mean_pd"] for row in group
            ])),
            "independent_mean_pd": float(np.mean([
                row["independent_mean_pd"] for row in group
            ])),
            "all_scheduled_mean_pd": float(np.mean([
                row["all_scheduled_mean_pd"] for row in group
            ])),
            "paired_diff_mean": float(np.mean([
                row["paired_diff"] for row in group
            ])),
            "paired_diff_bootstrap_ci95": _bootstrap_ci([
                row["paired_diff"] for row in group
            ], seed=20260805),
            "conditional_win_rate": float(np.mean([
                row["conditional_win"] for row in group
            ])),
            "conditional_worst_pd": float(np.mean([
                row["conditional_worst_pd"] for row in group
            ])),
            "independent_worst_pd": float(np.mean([
                row["independent_worst_pd"] for row in group
            ])),
        })
    payload = {
        "gate": "G2-nonsaturated-stress",
        "strength": strength,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g2_nonsaturated_stress_gate.json"
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
