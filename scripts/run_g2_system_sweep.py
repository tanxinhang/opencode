"""Gate G2: system-level budget sweep across N, Q, and B_max."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select


def owners_for(num_uavs: int, num_targets: int) -> list[int]:
    return [
        int(np.round((0.5 + q) * num_uavs / num_targets)) % num_uavs
        for q in range(num_targets)
    ]


def _score_topk(models, budget_bits, score_kind):
    scored = []
    for q, model in enumerate(models):
        candidates = [
            i for i in range(model.num_uavs)
            if i != model.owner
        ]
        if score_kind == "sensing_snr":
            scores = [
                model.delta[i] ** 2 / max(model.sigma0[i, i], 1e-12)
                for i in candidates
            ]
        elif score_kind == "communication":
            scores = [
                model.success_prob[i] / max(model.report_bits[i], 1)
                for i in candidates
            ]
        elif score_kind == "independent_deflection":
            scores = [
                model.success_prob[i] * model.delta[i] ** 2
                / (max(model.sigma0[i, i], 1e-12) * model.report_bits[i])
                for i in candidates
            ]
        else:
            raise ValueError(score_kind)
        scored.extend((
            float(score), int(model.report_bits[i]), q, i
        ) for i, score in zip(candidates, scores))
    scored.sort(key=lambda item: item[0], reverse=True)
    scheduled = [{model.owner} for model in models]
    used = 0
    for score, cost, q, i in scored:
        if used + cost > budget_bits:
            continue
        scheduled[q].add(i)
        used += cost
    return scheduled


def _pd_vector(models, scheduled, false_alarm_rate):
    return np.asarray([
        gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            scheduled[q], false_alarm_rate,
        )
        for q, model in enumerate(models)
    ])


def pd_greedy_select(
    models,
    budget_bits,
    qos_pd,
    qos_weights,
    false_alarm_rate,
):
    """Select reports by relative miss-deficit reduction then logit PD gain."""
    scheduled = [{model.owner} for model in models]
    used = 0

    def current_pd(q, model):
        return gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            scheduled[q], false_alarm_rate,
        )

    def candidates(stage):
        for q, model in enumerate(models):
            pd = current_pd(q, model)
            deficit = max(float(qos_pd[q]) - pd, 0.0)
            for i in range(model.num_uavs):
                if i == model.owner or i in scheduled[q]:
                    continue
                cost = int(model.report_bits[i])
                if used + cost > budget_bits:
                    continue
                new_pd = gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1,
                    scheduled[q] | {i}, false_alarm_rate,
                )
                relative_gain = max(
                    (max(1.0 - pd, 1e-6)
                     - max(1.0 - new_pd, 0.0))
                    / max(1.0 - pd, 1e-6),
                    0.0,
                )
                clip = 1e-6
                logit_gain = max(
                    np.log(max(min(new_pd, 1.0 - clip), clip)
                           / max(min(1.0 - new_pd, 1.0 - clip), clip))
                    - np.log(max(min(pd, 1.0 - clip), clip)
                             / max(min(1.0 - pd, 1.0 - clip), clip)),
                    0.0,
                )
                if stage == "qos":
                    score = qos_weights[q] * relative_gain / max(cost, 1)
                else:
                    absolute_gain = max(new_pd - pd, 0.0)
                    score = qos_weights[q] * absolute_gain / max(cost, 1)
                yield score, relative_gain, q, i, cost

    for stage in ("qos", "performance"):
        while True:
            options = list(candidates(stage))
            if not options:
                break
            options.sort(key=lambda item: item[0], reverse=True)
            score, gain, q, i, cost = options[0]
            if score <= 1e-14:
                break
            scheduled[q].add(i)
            used += cost
    return scheduled


def run_sweep(*, output: Path, seeds: int, budgets, seed_offset: int) -> None:
    base_cfg = load_config("config/demo.yaml")
    rows = []
    for num_uavs in (8, 12):
        for num_targets in (3, 5):
            for budget in budgets:
                cfg = replace(
                    base_cfg,
                    num_uavs=num_uavs,
                    num_targets=num_targets,
                    owners=owners_for(num_uavs, num_targets),
                    report_budget_bits=budget,
                )
                qos_min = np.full(
                    num_targets, float(base_cfg.qos_min_deflection[0])
                )
                qos_weights = np.linspace(1.0, 1.3, num_targets)
                performance_weights = np.ones(num_targets)
                for offset in range(seeds):
                    models = build_models(
                        cfg, np.random.default_rng(cfg.seed + offset + seed_offset)
                    )
                    start = perf_counter()
                    proposed = greedy_select(
                        models, budget, qos_min, qos_weights,
                        performance_weights, qos_first=False,
                    )
                    proposed_seconds = perf_counter() - start
                    proposed_pd = _pd_vector(
                        models, proposed.scheduled, cfg.false_alarm_rate
                    )
                    qos_pd = np.full(
                        num_targets, max(float(base_cfg.qos_min_deflection[0]), 0.5)
                    )
                    pd_scheduled = pd_greedy_select(
                        models, budget, qos_pd, qos_weights,
                        cfg.false_alarm_rate,
                    )
                    pd_aware_pd = _pd_vector(
                        models, pd_scheduled, cfg.false_alarm_rate
                    )
                    baseline_rows = {}
                    for name in (
                        "sensing_snr", "communication",
                        "independent_deflection",
                    ):
                        scheduled = _score_topk(models, budget, name)
                        baseline_rows[name] = _pd_vector(
                            models, scheduled, cfg.false_alarm_rate
                        )
                    all_scheduled = [
                        set(range(model.num_uavs)) for model in models
                    ]
                    all_pd = _pd_vector(
                        models, all_scheduled, cfg.false_alarm_rate
                    )
                    rows.append({
                        "num_uavs": num_uavs,
                        "num_targets": num_targets,
                        "budget_bits": budget,
                        "seed_offset": offset,
                        "proposed_mean_pd": float(np.mean(proposed_pd)),
                        "proposed_worst_pd": float(np.min(proposed_pd)),
                        "proposed_selection_ratio": float(np.mean([
                            len(s) - 1 for s in proposed.scheduled
                        ]) / num_uavs),
                        "proposed_seconds": proposed_seconds,
                        "pd_aware_mean_pd": float(np.mean(pd_aware_pd)),
                        "pd_aware_worst_pd": float(np.min(pd_aware_pd)),
                        "sensing_snr_mean_pd": float(np.mean(
                            baseline_rows["sensing_snr"]
                        )),
                        "communication_mean_pd": float(np.mean(
                            baseline_rows["communication"]
                        )),
                        "independent_deflection_mean_pd": float(np.mean(
                            baseline_rows["independent_deflection"]
                        )),
                        "all_scheduled_mean_pd": float(np.mean(all_pd)),
                        "all_scheduled_worst_pd": float(np.min(all_pd)),
                    })
    payload = {
        "gate": "G2",
        "summary": {
            "proposed_mean_pd": float(np.mean([
                row["proposed_mean_pd"] for row in rows
            ])),
            "proposed_worst_pd": float(np.mean([
                row["proposed_worst_pd"] for row in rows
            ])),
            "pd_aware_mean_pd": float(np.mean([
                row["pd_aware_mean_pd"] for row in rows
            ])),
            "pd_aware_worst_pd": float(np.mean([
                row["pd_aware_worst_pd"] for row in rows
            ])),
            "sensing_snr_mean_pd": float(np.mean([
                row["sensing_snr_mean_pd"] for row in rows
            ])),
            "communication_mean_pd": float(np.mean([
                row["communication_mean_pd"] for row in rows
            ])),
            "independent_deflection_mean_pd": float(np.mean([
                row["independent_deflection_mean_pd"] for row in rows
            ])),
            "all_scheduled_mean_pd": float(np.mean([
                row["all_scheduled_mean_pd"] for row in rows
            ])),
            "mean_proposed_seconds": float(np.mean([
                row["proposed_seconds"] for row in rows
            ])),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g2_system_sweep_smoke.json"
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    run_sweep(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        seed_offset=args.seed_offset,
    )


if __name__ == "__main__":
    main()
