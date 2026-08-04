"""G2 algorithm negative-result audit.

The conditional-deflection greedy, exact-PD greedy, and max-min PD greedy are
compared with static Top-K baselines under a fair global budget in a strongly
correlated system model.  This audit deliberately documents where the
candidate algorithms do *not* beat Top-K, so later work does not claim a
performance advantage that is not present.
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

from uav_otfs_isac.config import load_config
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select
from scripts.run_g2_system_sweep import _pd_vector, _score_topk


def _correlated_models(cfg, seed):
    models = build_models(cfg, np.random.default_rng(seed))
    output = []
    for model in models:
        sigma0 = model.sigma0.copy()
        sigma1 = model.sigma1.copy()
        candidates = [
            i for i in range(model.num_uavs) if i != model.owner
        ]
        scores = [
            model.delta[i] ** 2 / max(model.sigma0[i, i], 1e-12)
            for i in candidates
        ]
        order = np.argsort(scores)[::-1]
        first, second = candidates[order[0]], candidates[order[1]]
        scale0 = np.sqrt(sigma0[first, first] * sigma0[second, second])
        scale1 = np.sqrt(sigma1[first, first] * sigma1[second, second])
        sigma0[first, second] = sigma0[second, first] = 0.85 * scale0
        sigma1[first, second] = sigma1[second, first] = 0.85 * scale1
        output.append(replace(model, sigma0=sigma0, sigma1=sigma1))
    return output


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
            for i in range(model.num_uavs):
                if i == model.owner or i in scheduled[q]:
                    continue
                cost = int(model.report_bits[i])
                if used + cost > budget_bits:
                    continue
                new = gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1,
                    scheduled[q] | {i}, false_alarm_rate,
                )
                gain = max(new - current, 0.0) / max(cost, 1)
                if best is None or gain > best[0]:
                    best = (gain, q, i, cost)
        if best is None or best[0] <= 1e-14:
            break
        _, q, i, cost = best
        scheduled[q].add(i)
        used += cost
    return scheduled


def _maxmin_pd_greedy(models, budget_bits, false_alarm_rate):
    scheduled = [{model.owner} for model in models]
    used = 0

    def pd(q, model):
        return gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            scheduled[q], false_alarm_rate,
        )

    while True:
        best = None
        for q, model in enumerate(models):
            for i in range(model.num_uavs):
                if i == model.owner or i in scheduled[q]:
                    continue
                cost = int(model.report_bits[i])
                if used + cost > budget_bits:
                    continue
                scheduled[q].add(i)
                values = [pd(j, m) for j, m in enumerate(models)]
                scheduled[q].remove(i)
                key = (min(values), sum(values))
                if best is None or key > best[0]:
                    best = (key, q, i, cost)
        if best is None:
            break
        _, q, i, cost = best
        scheduled[q].add(i)
        used += cost
    return scheduled


def run_audit(*, output: Path, seeds: int, budgets, seed_offset: int) -> None:
    base = load_config("config/demo.yaml")
    rows = []
    for num_uavs in (8, 12):
        for num_targets in (3, 5):
            owners = [
                int(np.round((0.5 + q) * num_uavs / num_targets)) % num_uavs
                for q in range(num_targets)
            ]
            cfg = replace(
                base, num_uavs=num_uavs, num_targets=num_targets,
                owners=owners,
            )
            qos = np.zeros(num_targets)
            weights = np.ones(num_targets)
            perf = np.ones(num_targets)
            for budget in budgets:
                for offset in range(seeds):
                    seed = cfg.seed + offset + seed_offset
                    models = _correlated_models(cfg, seed)
                    proposed = greedy_select(
                        models, budget, qos, weights, perf, qos_first=False
                    )
                    proposed_pd = _pd_vector(
                        models, proposed.scheduled, cfg.false_alarm_rate
                    )
                    exact_pd = _exact_pd_greedy(
                        models, budget, cfg.false_alarm_rate
                    )
                    exact_pd_pd = _pd_vector(
                        models, exact_pd, cfg.false_alarm_rate
                    )
                    maxmin = _maxmin_pd_greedy(
                        models, budget, cfg.false_alarm_rate
                    )
                    maxmin_pd = _pd_vector(
                        models, maxmin, cfg.false_alarm_rate
                    )
                    independent = _score_topk(
                        models, budget, "independent_deflection"
                    )
                    independent_pd = _pd_vector(
                        models, independent, cfg.false_alarm_rate
                    )
                    rows.append({
                        "num_uavs": num_uavs,
                        "num_targets": num_targets,
                        "budget_bits": budget,
                        "seed_offset": offset,
                        "proposed_mean_pd": float(np.mean(proposed_pd)),
                        "proposed_worst_pd": float(np.min(proposed_pd)),
                        "exact_pd_mean_pd": float(np.mean(exact_pd_pd)),
                        "exact_pd_worst_pd": float(np.min(exact_pd_pd)),
                        "maxmin_mean_pd": float(np.mean(maxmin_pd)),
                        "maxmin_worst_pd": float(np.min(maxmin_pd)),
                        "independent_mean_pd": float(np.mean(independent_pd)),
                        "independent_worst_pd": float(np.min(independent_pd)),
                    })
    summary = {
        "proposed_mean_pd": float(np.mean([
            row["proposed_mean_pd"] for row in rows
        ])),
        "exact_pd_mean_pd": float(np.mean([
            row["exact_pd_mean_pd"] for row in rows
        ])),
        "maxmin_mean_pd": float(np.mean([
            row["maxmin_mean_pd"] for row in rows
        ])),
        "independent_mean_pd": float(np.mean([
            row["independent_mean_pd"] for row in rows
        ])),
        "proposed_beats_independent_mean": float(np.mean([
            row["proposed_mean_pd"] > row["independent_mean_pd"]
            for row in rows
        ])),
    }
    payload = {
        "gate": "G2-algorithm-negative-audit",
        "correlation_strength": 0.85,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g2_algorithm_negative_audit.json"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    run_audit(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        seed_offset=args.seed_offset,
    )


if __name__ == "__main__":
    main()
