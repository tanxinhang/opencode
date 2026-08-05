"""G2 correlation sweep: value of conditional re-ranking vs rho.

For each rho in {0, 0.3, 0.5, 0.7, 0.85}, a strongly correlated report pair
is injected into the model.  We compare Conditional-Deflection Greedy,
Exact-P_D Greedy, Static Independent-Deflection Top-K, All-scheduled, and
(on a small Oracle subset) Exhaustive Oracle under a fair global budget.
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
from uav_otfs_isac.oracle import exhaustive_oracle
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select
from scripts.run_g2_system_sweep import _pd_vector, _score_topk
from scripts.run_g2_algorithm_negative_gates import (
    _correlated_models,
    _exact_pd_greedy,
)


def _jaccard_distance(conditional, independent, models):
    distances = []
    for q, model in enumerate(models):
        left = set(conditional[q]) - {model.owner}
        right = set(independent[q]) - {model.owner}
        union = left | right
        if not union:
            distances.append(0.0)
        else:
            distances.append(1.0 - len(left & right) / len(union))
    return float(np.mean(distances))


def _redundancy(models, scheduled):
    values = []
    for q, model in enumerate(models):
        selected = [
            i for i in scheduled[q]
            if i != model.owner
        ]
        if len(selected) < 2:
            continue
        diagonal = np.sqrt(np.diag(model.sigma0))
        correlation = (
            model.sigma0 / np.outer(diagonal, diagonal)
        )[np.ix_(selected, selected)]
        off_diagonal = correlation[~np.eye(len(selected), dtype=bool)]
        values.append(float(np.mean(np.abs(off_diagonal))))
    return float(np.mean(values)) if values else 0.0


def _bits_used(models, scheduled):
    return float(sum(
        int(model.report_bits[i])
        for model, selected in zip(models, scheduled)
        for i in selected if i != model.owner
    ))


def _paired_summary(rows, key):
    values = np.asarray([row[key] for row in rows])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _bootstrap_ci(values, seed, replicates=2000):
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


def run_sweep(*, output: Path, seeds: int, oracle_seeds: int) -> None:
    base = load_config("config/demo.yaml")
    rho_values = (0.0, 0.3, 0.5, 0.7, 0.85)
    rows = []
    oracle_rows = []
    for rho in rho_values:
        for budget in (20, 40):
            cfg = replace(
                base, num_uavs=8, num_targets=3,
                owners=[0, 3, 6],
            )
            qos = np.zeros(3)
            weights = np.ones(3)
            perf = np.ones(3)
            diffs = []
            win = 0
            for offset in range(seeds):
                seed = cfg.seed + offset
                models = _correlated_models(cfg, seed)
                models = _set_correlation(models, rho)
                conditional = greedy_select(
                    models, budget, qos, weights, perf, qos_first=False
                )
                conditional_pd = _pd_vector(
                    models, conditional.scheduled, cfg.false_alarm_rate
                )
                independent = _score_topk(
                    models, budget, "independent_deflection"
                )
                independent_pd = _pd_vector(
                    models, independent, cfg.false_alarm_rate
                )
                exact = _exact_pd_greedy(
                    models, budget, cfg.false_alarm_rate
                )
                exact_pd = _pd_vector(
                    models, exact, cfg.false_alarm_rate
                )
                all_scheduled = [
                    set(range(model.num_uavs)) for model in models
                ]
                all_pd = _pd_vector(
                    models, all_scheduled, cfg.false_alarm_rate
                )
                diff = float(
                    np.mean(conditional_pd) - np.mean(independent_pd)
                )
                diffs.append(diff)
                win += int(diff > 0.0)
                rows.append({
                    "rho": rho,
                    "budget_bits": budget,
                    "seed_offset": offset,
                    "conditional_mean_pd": float(np.mean(conditional_pd)),
                    "conditional_worst_pd": float(np.min(conditional_pd)),
                    "exact_pd_mean_pd": float(np.mean(exact_pd)),
                    "exact_pd_worst_pd": float(np.min(exact_pd)),
                    "independent_mean_pd": float(np.mean(independent_pd)),
                    "independent_worst_pd": float(np.min(independent_pd)),
                    "all_scheduled_mean_pd": float(np.mean(all_pd)),
                    "paired_diff": diff,
                    "jaccard_distance": _jaccard_distance(
                        conditional.scheduled, independent, models
                    ),
                    "conditional_redundancy": _redundancy(
                        models, conditional.scheduled
                    ),
                    "independent_redundancy": _redundancy(
                        models, independent
                    ),
                    "conditional_bits": _bits_used(
                        models, conditional.scheduled
                    ),
                    "independent_bits": _bits_used(models, independent),
                })
        # Oracle subset: smaller N/Q so exhaustive enumeration is feasible.
        cfg = replace(
            base, num_uavs=6, num_targets=2,
            owners=[0, 3],
        )
        qos = np.zeros(2)
        weights = np.ones(2)
        perf = np.ones(2)
        for budget in (8, 12):
            for offset in range(oracle_seeds):
                seed = cfg.seed + offset
                models = _correlated_models(cfg, seed)
                models = _set_correlation(models, rho)
                conditional = greedy_select(
                    models, budget, qos, weights, perf, qos_first=False
                )
                conditional_pd = _pd_vector(
                    models, conditional.scheduled, cfg.false_alarm_rate
                )
                exact = _exact_pd_greedy(
                    models, budget, cfg.false_alarm_rate
                )
                exact_pd = _pd_vector(
                    models, exact, cfg.false_alarm_rate
                )
                oracle = exhaustive_oracle(
                    models, budget, qos, weights, perf
                )
                oracle_pd = _pd_vector(
                    models, oracle.scheduled, cfg.false_alarm_rate
                )
                oracle_rows.append({
                    "rho": rho,
                    "budget_bits": budget,
                    "seed_offset": offset,
                    "conditional_mean_pd": float(np.mean(conditional_pd)),
                    "exact_pd_mean_pd": float(np.mean(exact_pd)),
                    "oracle_mean_pd": float(np.mean(oracle_pd)),
                    "oracle_gap_conditional": float(
                        np.mean(oracle_pd) - np.mean(conditional_pd)
                    ),
                    "oracle_gap_exact_pd": float(
                        np.mean(oracle_pd) - np.mean(exact_pd)
                    ),
                })
    summary = []
    for rho in rho_values:
        for budget in (20, 40):
            group = [row for row in rows if row["rho"] == rho
                     and row["budget_bits"] == budget]
            summary.append({
                "rho": rho,
                "budget_bits": budget,
                "conditional_mean_pd": _paired_summary(
                    group, "conditional_mean_pd"
                ),
                "exact_pd_mean_pd": _paired_summary(
                    group, "exact_pd_mean_pd"
                ),
                "independent_mean_pd": _paired_summary(
                    group, "independent_mean_pd"
                ),
                "paired_diff_mean": float(np.mean(
                    [row["paired_diff"] for row in group]
                )),
                "paired_diff_bootstrap_ci95": _bootstrap_ci(
                    np.asarray([row["paired_diff"] for row in group]),
                    seed=20260805,
                ),
                "conditional_win_rate": float(np.mean([
                    row["paired_diff"] > 0.0 for row in group
                ])),
                "jaccard_distance": _paired_summary(
                    group, "jaccard_distance"
                ),
                "conditional_redundancy": _paired_summary(
                    group, "conditional_redundancy"
                ),
                "independent_redundancy": _paired_summary(
                    group, "independent_redundancy"
                ),
                "conditional_worst_pd": _paired_summary(
                    group, "conditional_worst_pd"
                ),
                "independent_worst_pd": _paired_summary(
                    group, "independent_worst_pd"
                ),
                "conditional_bits": _paired_summary(
                    group, "conditional_bits"
                ),
                "independent_bits": _paired_summary(
                    group, "independent_bits"
                ),
            })
    oracle_summary = []
    for rho in rho_values:
        for budget in (8, 12):
            group = [row for row in oracle_rows if row["rho"] == rho
                     and row["budget_bits"] == budget]
            oracle_summary.append({
                "rho": rho,
                "budget_bits": budget,
                "oracle_gap_conditional": _paired_summary(
                    group, "oracle_gap_conditional"
                ),
                "oracle_gap_exact_pd": _paired_summary(
                    group, "oracle_gap_exact_pd"
                ),
                "oracle_mean_pd": _paired_summary(
                    group, "oracle_mean_pd"
                ),
            })
    payload = {
        "gate": "G2-correlation-sweep",
        "rho_values": list(rho_values),
        "summary": summary,
        "oracle_summary": oracle_summary,
        "rows": rows,
        "oracle_rows": oracle_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "rho_sweep": summary,
        "oracle_gap": oracle_summary,
    }, indent=2))


def _set_correlation(models, rho):
    """Override the top-SNR pair correlation to rho."""
    from dataclasses import replace as _replace
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
        sigma0[first, second] = sigma0[second, first] = rho * scale0
        sigma1[first, second] = sigma1[second, first] = rho * scale1
        output.append(_replace(model, sigma0=sigma0, sigma1=sigma1))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g2_correlation_sweep.json"
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--oracle-seeds", type=int, default=10)
    args = parser.parse_args()
    run_sweep(
        output=Path(args.output),
        seeds=args.seeds,
        oracle_seeds=args.oracle_seeds,
    )


if __name__ == "__main__":
    main()
