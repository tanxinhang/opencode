"""G4 gate: expected-P_D greedy selection under the exact reception law.

The existing selectors optimize expected deflection or deterministic P_D.
Under tight bit budgets and correlated erasures, the honest objective is the
expected P_D over the exact post-communication reception law, evaluated with
the Gate G3 monotone fusion family.  This gate compares the new
expected-P_D greedy against the conditional-deflection greedy, deterministic
P_D greedy, Static ID Top-K, and All-scheduled, and audits the bounded-regime
submodularity and the empirical approximation ratio against an exhaustive
single-target oracle.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import (
    expected_gaussian_detection_probability,
    expected_pd_greedy_select,
    pd_inflection_condition,
)
from uav_otfs_isac.fusion import optimal_deflection
from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.reliability import with_common_state_erasures
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select
from scripts.run_g2_system_sweep import _score_topk, pd_greedy_select


def expected_pd_vector(models, scheduled, false_alarm_rate, grid):
    return np.asarray([
        expected_gaussian_detection_probability(
            model, scheduled[q], false_alarm_rate, pd_mode="optimal",
            grid=grid,
        )
        for q, model in enumerate(models)
    ])


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


def bounded_regime_model(rng, variance_ratio, correlated_strength=0.6):
    n = 5
    delta = rng.uniform(1.4, 2.6, n)
    sigma0 = np.diag(rng.uniform(0.8, 1.6, n))
    sigma1 = variance_ratio * sigma0
    model = TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(n),
        mu1=delta,
        sigma0=sigma0,
        sigma1=sigma1,
        success_prob=np.concatenate(([1.0], rng.uniform(0.85, 0.98, n - 1))),
        report_bits=np.array([0, 1, 1, 1, 1], dtype=int),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )
    model.validate()
    return with_common_state_erasures([model], correlated_strength)[0]


def submodularity_audit(rng, variance_ratio, false_alarm_rate, grid, instances):
    violations = 0
    tested = 0
    conditional_violations = 0
    conditional_tested = 0
    for _ in range(instances):
        model = bounded_regime_model(rng, variance_ratio)
        candidates = list(range(1, model.num_uavs))
        cache = {}

        def value(selected):
            key = frozenset(selected)
            if key not in cache:
                cache[key] = expected_gaussian_detection_probability(
                    model, key, false_alarm_rate, grid=grid
                )
            return cache[key]

        def inflection_ok(selected):
            return all(
                pd_inflection_condition(
                    optimal_deflection(model.delta, model.sigma0, received),
                    variance_ratio, false_alarm_rate,
                )
                for received, _ in _received_patterns_for_audit(
                    model, selected
                )
            )

        for size in range(3):
            for base in combinations(candidates, size):
                base = set(base)
                remaining = [x for x in candidates if x not in base]
                for extra_count in range(1, len(remaining)):
                    for extra in combinations(remaining, extra_count):
                        larger = set(extra) | base
                        for candidate in candidates:
                            if candidate in larger:
                                continue
                            marginal_base = (
                                value(base | {candidate}) - value(base)
                            )
                            marginal_larger = (
                                value(larger | {candidate}) - value(larger)
                            )
                            tested += 1
                            ok = inflection_ok(base | {candidate})
                            ok = ok and inflection_ok(larger | {candidate})
                            if ok:
                                conditional_tested += 1
                            if marginal_base < marginal_larger - 1e-7:
                                violations += 1
                                if ok:
                                    conditional_violations += 1
    return {
        "tested_edges": tested,
        "violations": violations,
        "conditional_tested_edges": conditional_tested,
        "conditional_violations": conditional_violations,
    }


def _received_patterns_for_audit(model, selected):
    from uav_otfs_isac.risk import received_pattern_distribution

    return received_pattern_distribution(model, selected)


def approximation_ratio_audit(rng, false_alarm_rate, grid, instances):
    ratios = []
    for _ in range(instances):
        model = bounded_regime_model(rng, 1.0)
        candidates = list(range(1, model.num_uavs))
        budget = 3
        oracle = 0.0
        for mask in range(1 << len(candidates)):
            selected = {model.owner}
            cost = 0
            for index, candidate in enumerate(candidates):
                if mask & (1 << index):
                    selected.add(candidate)
                    cost += 1
            if cost > budget:
                continue
            oracle = max(oracle, expected_gaussian_detection_probability(
                model, selected, false_alarm_rate, grid=grid
            ))
        if oracle < 0.5:
            continue
        greedy = float(expected_pd_greedy_select(
            [model], budget, false_alarm_rate, grid=grid
        ).expected_pd[0])
        ratios.append(float(greedy / oracle))
    if not ratios:
        return {"instances": 0}
    return {
        "instances": len(ratios),
        "mean_ratio": float(np.mean(ratios)),
        "min_ratio": float(np.min(ratios)),
        "above_one_minus_one_over_e": float(np.mean([
            ratio >= 1.0 - 1.0 / np.e - 1e-9 for ratio in ratios
        ])),
    }


def run_gate(
    *, output: Path, seeds: int, budgets, strength: float, grid: int,
    audit_instances: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    rows = []
    for budget in budgets:
        for offset in range(seeds):
            truth = with_common_state_erasures(
                build_models(cfg, np.random.default_rng(cfg.seed + offset)),
                strength,
            )
            proposed = greedy_select(
                truth, budget, np.asarray(cfg.qos_min_deflection),
                qos_weights, np.ones(cfg.num_targets), qos_first=True,
            )
            expected = expected_pd_greedy_select(
                truth, budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            pd_aware = pd_greedy_select(
                truth, budget, qos_pd, qos_weights, false_alarm_rate,
            )
            topk = _score_topk(truth, budget, "independent_deflection")
            all_scheduled = [
                set(range(model.num_uavs)) for model in truth
            ]
            proposed_vector = expected_pd_vector(
                truth, proposed.scheduled, false_alarm_rate, grid
            )
            expected_vector = expected_pd_vector(
                truth, expected.scheduled, false_alarm_rate, grid
            )
            hybrid = (
                expected.scheduled
                if float(np.mean(expected_vector)) >= float(np.mean(proposed_vector))
                else proposed.scheduled
            )
            methods = {
                "expected_pd_greedy": expected.scheduled,
                "proposed": proposed.scheduled,
                "hybrid": hybrid,
                "pd_greedy": pd_aware,
                "independent_topk": topk,
                "all_scheduled": all_scheduled,
            }
            for name, scheduled in methods.items():
                vector = expected_pd_vector(
                    truth, scheduled, false_alarm_rate, grid
                )
                rows.append({
                    "budget_bits": budget,
                    "seed_offset": offset,
                    "method": name,
                    "mean_expected_pd": float(np.mean(vector)),
                    "worst_expected_pd": float(np.min(vector)),
                })

    summary = []
    for budget in budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        expected_group = [row for row in group if row["method"] == "expected_pd_greedy"]
        proposed_group = [row for row in group if row["method"] == "proposed"]
        topk_group = [row for row in group if row["method"] == "independent_topk"]
        all_group = [row for row in group if row["method"] == "all_scheduled"]
        hybrid_group = [row for row in group if row["method"] == "hybrid"]
        expected = float(np.mean([
            row["mean_expected_pd"] for row in expected_group
        ])) if expected_group else 0.0
        proposed = float(np.mean([
            row["mean_expected_pd"] for row in proposed_group
        ])) if proposed_group else 0.0
        topk = float(np.mean([
            row["mean_expected_pd"] for row in topk_group
        ])) if topk_group else 0.0
        all_value = float(np.mean([
            row["mean_expected_pd"] for row in all_group
        ])) if all_group else 0.0
        hybrid_value = float(np.mean([
            row["mean_expected_pd"] for row in hybrid_group
        ])) if hybrid_group else 0.0
        hybrid_worst = float(np.mean([
            row["worst_expected_pd"] for row in hybrid_group
        ])) if hybrid_group else 0.0
        expected_worst = float(np.mean([
            row["worst_expected_pd"] for row in expected_group
        ]))
        proposed_worst = float(np.mean([
            row["worst_expected_pd"] for row in proposed_group
        ]))
        mean_gains = [
            expected_row["mean_expected_pd"] - proposed_row["mean_expected_pd"]
            for expected_row, proposed_row in zip(expected_group, proposed_group)
        ]
        worst_gains = [
            expected_row["worst_expected_pd"] - proposed_row["worst_expected_pd"]
            for expected_row, proposed_row in zip(expected_group, proposed_group)
        ]
        hybrid_gains = [
            hybrid_row["mean_expected_pd"] - proposed_row["mean_expected_pd"]
            for hybrid_row, proposed_row in zip(hybrid_group, proposed_group)
        ]
        hybrid_worst_gains = [
            hybrid_row["worst_expected_pd"] - proposed_row["worst_expected_pd"]
            for hybrid_row, proposed_row in zip(hybrid_group, proposed_group)
        ]
        summary.append({
            "budget_bits": budget,
            "expected_pd_mean": expected,
            "proposed_mean": proposed,
            "topk_mean": topk,
            "all_scheduled_mean": all_value,
            "expected_pd_worst": expected_worst,
            "proposed_worst": proposed_worst,
            "mean_gain_vs_proposed": float(np.mean(mean_gains)),
            "mean_gain_bootstrap_ci95": _bootstrap_ci(
                mean_gains, seed=20260805
            ),
            "worst_gain_vs_proposed": float(np.mean(worst_gains)),
            "win_rate_vs_proposed": float(np.mean([
                gain > 1e-6 for gain in mean_gains
            ])),
            "hybrid_mean": hybrid_value,
            "hybrid_worst": hybrid_worst,
            "hybrid_gain_vs_proposed": float(np.mean(hybrid_gains)),
            "hybrid_worst_gain_vs_proposed": float(np.mean(hybrid_worst_gains)),
            "hybrid_win_rate_vs_proposed": float(np.mean([
                gain > 1e-6 for gain in hybrid_gains
            ])),
            "gap_to_all_scheduled": all_value - expected,
        })

    rng_audit = np.random.default_rng(20260805)
    submodular_c1 = submodularity_audit(
        rng_audit, 1.0, false_alarm_rate, grid, instances=audit_instances
    )
    submodular_c05 = submodularity_audit(
        rng_audit, 0.5, false_alarm_rate, grid, instances=audit_instances
    )
    ratio_audit = approximation_ratio_audit(
        rng_audit, false_alarm_rate, grid, instances=audit_instances
    )

    payload = {
        "gate": "G4-expected-pd-greedy",
        "strength": strength,
        "grid": grid,
        "summary": summary,
        "submodularity": {
            "variance_ratio_1_0": submodular_c1,
            "variance_ratio_0_5": submodular_c05,
        },
        "approximation_ratio": ratio_audit,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": summary,
        "submodularity": payload["submodularity"],
        "approximation_ratio": ratio_audit,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/expected_pd_greedy_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--audit-instances", type=int, default=12)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        strength=args.strength,
        grid=args.grid,
        audit_instances=args.audit_instances,
    )


if __name__ == "__main__":
    main()
