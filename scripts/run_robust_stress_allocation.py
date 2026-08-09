"""Nominal versus worst-scenario exact chance-constrained allocation.

Each target is represented by two physical scenario models: a clean scene
and a combined INR/BSC/erasure/mobility stress.  The robust DP optimizes the
maximum scenario-weighted violation excess; the nominal DP sees only the
clean scenario.  The same nominal schedule is then evaluated on both
scenarios so the conservative gain is measured on a common worst-case scale.
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

from uav_otfs_isac.config import load_config
from uav_otfs_isac.robust_baselines import (
    evaluate_robust_schedule_worst_violation,
    robust_greedy_worst_case,
    worst_case_communication_top_k,
    worst_case_independent_post_top_k,
    worst_case_no_cooperation,
    worst_case_random_top_k,
    worst_case_sensing_top_k,
)
from uav_otfs_isac.robust_portfolio import (
    enumerate_robust_target_portfolios,
    optimize_robust_chance_constrained_portfolio,
    select_scenario_options,
)
from uav_otfs_isac.robustness_stress import StressProfile, build_stress_models


def worst_excess_for_schedule(
    option_groups,
    scheduled,
    target_weights,
    violation_limits,
) -> float:
    excesses = []
    scenario_count = len(option_groups[0][0].scenario_violations)
    for scenario in range(scenario_count):
        total = 0.0
        for q, group in enumerate(option_groups):
            option = next(
                option for option in group
                if option.scheduled == scheduled[q]
            )
            total += target_weights[q] * max(
                option.scenario_violations[scenario] - violation_limits[q],
                0.0,
            )
        excesses.append(total)
    return float(max(excesses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robust_stress_allocation.json")
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument("--qos-target", type=float, default=0.7)
    parser.add_argument("--violation-limit", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = np.asarray(cfg.qos_weights, dtype=float)
    limits = np.full(cfg.num_targets, args.violation_limit)
    minimum_quality = np.full(cfg.num_targets, args.qos_target)
    profiles = [
        StressProfile("clean"),
        StressProfile(
            "combined",
            interference_sources=((60.0, -20.0, 0.0),),
            interference_reference_inr=(0.01,),
            bit_flip_probability=0.05,
            success_probability_scale=0.9,
            mobility_std=2.0,
        ),
    ]
    summaries = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        clean = build_stress_models(cfg, seed, profiles[0])
        combined = build_stress_models(cfg, seed, profiles[1])
        scenario_groups = [
            [clean[q], combined[q]] for q in range(cfg.num_targets)
        ]
        option_groups = [
            enumerate_robust_target_portfolios(
                scenarios,
                args.qos_target,
                float(weights[q]),
                0.9,
                1.0,
                quality_mode="gaussian_pd",
                false_alarm_rate=cfg.false_alarm_rate,
            )
            for q, scenarios in enumerate(scenario_groups)
        ]
        for budget in args.budgets:
            nominal = optimize_robust_chance_constrained_portfolio(
                scenario_groups,
                budget_bits=budget,
                minimum_quality=minimum_quality,
                target_weights=weights,
                violation_limits=limits,
                quality_mode="gaussian_pd",
                false_alarm_rate=cfg.false_alarm_rate,
                option_groups=select_scenario_options(option_groups, 0),
            )
            robust = optimize_robust_chance_constrained_portfolio(
                scenario_groups,
                budget_bits=budget,
                minimum_quality=minimum_quality,
                target_weights=weights,
                violation_limits=limits,
                quality_mode="gaussian_pd",
                false_alarm_rate=cfg.false_alarm_rate,
            )
            baselines = [
                worst_case_sensing_top_k(
                    scenario_groups,
                    budget,
                    weights,
                    limits,
                    minimum_quality=minimum_quality,
                    false_alarm_rate=cfg.false_alarm_rate,
                    option_groups=option_groups,
                ),
                worst_case_communication_top_k(
                    scenario_groups,
                    budget,
                    weights,
                    limits,
                    minimum_quality=minimum_quality,
                    false_alarm_rate=cfg.false_alarm_rate,
                    option_groups=option_groups,
                ),
                worst_case_independent_post_top_k(
                    scenario_groups,
                    budget,
                    weights,
                    limits,
                    minimum_quality=minimum_quality,
                    false_alarm_rate=cfg.false_alarm_rate,
                    option_groups=option_groups,
                ),
                worst_case_random_top_k(
                    scenario_groups,
                    budget,
                    weights,
                    limits,
                    minimum_quality=minimum_quality,
                    false_alarm_rate=cfg.false_alarm_rate,
                    option_groups=option_groups,
                ),
                worst_case_no_cooperation(
                    scenario_groups,
                    weights,
                    limits,
                    minimum_quality=minimum_quality,
                    false_alarm_rate=cfg.false_alarm_rate,
                    option_groups=option_groups,
                ),
                robust_greedy_worst_case(
                    scenario_groups,
                    budget,
                    weights,
                    limits,
                    minimum_quality=minimum_quality,
                    false_alarm_rate=cfg.false_alarm_rate,
                    option_groups=option_groups,
                ),
            ]
            nominal_worst = worst_excess_for_schedule(
                option_groups,
                nominal.selection.scheduled,
                weights,
                limits,
            )
            nominal_worst_violation = (
                evaluate_robust_schedule_worst_violation(
                    option_groups, nominal.selection.scheduled
                )
            )
            robust_worst_violation = robust.worst_violation_probability_per_target
            summaries.append({
                "budget_bits": budget,
                "seed": seed,
                "nominal_worst_excess": nominal_worst,
                "robust_worst_excess": robust.worst_weighted_violation_excess,
                "nominal_mean_worst_violation_probability": float(
                    np.mean(nominal_worst_violation)
                ),
                "robust_mean_worst_violation_probability": float(
                    np.mean(robust_worst_violation)
                ),
                "nominal_worst_violation_probability_per_target": (
                    list(nominal_worst_violation)
                ),
                "robust_worst_violation_probability_per_target": (
                    robust_worst_violation.tolist()
                ),
                "excess_reduction": nominal_worst
                - robust.worst_weighted_violation_excess,
                "robust_feasible": robust.feasible,
                "baselines": {
                    baseline.name: {
                        "worst_excess": baseline.worst_excess,
                        "mean_worst_violation_probability": float(np.mean(
                            baseline.worst_violation_probability_per_target
                        )),
                        "worst_violation_probability_per_target": list(
                            baseline.worst_violation_probability_per_target
                        ),
                    }
                    for baseline in baselines
                },
            })

    payload = {
        "gate": "robust-stress-allocation",
        "config": args.config,
        "profiles": [profile.label for profile in profiles],
        "qos_target": args.qos_target,
        "violation_limit": args.violation_limit,
        "scenario_count": len(profiles),
        "summary": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
