from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.reliability import (
    alternating_failure_groups,
    grouped_failure_correlation,
    physical_failure_groups,
    with_grouped_common_state_erasures,
)
from uav_otfs_isac.risk import (
    attribute_failure_diversity_headroom,
    gaussian_pd_loss_distribution,
    optimize_chance_constrained_portfolio,
)
from uav_otfs_isac.scenario import build_models, uav_geometry


SCHEMES = (
    "alternating_proxy",
    "owner_angle_path",
    "formation_position",
    "link_midpoint",
)


def groups_for(scheme, models, positions, num_groups):
    if scheme == "alternating_proxy":
        return alternating_failure_groups(models, num_groups)
    return physical_failure_groups(models, positions, scheme, num_groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/physical_failure_domain_study.json")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--num-groups", type=int, default=2)
    args = parser.parse_args()
    cfg = load_config(args.config)
    limits = np.full(cfg.num_targets, args.epsilon)
    positions = uav_geometry(cfg.num_uavs)
    rows = []
    independent_baseline_rows = []
    maximum_marginal_error = 0.0
    gate_samples = []
    grouping_signatures = {}
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        independent = build_models(cfg, np.random.default_rng(seed))
        independent_solutions = {
            budget: optimize_chance_constrained_portfolio(
                independent, budget, args.minimum_pd, cfg.qos_weights, limits,
                quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate,
            ) for budget in args.budgets
        }
        for budget, solution in independent_solutions.items():
            scheduled = solution.portfolio.selection.scheduled
            violations = [
                gaussian_pd_loss_distribution(
                    model, scheduled[q], args.minimum_pd[q], cfg.false_alarm_rate
                ).violation_probability()
                for q, model in enumerate(independent)
            ]
            independent_baseline_rows.append({
                "seed": seed,
                "budget_bits": budget,
                "mean_violation_probability": float(np.mean(violations)),
                "worst_violation_probability": float(np.max(violations)),
                "system_feasible": bool(np.all(np.asarray(violations) <= limits + 1e-12)),
            })
        for scheme in SCHEMES:
            groups = groups_for(scheme, independent, positions, args.num_groups)
            grouping_signatures.setdefault(
                scheme, [[int(x) for x in group] for group in groups]
            )
            truth = with_grouped_common_state_erasures(
                independent, args.strength, groups
            )
            for model, labels in zip(truth, groups):
                marginals = model.pattern_probabilities @ model.reception_patterns
                maximum_marginal_error = max(
                    maximum_marginal_error,
                    float(np.max(np.abs(marginals - model.success_prob))),
                )
                gate_samples.append((scheme, *grouped_failure_correlation(model, labels)))
            for budget in args.budgets:
                independent_solution = independent_solutions[budget]
                aware = optimize_chance_constrained_portfolio(
                    truth, budget, args.minimum_pd, cfg.qos_weights, limits,
                    quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate,
                )
                independent_sets = independent_solution.portfolio.selection.scheduled
                aware_sets = aware.portfolio.selection.scheduled
                system_feasible = True
                for q, model in enumerate(truth):
                    attribution = attribute_failure_diversity_headroom(
                        independent[q], model, independent_sets[q], aware_sets[q],
                        budget, args.minimum_pd[q], groups[q],
                        false_alarm_rate=cfg.false_alarm_rate,
                    )
                    system_feasible &= attribution.aware_violation <= args.epsilon + 1e-12
                    rows.append({
                        "seed": seed, "scheme": scheme, "budget_bits": budget,
                        "target": q, "target_weight": float(cfg.qos_weights[q]),
                        "classification": attribution.classification,
                        "recoverable_headroom": attribution.recoverable_headroom,
                        "independent_violation": attribution.independent_violation,
                        "aware_violation": attribution.aware_violation,
                        "oracle_violation": attribution.oracle_violation,
                        "system_feasible": None,
                    })
                for row in rows[-cfg.num_targets:]:
                    row["system_feasible"] = bool(system_feasible)

    summary = []
    for scheme in SCHEMES:
        correlations = [item for item in gate_samples if item[0] == scheme]
        within = float(np.mean([item[1] for item in correlations]))
        between = float(np.mean([item[2] for item in correlations]))
        for budget in args.budgets:
            group = [row for row in rows if row["scheme"] == scheme and row["budget_bits"] == budget]
            counts = Counter(row["classification"] for row in group)
            denominator = sum(
                row["target_weight"] * row["recoverable_headroom"] for row in group
            )
            net = sum(
                row["target_weight"] * (row["independent_violation"] - row["aware_violation"])
                for row in group
            )
            positive = sum(
                row["target_weight"] * max(row["independent_violation"] - row["aware_violation"], 0.0)
                for row in group
            )
            summary.append({
                "scheme": scheme, "budget_bits": budget,
                "mean_within_group_failure_correlation": within,
                "mean_between_group_failure_correlation": between,
                "correlation_separation": within - between,
                "diversifiable_substitute_fraction": counts["diversifiable_substitute"] / len(group),
                "positive_headroom_fraction": float(np.mean([
                    row["recoverable_headroom"] > 1e-12 for row in group
                ])),
                "mean_recoverable_headroom": float(np.mean([
                    row["recoverable_headroom"] for row in group
                ])),
                "mean_aware_violation": float(np.mean([
                    row["aware_violation"] for row in group
                ])),
                "aggregate_headroom_capture_rate": net / denominator if denominator > 1e-12 else None,
                "positive_gain_capture_rate": positive / denominator if denominator > 1e-12 else None,
                "harm_fraction": float(np.mean([
                    row["aware_violation"] > row["independent_violation"] + 1e-12 for row in group
                ])),
                "outer_system_feasible_rate": float(np.mean([
                    row["system_feasible"] for row in group[::cfg.num_targets]
                ])),
            })
    midpoint_equivalent = grouping_signatures["formation_position"] == grouping_signatures["link_midpoint"]
    gates = {
        "physical_features_only": True,
        "maximum_marginal_error": maximum_marginal_error,
        "marginal_preservation_passed": maximum_marginal_error <= 1e-12,
        "within_exceeds_between": all(
            row["correlation_separation"] > 1e-6 for row in summary
        ),
        "midpoint_position_equivalent": midpoint_equivalent,
    }
    independent_summary = []
    for budget in args.budgets:
        group = [
            row for row in independent_baseline_rows
            if row["budget_bits"] == budget
        ]
        independent_summary.append({
            "budget_bits": budget,
            "mean_violation_probability": float(np.mean([
                row["mean_violation_probability"] for row in group
            ])),
            "mean_worst_violation_probability": float(np.mean([
                row["worst_violation_probability"] for row in group
            ])),
            "outer_system_feasible_rate": float(np.mean([
                row["system_feasible"] for row in group
            ])),
        })
    payload = {
        "seeds": args.seeds, "strength": args.strength,
        "num_groups": args.num_groups, "minimum_pd": args.minimum_pd,
        "epsilon": args.epsilon, "gates": gates,
        "grouping_signatures": grouping_signatures,
        "independent_erasure_baseline": independent_summary,
        "summary": summary, "instances": rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "gates": gates,
        "independent_erasure_baseline": independent_summary,
        "summary": summary,
    }, indent=2))
    if not gates["marginal_preservation_passed"]:
        raise SystemExit("C4 marginal-preservation gate failed")
    if not gates["within_exceeds_between"]:
        raise SystemExit("C4 correlation-separation gate failed")


if __name__ == "__main__":
    main()
