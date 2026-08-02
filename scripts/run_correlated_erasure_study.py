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
from uav_otfs_isac.reliability import (
    mean_off_diagonal_failure_correlation,
    with_common_state_erasures,
    with_grouped_common_state_erasures,
    alternating_failure_groups,
)
from uav_otfs_isac.risk import (
    gaussian_pd_loss_distribution,
    optimize_chance_constrained_portfolio,
)
from uav_otfs_isac.scenario import build_models


def evaluate_schedule(models, scheduled, minimum_pd, alpha):
    distributions = [
        gaussian_pd_loss_distribution(
            model, scheduled[q], minimum_pd[q], alpha
        )
        for q, model in enumerate(models)
    ]
    return {
        "violation": np.asarray([d.violation_probability() for d in distributions]),
        "mean_loss": np.asarray([d.mean for d in distributions]),
        "cvar": np.asarray([d.cvar(0.9) for d in distributions]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/correlated_erasure_study.json")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--failure-structure", choices=["global", "grouped"], default="global")
    args = parser.parse_args(); cfg = load_config(args.config)
    limits = np.full(cfg.num_targets, args.epsilon); rows = []
    for offset in range(args.seeds):
        seed = cfg.seed + offset
        independent_models = build_models(cfg, np.random.default_rng(seed))
        independent_solutions = {}
        for budget in args.budgets:
            independent_solutions[budget] = optimize_chance_constrained_portfolio(
                independent_models, budget, args.minimum_pd, cfg.qos_weights, limits,
                quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate)
        for strength in args.strengths:
            truth_models = (
                with_common_state_erasures(independent_models, strength)
                if args.failure_structure == "global"
                else with_grouped_common_state_erasures(
                    independent_models, strength,
                    alternating_failure_groups(independent_models, num_groups=2),
                )
            )
            marginal_error = max(
                float(np.max(np.abs(model.pattern_probabilities @ model.reception_patterns - model.success_prob)))
                for model in truth_models
            )
            mean_failure_correlation = float(np.mean([
                mean_off_diagonal_failure_correlation(model) for model in truth_models
            ]))
            for budget in args.budgets:
                independent_selection = independent_solutions[budget].portfolio.selection
                independent_prediction = evaluate_schedule(
                    independent_models, independent_selection.scheduled,
                    args.minimum_pd, cfg.false_alarm_rate
                )
                aware = optimize_chance_constrained_portfolio(
                    truth_models, budget, args.minimum_pd, cfg.qos_weights, limits,
                    quality_mode="gaussian_pd", false_alarm_rate=cfg.false_alarm_rate)
                all_scheduled = tuple(frozenset(range(model.num_uavs)) for model in truth_models)
                methods = {
                    "independent_assumption": independent_selection.scheduled,
                    "correlation_aware": aware.portfolio.selection.scheduled,
                    "all_scheduled": all_scheduled,
                }
                aware_pairs = {(q, i) for q, group in enumerate(methods["correlation_aware"])
                               for i in group if i != truth_models[q].owner}
                independent_pairs = {(q, i) for q, group in enumerate(methods["independent_assumption"])
                                     for i in group if i != truth_models[q].owner}
                union = aware_pairs | independent_pairs
                jaccard = len(aware_pairs & independent_pairs) / len(union) if union else 1.0
                for name, scheduled in methods.items():
                    metrics = evaluate_schedule(
                        truth_models, scheduled, args.minimum_pd, cfg.false_alarm_rate
                    )
                    rows.append({
                        "seed": seed, "strength": strength,
                        "mean_failure_correlation": mean_failure_correlation,
                        "maximum_marginal_error": marginal_error,
                        "budget_bits": budget, "method": name,
                        "mean_violation_probability": float(metrics["violation"].mean()),
                        "worst_target_violation_probability": float(metrics["violation"].max()),
                        "system_feasible": bool(np.all(metrics["violation"] <= limits + 1e-12)),
                        "weighted_mean_loss": float(np.asarray(cfg.qos_weights) @ metrics["mean_loss"]),
                        "weighted_cvar": float(np.asarray(cfg.qos_weights) @ metrics["cvar"]),
                        "independent_model_predicted_worst_violation": float(
                            independent_prediction["violation"].max()
                        ) if name == "independent_assumption" else None,
                        "worst_violation_optimism_gap": float(
                            metrics["violation"].max() - independent_prediction["violation"].max()
                        ) if name == "independent_assumption" else None,
                        "independent_aware_pair_jaccard": jaccard,
                    })
    summary = []
    for strength in args.strengths:
        for budget in args.budgets:
            for method in ("independent_assumption", "correlation_aware", "all_scheduled"):
                group = [r for r in rows if r["strength"] == strength
                         and r["budget_bits"] == budget and r["method"] == method]
                summary.append({
                    "strength": strength,
                    "mean_failure_correlation": float(np.mean([r["mean_failure_correlation"] for r in group])),
                    "budget_bits": budget, "method": method,
                    "mean_violation_probability": float(np.mean([r["mean_violation_probability"] for r in group])),
                    "mean_worst_target_violation_probability": float(np.mean([r["worst_target_violation_probability"] for r in group])),
                    "outer_system_feasible_rate": float(np.mean([r["system_feasible"] for r in group])),
                    "mean_weighted_loss": float(np.mean([r["weighted_mean_loss"] for r in group])),
                    "mean_weighted_cvar": float(np.mean([r["weighted_cvar"] for r in group])),
                    "mean_independent_aware_pair_jaccard": float(np.mean([r["independent_aware_pair_jaccard"] for r in group])),
                    "mean_worst_violation_optimism_gap": float(np.mean([
                        r["worst_violation_optimism_gap"] for r in group
                        if r["worst_violation_optimism_gap"] is not None
                    ])) if method == "independent_assumption" else None,
                    "maximum_marginal_error": float(np.max([r["maximum_marginal_error"] for r in group])),
                })
    payload = {"seeds": args.seeds, "failure_structure": args.failure_structure,
               "minimum_pd": args.minimum_pd,
               "epsilon": args.epsilon, "summary": summary, "instances": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
