from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.fusion import gaussian_detection_probability, optimal_deflection
from uav_otfs_isac.risk import optimize_chance_constrained_portfolio
from uav_otfs_isac.scenario import build_models


def subsets(model):
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    for mask in range(1 << len(candidates)):
        received = {model.owner}
        received.update(candidates[j] for j in range(len(candidates)) if mask & (1 << j))
        yield received


def isotonic_fit(x, y):
    order = np.argsort(x); x = np.asarray(x)[order]; y = np.asarray(y)[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y); counts = np.bincount(inverse)
    means = sums / counts
    blocks = [[i, i, counts[i], means[i]] for i in range(len(unique_x))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][3] <= blocks[i + 1][3] + 1e-15:
            i += 1; continue
        left, right = blocks[i], blocks[i + 1]
        count = left[2] + right[2]
        value = (left[2] * left[3] + right[2] * right[3]) / count
        blocks[i:i + 2] = [[left[0], right[1], count, value]]
        i = max(i - 1, 0)
    fitted = np.empty(len(unique_x))
    for start, end, _, value in blocks:
        fitted[start:end + 1] = value
    return unique_x, fitted


def inverse_isotonic(x, fitted, target):
    indices = np.flatnonzero(fitted >= target)
    return float(x[indices[0]]) if indices.size else float(x[-1] + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--output", default="results/deflection_calibration_study.json")
    parser.add_argument("--train-seeds", type=int, default=50)
    parser.add_argument("--test-seeds", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--minimum-pd", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args(); cfg = load_config(args.config)
    alpha = cfg.false_alarm_rate
    theory_threshold = np.square(
        norm.ppf(np.asarray(args.minimum_pd)) + norm.ppf(1.0 - alpha)
    )

    # Equal-covariance control: mapped D and P_D define identical violations.
    control_edges = 0; control_mismatches = 0; control_rows = []
    for offset in range(min(args.train_seeds, 30)):
        models = build_models(cfg, np.random.default_rng(cfg.seed + offset))
        equal_models = [replace(model, sigma1=model.sigma0.copy()) for model in models]
        for q, model in enumerate(equal_models):
            for received in subsets(model):
                d = optimal_deflection(model.delta, model.sigma0, received)
                pd = gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1, received, alpha)
                control_edges += 1
                control_mismatches += int((d < theory_threshold[q]) != (pd < args.minimum_pd[q]))
        for budget in args.budgets:
            limits = np.full(cfg.num_targets, args.epsilon)
            d_result = optimize_chance_constrained_portfolio(
                equal_models, budget, theory_threshold, cfg.qos_weights, limits,
                quality_mode="deflection", false_alarm_rate=alpha)
            pd_result = optimize_chance_constrained_portfolio(
                equal_models, budget, args.minimum_pd, cfg.qos_weights, limits,
                quality_mode="gaussian_pd", false_alarm_rate=alpha)
            control_rows.append({
                "budget_bits": budget,
                "primary_excess_difference": abs(
                    d_result.weighted_violation_excess - pd_result.weighted_violation_excess),
                "exact_schedule_match": d_result.portfolio.selection.scheduled == pd_result.portfolio.selection.scheduled,
            })

    # Train target-position-specific monotone D -> P_D mappings.
    train_x = [[] for _ in range(cfg.num_targets)]; train_y = [[] for _ in range(cfg.num_targets)]
    for offset in range(args.train_seeds):
        models = build_models(cfg, np.random.default_rng(cfg.seed + 1000 + offset))
        for q, model in enumerate(models):
            for received in subsets(model):
                train_x[q].append(optimal_deflection(model.delta, model.sigma0, received))
                train_y[q].append(gaussian_detection_probability(
                    model.mu0, model.mu1, model.sigma0, model.sigma1, received, alpha))
    calibrated_threshold = []
    for q in range(cfg.num_targets):
        x, fitted = isotonic_fit(train_x[q], train_y[q])
        calibrated_threshold.append(inverse_isotonic(x, fitted, args.minimum_pd[q]))

    test_rows = []; limits = np.full(cfg.num_targets, args.epsilon)
    for offset in range(args.test_seeds):
        seed = cfg.seed + 100000 + offset
        models = build_models(cfg, np.random.default_rng(seed))
        for budget in args.budgets:
            methods = {
                "uncalibrated_deflection": (cfg.qos_min_deflection, "deflection"),
                "theory_mapped_deflection": (theory_threshold, "deflection"),
                "isotonic_calibrated_deflection": (calibrated_threshold, "deflection"),
                "gaussian_pd": (args.minimum_pd, "gaussian_pd"),
            }
            for name, (thresholds, mode) in methods.items():
                result = optimize_chance_constrained_portfolio(
                    models, budget, thresholds, cfg.qos_weights, limits,
                    quality_mode=mode, false_alarm_rate=alpha)
                pd_violations = []
                for q, model in enumerate(models):
                    from uav_otfs_isac.risk import gaussian_pd_loss_distribution
                    pd_violations.append(gaussian_pd_loss_distribution(
                        model, result.portfolio.selection.scheduled[q],
                        args.minimum_pd[q], alpha).violation_probability())
                test_rows.append({"seed": seed, "budget_bits": budget, "method": name,
                                  "worst_pd_violation": max(pd_violations),
                                  "mean_pd_violation": float(np.mean(pd_violations))})
    summary = []
    for budget in args.budgets:
        for method in ("uncalibrated_deflection", "theory_mapped_deflection",
                       "isotonic_calibrated_deflection", "gaussian_pd"):
            group = [r for r in test_rows if r["budget_bits"] == budget and r["method"] == method]
            summary.append({"budget_bits": budget, "method": method,
                            "mean_worst_pd_violation": float(np.mean([r["worst_pd_violation"] for r in group])),
                            "mean_pd_violation": float(np.mean([r["mean_pd_violation"] for r in group]))})
    payload = {
        "equal_covariance_control": {
            "tested_received_sets": control_edges,
            "violation_event_mismatches": control_mismatches,
            "max_primary_excess_difference": max(r["primary_excess_difference"] for r in control_rows),
            "exact_schedule_match_rate": float(np.mean([r["exact_schedule_match"] for r in control_rows])),
        },
        "theory_threshold": theory_threshold.tolist(),
        "isotonic_calibrated_threshold": calibrated_threshold,
        "summary": summary, "test_instances": test_rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("equal_covariance_control", "theory_threshold", "isotonic_calibrated_threshold", "summary")}, indent=2))


if __name__ == "__main__":
    main()
