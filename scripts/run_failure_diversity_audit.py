from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.reliability import with_grouped_common_state_erasures
from uav_otfs_isac.risk import (
    gaussian_pd_loss_distribution,
    optimize_chance_constrained_portfolio,
)


GROUPS = np.array([-1, 0, 0, 1, 1])
SAME = frozenset({0, 1, 2})
CROSS = frozenset({0, 1, 3})


def received_pd(model, reports, alpha=0.05):
    return gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1,
        {model.owner, *reports}, alpha,
    )


def violation(model, scheduled, minimum_pd, alpha=0.05):
    return gaussian_pd_loss_distribution(
        model, scheduled, minimum_pd, alpha
    ).violation_probability()


def best_two_report_violation(model, minimum_pd, alpha=0.05):
    candidates = [1, 2, 3, 4]
    options = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            scheduled = frozenset({0, candidates[i], candidates[j]})
            options.append((violation(model, scheduled, minimum_pd, alpha), scheduled))
    return min(options, key=lambda item: (item[0], sorted(item[1])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/failure_diversity_audit.json")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--quality-deltas", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.32, 0.34, 0.36,
                                 0.38, 0.40, 0.50, 0.60])
    args = parser.parse_args()
    base = symmetric_diversity_model(success_probability=0.6)
    correlated = with_grouped_common_state_erasures([base], args.strength, GROUPS)[0]
    owner_pd = received_pd(base, [])
    one_pd = received_pd(base, [1])
    two_pd = received_pd(base, [1, 2])
    one_of_threshold = (owner_pd + one_pd) / 2.0
    two_of_threshold = (one_pd + two_pd) / 2.0

    independent_same = violation(base, SAME, one_of_threshold)
    independent_cross = violation(base, CROSS, one_of_threshold)
    correlated_same = violation(correlated, SAME, one_of_threshold)
    correlated_cross = violation(correlated, CROSS, one_of_threshold)
    chance = optimize_chance_constrained_portfolio(
        [correlated], 2, [one_of_threshold], [1.0], [0.0],
        quality_mode="gaussian_pd", false_alarm_rate=0.05,
    )
    oracle_violation, oracle_schedule = best_two_report_violation(
        correlated, one_of_threshold
    )
    selected = chance.portfolio.selection.scheduled[0]
    c0_pass = bool(
        np.isclose(independent_same, independent_cross)
        and correlated_cross < correlated_same - 1e-6
        and ({1, 2}.issubset(selected) is False)
        and oracle_violation < correlated_same - 1e-6
        and np.isclose(
            violation(correlated, selected, one_of_threshold), oracle_violation
        )
    )

    same_two = violation(correlated, SAME, two_of_threshold)
    cross_two = violation(correlated, CROSS, two_of_threshold)
    c1_pass = bool(correlated_cross < correlated_same and same_two < cross_two)

    c2_rows = []
    for quality_delta in args.quality_deltas:
        # Group A keeps quality 1+Delta; group B alternatives have quality 1-Delta.
        controlled = symmetric_diversity_model(
            report_delta=np.array([1.0 + quality_delta, 1.0 + quality_delta,
                                   1.0 - quality_delta, 1.0 - quality_delta]),
            success_probability=0.6,
        )
        truth = with_grouped_common_state_erasures([controlled], args.strength, GROUPS)[0]
        same_violation = violation(truth, SAME, one_of_threshold)
        cross_violation = violation(truth, CROSS, one_of_threshold)
        same_expected_pd = (
            1.0 - gaussian_pd_loss_distribution(truth, SAME, 1.0, 0.05).mean
        )
        cross_expected_pd = (
            1.0 - gaussian_pd_loss_distribution(truth, CROSS, 1.0, 0.05).mean
        )
        oracle_value, oracle_set = best_two_report_violation(truth, one_of_threshold)
        solution = optimize_chance_constrained_portfolio(
            [truth], 2, [one_of_threshold], [1.0], [0.0],
            quality_mode="gaussian_pd", false_alarm_rate=0.05,
        )
        selected_set = solution.portfolio.selection.scheduled[0]
        independent_choice = SAME
        independent_truth = violation(truth, independent_choice, one_of_threshold)
        chance_truth = violation(truth, selected_set, one_of_threshold)
        headroom = independent_truth - oracle_value
        use_ratio = None if headroom <= 1e-12 else (independent_truth - chance_truth) / headroom
        c2_rows.append({
            "quality_delta": quality_delta,
            "same_group_violation": same_violation,
            "cross_group_violation": cross_violation,
            "same_group_expected_pd": same_expected_pd,
            "cross_group_expected_pd": cross_expected_pd,
            "cross_group_sensing_cost": same_expected_pd - cross_expected_pd,
            "selected": sorted(selected_set),
            "oracle": sorted(oracle_set),
            "recoverable_headroom": headroom,
            "headroom_use_ratio": use_ratio,
        })

    payload = {
        "received_pd": {"owner": owner_pd, "one_report": one_pd, "two_reports": two_pd},
        "thresholds": {"one_of_two": one_of_threshold, "two_of_two": two_of_threshold},
        "C0": {
            "passed": c0_pass,
            "independent_same": independent_same,
            "independent_cross": independent_cross,
            "correlated_same": correlated_same,
            "correlated_cross": correlated_cross,
            "optimizer_schedule": sorted(selected),
            "oracle_schedule": sorted(oracle_schedule),
        },
        "C1": {
            "passed": c1_pass,
            "one_of_two_same": correlated_same,
            "one_of_two_cross": correlated_cross,
            "two_of_two_same": same_two,
            "two_of_two_cross": cross_two,
        },
        "C2": c2_rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not c0_pass:
        raise SystemExit("C0 failure-diversity gate failed")
    if not c1_pass:
        raise SystemExit("C1 threshold-polarity gate failed")


if __name__ == "__main__":
    main()
