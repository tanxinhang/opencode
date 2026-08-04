"""Paired G0-B comparison on identical imperfect path-candidate scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multistatic_g0b import evaluate_trial


METHODS = (
    "position_dbscan",
    "angle_position_dbscan",
    "identity_dbscan",
    "gated_identity_dbscan",
    "conflict_aware_dbscan",
    "bic_conflict",
    "geometry_doppler",
)


def run_comparison(
    trials: int = 100,
    seed: int = 20260803,
    transmitters: int = 8,
    targets: int = 6,
    scenario: str = "separated",
    clutter_model: str = "diffuse",
    view_false_target_probability: float = 0.1,
    geometry_mode: str = "independent_uniform",
    confidence_model: str = "separated",
    collision_gate_mode: str = "hard_null",
    score_calibrator=None,
    collision_support_calibrators=None,
    collision_statistic_thresholds=None,
    physics_collision_threshold=None,
    physics_frame_thresholds=None,
    physics_stepdown_thresholds=None,
    robust_final_velocity=False,
    physics_stepdown_activation_count=1,
    physics_glrt_refinement_iterations=0,
    physics_cascade_thresholds=None,
    final_joint_refinement_iterations=3,
    covariance_weighted_final_state=False,
    physics_conformal_null=None,
    physics_conformal_p_threshold=None,
) -> dict:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rows_by_method = {method: [] for method in METHODS}
    # Reinitialize each method with the same seed. evaluate_trial consumes only
    # scene/candidate RNG; association algorithms are deterministic.
    for method in METHODS:
        rng = np.random.default_rng(seed)
        rows_by_method[method] = [evaluate_trial(
            rng, transmitters, targets, 0.08, 0.4, True, method, scenario,
            clutter_model, view_false_target_probability, geometry_mode,
            confidence_model, collision_gate_mode, score_calibrator,
            collision_support_calibrators,
            collision_statistic_thresholds,
            physics_collision_threshold,
            physics_frame_thresholds,
            physics_stepdown_thresholds,
            robust_final_velocity,
            physics_stepdown_activation_count,
            physics_glrt_refinement_iterations,
            physics_cascade_thresholds,
            final_joint_refinement_iterations,
            covariance_weighted_final_state,
            physics_conformal_null,
            physics_conformal_p_threshold,
        ) for _ in range(trials)]

    summaries = []
    for method in METHODS:
        rows = rows_by_method[method]
        matched = sum(row["matched_targets"] for row in rows)
        summaries.append({
            "method": method,
            "target_count_accuracy": float(np.mean([
                row["target_count_correct"] for row in rows
            ])),
            "target_recall": float(np.mean([row["target_recall"] for row in rows])),
            "scene_exact_recovery": float(np.mean([
                row["scene_exact_recovery"] for row in rows
            ])),
            "position_set_exact_15m": float(np.mean([
                row["position_set_exact_15m"] for row in rows
            ])),
            "position_velocity_state_exact": float(np.mean([
                row["position_velocity_state_exact"] for row in rows
            ])),
            "mean_gospa_15m_p2": float(np.mean([
                row["gospa_15m_p2"] for row in rows
            ])),
            "path_association_precision": float(np.mean([
                row["path_association_precision"] for row in rows
            ])),
            "path_association_recall": float(np.mean([
                row["path_association_recall"] for row in rows
            ])),
            "path_association_f1": float(np.mean([
                row["path_association_f1"] for row in rows
            ])),
            "identity_association_accuracy": float(
                sum(row["correct_identity_associations"] for row in rows)
                / max(sum(row["grouped_true_candidates"] for row in rows), 1)
            ),
            "position_rmse_m": float(np.sqrt(
                sum(row["position_squared_error_sum"] for row in rows)
                / max(matched, 1)
            )),
            "mean_velocity_error_mps": float(
                sum(row["velocity_error_sum"] for row in rows) / max(matched, 1)
            ),
            "mean_time_ms": 1000.0 * float(np.mean([
                row["association_time_s"] for row in rows
            ])),
            "p95_time_ms": 1000.0 * float(np.percentile([
                row["association_time_s"] for row in rows
            ], 95)),
            "over_count_rate": float(np.mean([
                row["estimated_targets"] > targets for row in rows
            ])),
            "under_count_rate": float(np.mean([
                row["estimated_targets"] < targets for row in rows
            ])),
        })

    proposed = rows_by_method["bic_conflict"]
    paired = {}
    for method in METHODS:
        if method == "bic_conflict":
            continue
        baseline = rows_by_method[method]
        differences = np.asarray([
            proposed_row["scene_exact_recovery"] - baseline_row["scene_exact_recovery"]
            for proposed_row, baseline_row in zip(proposed, baseline)
        ])
        paired[method] = {
            "mean_scene_exact_gain_pp": 100.0 * float(np.mean(differences)),
            "proposed_wins": int(np.sum(differences > 0)),
            "baseline_wins": int(np.sum(differences < 0)),
            "ties": int(np.sum(differences == 0)),
        }
    return {
        "scope": (
            "paired unknown-N path-to-target association comparison on the "
            "same synthetic imperfect candidates; not an OTFS front-end comparison"
        ),
        "M": transmitters,
        "N": targets,
        "trials": trials,
        "scenario": scenario,
        "clutter_model": clutter_model,
        "view_false_target_probability": view_false_target_probability,
        "geometry_mode": geometry_mode,
        "confidence_model": confidence_model,
        "collision_gate_mode": collision_gate_mode,
        "seed": seed,
        "metric_definitions": {
            "position_set_exact_15m": "correct cardinality and one-to-one position matches within 15 m",
            "position_velocity_state_exact": "position-set exact plus every matched velocity error <= 1 m/s",
            "gospa_15m_p2": "GOSPA with p=2, cutoff=15 m, alpha=2",
            "path_association_precision": "correctly grouped true paths / all grouped candidates",
            "path_association_recall": "correctly grouped true paths / all retained true path candidates"
        },
        "summaries": summaries,
        "paired_proposed_bic_minus_baseline": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--targets", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--scenario", choices=(
            "separated", "paired_collision", "single_pair_collision",
            "two_pair_collision"
        ),
        default="separated",
    )
    parser.add_argument(
        "--clutter-model", choices=("diffuse", "correlated_sidelobes"),
        default="diffuse",
    )
    parser.add_argument("--view-false-target-probability", type=float, default=0.1)
    parser.add_argument(
        "--geometry-mode", choices=("independent_uniform", "nested_12"),
        default="independent_uniform",
    )
    parser.add_argument(
        "--confidence-model", choices=("separated", "overlap"),
        default="separated",
    )
    parser.add_argument(
        "--collision-gate-mode",
        choices=("hard_null", "posterior_support", "empirical_null",
                 "physics_glrt", "physics_conformal", "physics_frame_stratified",
                 "physics_stepdown"),
        default="hard_null",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/multistatic_baseline_comparison.json"),
    )
    args = parser.parse_args()
    payload = run_comparison(
        args.trials, args.seed, args.transmitters, args.targets, args.scenario,
        args.clutter_model, args.view_false_target_probability,
        args.geometry_mode, args.confidence_model, args.collision_gate_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
