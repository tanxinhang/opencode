"""Build submission-draft tables and figures from stored gate results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper_figures"


def load(name: str) -> dict:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def pp(value: float) -> str:
    return f"{100.0 * value:.2f} pp"


def ci_text(interval) -> str:
    return f"[{100.0 * interval[0]:.2f}, {100.0 * interval[1]:.2f}] pp"


def ms_text(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def pos_text(position) -> str:
    return "(" + ", ".join(str(float(x)) for x in position) + ")"


def build_rows() -> list[list[str]]:
    rows: list[list[str]] = []

    def add(gate: str, metric: str, value: str, source: str) -> None:
        rows.append([gate, metric, value, source])

    g3 = load("pd_optimal_fusion_gate.json")
    mono = g3["monotonicity"]
    gains = g3["gains"]
    prop = g3["proportional_regime"]
    greedy = g3["greedy"]
    add("G3", "Operating-point addition edges", str(mono["operating_edges"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Decreasing edges, P_D-optimal rule",
        str(mono["optimal_decreasing_edges"]), "pd_optimal_fusion_gate.json")
    add("G3", "Decreasing edges, deflection rule",
        str(mono["deflection_decreasing_edges"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Deflection-rule decreasing-edge rate",
        f"{100.0 * mono['deflection_decreasing_edge_rate']:.2f}%",
        "pd_optimal_fusion_gate.json")
    add("G3", "Maximum deflection-rule P_D drop",
        pp(mono["maximum_deflection_pd_drop"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Mean P_D gain over deflection rule per edge",
        pp(gains["mean_pd_gain_over_deflection"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Maximum P_D gain over deflection rule per edge",
        pp(gains["maximum_pd_gain_over_deflection"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Proportional-regime checks", str(prop["checks"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Proportional-regime max absolute error",
        f"{prop['maximum_absolute_error_vs_closed_form']:.2e}",
        "pd_optimal_fusion_gate.json")
    add("G3", "Greedy fusion gain on deflection schedule",
        pp(greedy["mean_fusion_gain_on_deflection_schedule"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Greedy scheduling gain under optimal rule",
        pp(greedy["mean_scheduling_gain_under_optimal_rule"]),
        "pd_optimal_fusion_gate.json")
    add("G3", "Greedy total mean gain",
        pp(greedy["mean_total_gain"]), "pd_optimal_fusion_gate.json")

    g4 = load("expected_pd_greedy_gate.json")
    for summary in g4["summary"]:
        budget = summary["budget_bits"]
        add("G4", f"B={budget} mean gain vs proposed",
            pp(summary["mean_gain_vs_proposed"]),
            "expected_pd_greedy_gate.json")
        add("G4", f"B={budget} mean gain bootstrap CI",
            ci_text(summary["mean_gain_bootstrap_ci95"]),
            "expected_pd_greedy_gate.json")
        add("G4", f"B={budget} worst-target gain vs proposed",
            pp(summary["worst_gain_vs_proposed"]),
            "expected_pd_greedy_gate.json")
        add("G4", f"B={budget} hybrid mean gain vs proposed",
            pp(summary["hybrid_gain_vs_proposed"]),
            "expected_pd_greedy_gate.json")
    sub = g4["submodularity"]
    for key, value in sub.items():
        add("G4", f"Submodularity edges ({key})", str(value["tested_edges"]),
            "expected_pd_greedy_gate.json")
        add("G4", f"Submodularity violations ({key})",
            str(value["violations"]), "expected_pd_greedy_gate.json")
    approx = g4["approximation_ratio"]
    add("G4", "Small-instance greedy ratio (mean)",
        f"{approx['mean_ratio']:.3f}", "expected_pd_greedy_gate.json")
    add("G4", "Small-instance greedy ratio (min)",
        f"{approx['min_ratio']:.3f}", "expected_pd_greedy_gate.json")

    g5 = load("ris_isac_gate.json")
    for summary in g5["summary"]:
        budget = summary["budget_bits"]
        add("G5", f"B={budget} aligned vs no-RIS mean gain",
            pp(summary["mean_gain_aligned_vs_no_ris"]),
            "ris_isac_gate.json")
        add("G5", f"B={budget} aligned vs no-RIS worst gain",
            pp(summary["worst_gain_aligned_vs_no_ris"]),
            "ris_isac_gate.json")
        add("G5", f"B={budget} QoS feasible rate",
            f"{100.0 * summary['ris_aligned']['qos_feasible_rate']:.0f}%",
            "ris_isac_gate.json")

    g5q = load("ris_phase_resolution_gate.json")
    for summary in g5q["summary"]:
        budget = summary["budget_bits"]
        for bits, value in (
            ("1", summary["bits_1"]),
            ("2", summary["bits_2"]),
            ("3", summary["bits_3"]),
        ):
            add("G5-Q", f"B={budget}, {bits}-bit mean gain vs no RIS",
                pp(value["mean_gain_vs_no_ris"]),
                "ris_phase_resolution_gate.json")
            add("G5-Q", f"B={budget}, {bits}-bit worst gain vs no RIS",
                pp(value["worst_gain_vs_no_ris"]),
                "ris_phase_resolution_gate.json")

    g5p = load("ris_physics_gate.json")
    for summary in g5p["summary"]:
        if summary["budget_bits"] not in (20, 30):
            continue
        if summary["elements"] not in (256, 1024) or summary["aperture_scale"] != 0.01:
            continue
        label = (
            f"B={summary['budget_bits']}, N={summary['elements']}, "
            f"aperture={summary['aperture_scale']}"
        )
        add("G5-P", f"{label} mean gain vs no RIS",
            pp(summary["mean_gain_aligned_vs_no_ris"]),
            "ris_physics_gate.json")
        add("G5-P", f"{label} worst gain vs no RIS",
            pp(summary["worst_gain_aligned_vs_no_ris"]),
            "ris_physics_gate.json")

    g5r = load("ris_joint_budget_gate.json")
    for summary in g5r["summary"]:
        if summary["total_budget_bits"] not in (40, 60) or summary["coherence_frames"] != 64:
            continue
        total = summary["total_budget_bits"]
        add("G5-R", f"Total B={total}, best 3-bit mean gain vs no RIS",
            pp(summary["best_quantized_mean_gain_vs_no_ris"]),
            "ris_joint_budget_gate.json")
        add("G5-R", f"Total B={total}, best 3-bit worst gain vs no RIS",
            pp(summary["best_quantized_worst_gain_vs_no_ris"]),
            "ris_joint_budget_gate.json")
        add("G5-R", f"Total B={total}, report budget at 3-bit",
            str(summary["best_quantized_report_budget_by_mean"]),
            "ris_joint_budget_gate.json")

    g5s = load("ris_placement_gate.json")
    for summary in g5s["summary"]:
        total = summary["total_budget_bits"]
        add("G5-S", f"Total B={total}, best position",
            pos_text(summary["best_position"]), "ris_placement_gate.json")
        add("G5-S", f"Total B={total}, worst gain vs fixed",
            pp(summary["worst_gain_best_vs_fixed"]),
            "ris_placement_gate.json")
        add("G5-S", f"Total B={total}, worst gain vs no RIS",
            pp(summary["worst_gain_best_vs_no_ris"]),
            "ris_placement_gate.json")

    g5t = load("ris_multigrid_gate.json")
    add("G5-T", "Best deployment", pos_text(g5t["best_position"]),
        "ris_multigrid_gate.json")
    add("G5-T", "Deployment evaluations", str(g5t["evaluations"]),
        "ris_multigrid_gate.json")
    add("G5-T", "Worst gain vs fixed",
        pp(g5t["worst_gain_best_vs_fixed"]), "ris_multigrid_gate.json")
    add("G5-T", "Worst gain vs no RIS",
        pp(g5t["worst_gain_best_vs_no_ris"]), "ris_multigrid_gate.json")

    g5u = load("deployment_theory_gate.json")
    add("G5-U", "Empirical Lipschitz constant",
        f"{g5u['lipschitz_estimate']:.3e}", "deployment_theory_gate.json")
    add("G5-U", "Spacing-5 suboptimality bound",
        pp(g5u["second_suboptimality_bound"]),
        "deployment_theory_gate.json")
    add("G5-U", "Fine-to-second improvement",
        pp(g5u["fine_to_second_improvement"]),
        "deployment_theory_gate.json")

    g5v = load("lipschitz_adaptive_deployment_gate.json")["result"]
    add("G5-V", "Best deployment", pos_text(g5v["best_point"]),
        "lipschitz_adaptive_deployment_gate.json")
    add("G5-V", "Worst-target expected P_D",
        f"{g5v['best_value']:.4f}",
        "lipschitz_adaptive_deployment_gate.json")
    add("G5-V", "Certificate gap",
        pp(g5v["certificate_gap"]),
        "lipschitz_adaptive_deployment_gate.json")
    add("G5-V", "Evaluations", str(g5v["evaluations"]),
        "lipschitz_adaptive_deployment_gate.json")

    g5w = load("epsilon_closed_deployment_gate.json")["result"]
    add("G5-W", "Best deployment (single seed)", pos_text(g5w["best_point"]),
        "epsilon_closed_deployment_gate.json")
    add("G5-W", "Certificate gap (single seed)",
        pp(g5w["certificate_gap"]),
        "epsilon_closed_deployment_gate.json")
    add("G5-W", "Evaluations (single seed)", str(g5w["evaluations"]),
        "epsilon_closed_deployment_gate.json")
    g5w3 = load("epsilon_closed_deployment_gate_3seeds.json")["result"]
    add("G5-W", "Certificate gap (3 seeds, original local box)",
        pp(g5w3["certificate_gap"]),
        "epsilon_closed_deployment_gate_3seeds.json")
    add("G5-W", "Evaluations (3 seeds, main search)",
        str(g5w3["evaluations"]),
        "epsilon_closed_deployment_gate_3seeds.json")
    add("G5-W", "Corner-refinement evaluations (3 seeds)",
        str(g5w3.get("refinement_evaluations", 0)),
        "epsilon_closed_deployment_gate_3seeds.json")
    local_closure = g5w3.get("local_closure")
    if local_closure is not None:
        add("G5-W", "Local epsilon-closed (3 seeds)",
            str(g5w3.get("local_epsilon_closed", False)),
            "epsilon_closed_deployment_gate_3seeds.json")
        add("G5-W", "Local certificate gap (3 seeds)",
            pp(local_closure["certificate_gap"]),
            "epsilon_closed_deployment_gate_3seeds.json")
        add("G5-W", "Local evaluations (3 seeds)",
            str(local_closure["evaluations"]),
            "epsilon_closed_deployment_gate_3seeds.json")
        add("G5-W", "Local closure bounds (3 seeds)",
            "; ".join(
                f"[{row[0]:g}, {row[1]:g}]"
                for row in g5w3["local_closure_bounds"]
            ),
            "epsilon_closed_deployment_gate_3seeds.json")

    g5dci = load("g5_deployment_ci_gate.json")
    for prefix, label in (
        ("g5s", "G5-S"),
        ("g5t", "G5-T"),
        ("g5v", "G5-V"),
        ("g5w", "G5-W"),
    ):
        for metric in ("mean", "worst"):
            section = g5dci["sections"][f"{prefix}_vs_fixed_{metric}"]
            add("G5-DCI", f"{label} vs fixed {metric} gain",
                pp(section["mean"]), "g5_deployment_ci_gate.json")
            add("G5-DCI", f"{label} vs fixed {metric} CI",
                ci_text(section["ci95"]), "g5_deployment_ci_gate.json")

    g5rf = load("global_resource_fairness_gate.json")
    for summary in g5rf["summary"]:
        scenario = summary["scenario"]
        add("G5-RF", f"{scenario} mean expected P_D",
            f"{summary['mean_expected_pd']:.3f}",
            "global_resource_fairness_gate.json")
        add("G5-RF", f"{scenario} worst expected P_D",
            f"{summary['worst_expected_pd']:.3f}",
            "global_resource_fairness_gate.json")
        add("G5-RF", f"{scenario} report/control bits",
            f"{summary['report_bits']:.0f}/{summary['control_bits']:.0f}",
            "global_resource_fairness_gate.json")
        add("G5-RF", f"{scenario} total time-bandwidth",
            f"{summary['total_time_bandwidth']:.0f}",
            "global_resource_fairness_gate.json")
        add("G5-RF", f"{scenario} QoS feasible rate",
            f"{100.0 * summary['qos_feasible_rate']:.0f}%",
            "global_resource_fairness_gate.json")

    g5sen = load("ris_sensitivity_gate.json")
    for summary in g5sen["summary"]:
        label = f"{summary['parameter']}={summary['parameter_value']:g}"
        add("G5-SEN", f"{label} report budget",
            str(summary["report_budget_bits"]),
            "ris_sensitivity_gate.json")
        add("G5-SEN", f"{label} mean gain vs no RIS",
            pp(summary["mean_gain_vs_no_ris"]["mean"]),
            "ris_sensitivity_gate.json")
        add("G5-SEN", f"{label} mean gain CI",
            ci_text(summary["mean_gain_vs_no_ris"]["ci95"]),
            "ris_sensitivity_gate.json")
        add("G5-SEN", f"{label} worst gain vs no RIS",
            pp(summary["worst_gain_vs_no_ris"]["mean"]),
            "ris_sensitivity_gate.json")
        add("G5-SEN", f"{label} worst gain CI",
            ci_text(summary["worst_gain_vs_no_ris"]["ci95"]),
            "ris_sensitivity_gate.json")
        add("G5-SEN", f"{label} QoS feasible rate",
            f"{100.0 * summary['qos_feasible_rate']:.0f}%",
            "ris_sensitivity_gate.json")

    sota = load("sota_baseline_gate.json")
    labels = {
        "s1_ris_deflection_topk": "RIS + deflection Top-K",
        "s2_no_ris_deflection_topk": "No-RIS + deflection Top-K",
        "s3_random_ris_deflection_topk": "Random RIS + deflection Top-K",
        "s4_uniform_soft_no_ris": "Uniform soft, no RIS",
        "hard_no_ris": "1-bit counting, no RIS",
        "hard_ris": "1-bit counting, RIS",
        "proposed_schedule_deflection": "Same schedule, deflection fusion",
    }
    for baseline, label in labels.items():
        for metric in ("mean", "worst"):
            section = sota["sections"][f"proposed_vs_{baseline}_{metric}"]
            add("G5-SOTA", f"Proposed vs {label}, {metric} gain",
                pp(section["mean"]), "sota_baseline_gate.json")
            add("G5-SOTA", f"Proposed vs {label}, {metric} CI",
                ci_text(section["ci95"]), "sota_baseline_gate.json")
        add("G5-SOTA", f"Proposed vs {label}, {metric} win rate",
                f"{100.0 * section['win_rate']:.0f}%",
                "sota_baseline_gate.json")

    g6 = load("budget_saturation_gate.json")
    for summary in g6["summary"]:
        label = f"{summary['scenario']}, total B={summary['total_budget_bits']}"
        add("G6", f"{label} report budget", str(summary["report_budget_bits"]),
            "budget_saturation_gate.json")
        add("G6", f"{label} greedy worst P_D",
            f"{summary['greedy_worst']:.3f}", "budget_saturation_gate.json")
        add("G6", f"{label} descent worst P_D",
            f"{summary['descent_worst']:.3f}", "budget_saturation_gate.json")
        add("G6", f"{label} all-scheduled worst P_D",
            f"{summary['all_worst']:.3f}", "budget_saturation_gate.json")
        add("G6", f"{label} selection gain worst",
            pp(summary["selection_gain_worst"]),
            "budget_saturation_gate.json")
        add("G6", f"{label} gap to all-scheduled worst",
            pp(summary["gap_to_all_scheduled_worst"]),
            "budget_saturation_gate.json")
        add("G6", f"{label} QoS rate",
            f"{100.0 * summary['descent_qos_rate']:.0f}%",
            "budget_saturation_gate.json")
    for scenario in ("no_ris", "ris"):
        minimum = g6["minimum_budget_for_qos"][scenario]
        add("G6", f"{scenario} minimum total budget for QoS",
            "none" if minimum["mean"] is None else str(int(minimum["mean"])),
            "budget_saturation_gate.json")
        add("G6", f"{scenario} QoS achievement rate",
            f"{100.0 * minimum['achieved_seed_rate']:.0f}%",
            "budget_saturation_gate.json")

    g7 = load("ris_shared_phase_gate.json")
    scenario_labels = {
        "no_ris": "No RIS",
        "random_shared_phase": "Random shared phase",
        "per_target_ideal_phase": "Per-target ideal phase",
        "shared_weak_aligned": "Shared weak-aligned",
        "shared_surrogate_optimized": "Shared surrogate-gradient",
        "shared_system_optimized": "Shared system-optimized",
    }
    for summary in g7["summary"]:
        label = f"{scenario_labels[summary['scenario']]}, B={summary['total_budget_bits']}"
        add("G7", f"{label} mean P_D",
            f"{summary['mean_expected_pd']:.3f}", "ris_shared_phase_gate.json")
        add("G7", f"{label} worst P_D",
            f"{summary['worst_expected_pd']:.3f}", "ris_shared_phase_gate.json")
        add("G7", f"{label} QoS rate",
            f"{100.0 * summary['qos_feasible_rate']:.0f}%",
            "ris_shared_phase_gate.json")
    for budget, optimization in g7["system_optimization_by_budget"].items():
        add("G7", f"System-optimized cosine, B={budget}",
            f"{optimization['steering_cosine']:.4f}",
            "ris_shared_phase_gate.json")
        add("G7", f"System-optimized worst P_D, B={budget}",
            f"{optimization['system_worst_pd']:.4f}",
            "ris_shared_phase_gate.json")
        add("G7", f"System-optimized evaluations, B={budget}",
            str(optimization["evaluations"]), "ris_shared_phase_gate.json")

    g8 = load("exact_quota_gate.json")
    for summary in g8["summary"]:
        label = f"{summary['scenario']}, B={summary['total_budget_bits']}"
        add("G8", f"{label} greedy worst P_D",
            f"{summary['greedy_worst']:.4f}", "exact_quota_gate.json")
        add("G8", f"{label} exact worst P_D",
            f"{summary['exact_worst']:.4f}", "exact_quota_gate.json")
        add("G8", f"{label} all-scheduled worst P_D",
            f"{summary['all_worst']:.4f}", "exact_quota_gate.json")
        add("G8", f"{label} exact-vs-greedy worst gain",
            pp(summary["selection_gain_worst"]), "exact_quota_gate.json")
        add("G8", f"{label} exact-to-all gap worst",
            pp(summary["gap_to_all_worst"]), "exact_quota_gate.json")

    g8k = load("exact_budget_gate.json")
    add("G8-K", "Controlled oracle match rate",
        f"{100.0 * g8k['controlled']['summary']['oracle_match_rate']:.1f}%",
        "exact_budget_gate.json")
    add("G8-K", "System never worse than greedy",
        f"{100.0 * g8k['variable_rate_system']['summary']['never_worse_than_greedy']:.1f}%",
        "exact_budget_gate.json")
    for cell in g8k["variable_rate_system"]["by_budget"]:
        label = f"B={cell['budget_bits']}"
        add("G8-K", f"{label} greedy worst P_D",
            ms_text(cell["greedy_worst_mean"], cell["greedy_worst_std"]),
            "exact_budget_gate.json")
        add("G8-K", f"{label} exact worst P_D",
            ms_text(cell["exact_worst_mean"], cell["exact_worst_std"]),
            "exact_budget_gate.json")
        add("G8-K", f"{label} exact-vs-greedy worst gain",
            ms_text(cell["gain_worst_mean"], cell["gain_worst_std"]),
            "exact_budget_gate.json")
        add("G8-K", f"{label} exact-vs-greedy worst gain p",
            f"{cell['gain_worst_paired_t']['p_one_sided']:.3f}",
            "exact_budget_gate.json")
        add("G8-K", f"{label} exact-vs-greedy worst gain CI95",
            f"[{cell['gain_worst_bootstrap_ci']['lower']:.4f}, "
            f"{cell['gain_worst_bootstrap_ci']['upper']:.4f}]",
            "exact_budget_gate.json")
        add("G8-K", f"{label} exact QoS rate",
            f"{100.0 * cell['exact_qos_rate']:.0f}%",
            "exact_budget_gate.json")

    g8m = load("exact_maxmin_gate.json")
    add("G8-M", "Controlled oracle match rate",
        f"{100.0 * g8m['controlled']['summary']['oracle_match_rate']:.1f}%",
        "exact_maxmin_gate.json")
    add("G8-M", "System never worse than greedy",
        f"{100.0 * g8m['variable_rate_system']['summary']['never_worse_than_greedy']:.1f}%",
        "exact_maxmin_gate.json")
    for cell in g8m["variable_rate_system"]["by_budget"]:
        label = f"B={cell['budget_bits']}"
        add("G8-M", f"{label} greedy worst P_D",
            ms_text(cell["greedy_worst_mean"], cell["greedy_worst_std"]),
            "exact_maxmin_gate.json")
        add("G8-M", f"{label} exact max-min worst P_D",
            ms_text(cell["exact_worst_mean"], cell["exact_worst_std"]),
            "exact_maxmin_gate.json")
        add("G8-M", f"{label} exact-vs-greedy worst gain",
            ms_text(cell["gain_worst_mean"], cell["gain_worst_std"]),
            "exact_maxmin_gate.json")
        add("G8-M", f"{label} exact-vs-greedy worst gain p",
            f"{cell['gain_worst_paired_t']['p_one_sided']:.3f}",
            "exact_maxmin_gate.json")
        add("G8-M", f"{label} exact-vs-greedy worst gain CI95",
            f"[{cell['gain_worst_bootstrap_ci']['lower']:.4f}, "
            f"{cell['gain_worst_bootstrap_ci']['upper']:.4f}]",
            "exact_maxmin_gate.json")

    g8s = load("scaled_maxmin_gate.json")
    add("G8-S", "Controlled max abs error vs exact",
        f"{g8s['controlled_summary']['max_abs_error']:.2e}",
        "scaled_maxmin_gate.json")
    add("G8-S", "Large-report set found",
        str(g8s["large_report_set"]["found"]),
        "scaled_maxmin_gate.json")
    if g8s["large_report_set"]["min_cost"] is not None:
        add("G8-S", "Large-report set min cost",
            str(g8s["large_report_set"]["min_cost"]),
            "scaled_maxmin_gate.json")
    for cell in g8s["controlled_by_budget"]:
        label = f"B={cell['budget_bits']}"
        add("G8-S", f"{label} exact worst P_D",
            ms_text(cell["exact_worst_mean"], cell["exact_worst_std"]),
            "scaled_maxmin_gate.json")
        add("G8-S", f"{label} scaled worst P_D",
            ms_text(cell["scaled_worst_mean"], cell["scaled_worst_std"]),
            "scaled_maxmin_gate.json")
        add("G8-S", f"{label} abs error vs exact",
            f"{cell['abs_error_mean']:.2e}",
            "scaled_maxmin_gate.json")
    for cell in g8s["scalability_benchmark"]:
        label = f"R={cell['num_reports']}"
        add("G8-S", f"{label} exhaustive subsets",
            str(cell["exhaustive_subsets"]),
            "scaled_maxmin_gate.json")
        add("G8-S", f"{label} min cost",
            "null" if cell["min_cost"] is None else str(cell["min_cost"]),
            "scaled_maxmin_gate.json")
        add("G8-S", f"{label} wall seconds",
            f"{cell['wall_seconds']:.4f}",
            "scaled_maxmin_gate.json")

    g8t = load("exact_selection_target_scalability.json")
    for cell in g8t["summary"]:
        label = f"Q={cell['num_targets']},B={cell['budget_bits']}"
        add("G8-target", f"{label} budget oracle match",
            f"{100.0 * cell['budget_oracle_match_rate']:.0f}%",
            "exact_selection_target_scalability.json")
        add("G8-target", f"{label} max-min oracle match",
            f"{100.0 * cell['maxmin_oracle_match_rate']:.0f}%",
            "exact_selection_target_scalability.json")
        add("G8-target", f"{label} max-min never worse",
            f"{100.0 * cell['maxmin_never_worse_rate']:.0f}%",
            "exact_selection_target_scalability.json")
        add("G8-target", f"{label} budget wall ms",
            f"{cell['budget_wall_mean_ms']:.1f}",
            "exact_selection_target_scalability.json")
        add("G8-target", f"{label} max-min wall ms",
            f"{cell['maxmin_wall_mean_ms']:.1f}",
            "exact_selection_target_scalability.json")

    g30e = load("exact_rate_certificate_gate.json")
    for cell in g30e["summary"]:
        label = f"B={cell['total_budget_bits']}"
        add("G30-E", f"{label} G30 profile exact worst P_D",
            f"{cell['g30_exact_value']:.4f}",
            "exact_rate_certificate_gate.json")
        add("G30-E", f"{label} exact-optimized worst P_D",
            f"{cell['exact_optimized_value']:.4f}",
            "exact_rate_certificate_gate.json")
        add("G30-E", f"{label} exact gain over G30",
            pp(cell["exact_gain_over_g30"]),
            "exact_rate_certificate_gate.json")
        add("G30-E", f"{label} greedy certificate false under exact",
            str(cell["greedy_certificate_false_under_exact"]),
            "exact_rate_certificate_gate.json")
        add("G30-E", f"{label} exact single-change local optimal",
            str(cell["exact_single_change_local_optimal"]),
            "exact_rate_certificate_gate.json")

    g9 = load("ris_subarray_gate.json")
    g9_labels = {
        "no_ris": "No RIS",
        "random_shared_phase": "Random shared phase",
        "shared_weak_aligned": "Shared weak-aligned",
        "per_target_ideal_phase": "Per-target ideal phase",
        "subarray_optimized": "Subarray multi-beam optimized",
    }
    for summary in g9["summary"]:
        label = f"{g9_labels[summary['scenario']]}, B={summary['total_budget_bits']}"
        add("G9", f"{label} mean P_D",
            f"{summary['mean_expected_pd']:.4f}", "ris_subarray_gate.json")
        add("G9", f"{label} worst P_D",
            f"{summary['worst_expected_pd']:.4f}", "ris_subarray_gate.json")
        add("G9", f"{label} QoS rate",
            f"{100.0 * summary['qos_feasible_rate']:.0f}%",
            "ris_subarray_gate.json")
    for budget, optimization in g9["optimization_by_budget"].items():
        add("G9", f"Subarray allocation, B={budget}",
            "(" + ", ".join(str(value) for value in optimization["allocation"]) + ")",
            "ris_subarray_gate.json")
        add("G9", f"Subarray objective worst P_D, B={budget}",
            f"{optimization['value']:.4f}", "ris_subarray_gate.json")

    g10 = load("ris_subarray_steering_gate.json")
    g10_labels = {
        "no_ris": "No RIS",
        "random_shared_phase": "Random shared phase",
        "shared_weak_aligned": "Shared weak-aligned",
        "per_target_ideal_phase": "Per-target ideal phase",
        "g9_subarray": "G9 subarray allocation",
        "g10_steering_optimized": "G10 steering optimized",
    }
    for summary in g10["summary"]:
        label = f"{g10_labels[summary['scenario']]}, B={summary['total_budget_bits']}"
        add("G10", f"{label} mean P_D",
            f"{summary['mean_expected_pd']:.4f}",
            "ris_subarray_steering_gate.json")
        add("G10", f"{label} worst P_D",
            f"{summary['worst_expected_pd']:.4f}",
            "ris_subarray_steering_gate.json")
        add("G10", f"{label} QoS rate",
            f"{100.0 * summary['qos_feasible_rate']:.0f}%",
            "ris_subarray_steering_gate.json")
    for budget, optimization in g10["steering_by_budget"].items():
        add("G10", f"Steering cosines, B={budget}",
            "(" + ", ".join(f"{value:.4f}" for value in optimization["steering_cosines"]) + ")",
            "ris_subarray_steering_gate.json")
        add("G10", f"Steering objective worst P_D, B={budget}",
            f"{optimization['value']:.4f}", "ris_subarray_steering_gate.json")

    g11 = load("ris_aperture_scaling_gate.json")
    selected = [
        summary for summary in g11["summary"]
        if summary["total_budget_bits"] == 20
        and (
            (summary["ris_elements"] == 256
             and summary["phase_bits"] == 3
             and summary["coherence_frames"] == 64)
            or (summary["ris_elements"] == 512
                and summary["phase_bits"] == 3
                and summary["coherence_frames"] == 256)
            or (summary["ris_elements"] == 1024
                and summary["phase_bits"] in (1, 3)
                and summary["coherence_frames"] == 256)
        )
    ]
    for summary in selected:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['ris_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}, "
            f"{summary['allocation_name']}"
        )
        add("G11", f"{label} report budget",
            str(summary["report_budget_bits"]),
            "ris_aperture_scaling_gate.json")
        add("G11", f"{label} mean P_D",
            f"{summary['mean_expected_pd']:.4f}",
            "ris_aperture_scaling_gate.json")
        add("G11", f"{label} worst P_D",
            f"{summary['worst_expected_pd']:.4f}",
            "ris_aperture_scaling_gate.json")
        add("G11", f"{label} QoS rate",
            f"{100.0 * summary['qos_feasible_rate']:.0f}%",
            "ris_aperture_scaling_gate.json")

    g12 = load("derived_architecture_gate.json")
    for summary in g12["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G12", f"{label} weak kappa",
            f"{summary['weak_kappa']:.3e}",
            "derived_architecture_gate.json")
        add("G12", f"{label} derived N*",
            f"{summary['derived_aperture']:.1f}",
            "derived_architecture_gate.json")
        add("G12", f"{label} rounded N",
            str(summary["rounded_aperture"]),
            "derived_architecture_gate.json")
        for candidate in summary["candidates"]:
            add(
                "G12",
                f"{label}, N={candidate['num_elements']} worst P_D",
                f"{candidate['worst_expected_pd']:.4f}",
                "derived_architecture_gate.json",
            )
            add(
                "G12",
                f"{label}, N={candidate['num_elements']} QoS rate",
                f"{100.0 * candidate['qos_feasible_rate']:.0f}%",
                "derived_architecture_gate.json",
            )

    g13 = load("waterfilling_architecture_gate.json")
    for summary in g13["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G13", f"{label} waterfilling allocation",
            "(" + ", ".join(str(value) for value in summary["waterfilling_allocation"]) + ")",
            "waterfilling_architecture_gate.json")
        add("G13", f"{label} equal worst P_D",
            f"{summary['exact']['equal']['worst_expected_pd']:.4f}",
            "waterfilling_architecture_gate.json")
        add("G13", f"{label} waterfilling worst P_D",
            f"{summary['exact']['waterfilling']['worst_expected_pd']:.4f}",
            "waterfilling_architecture_gate.json")
        add("G13", f"{label} waterfilling QoS rate",
            f"{100.0 * summary['exact']['waterfilling']['qos_feasible_rate']:.0f}%",
            "waterfilling_architecture_gate.json")

    g14 = load("exact_allocation_gate.json")
    for summary in g14["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        for method in ("equal", "separable", "exact"):
            add("G14", f"{label} {method} allocation",
                "(" + ", ".join(str(value) for value in summary[f"{method}_allocation"]) + ")",
                "exact_allocation_gate.json")
            add("G14", f"{label} {method} exact surrogate min",
                f"{summary['exact_surrogate'][method]:.4f}",
                "exact_allocation_gate.json")
            add("G14", f"{label} {method} system worst P_D",
                f"{summary['exact_system'][method]['worst_expected_pd']:.4f}",
                "exact_allocation_gate.json")
            add("G14", f"{label} {method} QoS rate",
                f"{100.0 * summary['exact_system'][method]['qos_feasible_rate']:.0f}%",
                "exact_allocation_gate.json")

    g15 = load("system_allocation_gate.json")
    for summary in g15["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G15", f"{label} system ascent allocation",
            "(" + ", ".join(str(value) for value in summary["system_ascent_allocation"]) + ")",
            "system_allocation_gate.json")
        for method in ("equal", "separable", "exact_surrogate", "system_ascent"):
            add("G15", f"{label} {method} worst P_D",
                f"{summary['exact_system'][method]['worst_expected_pd']:.6f}",
                "system_allocation_gate.json")
            add("G15", f"{label} {method} QoS rate",
                f"{100.0 * summary['exact_system'][method]['qos_feasible_rate']:.0f}%",
                "system_allocation_gate.json")

    g16 = load("single_move_certificate_gate.json")
    for summary in g16["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G16", f"{label} G15 allocation",
            "(" + ", ".join(str(value) for value in summary["g15_allocation"]) + ")",
            "single_move_certificate_gate.json")
        add("G16", f"{label} refined allocation",
            "(" + ", ".join(str(value) for value in summary["refined_allocation"]) + ")",
            "single_move_certificate_gate.json")
        add("G16", f"{label} G15 value",
            f"{summary['g15_value']:.6f}", "single_move_certificate_gate.json")
        add("G16", f"{label} refined value",
            f"{summary['refined_value']:.6f}", "single_move_certificate_gate.json")
        add("G16", f"{label} single-move local optimal",
            str(summary["certificate"]["local_optimal"]),
            "single_move_certificate_gate.json")
        add("G16", f"{label} maximum gradient",
            f"{summary['certificate']['maximum_gradient']:.3e}",
            "single_move_certificate_gate.json")

    g17 = load("multi_move_certificate_gate.json")
    for summary in g17["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G17", f"{label} G16 allocation",
            "(" + ", ".join(str(value) for value in summary["g16_allocation"]) + ")",
            "multi_move_certificate_gate.json")
        add("G17", f"{label} multi-block allocation",
            "(" + ", ".join(str(value) for value in summary["multi_block_allocation"]) + ")",
            "multi_move_certificate_gate.json")
        add("G17", f"{label} G16 value",
            f"{summary['g16_value']:.6f}", "multi_move_certificate_gate.json")
        add("G17", f"{label} multi-block value",
            f"{summary['multi_block_value']:.6f}",
            "multi_move_certificate_gate.json")
        add("G17", f"{label} multi-block rounds",
            str(summary["rounds"]), "multi_move_certificate_gate.json")
        add("G17", f"{label} multi-block local optimal",
            str(summary["certificate"]["local_optimal"]),
            "multi_move_certificate_gate.json")

    g18 = load("joint_placement_allocation_gate.json")
    for summary in g18["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G18", f"{label} G17 value",
            f"{summary['g17_value']:.6f}",
            "joint_placement_allocation_gate.json")
        add("G18", f"{label} final position",
            "(" + ", ".join(f"{value:.2f}" for value in summary["final_position"]) + ")",
            "joint_placement_allocation_gate.json")
        add("G18", f"{label} final allocation",
            "(" + ", ".join(str(value) for value in summary["final_allocation"]) + ")",
            "joint_placement_allocation_gate.json")
        add("G18", f"{label} final value",
            f"{summary['final_value']:.6f}",
            "joint_placement_allocation_gate.json")
        add("G18", f"{label} allocation local optimal",
            str(summary["allocation_certificate"]["local_optimal"]),
            "joint_placement_allocation_gate.json")
        add("G18", f"{label} position local optimal",
            str(summary["position_certificate"]["local_optimal"]),
            "joint_placement_allocation_gate.json")

    g19 = load("progressive_decentralization_gate.json")
    method_labels = {
        "centralized_full": "Centralized full",
        "local_schedule_optimal": "Local schedule, optimal fusion",
        "local_schedule_deflection": "Local schedule, deflection fusion",
        "owner_only": "Owner only",
        "hard_decision_local": "1-bit hard decision local",
    }
    for summary in g19["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G19", f"{label} {method_label} mean P_D",
                f"{values['mean']:.6f}",
                "progressive_decentralization_gate.json")
            add("G19", f"{label} {method_label} worst P_D",
                f"{values['worst']:.6f}",
                "progressive_decentralization_gate.json")
            add("G19", f"{label} {method_label} QoS rate",
                f"{100.0 * values['qos_rate']:.0f}%",
                "progressive_decentralization_gate.json")
            add("G19", f"{label} {method_label} worst loss vs centralized",
                pp(values["worst_loss_vs_centralized"]),
                "progressive_decentralization_gate.json")

    g20 = load("amplified_distributed_gate.json")
    method_labels = {
        "centralized_full": "Centralized full",
        "owner_only": "Owner only",
        "hard_default": "1-bit hard default",
        "hard_optimized": "1-bit hard optimized",
    }
    for summary in g20["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G20", f"{label} hard used bits",
            str(summary["hard_used_bits"]),
            "amplified_distributed_gate.json")
        add("G20", f"{label} mean optimized local P_FA",
            f"{summary['mean_optimized_local_pfa']:.4f}",
            "amplified_distributed_gate.json")
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G20", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.6f}",
                "amplified_distributed_gate.json")
            add("G20", f"{label} {method_label} QoS rate",
                f"{100.0 * values['qos_rate']:.0f}%",
                "amplified_distributed_gate.json")

    g21 = load("network_decentralization_gate.json")
    method_labels = {
        "centralized_soft": "Centralized soft",
        "hard_full_links": "Hard full links",
        "hard_top5": "Hard top-5 links",
        "hard_top3": "Hard top-3 links",
        "hard_top1": "Hard top-1 link",
        "peer_majority": "Peer majority",
    }
    for summary in g21["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        add("G21", f"{label} hard used bits",
            str(summary["hard_used_bits"]),
            "network_decentralization_gate.json")
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G21", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.6f}",
                "network_decentralization_gate.json")
            add("G21", f"{label} {method_label} QoS rate",
                f"{100.0 * values['qos_rate']:.0f}%",
                "network_decentralization_gate.json")

    g22 = load("degraded_consensus_gate.json")
    method_labels = {
        "centralized_soft": "Centralized soft",
        "peer_clean": "Peer clean",
        "obs_075": "Partial observability 0.75",
        "link_08": "Link reliability 0.8",
        "multihop_3x08": "Multi-hop 3x0.8",
        "severe": "Severe degradation",
    }
    for summary in g22["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G22", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.6f}",
                "degraded_consensus_gate.json")
            add("G22", f"{label} {method_label} QoS rate",
                f"{100.0 * values['qos_rate']:.0f}%",
                "degraded_consensus_gate.json")
            if values.get("mean_participation") is not None:
                add("G22", f"{label} {method_label} mean participation",
                    f"{values['mean_participation']:.4f}",
                    "degraded_consensus_gate.json")

    g23 = load("correlated_consensus_gate.json")
    method_labels = {
        "centralized_soft": "Centralized soft",
        "peer_clean": "Peer clean",
        "common_fail_02": "Common failure 0.2",
        "common_fail_04": "Common failure 0.4",
        "heterogeneous_obs": "Heterogeneous observability",
        "severe_combined": "Severe combined",
    }
    for summary in g23["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, N={summary['num_elements']}, "
            f"bits={summary['phase_bits']}, C={summary['coherence_frames']}"
        )
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G23", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.6f}",
                "correlated_consensus_gate.json")
            add("G23", f"{label} {method_label} QoS rate",
                f"{100.0 * values['qos_rate']:.0f}%",
                "correlated_consensus_gate.json")
            if values.get("mean_participation") is not None:
                add("G23", f"{label} {method_label} mean participation",
                    f"{values['mean_participation']:.4f}",
                    "correlated_consensus_gate.json")

    g24 = load("scalability_comparison_gate.json")
    for summary in g24["summary"]:
        label = (
            f"Q={summary['num_targets']}, M={summary['num_uavs']}, "
            f"M/Q={summary['uav_to_target_ratio']}"
        )
        add("G24", f"{label} total budget",
            str(summary["total_budget_bits"]),
            "scalability_comparison_gate.json")
        add("G24", f"{label} no-RIS worst P_D",
            f"{summary['no_ris_worst']:.4f}",
            "scalability_comparison_gate.json")
        add("G24", f"{label} RIS ideal worst P_D",
            f"{summary['ris_ideal_worst']:.4f}",
            "scalability_comparison_gate.json")
        add("G24", f"{label} peer majority worst P_D",
            f"{summary['peer_majority_worst']:.4f}",
            "scalability_comparison_gate.json")
        add("G24", f"{label} no-RIS QoS",
            f"{100.0 * summary['no_ris_qos']:.0f}%",
            "scalability_comparison_gate.json")
        add("G24", f"{label} RIS ideal QoS",
            f"{100.0 * summary['ris_ideal_qos']:.0f}%",
            "scalability_comparison_gate.json")
        add("G24", f"{label} peer majority QoS",
            f"{100.0 * summary['peer_majority_qos']:.0f}%",
            "scalability_comparison_gate.json")

    g25 = load("scaled_g18_scalability_gate.json")
    for summary in g25["summary"]:
        label = (
            f"Q={summary['num_targets']}, M={summary['num_uavs']}, "
            f"M/Q={summary['uav_to_target_ratio']}"
        )
        add("G25", f"{label} scaled-G18 allocation",
            "(" + ", ".join(str(value) for value in summary["allocation"]) + ")",
            "scaled_g18_scalability_gate.json")
        add("G25", f"{label} scaled-G18 position",
            "(" + ", ".join(f"{value:.2f}" for value in summary["position"]) + ")",
            "scaled_g18_scalability_gate.json")
        add("G25", f"{label} no-RIS worst P_D",
            f"{summary['no_ris_worst']:.4f}",
            "scaled_g18_scalability_gate.json")
        add("G25", f"{label} RIS ideal worst P_D",
            f"{summary['ris_ideal_worst']:.4f}",
            "scaled_g18_scalability_gate.json")
        add("G25", f"{label} scaled-G18 worst P_D",
            f"{summary['scaled_g18_worst']:.4f}",
            "scaled_g18_scalability_gate.json")
        add("G25", f"{label} peer majority worst P_D",
            f"{summary['peer_majority_worst']:.4f}",
            "scaled_g18_scalability_gate.json")
        add("G25", f"{label} scaled-G18 QoS",
            f"{100.0 * summary['scaled_g18_qos']:.0f}%",
            "scaled_g18_scalability_gate.json")

    g26 = load("mobility_blockage_gate.json")
    method_labels = {
        "no_ris": "No RIS",
        "ris_ideal": "RIS ideal",
        "ris_static_subarray": "RIS static subarray",
        "ris_adaptive_subarray": "RIS adaptive subarray",
    }
    for method, method_label in method_labels.items():
        values = g26["summary"]["methods"][method]
        add("G26", f"{method_label} worst over time",
            f"{values['worst_over_time']:.4f}",
            "mobility_blockage_gate.json")
        add("G26", f"{method_label} mean over time",
            f"{values['mean_over_time']:.4f}",
            "mobility_blockage_gate.json")
        add("G26", f"{method_label} QoS over time",
            f"{100.0 * values['qos_over_time']:.0f}%",
            "mobility_blockage_gate.json")
    add("G26", "Static allocation",
        "(" + ", ".join(str(value) for value in g26["summary"]["static_allocation"]) + ")",
        "mobility_blockage_gate.json")

    g27 = load("multi_ris_gate.json")
    for summary in g27["summary"]:
        label = f"B={summary['total_budget_bits']}"
        add("G27", f"{label} no-RIS worst P_D",
            f"{summary['no_ris_worst']:.4f}", "multi_ris_gate.json")
        add("G27", f"{label} no-RIS QoS",
            f"{100.0 * summary['no_ris_qos']:.0f}%", "multi_ris_gate.json")
        for method, values in summary["methods"].items():
            add("G27", f"{label} {method} worst P_D",
                f"{values['worst_expected_pd']:.4f}", "multi_ris_gate.json")
            add("G27", f"{label} {method} QoS",
                f"{100.0 * values['qos_rate']:.0f}%", "multi_ris_gate.json")
            add("G27", f"{label} {method} report budget",
                str(values["report_budget_bits"]), "multi_ris_gate.json")

    g28 = load("multi_ris_split_optimization_gate.json")
    summary = g28["summary"]
    add("G28", "One RIS value", f"{summary['one_ris_value']:.6f}",
        "multi_ris_split_optimization_gate.json")
    add("G28", "Equal split value", f"{summary['equal_split_value']:.6f}",
        "multi_ris_split_optimization_gate.json")
    add("G28", "Optimized split",
        str(summary["optimized_split"]),
        "multi_ris_split_optimization_gate.json")
    add("G28", "Optimized second position",
        "(" + ", ".join(f"{value:.2f}" for value in summary["optimized_second_position"]) + ")",
        "multi_ris_split_optimization_gate.json")
    add("G28", "Optimized value", f"{summary['optimized_value']:.6f}",
        "multi_ris_split_optimization_gate.json")
    add("G28", "Optimized QoS",
        f"{100.0 * summary['optimized_qos']:.0f}%",
        "multi_ris_split_optimization_gate.json")

    g29 = load("variable_rate_report_gate.json")
    method_labels = {
        "soft5": "Soft 5-bit",
        "soft3": "Soft 3-bit",
        "adaptive_soft": "Adaptive soft",
        "hard1": "Hard 1-bit",
    }
    for summary in g29["summary"]:
        label = f"B={summary['total_budget_bits']}"
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G29", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.4f}",
                "variable_rate_report_gate.json")
            add("G29", f"{label} {method_label} QoS",
                f"{100.0 * values['qos_rate']:.0f}%",
                "variable_rate_report_gate.json")

    g30 = load("global_rate_optimization_gate.json")
    for summary in g30["summary"]:
        label = f"B={summary['total_budget_bits']}"
        add("G30", f"{label} fixed 3-bit",
            f"{summary['fixed3']:.6f}",
            "global_rate_optimization_gate.json")
        add("G30", f"{label} fixed 5-bit",
            f"{summary['fixed5']:.6f}",
            "global_rate_optimization_gate.json")
        add("G30", f"{label} adaptive profile",
            f"{summary['adaptive']:.6f}",
            "global_rate_optimization_gate.json")
        add("G30", f"{label} optimized bits",
            "(" + ", ".join(str(value) for value in summary["optimized_bits"]) + ")",
            "global_rate_optimization_gate.json")
        add("G30", f"{label} optimized value",
            f"{summary['optimized_value']:.6f}",
            "global_rate_optimization_gate.json")
        add("G30", f"{label} single-change local optimal",
            str(summary["single_change_local_optimal"]),
            "global_rate_optimization_gate.json")

    g31 = load("hybrid_fusion_gate.json")
    method_labels = {
        "soft5": "Soft 5-bit",
        "hard1": "Hard 1-bit",
        "hybrid": "Soft/hard hybrid",
    }
    for summary in g31["summary"]:
        label = f"B={summary['total_budget_bits']}"
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G31", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.4f}",
                "hybrid_fusion_gate.json")
            add("G31", f"{label} {method_label} QoS",
                f"{100.0 * values['qos_rate']:.0f}%",
                "hybrid_fusion_gate.json")

    g32 = load("interference_sensitivity_gate.json")
    method_labels = {
        "no_ris": "No RIS",
        "ris_ideal": "RIS ideal",
        "peer_majority": "Peer majority",
    }
    for summary in g32["summary"]:
        label = f"INR={summary['inr_db']:g} dB"
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G32", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.4f}",
                "interference_sensitivity_gate.json")
            add("G32", f"{label} {method_label} QoS",
                f"{100.0 * values['qos_rate']:.0f}%",
                "interference_sensitivity_gate.json")

    g33 = load("spatial_interference_placement_gate.json")
    for summary in g33["summary"]:
        label = f"INR_ref={summary['inr_ref']:g}"
        add("G33", f"{label} mean INR",
            f"{summary['mean_inr']:.4f}",
            "spatial_interference_placement_gate.json")
        add("G33", f"{label} no-RIS worst P_D",
            f"{summary['no_ris_worst']:.4f}",
            "spatial_interference_placement_gate.json")
        add("G33", f"{label} fixed RIS worst P_D",
            f"{summary['fixed_ris_worst']:.4f}",
            "spatial_interference_placement_gate.json")
        add("G33", f"{label} optimized RIS worst P_D",
            f"{summary['optimized_ris_worst']:.4f}",
            "spatial_interference_placement_gate.json")
        add("G33", f"{label} optimized position",
            "(" + ", ".join(f"{value:.2f}" for value in summary["optimized_position"]) + ")",
            "spatial_interference_placement_gate.json")
        add("G33", f"{label} optimized QoS",
            f"{100.0 * summary['optimized_ris_qos']:.0f}%",
            "spatial_interference_placement_gate.json")

    g34 = load("multi_interference_placement_gate.json")
    summary = g34["summary"]
    add("G34", "Mean INR", f"{summary['mean_inr']:.4f}",
        "multi_interference_placement_gate.json")
    add("G34", "Max INR", f"{summary['max_inr']:.4f}",
        "multi_interference_placement_gate.json")
    add("G34", "No-RIS worst P_D", f"{summary['no_ris_worst']:.4f}",
        "multi_interference_placement_gate.json")
    add("G34", "Fixed RIS worst P_D", f"{summary['fixed_ris_worst']:.4f}",
        "multi_interference_placement_gate.json")
    add("G34", "Optimized RIS worst P_D",
        f"{summary['optimized_ris_worst']:.4f}",
        "multi_interference_placement_gate.json")
    add("G34", "Optimized position",
        "(" + ", ".join(f"{value:.2f}" for value in summary["optimized_position"]) + ")",
        "multi_interference_placement_gate.json")
    add("G34", "Optimized QoS",
        f"{100.0 * summary['optimized_ris_qos']:.0f}%",
        "multi_interference_placement_gate.json")

    g35 = load("upd_vs_ula_gate.json")
    method_labels = {
        "no_ris": "No RIS",
        "ula": "1-D ULA",
        "upa": "2-D UPA",
    }
    for summary in g35["summary"]:
        label = f"{summary['scenario']}, B={summary['total_budget_bits']}"
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G35", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.4f}",
                "upd_vs_ula_gate.json")
            add("G35", f"{label} {method_label} QoS",
                f"{100.0 * values['qos_rate']:.0f}%",
                "upd_vs_ula_gate.json")

    g36 = load("null_steering_gate.json")
    suppression = g36["interference_suppression"]
    add("G36", "Mean reflected INR aligned",
        f"{suppression['mean_reflected_inr_aligned']:.4f}",
        "null_steering_gate.json")
    add("G36", "Mean reflected INR null",
        f"{suppression['mean_reflected_inr_null']:.4f}",
        "null_steering_gate.json")
    add("G36", "Mean target gain aligned",
        f"{suppression['mean_array_gain_aligned']:.4f}",
        "null_steering_gate.json")
    add("G36", "Mean target gain null",
        f"{suppression['mean_array_gain_null']:.4f}",
        "null_steering_gate.json")
    for summary in g36["summary"]:
        label = f"B={summary['total_budget_bits']}"
        for method, method_label in (
            ("no_ris", "No RIS"),
            ("aligned", "Aligned UPA"),
            ("null_steered", "Null-steered UPA"),
        ):
            values = summary["methods"][method]
            add("G36", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.4f}",
                "null_steering_gate.json")
            add("G36", f"{label} {method_label} QoS",
                f"{100.0 * values['qos_rate']:.0f}%",
                "null_steering_gate.json")

    g37 = load("quantized_null_steering_gate.json")
    info = g37["interference_info"]
    for name, values in info.items():
        add("G37", f"{name} mean reflected INR",
            f"{values['mean_reflected_inr']:.6f}",
            "quantized_null_steering_gate.json")
        add("G37", f"{name} mean target gain",
            f"{values['mean_target_gain']:.4f}",
            "quantized_null_steering_gate.json")
    for summary in g37["summary"]:
        label = f"B={summary['total_budget_bits']}"
        for method, method_label in (
            ("no_ris", "No RIS"),
            ("aligned_quantized", "Aligned quantized"),
            ("continuous_quantized", "Continuous then quantized"),
            ("quantized_optimized", "Quantized optimized"),
        ):
            values = summary["methods"][method]
            add("G37", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.6f}",
                "quantized_null_steering_gate.json")
            add("G37", f"{label} {method_label} QoS",
                f"{100.0 * values['qos_rate']:.0f}%",
                "quantized_null_steering_gate.json")

    g38 = load("joint_null_placement_gate.json")
    summary = g38["summary"]
    add("G38", "No-RIS worst P_D", f"{summary['no_ris_worst']:.4f}",
        "joint_null_placement_gate.json")
    add("G38", "Fixed position value", f"{summary['fixed_value']:.6f}",
        "joint_null_placement_gate.json")
    add("G38", "Fixed reflected INR", f"{summary['fixed_reflected_inr']:.6f}",
        "joint_null_placement_gate.json")
    add("G38", "Optimized position",
        "(" + ", ".join(f"{value:.2f}" for value in summary["optimized_position"]) + ")",
        "joint_null_placement_gate.json")
    add("G38", "Optimized value", f"{summary['optimized_value']:.6f}",
        "joint_null_placement_gate.json")
    add("G38", "Optimized reflected INR",
        f"{summary['optimized_reflected_inr']:.6f}",
        "joint_null_placement_gate.json")
    add("G38", "Optimized QoS",
        f"{100.0 * summary['optimized_qos']:.0f}%",
        "joint_null_placement_gate.json")

    g39 = load("distributed_relaxation_gate.json")
    method_labels = {
        "centralized_soft": "Centralized soft",
        "peer_clean": "Peer clean",
        "peer_multihop": "Peer multi-hop",
        "hard_optimized": "Hard optimized",
    }
    for summary in g39["summary"]:
        label = (
            f"B={summary['total_budget_bits']}, "
            f"QoS={summary['qos_target']:.2f}"
        )
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G39", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.4f}",
                "distributed_relaxation_gate.json")
            add("G39", f"{label} {method_label} QoS feasible",
                str(values["qos_feasible"]),
                "distributed_relaxation_gate.json")

    g40 = load("low_budget_snr_distributed_gate.json")
    method_labels = {
        "centralized_soft": "Centralized soft",
        "peer_clean": "Peer clean",
        "peer_multihop": "Peer multi-hop",
        "hard_optimized": "Hard optimized",
    }
    for summary in g40["summary"]:
        label = f"B={summary['total_budget_bits']}, N={summary['ris_elements']}"
        for method, method_label in method_labels.items():
            values = summary["methods"][method]
            add("G40", f"{label} {method_label} worst P_D",
                f"{values['worst_expected_pd']:.4f}",
                "low_budget_snr_distributed_gate.json")
            add("G40", f"{label} {method_label} QoS feasible",
                str(values["qos_feasible"]),
                "low_budget_snr_distributed_gate.json")

    g41 = load("consensus_parity_boundary_gate.json")
    for summary in g41["summary"]:
        label = f"M={summary['num_uavs']}, B={summary['total_budget_bits']}"
        add("G41", f"{label} theoretical min UAVs",
            f"{summary['theoretical_min_uavs']:.1f}",
            "consensus_parity_boundary_gate.json")
        add("G41", f"{label} centralized worst P_D",
            f"{summary['centralized_worst']:.4f}",
            "consensus_parity_boundary_gate.json")
        add("G41", f"{label} peer worst P_D",
            f"{summary['peer_worst']:.4f}",
            "consensus_parity_boundary_gate.json")
        add("G41", f"{label} consensus wins",
            str(summary["consensus_wins"]),
            "consensus_parity_boundary_gate.json")

    g42 = load("optimized_parity_boundary_gate.json")
    for summary in g42["summary"]:
        label = f"M={summary['num_uavs']}, B={summary['total_budget_bits']}"
        add("G42", f"{label} M_min fixed", f"{summary['m_fixed']:.2f}",
            "optimized_parity_boundary_gate.json")
        add("G42", f"{label} M_min optimized",
            f"{summary['m_optimized']:.2f}",
            "optimized_parity_boundary_gate.json")
        add("G42", f"{label} centralized worst P_D",
            f"{summary['centralized_worst']:.4f}",
            "optimized_parity_boundary_gate.json")
        add("G42", f"{label} peer worst P_D",
            f"{summary['peer_worst']:.4f}",
            "optimized_parity_boundary_gate.json")
        add("G42", f"{label} consensus wins",
            str(summary["consensus_wins"]),
            "optimized_parity_boundary_gate.json")

    g43 = load("exact_parity_boundary_gate.json")
    for summary in g43["summary"]:
        label = f"M={summary['num_uavs']}"
        add("G43", f"{label} exact min feasible",
            "inf" if summary["exact_min_feasible"] == float("inf")
            else f"{summary['exact_min_feasible']:.0f}",
            "exact_parity_boundary_gate.json")
        add("G43", f"{label} Gaussian min",
            f"{summary['gaussian_min']:.2f}",
            "exact_parity_boundary_gate.json")
        add("G43", f"{label} centralized low-budget worst P_D",
            f"{summary['centralized_worst_low_budget']:.4f}",
            "exact_parity_boundary_gate.json")
        add("G43", f"{label} peer low-budget worst P_D",
            f"{summary['peer_worst_low_budget']:.4f}",
            "exact_parity_boundary_gate.json")
        add("G43", f"{label} consensus wins low budget",
            str(summary["consensus_wins_low_budget"]),
            "exact_parity_boundary_gate.json")

    g43b = load("exact_min_majority_gate.json")
    for summary in g43b["summary"]:
        label = f"M={summary['num_uavs']}"
        add("G43-B", f"{label} exact min voters",
            "null" if summary["exact_min_uavs"] is None
            else f"{summary['exact_min_uavs']}",
            "exact_min_majority_gate.json")
        add("G43-B", f"{label} feasibility monotone",
            str(summary["feasibility_monotone"]),
            "exact_min_majority_gate.json")
        add("G43-B", f"{label} max voters",
            str(summary["max_voters"]),
            "exact_min_majority_gate.json")
        if summary["best_local_alpha"] is not None:
            add("G43-B", f"{label} best local P_FA",
                f"{summary['best_local_alpha']:.4f}",
                "exact_min_majority_gate.json")

    g44 = load("fundamental_information_gate.json")
    for summary in g44["summary"]:
        label = f"B={summary['total_budget_bits']}"
        add("G44", f"{label} soft P_D",
            f"{summary['soft_pd']:.4f}",
            "fundamental_information_gate.json")
        add("G44", f"{label} soft normalized info",
            f"{summary['soft_info_norm']:.4f}",
            "fundamental_information_gate.json")
        add("G44", f"{label} hard P_D",
            f"{summary['hard_pd']:.4f}",
            "fundamental_information_gate.json")
        add("G44", f"{label} hard normalized info",
            f"{summary['hard_info_norm']:.4f}",
            "fundamental_information_gate.json")
        add("G44", f"{label} peer P_D",
            f"{summary['peer_pd']:.4f}",
            "fundamental_information_gate.json")
        add("G44", f"{label} peer normalized info",
            f"{summary['peer_info_norm']:.4f}",
            "fundamental_information_gate.json")
        add("G44", f"{label} full info",
            f"{summary['full_info']:.2f}",
            "fundamental_information_gate.json")

    g45 = load("resource_information_law_gate.json")
    for summary in g45["summary"]:
        label = (
            f"N={summary['ris_elements']}, B={summary['total_budget_bits']}"
        )
        add("G45", f"{label} predicted P_D",
            f"{summary['predicted_pd']:.4f}",
            "resource_information_law_gate.json")
        add("G45", f"{label} exact P_D",
            f"{summary['exact_pd']:.4f}",
            "resource_information_law_gate.json")
        add("G45", f"{label} absolute error",
            f"{summary['absolute_error']:.4f}",
            "resource_information_law_gate.json")
        add("G45", f"{label} predicted D",
            f"{summary['predicted_d']:.1f}",
            "resource_information_law_gate.json")

    g46 = load("exact_information_budget_gate.json")
    for summary in g46["summary"]:
        label = f"B={summary['total_budget_bits']}"
        for method, method_label in (
            ("soft", "Soft"),
            ("hard", "Hard"),
            ("peer", "Peer consensus"),
        ):
            add("G46", f"{label} {method_label} P_D",
                f"{summary[f'{method}_pd']:.4f}",
                "exact_information_budget_gate.json")
            add("G46", f"{label} {method_label} exact rho",
                f"{summary[f'{method}_rho_exact']:.4f}",
                "exact_information_budget_gate.json")
            add("G46", f"{label} {method_label} raw rho",
                f"{summary[f'{method}_rho_raw']:.4f}",
                "exact_information_budget_gate.json")
            add("G46", f"{label} {method_label} used report bits",
                f"{summary[f'{method}_used_bits']:.0f}",
                "exact_information_budget_gate.json")
            add("G46", f"{label} {method_label} raw inflation factor",
                f"{summary[f'{method}_raw_inflation_factor']:.3f}",
                "exact_information_budget_gate.json")

    g47 = load("architecture_switch_gate.json")
    for summary in g47["summary"]:
        label = f"B={summary['total_budget_bits']}"
        add("G47", f"{label} soft worst P_D",
            f"{summary['soft_worst_pd']:.4f}",
            "architecture_switch_gate.json")
        add("G47", f"{label} peer worst P_D",
            f"{summary['peer_worst_pd']:.4f}",
            "architecture_switch_gate.json")
        add("G47", f"{label} exact-switch worst P_D",
            f"{summary['exact_switch_worst_pd']:.4f}",
            "architecture_switch_gate.json")
        add("G47", f"{label} fixed-switch worst P_D",
            f"{summary['fixed_switch_worst_pd']:.4f}",
            "architecture_switch_gate.json")
        add("G47", f"{label} exact-switch gain vs soft",
            pp(summary["exact_gain_vs_soft"]),
            "architecture_switch_gate.json")
        add("G47", f"{label} fixed-switch gain vs soft",
            pp(summary["fixed_gain_vs_soft"]),
            "architecture_switch_gate.json")
        add("G47", f"{label} peer-selected rate (exact)",
            f"{100.0 * summary['peer_selected_rate_exact']:.0f}%",
            "architecture_switch_gate.json")
        add("G47", f"{label} peer-selected rate (fixed)",
            f"{100.0 * summary['peer_selected_rate_fixed']:.0f}%",
            "architecture_switch_gate.json")
        add("G47", f"{label} exact-switch rho",
            f"{summary['exact_switch_rho_exact']:.4f}",
            "architecture_switch_gate.json")
        add("G47", f"{label} exact-switch QoS",
            str(summary["exact_switch_qos_feasible"]),
            "architecture_switch_gate.json")

    g48 = load("target_wise_architecture_switch_gate.json")
    for summary in g48["summary"]:
        label = f"B={summary['total_budget_bits']}"
        add("G48", f"{label} soft worst P_D",
            f"{summary['soft_worst_pd']:.4f}",
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} peer worst P_D",
            f"{summary['peer_worst_pd']:.4f}",
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} global-switch worst P_D",
            f"{summary['global_switch_worst_pd']:.4f}",
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} target-wise worst P_D",
            f"{summary['target_wise_switch_worst_pd']:.4f}",
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} target-wise gain vs soft",
            pp(summary["target_wise_gain_vs_soft"]),
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} target-wise gain vs global switch",
            pp(summary["target_wise_gain_vs_global"]),
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} peer target-selection rate",
            f"{100.0 * summary['peer_target_selection_rate']:.0f}%",
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} target-wise rho",
            f"{summary['target_wise_rho_exact']:.4f}",
            "target_wise_architecture_switch_gate.json")
        add("G48", f"{label} target-wise QoS",
            str(summary["target_wise_qos_feasible"]),
            "target_wise_architecture_switch_gate.json")

    g49 = load("soft_reallocation_gate.json")
    for summary in g49["summary"]:
        label = f"B={summary['total_budget_bits']}"
        add("G49", f"{label} soft worst P_D",
            f"{summary['soft_worst_pd']:.4f}",
            "soft_reallocation_gate.json")
        add("G49", f"{label} target-wise worst P_D",
            f"{summary['target_wise_switch_worst_pd']:.4f}",
            "soft_reallocation_gate.json")
        add("G49", f"{label} reallocation worst P_D",
            f"{summary['reallocation_worst_pd']:.4f}",
            "soft_reallocation_gate.json")
        add("G49", f"{label} reallocation gain vs soft",
            pp(summary["reallocation_gain_vs_soft"]),
            "soft_reallocation_gate.json")
        add("G49", f"{label} reallocation gain vs global switch",
            pp(summary["reallocation_gain_vs_global"]),
            "soft_reallocation_gate.json")
        add("G49", f"{label} reallocation gain vs target-wise",
            pp(summary["reallocation_gain_vs_target_wise"]),
            "soft_reallocation_gate.json")
        add("G49", f"{label} reallocation rho",
            f"{summary['reallocation_rho_exact']:.4f}",
            "soft_reallocation_gate.json")
        add("G49", f"{label} target-wise used bits",
            f"{summary['target_wise_used_bits']:.0f}",
            "soft_reallocation_gate.json")
        add("G49", f"{label} reallocation used bits",
            f"{summary['reallocation_used_bits']:.0f}",
            "soft_reallocation_gate.json")
        add("G49", f"{label} reallocation QoS",
            str(summary["reallocation_qos_feasible"]),
            "soft_reallocation_gate.json")

    g50 = load("mode_ascent_gate.json")
    for summary in g50["summary"]:
        label = f"B={summary['total_budget_bits']}"
        add("G50", f"{label} soft worst P_D",
            f"{summary['soft_worst_pd']:.4f}",
            "mode_ascent_gate.json")
        add("G50", f"{label} target-wise worst P_D",
            f"{summary['target_wise_switch_worst_pd']:.4f}",
            "mode_ascent_gate.json")
        add("G50", f"{label} mode-ascent worst P_D",
            f"{summary['mode_ascent_worst_pd']:.4f}",
            "mode_ascent_gate.json")
        add("G50", f"{label} mode-ascent gain vs soft",
            pp(summary["mode_ascent_gain_vs_soft"]),
            "mode_ascent_gate.json")
        add("G50", f"{label} mode-ascent gain vs target-wise",
            pp(summary["mode_ascent_gain_vs_target_wise"]),
            "mode_ascent_gate.json")
        add("G50", f"{label} mode-ascent rho",
            f"{summary['mode_ascent_rho_exact']:.4f}",
            "mode_ascent_gate.json")
        add("G50", f"{label} mode-ascent used bits",
            f"{summary['mode_ascent_used_bits']:.0f}",
            "mode_ascent_gate.json")
        add("G50", f"{label} peer-to-soft switches",
            f"{summary['peer_to_soft_switches']:.2f}",
            "mode_ascent_gate.json")
        add("G50", f"{label} mode-ascent QoS",
            str(summary["mode_ascent_qos_feasible"]),
            "mode_ascent_gate.json")

    g51 = load("stochastic_mobility_gate.json")
    labels = {
        "no_ris_soft": "No-RIS soft",
        "no_ris_mode_ascent": "No-RIS mode ascent",
        "ris_static_mode_ascent": "Static RIS mode ascent",
        "ris_latency_mode_ascent": "Latency-1 RIS mode ascent",
        "ris_ideal_target_wise": "Ideal RIS target-wise",
        "ris_ideal_mode_ascent": "Ideal RIS mode ascent",
    }
    for method, method_label in labels.items():
        values = g51["summary"]["methods"][method]
        add("G51", f"{method_label} worst over time",
            f"{values['worst_over_time']:.4f}",
            "stochastic_mobility_gate.json")
        add("G51", f"{method_label} mean over time",
            f"{values['mean_over_time']:.4f}",
            "stochastic_mobility_gate.json")
        add("G51", f"{method_label} QoS over time",
            f"{100.0 * values['qos_over_time']:.0f}%",
            "stochastic_mobility_gate.json")
    add("G51", "Latency-1 vs static worst gain",
        pp(g51["summary"]["latency_over_static_worst_gain"]),
        "stochastic_mobility_gate.json")
    add("G51", "Ideal mode ascent vs target-wise worst gain",
        pp(g51["summary"]["ideal_ascent_over_target_wise_worst_gain"]),
        "stochastic_mobility_gate.json")
    add("G51", "Ideal mode ascent QoS gain",
        pp(g51["summary"]["ideal_ascent_qos_gain"]),
        "stochastic_mobility_gate.json")

    g52 = load("prediction_aware_ris_gate.json")
    labels = {
        "no_ris_soft": "No-RIS soft",
        "ris_static_mode_ascent": "Static RIS mode ascent",
        "ris_latency_mode_ascent": "Latency-1 RIS mode ascent",
        "ris_mmse_mode_ascent": "MMSE RIS mode ascent",
        "ris_ideal_target_wise": "Ideal RIS target-wise",
        "ris_ideal_mode_ascent": "Ideal RIS mode ascent",
    }
    for method, method_label in labels.items():
        values = g52["summary"]["methods"][method]
        add("G52", f"{method_label} worst over time",
            f"{values['worst_over_time']:.4f}",
            "prediction_aware_ris_gate.json")
        add("G52", f"{method_label} mean over time",
            f"{values['mean_over_time']:.4f}",
            "prediction_aware_ris_gate.json")
        add("G52", f"{method_label} QoS over time",
            f"{100.0 * values['qos_over_time']:.0f}%",
            "prediction_aware_ris_gate.json")
    add("G52", "MMSE vs latency-1 worst gain",
        pp(g52["summary"]["mmse_over_latency_worst_gain"]),
        "prediction_aware_ris_gate.json")
    add("G52", "MMSE QoS gain",
        pp(g52["summary"]["mmse_qos_gain"]),
        "prediction_aware_ris_gate.json")

    g53 = load("multi_step_prediction_gate.json")
    for horizon in g53["summary"]["horizons"]:
        h = horizon["horizon"]
        add("G53", f"h={h} prediction coefficient",
            f"{horizon['prediction_coefficient']:.4f}",
            "multi_step_prediction_gate.json")
        add("G53", f"h={h} error covariance scale",
            f"{horizon['error_covariance_scale']:.4f}",
            "multi_step_prediction_gate.json")
        add("G53", f"h={h} stale worst over time",
            f"{horizon['stale_worst_over_time']:.4f}",
            "multi_step_prediction_gate.json")
        add("G53", f"h={h} MMSE worst over time",
            f"{horizon['mmse_worst_over_time']:.4f}",
            "multi_step_prediction_gate.json")
        add("G53", f"h={h} MMSE vs stale worst gain",
            pp(horizon["mmse_over_stale_worst_gain"]),
            "multi_step_prediction_gate.json")
        add("G53", f"h={h} stale QoS over time",
            f"{100.0 * horizon['stale_qos_over_time']:.0f}%",
            "multi_step_prediction_gate.json")
        add("G53", f"h={h} MMSE QoS over time",
            f"{100.0 * horizon['mmse_qos_over_time']:.0f}%",
            "multi_step_prediction_gate.json")
    add("G53", "Oracle horizon worst over time",
        f"{g53['summary']['oracle_horizon_worst_over_time']:.4f}",
        "multi_step_prediction_gate.json")
    add("G53", "Oracle horizon QoS over time",
        f"{100.0 * g53['summary']['methods']['oracle_horizon_mode_ascent']['qos_over_time']:.0f}%",
        "multi_step_prediction_gate.json")
    add("G53", "Oracle horizon vs best fixed MMSE worst gain",
        pp(g53["summary"]["oracle_over_best_fixed_mmse_worst_gain"]),
        "multi_step_prediction_gate.json")
    for row in g53["summary"]["architecture_reconfiguration"]:
        delta = row["delta"]
        add("G53", f"Hysteresis delta={delta} worst over time",
            f"{row['worst_over_time']:.4f}",
            "multi_step_prediction_gate.json")
        add("G53", f"Hysteresis delta={delta} QoS over time",
            f"{100.0 * row['qos_over_time']:.0f}%",
            "multi_step_prediction_gate.json")
        add("G53", f"Hysteresis delta={delta} switches per seed",
            f"{row['mean_switches_per_seed']:.2f}",
            "multi_step_prediction_gate.json")
        add("G53", f"Hysteresis delta={delta} loss vs oracle",
            pp(g53["summary"]["oracle_horizon_worst_over_time"]
               - row["worst_over_time"]),
            "multi_step_prediction_gate.json")
    for row in g53["summary"]["switch_cost_analysis"]:
        cost = row["switch_cost_bits"]
        add("G53", f"Switch cost {cost} bit optimal delta",
            "none" if row["best_delta"] is None else f"{row['best_delta']:.3f}",
            "multi_step_prediction_gate.json")
        add("G53", f"Switch cost {cost} bit optimal worst",
            "none" if row["best_worst_over_time"] < 0 else
            f"{row['best_worst_over_time']:.4f}",
            "multi_step_prediction_gate.json")
        add("G53", f"Switch cost {cost} bit optimal switches",
            "none" if row["best_switches"] is None else
            f"{row['best_switches']:.2f}",
            "multi_step_prediction_gate.json")

    g54 = load("covariance_aware_ris_gate.json")
    labels = {
        "no_ris_soft": "No-RIS soft",
        "stale_mode_ascent": "Stale-h3 mode ascent",
        "mmse_mode_ascent": "MMSE-h3 mode ascent",
        "covariance_aware_mode_ascent": "Covariance-aware mode ascent",
        "ris_ideal_mode_ascent": "Ideal RIS mode ascent",
    }
    for method, method_label in labels.items():
        values = g54["summary"]["methods"][method]
        add("G54", f"{method_label} worst over time",
            f"{values['worst_over_time']:.4f}",
            "covariance_aware_ris_gate.json")
        add("G54", f"{method_label} mean over time",
            f"{values['mean_over_time']:.4f}",
            "covariance_aware_ris_gate.json")
        add("G54", f"{method_label} QoS over time",
            f"{100.0 * values['qos_over_time']:.0f}%",
            "covariance_aware_ris_gate.json")
    add("G54", "Covariance-aware vs MMSE worst gain (negative)",
        pp(g54["summary"]["robust_over_mmse_worst_gain"]),
        "covariance_aware_ris_gate.json")
    add("G54", "Covariance-aware QoS gain (negative)",
        pp(g54["summary"]["robust_qos_gain"]),
        "covariance_aware_ris_gate.json")
    add("G54", "Verdict", g54["summary"]["verdict"],
        "covariance_aware_ris_gate.json")

    return rows


def write_table(rows: list[list[str]]) -> None:
    csv_path = RESULTS / "paper_results_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gate", "metric", "value", "source"])
        writer.writerows(rows)

    md_path = RESULTS / "paper_results_table.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Gate | Metric | Value | Source |\n")
        handle.write("| --- | --- | --- | --- |\n")
        for row in rows:
            escaped = [cell.replace("|", "\\|") for cell in row]
            handle.write("| " + " | ".join(escaped) + " |\n")


def draw_g4_budget() -> None:
    g4 = load("expected_pd_greedy_gate.json")
    budgets = [s["budget_bits"] for s in g4["summary"]]
    proposed = [s["proposed_mean"] for s in g4["summary"]]
    expected = [s["expected_pd_mean"] for s in g4["summary"]]
    hybrid = [s["hybrid_mean"] for s in g4["summary"]]
    topk = [s["topk_mean"] for s in g4["summary"]]
    all_scheduled = [s["all_scheduled_mean"] for s in g4["summary"]]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(budgets, topk, "o--", label="Static Top-K", color="#6b7280")
    ax.plot(budgets, proposed, "s--", label="Proposed greedy", color="#2563eb")
    ax.plot(budgets, expected, "^--", label="Expected-P_D greedy", color="#16a34a")
    ax.plot(budgets, hybrid, "D-", label="Hybrid", color="#dc2626")
    ax.plot(budgets, all_scheduled, ":", label="All scheduled", color="#7c3aed")
    ax.set_xlabel("Report budget B (bits)")
    ax.set_ylabel("Mean expected P_D")
    ax.set_title("Gate G4: expected-P_D selection under correlated erasures")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "g4_pd_vs_budget.png", dpi=200)
    plt.close(fig)


def draw_g5_budget() -> None:
    g5 = load("ris_isac_gate.json")
    budgets = [s["budget_bits"] for s in g5["summary"]]
    no_ris = [s["no_ris"]["mean_expected_pd"] for s in g5["summary"]]
    random = [s["ris_random"]["mean_expected_pd"] for s in g5["summary"]]
    aligned = [s["ris_aligned"]["mean_expected_pd"] for s in g5["summary"]]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(budgets, no_ris, "o--", label="No RIS", color="#6b7280")
    ax.plot(budgets, random, "s--", label="Random RIS phase", color="#2563eb")
    ax.plot(budgets, aligned, "D-", label="Aligned RIS", color="#dc2626")
    ax.set_xlabel("Report budget B (bits)")
    ax.set_ylabel("Mean expected P_D")
    ax.set_title("Gate G5: RIS-assisted 6G sensing channel")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "g5_ris_pd_vs_budget.png", dpi=200)
    plt.close(fig)


def draw_g5q() -> None:
    g5q = load("ris_phase_resolution_gate.json")
    labels = ["Continuous", "1 bit", "2 bit", "3 bit"]
    mean_gain = []
    worst_gain = []
    for summary in g5q["summary"][:2]:
        values = [
            summary["continuous"]["mean_gain_vs_no_ris"],
            summary["bits_1"]["mean_gain_vs_no_ris"],
            summary["bits_2"]["mean_gain_vs_no_ris"],
            summary["bits_3"]["mean_gain_vs_no_ris"],
        ]
        mean_gain.append([100.0 * v for v in values])
        values = [
            summary["continuous"]["worst_gain_vs_no_ris"],
            summary["bits_1"]["worst_gain_vs_no_ris"],
            summary["bits_2"]["worst_gain_vs_no_ris"],
            summary["bits_3"]["worst_gain_vs_no_ris"],
        ]
        worst_gain.append([100.0 * v for v in values])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    x = range(len(labels))
    for index, budget in enumerate((20, 30)):
        ax = axes[index]
        ax.bar([i - 0.2 for i in x], mean_gain[index], width=0.38,
               label="Mean", color="#2563eb")
        ax.bar([i + 0.2 for i in x], worst_gain[index], width=0.38,
               label="Worst target", color="#dc2626")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"B={budget}")
        ax.grid(axis="y", alpha=0.25)
        if index == 1:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("P_D gain over no RIS (pp)")
    fig.suptitle("Gate G5-Q: finite-resolution RIS phase")
    fig.tight_layout()
    fig.savefig(FIGURES / "g5_phase_resolution_gain.png", dpi=200)
    plt.close(fig)


def draw_g5dci() -> None:
    g5dci = load("g5_deployment_ci_gate.json")
    labels = []
    means = []
    lows = []
    highs = []
    for prefix, label in (
        ("g5s", "G5-S"),
        ("g5t", "G5-T"),
        ("g5v", "G5-V"),
        ("g5w", "G5-W"),
    ):
        section = g5dci["sections"][f"{prefix}_vs_fixed_worst"]
        labels.append(label)
        means.append(100.0 * section["mean"])
        lows.append(100.0 * section["ci95"][0])
        highs.append(100.0 * section["ci95"][1])

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    y = range(len(labels))
    lower_errors = [mean - low for mean, low in zip(means, lows)]
    upper_errors = [high - mean for mean, high in zip(means, highs)]
    ax.errorbar(
        means, y, xerr=[lower_errors, upper_errors],
        fmt="o", color="#dc2626", capsize=4,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Worst-target P_D gain over fixed RIS (pp)")
    ax.set_title("Gate G5-DCI: deployment paired bootstrap intervals")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g5_deployment_ci_forest.png", dpi=200)
    plt.close(fig)


def draw_g5rf() -> None:
    g5rf = load("global_resource_fairness_gate.json")
    scenarios = []
    report_bits = []
    control_bits = []
    pd = []
    tb = []
    for summary in g5rf["summary"]:
        scenarios.append("No RIS" if summary["scenario"] == "no_ris" else "RIS")
        report_bits.append(summary["report_bits"])
        control_bits.append(summary["control_bits"])
        pd.append(summary["worst_expected_pd"])
        tb.append(summary["total_time_bandwidth"])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    x = range(len(scenarios))
    axes[0].bar(x, report_bits, label="Report bits", color="#2563eb")
    axes[0].bar(x, control_bits, bottom=report_bits, label="RIS control bits",
                color="#dc2626")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(scenarios)
    axes[0].set_ylabel("Bits per frame")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Bit ledger")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([i - 0.2 for i in x], pd, width=0.38, label="Worst P_D",
                color="#16a34a")
    axes[1].bar([i + 0.2 for i in x], tb, width=0.38, label="TB symbols",
                color="#7c3aed")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(scenarios)
    axes[1].set_ylabel("Worst expected P_D / TB symbols")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Performance and time-bandwidth")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Gate G5-RF: global resource ledger at total B=40")
    fig.tight_layout()
    fig.savefig(FIGURES / "g5_resource_ledger.png", dpi=200)
    plt.close(fig)


def draw_g5_sensitivity() -> None:
    g5sen = load("ris_sensitivity_gate.json")
    panels = (
        ("aperture_scale", "Aperture scale", True),
        ("elements", "RIS elements N", False),
        ("coherence_frames", "Coherence frames", False),
        ("direct_blockage", "Direct-path blockage", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    for ax, (parameter, xlabel, log_scale) in zip(axes.flat, panels):
        rows = [
            summary for summary in g5sen["summary"]
            if summary["parameter"] == parameter
        ]
        xs = [summary["parameter_value"] for summary in rows]
        mean_gain = [
            100.0 * summary["mean_gain_vs_no_ris"]["mean"]
            for summary in rows
        ]
        worst_gain = [
            100.0 * summary["worst_gain_vs_no_ris"]["mean"]
            for summary in rows
        ]
        ax.plot(xs, mean_gain, "o-", label="Mean", color="#2563eb")
        ax.plot(xs, worst_gain, "s--", label="Worst target", color="#dc2626")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("P_D gain over no RIS (pp)")
        ax.set_title(parameter.replace("_", " "))
        ax.grid(alpha=0.25)
        if log_scale:
            ax.set_xscale("log")
        if parameter == "aperture_scale":
            ax.legend(fontsize=8)
    fig.suptitle("Gate G5-SEN: RIS parameter sensitivity at total B=40")
    fig.tight_layout()
    fig.savefig(FIGURES / "g5_sensitivity.png", dpi=200)
    plt.close(fig)


def draw_g5_sota() -> None:
    sota = load("sota_baseline_gate.json")
    baselines = (
        ("s1_ris_deflection_topk", "RIS deflection Top-K"),
        ("s2_no_ris_deflection_topk", "No-RIS deflection Top-K"),
        ("s3_random_ris_deflection_topk", "Random RIS deflection Top-K"),
        ("s4_uniform_soft_no_ris", "Uniform soft no-RIS"),
        ("hard_no_ris", "1-bit counting no-RIS"),
        ("hard_ris", "1-bit counting RIS"),
        ("proposed_schedule_deflection", "Same schedule deflection fusion"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for axis_index, metric in enumerate(("mean", "worst")):
        ax = axes[axis_index]
        labels = []
        means = []
        lows = []
        highs = []
        for baseline, label in baselines:
            section = sota["sections"][f"proposed_vs_{baseline}_{metric}"]
            labels.append(label)
            means.append(100.0 * section["mean"])
            lows.append(100.0 * section["ci95"][0])
            highs.append(100.0 * section["ci95"][1])
        y = range(len(labels))
        lower_errors = [mean - low for mean, low in zip(means, lows)]
        upper_errors = [high - mean for mean, high in zip(means, highs)]
        ax.errorbar(
            means, y, xerr=[lower_errors, upper_errors],
            fmt="o", color="#2563eb", capsize=4,
        )
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"Proposed minus baseline, {metric} expected P_D (pp)")
        ax.set_title(metric.capitalize())
        ax.grid(alpha=0.25)
    fig.suptitle("Gate G5-SOTA: proposed chain versus literature-style baselines")
    fig.tight_layout()
    fig.savefig(FIGURES / "g5_sota_baselines.png", dpi=200)
    plt.close(fig)


def draw_g6_budget_saturation() -> None:
    g6 = load("budget_saturation_gate.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for scenario, color, label in (
        ("no_ris", "#6b7280", "No RIS"),
        ("ris", "#dc2626", "RIS"),
    ):
        rows = [
            summary for summary in g6["summary"]
            if summary["scenario"] == scenario
        ]
        budgets = [summary["total_budget_bits"] for summary in rows]
        axes[0].plot(
            budgets,
            [100.0 * summary["greedy_worst"] for summary in rows],
            "o--", color=color, label=f"{label} greedy",
        )
        axes[0].plot(
            budgets,
            [100.0 * summary["all_worst"] for summary in rows],
            ":", color=color, label=f"{label} all-scheduled",
        )
        axes[1].plot(
            budgets,
            [100.0 * summary["descent_qos_rate"] for summary in rows],
            "s-", color=color, label=label,
        )
    axes[0].axhline(85.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Total budget B (bits)")
    axes[0].set_ylabel("Worst-target expected P_D (%)")
    axes[0].set_title("Worst-target P_D versus budget")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Total budget B (bits)")
    axes[1].set_ylabel("QoS feasibility rate (%)")
    axes[1].set_title("QoS feasibility at 0.85")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Gate G6: budget saturation frontier")
    fig.tight_layout()
    fig.savefig(FIGURES / "g6_budget_saturation.png", dpi=200)
    plt.close(fig)


def draw_g7_shared_phase() -> None:
    g7 = load("ris_shared_phase_gate.json")
    scenarios = (
        ("no_ris", "No RIS", "#6b7280"),
        ("random_shared_phase", "Random shared", "#2563eb"),
        ("shared_weak_aligned", "Weak-aligned shared", "#ca8a04"),
        ("shared_system_optimized", "System-optimized shared", "#dc2626"),
        ("per_target_ideal_phase", "Per-target ideal", "#7c3aed"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scenario, label, color in scenarios:
        rows = [
            summary for summary in g7["summary"]
            if summary["scenario"] == scenario
        ]
        budgets = [summary["total_budget_bits"] for summary in rows]
        worst = [100.0 * summary["worst_expected_pd"] for summary in rows]
        ax.plot(budgets, worst, "o-", color=color, label=label)
    ax.axhline(85.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst-target expected P_D (%)")
    ax.set_title("Gate G7: shared single-phase RIS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g7_shared_phase.png", dpi=200)
    plt.close(fig)


def draw_g8_exact_quota() -> None:
    g8 = load("exact_quota_gate.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scenario, color, label in (
        ("no_ris", "#6b7280", "No RIS"),
        ("ris", "#dc2626", "RIS"),
    ):
        rows = [
            summary for summary in g8["summary"]
            if summary["scenario"] == scenario
        ]
        budgets = [summary["total_budget_bits"] for summary in rows]
        ax.plot(
            budgets,
            [100.0 * summary["greedy_worst"] for summary in rows],
            "o", color=color, label=f"{label} greedy=exact",
        )
        ax.plot(
            budgets,
            [100.0 * summary["all_worst"] for summary in rows],
            "--", color=color, label=f"{label} all-scheduled",
        )
    ax.axhline(85.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst-target expected P_D (%)")
    ax.set_title("Gate G8: exact quota selection upper bound")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g8_exact_quota.png", dpi=200)
    plt.close(fig)


def draw_g8k_exact_budget() -> None:
    g = load("exact_budget_gate.json")
    cells = g["variable_rate_system"]["by_budget"]
    x = [cell["budget_bits"] for cell in cells]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(
        x,
        [100.0 * cell["greedy_worst_mean"] for cell in cells],
        yerr=[100.0 * cell["greedy_worst_std"] for cell in cells],
        fmt="o-", color="#2563eb", label="Greedy worst P_D", capsize=3,
    )
    ax.errorbar(
        x,
        [100.0 * cell["exact_worst_mean"] for cell in cells],
        yerr=[100.0 * cell["exact_worst_std"] for cell in cells],
        fmt="s-", color="#dc2626",
        label="Exact lexicographic worst P_D", capsize=3,
    )
    ax.set_xlabel("Report budget B (bits)")
    ax.set_ylabel("Worst-target expected P_D (%)")
    ax.set_title("Gate G8-K: exact heterogeneous-cost selection (20 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g8k_exact_budget.png", dpi=200)
    plt.close(fig)


def draw_g8m_exact_maxmin() -> None:
    g = load("exact_maxmin_gate.json")
    cells = g["variable_rate_system"]["by_budget"]
    x = [cell["budget_bits"] for cell in cells]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(
        x,
        [100.0 * cell["greedy_worst_mean"] for cell in cells],
        yerr=[100.0 * cell["greedy_worst_std"] for cell in cells],
        fmt="o-", color="#2563eb", label="Greedy worst P_D", capsize=3,
    )
    ax.errorbar(
        x,
        [100.0 * cell["exact_worst_mean"] for cell in cells],
        yerr=[100.0 * cell["exact_worst_std"] for cell in cells],
        fmt="s-", color="#16a34a", label="Exact max-min worst P_D", capsize=3,
    )
    ax.set_xlabel("Report budget B (bits)")
    ax.set_ylabel("Worst-target expected P_D (%)")
    ax.set_title("Gate G8-M: exact max-min selection (20 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g8m_exact_maxmin.png", dpi=200)
    plt.close(fig)


def draw_g8s_scaled_maxmin() -> None:
    g = load("scaled_maxmin_gate.json")
    exact = [100.0 * row["exact_worst"] for row in g["controlled_rows"]]
    scaled = [100.0 * row["scaled_worst"] for row in g["controlled_rows"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(exact, scaled, s=42, color="#2563eb", zorder=3)
    bound = [min(exact + scaled), max(exact + scaled)]
    ax.plot(bound, bound, "--", color="#6b7280", label="Identity")
    ax.set_xlabel("Exact max-min worst P_D (%)")
    ax.set_ylabel("Scaled certificate worst P_D (%)")
    ax.set_title("Gate G8-S: scaled vs exact (5 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g8s_scaled_maxmin.png", dpi=200)
    plt.close(fig)


def draw_g8s_scalability_benchmark() -> None:
    g = load("scaled_maxmin_gate.json")
    cells = g["scalability_benchmark"]
    x = [cell["num_reports"] for cell in cells]
    times = [cell["wall_seconds"] for cell in cells]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, times, "o-", color="#2563eb")
    for cell in cells:
        ax.annotate(
            f"2^{cell['num_reports']} subsets",
            (cell["num_reports"], cell["wall_seconds"]),
            textcoords="offset points", xytext=(6, 8), fontsize=7,
        )
    ax.set_xlabel("Non-owner reports R")
    ax.set_ylabel("Branch-and-bound wall time (s)")
    ax.set_yscale("log")
    ax.set_title("Gate G8-S: exact certificate scaling")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g8s_scalability_benchmark.png", dpi=200)
    plt.close(fig)


def draw_g8_target_scalability() -> None:
    g = load("exact_selection_target_scalability.json")
    budgets = sorted({cell["budget_bits"] for cell in g["summary"]})
    target_counts = sorted({cell["num_targets"] for cell in g["summary"]})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = ("#2563eb", "#16a34a", "#dc2626")
    for budget, color in zip(budgets, colors):
        rows = [
            cell for cell in g["summary"]
            if cell["budget_bits"] == budget
        ]
        rows.sort(key=lambda cell: cell["num_targets"])
        ax.plot(
            [cell["num_targets"] for cell in rows],
            [cell["maxmin_wall_mean_ms"] for cell in rows],
            "o-", color=color, label=f"B={budget} bits",
        )
    ax.set_xticks(target_counts)
    ax.set_xlabel("Target count Q")
    ax.set_ylabel("Exact max-min wall time (ms)")
    ax.set_title("Gate G8-target: exact selection across target count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g8_target_scalability.png", dpi=200)
    plt.close(fig)


def draw_g43b_exact_min_majority() -> None:
    g = load("exact_min_majority_gate.json")
    rows = g["summary"]
    x = np.arange(len(rows))
    labels = [f"M={row['num_uavs']}" for row in rows]
    values = [
        row["exact_min_uavs"] if row["exact_min_uavs"] is not None else 0
        for row in rows
    ]
    colors = ["#dc2626" if row["exact_min_uavs"] is None else "#2563eb"
              for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(x, values, color=colors)
    for bar, row in zip(bars, rows):
        if row["exact_min_uavs"] is None:
            bar.set_height(1.0)
            ax.text(bar.get_x() + bar.get_width() / 2, 1.0, "infeasible",
                    ha="center", va="bottom", fontsize=8, color="#dc2626")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), f"{bar.get_height():.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Exact minimum voters")
    ax.set_title("Gate G43-B: exact minimum majority count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g43b_exact_min_majority.png", dpi=200)
    plt.close(fig)


def draw_g30e_exact_rate_certificate() -> None:
    g = load("exact_rate_certificate_gate.json")
    rows = g["summary"]
    x = np.arange(len(rows))
    g30 = [100.0 * row["g30_exact_value"] for row in rows]
    optimized = [100.0 * row["exact_optimized_value"] for row in rows]
    labels = [f"B={row['total_budget_bits']}" for row in rows]
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x - width / 2, g30, width, color="#2563eb",
           label="G30 profile (exact)")
    ax.bar(x + width / 2, optimized, width, color="#16a34a",
           label="Exact coordinate ascent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Worst-target expected P_D (%)")
    ax.set_title("Gate G30-E: exact rate certificate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g30e_exact_rate_certificate.png", dpi=200)
    plt.close(fig)


def draw_g9_subarray() -> None:
    g9 = load("ris_subarray_gate.json")
    scenarios = (
        ("no_ris", "No RIS", "#6b7280"),
        ("shared_weak_aligned", "Shared weak-aligned", "#ca8a04"),
        ("subarray_optimized", "Subarray multi-beam", "#dc2626"),
        ("per_target_ideal_phase", "Per-target ideal", "#7c3aed"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scenario, label, color in scenarios:
        rows = [
            summary for summary in g9["summary"]
            if summary["scenario"] == scenario
        ]
        budgets = [summary["total_budget_bits"] for summary in rows]
        worst = [100.0 * summary["worst_expected_pd"] for summary in rows]
        ax.plot(budgets, worst, "o-", color=color, label=label)
    ax.axhline(85.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst-target expected P_D (%)")
    ax.set_title("Gate G9: aperture-conserved subarray multi-beam")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g9_subarray_multibeam.png", dpi=200)
    plt.close(fig)


def draw_g10_subarray_steering() -> None:
    g10 = load("ris_subarray_steering_gate.json")
    scenarios = (
        ("no_ris", "No RIS", "#6b7280"),
        ("shared_weak_aligned", "Shared weak-aligned", "#ca8a04"),
        ("g9_subarray", "G9 subarray", "#2563eb"),
        ("g10_steering_optimized", "G10 steering optimized", "#dc2626"),
        ("per_target_ideal_phase", "Per-target ideal", "#7c3aed"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scenario, label, color in scenarios:
        rows = [
            summary for summary in g10["summary"]
            if summary["scenario"] == scenario
        ]
        budgets = [summary["total_budget_bits"] for summary in rows]
        worst = [100.0 * summary["worst_expected_pd"] for summary in rows]
        ax.plot(budgets, worst, "o-", color=color, label=label)
    ax.axhline(85.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst-target expected P_D (%)")
    ax.set_title("Gate G10: per-subarray steering optimization")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g10_subarray_steering.png", dpi=200)
    plt.close(fig)


def draw_g11_aperture_scaling() -> None:
    g11 = load("ris_aperture_scaling_gate.json")
    curves = (
        (1, 64, "equal", "1-bit, C=64, equal", "#2563eb"),
        (1, 256, "equal", "1-bit, C=256, equal", "#16a34a"),
        (3, 256, "equal", "3-bit, C=256, equal", "#dc2626"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for phase_bits, coherence, allocation, label, color in curves:
        rows = [
            summary for summary in g11["summary"]
            if summary["total_budget_bits"] == 20
            and summary["phase_bits"] == phase_bits
            and summary["coherence_frames"] == coherence
            and summary["allocation_name"] == allocation
        ]
        rows.sort(key=lambda row: row["ris_elements"])
        ax.plot(
            [row["ris_elements"] for row in rows],
            [100.0 * row["worst_expected_pd"] for row in rows],
            "o-", color=color, label=label,
        )
    ax.axhline(85.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("RIS elements N")
    ax.set_ylabel("Worst-target expected P_D at B=20 (%)")
    ax.set_title("Gate G11: aperture scaling under fixed total budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g11_aperture_scaling.png", dpi=200)
    plt.close(fig)


def draw_g12_derived_architecture() -> None:
    g12 = load("derived_architecture_gate.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for phase_bits, coherence, color, label in (
        (1, 64, "#2563eb", "1-bit, C=64"),
        (3, 128, "#16a34a", "3-bit, C=128"),
        (3, 256, "#dc2626", "3-bit, C=256"),
    ):
        rows = [
            summary for summary in g12["summary"]
            if summary["total_budget_bits"] == 20
            and summary["phase_bits"] == phase_bits
            and summary["coherence_frames"] == coherence
        ]
        if not rows:
            continue
        summary = rows[0]
        aperture = np.linspace(0.0, 2048.0, 401)
        kappa = summary["weak_kappa"]
        rate = phase_bits / coherence
        surrogate = [
            (1.0 + kappa * n * n) ** 2 * max(20.0 - rate * n, 0.0)
            for n in aperture
        ]
        axes[0].plot(aperture, surrogate, color=color, label=label)
        axes[0].axvline(
            summary["derived_aperture"], color=color, linestyle=":",
            linewidth=1.0,
        )
        for candidate in summary["candidates"]:
            axes[1].plot(
                candidate["num_elements"],
                100.0 * candidate["worst_expected_pd"],
                "o", color=color,
            )
    axes[0].set_xlabel("RIS elements N")
    axes[0].set_ylabel("Derived surrogate J(N), B=20")
    axes[0].set_title("First-order aperture condition")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(85.0, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Evaluated N around N*")
    axes[1].set_ylabel("Exact worst-target P_D at B=20 (%)")
    axes[1].set_title("Exact validation of derived N*")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g12_derived_architecture.png", dpi=200)
    plt.close(fig)


def draw_g13_waterfilling() -> None:
    g13 = load("waterfilling_architecture_gate.json")
    equal = [
        summary["exact"]["equal"]["worst_expected_pd"]
        for summary in g13["summary"]
    ]
    water = [
        summary["exact"]["waterfilling"]["worst_expected_pd"]
        for summary in g13["summary"]
    ]
    labels = [
        f"B{s['total_budget_bits']} N{s['num_elements']} b{s['phase_bits']}"
        for s in g13["summary"]
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([0.7, 1.0], [0.7, 1.0], "--", color="#6b7280",
            label="No gain line")
    ax.scatter(equal, water, color="#dc2626", s=48)
    for x, y, label in zip(equal, water, labels):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Equal allocation worst P_D")
    ax.set_ylabel("Waterfilling worst P_D")
    ax.set_title("Gate G13: max-min deflection water-filling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "g13_waterfilling_architecture.png", dpi=200)
    plt.close(fig)


def draw_g14_exact_allocation() -> None:
    g14 = load("exact_allocation_gate.json")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g14["summary"]
    ]
    methods = ("equal", "separable", "exact")
    colors = ("#6b7280", "#2563eb", "#dc2626")
    x = np.arange(len(g14["summary"]))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["exact_system"][method]["worst_expected_pd"]
            for summary in g14["summary"]
        ]
        ax.bar(
            x + (index - 1) * width, values, width, color=colors[index],
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Exact system worst-target P_D (%)")
    ax.set_title("Gate G14: surrogate exactness vs system performance")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g14_exact_allocation.png", dpi=200)
    plt.close(fig)


def draw_g15_system_allocation() -> None:
    g15 = load("system_allocation_gate.json")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g15["summary"]
    ]
    methods = ("equal", "separable", "exact_surrogate", "system_ascent")
    colors = ("#6b7280", "#2563eb", "#ca8a04", "#dc2626")
    x = np.arange(len(g15["summary"]))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["exact_system"][method]["worst_expected_pd"]
            for summary in g15["summary"]
        ]
        ax.bar(
            x + (index - 1.5) * width, values, width, color=colors[index],
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Exact system worst-target P_D (%)")
    ax.set_title("Gate G15: greedy-aware system-level allocation")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g15_system_allocation.png", dpi=200)
    plt.close(fig)


def draw_g16_single_move_certificate() -> None:
    g16 = load("single_move_certificate_gate.json")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g16["summary"]
    ]
    g15_values = [100.0 * s["g15_value"] for s in g16["summary"]]
    refined_values = [100.0 * s["refined_value"] for s in g16["summary"]]
    x = np.arange(len(g16["summary"]))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - width / 2, g15_values, width, label="G15 (8-element)",
           color="#2563eb")
    ax.bar(x + width / 2, refined_values, width,
           label="G16 (1-element + certificate)", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("System objective worst P_D (%)")
    ax.set_title("Gate G16: single-move refinement and local certificate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g16_single_move_certificate.png", dpi=200)
    plt.close(fig)


def draw_g17_multi_move_certificate() -> None:
    g17 = load("multi_move_certificate_gate.json")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g17["summary"]
    ]
    g16_values = [100.0 * s["g16_value"] for s in g17["summary"]]
    multi_values = [100.0 * s["multi_block_value"] for s in g17["summary"]]
    x = np.arange(len(g17["summary"]))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - width / 2, g16_values, width, label="G16 (single move)",
           color="#2563eb")
    ax.bar(x + width / 2, multi_values, width,
           label="G17 (multi-block, T<=3)", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("System objective worst P_D (%)")
    ax.set_title("Gate G17: bounded multi-block local certificate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g17_multi_move_certificate.png", dpi=200)
    plt.close(fig)


def draw_g18_joint_placement() -> None:
    g18 = load("joint_placement_allocation_gate.json")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g18["summary"]
    ]
    g17_values = [100.0 * s["g17_value"] for s in g18["summary"]]
    final_values = [100.0 * s["final_value"] for s in g18["summary"]]
    x = np.arange(len(g18["summary"]))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - width / 2, g17_values, width, label="G17 (fixed position)",
           color="#2563eb")
    ax.bar(x + width / 2, final_values, width,
           label="G18 (joint placement+allocation)", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("System objective worst P_D (%)")
    ax.set_title("Gate G18: joint placement-allocation local certificate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g18_joint_placement_allocation.png", dpi=200)
    plt.close(fig)


def draw_g19_progressive_decentralization() -> None:
    g19 = load("progressive_decentralization_gate.json")
    methods = ("centralized_full", "local_schedule_optimal",
               "local_schedule_deflection", "owner_only",
               "hard_decision_local")
    colors = ("#6b7280", "#2563eb", "#16a34a", "#ca8a04", "#dc2626")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for summary_index, summary in enumerate(g19["summary"]):
        labels = [f"B{s['total_budget_bits']}\nN{s['num_elements']}"
                  for s in g19["summary"]]
        for method_index, method in enumerate(methods):
            worst = 100.0 * summary["methods"][method]["worst"]
            qos = 100.0 * summary["methods"][method]["qos_rate"]
            axes[0].bar(
                summary_index + (method_index - 2) * 0.16, worst, 0.15,
                color=colors[method_index], label=method if summary_index == 0 else None,
            )
            axes[1].bar(
                summary_index + (method_index - 2) * 0.16, qos, 0.15,
                color=colors[method_index],
            )
    axes[0].set_xticks(range(len(g19["summary"])))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Worst P_D (%)")
    axes[0].set_title("Worst-target P_D")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(range(len(g19["summary"])))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("QoS feasibility (%)")
    axes[1].set_title("QoS at 0.85")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Gate G19: progressive decentralization")
    fig.tight_layout()
    fig.savefig(FIGURES / "g19_progressive_decentralization.png", dpi=200)
    plt.close(fig)


def draw_g20_amplified_distributed() -> None:
    g20 = load("amplified_distributed_gate.json")
    methods = ("centralized_full", "owner_only", "hard_default",
               "hard_optimized")
    colors = ("#6b7280", "#ca8a04", "#2563eb", "#dc2626")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g20["summary"]
    ]
    x = np.arange(len(g20["summary"]))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g20["summary"]
        ]
        ax.bar(
            x + (index - 1.5) * width, values, width,
            color=colors[index], label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G20: amplified distributed hard detection")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g20_amplified_distributed.png", dpi=200)
    plt.close(fig)


def draw_g21_network_decentralization() -> None:
    g21 = load("network_decentralization_gate.json")
    methods = ("centralized_soft", "hard_full_links", "hard_top3",
               "peer_majority")
    colors = ("#6b7280", "#2563eb", "#ca8a04", "#dc2626")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g21["summary"]
    ]
    x = np.arange(len(g21["summary"]))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g21["summary"]
        ]
        ax.bar(
            x + (index - 1.5) * width, values, width,
            color=colors[index], label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G21: network-level progressive decentralization")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g21_network_decentralization.png", dpi=200)
    plt.close(fig)


def draw_g22_degraded_consensus() -> None:
    g22 = load("degraded_consensus_gate.json")
    methods = ("peer_clean", "multihop_3x08", "link_08", "obs_075",
               "severe", "centralized_soft")
    colors = ("#dc2626", "#16a34a", "#2563eb", "#ca8a04", "#6b7280",
              "#7c3aed")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g22["summary"]
    ]
    x = np.arange(len(g22["summary"]))
    width = 0.13
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g22["summary"]
        ]
        ax.bar(
            x + (index - 2.5) * width, values, width,
            color=colors[index], label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G22: degraded multi-hop consensus")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g22_degraded_consensus.png", dpi=200)
    plt.close(fig)


def draw_g23_correlated_consensus() -> None:
    g23 = load("correlated_consensus_gate.json")
    methods = ("peer_clean", "common_fail_02", "common_fail_04",
               "heterogeneous_obs", "severe_combined", "centralized_soft")
    colors = ("#dc2626", "#f97316", "#ca8a04", "#2563eb", "#6b7280",
              "#7c3aed")
    labels = [
        f"B{s['total_budget_bits']}\nN{s['num_elements']}"
        for s in g23["summary"]
    ]
    x = np.arange(len(g23["summary"]))
    width = 0.13
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g23["summary"]
        ]
        ax.bar(
            x + (index - 2.5) * width, values, width,
            color=colors[index], label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G23: correlated failure and heterogeneous consensus")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g23_correlated_consensus.png", dpi=200)
    plt.close(fig)


def draw_g24_scalability_comparison() -> None:
    g24 = load("scalability_comparison_gate.json")
    methods = (
        ("no_ris_worst", "No RIS", "#6b7280"),
        ("ris_ideal_worst", "RIS ideal", "#dc2626"),
        ("peer_majority_worst", "Peer majority", "#2563eb"),
    )
    target_counts = sorted({row["num_targets"] for row in g24["summary"]})
    fig, axes = plt.subplots(1, len(target_counts), figsize=(12.0, 4.0),
                             sharey=True)
    for axis, num_targets in zip(axes, target_counts):
        rows = [
            row for row in g24["summary"]
            if row["num_targets"] == num_targets
        ]
        rows.sort(key=lambda row: row["num_uavs"])
        for key, label, color in methods:
            axis.plot(
                [row["num_uavs"] for row in rows],
                [100.0 * row[key] for row in rows],
                "o-", color=color, label=label,
            )
        axis.axhline(85.0, color="black", linestyle=":", linewidth=0.8)
        axis.set_xlabel("UAV count M")
        axis.set_title(f"Q={num_targets}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Worst-target P_D (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gate G24: scalability comparison across M and Q")
    fig.tight_layout()
    fig.savefig(FIGURES / "g24_scalability_comparison.png", dpi=200)
    plt.close(fig)


def draw_g25_scaled_g18_scalability() -> None:
    g25 = load("scaled_g18_scalability_gate.json")
    methods = (
        ("no_ris_worst", "No RIS", "#6b7280"),
        ("ris_ideal_worst", "RIS ideal", "#7c3aed"),
        ("scaled_g18_worst", "Scaled G18", "#dc2626"),
        ("peer_majority_worst", "Peer majority", "#2563eb"),
    )
    target_counts = sorted({row["num_targets"] for row in g25["summary"]})
    fig, axes = plt.subplots(1, len(target_counts), figsize=(12.0, 4.0),
                             sharey=True)
    for axis, num_targets in zip(axes, target_counts):
        rows = [
            row for row in g25["summary"]
            if row["num_targets"] == num_targets
        ]
        rows.sort(key=lambda row: row["num_uavs"])
        for key, label, color in methods:
            axis.plot(
                [row["num_uavs"] for row in rows],
                [100.0 * row[key] for row in rows],
                "o-", color=color, label=label,
            )
        axis.axhline(85.0, color="black", linestyle=":", linewidth=0.8)
        axis.set_xlabel("UAV count M")
        axis.set_title(f"Q={num_targets}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Worst-target P_D (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gate G25: scaled white-box G18 across M and Q")
    fig.tight_layout()
    fig.savefig(FIGURES / "g25_scaled_g18_scalability.png", dpi=200)
    plt.close(fig)


def draw_g26_mobility_blockage() -> None:
    g26 = load("mobility_blockage_gate.json")
    methods = ("no_ris", "ris_static_subarray", "ris_adaptive_subarray",
               "ris_ideal")
    labels = ["No RIS", "Static subarray", "Adaptive subarray", "RIS ideal"]
    colors = ("#6b7280", "#2563eb", "#dc2626", "#7c3aed")
    x = np.arange(len(methods))
    worst = [
        100.0 * g26["summary"]["methods"][method]["worst_over_time"]
        for method in methods
    ]
    qos = [
        100.0 * g26["summary"]["methods"][method]["qos_over_time"]
        for method in methods
    ]
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, worst, width, color=colors, label="Worst P_D")
    ax.bar(x + width / 2, qos, width, color=colors, alpha=0.55,
           label="QoS over time")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Percent")
    ax.set_title("Gate G26: mobility and time-varying blockage")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g26_mobility_blockage.png", dpi=200)
    plt.close(fig)


def draw_g27_multi_ris() -> None:
    g27 = load("multi_ris_gate.json")
    methods = ("one_ris", "two_ris", "three_ris")
    labels = ["1 RIS", "2 RIS", "3 RIS"]
    colors = ("#2563eb", "#ca8a04", "#dc2626")
    x = np.arange(len(g27["summary"]))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g27["summary"]
        ]
        ax.bar(
            x + (index - 1) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={s['total_budget_bits']}" for s in g27["summary"]])
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G27: multi-RIS deployment under fixed total aperture")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g27_multi_ris.png", dpi=200)
    plt.close(fig)


def draw_g28_multi_ris_split() -> None:
    g28 = load("multi_ris_split_optimization_gate.json")
    summary = g28["summary"]
    labels = ["1 RIS", "2 RIS equal split", "2 RIS optimized"]
    values = [
        100.0 * summary["one_ris_value"],
        100.0 * summary["equal_split_value"],
        100.0 * summary["optimized_value"],
    ]
    colors = ("#6b7280", "#2563eb", "#dc2626")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G28: multi-RIS split/placement optimization")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g28_multi_ris_split.png", dpi=200)
    plt.close(fig)


def draw_g29_variable_rate() -> None:
    g29 = load("variable_rate_report_gate.json")
    methods = ("soft5", "soft3", "adaptive_soft", "hard1")
    labels = ["5-bit", "3-bit", "Adaptive", "1-bit hard"]
    colors = ("#6b7280", "#2563eb", "#dc2626", "#ca8a04")
    x = np.arange(len(g29["summary"]))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g29["summary"]
        ]
        ax.bar(
            x + (index - 1.5) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={s['total_budget_bits']}" for s in g29["summary"]])
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G29: variable-rate soft/hard reporting")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g29_variable_rate.png", dpi=200)
    plt.close(fig)


def draw_g30_global_rate() -> None:
    g30 = load("global_rate_optimization_gate.json")
    methods = ("fixed3", "fixed5", "adaptive", "optimized_value")
    labels = ["3-bit", "5-bit", "Adaptive", "Optimized"]
    colors = ("#6b7280", "#2563eb", "#ca8a04", "#dc2626")
    x = np.arange(len(g30["summary"]))
    width = 0.2
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for index, method in enumerate(methods):
        key = "optimized_value" if method == "optimized_value" else method
        values = [100.0 * summary[key] for summary in g30["summary"]]
        ax.bar(
            x + (index - 1.5) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={s['total_budget_bits']}" for s in g30["summary"]])
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G30: global rate-profile optimization")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g30_global_rate.png", dpi=200)
    plt.close(fig)


def draw_g31_hybrid_fusion() -> None:
    g31 = load("hybrid_fusion_gate.json")
    methods = ("soft5", "hybrid", "hard1")
    labels = ["Soft 5-bit", "Hybrid", "Hard 1-bit"]
    colors = ("#2563eb", "#dc2626", "#ca8a04")
    x = np.arange(len(g31["summary"]))
    width = 0.28
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g31["summary"]
        ]
        ax.bar(
            x + (index - 1) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={s['total_budget_bits']}" for s in g31["summary"]])
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G31: exact soft/hard hybrid fusion")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g31_hybrid_fusion.png", dpi=200)
    plt.close(fig)


def draw_g32_interference_sensitivity() -> None:
    g32 = load("interference_sensitivity_gate.json")
    methods = ("no_ris", "ris_ideal", "peer_majority")
    colors = ("#6b7280", "#dc2626", "#2563eb")
    labels = [f"{s['inr_db']:g} dB" for s in g32["summary"]]
    x = np.arange(len(g32["summary"]))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g32["summary"]
        ]
        ax.bar(
            x + (index - 1) * width, values, width,
            color=colors[index], label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G32: interference sensitivity")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g32_interference_sensitivity.png", dpi=200)
    plt.close(fig)


def draw_g33_spatial_interference() -> None:
    g33 = load("spatial_interference_placement_gate.json")
    labels = [f"{s['inr_ref']:g}" for s in g33["summary"]]
    methods = ("no_ris_worst", "fixed_ris_worst", "optimized_ris_worst")
    names = ("No RIS", "Fixed RIS", "Optimized RIS")
    colors = ("#6b7280", "#2563eb", "#dc2626")
    x = np.arange(len(g33["summary"]))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, (method, name) in enumerate(zip(methods, names)):
        values = [100.0 * summary[method] for summary in g33["summary"]]
        ax.bar(
            x + (index - 1) * width, values, width,
            color=colors[index], label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Reference INR")
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G33: spatial interference and RIS placement")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g33_spatial_interference.png", dpi=200)
    plt.close(fig)


def draw_g34_multi_interference() -> None:
    g34 = load("multi_interference_placement_gate.json")
    summary = g34["summary"]
    labels = ["No RIS", "Fixed RIS", "Optimized RIS"]
    values = [
        100.0 * summary["no_ris_worst"],
        100.0 * summary["fixed_ris_worst"],
        100.0 * summary["optimized_ris_worst"],
    ]
    colors = ("#6b7280", "#2563eb", "#dc2626")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G34: multi-interference placement")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g34_multi_interference.png", dpi=200)
    plt.close(fig)


def draw_g35_ula_vs_upd() -> None:
    g35 = load("upd_vs_ula_gate.json")
    scenarios = sorted({s["scenario"] for s in g35["summary"]})
    methods = ("no_ris", "ula", "upa")
    labels = ["No RIS", "ULA", "UPA"]
    colors = ("#6b7280", "#2563eb", "#dc2626")
    x = np.arange(len(scenarios))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * max(
                s["methods"][method]["worst_expected_pd"]
                for s in g35["summary"]
                if s["scenario"] == scenario
            )
            for scenario in scenarios
        ]
        ax.bar(
            x + (index - 1) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G35: 1-D ULA vs 2-D UPA")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g35_ula_vs_upd.png", dpi=200)
    plt.close(fig)


def draw_g36_null_steering() -> None:
    g36 = load("null_steering_gate.json")
    methods = ("no_ris", "aligned", "null_steered")
    labels = ["No RIS", "Aligned UPA", "Null-steered UPA"]
    colors = ("#6b7280", "#2563eb", "#dc2626")
    x = np.arange(len(g36["summary"]))
    width = 0.28
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g36["summary"]
        ]
        ax.bar(
            x + (index - 1) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={s['total_budget_bits']}" for s in g36["summary"]])
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G36: UPA null-steering")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g36_null_steering.png", dpi=200)
    plt.close(fig)


def draw_g37_quantized_null_steering() -> None:
    g37 = load("quantized_null_steering_gate.json")
    methods = ("aligned_quantized", "continuous_quantized",
               "quantized_optimized")
    labels = ["Aligned quantized", "Continuous then quantized",
              "Quantized optimized"]
    colors = ("#6b7280", "#2563eb", "#dc2626")
    x = np.arange(len(g37["summary"]))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g37["summary"]
        ]
        ax.bar(
            x + (index - 1) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={s['total_budget_bits']}" for s in g37["summary"]])
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G37: directly quantized null-steering")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g37_quantized_null_steering.png", dpi=200)
    plt.close(fig)


def draw_g38_joint_null_placement() -> None:
    g38 = load("joint_null_placement_gate.json")
    summary = g38["summary"]
    labels = ["No RIS", "Fixed nulling", "Joint optimized"]
    values = [
        100.0 * summary["no_ris_worst"],
        100.0 * summary["fixed_value"],
        100.0 * summary["optimized_value"],
    ]
    colors = ("#6b7280", "#2563eb", "#dc2626")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G38: joint quantized nulling and placement")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g38_joint_null_placement.png", dpi=200)
    plt.close(fig)


def draw_g39_distributed_relaxation() -> None:
    g39 = load("distributed_relaxation_gate.json")
    methods = ("centralized_soft", "peer_clean", "peer_multihop",
               "hard_optimized")
    labels = ["Centralized", "Peer clean", "Peer multi-hop", "Hard"]
    colors = ("#6b7280", "#2563eb", "#16a34a", "#ca8a04")
    budgets = sorted({s["total_budget_bits"] for s in g39["summary"]})
    qos_targets = sorted({s["qos_target"] for s in g39["summary"]})
    fig, axes = plt.subplots(1, len(budgets), figsize=(12.0, 4.0),
                             sharey=True)
    for axis, budget in zip(axes, budgets):
        x = np.arange(len(qos_targets))
        width = 0.2
        for index, method in enumerate(methods):
            values = [
                100.0 * summary["methods"][method]["worst_expected_pd"]
                for summary in g39["summary"]
                if summary["total_budget_bits"] == budget
            ]
            axis.bar(
                x + (index - 1.5) * width, values, width,
                color=colors[index], label=labels[index],
            )
        axis.set_xticks(x)
        axis.set_xticklabels([f"{target:.2f}" for target in qos_targets])
        axis.set_xlabel("QoS target")
        axis.set_title(f"B={budget}")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Worst P_D (%)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Gate G39: distributed features under relaxed thresholds")
    fig.tight_layout()
    fig.savefig(FIGURES / "g39_distributed_relaxation.png", dpi=200)
    plt.close(fig)


def draw_g40_low_budget_snr_distributed() -> None:
    g40 = load("low_budget_snr_distributed_gate.json")
    methods = ("centralized_soft", "peer_clean", "peer_multihop",
               "hard_optimized")
    labels = ["Centralized", "Peer clean", "Peer multi-hop", "Hard"]
    colors = ("#6b7280", "#2563eb", "#16a34a", "#ca8a04")
    x = np.arange(len(g40["summary"]))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, method in enumerate(methods):
        values = [
            100.0 * summary["methods"][method]["worst_expected_pd"]
            for summary in g40["summary"]
        ]
        ax.bar(
            x + (index - 1.5) * width, values, width,
            color=colors[index], label=labels[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={s['total_budget_bits']}" for s in g40["summary"]])
    ax.axhline(70.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Worst-target P_D (%)")
    ax.set_title("Gate G40: low-budget/low-SNR distributed")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g40_low_budget_snr_distributed.png", dpi=200)
    plt.close(fig)


def draw_g41_consensus_parity() -> None:
    g41 = load("consensus_parity_boundary_gate.json")
    uav_counts = sorted({s["num_uavs"] for s in g41["summary"]})
    budgets = sorted({s["total_budget_bits"] for s in g41["summary"]})
    wins = np.zeros((len(uav_counts), len(budgets)), dtype=float)
    for index, m in enumerate(uav_counts):
        for j, budget in enumerate(budgets):
            rows = [
                s for s in g41["summary"]
                if s["num_uavs"] == m and s["total_budget_bits"] == budget
            ]
            wins[index, j] = 1.0 if rows and rows[0]["consensus_wins"] else 0.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    image = ax.imshow(wins, cmap="RdYlGn", vmin=0, vmax=1,
                      aspect="auto")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([f"B={b}" for b in budgets])
    ax.set_yticks(range(len(uav_counts)))
    ax.set_yticklabels([f"M={m}" for m in uav_counts])
    ax.set_title("Gate G41: consensus wins over centralized")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "g41_consensus_parity.png", dpi=200)
    plt.close(fig)


def draw_g42_optimized_parity() -> None:
    g42 = load("optimized_parity_boundary_gate.json")
    fixed = [summary["m_fixed"] for summary in g42["summary"]]
    optimized = [summary["m_optimized"] for summary in g42["summary"]]
    labels = [
        f"M={s['num_uavs']} B={s['total_budget_bits']}"
        for s in g42["summary"]
    ]
    x = np.arange(len(g42["summary"]))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(x, fixed, "o-", label="Fixed local P_FA", color="#6b7280")
    ax.plot(x, optimized, "s-", label="Optimized local P_FA", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_ylabel("Theoretical M_min")
    ax.set_title("Gate G42: optimized local threshold boundary")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g42_optimized_parity.png", dpi=200)
    plt.close(fig)


def draw_g43_exact_parity() -> None:
    g43 = load("exact_parity_boundary_gate.json")
    labels = [f"M={s['num_uavs']}" for s in g43["summary"]]
    gaussian = [s["gaussian_min"] for s in g43["summary"]]
    exact = [
        0.0 if s["exact_min_feasible"] == float("inf")
        else s["exact_min_feasible"]
        for s in g43["summary"]
    ]
    x = np.arange(len(g43["summary"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - 0.2, gaussian, width=0.4, label="Gaussian M_min",
           color="#2563eb")
    ax.bar(x + 0.2, exact, width=0.4, label="Exact feasible M",
           color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("M")
    ax.set_title("Gate G43: exact Poisson-binomial parity boundary")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g43_exact_parity.png", dpi=200)
    plt.close(fig)


def draw_g44_fundamental_information() -> None:
    g44 = load("fundamental_information_gate.json")
    points = []
    for summary in g44["summary"]:
        points.append(("Soft", summary["soft_info_norm"], summary["soft_pd"]))
        points.append(("Hard", summary["hard_info_norm"], summary["hard_pd"]))
        points.append(("Peer", summary["peer_info_norm"], summary["peer_pd"]))
    colors = {"Soft": "#2563eb", "Hard": "#ca8a04", "Peer": "#dc2626"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method in ("Soft", "Hard", "Peer"):
        xs = [point[1] for point in points if point[0] == method]
        ys = [100.0 * point[2] for point in points if point[0] == method]
        ax.plot(xs, ys, "o-", color=colors[method], label=method)
    ax.set_xlabel("Normalized information budget")
    ax.set_ylabel("Worst P_D (%)")
    ax.set_title("Gate G44: information-budget view")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g44_fundamental_information.png", dpi=200)
    plt.close(fig)


def draw_g45_resource_information_law() -> None:
    g45 = load("resource_information_law_gate.json")
    predicted = [summary["predicted_pd"] for summary in g45["summary"]]
    exact = [summary["exact_pd"] for summary in g45["summary"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(exact, predicted, color="#dc2626", s=40)
    ax.plot([0.6, 1.0], [0.6, 1.0], "--", color="#6b7280",
            label="Perfect law")
    ax.set_xlabel("Exact worst P_D")
    ax.set_ylabel("Predicted P_D (closed form)")
    ax.set_title("Gate G45: closed-form resource-information law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g45_resource_information_law.png", dpi=200)
    plt.close(fig)


def draw_g46_exact_information_budget() -> None:
    g46 = load("exact_information_budget_gate.json")
    fig, (ax_raw, ax_exact) = plt.subplots(
        1, 2, figsize=(10.6, 4.0), sharey=True
    )
    colors = {"soft": "#2563eb", "hard": "#ca8a04", "peer": "#dc2626"}
    markers = {"soft": "o", "hard": "^", "peer": "D"}
    for method, label in (
        ("soft", "Soft"),
        ("hard", "Hard"),
        ("peer", "Peer consensus"),
    ):
        raw_x = [summary[f"{method}_rho_raw"] for summary in g46["summary"]]
        exact_x = [summary[f"{method}_rho_exact"] for summary in g46["summary"]]
        ys = [100.0 * summary[f"{method}_pd"] for summary in g46["summary"]]
        ax_raw.plot(
            raw_x, ys, marker=markers[method], linestyle="--" if method != "peer" else "None",
            color=colors[method], label=label,
        )
        ax_exact.plot(
            exact_x, ys, marker=markers[method], linestyle="--" if method != "peer" else "None",
            color=colors[method], label=label,
        )
    ax_raw.set_xlabel("Raw normalized information $\\rho_{raw}$")
    ax_raw.set_ylabel("Worst $P_D$ (%)")
    ax_raw.set_title("Gate G46: raw coordinate")
    ax_raw.grid(alpha=0.25)
    ax_exact.set_xlabel("Exact effective information $\\rho_{exact}$")
    ax_exact.set_title("Gate G46: exact coordinate")
    ax_exact.grid(alpha=0.25)
    ax_exact.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "g46_exact_information_budget.png", dpi=200)
    plt.close(fig)


def draw_g47_architecture_switch() -> None:
    g47 = load("architecture_switch_gate.json")
    budgets = [summary["total_budget_bits"] for summary in g47["summary"]]
    soft = [summary["soft_worst_pd"] for summary in g47["summary"]]
    peer = [summary["peer_worst_pd"] for summary in g47["summary"]]
    exact = [summary["exact_switch_worst_pd"] for summary in g47["summary"]]
    fixed = [summary["fixed_switch_worst_pd"] for summary in g47["summary"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(budgets, soft, "o--", color="#2563eb", label="Centralized soft")
    ax.plot(budgets, peer, "s:", color="#dc2626", label="Peer consensus")
    ax.plot(budgets, exact, "^-", color="#16a34a", label="Exact switch")
    ax.plot(budgets, fixed, "D-", color="#7c3aed", label="Fixed threshold")
    ax.axvspan(0, 10, color="#f59e0b", alpha=0.08, label="Peer-preferred")
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst $P_D$")
    ax.set_title("Gate G47: centralized/distributed architecture switch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g47_architecture_switch.png", dpi=200)
    plt.close(fig)


def draw_g48_target_wise_switch() -> None:
    g48 = load("target_wise_architecture_switch_gate.json")
    budgets = [summary["total_budget_bits"] for summary in g48["summary"]]
    soft = [summary["soft_worst_pd"] for summary in g48["summary"]]
    peer = [summary["peer_worst_pd"] for summary in g48["summary"]]
    global_switch = [
        summary["global_switch_worst_pd"] for summary in g48["summary"]
    ]
    target_wise = [
        summary["target_wise_switch_worst_pd"] for summary in g48["summary"]
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(budgets, soft, "o--", color="#2563eb", label="Centralized soft")
    ax.plot(budgets, peer, "s:", color="#dc2626", label="Peer consensus")
    ax.plot(budgets, global_switch, "^-", color="#16a34a",
            label="Global switch")
    ax.plot(budgets, target_wise, "D-", color="#7c3aed",
            label="Target-wise switch")
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst $P_D$")
    ax.set_title("Gate G48: target-wise architecture switch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g48_target_wise_switch.png", dpi=200)
    plt.close(fig)


def draw_g49_soft_reallocation() -> None:
    g49 = load("soft_reallocation_gate.json")
    budgets = [summary["total_budget_bits"] for summary in g49["summary"]]
    soft = [summary["soft_worst_pd"] for summary in g49["summary"]]
    peer = [summary["peer_worst_pd"] for summary in g49["summary"]]
    target_wise = [
        summary["target_wise_switch_worst_pd"] for summary in g49["summary"]
    ]
    realloc = [
        summary["reallocation_worst_pd"] for summary in g49["summary"]
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(budgets, soft, "o--", color="#2563eb", label="Centralized soft")
    ax.plot(budgets, peer, "s:", color="#dc2626", label="Peer consensus")
    ax.plot(budgets, target_wise, "^-", color="#16a34a",
            label="Target-wise switch")
    ax.plot(budgets, realloc, "D-", color="#7c3aed",
            label="Reallocation")
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst $P_D$")
    ax.set_title("Gate G49: soft-report reallocation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g49_soft_reallocation.png", dpi=200)
    plt.close(fig)


def draw_g50_mode_ascent() -> None:
    g50 = load("mode_ascent_gate.json")
    budgets = [summary["total_budget_bits"] for summary in g50["summary"]]
    soft = [summary["soft_worst_pd"] for summary in g50["summary"]]
    peer = [summary["peer_worst_pd"] for summary in g50["summary"]]
    target_wise = [
        summary["target_wise_switch_worst_pd"] for summary in g50["summary"]
    ]
    ascent = [
        summary["mode_ascent_worst_pd"] for summary in g50["summary"]
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(budgets, soft, "o--", color="#2563eb", label="Centralized soft")
    ax.plot(budgets, peer, "s:", color="#dc2626", label="Peer consensus")
    ax.plot(budgets, target_wise, "^-", color="#16a34a",
            label="Target-wise switch")
    ax.plot(budgets, ascent, "D-", color="#7c3aed", label="Mode ascent")
    ax.set_xlabel("Total budget B (bits)")
    ax.set_ylabel("Worst $P_D$")
    ax.set_title("Gate G50: two-sided mode ascent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g50_mode_ascent.png", dpi=200)
    plt.close(fig)


def draw_g51_stochastic_mobility() -> None:
    g51 = load("stochastic_mobility_gate.json")
    labels = [
        "No-RIS\nsoft",
        "No-RIS\nmode\nascent",
        "Static RIS\nmode\nascent",
        "Latency-1\nRIS mode\nascent",
        "Ideal RIS\ntarget-wise",
        "Ideal RIS\nmode\nascent",
    ]
    order = [
        "no_ris_soft",
        "no_ris_mode_ascent",
        "ris_static_mode_ascent",
        "ris_latency_mode_ascent",
        "ris_ideal_target_wise",
        "ris_ideal_mode_ascent",
    ]
    worst = [
        100.0 * g51["summary"]["methods"][method]["worst_over_time"]
        for method in order
    ]
    qos = [
        100.0 * g51["summary"]["methods"][method]["qos_over_time"]
        for method in order
    ]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(x - 0.2, worst, width=0.4, label="Worst over time",
           color="#2563eb")
    ax.bar(x + 0.2, qos, width=0.4, label="QoS over time",
           color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Percent")
    ax.set_title("Gate G51: stochastic mobility with RIS latency")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g51_stochastic_mobility.png", dpi=200)
    plt.close(fig)


def draw_g52_prediction_aware_ris() -> None:
    g52 = load("prediction_aware_ris_gate.json")
    labels = [
        "No-RIS\nsoft",
        "Static RIS\nmode\nascent",
        "Latency-1\nRIS mode\nascent",
        "MMSE\nRIS mode\nascent",
        "Ideal RIS\ntarget-wise",
        "Ideal RIS\nmode\nascent",
    ]
    order = [
        "no_ris_soft",
        "ris_static_mode_ascent",
        "ris_latency_mode_ascent",
        "ris_mmse_mode_ascent",
        "ris_ideal_target_wise",
        "ris_ideal_mode_ascent",
    ]
    worst = [
        100.0 * g52["summary"]["methods"][method]["worst_over_time"]
        for method in order
    ]
    qos = [
        100.0 * g52["summary"]["methods"][method]["qos_over_time"]
        for method in order
    ]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(x - 0.2, worst, width=0.4, label="Worst over time",
           color="#2563eb")
    ax.bar(x + 0.2, qos, width=0.4, label="QoS over time",
           color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Percent")
    ax.set_title("Gate G52: prediction-aware RIS phase")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g52_prediction_aware_ris.png", dpi=200)
    plt.close(fig)


def draw_g53_multi_step_prediction() -> None:
    g53 = load("multi_step_prediction_gate.json")
    horizons = [row["horizon"] for row in g53["summary"]["horizons"]]
    stale = [
        100.0 * row["stale_worst_over_time"]
        for row in g53["summary"]["horizons"]
    ]
    mmse = [
        100.0 * row["mmse_worst_over_time"]
        for row in g53["summary"]["horizons"]
    ]
    error_scale = [
        100.0 * row["error_covariance_scale"]
        for row in g53["summary"]["horizons"]
    ]
    ideal = 100.0 * g53["summary"]["methods"]["ris_ideal_mode_ascent"]["worst_over_time"]
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9.6, 4.0)
    )
    ax_left.plot(horizons, stale, "o--", color="#dc2626", label="Stale phase")
    ax_left.plot(horizons, mmse, "^--", color="#2563eb", label="MMSE phase")
    ax_left.axhline(
        100.0 * g53["summary"]["oracle_horizon_worst_over_time"],
        color="#7c3aed", linestyle="-.", label="Oracle horizon",
    )
    ax_left.axhline(ideal, color="#16a34a", linestyle=":", label="Ideal phase")
    ax_left.set_xlabel("Reconfiguration horizon h")
    ax_left.set_ylabel("Worst-over-time $P_D$ (%)")
    ax_left.set_title("Gate G53: worst P_D by horizon")
    ax_left.legend(fontsize=8)
    ax_left.grid(alpha=0.25)
    ax_right.plot(horizons, error_scale, "s-", color="#7c3aed")
    ax_right.set_xlabel("Reconfiguration horizon h")
    ax_right.set_ylabel("Error covariance scale (%)")
    ax_right.set_title("AR(1) $1 - \\rho^{2h}$")
    ax_right.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g53_multi_step_prediction.png", dpi=200)
    plt.close(fig)


def draw_g54_covariance_aware_negative() -> None:
    g54 = load("covariance_aware_ris_gate.json")
    labels = [
        "No-RIS\nsoft",
        "Stale-h3\nmode\nascent",
        "MMSE-h3\nmode\nascent",
        "Covariance\naware",
        "Ideal RIS\nmode\nascent",
    ]
    order = [
        "no_ris_soft",
        "stale_mode_ascent",
        "mmse_mode_ascent",
        "covariance_aware_mode_ascent",
        "ris_ideal_mode_ascent",
    ]
    worst = [
        100.0 * g54["summary"]["methods"][method]["worst_over_time"]
        for method in order
    ]
    qos = [
        100.0 * g54["summary"]["methods"][method]["qos_over_time"]
        for method in order
    ]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(x - 0.2, worst, width=0.4, label="Worst over time",
           color="#2563eb")
    ax.bar(x + 0.2, qos, width=0.4, label="QoS over time",
           color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Percent")
    ax.set_title("Gate G54: covariance-aware phase (negative)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "g54_covariance_aware_negative.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    rows = build_rows()
    write_table(rows)
    draw_g4_budget()
    draw_g5_budget()
    draw_g5q()
    draw_g5dci()
    draw_g5rf()
    draw_g5_sensitivity()
    draw_g5_sota()
    draw_g6_budget_saturation()
    draw_g7_shared_phase()
    draw_g8_exact_quota()
    draw_g8k_exact_budget()
    draw_g8m_exact_maxmin()
    draw_g8s_scaled_maxmin()
    draw_g8s_scalability_benchmark()
    draw_g8_target_scalability()
    draw_g9_subarray()
    draw_g10_subarray_steering()
    draw_g11_aperture_scaling()
    draw_g12_derived_architecture()
    draw_g13_waterfilling()
    draw_g14_exact_allocation()
    draw_g15_system_allocation()
    draw_g16_single_move_certificate()
    draw_g17_multi_move_certificate()
    draw_g18_joint_placement()
    draw_g19_progressive_decentralization()
    draw_g20_amplified_distributed()
    draw_g21_network_decentralization()
    draw_g22_degraded_consensus()
    draw_g23_correlated_consensus()
    draw_g24_scalability_comparison()
    draw_g25_scaled_g18_scalability()
    draw_g26_mobility_blockage()
    draw_g27_multi_ris()
    draw_g28_multi_ris_split()
    draw_g29_variable_rate()
    draw_g30_global_rate()
    draw_g31_hybrid_fusion()
    draw_g32_interference_sensitivity()
    draw_g33_spatial_interference()
    draw_g34_multi_interference()
    draw_g35_ula_vs_upd()
    draw_g36_null_steering()
    draw_g37_quantized_null_steering()
    draw_g38_joint_null_placement()
    draw_g39_distributed_relaxation()
    draw_g40_low_budget_snr_distributed()
    draw_g41_consensus_parity()
    draw_g42_optimized_parity()
    draw_g43_exact_parity()
    draw_g43b_exact_min_majority()
    draw_g44_fundamental_information()
    draw_g45_resource_information_law()
    draw_g46_exact_information_budget()
    draw_g47_architecture_switch()
    draw_g48_target_wise_switch()
    draw_g49_soft_reallocation()
    draw_g50_mode_ascent()
    draw_g51_stochastic_mobility()
    draw_g52_prediction_aware_ris()
    draw_g53_multi_step_prediction()
    draw_g54_covariance_aware_negative()
    draw_g30e_exact_rate_certificate()
    print(f"wrote {len(rows)} table rows")
    print("wrote paper_results_table.csv and paper_results_table.md")
    for path in sorted(FIGURES.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
