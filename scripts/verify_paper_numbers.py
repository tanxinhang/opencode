"""Verify that paper/submission.md key numbers match the result JSONs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def load(name: str) -> dict:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tol: float = 1e-2) -> bool:
    return abs(actual - expected) <= tol


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" ({detail})" if detail else ""))
    if not condition:
        sys.exit(1)


def main() -> None:
    g3 = load("pd_optimal_fusion_gate.json")
    mono = g3["monotonicity"]
    gains = g3["gains"]
    check("G3 operating edges", mono["operating_edges"] == 1318)
    check("G3 optimal family has no decreasing edge",
          mono["optimal_decreasing_edges"] == 0)
    check("G3 deflection decreasing edges", mono["deflection_decreasing_edges"] == 258)
    check("G3 deflection decreasing rate",
          close(mono["deflection_decreasing_edge_rate"], 0.19575, 1e-3))
    check("G3 max deflection drop",
          close(mono["maximum_deflection_pd_drop"], 0.16059, 1e-3))
    check("G3 mean gain over deflection",
          close(gains["mean_pd_gain_over_deflection"], 0.00634, 1e-3))
    check("G3 max gain over deflection",
          close(gains["maximum_pd_gain_over_deflection"], 0.21163, 1e-3))

    g4 = load("expected_pd_greedy_gate.json")
    cell = next(r for r in g4["summary"] if r["budget_bits"] == 20)
    check("G4 B20 mean gain",
          close(cell["mean_gain_vs_proposed"], 0.01140, 1e-3))
    check("G4 B20 mean CI",
          close(cell["mean_gain_bootstrap_ci95"][0], 0.00470, 1e-3)
          and close(cell["mean_gain_bootstrap_ci95"][1], 0.01743, 1e-3))
    check("G4 B20 worst gain",
          close(cell["worst_gain_vs_proposed"], 0.07563, 1e-3))
    check("G4 B20 win rate", cell["win_rate_vs_proposed"] == 0.85)
    check("G4 B20 hybrid mean/worst",
          close(cell["hybrid_gain_vs_proposed"], 0.01440, 1e-3)
          and close(cell["hybrid_worst_gain_vs_proposed"], 0.06503, 1e-3))
    cell40 = next(r for r in g4["summary"] if r["budget_bits"] == 40)
    check("G4 B40 greedy mean negative",
          close(cell40["mean_gain_vs_proposed"], -0.00915, 1e-3))

    g5 = load("ris_isac_gate.json")
    row = next(r for r in g5["summary"] if r["budget_bits"] == 20)
    check("G5 B20 aligned mean gain",
          close(row["mean_gain_aligned_vs_no_ris"], 0.12329, 1e-3))
    check("G5 B20 aligned worst gain",
          close(row["worst_gain_aligned_vs_no_ris"], 0.17793, 1e-3))
    check("G5 B20 QoS rate", row["ris_aligned"]["qos_feasible_rate"] == 0.95)

    physics = load("ris_physics_gate.json")
    prow = next(
        r for r in physics["summary"]
        if r["elements"] == 1024 and abs(r["aperture_scale"] - 0.01) < 1e-9
        and r["budget_bits"] == 20
    )
    check("G5-P 1024/0.01 mean/worst gain",
          close(prow["mean_gain_aligned_vs_no_ris"], 0.16264, 1e-3)
          and close(prow["worst_gain_aligned_vs_no_ris"], 0.24777, 1e-3))
    check("G5-P QoS rate", prow["qos_feasible_rate"] == 1.0)

    g5t = load("ris_multigrid_gate.json")
    check("G5-T worst gain vs fixed",
          close(g5t["worst_gain_best_vs_fixed"], 0.09847, 1e-3))
    check("G5-T worst gain vs no RIS",
          close(g5t["worst_gain_best_vs_no_ris"], 0.19528, 1e-3))

    g9 = load("ris_subarray_gate.json")
    g9b28 = next(
        r for r in g9["summary"]
        if r["scenario"] == "subarray_optimized"
        and r["total_budget_bits"] == 28
    )
    g9_shared = next(
        r for r in g9["summary"]
        if r["scenario"] == "shared_weak_aligned"
        and r["total_budget_bits"] == 28
    )
    g9_no_ris = next(
        r for r in g9["summary"]
        if r["scenario"] == "no_ris" and r["total_budget_bits"] == 28
    )
    check("G9 B28 subarray",
          close(g9b28["worst_expected_pd"], 0.91330, 1e-3)
          and g9b28["qos_feasible_rate"] == 1.0)
    check("G9 B28 gains vs shared/no-RIS",
          close(g9b28["worst_expected_pd"] - g9_shared["worst_expected_pd"],
                0.0524, 1e-3)
          and close(g9b28["worst_expected_pd"] - g9_no_ris["worst_expected_pd"],
                    0.1366, 1e-3))

    g11 = load("ris_aperture_scaling_gate.json")
    g11row = next(
        r for r in g11["summary"]
        if r["ris_elements"] == 1024 and r["phase_bits"] == 3
        and r["coherence_frames"] == 256 and r["total_budget_bits"] == 20
        and r["allocation_name"] == "equal"
    )
    check("G11 N1024 b3 C256",
          g11row["report_budget_bits"] == 8
          and g11row["qos_feasible_rate"] == 1.0
          and close(g11row["worst_expected_pd"], 0.98167, 1e-3))

    sota = load("sota_baseline_gate.json")
    s1 = sota["sections"]["proposed_vs_s1_ris_deflection_topk_mean"]
    s1w = sota["sections"]["proposed_vs_s1_ris_deflection_topk_worst"]
    s2 = sota["sections"]["proposed_vs_s2_no_ris_deflection_topk_mean"]
    s2w = sota["sections"]["proposed_vs_s2_no_ris_deflection_topk_worst"]
    s3 = sota["sections"]["proposed_vs_s3_random_ris_deflection_topk_mean"]
    s3w = sota["sections"]["proposed_vs_s3_random_ris_deflection_topk_worst"]
    s4 = sota["sections"]["proposed_vs_s4_uniform_soft_no_ris_mean"]
    s4w = sota["sections"]["proposed_vs_s4_uniform_soft_no_ris_worst"]
    hard_no_ris = sota["sections"]["proposed_vs_hard_no_ris_mean"]
    hard_no_ris_w = sota["sections"]["proposed_vs_hard_no_ris_worst"]
    hard_ris = sota["sections"]["proposed_vs_hard_ris_mean"]
    hard_ris_w = sota["sections"]["proposed_vs_hard_ris_worst"]
    same_sched = sota["sections"]["proposed_vs_proposed_schedule_deflection_mean"]
    same_sched_w = sota["sections"]["proposed_vs_proposed_schedule_deflection_worst"]
    check("G5-SOTA RIS deflection mean",
          close(s1["mean"], 0.00684, 1e-3) and s1["win_rate"] == 1.0)
    check("G5-SOTA RIS deflection worst",
          close(s1w["mean"], 0.01628, 1e-3))
    check("G5-SOTA no-RIS deflection",
          close(s2["mean"], 0.15168, 1e-3)
          and close(s2w["mean"], 0.27483, 1e-3))
    check("G5-SOTA random-RIS deflection",
          close(s3["mean"], 0.14382, 1e-3)
          and close(s3w["mean"], 0.25410, 1e-3))
    check("G5-SOTA uniform soft",
          close(s4["mean"], 0.21850, 1e-3)
          and close(s4w["mean"], 0.46136, 1e-3))
    check("G5-SOTA hard no-RIS",
          close(hard_no_ris["mean"], 0.75668, 1e-3)
          and close(hard_no_ris_w["mean"], 0.79935, 1e-3))
    check("G5-SOTA hard RIS",
          close(hard_ris["mean"], 0.52105, 1e-3)
          and close(hard_ris_w["mean"], 0.64481, 1e-3))
    check("G5-SOTA same-schedule deflection",
          close(same_sched["mean"], 0.00249, 1e-3)
          and close(same_sched_w["mean"], 0.00447, 1e-3))

    g8k = load("exact_budget_gate.json")
    g8m = load("exact_maxmin_gate.json")
    g8s = load("scaled_maxmin_gate.json")
    check("G8-K seeds 20", g8k["seeds"] == 20)
    check("G8-M seeds 500", g8m["seeds"] == 500)
    check("G8-S seeds 20", g8s["seeds"] == 20)
    check("G8-K oracle match 100%",
          g8k["controlled"]["summary"]["oracle_match_rate"] == 1.0)
    check("G8-M oracle match 100%",
          g8m["controlled"]["summary"]["oracle_match_rate"] == 1.0)
    g8k7 = next(r for r in g8k["variable_rate_system"]["by_budget"]
                if r["budget_bits"] == 7)
    g8m7 = next(r for r in g8m["variable_rate_system"]["by_budget"]
                if r["budget_bits"] == 7)
    check("G8-K B7 gain", close(g8k7["gain_worst_mean"], 0.0257, 1e-3))
    check("G8-M B7 gain", close(g8m7["gain_worst_mean"], 0.0240, 1e-3))
    check("G8-M B7 significant",
          g8m7["gain_worst_paired_t"]["p_one_sided"] < 0.05)
    check("G8-S zero max abs error", g8s["controlled_summary"]["max_abs_error"] == 0.0)
    check("G8-S benchmark 8 rows", len(g8s["scalability_benchmark"]) == 8)
    r40 = next(r for r in g8s["scalability_benchmark"]
               if r["num_reports"] == 40)
    check("G8-S R40 exact min cost",
          r40["min_cost"] == 1 and r40["exhaustive_subsets"] == 1099511627776)

    g8d = load("scaled_difficulty_gate.json")
    check("G8-D ten difficulty cells", len(g8d["rows"]) == 10)
    check("G8-D exhaustive match",
          g8d["all_match_exhaustive"] is True)
    critical85 = next(r for r in g8d["rows"] if r["label"] == "critical-0.85")
    check("G8-D critical 0.85 needs 7 reports",
          critical85["min_cost"] == 7 and critical85["num_selected"] == 7)
    check("G8-D critical 0.85 nodes",
          critical85["nodes"] == 4225 and critical85["max_depth"] == 10)
    correlated = next(
        r for r in g8d["rows"] if r["label"] == "correlated-redundant"
    )
    check("G8-D correlated upper cuts",
          correlated["prune_upper"] == 15 and correlated["nodes"] == 113)
    check("G8-D correlated exhaustive match",
          correlated["matches_exhaustive"] is True)

    g8stats = load("exact_selection_stats.json")
    check("G8-stats 500 seeds", g8stats["seeds"] == 500)
    check("G8-stats all Holm significant",
          all(cell["holm_p_two_sided_t"] < 0.05
              for cell in g8stats["sections"]["variable_rate_system"]))

    fab = load("factorial_ablation.json")
    check("factorial 500 seeds", fab["seeds"] == 500)
    check("factorial seven factors",
          set(fab["summary"]) == {
              "full", "fusion_off", "selection_off", "ris_off",
              "quantization_off", "communication_off", "maxmin_off",
          })
    check("factorial RIS dominant",
          close(fab["summary"]["ris_off"]["worst_pd_gain_vs_full"], 0.2232, 1e-3))
    check("factorial full QoS",
          fab["summary"]["full"]["qos_rate"] == 1.0)

    hard = load("hard_maxmin_scenario.json")
    check("hard scenario 20 seeds", hard["seeds"] == 20)
    hard8 = next(r for r in hard["summary"] if r["budget_bits"] == 8)
    hard10 = next(r for r in hard["summary"] if r["budget_bits"] == 10)
    check("hard B8 gain over greedy",
          close(hard8["maxmin_gain_over_greedy_pp"], 2.445, 1e-3))
    check("hard B8 gain over lex",
          close(hard8["maxmin_gain_over_lex_pp"], 3.454, 1e-3))
    check("hard B10 gain over greedy",
          close(hard10["maxmin_gain_over_greedy_pp"], 3.262, 1e-3))

    quant = load("quantization_study.json")
    check("quantization study 10 seeds", quant["seeds"] == 10)
    quant18 = next(r for r in quant["summary"] if r["budget_bits"] == 18)
    quant20 = next(r for r in quant["summary"] if r["budget_bits"] == 20)
    quant24 = next(r for r in quant["summary"] if r["budget_bits"] == 24)
    check("quantization B18 gain",
          close(quant18["variable_gain_mean_pp"], 0.802, 1e-3))
    check("quantization B20 gain",
          close(quant20["variable_gain_mean_pp"], 3.055, 1e-3))
    check("quantization B24 gain",
          close(quant24["variable_gain_mean_pp"], -1.540, 1e-3))
    check("quantization absolute P_D reasonable",
          quant20["variable_worst_mean"] >= 0.90
          and quant24["fixed_worst_mean"] >= 0.90)
    check("quantization greedy B18 over pattern",
          close(quant18["greedy_gain_over_pattern_mean_pp"], 1.606, 1e-3))
    check("quantization greedy B24 over pattern",
          close(quant24["greedy_gain_over_pattern_mean_pp"], 0.854, 1e-3))
    check("quantization greedy absolute P_D",
          quant18["greedy_worst_mean"] >= 0.90)

    joint = load("quantization_joint_gate.json")
    check("joint gate 10 seeds", joint["seeds"] == 10)
    joint18 = next(r for r in joint["summary"] if r["budget_bits"] == 18)
    joint20 = next(r for r in joint["summary"] if r["budget_bits"] == 20)
    joint24 = next(r for r in joint["summary"] if r["budget_bits"] == 24)
    check("joint B18 over greedy",
          close(joint18["exact_over_greedy_mean_pp"], 0.894, 1e-3))
    check("joint B20 over greedy",
          close(joint20["exact_over_greedy_mean_pp"], 0.633, 1e-3))
    check("joint B24 matches greedy",
          close(joint24["exact_over_greedy_mean_pp"], 0.0, 1e-3))

    multi = load("joint_multi_gate.json")
    check("multi-target joint gate 20 seeds", multi["seeds"] == 20)
    multi14 = next(r for r in multi["summary"] if r["budget_bits"] == 14)
    multi16 = next(r for r in multi["summary"] if r["budget_bits"] == 16)
    multi18 = next(r for r in multi["summary"] if r["budget_bits"] == 18)
    check("multi-target B14 gain",
          close(multi14["joint_over_greedy_mean_pp"], 4.948, 1e-3))
    check("multi-target B16 gain",
          close(multi16["joint_over_greedy_mean_pp"], 3.611, 1e-3))
    check("multi-target B18 gain",
          close(multi18["joint_over_greedy_mean_pp"], 3.722, 1e-3))
    check("multi-target absolute P_D",
          multi14["exact_joint_pd_mean"] >= 0.80)

    scale = load("joint_scale_gate.json")
    check("joint scale 10 seeds", scale["seeds"] == 10)
    check("joint scale six Q", len(scale["target_counts"]) == 6)
    scale20 = next(r for r in scale["summary"] if r["target_count"] == 20)
    check("joint scale Q20 gain",
          close(scale20["joint_over_greedy_mean_pp"], 2.618, 1e-3))
    check("joint scale Q20 frontier",
          scale20["max_frontier_size"] == 173)
    check("joint scale Q20 DP fast",
          scale20["dp_wall_seconds_mean"] < 0.01)

    g8t = load("exact_selection_target_scalability.json")
    check("G8-target 9 summary cells", len(g8t["summary"]) == 9)
    check("G8-target all oracle match",
          all(cell["budget_oracle_match_rate"] == 1.0
              and cell["maxmin_oracle_match_rate"] == 1.0
              for cell in g8t["summary"]))
    check("G8-target all never worse",
          all(cell["budget_never_worse_rate"] == 1.0
              and cell["maxmin_never_worse_rate"] == 1.0
              for cell in g8t["summary"]))
    q5b16 = next(cell for cell in g8t["summary"]
                 if cell["num_targets"] == 5 and cell["budget_bits"] == 16)
    check("G8-target Q5 B16 wall < 500 ms",
          q5b16["maxmin_wall_mean_ms"] < 500.0)

    g25 = load("scaled_g18_scalability_gate.json")
    q6 = next(r for r in g25["summary"]
              if r["num_targets"] == 6 and r["num_uavs"] == 12)
    check("G25 Q6 M12 scaled", close(q6["scaled_g18_worst"], 0.92168, 1e-3))
    check("G25 Q6 M12 peer", close(q6["peer_majority_worst"], 0.79176, 1e-3))
    check("G25 Q6 M12 ideal", close(q6["ris_ideal_worst"], 0.93374, 1e-3))

    g47 = load("architecture_switch_gate.json")
    b8 = next(r for r in g47["summary"] if r["total_budget_bits"] == 8)
    b12 = next(r for r in g47["summary"] if r["total_budget_bits"] == 12)
    check("G47 B8 gain", close(b8["exact_gain_vs_soft"], 0.10676, 1e-3))
    check("G47 B12 gain", close(b12["exact_gain_vs_soft"], 0.05677, 1e-3))

    g48 = load("target_wise_architecture_switch_gate.json")
    b12 = next(r for r in g48["summary"] if r["total_budget_bits"] == 12)
    b16 = next(r for r in g48["summary"] if r["total_budget_bits"] == 16)
    check("G48 B12 gain vs global",
          close(b12["target_wise_gain_vs_global"], 0.00485, 1e-3))
    check("G48 B16 gain vs global",
          close(b16["target_wise_gain_vs_global"], 0.01549, 1e-3))

    g49 = load("soft_reallocation_gate.json")
    b16 = next(r for r in g49["summary"] if r["total_budget_bits"] == 16)
    b40 = next(r for r in g49["summary"] if r["total_budget_bits"] == 40)
    check("G49 B16 gain vs target-wise",
          close(b16["reallocation_gain_vs_target_wise"], 0.00754, 1e-3))
    check("Table 4 B40 reallocation",
          close(b40["reallocation_worst_pd"], 0.94123, 1e-3))

    g43 = load("exact_parity_boundary_gate.json")
    m6 = next(r for r in g43["summary"] if r["num_uavs"] == 6)
    check("G43 M6 exact feasible", m6["exact_min_feasible"] == 6.0)
    check("G43 M6 Gaussian min", close(m6["gaussian_min"], 13.36245, 1e-2))

    g43b = load("exact_min_majority_gate.json")
    mins = [r["exact_min_uavs"] for r in g43b["summary"]]
    check("G43-B exact mins", mins == [None, 14, 17, 16, 19])
    m6 = next(r for r in g43b["summary"] if r["num_uavs"] == 6)
    check("G43-B M6 voter semantics",
          m6["max_voters"] == 15 and m6["exact_min_voters"] == 14)

    g30e = load("exact_rate_certificate_gate.json")
    b28 = next(r for r in g30e["summary"] if r["total_budget_bits"] == 28)
    b40 = next(r for r in g30e["summary"] if r["total_budget_bits"] == 40)
    check("G30-E B28 exact local optimum",
          close(b28["g30_exact_value"], 0.98795, 1e-3)
          and b28["exact_single_change_local_optimal"] is True)
    check("G30-E B40 correction",
          close(b40["g30_exact_value"], 0.99114, 1e-3)
          and close(b40["exact_optimized_value"], 0.99156, 1e-3)
          and b40["greedy_certificate_false_under_exact"] is True)

    print("All paper-number checks passed.")


if __name__ == "__main__":
    main()
