"""Gate: detection-aware quantization and information-gradient allocation.

Part A (design metric correctness) sweeps the uniform-quantizer span knob
over randomized links and scores every span by three metrics: the KL mean
drift ``I+``, the Chernoff information, and the exact ``P_D(4)``.  The
gate measures how often the ``argmax`` of each metric picks a span whose
exact ``P_D(4)`` is within 1% of the best span (``I+`` is expected to fail
frequently: it ranks coarse, mass-concentrating quantizers above the
true optimum).

Part B (1-bit LLR structure) compares the single-threshold 1-bit quantizer
with the two-sided window family derived from the Gaussian LLR quadratic
(``var1 > var0`` => convex LLR => two-tail superlevel sets) in exact KL.

Part C (information gradient) allocates report bits by marginal ``I+``
(sum and max-min water-filling) and compares the realized exact worst-target
``P_D(4)`` against the exact max-min ``P_D`` allocation (exhaustive, small
scale).
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.detection_information import (
    chernoff_information,
    post_communication_likelihoods,
    sequential_pd,
)
from uav_otfs_isac.detection_quantization import (
    information_waterfilling,
    link_information_vs_bits,
    maxmin_pd_allocation,
    one_bit_kl_scan,
    quantizer_edges,
)


def make_link(rng: np.random.Generator, snr_range=(-3.0, 8.0)) -> dict:
    l_acc = 4
    snr_db = float(rng.uniform(*snr_range))
    noncentrality = l_acc * 10 ** (snr_db / 10.0)
    mu0, var0 = float(l_acc), float(l_acc)
    mu1 = mu0 + noncentrality
    var1 = var0 + 2.0 * noncentrality
    flip = float(rng.uniform(0.01, 0.15))
    success = float(rng.uniform(0.5, 0.98))
    return {"mu0": mu0, "var0": var0, "mu1": mu1, "var1": var1,
            "flip": flip, "success": success}


def link_metrics(link: dict, bits: int, span_std: float,
                 alpha: float, grid_step: float) -> dict:
    edges, values = quantizer_edges(
        link["mu0"], link["var0"], link["mu1"], link["var1"],
        bits, span_std,
    )
    info = post_communication_likelihoods(
        link["mu0"], link["var0"], link["mu1"], link["var1"],
        edges, values, bits, link["flip"], link["success"],
    )
    return {
        "i_plus": float(info["kl_plus"]),
        "chernoff": float(info["chernoff"]),
        "pd4": float(sequential_pd(
            info["p1_y"], info["p0_y"], 4, alpha, grid_step,
        )["pd"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/detection_quantization_gate.json")
    parser.add_argument("--instances", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--spans", type=int, nargs="+",
                        default=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
                                 16.0, 20.0, 24.0])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--alloc-instances", type=int, default=12)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---------- Part A: design-metric correctness on the span knob ----------
    span_opt_counts = {"i_plus": 0, "chernoff": 0, "pd4": 0}
    i_plus_fail = 0
    chernoff_fail = 0
    i_plus_span_dist = []
    chernoff_span_dist = []
    for _ in range(args.instances):
        link = make_link(rng)
        metrics = {
            span: link_metrics(link, args.bits, span, args.alpha, args.grid_step)
            for span in args.spans
        }
        spans = list(args.spans)
        best_pd4_span = spans[int(np.argmax([metrics[s]["pd4"] for s in spans]))]
        best_iplus_span = spans[int(np.argmax([metrics[s]["i_plus"] for s in spans]))]
        best_chernoff_span = spans[int(np.argmax([metrics[s]["chernoff"] for s in spans]))]
        span_opt_counts["pd4"] += 1
        span_opt_counts["i_plus"] += int(best_iplus_span == best_pd4_span)
        span_opt_counts["chernoff"] += int(best_chernoff_span == best_pd4_span)
        if metrics[best_iplus_span]["pd4"] < 0.99 * metrics[best_pd4_span]["pd4"]:
            i_plus_fail += 1
        if metrics[best_chernoff_span]["pd4"] < 0.99 * metrics[best_pd4_span]["pd4"]:
            chernoff_fail += 1
        i_plus_span_dist.append(best_iplus_span)
        chernoff_span_dist.append(best_chernoff_span)

    # ---------- Part B: 1-bit LLR structure (window vs single) ----------
    window_gains = []
    structures = {}
    for _ in range(20):
        link = make_link(rng)
        scan = one_bit_kl_scan(
            link["mu0"], link["var0"], link["mu1"], link["var1"],
            link["flip"], link["success"], grid=201,
        )
        window_gains.append(float(scan["window_gain_over_single"]))
        structures[scan["llr_structure"]["kind"]] = (
            structures.get(scan["llr_structure"]["kind"], 0) + 1
        )

    # ---------- Part C: information-gradient allocation ----------
    alloc_rows = []
    for _ in range(args.alloc_instances):
        targets = [{"reports": [make_link(rng) for _ in range(2)]}
                   for _ in range(3)]
        # per-target per-report I+ profiles
        profiles_per_report = [
            [link_information_vs_bits(
                r["mu0"], r["var0"], r["mu1"], r["var1"],
                r["flip"], r["success"], 5,
            ) for r in t["reports"]]
            for t in targets
        ]
        # sum-I+ water filling over all (target, report) profiles
        flat = [p for t in profiles_per_report for p in t]
        bits_sum, _ = information_waterfilling(flat, args.budget)
        # max-min I+ water filling
        bits_min, _ = information_waterfilling(flat, args.budget, max_min=True)

        def option_metrics(combo):
            """Worst-target exact P_D(4) and worst-target Chernoff of a
            flat bit assignment ``combo`` (6 entries: 3 targets x 2 reports)."""
            worst_pd = 0.0
            worst_chernoff = 0.0
            for t in range(3):
                best_pd = 0.0
                best_c = 0.0
                for r in range(2):
                    bits = combo[t * 2 + r]
                    if bits == 0:
                        continue
                    link = targets[t]["reports"][r]
                    edges, values = quantizer_edges(
                        link["mu0"], link["var0"], link["mu1"], link["var1"],
                        bits, 4.0,
                    )
                    info = post_communication_likelihoods(
                        link["mu0"], link["var0"], link["mu1"], link["var1"],
                        edges, values, bits, link["flip"], link["success"],
                    )
                    pd = float(sequential_pd(
                        info["p1_y"], info["p0_y"], 4, args.alpha,
                        args.grid_step,
                    )["pd"])
                    c = float(chernoff_information(info["p1_y"], info["p0_y"]))
                    best_pd = max(best_pd, pd)
                    best_c = max(best_c, c)
                worst_pd = min(worst_pd, best_pd) if t > 0 else best_pd
                worst_chernoff = min(worst_chernoff, best_c) if t > 0 else best_c
            return worst_pd, worst_chernoff

        def to_assignment(bits_list):
            return [int(b) for b in bits_list]

        worst_sum, _ = option_metrics(to_assignment(bits_sum))
        worst_min, _ = option_metrics(to_assignment(bits_min))
        # floor-cover exact allocation (the theorem: min-cost coverage of
        # the worst-target floor, no concavity assumption); bits are capped
        # at 5 per option to match the exhaustive reference and the
        # water-filling profiles
        floor_targets = [
            [dict(r, bits_max=5) for r in t["reports"]] for t in targets
        ]
        floor = maxmin_pd_allocation(floor_targets, args.budget, metric="pd")
        # exact max-min P_D and exact max-min Chernoff (exhaustive).
        # Bits run 0..5 per report, matching the water-filling profiles, so
        # the exhaustive reference is a fair ground truth (the previous
        # 0..3 cap biased the comparison against the water-fillers).
        options = list(product(range(6), repeat=6))
        best_worst_pd = 0.0
        best_worst_c = 0.0
        worst_pd_of_chernoff_opt = 0.0
        for combo in options:
            if sum(combo) > args.budget:
                continue
            w_pd, w_c = option_metrics(combo)
            if w_pd > best_worst_pd:
                best_worst_pd = w_pd
            if w_c > best_worst_c:
                best_worst_c = w_c
                worst_pd_of_chernoff_opt = w_pd
        alloc_rows.append({
            "sum_i_plus_worst_pd": worst_sum,
            "maxmin_i_plus_worst_pd": worst_min,
            "floor_cover_worst_pd": float(floor["worst_metric"]),
            "exact_maxmin_worst_pd": best_worst_pd,
            "exact_maxmin_chernoff": best_worst_c,
            "worst_pd_of_chernoff_opt": worst_pd_of_chernoff_opt,
        })

    summary_a = {
        "instances": args.instances,
        "span_agreement_with_pd4_opt": span_opt_counts,
        "i_plus_failure_fraction": i_plus_fail / args.instances,
        "chernoff_failure_fraction": chernoff_fail / args.instances,
        "i_plus_span_distribution": {
            str(s): int(i_plus_span_dist.count(s)) for s in args.spans
        },
        "chernoff_span_distribution": {
            str(s): int(chernoff_span_dist.count(s)) for s in args.spans
        },
    }
    summary_b = {
        "instances": 20,
        "window_gain_mean": float(np.mean(window_gains)),
        "window_gain_max": float(np.max(window_gains)),
        "structures": structures,
    }
    summary_c = {
        "instances": args.alloc_instances,
        "budget": args.budget,
        "mean_worst_pd_sum_i_plus": float(np.mean(
            [r["sum_i_plus_worst_pd"] for r in alloc_rows])),
        "mean_worst_pd_maxmin_i_plus": float(np.mean(
            [r["maxmin_i_plus_worst_pd"] for r in alloc_rows])),
        "mean_worst_pd_floor_cover": float(np.mean(
            [r["floor_cover_worst_pd"] for r in alloc_rows])),
        "floor_cover_exactness_fraction": sum(
            1 for r in alloc_rows
            if abs(r["floor_cover_worst_pd"] - r["exact_maxmin_worst_pd"])
            <= 1e-9
        ) / args.alloc_instances,
        "mean_worst_pd_exact": float(np.mean(
            [r["exact_maxmin_worst_pd"] for r in alloc_rows])),
        "mean_worst_chernoff_exact": float(np.mean(
            [r["exact_maxmin_chernoff"] for r in alloc_rows])),
        "mean_worst_pd_of_chernoff_opt": float(np.mean(
            [r["worst_pd_of_chernoff_opt"] for r in alloc_rows])),
        "chernoff_opt_matches_exact_pd": sum(
            1 for r in alloc_rows
            if r["worst_pd_of_chernoff_opt"] >= r["exact_maxmin_worst_pd"] - 1e-9
        ) / args.alloc_instances,
        "exact_wins_over_sum_i_plus": sum(
            1 for r in alloc_rows
            if r["exact_maxmin_worst_pd"] > r["sum_i_plus_worst_pd"] + 1e-9
        ) / args.alloc_instances,
        "exact_wins_over_maxmin_i_plus": sum(
            1 for r in alloc_rows
            if r["exact_maxmin_worst_pd"] > r["maxmin_i_plus_worst_pd"] + 1e-9
        ) / args.alloc_instances,
    }

    passed = (
        summary_a["i_plus_failure_fraction"] > 0.3
        and summary_a["chernoff_failure_fraction"] < 0.25
        and summary_c["exact_wins_over_sum_i_plus"] > 0.9
        and summary_c["chernoff_opt_matches_exact_pd"] > 0.5
        and summary_c["floor_cover_exactness_fraction"] == 1.0
    )
    payload = {
        "gate": "detection-aware-quantization-and-information-gradient",
        "bits": args.bits,
        "alpha": args.alpha,
        "part_a": summary_a,
        "part_b": summary_b,
        "part_c": summary_c,
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Part A: design-metric correctness on the span knob")
    print(f"  span agreement with exact P_D(4) argmax: {summary_a['span_agreement_with_pd4_opt']}")
    print(f"  I+ failure fraction (picks span with P_D(4) < 99% of best): "
          f"{summary_a['i_plus_failure_fraction']:.3f}")
    print(f"  Chernoff failure fraction: {summary_a['chernoff_failure_fraction']:.3f}")
    print("Part B: 1-bit LLR structure")
    print(f"  window gain over single threshold: mean {summary_b['window_gain_mean']:.5f} "
          f"max {summary_b['window_gain_max']:.5f}; structures {structures}")
    print("Part C: information-gradient allocation (worst-target P_D(4))")
    print(f"  sum-I+ waterfilling: {summary_c['mean_worst_pd_sum_i_plus']:.4f}")
    print(f"  maxmin-I+ waterfilling: {summary_c['mean_worst_pd_maxmin_i_plus']:.4f}")
    print(f"  exact max-min P_D: {summary_c['mean_worst_pd_exact']:.4f}")
    print(f"  floor-cover allocation: {summary_c['mean_worst_pd_floor_cover']:.4f} "
          f"(exactness vs exhaustive: {summary_c['floor_cover_exactness_fraction']:.2f})")
    print(f"  exact max-min Chernoff (proxy): {summary_c['mean_worst_chernoff_exact']:.4f}")
    print(f"  worst P_D(4) of the Chernoff-optimal allocation: "
          f"{summary_c['mean_worst_pd_of_chernoff_opt']:.4f} "
          f"(matches exact P_D in {summary_c['chernoff_opt_matches_exact_pd']:.2f} of instances)")
    print(f"  exact wins fraction: {summary_c['exact_wins_over_sum_i_plus']:.2f} "
          f"(sum-I+), {summary_c['exact_wins_over_maxmin_i_plus']:.2f} (maxmin-I+)")
    print(f"passed={passed}")


if __name__ == "__main__":
    main()