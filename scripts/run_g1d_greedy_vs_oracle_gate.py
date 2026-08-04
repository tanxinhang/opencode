"""Gate G1-D: open-loop approximation, exact greedy, SAA, and Oracle."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.expectation import expected_deflection_exact
from uav_otfs_isac.fusion import conditional_marginal_deflection
from uav_otfs_isac.oracle import exhaustive_oracle
from uav_otfs_isac.selection import greedy_select


def _model(success_probability, correlation, heterogeneous_costs):
    model = symmetric_diversity_model(
        np.asarray([0.6, 0.9, 1.2, 1.5]),
        success_probability=success_probability,
    )
    sigma0 = model.sigma0.copy()
    if correlation > 0.0:
        sigma0[1, 2] = sigma0[2, 1] = correlation
    report_bits = (
        np.asarray([0, 1, 1, 2, 1])
        if heterogeneous_costs else model.report_bits
    )
    return replace(
        model,
        sigma0=sigma0,
        report_bits=report_bits,
    )


def _gains(model, budget_bits):
    pairs = []
    base = expected_deflection_exact(model, {model.owner})
    for uav in range(model.num_uavs):
        if uav == model.owner or model.report_bits[uav] > budget_bits:
            continue
        first_order = (
            model.success_prob[uav]
            * (
                model.delta[uav] ** 2 / max(model.sigma0[uav, uav], 1e-12)
            )
            / model.report_bits[uav]
        )
        exact_gain = (
            expected_deflection_exact(model, {model.owner, uav}) - base
        )
        pairs.append((first_order, exact_gain))
    return pairs


def run_gate(*, output: Path, seed: int) -> None:
    rows = []
    for success_probability in (0.95, 0.6):
        for correlation in (0.0, 0.7):
            for heterogeneous_costs in (False, True):
                model = _model(
                    success_probability, correlation, heterogeneous_costs
                )
                budget_bits = 3
                common = dict(
                    models=[model],
                    budget_bits=budget_bits,
                    qos_min=[0.0],
                    qos_weights=[1.0],
                    performance_weights=[1.0],
                )
                first_order = greedy_select(
                    **common, gain_mode="first_order", qos_first=False
                )
                exact = greedy_select(
                    **common, gain_mode="exact", qos_first=False
                )
                saa = greedy_select(
                    **common, gain_mode="exact", qos_first=False,
                    mode="saa", rng=np.random.default_rng(seed),
                )
                oracle = exhaustive_oracle(**common)
                oracle_set = set(oracle.scheduled[0]) - {model.owner}
                pairs = _gains(model, budget_bits)
                predicted = np.asarray([p[0] for p in pairs])
                actual = np.asarray([p[1] for p in pairs])
                spearman = (
                    float(spearmanr(predicted, actual)[0])
                    if len(predicted) >= 2 else None
                )
                rows.append({
                    "success_probability": success_probability,
                    "correlation": correlation,
                    "heterogeneous_costs": heterogeneous_costs,
                    "oracle": sorted(oracle_set),
                    "first_order_set": sorted(
                        set(first_order.scheduled[0]) - {model.owner}
                    ),
                    "exact_greedy_set": sorted(
                        set(exact.scheduled[0]) - {model.owner}
                    ),
                    "saa_greedy_set": sorted(
                        set(saa.scheduled[0]) - {model.owner}
                    ),
                    "oracle_deflection": float(
                        oracle.expected_deflection[0]
                    ),
                    "first_order_deflection": float(
                        first_order.expected_deflection[0]
                    ),
                    "exact_greedy_deflection": float(
                        exact.expected_deflection[0]
                    ),
                    "saa_greedy_deflection": float(
                        saa.expected_deflection[0]
                    ),
                    "first_order_matches_oracle": bool(
                        sorted(first_order.scheduled[0] - {model.owner})
                        == sorted(oracle_set)
                    ),
                    "exact_greedy_matches_oracle": bool(
                        sorted(exact.scheduled[0] - {model.owner})
                        == sorted(oracle_set)
                    ),
                    "saa_greedy_matches_oracle": bool(
                        sorted(saa.scheduled[0] - {model.owner})
                        == sorted(oracle_set)
                    ),
                    "first_order_vs_exact_spearman": spearman,
                })
    payload = {
        "gate": "G1-D",
        "summary": {
            "first_order_oracle_match_rate": float(np.mean([
                row["first_order_matches_oracle"] for row in rows
            ])),
            "exact_greedy_oracle_match_rate": float(np.mean([
                row["exact_greedy_matches_oracle"] for row in rows
            ])),
            "saa_greedy_oracle_match_rate": float(np.mean([
                row["saa_greedy_matches_oracle"] for row in rows
            ])),
            "mean_first_order_deflection_gap": float(np.mean([
                row["oracle_deflection"] - row["first_order_deflection"]
                for row in rows
            ])),
            "mean_exact_greedy_deflection_gap": float(np.mean([
                row["oracle_deflection"] - row["exact_greedy_deflection"]
                for row in rows
            ])),
            "mean_first_order_vs_exact_spearman": float(np.mean([
                row["first_order_vs_exact_spearman"]
                for row in rows if row["first_order_vs_exact_spearman"] is not None
            ])),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g1d_greedy_vs_oracle_smoke.json"
    )
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    run_gate(output=Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
