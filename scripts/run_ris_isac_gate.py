"""G5 gate: RIS-assisted 6G UAV-OTFS-ISAC scenario and channel.

The sensing channel is upgraded from a single-hop direct view to a direct
plus RIS-cascaded path.  The RIS phase profile steers array gain toward the
weak target, and the resulting evidence SNR gain is injected before
quantization, BSC, and erasure reporting.  This gate compares no-RIS, random
RIS phase, and aligned RIS phase under the expected-P_D greedy selector at
tight budgets, reporting mean/worst expected P_D and QoS feasibility.
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
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, ris_elements: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    ris = RisConfig(
        position=np.array([55.0, 15.0, 12.0]),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
    )
    aligned_phases = [ris_beam_phase(target, ris) for target in targets]
    rows = []
    for budget in budgets:
        for offset in range(seeds):
            seed = cfg.seed + offset
            model_rng = np.random.default_rng(seed)
            no_ris_models = build_models(cfg, model_rng)
            channel_rng = np.random.default_rng(seed + 100000)
            aligned_gain = ris_gain_matrix(
                ris, targets, cfg.num_uavs, aligned_phases
            )
            channel_rng = np.random.default_rng(seed + 200000)
            random_phases = [
                channel_rng.uniform(0.0, 2.0 * np.pi, ris.num_elements)
                for _ in targets
            ]
            random_gain = ris_gain_matrix(
                ris, targets, cfg.num_uavs, random_phases
            )
            model_rng_aligned = np.random.default_rng(seed)
            aligned_models = build_models(
                cfg, model_rng_aligned, snr_gain=aligned_gain
            )
            model_rng_random = np.random.default_rng(seed)
            random_models = build_models(
                cfg, model_rng_random, snr_gain=random_gain
            )
            scenarios = {
                "no_ris": no_ris_models,
                "ris_random": random_models,
                "ris_aligned": aligned_models,
            }
            for name, models in scenarios.items():
                selection = expected_pd_greedy_select(
                    models, budget, false_alarm_rate, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                vector = np.asarray(selection.expected_pd)
                rows.append({
                    "budget_bits": budget,
                    "seed_offset": offset,
                    "scenario": name,
                    "mean_expected_pd": float(np.mean(vector)),
                    "worst_expected_pd": float(np.min(vector)),
                    "qos_deficit": float(np.sum(np.maximum(
                        qos_pd - vector, 0.0
                    ))),
                    "qos_feasible": bool(np.all(vector >= qos_pd - 1e-9)),
                })

    summary = []
    for budget in budgets:
        group = [row for row in rows if row["budget_bits"] == budget]

        def aggregate(scenario):
            subset = [row for row in group if row["scenario"] == scenario]
            return {
                "mean_expected_pd": float(np.mean([
                    row["mean_expected_pd"] for row in subset
                ])),
                "worst_expected_pd": float(np.mean([
                    row["worst_expected_pd"] for row in subset
                ])),
                "qos_deficit": float(np.mean([
                    row["qos_deficit"] for row in subset
                ])),
                "qos_feasible_rate": float(np.mean([
                    row["qos_feasible"] for row in subset
                ])),
            }

        no_ris = aggregate("no_ris")
        aligned = aggregate("ris_aligned")
        random_ris = aggregate("ris_random")
        aligned_mean_gains = [
            row["mean_expected_pd"]
            - no_row["mean_expected_pd"]
            for row, no_row in zip(
                [r for r in group if r["scenario"] == "ris_aligned"],
                [r for r in group if r["scenario"] == "no_ris"],
            )
        ]
        aligned_worst_gains = [
            row["worst_expected_pd"]
            - no_row["worst_expected_pd"]
            for row, no_row in zip(
                [r for r in group if r["scenario"] == "ris_aligned"],
                [r for r in group if r["scenario"] == "no_ris"],
            )
        ]
        summary.append({
            "budget_bits": budget,
            "no_ris": no_ris,
            "ris_random": random_ris,
            "ris_aligned": aligned,
            "mean_gain_aligned_vs_no_ris": float(np.mean(aligned_mean_gains)),
            "worst_gain_aligned_vs_no_ris": float(np.mean(aligned_worst_gains)),
            "mean_gain_aligned_vs_random": (
                aligned["mean_expected_pd"] - random_ris["mean_expected_pd"]
            ),
            "worst_gain_aligned_vs_random": (
                aligned["worst_expected_pd"] - random_ris["worst_expected_pd"]
            ),
        })

    payload = {
        "gate": "G5-ris-isac",
        "ris_elements": ris_elements,
        "ris_position": ris.position.tolist(),
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_isac_gate.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=16)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        grid=args.grid,
        ris_elements=args.ris_elements,
    )


if __name__ == "__main__":
    main()
