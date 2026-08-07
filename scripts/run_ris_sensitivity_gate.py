"""G5-SEN gate: RIS parameter sensitivity under the joint resource ledger.

The gate sweeps aperture scale, RIS element count, coherence frame count, and
direct-path blockage while keeping the total budget identity

``B_report = B_total - N_ris * phase_bits / coherence_frames``

exact.  Every cell runs the physics-based RIS channel, aligned and random
phase profiles, and the expected-P_D greedy selector.  The output stores
per-seed rows plus summaries with paired bootstrap 95% CIs.
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
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def bootstrap_ci(differences, seed=20260805, replicates=2000):
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = np.arange(differences.size)
    samples = []
    for _ in range(replicates):
        sample = rng.choice(indices, size=differences.size, replace=True)
        samples.append(float(np.mean(differences[sample])))
    return {
        "mean": float(np.mean(differences)),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "win_rate": float(np.mean(differences > 1e-6)),
        "pairs": int(differences.size),
    }


def run_gate(
    *,
    output: Path,
    seeds: int,
    total_budget: int,
    phase_bits: int,
    grid: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris_position = np.array([55.0, 15.0, 12.0])

    sweep_options = {
        "aperture_scale": {
            "values": [1e-3, 3e-3, 1e-2, 3e-2],
            "elements": 256,
            "coherence_frames": 64,
            "direct_blockage": 0.01,
            "aperture_scale": None,
        },
        "elements": {
            "values": [64, 128, 256, 512, 1024],
            "elements": None,
            "coherence_frames": 256,
            "direct_blockage": 0.01,
            "aperture_scale": 1e-2,
        },
        "coherence_frames": {
            "values": [16, 32, 64, 128, 256],
            "elements": 256,
            "coherence_frames": None,
            "direct_blockage": 0.01,
            "aperture_scale": 1e-2,
        },
        "direct_blockage": {
            "values": [0.001, 0.01, 0.1, 1.0],
            "elements": 256,
            "coherence_frames": 64,
            "direct_blockage": None,
            "aperture_scale": 1e-2,
        },
    }

    no_ris_rows = []
    for offset in range(seeds):
        seed = cfg.seed + offset
        models = build_models(cfg, np.random.default_rng(seed))
        selection = expected_pd_greedy_select(
            models, total_budget, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights, grid=grid,
        )
        vector = np.asarray(selection.expected_pd)
        no_ris_rows.append({
            "seed_offset": offset,
            "mean": float(np.mean(vector)),
            "worst": float(np.min(vector)),
            "qos_feasible": bool(np.all(vector >= qos_pd - 1e-9)),
        })

    rows = []
    for parameter, options in sweep_options.items():
        for offset in range(seeds):
            seed = cfg.seed + offset
            rng_phase = np.random.default_rng(seed + 500000)
            for value in options["values"]:
                elements = options["elements"] if parameter != "elements" else value
                coherence = (
                    options["coherence_frames"]
                    if parameter != "coherence_frames"
                    else value
                )
                aperture = (
                    options["aperture_scale"]
                    if parameter != "aperture_scale"
                    else value
                )
                blockage = (
                    options["direct_blockage"]
                    if parameter != "direct_blockage"
                    else value
                )
                ris = RisConfig(
                    position=ris_position,
                    num_elements=int(elements),
                    weak_target_id=cfg.num_targets - 1,
                    phase_bits=phase_bits,
                )
                overhead = ris_control_overhead_bits(
                    ris, coherence_frames=int(coherence)
                )
                report_budget = int(total_budget - overhead)
                if report_budget < 0:
                    rows.append({
                        "parameter": parameter,
                        "parameter_value": float(value),
                        "elements": int(elements),
                        "coherence_frames": int(coherence),
                        "aperture_scale": float(aperture),
                        "direct_blockage": float(blockage),
                        "seed_offset": offset,
                        "scenario": "infeasible",
                        "report_budget_bits": report_budget,
                        "mean_expected_pd": None,
                        "worst_expected_pd": None,
                        "qos_feasible": False,
                    })
                    continue

                aligned_phases = [ris_beam_phase(target, ris) for target in targets]
                aligned_gain = ris_physics_gain_matrix(
                    ris, transmitter_positions, targets, receiver, aperture,
                    direct_blockage=blockage, phase_per_target=aligned_phases,
                )
                random_phases = [
                    rng_phase.uniform(0.0, 2.0 * np.pi, int(elements))
                    for _ in targets
                ]
                random_gain = ris_physics_gain_matrix(
                    ris, transmitter_positions, targets, receiver, aperture,
                    direct_blockage=blockage, phase_per_target=random_phases,
                )
                for scenario, gain in (
                    ("ris_aligned", aligned_gain),
                    ("ris_random", random_gain),
                ):
                    models = build_models(
                        cfg, np.random.default_rng(seed), snr_gain=gain
                    )
                    selection = expected_pd_greedy_select(
                        models, report_budget, false_alarm_rate,
                        qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                    )
                    vector = np.asarray(selection.expected_pd)
                    rows.append({
                        "parameter": parameter,
                        "parameter_value": float(value),
                        "elements": int(elements),
                        "coherence_frames": int(coherence),
                        "aperture_scale": float(aperture),
                        "direct_blockage": float(blockage),
                        "seed_offset": offset,
                        "scenario": scenario,
                        "report_budget_bits": report_budget,
                        "mean_expected_pd": float(np.mean(vector)),
                        "worst_expected_pd": float(np.min(vector)),
                        "qos_feasible": bool(np.all(vector >= qos_pd - 1e-9)),
                    })

    summary = []
    for parameter, options in sweep_options.items():
        for value in options["values"]:
            cell_rows = [
                row for row in rows
                if row["parameter"] == parameter
                and row["parameter_value"] == float(value)
            ]
            aligned = [row for row in cell_rows if row["scenario"] == "ris_aligned"]
            random_ris = [row for row in cell_rows if row["scenario"] == "ris_random"]
            if not aligned:
                continue
            report_budget = aligned[0]["report_budget_bits"]
            aligned_mean = [row["mean_expected_pd"] for row in aligned]
            aligned_worst = [row["worst_expected_pd"] for row in aligned]
            random_mean = [row["mean_expected_pd"] for row in random_ris]
            no_ris_mean = [row["mean"] for row in no_ris_rows]
            no_ris_worst = [row["worst"] for row in no_ris_rows]
            mean_gain = [
                aligned_value - no_ris_value
                for aligned_value, no_ris_value in zip(aligned_mean, no_ris_mean)
            ]
            worst_gain = [
                aligned_value - no_ris_value
                for aligned_value, no_ris_value in zip(aligned_worst, no_ris_worst)
            ]
            summary.append({
                "parameter": parameter,
                "parameter_value": float(value),
                "elements": aligned[0]["elements"],
                "coherence_frames": aligned[0]["coherence_frames"],
                "aperture_scale": aligned[0]["aperture_scale"],
                "direct_blockage": aligned[0]["direct_blockage"],
                "report_budget_bits": report_budget,
                "aligned_mean": float(np.mean(aligned_mean)),
                "aligned_worst": float(np.mean(aligned_worst)),
                "random_mean": float(np.mean(random_mean)),
                "mean_gain_vs_no_ris": bootstrap_ci(mean_gain),
                "worst_gain_vs_no_ris": bootstrap_ci(worst_gain),
                "qos_feasible_rate": float(np.mean([
                    row["qos_feasible"] for row in aligned
                ])),
            })

    payload = {
        "gate": "G5-SEN-ris-sensitivity",
        "total_budget_bits": total_budget,
        "phase_bits": phase_bits,
        "grid": grid,
        "ris_position": ris_position.tolist(),
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_sensitivity_gate.json")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--grid", type=int, default=512)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        total_budget=args.total_budget,
        phase_bits=args.phase_bits,
        grid=args.grid,
    )


if __name__ == "__main__":
    main()
