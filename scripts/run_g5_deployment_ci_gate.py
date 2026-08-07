"""G5-DCI gate: paired bootstrap CIs for RIS deployment gains.

G5-CI covered G5/G5-P/G5-S/G5-R.  This gate stores per-seed expected-P_D
values for the fixed, G5-S, G5-T, G5-V, and G5-W deployments and computes
paired bootstrap 95% CIs for deployment gains versus no-RIS and versus the
original fixed position, under the joint control/report budget identity.
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


DEPLOYMENTS = {
    "fixed": [55.0, 15.0, 12.0],
    "g5s": [0.0, 20.0, 8.0],
    "g5t": [0.0, 30.0, 6.0],
    "g5v": [6.25, 39.375, 4.5],
    "g5w": [11.875, 34.21875, 6.5],
}


def bootstrap_ci(values, seed, replicates=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(values))
    samples = []
    for _ in range(replicates):
        sample = rng.choice(indices, size=len(indices), replace=True)
        samples.append(float(np.mean(values[sample])))
    return {
        "mean": float(np.mean(values)),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "win_rate": float(np.mean(values > 1e-6)),
        "pairs": len(values),
    }


def run_gate(
    *, output: Path, seeds: int, total_budget: int, coherence_frames: int,
    grid: int, ris_elements: int, aperture_scale: float, phase_bits: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris = RisConfig(
        position=np.zeros(3),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
    report_budget = int(total_budget - overhead)

    rows = []
    for offset in range(seeds):
        seed = cfg.seed + offset
        no_ris_models = build_models(cfg, np.random.default_rng(seed))
        no_ris_selection = expected_pd_greedy_select(
            no_ris_models, total_budget, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights, grid=grid,
        )
        no_ris_mean = float(np.mean(no_ris_selection.expected_pd))
        no_ris_worst = float(np.min(no_ris_selection.expected_pd))
        row = {
            "seed_offset": offset,
            "no_ris_mean": no_ris_mean,
            "no_ris_worst": no_ris_worst,
        }
        for name, position in DEPLOYMENTS.items():
            config = RisConfig(
                position=np.asarray(position, dtype=float),
                num_elements=ris_elements,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            )
            phases = [ris_beam_phase(target, config) for target in targets]
            gain = ris_physics_gain_matrix(
                config, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=0.01,
                phase_per_target=phases,
            )
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            row[f"{name}_mean"] = float(np.mean(selection.expected_pd))
            row[f"{name}_worst"] = float(np.min(selection.expected_pd))
        rows.append(row)

    sections = {}
    for name in DEPLOYMENTS:
        for metric in ("mean", "worst"):
            vs_no = [
                row[f"{name}_{metric}"]
                - row[f"no_ris_{metric}"]
                for row in rows
            ]
            vs_fixed = [
                row[f"{name}_{metric}"]
                - row[f"fixed_{metric}"]
                for row in rows
            ]
            sections[f"{name}_vs_no_ris_{metric}"] = bootstrap_ci(vs_no, 20260805)
            sections[f"{name}_vs_fixed_{metric}"] = bootstrap_ci(vs_fixed, 20260805)

    payload = {
        "gate": "G5-DCI-deployment-paired-ci",
        "total_budget_bits": total_budget,
        "coherence_frames": coherence_frames,
        "report_budget_bits": report_budget,
        "control_overhead_bits": overhead,
        "sections": sections,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(sections, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g5_deployment_ci_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        total_budget=args.total_budget,
        coherence_frames=args.coherence_frames,
        grid=args.grid,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
    )


if __name__ == "__main__":
    main()
