"""G5-RF gate: global resource fairness ledger for RIS-assisted ISAC.

One table accounts for sensing energy, identity resources, report bits, RIS
control bits, and OTFS time-bandwidth occupation, for no-RIS and RIS
deployments under the same total bit budget:

``B_total = B_report + N * phase_bits / coherence_frames``.

Under the conservative 1-symbol-per-bit ledger, the RIS deployment has the
same total time-bandwidth as no-RIS when the control overhead is deducted
from the report budget, so the fixed-total-TB path is exactly the
fixed-total-bits path for the report/control plane.  The RIS is passive, so
control signaling costs time-bandwidth but no transmit energy.
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


def run_gate(
    *, output: Path, seeds: int, total_budget: int, coherence_frames: int,
    grid: int, ris_elements: int, aperture_scale: float, phase_bits: int,
    deployment_position,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris = RisConfig(
        position=np.asarray(deployment_position, dtype=float),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    control_bits = ris_control_overhead_bits(
        ris, coherence_frames=coherence_frames
    )
    ris_report_budget = int(total_budget - control_bits)
    phases = [ris_beam_phase(target, ris) for target in targets]
    gain = ris_physics_gain_matrix(
        ris, transmitter_positions, targets, receiver,
        aperture_scale, direct_blockage=0.01, phase_per_target=phases,
    )

    otfs_symbols_per_frame = cfg.otfs.doppler_bins * cfg.otfs.delay_bins
    frames = 1
    amplitude = 1.0
    sensing_energy = frames * cfg.num_uavs * amplitude**2
    identity_resources = cfg.num_uavs
    sensing_time_bandwidth = frames * otfs_symbols_per_frame
    identity_time_bandwidth = cfg.num_uavs * otfs_symbols_per_frame

    rows = []
    for offset in range(seeds):
        seed = cfg.seed + offset
        scenarios = {
            "no_ris": {
                "report_budget": total_budget,
                "models": build_models(cfg, np.random.default_rng(seed)),
                "control_bits": 0.0,
            },
            "ris": {
                "report_budget": ris_report_budget,
                "models": build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain
                ),
                "control_bits": control_bits,
            },
        }
        for name, scenario in scenarios.items():
            selection = expected_pd_greedy_select(
                scenario["models"], scenario["report_budget"],
                false_alarm_rate, qos_pd=qos_pd, qos_weights=qos_weights,
                grid=grid,
            )
            report_bits = selection.used_bits
            control_bits_used = scenario["control_bits"]
            total_bits = report_bits + control_bits_used
            report_time_bandwidth = report_bits
            control_time_bandwidth = control_bits_used
            total_time_bandwidth = (
                sensing_time_bandwidth
                + identity_time_bandwidth
                + report_time_bandwidth
                + control_time_bandwidth
            )
            total_occupation = (
                sensing_energy
                + identity_resources
                + report_bits
                + control_bits_used
            )
            rows.append({
                "seed_offset": offset,
                "scenario": name,
                "mean_expected_pd": float(np.mean(selection.expected_pd)),
                "worst_expected_pd": float(np.min(selection.expected_pd)),
                "qos_feasible": bool(np.all(
                    selection.expected_pd >= qos_pd - 1e-9
                )),
                "report_bits": report_bits,
                "control_bits": control_bits_used,
                "total_bits": total_bits,
                "sensing_energy": sensing_energy,
                "identity_resources": identity_resources,
                "report_time_bandwidth": report_time_bandwidth,
                "control_time_bandwidth": control_time_bandwidth,
                "total_time_bandwidth": total_time_bandwidth,
                "total_occupation": total_occupation,
            })

    summary = []
    for name in ("no_ris", "ris"):
        group = [row for row in rows if row["scenario"] == name]
        summary.append({
            "scenario": name,
            "mean_expected_pd": float(np.mean([
                row["mean_expected_pd"] for row in group
            ])),
            "worst_expected_pd": float(np.mean([
                row["worst_expected_pd"] for row in group
            ])),
            "qos_feasible_rate": float(np.mean([
                row["qos_feasible"] for row in group
            ])),
            "report_bits": float(np.mean([
                row["report_bits"] for row in group
            ])),
            "control_bits": float(np.mean([
                row["control_bits"] for row in group
            ])),
            "total_bits": float(np.mean([
                row["total_bits"] for row in group
            ])),
            "sensing_energy": float(np.mean([
                row["sensing_energy"] for row in group
            ])),
            "identity_resources": float(np.mean([
                row["identity_resources"] for row in group
            ])),
            "report_time_bandwidth": float(np.mean([
                row["report_time_bandwidth"] for row in group
            ])),
            "control_time_bandwidth": float(np.mean([
                row["control_time_bandwidth"] for row in group
            ])),
            "total_time_bandwidth": float(np.mean([
                row["total_time_bandwidth"] for row in group
            ])),
            "total_occupation": float(np.mean([
                row["total_occupation"] for row in group
            ])),
        })

    no_ris = summary[0]
    ris_summary = summary[1]
    payload = {
        "gate": "G5-RF-global-resource-fairness",
        "deployment_position": list(deployment_position),
        "total_budget_bits": total_budget,
        "coherence_frames": coherence_frames,
        "ris_elements": ris_elements,
        "phase_bits": phase_bits,
        "otfs_symbols_per_frame": otfs_symbols_per_frame,
        "ledger_note": (
            "1 report/control bit = 1 time-bandwidth symbol (conservative); "
            "RIS is passive, so control signaling costs time-bandwidth but no "
            "transmit energy."
        ),
        "summary": summary,
        "gains": {
            "mean_expected_pd_gain": (
                ris_summary["mean_expected_pd"] - no_ris["mean_expected_pd"]
            ),
            "worst_expected_pd_gain": (
                ris_summary["worst_expected_pd"] - no_ris["worst_expected_pd"]
            ),
            "total_time_bandwidth_difference": (
                ris_summary["total_time_bandwidth"]
                - no_ris["total_time_bandwidth"]
            ),
            "total_occupation_difference": (
                ris_summary["total_occupation"] - no_ris["total_occupation"]
            ),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gains": payload["gains"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/global_resource_fairness_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument(
        "--deployment", type=float, nargs="+",
        default=[0.0, 30.0, 6.0],
    )
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
        deployment_position=tuple(args.deployment),
    )


if __name__ == "__main__":
    main()
