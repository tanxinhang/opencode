"""G11 gate: fixed-budget RIS aperture scaling with control-overhead ledger.

The proposed performance can be improved either by the algorithm or by the
sensing architecture.  G8 proved the algorithm layer has zero headroom under
the audited equal-cost model.  This gate varies the physical architecture:
RIS element count, phase resolution, and coherence-frame amortization, all
under the exact identity ``B_report = B_total - N * phase_bits /
coherence_frames``.  Three aperture allocations are evaluated per
configuration, so the result shows whether more aperture alone can close the
B=20 QoS gap without consuming extra total budget.
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
    ris_control_overhead_bits,
)
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def equal_allocation(num_elements: int, count: int) -> list[int]:
    quotient, remainder = divmod(num_elements, count)
    allocation = [quotient] * count
    for index in range(remainder):
        allocation[index] += 1
    return allocation


def weak_biased_allocation(
    num_elements: int, weak_fraction: float
) -> list[int]:
    weak = int(round(num_elements * weak_fraction))
    rest = num_elements - weak
    first = rest // 2
    second = rest - first
    return [first, second, weak]


def run_gate(
    *, output: Path, seeds: int, total_budgets, ris_elements_options,
    phase_bits_options, coherence_options, grid: int, qos_target: float,
    aperture_scale: float, direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris_position = np.array([0.0, 30.0, 6.0])
    allocations = {
        "equal": lambda n: equal_allocation(n, cfg.num_targets),
        "weak60": lambda n: weak_biased_allocation(n, 0.6),
        "weak80": lambda n: weak_biased_allocation(n, 0.8),
    }

    rows = []
    for total_budget in total_budgets:
        for offset in range(seeds):
            seed = cfg.seed + offset
            no_ris_models = build_models(cfg, np.random.default_rng(seed))
            selection = expected_pd_greedy_select(
                no_ris_models, total_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            values = np.asarray(selection.expected_pd)
            rows.append({
                "seed_offset": offset,
                "scenario": "no_ris",
                "total_budget_bits": total_budget,
                "ris_elements": 0,
                "phase_bits": 0,
                "coherence_frames": 0,
                "report_budget_bits": total_budget,
                "allocation_name": "none",
                "mean_expected_pd": float(np.mean(values)),
                "worst_expected_pd": float(np.min(values)),
                "qos_feasible": bool(np.all(values >= qos_target - 1e-9)),
            })
        for ris_elements in ris_elements_options:
            for phase_bits in phase_bits_options:
                for coherence_frames in coherence_options:
                    ris = RisConfig(
                        position=ris_position,
                        num_elements=ris_elements,
                        weak_target_id=cfg.num_targets - 1,
                        phase_bits=phase_bits,
                    )
                    overhead = ris_control_overhead_bits(
                        ris, coherence_frames=coherence_frames
                    )
                    report_budget = int(total_budget - overhead)
                    if report_budget < 0:
                        continue
                    for allocation_name, allocation_fn in allocations.items():
                        allocation = allocation_fn(ris_elements)
                        phase = multi_beam_phase(
                            ris, targets, allocation
                        )
                        gain = shared_phase_gain_matrix(
                            ris, transmitter_positions, targets, receiver,
                            aperture_scale,
                            direct_blockage=direct_blockage, phase=phase,
                        )
                        for offset in range(seeds):
                            seed = cfg.seed + offset
                            models = build_models(
                                cfg, np.random.default_rng(seed),
                                snr_gain=gain,
                            )
                            selection = expected_pd_greedy_select(
                                models, report_budget, false_alarm_rate,
                                qos_pd=qos_pd, qos_weights=qos_weights,
                                grid=grid,
                            )
                            values = np.asarray(selection.expected_pd)
                            rows.append({
                                "seed_offset": offset,
                                "scenario": "ris_subarray",
                                "total_budget_bits": total_budget,
                                "ris_elements": ris_elements,
                                "phase_bits": phase_bits,
                                "coherence_frames": coherence_frames,
                                "report_budget_bits": report_budget,
                                "allocation_name": allocation_name,
                                "mean_expected_pd": float(np.mean(values)),
                                "worst_expected_pd": float(np.min(values)),
                                "qos_feasible": bool(np.all(
                                    values >= qos_target - 1e-9
                                )),
                            })

    summary = []
    for ris_elements in ris_elements_options:
        for phase_bits in phase_bits_options:
            for coherence_frames in coherence_options:
                for allocation_name in allocations:
                    for total_budget in total_budgets:
                        group = [
                            row for row in rows
                            if row["ris_elements"] == ris_elements
                            and row["phase_bits"] == phase_bits
                            and row["coherence_frames"] == coherence_frames
                            and row["allocation_name"] == allocation_name
                            and row["total_budget_bits"] == total_budget
                        ]
                        if not group:
                            continue
                        summary.append({
                            "total_budget_bits": total_budget,
                            "ris_elements": ris_elements,
                            "phase_bits": phase_bits,
                            "coherence_frames": coherence_frames,
                            "report_budget_bits": group[0]["report_budget_bits"],
                            "allocation_name": allocation_name,
                            "mean_expected_pd": float(np.mean([
                                row["mean_expected_pd"] for row in group
                            ])),
                            "worst_expected_pd": float(np.mean([
                                row["worst_expected_pd"] for row in group
                            ])),
                            "qos_feasible_rate": float(np.mean([
                                row["qos_feasible"] for row in group
                            ])),
                        })

    payload = {
        "gate": "G11-ris-aperture-scaling",
        "qos_target": qos_target,
        "aperture_scale": aperture_scale,
        "direct_blockage": direct_blockage,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_aperture_scaling_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--total-budgets", type=int, nargs="+", default=[20, 28, 40])
    parser.add_argument("--ris-elements", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--phase-bits", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--coherence-frames", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        total_budgets=args.total_budgets,
        ris_elements_options=args.ris_elements,
        phase_bits_options=args.phase_bits,
        coherence_options=args.coherence_frames,
        grid=args.grid,
        qos_target=args.qos_target,
        aperture_scale=args.aperture_scale,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
