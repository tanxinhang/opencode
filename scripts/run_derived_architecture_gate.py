"""G12 gate: model-driven RIS architecture optimization.

Instead of exhaustively sweeping the four design variables, the gate derives
the weak-target deflection surrogate

``J(N) = beta (1 + kappa N^2)^2 (R - L N)``,

where ``kappa`` follows from geometry and phase quantization, ``R = B_total``
and ``L = phase_bits / coherence_frames``.  The first-order condition is a
quadratic, so the candidate aperture is obtained in closed form.  The gate
then evaluates the exact expected-P_D system at that aperture and its two
neighbouring 64-element steps, confirming the derived point without a
high-dimensional grid.
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

from uav_otfs_isac.architecture_objective import (
    aperture_constants,
    derived_surrogate_objective,
    optimal_aperture_formula,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import RisConfig, ris_quantized_gain_loss
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def equal_allocation(num_elements: int, count: int) -> list[int]:
    quotient, remainder = divmod(num_elements, count)
    allocation = [quotient] * count
    for index in range(remainder):
        allocation[index] += 1
    return allocation


def run_gate(
    *, output: Path, seeds: int, total_budgets, phase_bits_options,
    coherence_options, grid: int, qos_target: float, aperture_scale: float,
    direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris_position = np.array([0.0, 30.0, 6.0])
    reference_ris = RisConfig(
        position=ris_position,
        num_elements=256,
        weak_target_id=cfg.num_targets - 1,
    )
    constants = aperture_constants(
        reference_ris, transmitter_positions, targets, receiver,
        aperture_scale, direct_blockage=direct_blockage,
    )
    weak_constant = float(constants[cfg.num_targets - 1])

    rows = []
    summary = []
    for total_budget in total_budgets:
        for phase_bits in phase_bits_options:
            for coherence_frames in coherence_options:
                quantization = ris_quantized_gain_loss(phase_bits)
                kappa = weak_constant * quantization / 9.0
                derived_aperture = optimal_aperture_formula(
                    total_budget, phase_bits, coherence_frames, kappa
                )
                if derived_aperture is None:
                    continue
                rounded = int(np.clip(
                    round(derived_aperture / 64.0) * 64, 64, 2048
                ))
                surrogate = derived_surrogate_objective(
                    derived_aperture, total_budget, phase_bits,
                    coherence_frames, kappa,
                )
                candidates = sorted(set((
                    max(64, rounded - 64),
                    rounded,
                    min(2048, rounded + 64),
                )))
                cell = {
                    "total_budget_bits": total_budget,
                    "phase_bits": phase_bits,
                    "coherence_frames": coherence_frames,
                    "weak_kappa": kappa,
                    "derived_aperture": derived_aperture,
                    "rounded_aperture": rounded,
                    "surrogate_value": surrogate,
                    "candidates": [],
                }
                for num_elements in candidates:
                    overhead = (
                        num_elements * phase_bits / coherence_frames
                    )
                    report_budget = int(total_budget - overhead)
                    if report_budget < 0:
                        continue
                    ris = RisConfig(
                        position=ris_position,
                        num_elements=num_elements,
                        weak_target_id=cfg.num_targets - 1,
                        phase_bits=phase_bits,
                    )
                    allocation = equal_allocation(
                        num_elements, cfg.num_targets
                    )
                    phase = multi_beam_phase(ris, targets, allocation)
                    gain = shared_phase_gain_matrix(
                        ris, transmitter_positions, targets, receiver,
                        aperture_scale,
                        direct_blockage=direct_blockage, phase=phase,
                    )
                    worsts = []
                    means = []
                    qos = []
                    for offset in range(seeds):
                        seed = cfg.seed + offset
                        models = build_models(
                            cfg, np.random.default_rng(seed), snr_gain=gain
                        )
                        selection = expected_pd_greedy_select(
                            models, report_budget, false_alarm_rate,
                            qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                        )
                        values = np.asarray(selection.expected_pd)
                        means.append(float(np.mean(values)))
                        worsts.append(float(np.min(values)))
                        qos.append(bool(np.all(values >= qos_target - 1e-9)))
                    cell["candidates"].append({
                        "num_elements": num_elements,
                        "report_budget_bits": report_budget,
                        "mean_expected_pd": float(np.mean(means)),
                        "worst_expected_pd": float(np.mean(worsts)),
                        "qos_feasible_rate": float(np.mean(qos)),
                    })
                summary.append(cell)

    payload = {
        "gate": "G12-derived-architecture-objective",
        "qos_target": qos_target,
        "aperture_scale": aperture_scale,
        "direct_blockage": direct_blockage,
        "weak_aperture_constant": weak_constant,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "weak_aperture_constant": weak_constant,
        "summary": summary,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/derived_architecture_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--total-budgets", type=int, nargs="+", default=[20, 28, 40])
    parser.add_argument("--phase-bits", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--coherence-frames", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        total_budgets=args.total_budgets,
        phase_bits_options=args.phase_bits,
        coherence_options=args.coherence_frames,
        grid=args.grid,
        qos_target=args.qos_target,
        aperture_scale=args.aperture_scale,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
