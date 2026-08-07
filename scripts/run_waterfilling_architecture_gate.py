"""G13 gate: KKT water-filling allocation from the derived objective.

The allocation is not enumerated.  For target ``q`` the owner-only
deflection surrogate is ``D_q(a_q) = beta_q (1 + kappa_q a_q^2)^2``, and the
max-min KKT condition is ``4 beta_q kappa_q a_q (1 + kappa_q a_q^2) =
lambda``.  This gate solves that scalar equation for each target, chooses
the unique lambda with ``sum a_q = N``, and validates the resulting
allocation with the exact expected-P_D system at the G12-derived aperture.
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
    deflection_surrogate,
    waterfilling_allocation,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.fusion import optimal_deflection
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_quantized_gain_loss,
)
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def equal_allocation(num_elements: int, count: int) -> list[int]:
    quotient, remainder = divmod(num_elements, count)
    allocation = [quotient] * count
    for index in range(remainder):
        allocation[index] += 1
    return allocation


def run_gate(
    *, output: Path, seeds: int, configurations, grid: int,
    qos_target: float, aperture_scale: float, direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris_position = np.array([0.0, 30.0, 6.0])

    seed_list = [cfg.seed + offset for offset in range(seeds)]
    reference_ris = RisConfig(
        position=ris_position,
        num_elements=256,
        weak_target_id=cfg.num_targets - 1,
    )
    constants = aperture_constants(
        reference_ris, transmitter_positions, targets, receiver,
        aperture_scale, direct_blockage=direct_blockage,
    )
    base_deflections = []
    for seed in seed_list:
        models = build_models(cfg, np.random.default_rng(seed))
        base_deflections.append([
            optimal_deflection(
                model.delta, model.sigma0, {model.owner}
            )
            for model in models
        ])
    base = np.mean(base_deflections, axis=0)

    summary = []
    for config in configurations:
        num_elements, phase_bits, coherence_frames, total_budget = config
        effective_constants = constants * ris_quantized_gain_loss(phase_bits)
        allocation = waterfilling_allocation(
            num_elements, effective_constants, base
        )
        equal = equal_allocation(num_elements, cfg.num_targets)
        surrogate_water = float(np.min(deflection_surrogate(
            allocation, phase_bits, constants, base
        )))
        surrogate_equal = float(np.min(deflection_surrogate(
            equal, phase_bits, constants, base
        )))
        cell = {
            "total_budget_bits": total_budget,
            "num_elements": num_elements,
            "phase_bits": phase_bits,
            "coherence_frames": coherence_frames,
            "base_deflections": base.tolist(),
            "effective_constants": effective_constants.tolist(),
            "waterfilling_allocation": list(allocation),
            "equal_allocation": equal,
            "surrogate_water_min": surrogate_water,
            "surrogate_equal_min": surrogate_equal,
            "exact": {},
        }
        for name, candidate_allocation in (
            ("equal", equal),
            ("waterfilling", allocation),
        ):
            overhead = num_elements * phase_bits / coherence_frames
            report_budget = int(total_budget - overhead)
            ris = RisConfig(
                position=ris_position,
                num_elements=num_elements,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            )
            phase = multi_beam_phase(ris, targets, candidate_allocation)
            gain = shared_phase_gain_matrix(
                ris, transmitter_positions, targets, receiver,
                aperture_scale,
                direct_blockage=direct_blockage, phase=phase,
            )
            means = []
            worsts = []
            qos = []
            for seed in seed_list:
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
            cell["exact"][name] = {
                "report_budget_bits": report_budget,
                "mean_expected_pd": float(np.mean(means)),
                "worst_expected_pd": float(np.mean(worsts)),
                "qos_feasible_rate": float(np.mean(qos)),
            }
        summary.append(cell)

    payload = {
        "gate": "G13-waterfilling-architecture",
        "qos_target": qos_target,
        "aperture_scale": aperture_scale,
        "direct_blockage": direct_blockage,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/waterfilling_architecture_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument(
        "--configurations",
        nargs="+",
        type=lambda value: tuple(int(part) for part in value.split(",")),
        default=[
            (1024, 1, 64, 20),
            (704, 3, 128, 20),
            (1344, 3, 256, 20),
            (960, 3, 128, 28),
            (2048, 3, 256, 40),
        ],
    )
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        configurations=args.configurations,
        grid=args.grid,
        qos_target=args.qos_target,
        aperture_scale=args.aperture_scale,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
