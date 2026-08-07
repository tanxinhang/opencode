"""G5-V gate: Lipschitz-adaptive RIS deployment search with certificate.

The deployment objective is the mean worst-target expected P_D over the seed
set with the physics channel and joint control/report budget.  An initial
coarse evaluation estimates a Lipschitz constant, then
:func:`uav_otfs_isac.deployment_search.lipschitz_adaptive_search` refines
only boxes whose upper bound can still beat the current best.  The result
carries an epsilon-optimality certificate under the estimated Lipschitz
constant and reports the number of objective evaluations.
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
from uav_otfs_isac.deployment_search import (
    estimate_coordinate_lipschitz,
    estimate_lipschitz,
    lipschitz_adaptive_search,
)
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
    epsilon: float, max_evaluations: int,
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
    seed_list = [cfg.seed + offset for offset in range(seeds)]

    def objective(position):
        config = RisConfig(
            position=position,
            num_elements=ris_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        phases = [ris_beam_phase(target, config) for target in targets]
        gain = ris_physics_gain_matrix(
            config, transmitter_positions, targets, receiver,
            aperture_scale, direct_blockage=0.01, phase_per_target=phases,
        )
        worsts = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            worsts.append(float(np.min(selection.expected_pd)))
        return float(np.mean(worsts))

    bounds = np.array([
        [-20.0, 20.0],
        [15.0, 45.0],
        [0.0, 12.0],
    ])
    initial_positions = [
        np.array([bounds[0, 0], bounds[1, 0], bounds[2, 0]]),
        np.array([bounds[0, 1], bounds[1, 0], bounds[2, 0]]),
        np.array([bounds[0, 0], bounds[1, 1], bounds[2, 0]]),
        np.array([bounds[0, 1], bounds[1, 1], bounds[2, 0]]),
        np.array([bounds[0, 0], bounds[1, 0], bounds[2, 1]]),
        np.array([bounds[0, 1], bounds[1, 0], bounds[2, 1]]),
        np.array([bounds[0, 0], bounds[1, 1], bounds[2, 1]]),
        np.array([bounds[0, 1], bounds[1, 1], bounds[2, 1]]),
        np.array([0.0, 20.0, 8.0]),
    ]
    initial_values = [objective(position) for position in initial_positions]
    lipschitz = estimate_lipschitz(
        np.asarray(initial_values), np.asarray(initial_positions)
    )
    used_lipschitz = 2.0 * lipschitz
    coordinate_lipschitz = estimate_coordinate_lipschitz(
        np.asarray(initial_values), np.asarray(initial_positions)
    )
    used_coordinate_lipschitz = 2.0 * coordinate_lipschitz
    result = lipschitz_adaptive_search(
        objective, bounds, lipschitz=used_lipschitz, epsilon=epsilon,
        max_evaluations=max_evaluations,
        coordinate_lipschitz=used_coordinate_lipschitz,
    )
    result["best_point"] = np.asarray(result["best_point"]).tolist()
    result["lipschitz_estimate"] = lipschitz
    result["used_lipschitz"] = used_lipschitz
    result["coordinate_lipschitz_estimate"] = coordinate_lipschitz.tolist()
    result["used_coordinate_lipschitz"] = used_coordinate_lipschitz.tolist()
    result["bounds"] = bounds.tolist()
    result["initial_evaluations"] = len(initial_positions)
    result["seed_count"] = seeds
    result["report_budget_bits"] = report_budget
    payload = {
        "gate": "G5-V-lipschitz-adaptive-deployment",
        "result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/lipschitz_adaptive_deployment_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--max-evaluations", type=int, default=200)
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
        epsilon=args.epsilon,
        max_evaluations=args.max_evaluations,
    )


if __name__ == "__main__":
    main()
