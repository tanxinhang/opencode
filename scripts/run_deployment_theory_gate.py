"""G5-U gate: Lipschitz grid-search bound for RIS deployment.

For an ``L``-Lipschitz deployment objective, a grid with spacing ``h`` has
suboptimality at most ``L h sqrt(d) / 2``.  This gate estimates ``L``
empirically from the evaluated RIS deployments, computes the bound at the
fine and second-fine spacings, and checks that the actual fine-to-second-fine
improvement stays within the bound.  The objective is the mean worst-target
expected P_D over the seed set, with the physics channel and joint
control/report budget.
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
    estimate_lipschitz,
    grid_search_suboptimality_bound,
)
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from scripts.run_ris_multigrid_gate import COARSE_POSITIONS, fine_grid


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

    coarse_values = [objective(position) for position in COARSE_POSITIONS]
    coarse_index = int(np.argmax(coarse_values))
    coarse_position = COARSE_POSITIONS[coarse_index]
    fine_positions = fine_grid(coarse_position, span=10.0, z_span=2.0)
    fine_values = [objective(position) for position in fine_positions]
    fine_index = int(np.argmax(fine_values))
    fine_position = fine_positions[fine_index]
    second_positions = fine_grid(fine_position, span=5.0, z_span=1.0)
    second_values = [objective(position) for position in second_positions]
    second_index = int(np.argmax(second_values))
    second_position = second_positions[second_index]

    all_positions = np.vstack((
        np.asarray(COARSE_POSITIONS),
        np.asarray(fine_positions),
        np.asarray(second_positions),
    ))
    all_values = np.concatenate((
        np.asarray(coarse_values),
        np.asarray(fine_values),
        np.asarray(second_values),
    ))
    lipschitz = estimate_lipschitz(all_values, all_positions)
    fine_spacing = 10.0
    second_spacing = 5.0
    fine_bound = grid_search_suboptimality_bound(
        lipschitz, fine_spacing, 3
    )
    second_bound = grid_search_suboptimality_bound(
        lipschitz, second_spacing, 3
    )
    coarse_to_fine = float(np.max(fine_values) - np.max(coarse_values))
    fine_to_second = float(np.max(second_values) - np.max(fine_values))

    payload = {
        "gate": "G5-U-deployment-theory",
        "seeds": seeds,
        "lipschitz_estimate": lipschitz,
        "fine_spacing": fine_spacing,
        "second_spacing": second_spacing,
        "fine_suboptimality_bound": fine_bound,
        "second_suboptimality_bound": second_bound,
        "coarse_position": coarse_position.tolist(),
        "coarse_value": float(np.max(coarse_values)),
        "fine_position": fine_position.tolist(),
        "fine_value": float(np.max(fine_values)),
        "second_position": second_position.tolist(),
        "second_value": float(np.max(second_values)),
        "coarse_to_fine_improvement": coarse_to_fine,
        "fine_to_second_improvement": fine_to_second,
        "fine_bound_holds": bool(fine_to_second <= fine_bound + 1e-9),
        "second_bound_holds": bool(fine_to_second <= second_bound + 1e-9),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/deployment_theory_gate.json")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--grid", type=int, default=256)
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
