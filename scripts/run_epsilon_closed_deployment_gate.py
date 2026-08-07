"""G5-W gate: practical epsilon-closed RIS deployment search.

G5-V reported a bounded certificate that did not close within its evaluation
budget.  This gate uses a two-phase procedure: first the finite G5-S
candidate search localizes the deployment, then a small axis-aligned box
around the coarse optimum is searched with Lipschitz branch-and-bound until
the certificate gap is at most ``epsilon`` or the evaluation budget is
exhausted.  The result reports whether the certificate actually closed.
"""

from __future__ import annotations

import argparse
from itertools import product
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
from scripts.run_ris_multigrid_gate import COARSE_POSITIONS


def run_gate(
    *, output: Path, seeds: int, total_budget: int, coherence_frames: int,
    grid: int, ris_elements: int, aperture_scale: float, phase_bits: int,
    epsilon: float, max_evaluations: int, local_radius: float,
    local_evaluations: int, resume: Path | None,
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

    if resume is not None:
        with resume.open(encoding="utf-8") as handle:
            resume_result = json.load(handle)["result"]
        result = dict(resume_result)
        result["best_point"] = np.asarray(result["best_point"]).tolist()
        bounds = np.asarray(result["bounds"])
        result["epsilon_closed"] = bool(
            result["certificate_gap"] <= epsilon + 1e-12
        )
    else:
        coarse_values = [objective(position) for position in COARSE_POSITIONS]
        coarse_index = int(np.argmax(coarse_values))
        coarse_position = COARSE_POSITIONS[coarse_index]
        coarse_value = float(coarse_values[coarse_index])

        # Phase 2 operates in the local box identified by the G5-T/G5-V
        # audits around the blocked weak target.
        lo = np.array([-5.0, 30.0, 0.0])
        hi = np.array([15.0, 45.0, 8.0])
        bounds = np.column_stack((lo, hi))
        initial_positions = [
            np.array([bounds[0, 0], bounds[1, 0], bounds[2, 0]]),
            np.array([bounds[0, 1], bounds[1, 0], bounds[2, 0]]),
            np.array([bounds[0, 0], bounds[1, 1], bounds[2, 0]]),
            np.array([bounds[0, 1], bounds[1, 1], bounds[2, 0]]),
            np.array([bounds[0, 0], bounds[1, 0], bounds[2, 1]]),
            np.array([bounds[0, 1], bounds[1, 0], bounds[2, 1]]),
            np.array([bounds[0, 0], bounds[1, 1], bounds[2, 1]]),
            np.array([bounds[0, 1], bounds[1, 1], bounds[2, 1]]),
            0.5 * (bounds[:, 0] + bounds[:, 1]),
        ]
        initial_values = [objective(position) for position in initial_positions]
        lipschitz = 2.0 * estimate_lipschitz(
            np.asarray(initial_values), np.asarray(initial_positions)
        )
        coordinate_lipschitz = estimate_coordinate_lipschitz(
            np.asarray(initial_values), np.asarray(initial_positions)
        )
        used_coordinate_lipschitz = 2.0 * coordinate_lipschitz
        result = lipschitz_adaptive_search(
            objective, bounds, lipschitz=lipschitz, epsilon=epsilon,
            max_evaluations=max_evaluations,
            coordinate_lipschitz=used_coordinate_lipschitz,
            return_boxes=True,
        )
        boxes = result.pop("boxes")
        refinement_evaluations = 0
        if result["certificate_gap"] > epsilon + 1e-12:
            ordered = sorted(boxes, key=lambda box: box["upper"], reverse=True)
            best = result["best_value"]
            best_point = np.asarray(result["best_point"], dtype=float)
            for box in ordered[:50]:
                center = np.asarray(box["center"], dtype=float)
                half = np.asarray(box["half"], dtype=float)
                for signs in product((-1.0, 1.0), repeat=bounds.shape[0]):
                    candidate = center + np.asarray(signs) * half
                    candidate_value = float(objective(candidate))
                    refinement_evaluations += 1
                    if candidate_value > best:
                        best = candidate_value
                        best_point = candidate
            result["best_value"] = best
            result["best_point"] = best_point.tolist()
            result["certificate_gap"] = result["global_upper"] - best
        result["refinement_evaluations"] = refinement_evaluations
        result["best_point"] = np.asarray(result["best_point"]).tolist()
        result["coarse_position"] = coarse_position.tolist()
        result["coarse_value"] = coarse_value
        result["bounds"] = bounds.tolist()
        result["lipschitz_estimate"] = lipschitz
        result["coordinate_lipschitz_estimate"] = (
            used_coordinate_lipschitz.tolist()
        )
        result["epsilon_closed"] = bool(
            result["certificate_gap"] <= epsilon + 1e-12
        )
        result["seed_count"] = seeds
        result["report_budget_bits"] = report_budget
    if local_evaluations > 0 and result["certificate_gap"] > epsilon + 1e-12:
        local_center = np.asarray(result["best_point"], dtype=float)
        local_half = np.full(bounds.shape[0], local_radius)
        local_lo = np.maximum(local_center - local_half, bounds[:, 0])
        local_hi = np.minimum(local_center + local_half, bounds[:, 1])
        local_bounds = np.column_stack((local_lo, local_hi))
        local_initial = [
            np.array([local_bounds[0, 0], local_bounds[1, 0], local_bounds[2, 0]]),
            np.array([local_bounds[0, 1], local_bounds[1, 0], local_bounds[2, 0]]),
            np.array([local_bounds[0, 0], local_bounds[1, 1], local_bounds[2, 0]]),
            np.array([local_bounds[0, 1], local_bounds[1, 1], local_bounds[2, 0]]),
            np.array([local_bounds[0, 0], local_bounds[1, 0], local_bounds[2, 1]]),
            np.array([local_bounds[0, 1], local_bounds[1, 0], local_bounds[2, 1]]),
            np.array([local_bounds[0, 0], local_bounds[1, 1], local_bounds[2, 1]]),
            np.array([local_bounds[0, 1], local_bounds[1, 1], local_bounds[2, 1]]),
            0.5 * (local_bounds[:, 0] + local_bounds[:, 1]),
        ]
        local_values = [objective(position) for position in local_initial]
        local_lipschitz = 2.0 * estimate_lipschitz(
            np.asarray(local_values), np.asarray(local_initial)
        )
        local_coordinate = 2.0 * estimate_coordinate_lipschitz(
            np.asarray(local_values), np.asarray(local_initial)
        )
        local_result = lipschitz_adaptive_search(
            objective, local_bounds, lipschitz=local_lipschitz,
            epsilon=epsilon, max_evaluations=local_evaluations,
            coordinate_lipschitz=local_coordinate,
        )
        local_closed = bool(
            local_result["certificate_gap"] <= epsilon + 1e-12
        )
        local_result["best_point"] = np.asarray(
            local_result["best_point"]
        ).tolist()
        result["local_closure"] = local_result
        result["local_closure_bounds"] = local_bounds.tolist()
        result["local_epsilon_closed"] = local_closed
        if local_result["best_value"] > result["best_value"]:
            result["best_value"] = local_result["best_value"]
            result["best_point"] = local_result["best_point"]
            result["certificate_gap"] = (
                result["global_upper"] - result["best_value"]
            )
            result["epsilon_closed"] = bool(
                result["certificate_gap"] <= epsilon + 1e-12
            )
    result["seed_count"] = seeds
    result["report_budget_bits"] = report_budget
    payload = {
        "gate": "G5-W-epsilon-closed-deployment",
        "result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/epsilon_closed_deployment_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--max-evaluations", type=int, default=300)
    parser.add_argument("--local-radius", type=float, default=2.0)
    parser.add_argument("--local-evaluations", type=int, default=400)
    parser.add_argument("--resume", type=Path, default=None)
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
        local_radius=args.local_radius,
        local_evaluations=args.local_evaluations,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
