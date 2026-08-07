"""G23 gate: correlated failure and heterogeneous observability consensus.

This gate adds two realistic network properties to the G22 model:

- ``common_failure_probability p_c``: a network-wide outage that disables
  every vote simultaneously;
- heterogeneous per-UAV observability ``obs_i`` derived from the distance
  from UAV ``i`` to the target.

The effective participation is
``obs_i * (1 - p_c) * (1 - (1 - r)^hops)`` and the counting majority rule is
still optimized exactly under the global P_FA constraint.
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
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import (
    degraded_peer_majority_fusion,
    peer_majority_fusion,
)


def geometry_observability(transmitter_positions, target_position):
    distances = np.linalg.norm(
        transmitter_positions - np.asarray(target_position, dtype=float),
        axis=1,
    )
    normalized = distances / max(float(np.max(distances)), 1e-9)
    return np.clip(1.0 - 0.5 * normalized, 0.2, 1.0)


def run_gate(
    *, output: Path, seeds: int, grid: int, qos_target: float,
    aperture_scale: float, direct_blockage: float, g18_result: Path,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    with g18_result.open(encoding="utf-8") as handle:
        g18 = json.load(handle)
    observability_vectors = [
        geometry_observability(transmitter_positions, target)
        for target in targets
    ]
    stages = {
        "peer_clean": dict(
            observability=1.0, per_hop_reliability=1.0, hops=1,
            common_failure_probability=0.0,
        ),
        "common_fail_02": dict(
            observability=1.0, per_hop_reliability=1.0, hops=1,
            common_failure_probability=0.2,
        ),
        "common_fail_04": dict(
            observability=1.0, per_hop_reliability=1.0, hops=1,
            common_failure_probability=0.4,
        ),
        "heterogeneous_obs": dict(
            observability=None, per_hop_reliability=1.0, hops=1,
            common_failure_probability=0.0,
        ),
        "severe_combined": dict(
            observability=None, per_hop_reliability=0.8, hops=2,
            common_failure_probability=0.2,
        ),
    }
    summary = []
    for config_cell in g18["summary"]:
        num_elements = config_cell["num_elements"]
        phase_bits = config_cell["phase_bits"]
        coherence_frames = config_cell["coherence_frames"]
        total_budget = config_cell["total_budget_bits"]
        report_budget = config_cell["report_budget_bits"]
        position = config_cell["final_position"]
        allocation = config_cell["final_allocation"]
        ris = RisConfig(
            position=np.asarray(position, dtype=float),
            num_elements=num_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        phase = multi_beam_phase(ris, targets, allocation)
        gain = shared_phase_gain_matrix(
            ris, transmitter_positions, targets, receiver,
            aperture_scale,
            direct_blockage=direct_blockage, phase=phase,
        )
        accumulators = {
            "centralized_soft": [],
            **{name: [] for name in stages},
        }
        participations = {name: [] for name in stages}
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            central = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            accumulators["centralized_soft"].append(
                float(np.min(central.expected_pd))
            )
            for name, kwargs in stages.items():
                values = []
                for q, model in enumerate(models):
                    stage_kwargs = dict(kwargs)
                    if stage_kwargs["observability"] is None:
                        stage_kwargs["observability"] = observability_vectors[q]
                    result = degraded_peer_majority_fusion(
                        model, false_alarm_rate, **stage_kwargs
                    )
                    values.append(float(result["pd"]))
                    participations[name].append(
                        float(result["mean_participation"])
                    )
                accumulators[name].append(float(np.min(values)))
        cell = {
            "num_elements": num_elements,
            "phase_bits": phase_bits,
            "coherence_frames": coherence_frames,
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "position": position,
            "allocation": allocation,
            "methods": {},
        }
        for name, values in accumulators.items():
            worst = float(np.mean(values))
            cell["methods"][name] = {
                "worst_expected_pd": worst,
                "qos_rate": float(worst >= qos_target - 1e-9),
                "mean_participation": (
                    float(np.mean(participations[name]))
                    if name in participations else None
                ),
            }
        summary.append(cell)

    payload = {
        "gate": "G23-correlated-failure-consensus",
        "seeds": seeds,
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
    parser.add_argument("--output", default="results/correlated_consensus_gate.json")
    parser.add_argument("--g18-result", type=Path,
                        default="results/joint_placement_allocation_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        g18_result=args.g18_result,
        seeds=args.seeds,
        grid=args.grid,
        qos_target=args.qos_target,
        aperture_scale=args.aperture_scale,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
