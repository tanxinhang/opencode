"""G29 gate: variable-rate soft/hard reporting.

The report chain is upgraded from a fixed 5-bit soft report to per-UAV
variable quantizer bits.  The gate compares:

- fixed 5-bit soft reports (3 quantizer bits);
- fixed 3-bit soft reports (1 quantizer bit);
- adaptive soft rates: high-rate reports on the best sensing links, low-rate
  reports elsewhere, with the same per-target budget;
- optimized 1-bit hard decisions.

All policies use the same total budget and the same RIS control overhead.
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
from uav_otfs_isac.sota_baselines import optimized_hard_decision_fusion


def adaptive_bits_profile(model, budget_bits, num_targets):
    """Per-target equal-budget rate profile based on sensing quality."""
    per_target_budget = budget_bits // num_targets
    high_cost = 5
    low_cost = 3
    high_count = max(
        0, min(
            model.num_uavs - 1,
            (per_target_budget - low_cost) // (high_cost - low_cost),
        )
    )
    candidates = sorted(
        (
            float(model.delta[i] ** 2 / model.sigma0[i, i]),
            i,
        )
        for i in range(model.num_uavs)
        if i != model.owner
    )
    candidates.reverse()
    bits = np.full(model.num_uavs, 1, dtype=int)
    bits[model.owner] = 3
    for _, uav in candidates[:high_count]:
        bits[uav] = 3
    return bits


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, qos_target: float,
    ris_elements: int, aperture_scale: float, phase_bits: int,
    coherence_frames: int, direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
    phases = [ris_beam_phase(target, ris) for target in targets]
    gain = ris_physics_gain_matrix(
        ris, transmitter_positions, targets, receiver, aperture_scale,
        direct_blockage=direct_blockage, phase_per_target=phases,
    )
    summary = []
    for total_budget in budgets:
        report_budget = int(total_budget - overhead)
        methods = {
            "soft5": [],
            "soft3": [],
            "adaptive_soft": [],
            "hard1": [],
        }
        for seed in seed_list:
            soft5_models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            soft5_selection = expected_pd_greedy_select(
                soft5_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["soft5"].append(float(np.min(soft5_selection.expected_pd)))

            soft3_models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                quantizer_bits_per_uav=[1] * cfg.num_uavs,
            )
            soft3_selection = expected_pd_greedy_select(
                soft3_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["soft3"].append(float(np.min(soft3_selection.expected_pd)))

            base_models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            bits = adaptive_bits_profile(
                base_models[0], report_budget, cfg.num_targets
            )
            adaptive_models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                quantizer_bits_per_uav=bits,
            )
            adaptive_selection = expected_pd_greedy_select(
                adaptive_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["adaptive_soft"].append(
                float(np.min(adaptive_selection.expected_pd))
            )

            hard_values = []
            for model in adaptive_models:
                reports = sorted(set(range(model.num_uavs)) - {model.owner})
                per_target = max(1, report_budget // cfg.num_targets)
                candidates = sorted(
                    (
                        float(model.delta[i] ** 2 / model.sigma0[i, i]),
                        i,
                    )
                    for i in reports
                )
                candidates.reverse()
                hard_schedule = {model.owner}
                for _, uav in candidates[:per_target]:
                    hard_schedule.add(uav)
                hard_values.append(float(optimized_hard_decision_fusion(
                    model, hard_schedule, false_alarm_rate
                )["pd"]))
            methods["hard1"].append(float(np.min(hard_values)))
        cell = {
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "methods": {},
        }
        for name, values in methods.items():
            worst = float(np.mean(values))
            cell["methods"][name] = {
                "worst_expected_pd": worst,
                "qos_rate": float(worst >= qos_target - 1e-9),
            }
        summary.append(cell)

    payload = {
        "gate": "G29-variable-rate-reporting",
        "seeds": seeds,
        "qos_target": qos_target,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/variable_rate_report_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 28, 40])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        grid=args.grid,
        qos_target=args.qos_target,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
