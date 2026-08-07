"""G45 gate: closed-form resource-information law.

In the proportional-covariance regime,

``P_D = Phi( (sqrt(D_pred) - z_FA) / sqrt(c) )``,

with

``D_pred = d0 * (1 + n_soft) * mean(gain^2)``,

where ``d0`` is the owner-only deflection calibrated from the model,
``n_soft`` is the number of soft reports per target affordable under the
report budget, and ``gain`` is the RIS gain matrix.  The gate compares this
derived law with the exact system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.expected_pd import expected_gaussian_detection_probability
from uav_otfs_isac.fusion import optimal_deflection
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def run_gate(
    *, output: Path, seeds: int, budgets, element_options, grid: int,
    aperture_scale: float, phase_bits: int, coherence_frames: int,
    direct_blockage: float, variance_ratio: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    z = norm.ppf(1.0 - false_alarm_rate)
    qos_pd = np.full(cfg.num_targets, 0.70)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    summary = []
    for num_elements in element_options:
        ris = RisConfig(
            position=np.array([0.0, 30.0, 6.0]),
            num_elements=num_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        overhead = ris_control_overhead_bits(
            ris, coherence_frames=coherence_frames
        )
        phases = [ris_beam_phase(target, ris) for target in targets]
        gain = ris_physics_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase_per_target=phases,
        )
        mean_gain_sq = float(np.mean(gain**2))
        owner_deflections = []
        for seed in seed_list:
            models = build_models(cfg, np.random.default_rng(seed))
            owner_deflections.extend([
                optimal_deflection(
                    model.delta, model.sigma0, {model.owner}
                )
                for model in models
            ])
        d0 = float(np.mean(owner_deflections))
        owner_pd_values = []
        for seed in seed_list:
            models = build_models(cfg, np.random.default_rng(seed))
            owner_pd_values.append(float(np.min([
                expected_gaussian_detection_probability(
                    model, {model.owner}, false_alarm_rate,
                    pd_mode="optimal", grid=grid,
                )
                for model in models
            ])))
        owner_pd = float(np.mean(owner_pd_values))
        c_eff = float((
            (np.sqrt(d0) - z) / norm.ppf(owner_pd)
        ) ** 2)
        for total_budget in budgets:
            report_budget = int(total_budget - overhead)
            if report_budget < 0:
                continue
            n_soft = max(0, report_budget // (5 * cfg.num_targets))
            d_pred = d0 * (1.0 + n_soft) * mean_gain_sq
            predicted_pd = float(norm.cdf(
                (np.sqrt(d_pred) - z) / np.sqrt(c_eff)
            ))
            exact_values = []
            for seed in seed_list:
                models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain
                )
                selection = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                exact_values.append(float(np.min(selection.expected_pd)))
            exact_pd = float(np.mean(exact_values))
            summary.append({
                "ris_elements": num_elements,
                "total_budget_bits": total_budget,
                "report_budget_bits": report_budget,
                "n_soft": n_soft,
                "d0": d0,
                "mean_gain_sq": mean_gain_sq,
                "calibrated_c": c_eff,
                "predicted_d": d_pred,
                "predicted_pd": predicted_pd,
                "exact_pd": exact_pd,
                "absolute_error": abs(predicted_pd - exact_pd),
            })

    payload = {
        "gate": "G45-resource-information-law",
        "seeds": seeds,
        "variance_ratio": variance_ratio,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/resource_information_law_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[12, 16, 28, 40])
    parser.add_argument("--element-options", type=int, nargs="+",
                        default=[64, 128, 256])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    parser.add_argument("--variance-ratio", type=float, default=1.0)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        element_options=args.element_options,
        grid=args.grid,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
        variance_ratio=args.variance_ratio,
    )


if __name__ == "__main__":
    main()
