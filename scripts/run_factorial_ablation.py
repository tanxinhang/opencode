"""Factorial ablation for fusion / selection / RIS / quantization / comm / max-min.

Each factor fixes the other five components and toggles one of:

- fusion: P_D-optimal linear fusion vs deflection-optimal fusion
- selection: exact max-min vs forward greedy
- RIS: optimized geometry-aware power gain vs no RIS
- quantization: variable-rate reports vs fixed 3-bit reports
- communication: noisy/correlated channel vs clean channel
- max-min: worst-target max-min objective vs lexicographic mean objective

Run with ``--seeds 50`` or ``--seeds 100`` for the full statistical audit.
The default is a two-seed smoke test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.exact_quota_selection import exact_budget_select, exact_maxmin_select
from uav_otfs_isac.expected_pd import (
    expected_gaussian_detection_probability,
    expected_pd_greedy_select,
)
from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


FACTORS = (
    "full",
    "fusion_off",
    "selection_off",
    "ris_off",
    "quantization_off",
    "communication_off",
    "maxmin_off",
)


def _fixed_rate_models(
    models: list[TargetEvidenceModel],
    bits: int = 3,
) -> list[TargetEvidenceModel]:
    return [
        replace(
            model,
            report_bits=np.array(
                [0 if i == model.owner else bits for i in range(model.num_uavs)],
                dtype=int,
            ),
        )
        for model in models
    ]


def _clean_channel_models(
    models: list[TargetEvidenceModel],
) -> list[TargetEvidenceModel]:
    return [
        replace(
            model,
            bit_flip_prob=np.zeros(model.num_uavs),
            success_prob=np.ones(model.num_uavs),
            reception_patterns=None,
            pattern_probabilities=None,
            reception_state_probabilities=None,
            conditional_success_probabilities=None,
        )
        for model in models
    ]


def _select(
    models: list[TargetEvidenceModel],
    budget_bits: int,
    false_alarm_rate: float,
    grid: int,
    mode: str,
):
    qos_pd = np.full(len(models), 0.85)
    if mode == "exact_maxmin":
        return exact_maxmin_select(
            models, budget_bits, false_alarm_rate,
            qos_pd=qos_pd, grid=grid, max_exhaustive_reports=10,
        )
    if mode == "lexicographic":
        return exact_budget_select(
            models, budget_bits, false_alarm_rate,
            qos_pd=qos_pd, grid=grid, max_exhaustive_reports=10,
        )
    if mode == "greedy":
        return expected_pd_greedy_select(
            models, budget_bits, false_alarm_rate,
            qos_pd=qos_pd, grid=grid,
        )
    raise ValueError(f"unknown selection mode {mode}")


def _evaluate(
    models: list[TargetEvidenceModel],
    scheduled,
    false_alarm_rate: float,
    pd_mode: str,
    grid: int,
) -> dict:
    values = np.asarray([
        expected_gaussian_detection_probability(
            model, group, false_alarm_rate, pd_mode=pd_mode, grid=grid,
        )
        for model, group in zip(models, scheduled)
    ])
    used = sum(
        int(model.report_bits[i])
        for model, group in zip(models, scheduled)
        for i in group
        if i != model.owner
    )
    return {
        "worst_pd": float(np.min(values)),
        "mean_pd": float(np.mean(values)),
        "qos_feasible": bool(np.min(values) >= 0.85 - 1e-9),
        "used_bits": used,
    }


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def _summary(rows: list[dict]) -> dict:
    by_factor: dict[str, list[dict]] = {}
    for row in rows:
        by_factor.setdefault(row["factor"], []).append(row)
    full_worst = [r["worst_pd"] for r in by_factor["full"]]
    out = {}
    for factor, cell_rows in by_factor.items():
        worst = [r["worst_pd"] for r in cell_rows]
        mean = [r["mean_pd"] for r in cell_rows]
        worst_mean, worst_std = _mean_std(worst)
        mean_mean, mean_std = _mean_std(mean)
        full_mean = float(np.mean(full_worst))
        out[factor] = {
            "n_seeds": len(cell_rows),
            "worst_pd_mean": worst_mean,
            "worst_pd_std": worst_std,
            "worst_pd_median": float(np.median(worst)),
            "mean_pd_mean": mean_mean,
            "mean_pd_std": mean_std,
            "qos_rate": float(
                sum(r["qos_feasible"] for r in cell_rows) / len(cell_rows)
            ),
            "used_bits_mean": float(np.mean([r["used_bits"] for r in cell_rows])),
            "worst_pd_gain_vs_full": full_mean - worst_mean,
        }
    return out


def run_ablation(*, output: Path, seeds: int, budget: int, grid: int) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=256,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=3,
    )
    phases = [ris_beam_phase(target, ris) for target in targets]
    gain = ris_physics_gain_matrix(
        ris, transmitter_positions, targets, receiver,
        aperture_scale=1e-2,
        direct_blockage=0.01,
        phase_per_target=phases,
    )

    rows: list[dict] = []
    started = time.perf_counter()
    for seed_offset in range(seeds):
        seed = cfg.seed + seed_offset
        rng = np.random.default_rng(seed)
        noisy_ris = build_models(cfg, rng, snr_gain=gain)
        noisy_no_ris = build_models(cfg, np.random.default_rng(seed + 1000))
        clean_ris = _clean_channel_models(
            build_models(cfg, np.random.default_rng(seed + 2000), snr_gain=gain)
        )
        fixed_ris = _fixed_rate_models(
            build_models(cfg, np.random.default_rng(seed + 3000), snr_gain=gain)
        )

        full = _select(noisy_ris, budget, false_alarm_rate, grid, "exact_maxmin")
        cells = {
            "full": _evaluate(
                noisy_ris, full.scheduled, false_alarm_rate, "optimal", grid
            ),
            "fusion_off": _evaluate(
                noisy_ris, full.scheduled, false_alarm_rate, "deflection", grid
            ),
            "selection_off": _evaluate(
                noisy_ris,
                _select(noisy_ris, budget, false_alarm_rate, grid, "greedy").scheduled,
                false_alarm_rate, "optimal", grid,
            ),
            "ris_off": _evaluate(
                noisy_no_ris,
                _select(noisy_no_ris, budget, false_alarm_rate, grid, "exact_maxmin").scheduled,
                false_alarm_rate, "optimal", grid,
            ),
            "quantization_off": _evaluate(
                fixed_ris,
                _select(fixed_ris, budget, false_alarm_rate, grid, "exact_maxmin").scheduled,
                false_alarm_rate, "optimal", grid,
            ),
            "communication_off": _evaluate(
                clean_ris,
                _select(clean_ris, budget, false_alarm_rate, grid, "exact_maxmin").scheduled,
                false_alarm_rate, "optimal", grid,
            ),
            "maxmin_off": _evaluate(
                noisy_ris,
                _select(noisy_ris, budget, false_alarm_rate, grid, "lexicographic").scheduled,
                false_alarm_rate, "optimal", grid,
            ),
        }
        for factor, metrics in cells.items():
            rows.append({
                "seed": seed,
                "factor": factor,
                **metrics,
            })
        if (seed_offset + 1) % 10 == 0:
            print(
                f"seeds {seed_offset + 1}/{seeds}, "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )

    payload = {
        "gate": "factorial-ablation-fusion-selection-ris-quant-comm-maxmin",
        "seeds": seeds,
        "budget_bits": budget,
        "grid": grid,
        "rows": rows,
        "summary": _summary(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/factorial_ablation.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--grid", type=int, default=64)
    args = parser.parse_args()
    run_ablation(
        output=Path(args.output),
        seeds=args.seeds,
        budget=args.budget,
        grid=args.grid,
    )


if __name__ == "__main__":
    main()
