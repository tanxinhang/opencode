"""G1-A grouped consistency: SNR groups for deflection vs P_D-gain proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.evidence_calibration import (
    collect_evidence,
    delta_deflection_vs_delta_pd,
    estimate_moments,
    evidence_matrices,
    moment_health,
)
from uav_otfs_isac.front_end import (
    FrontEndConfig,
    identity_patterns,
    precompute_templates,
)


def run_gate(*, output: Path, trials_per_group: int, seed: int) -> None:
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    amplitudes = (0.8, 1.0, 1.3)
    train_trials = max(1, trials_per_group // 2)
    test_trials = max(1, trials_per_group - train_trials)
    rows = []
    for group_index, amplitude in enumerate(amplitudes):
        train_records = collect_evidence(
            config, patterns, templates,
            np.random.default_rng(seed + group_index * 2),
            trials=train_trials, integration_frames=1, amplitude=amplitude,
        )
        test_records = collect_evidence(
            config, patterns, templates,
            np.random.default_rng(seed + group_index * 2 + 1),
            trials=test_trials, integration_frames=1, amplitude=amplitude,
        )
        train_moments = estimate_moments(
            evidence_matrices(train_records, len(patterns))
        )
        test_moments = estimate_moments(
            evidence_matrices(test_records, len(patterns))
        )
        health = moment_health(train_moments)
        deflection = delta_deflection_vs_delta_pd(
            train_moments, actual_moments=test_moments,
            predicted_score_mode="deflection",
        )
        pd_gain = delta_deflection_vs_delta_pd(
            train_moments, actual_moments=test_moments,
            predicted_score_mode="pd_gain",
        )
        rows.append({
            "amplitude": amplitude,
            "train_trials_per_hypothesis": train_trials,
            "test_trials_per_hypothesis": test_trials,
            "moment_health": health,
            "deflection_spearman": deflection["spearman"],
            "deflection_ci95": deflection["spearman_bootstrap_ci95"],
            "pd_gain_spearman": pd_gain["spearman"],
            "pd_gain_ci95": pd_gain["spearman_bootstrap_ci95"],
        })
    summary = {
        "deflection_all_above_0_6": bool(all(
            row["deflection_spearman"] is not None
            and row["deflection_spearman"] >= 0.6 for row in rows
        )),
        "pd_gain_all_above_0_6": bool(all(
            row["pd_gain_spearman"] is not None
            and row["pd_gain_spearman"] >= 0.6 for row in rows
        )),
        "rows": rows,
    }
    payload = {
        "gate": "G1-A-grouped-consistency",
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g1a_grouped_consistency_smoke.json"
    )
    parser.add_argument("--trials-per-group", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        trials_per_group=args.trials_per_group,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
