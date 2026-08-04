"""Gate G1-A: evidence-moment calibration from the toy front end.

This is the first G1 smoke: export per-UAV raw matched-filter evidence under
H0/H1, estimate (mu_h, Sigma_h) with shrinkage, and check whether predicted
deflection orders actual Gaussian P_D at a fixed false-alarm rate.
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


def run_gate(
    *,
    output: Path,
    trials: int,
    integration_frames: int,
    seed: int,
    amplitude: float,
    gain_mode: str,
    predicted_mode: str,
) -> None:
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    train_trials = max(1, trials // 2)
    test_trials = max(1, trials - train_trials)
    train_records = collect_evidence(
        config, patterns, templates, np.random.default_rng(seed),
        trials=train_trials,
        integration_frames=integration_frames,
        amplitude=amplitude,
    )
    test_records = collect_evidence(
        config, patterns, templates, np.random.default_rng(seed + 1),
        trials=test_trials,
        integration_frames=integration_frames,
        amplitude=amplitude,
    )
    train_moments = estimate_moments(
        evidence_matrices(train_records, len(patterns))
    )
    test_moments = estimate_moments(
        evidence_matrices(test_records, len(patterns))
    )
    moments = train_moments
    health = moment_health(moments)
    correlation = delta_deflection_vs_delta_pd(
        moments, actual_gain_mode=gain_mode,
        actual_moments=test_moments,
        predicted_score_mode=predicted_mode,
    )
    payload = {
        "gate": "G1-A",
        "train_trials_per_hypothesis": train_trials,
        "test_trials_per_hypothesis": test_trials,
        "integration_frames": integration_frames,
        "amplitude": amplitude,
        "num_uavs": len(patterns),
        "moment_health": health,
        "h0_mean": moments["means"]["h0"].tolist(),
        "h1_mean": moments["means"]["h1"].tolist(),
        "h0_covariance": moments["covariances"]["h0"].tolist(),
        "h1_covariance": moments["covariances"]["h1"].tolist(),
        "deflection_vs_pd": correlation,
        "actual_gain_mode": gain_mode,
        "predicted_score_mode": predicted_mode,
        "test_moments": {
            "h0_mean": test_moments["means"]["h0"].tolist(),
            "h1_mean": test_moments["means"]["h1"].tolist(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "moment_health": health,
        "modes": {
            mode: {
                "spearman": value["spearman"],
                "spearman_bootstrap_ci95": value.get(
                    "spearman_bootstrap_ci95"
                ),
            }
            for mode, value in correlation["modes"].items()
        },
        "deflection_vs_pd": {
            key: correlation[key] for key in
            ("spearman", "spearman_p_value", "spearman_bootstrap_ci95", "pairs")
        },
        "h0_mean": payload["h0_mean"],
        "h1_mean": payload["h1_mean"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/evidence_calibration_smoke.json"
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--integration-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument(
        "--gain-mode",
        choices=("pd_gain", "logit_gain", "relative_deficit_reduction"),
        default="relative_deficit_reduction",
    )
    parser.add_argument(
        "--predicted-mode",
        choices=("deflection", "pd_gain", "logit_pd"),
        default="deflection",
    )
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        trials=args.trials,
        integration_frames=args.integration_frames,
        seed=args.seed,
        amplitude=args.amplitude,
        gain_mode=args.gain_mode,
        predicted_mode=args.predicted_mode,
    )


if __name__ == "__main__":
    main()
