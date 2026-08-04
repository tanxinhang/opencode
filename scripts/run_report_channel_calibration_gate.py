"""Gate G1-B: exact versus Monte Carlo report-channel moments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.report_channel_calibration import (
    exact_received_moments,
    relative_errors,
    simulate_received_moments,
)
from uav_otfs_isac.reporting import quantizer_from_gaussian_range


def run_gate(*, output: Path, trials: int, seed: int) -> None:
    mu = np.asarray([0.5, 1.0, 1.5, 2.0])
    sigma = np.asarray([1.0, 1.0, 1.0, 1.0])
    rows = []
    for bits in (1, 2, 3, 4):
        for bit_flip_probability in (0.01, 0.08):
            for success_probability in (0.9, 0.7):
                for correlation in (0.0, 0.5, 0.9):
                    covariance = np.eye(4) * sigma ** 2
                    covariance[0, 1] = covariance[1, 0] = (
                        correlation * sigma[0] * sigma[1]
                    )
                    edges, values = quantizer_from_gaussian_range(
                        mu, covariance, mu + 1.0, covariance, bits
                    )
                    success = np.full(4, success_probability)
                    exact = exact_received_moments(
                        mu, covariance, edges, values, bits,
                        bit_flip_probability, success,
                    )
                    simulated = simulate_received_moments(
                        mu, covariance, edges, values, bits,
                        bit_flip_probability, success, trials, seed,
                    )
                    errors = relative_errors(exact, simulated)
                    rows.append({
                        "bits": bits,
                        "bit_flip_probability": bit_flip_probability,
                        "success_probability": success_probability,
                        "correlation": correlation,
                        "mean_relative_error": errors["mean_relative_error"],
                        "covariance_relative_error": (
                            errors["covariance_relative_error"]
                        ),
                        "mean_pass": errors["mean_relative_error"] < 0.05,
                        "covariance_pass": (
                            errors["covariance_relative_error"] < 0.10
                        ),
                    })
    summary = {
        "mean_error_max": float(np.max([
            row["mean_relative_error"] for row in rows
        ])),
        "covariance_error_max": float(np.max([
            row["covariance_relative_error"] for row in rows
        ])),
        "all_mean_pass": all(row["mean_pass"] for row in rows),
        "all_covariance_pass": all(row["covariance_pass"] for row in rows),
    }
    payload = {
        "gate": "G1-B",
        "trials": trials,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/report_channel_calibration_smoke.json"
    )
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    run_gate(output=Path(args.output), trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
