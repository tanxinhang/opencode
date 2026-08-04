"""Gate G1-C: conditional set-dependent ranking vs Static ID Top-K.

The current algorithm is a Conditional-Deflection Greedy.  Under a diagonal
covariance with equal costs it must degenerate to Static Individual-Deflection
Top-K; under a correlated covariance it should prefer a lower-SNR but weakly
correlated report over a redundant high-SNR report.
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

from uav_otfs_isac.fusion import (
    conditional_marginal_deflection,
    gaussian_detection_probability,
)


def _greedy(mu0, mu1, cov0, budget):
    delta = np.asarray(mu1) - np.asarray(mu0)
    selected = []
    for _ in range(budget):
        gains = [
            conditional_marginal_deflection(delta, cov0, tuple(selected), i)
            if selected else (
                delta[i] ** 2 / max(cov0[i, i], 1e-12)
            )
            for i in range(len(mu0)) if i not in selected
        ]
        candidates = [i for i in range(len(mu0)) if i not in selected]
        selected.append(candidates[int(np.argmax(gains))])
    return tuple(sorted(selected))


def _static_id_topk(mu0, mu1, cov0, budget):
    delta = np.asarray(mu1) - np.asarray(mu0)
    scores = [
        delta[i] ** 2 / max(cov0[i, i], 1e-12)
        for i in range(len(mu0))
    ]
    return tuple(sorted(np.argsort(scores)[::-1][:budget]))


def run_gate(*, output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Diagonal, equal-cost case: greedy must equal Static ID Top-K.
    mu0 = np.zeros(4)
    mu1 = np.asarray([1.0, 2.0, 3.0, 4.0])
    cov0 = np.eye(4)
    budget = 2
    degeneracy = {
        "greedy": [int(x) for x in _greedy(mu0, mu1, cov0, budget)],
        "static_id_topk": [
            int(x) for x in _static_id_topk(mu0, mu1, cov0, budget)
        ],
        "identical": _greedy(mu0, mu1, cov0, budget)
        == _static_id_topk(mu0, mu1, cov0, budget),
    }

    # Correlated case: UAV1 and UAV2 high SNR but correlated; UAV3 weaker but
    # nearly independent.  After UAV1, conditional gain should prefer UAV3.
    mu0 = np.zeros(3)
    mu1 = np.asarray([3.0, 2.9, 2.0])
    cov0 = np.asarray([
        [1.0, 0.9, 0.05],
        [0.9, 1.0, 0.05],
        [0.05, 0.05, 1.0],
    ])
    cov1 = cov0 * 0.5
    delta = mu1 - mu0
    first = int(np.argmax([
        delta[i] ** 2 / max(cov0[i, i], 1e-12) for i in range(3)
    ]))
    gain_uav2 = conditional_marginal_deflection(delta, cov0, {first}, 1)
    gain_uav3 = conditional_marginal_deflection(delta, cov0, {first}, 2)
    pd_uav2 = gaussian_detection_probability(
        mu0, mu1, cov0, cov1, (first, 1), 0.05
    )
    pd_uav3 = gaussian_detection_probability(
        mu0, mu1, cov0, cov1, (first, 2), 0.05
    )
    static_choice = _static_id_topk(mu0, mu1, cov0, 2)
    greedy_choice = _greedy(mu0, mu1, cov0, 2)
    correlation_penetration = {
        "first_report": first,
        "conditional_gain_uav2": float(gain_uav2),
        "conditional_gain_uav3": float(gain_uav3),
        "pd_uav2_set": float(pd_uav2),
        "pd_uav3_set": float(pd_uav3),
        "static_choice": [int(x) for x in static_choice],
        "greedy_choice": [int(x) for x in greedy_choice],
        "prefers_low_correlation": bool(
            gain_uav3 > gain_uav2 and pd_uav3 > pd_uav2
        ),
    }
    payload = {
        "gate": "G1-C",
        "degeneracy": degeneracy,
        "correlation_penetration": correlation_penetration,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/g1c_conditional_ranking_smoke.json"
    )
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    run_gate(output=Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
