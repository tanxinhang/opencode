"""G8-S difficulty gate: critical thresholds, similar weak reports, K layers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.scalable_selection import (
    _minimum_cost_bruteforce,
    minimum_cost_to_threshold,
)


def _model(deltas: np.ndarray, costs: np.ndarray) -> TargetEvidenceModel:
    n = len(deltas)
    if len(costs) != n:
        raise ValueError("one cost per report, owner cost zero")
    model = TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(n),
        mu1=np.asarray(deltas, dtype=float),
        sigma0=np.eye(n),
        sigma1=np.eye(n),
        success_prob=np.ones(n),
        report_bits=np.asarray(costs, dtype=int),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )
    model.validate()
    return model


def _row(
    label: str,
    threshold: float,
    model: TargetEvidenceModel,
    grid: int,
    *,
    exhaustive: bool,
) -> dict:
    stats: dict = {}
    started = time.perf_counter()
    result = minimum_cost_to_threshold(
        model,
        threshold,
        0.05,
        grid=grid,
        max_cost=24,
        max_exhaustive_reports=0,
        stats=stats,
    )
    elapsed = time.perf_counter() - started
    brute = None
    if exhaustive:
        brute = _minimum_cost_bruteforce(
            model, threshold, 0.05, pd_mode="optimal", grid=grid
        )
    nodes = stats.get("nodes", 0)
    pruned = (
        stats.get("prune_upper", 0)
        + stats.get("prune_cost", 0)
        + stats.get("prune_leaf", 0)
    )
    return {
        "label": label,
        "threshold": threshold,
        "num_reports": model.num_uavs - 1,
        "min_cost": None if result is None else int(result[0]),
        "num_selected": (
            None if result is None else len(result[1]) - 1
        ),
        "brute_min_cost": None if brute is None else int(brute[0]),
        "matches_exhaustive": (
            result is not None
            and brute is not None
            and int(result[0]) == int(brute[0])
        ),
        "nodes": nodes,
        "prune_upper": stats.get("prune_upper", 0),
        "prune_cost": stats.get("prune_cost", 0),
        "prune_leaf": stats.get("prune_leaf", 0),
        "prune_rate": pruned / max(nodes, 1),
        "max_depth": stats.get("max_depth", 0),
        "wall_seconds": elapsed,
    }


def run_gate(*, output: Path, grid: int) -> None:
    rows: list[dict] = []

    # Critical threshold sweep on a model whose single-report evidence is
    # moderately informative but whose best subset needs several reports.
    num_reports = 10
    deltas = np.concatenate(([0.8], np.linspace(1.2, 0.6, num_reports)))
    costs = np.concatenate(([0], np.ones(num_reports, dtype=int)))
    base = _model(deltas, costs)
    for threshold in (0.75, 0.80, 0.85, 0.90):
        rows.append(_row(
            f"critical-{threshold:.2f}",
            threshold,
            base,
            grid,
            exhaustive=True,
        ))

    # Similar weak reports: near-equal deltas force the B&B to resolve many
    # near-ties instead of being pruned by one dominant report.
    similar_deltas = np.concatenate(
        ([0.8], np.linspace(1.1, 0.9, num_reports))
    )
    similar = _model(similar_deltas, costs)
    rows.append(_row(
        "similar-weak",
        0.85,
        similar,
        grid,
        exhaustive=True,
    ))

    # K-layer: equal evidence and equal cost, so the optimal number of
    # reports is governed directly by the threshold.
    equal = _model(np.concatenate(([0.8], np.full(num_reports, 1.0))), costs)
    for threshold in (0.75, 0.80, 0.85, 0.90):
        rows.append(_row(
            f"k-layer-{threshold:.2f}",
            threshold,
            equal,
            grid,
            exhaustive=True,
        ))

    payload = {
        "gate": "G8-S-difficulty",
        "grid": grid,
        "all_match_exhaustive": all(r["matches_exhaustive"] for r in rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scaled_difficulty_gate.json")
    parser.add_argument("--grid", type=int, default=96)
    args = parser.parse_args()
    run_gate(output=Path(args.output), grid=args.grid)


if __name__ == "__main__":
    main()
