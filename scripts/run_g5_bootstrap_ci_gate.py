"""G5-CI gate: paired bootstrap CIs for the G5 series gains.

Loads the per-seed G5 result files and computes paired bootstrap 95%
confidence intervals and win rates for the key comparisons: aligned versus
no-RIS, aligned versus random phase, best placement versus fixed placement,
and best quantized allocation versus no-RIS.
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


def bootstrap_ci(values, seed, replicates=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(values))
    samples = []
    for _ in range(replicates):
        sample = rng.choice(indices, size=len(indices), replace=True)
        samples.append(float(np.mean(values[sample])))
    return {
        "mean": float(np.mean(values)),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "win_rate": float(np.mean(values > 1e-6)),
        "pairs": len(values),
    }


def _grouped_rows(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        result.setdefault(key, []).append(row)
    return result


def _paired(
    left_rows, right_rows, metric="mean_expected_pd"
):
    left_by_seed = {row["seed_offset"]: row for row in left_rows}
    differences = []
    for right in right_rows:
        seed = right["seed_offset"]
        if seed not in left_by_seed:
            continue
        differences.append(
            float(right[metric]) - float(left_by_seed[seed][metric])
        )
    return differences


def run_gate(*, output: Path, results_dir: Path, seed: int) -> None:
    payload = {"gate": "G5-CI-paired-bootstrap", "sections": {}}

    isac = json.loads((results_dir / "ris_isac_gate.json").read_text(encoding="utf-8"))
    isac_groups = _grouped_rows(isac["rows"], ("budget_bits",))
    isac_section = []
    for budget, group in sorted(isac_groups.items()):
        no_ris = [row for row in group if row["scenario"] == "no_ris"]
        aligned = [row for row in group if row["scenario"] == "ris_aligned"]
        for metric in ("mean_expected_pd", "worst_expected_pd"):
            diffs = _paired(no_ris, aligned, metric)
            isac_section.append({
                "budget_bits": budget,
                "metric": metric,
                "comparison": "aligned_vs_no_ris",
                **bootstrap_ci(diffs, seed),
            })
    payload["sections"]["ris_isac"] = isac_section

    physics = json.loads(
        (results_dir / "ris_physics_gate.json").read_text(encoding="utf-8")
    )
    physics_groups = _grouped_rows(
        physics["rows"], ("budget_bits", "elements", "aperture_scale")
    )
    physics_section = []
    for key, group in sorted(physics_groups.items()):
        aligned = [row for row in group if row["scenario"] == "ris_aligned"]
        random_ris = [row for row in group if row["scenario"] == "ris_random"]
        for metric in ("mean_expected_pd", "worst_expected_pd"):
            aligned_vs_no = [
                row[metric] - row["no_ris_mean" if metric == "mean_expected_pd" else "no_ris_worst"]
                for row in aligned
            ]
            aligned_vs_random = _paired(random_ris, aligned, metric)
            physics_section.append({
                "budget_bits": key[0],
                "elements": key[1],
                "aperture_scale": key[2],
                "metric": metric,
                "comparison": "aligned_vs_no_ris",
                **bootstrap_ci(aligned_vs_no, seed),
            })
            physics_section.append({
                "budget_bits": key[0],
                "elements": key[1],
                "aperture_scale": key[2],
                "metric": metric,
                "comparison": "aligned_vs_random",
                **bootstrap_ci(aligned_vs_random, seed),
            })
    payload["sections"]["ris_physics"] = physics_section

    placement = json.loads(
        (results_dir / "ris_placement_gate.json").read_text(encoding="utf-8")
    )
    placement_groups = _grouped_rows(
        placement["rows"], ("total_budget_bits", "coherence_frames")
    )
    placement_section = []
    fixed_position = [55.0, 15.0, 12.0]
    best_position = [0.0, 20.0, 8.0]
    for key, group in sorted(placement_groups.items()):
        fixed = [row for row in group if row["ris_position"] == fixed_position]
        best = [row for row in group if row["ris_position"] == best_position]
        for metric in ("mean_expected_pd", "worst_expected_pd"):
            best_vs_no = [
                row[metric]
                - row["no_ris_mean" if metric == "mean_expected_pd" else "no_ris_worst"]
                for row in best
            ]
            best_vs_fixed = _paired(fixed, best, metric)
            placement_section.append({
                "total_budget_bits": key[0],
                "coherence_frames": key[1],
                "metric": metric,
                "comparison": "best_vs_no_ris",
                **bootstrap_ci(best_vs_no, seed),
            })
            placement_section.append({
                "total_budget_bits": key[0],
                "coherence_frames": key[1],
                "metric": metric,
                "comparison": "best_vs_fixed",
                **bootstrap_ci(best_vs_fixed, seed),
            })
    payload["sections"]["ris_placement"] = placement_section

    joint = json.loads(
        (results_dir / "ris_joint_budget_gate.json").read_text(encoding="utf-8")
    )
    joint_groups = _grouped_rows(
        joint["rows"], ("total_budget_bits", "coherence_frames")
    )
    joint_section = []
    for key, group in sorted(joint_groups.items()):
        quantized = [row for row in group if row["phase_bits"] == 3]
        for metric in ("mean_expected_pd", "worst_expected_pd"):
            diffs = [
                row[metric]
                - row["no_ris_mean" if metric == "mean_expected_pd" else "no_ris_worst"]
                for row in quantized
            ]
            joint_section.append({
                "total_budget_bits": key[0],
                "coherence_frames": key[1],
                "metric": metric,
                "comparison": "bits3_vs_no_ris",
                **bootstrap_ci(diffs, seed),
            })
    payload["sections"]["ris_joint_budget"] = joint_section

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["sections"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/g5_bootstrap_ci_gate.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        results_dir=Path(args.results_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
