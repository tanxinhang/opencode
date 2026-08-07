"""Two-sided and multiplicity-corrected statistics for exact selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "exact_maxmin_gate.json"
OUT = ROOT / "results" / "exact_selection_stats.json"


def paired_stats(gains: list[float]) -> dict:
    g = np.asarray(gains, dtype=float)
    n = len(g)
    if np.allclose(g, 0.0):
        return {
            "n": n,
            "mean_pp": 0.0,
            "median_pp": 0.0,
            "std_pp": 0.0,
            "p_two_sided_t": 1.0,
            "wilcoxon_stat": 0.0,
            "p_wilcoxon": 1.0,
        }
    t, p_two = stats.ttest_rel(g, np.zeros_like(g))
    w, p_w = stats.wilcoxon(g, zero_method="wilcox")
    return {
        "n": n,
        "mean_pp": float(np.mean(g) * 100.0),
        "median_pp": float(np.median(g) * 100.0),
        "std_pp": float(np.std(g, ddof=1) * 100.0),
        "p_two_sided_t": float(p_two),
        "wilcoxon_stat": float(w),
        "p_wilcoxon": float(p_w),
    }


def holm(p_vals: list[float]) -> list[float]:
    p = np.asarray(p_vals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return [float(v) for v in adjusted]


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    summary: dict[str, list[dict]] = {}
    for section, budget_key in (
        ("variable_rate_system", "report_budget_bits"),
        ("controlled", "budget_bits"),
    ):
        per_budget: dict[int, list[float]] = {}
        for row in data[section]["rows"]:
            per_budget.setdefault(row[budget_key], []).append(row["gain_worst"])
        cells = []
        for budget in sorted(per_budget):
            cell = paired_stats(per_budget[budget])
            cell["budget_bits"] = budget
            cells.append(cell)
        p_values = [c["p_two_sided_t"] for c in cells]
        for cell, p_adj in zip(cells, holm(p_values)):
            cell["holm_p_two_sided_t"] = p_adj
        summary[section] = cells

    out = {
        "seeds": data["seeds"],
        "grid": data["grid"],
        "generator": "scripts/audit_exact_selection_stats.py",
        "sections": summary,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
