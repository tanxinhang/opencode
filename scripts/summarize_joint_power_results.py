"""Consolidated summary of joint power-bit performance results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "paper_figures"


def load(name: str):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def print_table(title: str, header: list[str], rows: list[list]):
    print(f"\n== {title} ==")
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(value) for value in row))


def main() -> None:
    homogeneous = load("joint_power_comparison.json")
    heterogeneous = load("joint_power_comparison_heterogeneous.json")
    scaling = load("joint_power_scaling.json")
    comm_mismatch = load("joint_power_comm_mismatch_gate.json")
    qos = load("qos_weighted_maxmin_gate.json")
    unknown = load("unknown_environment_gate.json")
    robust = load("robust_curriculum_gate.json")
    priority = load("priority_middleware_gate.json")

    header = [
        "Budget",
        "MAPPO",
        "MAPPO-NOMP",
        "MAPPO-Probe-NOMP",
        "MAPPO-Adapter",
        "MAPPO-Bandit",
        "Greedy",
        "WTA",
        "UCB-WTA",
        "UCB-NOMP",
        "NOMP",
        "Exact",
    ]
    print_table(
        "Clean homogeneous Q=2",
        header,
        [[
            row["budget"],
            round(row["mappo_worst_mean"], 4),
            round(row["mappo_nomp_worst_mean"], 4),
            round(row["mappo_probe_nomp_worst_mean"], 4),
            round(row["mappo_adapter_nomp_worst_mean"], 4),
            round(row["mappo_bandit_adapter_nomp_worst_mean"], 4),
            round(row["greedy_worst_mean"], 4),
            round(row["wta_greedy_worst_mean"], 4),
            round(row["ucb_wta_greedy_worst_mean"], 4),
            round(row["ucb_nomp_greedy_worst_mean"], 4),
            round(row["nomp_greedy_worst_mean"], 4),
            round(row["exact_worst_mean"], 4),
        ] for row in homogeneous["summary"]],
    )
    print_table(
        "Clean heterogeneous Q=2",
        header,
        [[
            row["budget"],
            round(row["mappo_worst_mean"], 4),
            round(row["mappo_nomp_worst_mean"], 4),
            round(row["mappo_probe_nomp_worst_mean"], 4),
            round(row["mappo_adapter_nomp_worst_mean"], 4),
            round(row["mappo_bandit_adapter_nomp_worst_mean"], 4),
            round(row["greedy_worst_mean"], 4),
            round(row["wta_greedy_worst_mean"], 4),
            round(row["ucb_wta_greedy_worst_mean"], 4),
            round(row["ucb_nomp_greedy_worst_mean"], 4),
            round(row["nomp_greedy_worst_mean"], 4),
            round(row["exact_worst_mean"], 4),
        ] for row in heterogeneous["summary"]],
    )

    print_table(
        "Heterogeneous scaling (per-target budget 4Q)",
        ["Q", "Budget", "Greedy", "WTA", "UCB-NOMP", "NOMP", "Exact"],
        [[
            row["targets"],
            row["budget"],
            round(row["greedy_worst_mean"], 4),
            round(row["wta_greedy_worst_mean"], 4),
            round(row["ucb_nomp_greedy_worst_mean"], 4),
            round(row["nomp_greedy_worst_mean"], 4),
            round(row["exact_worst_mean"], 4),
        ] for row in scaling["rows"]],
    )

    print_table(
        "Per-link communication mismatch",
        ["Budget", "WTA", "UCB-NOMP", "NOMP", "Robust Exact", "WTA gap", "NOMP gap"],
        [[
            row["budget"],
            round(row["wta_greedy_worst_mean"], 4),
            round(row["ucb_nomp_greedy_worst_mean"], 4),
            round(row["nomp_greedy_worst_mean"], 4),
            round(row["robust_exact_worst_mean"], 4),
            round(row["wta_gap_to_exact"], 4),
            round(row["nomp_gap_to_exact"], 4),
        ] for row in comm_mismatch["summary"]],
    )

    print_table(
        "QoS-weighted max-min",
        ["Budget", "Plain NOMP", "QoS-NOMP", "UCB-NOMP", "Exact", "Improvement", "Gap"],
        [[
            row["budget"],
            round(row["plain_nomp_qos_worst_mean"], 4),
            round(row["qos_nomp_qos_worst_mean"], 4),
            round(row["ucb_nomp_qos_worst_mean"], 4),
            round(row["exact_qos_worst_mean"], 4),
            round(row["qos_improvement"], 4),
            round(row["qos_nomp_gap_to_exact"], 4),
        ] for row in qos["summary"]],
    )

    print_table(
        "Unknown environments (average over budgets)",
        ["Environment", "MAPPO", "Adapter", "Bandit", "UCB-NOMP", "NOMP", "Exact"],
        [[
            env,
            round(np.mean([
                row["mappo_worst_mean"] for row in unknown["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["mappo_adapter_worst_mean"] for row in unknown["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["mappo_bandit_worst_mean"] for row in unknown["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["ucb_nomp_worst_mean"] for row in unknown["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["nomp_greedy_worst_mean"] for row in unknown["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["exact_worst_mean"] for row in unknown["rows"]
                if row["environment"] == env
            ]), 4),
        ] for env in ["in_distribution", "channel_shift", "weak", "strong"]],
    )

    print_table(
        "Robust curriculum (average over budgets)",
        ["Environment", "Robust MAPPO", "Robust Bandit", "NOMP", "Exact"],
        [[
            env,
            round(np.mean([
                row["robust_mappo_worst_mean"] for row in robust["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["robust_bandit_worst_mean"] for row in robust["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["nomp_worst_mean"] for row in robust["rows"]
                if row["environment"] == env
            ]), 4),
            round(np.mean([
                row["exact_worst_mean"] for row in robust["rows"]
                if row["environment"] == env
            ]), 4),
        ] for env in ["in_distribution", "weak", "channel_shift"]],
    )

    print_table(
        "Priority middleware (hard scenarios)",
        ["Case", "Plain NOMP", "Priority-NOMP", "Gain"],
        [[
            f"R={row['reports']} B={row['budget']}",
            round(row["plain_nomp_worst"], 4),
            round(row["priority_nomp_worst"], 4),
            round(row["gain"], 4),
        ] for row in priority["rows"]],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    budgets = [row["budget"] for row in heterogeneous["summary"]]
    ax.plot(budgets, [row["greedy_worst_mean"] for row in heterogeneous["summary"]],
            "s-", color="#bdbdbd", label="Greedy")
    ax.plot(budgets, [row["wta_greedy_worst_mean"] for row in heterogeneous["summary"]],
            "v-", color="#3182bd", label="WTA-Greedy")
    ax.plot(budgets, [row["ucb_nomp_greedy_worst_mean"] for row in heterogeneous["summary"]],
            "d--", color="#a1d99b", label="UCB-NOMP")
    ax.plot(budgets, [row["nomp_greedy_worst_mean"] for row in heterogeneous["summary"]],
            "D-", color="#31a354", label="NOMP-Greedy")
    ax.plot(budgets, [row["exact_worst_mean"] for row in heterogeneous["summary"]],
            "*-", color="#e6550d", label="Exact")
    ax.set_title("Clean heterogeneous Q=2")
    ax.set_xlabel("Budget B")
    ax.set_ylabel("Mean worst P_D")
    ax.grid(alpha=0.3)

    ax = axes[1]
    qs = [row["targets"] for row in scaling["rows"]]
    ax.plot(qs, [row["wta_greedy_worst_mean"] for row in scaling["rows"]],
            "v-", color="#3182bd", label="WTA-Greedy")
    ax.plot(qs, [row["ucb_nomp_greedy_worst_mean"] for row in scaling["rows"]],
            "d--", color="#a1d99b", label="UCB-NOMP")
    ax.plot(qs, [row["nomp_greedy_worst_mean"] for row in scaling["rows"]],
            "D-", color="#31a354", label="NOMP-Greedy")
    ax.plot(qs, [row["exact_worst_mean"] for row in scaling["rows"]],
            "*-", color="#e6550d", label="Exact")
    ax.set_title("Target scaling (budget 4Q)")
    ax.set_xlabel("Number of targets Q")
    ax.grid(alpha=0.3)

    ax = axes[2]
    budgets = [row["budget"] for row in comm_mismatch["summary"]]
    ax.plot(budgets, [row["wta_greedy_worst_mean"] for row in comm_mismatch["summary"]],
            "v-", color="#3182bd", label="WTA-Greedy")
    ax.plot(budgets, [row["ucb_nomp_greedy_worst_mean"] for row in comm_mismatch["summary"]],
            "d--", color="#a1d99b", label="UCB-NOMP")
    ax.plot(budgets, [row["nomp_greedy_worst_mean"] for row in comm_mismatch["summary"]],
            "D-", color="#31a354", label="NOMP-Greedy")
    ax.plot(budgets, [row["robust_exact_worst_mean"] for row in comm_mismatch["summary"]],
            "*-", color="#e6550d", label="Robust Exact")
    ax.set_title("Per-link comm mismatch")
    ax.set_xlabel("Budget B")
    ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / "joint_power_overall_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFigure written to {FIGURES / 'joint_power_overall_comparison.png'}")


if __name__ == "__main__":
    main()
