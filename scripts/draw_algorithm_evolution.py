"""Draw the algorithm-evolution diagram used by paper/submission.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures" / "algorithm_evolution.png"


def _box(ax, x, y, width, height, text, color, fontsize=7.2):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor="#1f2937", facecolor=color, alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize, color="#111827",
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, color="#374151"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.2, color=color,
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    boxes = [
        (0.10, 2.6, 1.95, 1.20,
         "Classical distributed detection\nTenney-Sandell, Chair-Varshney",
         "#dbeafe"),
        (2.35, 2.6, 1.95, 1.20,
         "Exact Poisson-binomial counting\nand consensus (Wang; Olfati-Saber)",
         "#dcfce7"),
        (4.60, 2.6, 1.95, 1.20,
         "KKT P_D-optimal linear family\nset-monotone at P_D > 0.5",
         "#fef9c3"),
        (6.85, 2.6, 1.95, 1.20,
         "Expected-P_D greedy selection\nmonotone submodular regime",
         "#f3e8ff"),
        (9.10, 2.6, 1.95, 1.20,
         "Exact budget / max-min selection\nheterogeneous report costs",
         "#ffedd5"),
        (11.35, 2.6, 1.95, 1.20,
         "Scaled exact-threshold certificate\nbranch-and-bound proof",
         "#cffafe"),
    ]
    for x, y, w, h, text, color in boxes:
        _box(ax, x, y, w, h, text, color)

    bottom_box = (
        2.80, 0.50, 7.90, 1.20,
        "RIS control/report resource identity, placement, distributed "
        "architecture switch (G5-G54)",
        "#e0e7ff",
    )
    _box(ax, *bottom_box, fontsize=7.6)

    for index in range(len(boxes) - 1):
        x1 = boxes[index][0] + boxes[index][2]
        x2 = boxes[index + 1][0]
        _arrow(ax, x1, boxes[index][1] + boxes[index][3] / 2,
               x2, boxes[index + 1][1] + boxes[index + 1][3] / 2)

    for index in (3, 4, 5):
        _arrow(
            ax,
            boxes[index][0] + boxes[index][2] / 2,
            boxes[index][1],
            bottom_box[0] + 1.0,
            bottom_box[1] + bottom_box[3],
            color="#6d28d9",
        )

    ax.text(
        6.75, 4.25,
        "Algorithm evolution: each stage keeps the prior element as a "
        "baseline or a degenerate case",
        ha="center", va="center", fontsize=10, color="#111827",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
