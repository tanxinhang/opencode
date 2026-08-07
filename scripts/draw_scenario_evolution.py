"""Draw the scenario-evolution diagram used by paper/submission.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures" / "scenario_evolution.png"


def _box(ax, x, y, width, height, text, color, fontsize=7.0):
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
    fig, ax = plt.subplots(figsize=(13.5, 4.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        (0.10, 2.5, 1.95, 1.25,
         "Single-UAV OTFS sensing\nDD-domain returns",
         "#dbeafe"),
        (2.35, 2.5, 1.95, 1.25,
         "Multistatic UAV swarm\ncorrelated soft evidence",
         "#dcfce7"),
        (4.60, 2.5, 1.95, 1.25,
         "Quantization + BSC\n+ correlated erasures",
         "#fef9c3"),
        (6.85, 2.5, 1.95, 1.25,
         "Selective soft-information fusion\nfinite report-bit budget",
         "#f3e8ff"),
        (9.10, 2.5, 1.95, 1.25,
         "RIS-assisted direct + cascaded\nsensing channel",
         "#ffedd5"),
        (11.35, 2.5, 1.95, 1.25,
         "One resource identity\nreport vs RIS control bits",
         "#cffafe"),
    ]
    for x, y, w, h, text, color in boxes:
        _box(ax, x, y, w, h, text, color)

    bottom = (
        2.80, 0.45, 7.90, 1.10,
        "Sensing/communication principles: global P_FA calibration, "
        "time-bandwidth ledger, QoS feasibility, exact reception law",
        "#e0e7ff",
    )
    _box(ax, *bottom, fontsize=7.5)

    for index in range(len(boxes) - 1):
        x1 = boxes[index][0] + boxes[index][2]
        x2 = boxes[index + 1][0]
        _arrow(ax, x1, boxes[index][1] + boxes[index][3] / 2,
               x2, boxes[index + 1][1] + boxes[index + 1][3] / 2)

    for index in (1, 2, 3, 4, 5):
        _arrow(
            ax,
            boxes[index][0] + boxes[index][2] / 2,
            boxes[index][1],
            bottom[0] + 1.0,
            bottom[1] + bottom[3],
            color="#6d28d9",
        )

    ax.text(
        6.75, 4.05,
        "Scenario evolution: each layer is kept as a baseline or a "
        "degenerate case of the next",
        ha="center", va="center", fontsize=10, color="#111827",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
