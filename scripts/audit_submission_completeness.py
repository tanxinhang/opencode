"""Completeness audit for paper/submission.md."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "paper" / "submission.md"
BIB = ROOT / "paper" / "references.bib"
FIGURES = ROOT / "paper_figures"


def main() -> None:
    if not MD.exists():
        print(f"missing {MD}")
        sys.exit(1)
    text = MD.read_text(encoding="utf-8")
    checks = [
        ("title", text.startswith("# Exact Budgeted Soft-Information Fusion")),
        ("abstract", "## Abstract" in text),
        ("keywords", "**Keywords:**" in text),
        ("three contributions", "makes three contributions" in text),
        ("related work sections",
         all(f"### 2.{n} " in text for n in range(1, 7))),
        ("system model", "## 3. System Model" in text),
        ("resource identity",
         "B_{\\mathrm{total}}" in text or "B_{total}" in text),
        ("method", "## 4. Method" in text),
        ("theorems in method",
         text.count("**Theorem ") >= 4),
        ("proof-sketch appendix",
         "## Appendix A. Proof Sketches" in text),
        ("complexity section", "### 4.5 Complexity" in text),
        ("simulation setup", "## 5. Simulation Setup" in text),
        ("reproducibility", "### 5.1 Reproducibility" in text),
        ("results", "## 6. Results" in text),
        ("limitations", "## 7. Discussion and Limitations" in text),
        ("conclusion", "## 8. Conclusion" in text),
        ("references", "## References" in text),
    ]
    for number in range(1, 23):
        checks.append((f"reference [{number}]", f"\n[{number}] " in text))
    for number in range(1, 7):
        checks.append((f"Table {number}", f"**Table {number}." in text))
    for name in ("algorithm_evolution", "g8_target_scalability", "scenario_evolution"):
        checks.append((f"Figure {name}", (FIGURES / f"{name}.png").exists()))
    if BIB.exists():
        bib = BIB.read_text(encoding="utf-8")
        checks.append(("bib entries", bib.count("@") >= 22))
    else:
        checks.append(("bib file", False))

    failed = False
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        failed = failed or not passed
    if failed:
        sys.exit(1)
    print("submission.md completeness audit passed.")


if __name__ == "__main__":
    main()
