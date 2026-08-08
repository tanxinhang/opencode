"""Structural audit for paper/main.tex."""

from __future__ import annotations

import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"


def main() -> None:
    if not TEX.exists():
        print(f"missing {TEX}; run scripts/md_to_latex.py first")
        sys.exit(1)
    text = TEX.read_text(encoding="utf-8")
    begins = re.findall(r"\\begin\{([^}]*)\}", text)
    ends = re.findall(r"\\end\{([^}]*)\}", text)
    checks = [
        ("balanced braces", text.count("{") == text.count("}")),
        ("balanced environments",
         sorted(begins) == sorted(ends)),
        ("documentclass",
         r"\documentclass[conference]{IEEEtran}" in text),
        ("document body",
         r"\begin{document}" in text and r"\end{document}" in text),
        ("maketitle", r"\maketitle" in text),
        ("bibtex used",
         r"\bibliographystyle{IEEEtran}" in text
         and r"\bibliography{references}" in text
         and text.count(r"\bibitem") == 0),
        ("seven tables", text.count(r"\begin{table}") == 7),
        ("three figures", text.count(r"\begin{figure}") == 3),
        ("sixteen equations", text.count(r"\begin{equation}") == 16),
        ("theorems present",
         text.count(r"\begin{theorem}") >= 4),
    ]
    failed = False
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        failed = failed or not passed
    if failed:
        sys.exit(1)
    print("main.tex structural audit passed.")


if __name__ == "__main__":
    main()
