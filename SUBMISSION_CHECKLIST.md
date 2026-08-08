# Submission Checklist

Status as of 2026-08-07.  The manuscript has been structurally revised toward
the recommended convergence: post-communication correlated value model,
P_D-optimal linear fusion, exact heterogeneous-cost max-min selection, and
verifiable B&B scaling.  The remaining items are statistical scale-up,
venue-specific packaging, and final polish.

## Artifacts

- `paper/submission.md` -- canonical manuscript (Markdown).
- `paper/submission.docx` -- Word conversion.
- `paper/main.tex` -- IEEEtran LaTeX source.
- `paper/main.pdf` -- compiled 9-page PDF (Tectonic 0.17.0; about 6.5 pages
  of core body plus appendix/proofs/references).
- `paper/references.bib` -- BibTeX library (22 entries).
- `paper_figures/` -- algorithm evolution, scenario evolution, and
  target-count scalability figures.
- `results/paper_results_table.md` and `.csv` -- 2514-row unified result
  table.

## Verification commands

Run from the repository root:

```powershell
python -m pytest -q
python scripts/verify_paper_numbers.py
python scripts/audit_exact_selection_stats.py
python scripts/run_scaled_difficulty_gate.py --grid 96
python scripts/run_factorial_ablation.py --seeds 500 --budget 20 --grid 64
python scripts/run_hard_maxmin_scenario.py --seeds 20 --budgets 8 10 --grid 64
python scripts/run_quantization_study.py --seeds 10 --budgets 9 12 15 --grid 64
python scripts/audit_submission_completeness.py
python scripts/audit_submission_docx.py
python scripts/audit_submission_latex.py
```

Expected: all green.

## Before sending to a venue

1. The PDF has already been compiled in this workspace.  Recompile with the
   bundled Tectonic engine (run from `paper/`):

   ```powershell
   ..\.tools\tectonic\tectonic.exe --keep-logs main.tex
   ```

   Alternatively use TeX Live/MiKTeX (`pdflatex`, `bibtex`, `pdflatex`
   twice).  The current build has no overfull-box warnings; underfull
   warnings are cosmetic.

2. Open `paper/submission.docx` in Word or WPS and inspect page breaks,
   table widths, math notation, and figure placement.  The current DOCX was
   generated without a rendering environment, so this visual check is
   mandatory before submission.

3. Replace the placeholder author/affiliation block in `paper/main.tex`
   (and add any acknowledgments/funding) with the real author list.

4. Choose the target venue and adjust the template if it is not IEEEtran
   conference style; keep the manuscript content unchanged.

5. Re-run `python scripts/verify_paper_numbers.py` after any number edit to
   ensure the manuscript and result JSONs stay in sync.

## Optional technical follow-ups

- Increase G30-E from 2 seeds to 5-10 seeds (requires re-running G30 with
  the same seed set) to strengthen the exact rate-profile certificate.
- Replace the empirical Lipschitz constant in G5-W with an analytically
  valid upper bound to turn the epsilon-closed deployment certificate into
  a fully proven global bound.
- Convert the 12-seed SOTA baseline comparison into externally reproduced
  numbers from the cited works.
