# Documentation Consistency Audit

Audit date: 2026-08-04.

## Scope

- README.md
- PAPER_OUTLINE.md
- THIRD_PARTY.md
- UAV_OTFS_ISAC_论证与系统模型_revised_final.docx
- UAV_OTFS_ISAC_论证与系统模型_revised_final_G0C.docx
- UAV_OTFS_ISAC_System_Model_revised.docx
- Relevant code and scripts under `uav_otfs_isac/`, `scripts/`, `tests/`

## What was checked

- Key quantitative claims across README, paper outline, and Word appendices:
  `96.7%`, `0.886/0.885/0.884/0.743/0.933`, `0.869/0.859/77.5%/0.880`,
  `0.62/0.61/0.58`, `4.08%/8.51%`, `0.90/0.161`.
- Status wording for G0-C, G1-A/B/C/D, G2, and the older physical Gate 1.
- File references and documentation-file lists.
- Third-party reuse boundaries.
- Idempotency of the Word appendix generator.

## Issues found and fixed

1. README listed Word appendices as A and B only, while the generated
   documents contain A/B/C.  Fixed to A/B/C.
2. The older physical "Gate 1" (identity-code collision) was not explicitly
   distinguished from the G1-A/B/C/D validation gates.  Added a clarifying
   note.
3. THIRD_PARTY.md did not list the newer independently implemented receiver
   components.  Extended the original-implementation section.
4. README said "unimplemented OTFS path-candidate front end" even though the
   toy Gate G0-C front end exists.  Changed to "full SDR-grade OTFS
   path-candidate front end".
5. README said G2 runs "only after G1 passes" while a G2 smoke already
   exists.  Clarified that the formal sweep follows G1 and the smoke is
   included to expose fairness/metric issues.

## Verified consistent

- G0-C separated-scene recovery `96.7%` (README, paper outline, Word A.3).
- G1-A saturation-robust Spearman `0.62` was a same-sample smoke; after
  train/test geometry separation the held-out smoke is `0.47-0.49` (README,
  paper outline, Word B.1) and the 10k formal run is pending.
- G1-B mean/covariance errors `4.08%` / `8.51%` (README, paper outline,
  Word B.2).
- G1-D first-order vs exact Spearman `0.90` and Oracle match rate `50%`
  (README, paper outline, Word B.4).
- G2 fair-budget numbers `0.886/0.886/0.884/0.743/0.933` and correlated-model
  `0.869/0.859/77.5%/0.880` (README, paper outline, Word B.5).
- Documentation file list now includes PAPER_OUTLINE.md and all three Word
  appendices.

## Residual open items (not contradictions)

- Equal bandwidth/frame-budget/communication-rate accounting.
- Formal G1-A run at 10 000 trials per hypothesis with bootstrap CI.
- G2 correlated audit at 10-20 seeds with win-rate CI.
- Same angle-DD collision decomposition and strong FWER.
- Bandwidth-consistent SDR front end.
