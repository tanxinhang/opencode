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
- G1-A saturation-robust Spearman `0.62` was a same-sample smoke; the 10k
  formal run with train/test geometry separation gives `0.588` (CIs
  `[0.23,0.83]` / `[0.21,0.84]`), below the `0.6` gate; predicted-score
  ablation with exact `P_D` gain gives formal 10k `0.996` (CI
  `[0.98,1.00]`) and logit/relative-deficit `0.994`, so the deflection proxy
  is the failure point and a `P_D`-gain selector passes G1-A; grouped
  consistency across amplitudes 0.8/1.0/1.3 confirms deflection
  `0.55/0.33/0.40` (<0.6) and P_D-gain `0.97/0.89/0.77` (>0.6) (README,
  paper outline, Word B.1).
- G1-B mean/covariance errors `4.08%` / `8.51%` (README, paper outline,
  Word B.2).
- G1-D first-order vs exact Spearman `0.90` and Oracle match rate `50%`
  (README, paper outline, Word B.4).
- G2 20-seed fair-budget numbers `0.898/0.898/0.897/0.773/0.935` with
  exact-PD greedy `0.900`, and correlated-model
  `0.870/0.855/83.1%/0.880`, plus a multi-rho sweep
  `0/0.3/0.5/0.7/0.85` with positive paired-diff CIs for `rho>=0.3`, and a
  non-saturated stress gate with gains `+0.172`/`+0.114` at `B=6/9` (README,
  paper outline, Word B.5), and a scaling sweep where Exact-PD gains grow
  from `+0.006` to `+0.067` as `(M,Q)` goes from `(8,3)` to `(16,8)`, plus a
  non-saturated scaling-stress gate with `+0.114` mean gain at `Q=3/5/8` and
  worst-target gains `+0.205` to `+0.272`.
- Resource fairness same-scale table: fixed per-frame energy
  `86.7% -> 100%` with 4x energy; fixed total energy `86.7% -> 50%`;
  time-bandwidth `2575` vs `4111` symbols; fixed-TB path requires grid scaling
  (README, paper outline, Word A.4).
- Documentation file list now includes PAPER_OUTLINE.md and all three Word
  appendices.

## Residual open items (not contradictions)

- Equal bandwidth/frame-budget/communication-rate accounting now has a
  same-scale resource table; full time-bandwidth and latency accounting is
  still incomplete.
- Formal G1-A run at 10 000 trials per hypothesis with bootstrap CI.
- G2 correlated audit at 10-20 seeds with win-rate CI.
- Same angle-DD collision decomposition and strong FWER.
- Bandwidth-consistent SDR front end.
