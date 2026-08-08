# Review Response and Revision Tracker

Status: 2026-08-07.  The external review recommends a structural revision,
not minor polish.  This file maps every P0/P1/P2 point to the current state
and the committed fix.

## P0 (blocking)

| # | Review point | Fix |
|---|--------------|-----|
| 1 | Communication-closed statistical model not written | DONE: full quantization -> bit mapping -> BSC -> dequantization -> erasure mask -> reconstructed moments chain and a concrete correlated-erasure model in Section 3. |
| 2 | KKT proof incomplete; monotonicity coupled to KKT | DONE: stationarity derivation, `lambda>0` from `P_D>0.5`, one-parameter coverage, and zero-extension feasible-region nesting decoupled from KKT. |
| 3 | Submodularity conditions vague | DONE: explicit concavity boundary `c >= z^2/4`, strong-evidence region, modularity conditions on `D(S)`, and no resource-coupled erasure requirement. |
| 4 | DP exactness only under separable targets | DONE: "under additive report costs and target-separable option sets with no cross-target coupling" stated at Theorems 1-2. |
| 5 | B&B exactness proof outsourced | DONE: Cauchy upper-bound formula and pruning proof sketch included in Section 4.4 and Appendix A; numerical epsilon-exactness recorded. |
| 6 | Poisson-binomial independence and M contradiction | DONE: conditional-independence requirement, mixture form under common state, renamed exact first-feasible prefix, voter-count semantics fixed. |
| 7 | "Exact" overclaim | DONE: combinatorial exactness separated from discretized/tolerance-controlled value oracle and one-dimensional fusion search. |

## P1 (experiments and modeling)

| # | Review point | Plan |
|---|--------------|------|
| 1 | RIS model too strong | DONE: wording is now geometry-aware normalized power-gain model with coherent cross-term caveat; RIS demoted to an application instance. |
| 2 | OTFS is a label | DONE: title/abstract rewritten without OTFS; DD-domain sensing is background and explicitly not required by the theory. |
| 3 | Statistical evidence weak | DONE: 500-seed two-sided paired t, Wilcoxon, and Holm correction are all significant at every system-level budget; absolute P_D and effect sizes are reported. |
| 4 | Ablations mixed | DONE: 500-seed factorial ablation toggles each factor one at a time; a hard weak-target scenario shows max-min gains of 2.45-3.45 pp; a quantization study adds a water-filling-inspired greedy plus an exact joint oracle; in a strong-vs-weak multi-target budget competition the exact joint allocation beats greedy by 3.61-4.95 pp at B=18/16/14 over 20 seeds. |
| 5 | Hard B&B instances missing | DONE: `scripts/run_scaled_difficulty_gate.py` adds critical-threshold, similar-weak, K-report, and correlated-redundant layers with node counts, recursion depth, prune rate, and exhaustive match checks. A dual Cauchy bound (min over `mu>=0`) tightens the old `mu=0` bound and cuts the correlated-redundant tree from 1791 to 113 nodes. |

## P2 (submission engineering)

| # | Review point | Plan |
|---|--------------|------|
| 1 | LaTeX auto-numbering broken | DONE: manual section/subsection numbers stripped, single contribution enumerate, `\caption` inside figure/table environments. |
| 2 | BibTeX unused | DONE: `\cite` + `\bibliography{references}`; Tectonic runs BibTeX and produces 22 auto-numbered references. |
| 3 | Times font substitution | PARTIAL: Tectonic substitutes Latin Modern for missing Times shapes; a reliable font setup remains for final submission. |
| 4 | DOCX math raw | PARTIAL: `submission.docx` rebuilt from Markdown; raw LaTeX equations remain a limitation of the Markdown-to-DOCX path. |
| 5 | Author placeholder | PENDING: real authors must be filled before submission. |

## Recommended convergence

Keep: post-communication correlated Gaussian value model, P_D-optimal linear
fusion, heterogeneous-cost exact max-min selection, verifiable B&B scaling.
Demote: peer majority, architecture switching, rate-profile optimization,
RIS placement Lipschitz certificate, multi-subarray details, OTFS as the
methodological foundation (unless the DD-domain signal-statistics link is
added).
