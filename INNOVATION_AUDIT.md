# Innovation Audit

Audit date: 2026-08-05.

## Question

Does the current performance satisfy the innovation requirement, i.e., can the
paper honestly claim novelty in either the scenario or the algorithm?

## Current claims

- Scenario: UAV-OTFS-ISAC with communication-corrupted, correlated soft
  evidence feeding selective fusion and system-level `P_D`.
- Algorithm: Conditional-Deflection Greedy, with Exact-`P_D`-Gain Greedy as
  the strongest candidate after G1-A.
- Validation: G1-A/B/C/D gates and fair G2 sweeps.

## What the evidence supports

- The integrated scenario is not a copy of one existing work; it combines
  OTFS soft evidence, per-path Fisher covariance, quantization/BSC/erasure,
  and correlated selective fusion into one audited chain.
- Correlation penetration is demonstrated at system level: under strong
  correlation, Conditional Greedy beats Static ID Top-K in 83.1% of
  configurations, and multi-`rho` sweeps give positive paired-diff CIs for
  `rho>=0.3` in most cells.
- Exact-`P_D`-gain greedy passes G1-A formally (Spearman 0.996, CI
  [0.98, 1.00]) while deflection does not (0.588).

## Where the innovation is weak

1. The algorithm has no formal selection property.  There is no submodularity,
   approximation ratio, or regret bound.  Exact-`P_D`-gain greedy is greedy on
   the actual objective, not a new algorithmic family.
2. Absolute gains in the saturated default G2 are small (0.01-0.02), and one
   cell (`rho=0.85`, `B=20`) has a paired-diff CI that crosses zero.  The
   non-saturated stress gate now provides the large-gain evidence:
   `+0.172` at `B=6` and `+0.114` at `B=9`, with 100% win rate and
   significant paired-diff CIs.
3. The front end is a toy-resolution model, not a bandwidth-consistent SDR.
   Physical realism can be challenged.
4. Resource accounting is incomplete.  Equal bandwidth/frame/rate accounting
   is not yet a unified table, so the four-frame integration gain is still
   open to the "you spent more resources" critique.
5. G1-A grouped consistency is now covered for SNR/amplitude groups
   (0.8/1.0/1.3): deflection 0.55/0.33/0.40, P_D-gain 0.97/0.89/0.77.
   Doppler/leakage/correlation groups and scatter/calibration artifacts are
   still optional.

## Verdict

- If the target venue accepts "scenario + validation methodology" as a
  contribution, the current evidence is borderline sufficient, but the
  following are required before submission:
  1. Switch the main selector to Exact-`P_D`-Gain Greedy and state the
     deflection failure as a negative result.
  2. Add the unified resource accounting table.
  3. Add Doppler/leakage/correlation grouped consistency and
     scatter/calibration data if the venue asks for them.
  4. Report G2 gains with paired CIs and be explicit that they are small in
     saturated regimes but large and significant in non-saturated regimes.
- If the venue requires algorithmic novelty, the current performance is not
  sufficient.  A new algorithm with a formal property, or a large consistent
  performance gain, is needed.

## Recommended next steps

1. Formalize the Exact-`P_D`-Gain Greedy objective and report its monotonicity
   and complexity; attempt a bounded-regime guarantee.
2. Complete the resource accounting gate (three fairness paths).
3. Run G1-A grouped consistency and produce scatter/calibration artifacts.
4. Update README, PAPER_OUTLINE, and Word appendices to reflect this audit.
