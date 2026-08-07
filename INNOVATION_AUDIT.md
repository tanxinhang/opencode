# Innovation Audit

Audit date: 2026-08-05 (updated 2026-08-06).

## Question

Does the current performance satisfy the innovation requirement, i.e., can the
paper honestly claim novelty in either the scenario or the algorithm?

## Current claims

- Scenario: RIS-assisted 6G UAV-OTFS-ISAC with a direct-plus-cascaded
  sensing channel and communication-corrupted, correlated soft evidence
  feeding selective fusion and system-level `P_D`.
- Algorithm: Conditional-Deflection Greedy, with Exact-`P_D`-Gain Greedy as
  the strongest candidate after G1-A, plus an exact heterogeneous-cost
  budget-selection certificate (G8-K) for the selection layer.
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
- The `P_D`-gain advantage scales with system size: at `(M,Q)=(16,8)` it
  reaches `+0.067` mean `P_D` over Static ID Top-K with `+0.127` worst-target
  improvement, so the mechanism becomes more valuable as the system grows.
- In a non-saturated scaling-stress model the advantage is visible and
  persistent: `+0.114` mean `P_D` at `Q=3/5/8`, with worst-target gains
  growing from `+0.205` to `+0.272`.
- Expected-`P_D`-gain greedy (Gate G4) adds +1.14pp mean expected `P_D`
  (bootstrap CI [0.47, 1.74], 85% win) and +7.56pp worst-target at `B=20`
  under correlated erasures; a two-candidate hybrid is never worse and gives
  +1.44pp mean / +6.50pp worst-target.
- Because variable-rate reporting (G29) makes report costs heterogeneous,
  the equal-cost exact certificate no longer covers the communication-budget
  feasible set.  G8-K closes that gap with an exact multiple-choice knapsack
  DP and matches an exhaustive global oracle in 100% of 100 controlled cells
  (20 seeds x 5 budgets); on the variable-rate demo scenario it is never
  worse than forward greedy on the lexicographic score in all 100 cells and
  recovers spare-bit headroom, with mean worst-target gains +1.27pp at
  `B=5` (p=0.015) and +2.57pp at `B=7` (p=0.009), and a negative -1.00pp at
  `B=9` that documents the lexicographic objective's worst-target trade-off.
- G8-M supplies the exact max-min version of the same selector, which
  matches G30's worst-target objective: threshold feasibility is solved by a
  multiple-choice knapsack DP and the optimal threshold by binary search.
  It matches an exhaustive max-min oracle in 100% of 100 controlled cells and
  is never worse than forward greedy in the tight-budget variable-rate demo;
  controlled mean worst-target gains are +5.37pp at `B=5` (p<1e-6), +8.24pp
  at `B=7` (p<1e-6), +0.39pp at `B=9` (p=0.083), and +3.33pp at `B=11`
  (p=3.9e-4), and system gains are significant at B=5/7/9/11 (p<0.05) with
  95% bootstrap CIs excluding zero.
- G8-S scales the max-min certificate to larger report sets: per-target
  minimum cost to a threshold is solved by branch-and-bound with a closed-form
  Cauchy upper bound on the `P_D`-optimal linear-score shift, and global
  feasibility is exactly the sum of per-target minima.  The worst case
  remains exponential, but the pruning is exact and is verified against
  exhaustive enumeration on the 20-seed set with zero absolute error; small
  models delegate to exact enumeration, and a cost-bounded minimality proof
  before the search cuts a 16-report threshold-0.9 case from about 60s to
  about 1.5s.
- G43-B evaluates the exact Poisson-binomial `M_min` over every voter prefix
  and audits monotonicity; the audited 8/12/16-UAV sequences are
  non-monotone (exact minimum voters 14/17/16/19 at M=6/8/12/16), so the
  previous binary-search shortcut is formally rejected for these cells.
- G30-E re-runs the G30 greedy certificate with the exact max-min selector
  on the same 2-seed/grid-256 audit: at B=28 the G30 profile remains an exact
  local optimum, while at B=40 the greedy certificate is false under the
  exact objective and exact ascent finds +0.04pp, so the paper reports the
  exact-objective certificate rather than the greedy one.
- RIS-assisted sensing (Gate G5) turns the blocked weak target into a
  controllable NLoS illumination: at `B=20`, aligned RIS plus expected-`P_D`
  greedy raises mean expected `P_D` by +12.3pp and worst-target by +17.8pp
  over no RIS, and the QoS feasibility rate rises from 0% to 95%.
- Finite-resolution RIS phase (G5-Q) keeps most of the gain: 1/2/3-bit
  quantization still gives +10.8/+11.9/+12.2pp mean expected `P_D` over no
  RIS at `B=20`, matching the closed-form `sinc^2(1/2^b)` array-gain loss,
  with amortized control overhead under 0.5 bit per frame.
- Physics-based RIS channel (G5-P) uses the two-way bistatic radar law for
  the direct path and a three-leg cascaded loss for the RIS path: with a
  1024-element RIS and aperture scale `1e-2`, aligned RIS gives +16.3pp mean
  and +24.8pp worst-target expected `P_D` over no RIS at `B=20` with 100%
  QoS feasibility.
- Joint RIS control/report budget allocation (G5-R) charges
  `N * phase_bits / coherence_frames` bits against the same total budget:
  3-bit phase with the remaining bits on reports keeps +9.3pp mean and
  +12.9pp worst-target expected `P_D` over no RIS at total budget 40,
  within 0.3pp of the free-continuous upper bound.
- Joint RIS placement (G5-S) turns topology into a deployable degree of
  freedom: the best candidate position adds +7.1pp worst-target expected
  `P_D` over the fixed deployment at total budget 40, and +16.7pp over no
  RIS.
- Multigrid placement refinement (G5-T) adds another +2.8pp worst-target
  over the finite candidate search with only 34 deployment evaluations,
  reaching +19.5pp over no RIS at total budget 40.
- A Lipschitz grid-search certificate (G5-U) bounds further-refinement loss
  by `L h sqrt(d)/2`: with empirical `L = 2.97e-3`, the second refinement
  improves worst-target expected P_D by only +0.46pp, inside the 1.29pp
  bound at spacing 5.
- Lipschitz-adaptive branch-and-bound (G5-V) finds `(6.25, 39.375, 4.5)`
  with worst-target expected P_D 0.987 after 251 evaluations and certifies
  the deployment optimum within +1.03pp under the used Lipschitz constant
  `3.43e-3`.
- A localized two-phase search (G5-W) closes the certificate to 0.10pp in
  111 evaluations on a single-seed objective, and reports a bounded 0.79pp
  gap on the 3-seed averaged objective rather than claiming convergence.
- Paired bootstrap CIs (G5-CI) confirm the primary gains are significant:
  aligned RIS vs no RIS at B=20 has +12.33pp mean (CI [11.66, 13.02]) and
  +17.79pp worst-target (CI [15.73, 20.03]) with 100% win rate.
- Deployment paired CIs (G5-DCI) close the deployment-search gap: G5-T vs
  fixed gives +9.85pp worst-target (CI [7.52, 12.08]) and G5-V vs fixed
  +10.45pp worst-target (CI [7.78, 13.08]), both with 100% win rate.
- A global resource fairness ledger (G5-RF) shows the RIS gain is not bought
  with extra resources: at total budget 40, the G5-T deployment uses 37
  total bits and 4645 TB symbols versus 40 bits and 4648 symbols for no-RIS,
  while raising mean expected P_D by +12.7pp and worst-target by +19.5pp.
- An exact information-coordinate audit (G46) shows that raw deflection/KL
  overstates the P_D-consistent budget by 2.38-2.78x; `rho_exact` orders
  soft P_D monotonically (0.774 at 0.205 -> 0.933 at 0.351).
- A centralized/consensus architecture switch (G47) turns that budget
  comparison into a feasible detector: at B=8/12 peer consensus raises worst
  P_D to 0.881 (+10.68/+5.68pp) and makes the 0.85 QoS target feasible,
  while B>=16 returns to centralized soft.
- A target-wise architecture switch (G48) uses the order inequality
  `min_q max(a_q,b_q) >= max(min_q a_q, min_q b_q)` to guarantee no loss
  versus the global switch and adds +0.49/+1.55/+1.55pp worst P_D at
  B=12/16/20.
- An additive soft-report reallocation (G49) spends the freed peer-target
  bits on centralized targets with exact expected-P_D marginals; it adds
  +0.75pp at B=16/20 over G48 and +1.55/+0.85pp at B=28/40 with a
  nondecreasing-worst-P_D certificate.
- A limiting-target mode ascent (G50) lets a peer target switch back to
  centralized soft only when the switch strictly raises the worst P_D; it
  adds +0.39pp at B=12 over G48 (0.8858 -> 0.8898).
- Stochastic mobility with RIS reconfiguration latency (G51) couples the
  dynamic 6G scene with per-frame target-wise/mode-ascent fusion: ideal mode
  ascent reaches 0.852 worst-over-time P_D and 90.625% QoS, versus 0.847 and
  81.25% for target-wise switching.
- MMSE prediction-aware RIS (G52) uses the AR(1) conditional mean to design
  the next-frame phase, improving latency-1 worst-over-time P_D from 0.7217
  to 0.7283 and QoS from 43.75% to 46.875%.
- Multi-step MMSE prediction (G53) gives stale-phase worst-P_D gains
  +0.65/+3.24/+5.24pp for h=1/2/3, matching the `1-rho^{2h}` error-covariance
  growth; exact per-frame horizon selection adds +0.86pp over the best fixed
  MMSE, and hysteresis architecture reconfiguration halves switches while
  bounding worst loss by delta, with a cost-aware delta frontier under
  per-switch control bits.
- A negative G54 audit rejects expected-gain covariance-aware phase design:
  it improves its surrogate but degrades exact worst P_D from 0.7200 to
  0.6557 under quantization, reinforcing the exact-system evaluation rule.

## Where the innovation is weak

1. The selection algorithm still has no universal approximation ratio or
   regret bound for arbitrary covariance.  Gate G4 now supplies a formal
   objective (expected `P_D` over the exact reception law), set
   monotonicity, and bounded-regime submodularity with a classical `1 - 1/e`
   property and empirical ratio 1.0 on small instances.  Gate G3 supplies
   the monotone `P_D`-optimal fusion family.
2. Absolute gains in the saturated default G2 are small (0.01-0.02), and one
   cell (`rho=0.85`, `B=20`) has a paired-diff CI that crosses zero.  The
   non-saturated stress gate now provides the large-gain evidence:
   `+0.172` at `B=6` and `+0.114` at `B=9`, with 100% win rate and
   significant paired-diff CIs.
3. The front end is a toy-resolution model, and the RIS channel is a
   controlled additive-power model rather than a full cascaded-channel SDR.
   Physical realism can be challenged.
4. Resource accounting now has a same-scale table showing the integration
   gain is resource-driven (fixed total energy drops 86.7% -> 50%), with a
   time-bandwidth ledger (L=1 2575 vs L=4 4111 symbols).  The fixed-total-TB
   path still needs OTFS grid scaling.
5. G1-A grouped consistency is now covered for SNR/amplitude groups
   (0.8/1.0/1.3): deflection 0.55/0.33/0.40, P_D-gain 0.97/0.89/0.77.
   Doppler/leakage/correlation groups and scatter/calibration artifacts are
   still optional.

## Verdict

- If the target venue accepts "scenario + validation methodology" as a
  contribution, the current evidence is borderline sufficient, but the
  following are required before submission:
  1. Switch the main selector to Exact-`P_D`-Gain Greedy and state the
     deflection failure as a negative result; use the Gate G3
     `P_D`-optimal linear fusion rule so the underlying `P_D` is
     set-monotone at operating points.
  2. Complete the same-scale resource accounting table.
  3. Add Doppler/leakage/correlation grouped consistency and
     scatter/calibration data if the venue asks for them.
  4. Report G2 gains with paired CIs and be explicit that they are small in
     saturated regimes but large and significant in non-saturated regimes.
- If the venue requires algorithmic novelty, Gates G3 and G4 now supply two
  formal properties: the KKT-derived monotone `P_D`-optimal fusion family,
  and the monotone/submodular expected-`P_D` selection objective with real
  tight-budget gains.  Gate G5 additionally replaces the old scenario with a
  6G-relevant RIS-assisted channel that produces large, QoS-meaningful gains.
  A universal approximation guarantee for arbitrary covariance and a full
  RIS physical model would further strengthen the claim.

## Recommended next steps

1. Extend the Gate G4 bounded-regime guarantee to non-proportional covariance
   or derive a curvature-aware approximation ratio; the fusion-level
   monotonicity is now covered by Gate G3 and the expected-`P_D` monotonicity
   by Gate G4.
2. Complete the resource accounting gate (three fairness paths).
3. Run G1-A grouped consistency and produce scatter/calibration artifacts.
4. Update README, PAPER_OUTLINE, and Word appendices to reflect this audit.
5. Refine the RIS channel to per-element mutual coupling/polarization and
   close the multi-seed adaptive certificate within a practical evaluation
   budget; phase quantization, control overhead, two-way/cascaded path-loss
   physics, joint control/report-bit allocation, finite-deployment placement,
   multigrid refinement, a Lipschitz grid-search certificate, adaptive
   branch-and-bound, and an epsilon-closed single-objective certificate are
   now covered by G5-Q/G5-P/G5-R/G5-S/G5-T/G5-U/G5-V/G5-W, and paired
   bootstrap CIs are covered for G5/G5-P/G5-S/G5-R and the deployment gains
   of G5-S/T/V/W (G5-CI/G5-DCI); the global resource fairness ledger
   (G5-RF) closes the report/control fixed-total-TB path, leaving only the
   sensing OTFS-grid scaling path open.
