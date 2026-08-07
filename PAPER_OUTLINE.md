# UAV-ISAC Exact Budgeted Soft-Information Fusion Paper Outline

## Working title

Exact Budgeted Soft-Information Fusion for UAV-ISAC under Correlated
Reporting Losses

## Novelty positioning (must be explicit in the paper)

- New: the post-communication correlated value model (quantization/BSC/
  correlated-erasure statistics enter H0/H1 moments before fusion).
- New: the `P_D`-optimal one-parameter fusion family and the exact
  heterogeneous-cost budget/max-min selection certificates with a stated
  branch-and-bound pruning bound.
- Not new: the conditional marginal-deflection greedy itself, which is
  retained as a baseline/extension.
- Claim: combinatorial exactness under target-separable, additive-cost
  assumptions; system-level gains are small and are reported with two-sided
  statistics and Holm correction.

## Abstract bullets

- Multistatic UAV-ISAC sensing produces correlated soft evidence that is
  quantized, corrupted, and partially erased before fusion.
- A moment-matched Gaussian model carries communication loss into evidence
  moments rather than applying a post-hoc reliability coefficient.
- A conditional set-dependent greedy re-ranks reports as the selected set
  grows; it degenerates to static individual-deflection Top-K under
  independent evidence and equal cost.
- Gate G1-A/B/C/D validate evidence calibration, report-channel closure,
  conditional ranking value, and greedy-vs-Oracle behavior.
- Under a strongly correlated system model, the conditional greedy beats
  static Independent-Deflection Top-K in 77.5% of audited configurations.
- A KKT-derived `P_D`-optimal linear fusion family makes the underlying
  detection probability set-monotone, and an expected-`P_D`-gain greedy over
  the exact reception law adds a monotone, bounded-regime submodular
  selection objective with verified tight-budget gains.
- A geometry-aware normalized RIS power-gain model is instantiated as an
  application: the
  direct plus RIS-cascaded path whose phase profile is a controllable
  physical resource, turning a blocked weak target into a feasible NLoS
  illumination and raising worst-target `P_D` by about 15-18pp at tight
  budgets.
- An exact information-coordinate audit and a target-wise
  centralized/consensus architecture switch show that scarce-report regimes
  should fall back to peer consensus per target (0.85 QoS becomes feasible
  at B=8) without consuming extra report bits; freed soft-report bits are
  reallocated and limiting peer targets may switch back to centralized soft
  only when the switch strictly improves the worst target.  The same mode
  switching is validated under AR(1) stochastic mobility with RIS
  reconfiguration latency and multi-step conditional-mean prediction.

## 1. Introduction

- Motivation: UAV-OTFS-ISAC reports are correlated, communication-corrupted,
  and budget-limited.
- Gap: existing work optimizes trajectories, beamforming, or bandwidth, but
  not selective fusion of post-communication correlated soft evidence.
- Scope: fixed geometry/waveform, aligned candidate targets, one fusion
  center, and a controllable RIS-assisted sensing channel.

## 2. Related work

- OTFS-ISAC: waveform and receiver design for doubly selective DD-domain
  sensing, but without post-communication soft-report fusion.
- RIS-assisted ISAC/OTFS: controllable propagation and beamforming, but not a
  joint bit-budget coupling of RIS control and report selection.
- UAV-ISAC: trajectory, deployment, and beamforming surveys, but not a
  selective soft-information reporting chain.
- Distributed detection: quantized soft fusion under bandwidth constraints,
  but mostly independent observations and per-sensor BSC.
- Communication-constrained ISAC: time/frequency/power allocation, but not
  per-report bits and RIS control/placement under an expected-P_D objective.
- Full survey and citation details in `RELATED_WORK.md`.

## 3. System model and scenario

- OTFS DD evidence with fractional Doppler and leakage.
- RIS-assisted direct-plus-cascaded sensing channel with a controllable
  phase profile; alignment is an additive-power gain that is monotone in
  array gain.
- Per-path Fisher-type covariance from matched-filter curvature.
- Quantization, BSC, detectable erasure, random effective received set.
- Moment-matched Gaussian fusion and expected deflection.
- Multi-target QoS and budget constraints.

## 4. Method

- Static Individual-Deflection Top-K (baseline).
- Conditional-Deflection Greedy (main adapted method).
- Exact-P_D-Gain Greedy (alternative, best in correlated smoke).
- P_D-optimal linear fusion family: `w(mu)` interpolates between the
  H1-whitened matched filter and the deflection-optimal score; its KKT
  optimum is the P_D-optimal linear score and is set-monotone at
  `P_D >= 0.5`.
- Expected-P_D-gain greedy over the exact post-communication reception law,
  using the monotone fusion family; the expected objective is set-monotone
  and monotone submodular in the proportional-covariance strong-evidence
  regime.
- RIS phase-profile selection from a target-aligned beam codebook, jointly
  evaluated with report selection under the expected-P_D objective.
- Joint RIS control-bit and report-bit allocation under one total budget
  `B_total = B_report + N * phase_bits / coherence_frames`.
- Joint RIS placement search over a deployment candidate set, selected by
  worst-target expected P_D under the same budget identity.
- Coarse-to-fine multigrid refinement of the RIS deployment, adding a
  bounded number of local deployments around the coarse optimum.
- Lipschitz grid-search certificate: for an `L`-Lipschitz deployment
  objective, spacing `h` grid search has suboptimality at most
  `L h sqrt(d)/2`.
- Lipschitz-adaptive branch-and-bound deployment search that splits only
  boxes whose upper bound can beat the current best and carries a bounded
  optimality certificate.
- Practical epsilon-closed two-phase deployment search: localize with the
  finite/fine candidate search, then close the certificate in a small box.
- Exhaustive Oracle (small instances).
- Not claimed: submodularity, approximation ratio, universal Top-K dominance.
- Formal statements and proofs are collected in `FORMAL_PROOFS.md`.

## 5. Gates and results

### G0-C waveform front end

- Separated-scene recovery 96.7% with four-frame integration and
  sidelobe-aware CFAR.
- Per-path covariance reduces GOSPA by roughly 30%.
- Resource fairness (same-scale, 30 trials per column): fixed per-frame energy
  raises 86.7% to 100% with 4x energy; fixed total energy drops to 50%, so
  the gain is resource-driven.  Time-bandwidth: L=1 total 2575 symbols vs
  L=4 4111; fixed-total-time-bandwidth path requires grid scaling.

### G1-A evidence-moment calibration

- Positive-definite covariance after shrinkage.
- Formal 10k run (5000 train / 5000 test geometry): Spearman 0.588 for
  relative miss-deficit reduction and logit gain (CIs [0.23, 0.83] and
  [0.21, 0.84]); deflection does not pass the 0.6 gate.  With exact `P_D`
  gain or logit `P_D` gain as the predicted score, held-out Spearman is
  0.996/0.994 in the formal 10k run (CIs [0.98, 1.00] and [0.97, 1.00]),
  so a `P_D`-gain selector passes G1-A formally.
- Grouped consistency (amplitudes 0.8/1.0/1.3): deflection Spearman
  0.55/0.33/0.40, all below 0.6; `P_D`-gain predicted 0.97/0.89/0.77, all
  above 0.6.

### G1-B report-channel closure

- Exact vs Monte Carlo moments across bits 1-4, BER, erasure, correlation.
- Max mean error 4.08%, max covariance error 8.51%.

### G1-C conditional ranking value

- Degeneracy test passes.
- Correlated scenario: greedy chooses low-correlation report with higher P_D.

### G1-D greedy vs Oracle

- First-order vs exact marginal gain Spearman 0.90.
- Greedy matches Oracle in 50% of small configs; budget interactions remain.

### G2 system-level sweep

- 20-seed fair global-budget comparison: proposed 0.898 vs Sensing 0.898,
  Independent 0.897, Communication 0.773, All-scheduled 0.935; exact-P_D
  greedy 0.900 (best).
- Strong-correlation model (20 seeds): proposed 0.870 vs Independent 0.855,
  wins in 83.1% of configurations; exact-P_D greedy 0.880.
- Multi-rho sweep (0/0.3/0.5/0.7/0.85): conditional beats Static ID Top-K
  with positive paired-diff CIs for rho>=0.3 in most cells; at rho=0.85,
  B=20 the CI crosses zero.  Exact-P_D greedy is strongest at every rho.
- Non-saturated stress gate: at B=6 conditional mean P_D 0.692 vs 0.520
  (+0.172, CI [0.161,0.181], win 100%); at B=9 0.813 vs 0.699 (+0.114, CI
  [0.105,0.123]); worst-target P_D improves from 0.471 to 0.569 and 0.577 to
  0.782.  Saturation at B=12 removes the gain.
- Scaling (10 seeds): Exact-P_D gains over Static ID Top-K grow with M/Q:
  (8,3) +0.006, (12,5) +0.007, (16,5) +0.015, (16,8) +0.067 mean P_D;
  worst-target +0.127 at (16,8).  Runtime 0.25 s vs 0.15 s at (16,8).
- Non-saturated scaling-stress (20 seeds): conditional beats Static ID Top-K
  by +0.114 mean P_D at Q=3/5/8; worst-target improvement grows from +0.205
  to +0.272 with Q.

### G3 P_D-optimal linear fusion

- The deflection-optimal score is not set-monotone under unequal H1
  covariance; in the controlled unequal-covariance audit it decreases on
  258/1318 (19.6%) operating-point edges with a maximum 16.1pp drop.
- The one-parameter KKT family gives 0 decreasing edges at the same operating
  points, a mean 0.63pp and maximum 21.2pp P_D gain over the
  deflection-optimal score per addition edge.
- In the proportional-covariance regime the family degenerates to the exact
  closed form `P_D = Phi((sqrt(D) - z_FA) / sqrt(c))` (max absolute error
  `1.3e-15` over 960 checks).
- Greedy-level audit (deterministic reports, unit report cost): fusion gain
  +0.83pp mean P_D on the deflection schedule; re-running exact-P_D greedy
  under the optimal rule adds +0.16pp mean on average, +0.99pp total;
  scheduling gain is positive in 23.3% of instances, so the claim is the
  monotone fusion rule, not universal scheduling dominance.

### G4 expected-P_D greedy under the reception law

- At B=20 with correlated erasures (20 seeds), expected-P_D greedy gives
  +1.14pp mean expected P_D (bootstrap CI [0.47, 1.74], 85% win) and +7.56pp
  worst-target gain over the proposed selector; the hybrid policy (keep the
  better of the two candidate schedules by expected P_D) gives +1.44pp mean
  and +6.50pp worst-target.
- At B=30 the mean gain is +0.04pp (hybrid +0.36pp) with +3.05pp
  worst-target gain; at B=40 the expected-P_D greedy alone is -0.91pp in
  mean, while the hybrid stays never worse (+0.02pp).
- Theory: expected P_D over a mixture of monotone set functions is monotone;
  in the proportional regime with strong evidence it is monotone submodular
  (0 violations on 3040 audited edges), and greedy matches the exhaustive
  single-target oracle with ratio 1.0 on 20 small instances.

### G5 RIS-assisted 6G sensing channel

- The sensing channel is a direct plus RIS-cascaded path; the RIS phase
  profile adds power `(ris_strength * array_gain)^2` to the evidence SNR, so
  alignment is monotone in link quality.
- At B=20 (20 seeds), aligned RIS plus expected-P_D greedy gives +12.3pp mean
  expected P_D and +17.8pp worst-target over no RIS, and +9.0pp mean over
  random RIS phase; QoS feasibility at the 0.85 target rises from 0% to 95%.
- At B=30 the aligned-RIS gains are +10.9pp mean and +15.2pp worst-target
  over no RIS, with 100% QoS feasibility.

### G5-Q finite-resolution RIS phase

- The closed-form mean array-gain loss of b-bit phase quantization is
  `sinc^2(1/2^b)`, verified numerically (1-bit 0.405, 2-bit 0.811, 3-bit
  0.950).
- At B=20, 1/2/3-bit quantized RIS retain +10.8/+11.9/+12.2pp mean expected
  P_D over no RIS (worst-target +15.1/+17.1/+17.6pp), versus +12.3/+17.8pp
  for ideal continuous phase.
- Amortized control overhead over a 100-frame coherence block is 0.16/0.32/
  0.48 bits per frame for 16 elements, so RIS control is not free but is
  negligible relative to the sensing report budget.

### G5-P physics-based RIS cascaded channel

- The direct path uses the two-way bistatic radar law
  `1 / (R_tx^2 R_rx^2)`; the RIS path uses the three-leg cascaded loss with
  an `N^2` coherent array gain and an optional direct-path blockage.
- With a 1024-element RIS and aperture scale `1e-2`, aligned RIS gives
  +16.3pp mean and +24.8pp worst-target expected P_D over no RIS at B=20
  (100% QoS feasibility), and +13.9pp / +21.8pp at B=30; aligned beats random
  phase by +15.7pp / +13.3pp mean.
- A 256-element RIS with aperture scale `1e-2` retains +10.8pp mean and
  +13.3pp worst-target at B=20, so the physics-based channel keeps the 6G
  mechanism meaningful without free channel assumptions.

### G5-R joint RIS control and report budget

- The control and report planes compete for one total budget; the best
  realizable allocation is 3-bit phase with the remaining bits spent on
  reports.  At total budget 40 and a 64-frame coherence block this gives
  +9.3pp mean and +12.9pp worst-target expected P_D over no RIS, within
  0.3pp of the free-continuous upper bound.
- At total budget 60 the quantized joint allocation gives +8.6pp / +13.4pp,
  so charging RIS control overhead honestly does not erase the 6G gain.

### G5-S joint RIS placement

- The fixed RIS position was far from the blocked weak target.  A small
  deployment search selects `(0, 20, 8)`: at total budget 40 and 64-frame
  coherence it raises worst-target expected P_D from 0.882 (fixed position)
  to 0.952, i.e. +7.1pp over fixed and +16.7pp over no RIS, with +11.7pp mean
  gain; at total budget 60 the worst-target gain over fixed is +6.6pp.
- Placement is therefore a deployable degree of freedom, jointly optimized
  with phase resolution and report allocation.

### G5-T multigrid RIS placement

- One local refinement around the G5-S coarse optimum gives `(0, 30, 6)`:
  worst-target expected P_D rises from 0.952 to 0.980 at total budget 40
  (+2.8pp over the finite search, +9.8pp over the fixed position, +19.5pp
  over no RIS), with +12.7pp mean gain.
- The search evaluates 34 deployments (7 coarse + 27 fine); each refinement
  halves the local grid spacing at a bounded additive evaluation cost.

### G5-U Lipschitz deployment certificate

- The deployment objective (worst-target expected P_D over seeds) has an
  empirical Lipschitz constant `2.97e-3`.  The second refinement to
  `(0, 35, 5)` improves worst-target expected P_D from 0.983 to 0.988
  (+0.46pp), inside the `1.29pp` bound at spacing 5 and the `2.57pp` bound at
  spacing 10.
- The lemma is proven for any L-Lipschitz function and checked numerically on
  the physics-channel objective; it bounds the loss of further refinement
  without requiring gradient information.

### G5-V Lipschitz-adaptive deployment search

- The adaptive search finds `(6.25, 39.375, 4.5)` with worst-target expected
  P_D 0.987 after 251 evaluations in a local deployment box, and certifies
  the deployment optimum within +1.03pp under the used Lipschitz constant
  `3.43e-3`.
- The certificate is reported honestly as a bounded gap rather than a fully
  epsilon-closed result at this evaluation budget.

### G5-W epsilon-closed deployment certificate

- In the localized box, the adaptive search closes the certificate to 0.10pp
  within 111 evaluations on a single-seed objective (`epsilon_closed =
  true`), finding `(11.875, 34.21875, 6.5)` with worst-target expected P_D
  0.983.
- On the 3-seed averaged objective the certificate over the original local
  box is bounded at 0.16pp after 3001 main-search evaluations plus 400
  corner-refinement evaluations; a second branch-and-bound run inside a 2 m
  box around the best point closes to 0.09pp in 23 evaluations
  (`local_epsilon_closed = true`).  The original-box gap is reported
  honestly instead of being conflated with the local closure.

### G5-CI paired bootstrap intervals

- Aligned RIS vs no RIS at B=20: mean expected P_D +12.33pp (CI
  [11.66, 13.02]) and worst-target +17.79pp (CI [15.73, 20.03]), 100% win.
- Physics RIS (1024 elements, aperture `1e-2`, B=20): +16.26pp mean (CI
  [14.82, 17.77]) and +24.78pp worst-target (CI [23.24, 26.42]).
- Best placement vs fixed at total budget 40: +7.06pp worst-target (CI
  [5.76, 8.33]); joint 3-bit allocation vs no RIS: +8.26pp mean (CI
  [7.28, 9.20]).

### G5-DCI deployment paired intervals

- Per-seed deployment rows close the previous CI gap: G5-T vs fixed gives
  +4.42pp mean (CI [3.09, 5.85]) and +9.85pp worst-target (CI
  [7.52, 12.08]); G5-V vs fixed +10.45pp worst-target (CI [7.78, 13.08]);
  G5-W vs fixed +9.10pp worst-target (CI [7.20, 10.93]), all with 100% win
  rate.

### G5-RF global resource fairness ledger

- Under `B_total = B_report + N * phase_bits / coherence_frames` and a
  conservative 1-symbol-per-bit ledger, the G5-T deployment uses 25 report
  bits plus 12 control bits (4645 TB symbols) versus 40 report bits (4648
  symbols) for no-RIS, while raising mean expected P_D from 0.863 to 0.990
  (+12.7pp) and worst-target from 0.785 to 0.980 (+19.5pp), with QoS
  feasibility 0% to 100%.
- The gain is not resource-driven in reverse: the passive RIS uses slightly
  less total time-bandwidth and total occupation than no-RIS.

### G5-SEN parameter sensitivity

- At total B=40, mean gain over no RIS rises from +1.3pp to +11.0pp as
  `aperture_scale` goes from `1e-3` to `3e-2`; worst-target gain rises from
  +2.3pp to +14.1pp.
- Mean gain rises from +1.1pp (`N=64`, report budget 39 bits) to +13.3pp
  (`N=1024`, report budget 28 bits), so RIS element gain dominates the
  control-overhead cost in the audited ledger.
- Increasing `coherence_frames` from 32 to 256 raises mean gain from +7.3pp
  to +8.7pp and worst-target gain from +8.9pp to +10.5pp; `C=16` is
  infeasible under the joint budget.
- Direct-path blockage is the regime condition: mean gains are
  +10.1/+8.2/+3.5/+2.2pp for blockage 0.001/0.01/0.1/1.0, and the
  worst-target gain CI crosses zero at blockage 1.0.  The paper therefore
  claims RIS NLoS illumination for a blocked weak target, not universal
  dominance.

### G5-SOTA literature-style baselines

- At total B=40 and the G5-T deployment, the proposed chain beats the
  strongest soft baseline (RIS + deflection Top-K) by +0.68pp mean and
  +1.63pp worst-target expected P_D with a positive 12-seed bootstrap CI.
- Against no-RIS deflection Top-K the gains are +15.2pp mean and +27.5pp
  worst-target; against random RIS deflection Top-K they are +14.4pp and
  +25.4pp; against uniform one-report soft allocation they are +21.8pp and
  +46.1pp.
- The exact 1-bit counting baseline is held to a fusion-level P_FA near
  0.008; the proposed method gains +75.7pp/+79.9pp (no RIS) and
  +52.1pp/+64.5pp (RIS), all with 100% win rate.
- On the same proposed schedule, replacing deflection fusion with the
  P_D-optimal fusion family gives +0.25pp mean and +0.45pp worst-target,
  isolating the fusion contribution from the selection contribution.

### G6 budget saturation frontier

- Without RIS, worst-target expected P_D saturates near 0.788 and the 0.85
  QoS target is not reached at any tested total budget up to 44.
- With the G5-T RIS deployment, `B_total=20` (8 report bits after control
  overhead) already reaches 100% QoS feasibility; the minimum budget for QoS
  is therefore 20, not 40 or higher.
- Discrete coordinate ascent over add/remove/swap moves from the forward
  greedy schedule gives zero additional gain in all audited cells, so the
  forward greedy is a single-move local optimum in this scenario.  The
  remaining headroom is architectural (RIS phase/placement), where
  continuous projected gradient is the natural next step.

### G7 continuous shared-phase RIS optimization

- A single physical phase profile is parameterized by one ULA steering
  cosine and optimized with analytic array-power gradients.  The
  worst-array-power surrogate is a documented negative result: it improves
  the surrogate but reduces system expected P_D.
- System-level grid-plus-refine optimization recovers the weak-target
  steering cosine (`u ~ -0.99`).  At `B_total=20`, the shared
  system-optimized profile gives worst-target expected P_D 0.831, which is
  +8.2pp over no-RIS, +31.9pp over random shared phase, and -12.3pp below
  per-target ideal phase.
- QoS feasibility of the shared single beam is 50%/67%/67% at total budgets
  20/28/40, versus 100% for per-target ideal phase.  A single shared beam is
  therefore physically limited; subarray-based multi-beam phase profiles are
  the next architecture step.

### G8 exact quota-constrained selection

- The audited model has equal report costs, so the report-budget constraint
  is a cardinality constraint.  For each target every report subset is
  evaluated exactly and the best subset of each size is retained; all
  per-target report quotas are searched globally.
- In every audited budget/scenario cell the exact selector equals forward
  greedy to numerical precision (0.0pp gain), so forward greedy is already
  globally optimal for the selection layer under the audited equal-cost
  model.
- The remaining gap to all-scheduled (up to 3.6pp worst at B=20 no-RIS,
  1.7pp worst at B=40 RIS) is therefore an architectural resource gap, not
  selection headroom.

### G8-K exact budget-constrained selection under heterogeneous costs

- Why this gate exists: G29 turns per-UAV quantizer bits into a design
  variable, so report costs are heterogeneous and the equal-cost exact
  certificate of G8 no longer covers the feasible set under the
  communication-bit budget.
- The equal-cost quota enumeration is generalized to a multiple-choice
  knapsack DP over targets and total bits.  Each target contributes an
  enumerated report subset with its exact bit cost, and DP states are pruned
  only by componentwise Pareto dominance.
- The DP is exact for the two-stage score (QoS gap, weighted expected `P_D`,
  worst target): on controlled 3-target/4-report models it matches an
  exhaustive global oracle in 100% of 100 cells (20 seeds x 5 budgets).
- On the variable-rate demo scenario the exact schedule is never worse than
  forward greedy on the lexicographic score in all 100 cells; mean
  worst-target gains are +1.27pp at `B=5` (p=0.015) and +2.57pp at `B=7`
  (p=0.009), while at `B=9` the worst-target gain is -1.00pp (p=0.895),
  documenting the lexicographic trade-off.
- Complexity is polynomial in targets and budget for fixed per-target report
  counts; the exponential part is the exhaustive subset evaluation already
  used by G8.  Per-target cost-value Pareto dominance removes options that
  are no cheaper and no better than another subset, so the DP option set is
  the cost-value frontier rather than all enumerated subsets.

### G8-M exact max-min budget selection under heterogeneous costs

- Why this gate exists: the audited system objective is worst-target
  expected `P_D`, so the G8-K lexicographic selector (QoS gap, weighted mean,
  worst target) is not the same problem as G30's max-min objective.
- For a threshold `t`, feasibility asks whether each target can choose an
  enumerated subset with value at least `t` under the total bit budget; this
  is a multiple-choice knapsack feasibility problem and is solved exactly by
  a cost DP.
- Feasibility is monotone in `t`, so the exact max-min value is found by
  binary search over the finite set of per-target subset values.  The
  returned schedule is feasible at the optimal threshold.
- On controlled 3-target/4-report models the selector matches an exhaustive
  global max-min oracle in 100% of 100 cells (20 seeds x 5 budgets); in the
  tight-budget variable-rate demo it is never worse than forward greedy in
  all 100 cells.  Controlled gains are +5.37pp at `B=5` (p<1e-6), +8.24pp
  at `B=7` (p<1e-6), +0.39pp at `B=9` (p=0.083), and +3.33pp at `B=11`
  (p=3.9e-4); system gains are significant at B=5/7/9/11 (p<0.05) with
  95% bootstrap CIs excluding zero.
- The same per-target dominance rule is applied before threshold search, so
  dominated options never enter the feasibility DP.
- The threshold-feasibility DP keeps the componentwise-Pareto frontier at
  every accumulated cost, so among schedules attaining the optimal max-min
  threshold the returned one is also lexicographically best in QoS gap,
  weighted mean, and worst target.

### G8-S scaled exact-threshold max-min selection

- Why this gate exists: G8-M enumerates all per-target subsets, so the
  exact certificate stops being practical as the report count grows.
- For threshold `t`, let `m_q(t)` be the minimum bit cost for target `q` to
  reach `t`.  A global schedule with all values at least `t` exists if and
  only if `sum_q m_q(t) <= B`, because the per-target minima are jointly
  attainable and every feasible subset costs at least its minimum.
- `minimum_cost_to_threshold` uses branch-and-bound with a closed-form
  Cauchy upper bound on the `P_D`-optimal linear-score shift, valid at every
  operating point; pruning happens only when that bound is below `t` or when
  cost cannot beat the best known feasible subset.  Small low-`P_D` models
  use exact subset enumeration instead; when every model is within
  `max_exhaustive_reports`, `scaled_maxmin_select` delegates directly to the
  exact selector rather than running an epsilon binary search.
- On the 20-seed controlled models the scaled selector agrees with exact
  enumeration to zero absolute error; on a synthetic 12-report model it finds the
  minimum-cost subset without enumerating all 4096 subsets.  The worst case
  remains exponential and is reported as a pruning certificate rather than a
  polynomial guarantee.  A report-count benchmark covers
  R=8/12/16/20/24/28/32/40 (up to about $1.1\times 10^{12}$ exhaustive
  subsets) in 24-60 ms with exact minimum costs of 1-2 bits.
- A greedy warm start supplies a cost upper bound; when that bound is small,
  all cost-bounded subsets below it are enumerated exactly to prove
  minimality before branch-and-bound.  This reduces a 16-report,
  threshold-0.9 case from about 60s to about 1.5s on the current machine.

### G8-target exact selection across target count

- At Q=3/4/5 and B=8/12/16 (3 seeds, grid 32), the budget and max-min
  selectors match their exhaustive oracles in 100% of all 27 cells and are
  never worse than forward greedy.
- Mean wall time grows from about 180 ms at Q=3 to about 300-360 ms at Q=5,
  so the exact certificate remains practical as the target count grows.

### G9 aperture-conserved subarray multi-beam RIS

- The RIS aperture is partitioned into disjoint target-aligned subarrays;
  total elements and per-element phase-bits are unchanged, so the control
  overhead is exactly the same as G5.
- A discrete coordinate-ascent search moves 32/16/8-element blocks between
  targets while maximizing mean worst-target expected P_D over seeds.
- Optimized allocations are budget-dependent: `B=20` gives `(6,85,165)`,
  `B=28` gives `(6,149,101)`, and `B=40` gives `(6,173,77)` elements across
  the three targets.
- At `B_total=28`, the optimized subarray profile raises worst-target
  expected P_D to 0.913, i.e. +5.2pp over single shared weak-aligned phase,
  +13.7pp over no-RIS, and reaches 100% QoS feasibility; it remains 6.7pp
  below the per-target ideal upper bound.

### G10 per-subarray steering-cosine optimization

- Fixing the G9 aperture allocations, each subarray steering cosine is
  optimized by coordinate ascent over a bounded grid; total aperture and
  control overhead are unchanged.
- Optimized steering mostly rotates the small first block toward the strong
  target direction (`u ~ 0.96-0.99`) while keeping the weak block aligned.
- Worst-target expected P_D improves to 0.858/0.916/0.935 at B=20/28/40,
  i.e. +0.41/+0.23/+0.14pp over G9; QoS remains 50% at B=20 and 100% at
  B=28/40.
- The per-target ideal gap is 9.6/6.5/4.8pp worst, so steering refinement is
  useful but secondary to aperture allocation.

### G11 fixed-budget RIS aperture scaling

- The RIS path power scales as `N^2`, while the control overhead scales as
  `N * phase_bits / coherence_frames`; the feasible report budget is
  `B_total - overhead`.
- At `B_total=20`, `N=1024`, 3-bit phase, `C=256`, and an equal subarray
  allocation, the system reaches 100% QoS feasibility with only 8 report
  bits and worst-target expected P_D 0.982.
- `N=512` with 3-bit phase and `C=256` also reaches 100% QoS at B=20 with
  worst P_D 0.943; the original `N=256`, 3-bit, `C=64` configuration stays
  at 50%.
- This is the direct architecture-level answer to the saturation question:
  proposed performance is limited by aperture and control-overhead
  amortization, not by the report-selection algorithm.

### G12 model-driven architecture derivation

- The design is not a four-variable exhaustive search.  Under the subarray
  approximation, the array gain of an `a_q`-element block is `a_q/N`, so the
  RIS-to-direct power ratio is `K_q a_q^2 sinc^2(1/2^b)`.
- Local deflection scales quadratically with evidence SNR, so the
  equal-allocation weak-target surrogate is
  `J(N) = beta (1 + kappa N^2)^2 (R - LN)` with
  `kappa = K_weak sinc^2(1/2^b)/9`, `L = b/C`, `R = B_total`.
- The first-order condition is the quadratic
  `5 kappa L N^2 - 4 kappa R N + L = 0`, giving `N*` in closed form.
- Exact validation: `B=20, b=1, C=64` gives `N* = 1016 -> 1024` and 100%
  QoS; `B=20, b=3, C=256` gives `N* = 1363 -> 1344` and worst P_D 0.974.
  The derivation therefore determines which architecture variables matter
  and why, and the exact system confirms the predicted operating point.

### G13 max-min deflection water-filling

- The allocation is derived from `D_q(a_q) = beta_q (1 + kappa_q a_q^2)^2`,
  a monotone convex surrogate.  Aperture is moved from the current
  highest-D target to the lowest-D target until the minimum stops improving;
  this is the max-min water-filling fixed point, not an enumeration.
- A first implementation equalized marginal derivatives
  `dD_q/da_q`, which is the wrong KKT for a max-min objective and degraded
  exact P_D; it is recorded as a rejected branch.
- At G12-derived apertures, correct water-filling raises worst-target P_D
  from 0.900 to 0.911 (`N=1024,b=1,C=64,B=20`), from 0.974 to 0.992
  (`N=1344,b=3,C=256,B=20`), and from 0.999599 to 0.999995
  (`N=2048,b=3,C=256,B=40`), all with 100% QoS.

### G14 exact-array-factor allocation

- The surrogate is upgraded to
  `D_q(a) = beta_q (1 + K0_q N^2 G_q(a))^2`, where `G_q(a)` is the exact
  squared array factor including cross-block interference.
- Max-min water-filling on this exact surrogate raises the surrogate minimum
  in every tested configuration, but exact system P_D does not consistently
  improve: `N=1024` is 0.8pp worse, `N=1344` is statistically unchanged,
  and `N=2048` is 0.00008pp better.
- The conclusion is recorded as a negative/equivocal result: surrogate
  exactness is necessary but not sufficient for system-level optimality.

### G15 greedy-aware system-level allocation

- The allocation objective is the exact system function
  `F(a) = mean_seed min_q E_PD(q, S_q(a))`, where `S_q(a)` is the
  expected-P_D greedy schedule under the allocation.
- Coordinate ascent over single-block aperture transfers accepts only moves
  that increase `F`; the stopping point is a local optimum of the true
  objective, not a surrogate.
- Exact validation improves worst P_D from 0.911 to 0.924
  (`N=1024,b=1,C=64`), 0.911 to 0.927 (`N=704,b=3,C=128`), and 0.981 to
  0.985 (`N=960,b=3,C=128,B=28`).
- At `N=2048,b=3,C=256,B=40`, the coarse 8-element local search ends 0.0018pp
  below the exact-surrogate allocation; this is a local-search limitation,
  not a system-optimality claim.

### G16 single-element refinement and local certificate

- Starting from each G15 allocation, 4/2/1-element coordinate ascent improves
  all five configurations.
- `exact_single_move_gradients` evaluates every one-element transfer of the
  exact system objective; all five final allocations satisfy
  `local_optimal=true` with maximum gradient `<= 0`.
- Final values: 0.924107 (`N=1024`), 0.927345 (`N=704`), 0.991896
  (`N=1344`), 0.985738 (`N=960,B=28`), and 0.999986 (`N=2048,B=40`).
- This is the first system-level local optimality certificate in the
  allocation line; it covers single-element transfers, not all multi-block
  moves.

### G17 bounded multi-block certificate

- Every zero-sum reallocation moving at most `T=3` elements in total is
  evaluated exactly on the system objective, iterated to the best point in
  that neighborhood.
- Four configurations are already multi-block local optima; `N=2048,B=40`
  improves from 0.999986 to 0.999988 in 7 rounds.
- All five final allocations satisfy `local_optimal=true` with respect to
  the `T<=3` neighborhood, extending the G16 certificate from single moves
  to simultaneous multi-block moves.

### G18 joint RIS placement and allocation

- The exact system objective `F(s,a)` is optimized by alternating coordinate
  ascent: allocation uses the T<=3 multi-block certificate, and position
  uses 2/1/0.5-meter coordinate steps.
- All three tested configurations improve and certify both degrees of
  freedom:
  - `N=1024,B=20`: 0.925224 at `(-2,30,6)`, allocation `(271,479,274)`;
  - `N=1344,B=20`: 0.992907 at `(0.5,31,6)`, allocation `(649,443,252)`;
  - `N=2048,B=40`: 0.999997 at `(6.5,34,5)`, allocation `(956,619,473)`.
- This closes the placement-allocation loop with joint local certificates.

### G19 progressive decentralization

- Decentralization is opened in four stages: fair local scheduling,
  deflection fusion instead of P_D-optimal fusion, owner-only decisions, and
  1-bit hard decisions with counting fusion.
- At B=40/N=2048, local scheduling loses only 0.0013pp, deflection fusion
  0.0026pp, owner-only 0.014pp, while 1-bit hard decisions lose 18.8pp and
  QoS drops to 50%.
- At B=20/N=1024 with a 4-bit report budget, 5-bit soft reports cannot be
  transmitted, so centralized soft fusion equals owner-only; 1-bit hard
  decisions can send three reports but cannot meet the global P_FA=0.05 with
  one vote per target, so they are infeasible/worse (QoS 0%).  The earlier
  +6.0pp claim was corrected after enforcing the global P_FA constraint.

### G20 amplified distributed hard detection

- The distributed branch is no longer a fixed baseline: local P_FA and the
  counting threshold are optimized per target under the global P_FA
  constraint.
- At B=40/N=2048, optimized 1-bit detection raises worst P_D from 0.812
  (fixed local P_FA) to 0.944, and QoS from 0% to 100%; centralized soft
  fusion remains at 0.999997.
- At B=20 with one vote per target, no counting rule meets global P_FA, so
  the distributed 1-bit branch is infeasible; this is stated honestly.

### G21 network-level decentralization

- Report links and owner fusion are removed entirely.  Every UAV makes a
  local 1-bit decision and the target is declared by an optimized majority
  threshold over all `M=8` UAVs.
- At B=20/N=1024, peer majority reaches worst P_D 0.955 (centralized soft
  0.925); at B=20/N=1344 it reaches 0.998 (centralized 0.993); at
  B=40/N=2048 it reaches 0.9999977 (centralized 0.9999967), all 100% QoS.
- This shows that with high local SNR, fully distributed consensus voting can
  match or exceed centralized soft fusion, and it requires zero report bits.

### G22 degraded multi-hop consensus

- Partial observability and per-hop link erasure are introduced through the
  effective participation `obs * (1 - (1 - r)^hops)`.
- At B=40/N=2048, observability 0.75 gives worst P_D 0.966 and link
  reliability 0.8 gives 0.977, both below centralized 0.999997; three hops at
  0.8 recover to 0.9998; severe degradation (participation 0.546) drops to
  0.877.
- The distributed advantage is therefore conditional on network quality, not
  a universal property.

### G23 correlated failure and heterogeneous observability

- Effective participation is
  `obs_i * (1 - p_c) * (1 - (1 - r)^hops)`, with `p_c` a network-wide common
  failure and per-UAV `obs_i` derived from target distance.
- At B=40/N=2048, common failure 0.2/0.4 gives 0.977/0.909, heterogeneous
  observability gives 0.936, and the severe combination gives 0.858, all
  below centralized 0.999997.
- Correlated failures remove the distributed advantage faster than
  independent degradation.

### G24 scalability across target and UAV counts

- The gate varies Q in {2,4,6} and M/Q in {1,2,3} with report budget
  `20*Q` bits plus the fixed RIS control overhead.
- RIS ideal phase reaches 100% QoS in every tested cell except Q=6,M=6;
  peer majority reaches 100% QoS when M/Q>=3 for Q=2 and Q=6, and Q=4.
- No-RIS is topology-sensitive: Q=2,M=4 gives worst P_D 0.460, while
  Q=2,M=2 gives 0.915.
- Consensus voting needs enough UAVs per target to compensate for losing
  soft information.

### G25 scaled white-box G18

- For Q>3 the exhaustive multi-block certificate is replaced by the derived
  max-min water-filling allocation and exact 0.5m position ascent; the model
  and system objective remain exact.
- The scaled G18 keeps 100% QoS in every tested cell except Q=6,M=6, where
  the ideal-phase upper bound also fails.
- At Q=6,M=12 it reaches worst P_D 0.922, versus 0.792 for peer majority and
  0.934 for ideal phase; at Q=4,M=8 it reaches 0.964 versus 0.915 for peer.

### G26 mobility and time-varying blockage

- UAVs rotate along smooth trajectories and targets move on bounded paths;
  the weak-target direct-path blockage varies sinusoidally over frames.
- Worst-over-time no-RIS QoS is 0%; RIS ideal is 100%; static subarray is
  68.75%; adaptive subarray recomputed each frame is 81.25%.
- Adaptive subarray improves worst P_D from 0.841 to 0.847 and mean from
  0.869 to 0.874 over the static allocation.

### G27 multi-RIS deployment

- Total aperture is fixed at 256 elements and the same control-overhead
  identity is used for one/two/three RISs.
- One RIS with 256 elements is the best configuration at every budget:
  worst P_D 0.955/0.980/0.983 at B=20/28/40.
- Two/three RISs are worse (e.g., B=28: 0.923/0.927) because non-coherent
  power addition loses the `N^2` coherent aperture; placement diversity only
  partially compensates.

### G28 multi-RIS split and placement optimization

- For a single target the convex power sum has its maximum at an extreme
  split, so multi-RIS only helps through multi-target geometry differences.
- Equal split gives 0.924; exact-system local optimization gives
  `(8, 248)` elements with the second RIS at `(4,42,2)`, reaching 0.986 and
  slightly exceeding the single-RIS 0.981.

### G29 variable-rate soft/hard reporting

- `build_models` now supports per-UAV quantizer bits, so report cost and
  quantization fidelity are jointly variable.
- Fixed 5-bit soft is best at B=20/28 (0.953/0.977); adaptive soft rates beat
  it at B=40 (0.988 vs 0.981); 1-bit hard remains weakest.

### G30 global rate-profile optimization

- Per-UAV quantizer bits are optimized by exact-system coordinate ascent;
  each UAV changes its rate by one bit at a time.
- B=28: optimized 0.988 vs fixed 5-bit 0.981 and adaptive 0.974.
- B=40: optimized 0.991 vs fixed 5-bit 0.987 and adaptive 0.987.
- All final profiles satisfy `single_change_local_optimal=true`.
- G30-E re-checks the profiles with the exact max-min selector G8-M on the
  same 2-seed/grid-256 audit as G30.  At B=28 the G30 profile remains a
  single-rate exact local optimum at 0.9879 (zero gain).  At B=40 the greedy
  certificate is false under the exact objective: exact coordinate ascent
  improves the profile from 0.9911 to 0.9916 and certifies it as a
  single-rate exact local optimum.

### G31 exact soft/hard hybrid fusion

- The combined score is an exact Gaussian-plus-hard LLR rule, evaluated by
  enumerating hard-decision patterns and searching the soft threshold to meet
  the global P_FA.
- At B=28/40, hybrid reaches 0.977/0.969 versus pure soft 5-bit
  0.977/0.981 and hard-only 0.843/0.736.
- Hybrid is not automatically better than soft-only; the schedule must be
  optimized.

### G32 interference sensitivity

- Effective SINR is `SNR / (1 + INR)` with per-UAV INR injected into the
  moment-matched model.
- INR=0 dB: only RIS ideal reaches 100% QoS; INR=3 dB: all architectures
  fail QoS; INR=10/20 dB: worst P_D below 0.2/0.06.

### G33 spatial interference and RIS placement

- INR at each UAV follows `inr_ref * (d_ref/d_i)^2` from a fixed source.
- No-RIS fails all strengths; fixed RIS keeps 100% QoS; RIS position
  optimization adds +0.3pp worst P_D.

### G34 multiple interference sources

- Per-UAV INR is the sum of free-space path losses from three sources.
- Mean INR 0.087; no-RIS 0.810 QoS 0%; fixed RIS 0.983 QoS 100%; optimized
  placement 0.987 QoS 100%.

### G35 1-D ULA versus 2-D UPA

- With 256 elements, UPA and ULA are almost identical in clean and spatial
  interference scenarios.
- The 2-D aperture does not add P_D in the current geometry; its value would
  require elevation diversity or null-steering.

### G36 UPA null-steering

- Phases are optimized on `J = G_target - lambda sum G_interference` with an
  analytic gradient.
- Reflected INR falls from 0.0267 to 0.0106 (-60%) while target gain drops
  from 1.000 to 0.984; B=40 worst P_D improves from 0.98112 to 0.98216.

### G37 directly quantized null-steering

- Discrete coordinate ascent over `2^b` phase levels directly optimizes the
  scalarized array power.
- Reflected INR 0.01052 vs continuous-quantized 0.01056; B=40 worst P_D
  0.982166 vs 0.982165.

### G38 joint quantized nulling and placement

- Each candidate position redesigns quantized null-steering phases for all
  targets; position coordinate ascent maximizes the exact system objective.
- B=40 worst P_D improves from 0.98217 (fixed) to 0.98481 (optimized), while
  reflected INR rises from 0.0105 to 0.0296, an explicit target-gain versus
  reflected-interference trade-off.

### G39 distributed features under relaxed thresholds

- Budgets 20/24/28 and QoS targets 0.70/0.75/0.80 make the distributed
  branch feasible.
- Peer multi-hop stays at worst P_D 0.953; optimized hard stays around
  0.84-0.86; centralized soft ranges 0.953-0.977.

### G40 low-budget/low-SNR distributed

- N=128, spatial interference, budgets 12/16/20.
- At B=12 centralized drops to 0.786; peer clean 0.858 and peer multi-hop
  0.855 outperform it; hard optimized 0.765 is feasible at QoS 0.70.

### G41 consensus parity boundary

- Gaussian approximation gives `M_min ~ 14-17` for the audited local
  decisions.
- Empirical wins: M>=8 at B=8/12, M=6 at B=8; centralized regains the lead
  at B>=16.

### G42 optimized-local-threshold boundary

- Minimizing `M_min` over the local P_FA grid lowers the theoretical bound by
  9-13%.
- Example M=16: 13.70 -> 12.14, closer to the exact wins at B=8/12.

### G43 exact Poisson-binomial boundary

- Exact majority feasibility uses Poisson-binomial tails and starts at M=6,
  matching empirical wins.
- Gaussian approximation predicted M_min=13.36, so exact enumeration closes
  the theory-empirics gap.

### G43-B exact minimum majority count and monotonicity audit

- The exact Poisson-binomial feasibility is evaluated on every prefix of the
  voter sequence, so `M_min` is the first feasible prefix rather than a
  coarse-grid estimate.
- The audit checks monotonicity explicitly.  A homogeneous example with
  `p0=0.1`, `p1=0.7`, `alpha=0.05`, `beta=0.7` has feasibility trace
  `[F,F,T,F,T]`, so `M=3` feasible and `M=4` infeasible; binary search over
  `M` is therefore not valid in general without a monotonicity certificate.
  In the audited run, the system voter sequences at `num_uavs=8/12/16` are
  also non-monotone, confirming that the precondition must be checked per
  sequence.

### G44 fundamental information budget

- Normalized information `rho = J/D_full` explains the soft family
  monotonically: P_D rises from 0.774 at rho=0.507 to 0.933 at rho=0.946.
- Consensus keeps rho nonzero when soft reports are unaffordable; the full
  principle is in `FUNDAMENTAL_PRINCIPLE.md`.

### G45 closed-form resource law (negative)

- A simple `Phi((sqrt(d0(1+n)g^2)-z)/sqrt(c))` law is tested across
  N=64/128/256 and B=12-40.
- It overestimates P_D by up to 30pp and saturates to 1 for N>=128; the law
  is rejected because quantization/correlation break the closed-form
  assumption.

### G46 exact information budget

- Each method is mapped to the exact effective deflection
  `D_eff=(Phi^{-1}(P_D)+z_FA)^2`, so `rho_exact=D_eff/D_full` is the
  P_D-consistent information coordinate.
- Soft raw `rho` overestimates `rho_exact` by a factor 2.38-2.78 because the
  raw deflection ignores quantization loss and correlation; the raw
  coordinate cannot serve as a performance law.
- With N=128, interference, and the same 4-seed audit, soft P_D rises from
  0.774 at `rho_exact=0.205` to 0.933 at `rho_exact=0.351`; peer consensus
  has `rho_exact=0.284` at P_D 0.881 and wins only where report bits are
  scarce, while optimized 1-bit hard fusion has `rho_exact<=0.199`.
- All three methods are budget-consistent: soft/hard used report bits never
  exceed the report budget, and peer consensus uses zero report bits.

### G47 exact centralized/distributed architecture switch

- The mode selector compares exact worst-target P_D between centralized soft
  fusion and peer majority and picks the higher detector; both branches are
  calibrated to the same global P_FA, so the switch is feasible by
  construction.
- At B=8/12, peer is selected in 100% of seeds and raises worst P_D from
  0.774/0.824 to 0.881 (+10.68/+5.68pp), making the 0.85 QoS target
  feasible.
- At B>=16, the exact switch returns to centralized soft and never loses;
  the fixed `report_budget < 10` threshold reproduces the exact mode choice
  in this audit and is treated as a design parameter, not a universal law.

### G48 target-wise architecture switch

- The global switch is refined to per-target mode selection:
  `P_D,q = max(soft_q, peer_q)`, which is never worse because
  `min_q max(a_q,b_q) >= max(min_q a_q, min_q b_q)`.
- At B=12/16/20, target-wise switching adds +0.49/+1.55/+1.55pp worst P_D
  over the global switch; at B=8 it reaches 0.881 and keeps the 0.85 QoS
  target feasible.
- Peer is selected for roughly 92% of targets at B=8, 83% at B=12, and 50%
  at B=16, so the per-target mixture is not a global binary choice.

### G49 soft-report reallocation

- The freed soft-report bits of peer-selected targets are greedily added
  back to the remaining centralized targets with exact expected-P_D marginal
  gains; the update is additive, so centralized P_D and worst P_D are
  nondecreasing.
- Reallocation adds +0.75pp at B=16/20 over G48 (worst P_D 0.9250) and
  +1.55/+0.85pp at B=28/40; B=8 has no free report budget and keeps QoS
  feasible via peer consensus.

### G50 two-sided mode ascent

- After reallocation, a limiting peer target may spend unused report bits on
  its original soft schedule and switch back to centralized soft, but only
  when the upgraded P_D strictly raises the current worst target.
- This adds +0.39pp at B=12 over G48 (0.8858 -> 0.8898) using 3.75 report
  bits on average, while B=16-40 matches G49; failed and tied-worst upgrade
  attempts are discarded, so the worst P_D is nondecreasing.

### G51 stochastic mobility with RIS reconfiguration latency

- The deterministic trajectories of G26 are upgraded to AR(1)-correlated
  random UAV/target positions plus a random time-varying blockage.
- Under N=128 and total B=40, no-RIS worst-over-time P_D is 0.524; static
  RIS mode ascent reaches 0.705, latency-1 RIS 0.722, ideal target-wise
  0.847, and ideal mode ascent 0.852.
- Ideal mode ascent raises QoS over time from 81.25% to 90.625% versus ideal
  target-wise switching, and latency-1 beats static by +1.64pp worst P_D.

### G52 MMSE prediction-aware RIS

- Under the AR(1) trajectory, the RIS phase is designed from the conditional
  mean predictor `hat p_t = n_t + rho (p_{t-1} - n_{t-1})`, which minimizes
  the mean squared position error.
- MMSE prediction improves latency-1 worst-over-time P_D from 0.7217 to
  0.7283 (+0.65pp) and QoS from 43.75% to 46.875%; ideal per-frame phase
  remains the upper bound.

### G53 multi-step MMSE prediction

- The predictor generalizes to latency `h`:
  `hat p_{t|t-h} = n_t + rho^h (p_{t-h} - n_{t-h})`, with error covariance
  `(1 - rho^{2h}) sigma^2 I`.
- MMSE over stale-phase worst P_D grows with h: +0.65pp at h=1,
  +3.24pp at h=2, and +5.24pp at h=3, consistent with the increasing
  prediction-error covariance.
- Exact per-frame horizon selection over stale/MMSE h=1/2/3 raises the best
  fixed MMSE worst P_D from 0.7283 to 0.7369 (+0.86pp) and QoS from 46.875%
  to 53.125%.
- A hysteresis architecture reconfiguration with delta=0.02 keeps QoS at
  53.125%, reduces switches from 4.50 to 2.25 per seed, and loses at most
  0.00104 worst P_D, inside the delta bound.
- Under a 6-bit control budget, per-switch costs of 1/3/6 bits select
  optimal delta 0.00/0.03/0.05 with worst P_D 0.7369/0.7250/0.7217 and
  4.50/1.50/0.75 switches per seed.

### G54 covariance-aware phase (negative)

- A covariance-aware phase maximizes the expected squared array gain under
  the AR(1) direction error and is monotone in that surrogate.
- Under 3-bit quantization and exact P_D, it degrades worst-over-time P_D
  from 0.7200 (MMSE) to 0.6557 (-6.43pp) at h=3, so it is rejected; the
  surrogate is not a system-level design criterion.

## 6. Boundaries and open items

- Equal bandwidth/frame-budget/communication-rate accounting is not yet
  complete.
- The G5-RF ledger resolves the report/control fixed-total-TB path under the
  1-symbol-per-bit convention; the sensing OTFS-grid scaling path from the
  G0-C ledger remains open.
- The RIS channel is not a full cascaded-channel SDR; phase quantization and
  control overhead are covered by G5-Q, two-way/cascaded path-loss physics by
  G5-P, parameter sensitivity by G5-SEN, finite-deployment placement by
  G5-S, while per-element mutual
  coupling, polarization, fully continuous placement optimization, and full
  waveform-level modeling remain open (G5-T is a coarse-to-fine finite
  refinement, not a gradient method).
- Paired bootstrap CIs now cover G5/G5-P/G5-S/G5-R and the deployment gains
  of G5-S/T/V/W (G5-CI/G5-DCI).
- The SOTA comparison is a 12-seed draft with internal and literature-style
  baselines; external numerical results from other systems should be added
  only after their channel and budget assumptions are matched.
- The G6 gate shows selection saturation is not the binding constraint;
  continuous gradient updates for RIS phase/placement and a differentiable
  relaxation of the bit allocation remain future work.
- G7 shows a single shared ULA beam cannot simultaneously serve all targets
  as well as per-target time multiplexing; subarray-based multi-beam and
  multi-RIS designs remain open.
- G8 exactness holds only under equal report costs and exhaustive subset
  enumeration; the nonuniform-cost knapsack variant is now implemented as
  G8-K, with the exhaustive global oracle match verified on the controlled
  model.
- G9 uses a 1-D ULA and contiguous element blocks; interleaved subarrays,
  non-contiguous allocations, and joint phase/placement optimization remain
  open.
- G10 optimizes block steering by a local coordinate ascent and does not
  prove global optimality of the steering cosines.
- G11 shows a feasible high-aperture operating point at B=20 but does not
  optimize coherence length jointly with phase bits and aperture; a
  three-way trade-off search remains open.
- G12 derives `N*` from the weak-target equal-allocation surrogate; the
  allocation vector is not yet included in the first-order derivation.
- G13 water-filling ignores cross-block interference in the surrogate and
  is validated only on the exact system at the tested configurations.
- G14 shows the exact-array surrogate is still not aligned with the greedy
  expected-P_D objective; a system-level first-order condition is required.
- G15 achieves a system-level local optimum but with 8-element transfers;
  finer moves, multi-block moves, and a convergence certificate remain open.
- G16 certificates cover single-element transfers; multi-block and joint
  placement-allocation certificates remain open.
- G17 covers up to three simultaneously moved elements; larger `T` and joint
  placement remain open.
- G18 is a local joint certificate; global placement and multi-block
  certificates remain open.
- G19 uses fixed report-bit granularity; variable-length reporting and
  hybrid soft/hard report policies remain open.
- G20 optimizes scalar local P_FA per target; joint local-threshold and
  vote-schedule design remains open.
- G21 peer majority ignores report-link erasure because no report links are
  used; a fully connected local-decision network is still assumed.
- G22 uses independent per-hop erasure and scalar observability; correlated
  topology failures and heterogeneous observability remain open.
- G23 uses one common failure probability for the whole network; grouped
  regional failures and time-varying topology remain open.
- G24 uses per-target ideal phase and fixed per-target budget; joint
  architecture scaling with optimal aperture per target is open.
- G25 does not claim the T<=3 allocation certificate for Q>3; only position
  certificate and derived water-filling are used.
- G26 uses deterministic trajectories and a sinusoidal blockage; stochastic
  mobility and trajectory optimization remain open.
- G27 uses fixed total aperture and fixed RIS positions; joint total-aperture
  allocation across RISs is open.
- G28 is a local split/placement search without a global certificate.
- G29 uses a fixed per-target equal-budget rate profile; global rate
  optimization remains open.
- G30 optimizes one rate per UAV; per-target rate profiles and soft/hard
  hybrid fusion remain open.
- G31 uses a fixed hybrid schedule; joint rate/hybrid schedule optimization
  remains open.
- G32 uses uniform INR across UAVs; spatial interference patterns and
  suppression remain open.
- G33 uses one fixed interference source; multiple sources and null-steering
  are open.
- G34 sums direct interference but does not implement RIS null-steering.
- G35 shows UPA is not automatically better than ULA in this geometry.
- G36 null-steering is per-target continuous-phase optimization; quantized
  nulling and multi-target shared phases remain open.
- G37 directly optimizes quantization but still one phase vector per target.
- G38 is a local joint search; shared multi-target nulling remains open.
- G39 relaxes the QoS threshold rather than changing the channel; very low
  budget/SNR regimes remain untested.
- G40 covers the low-budget regime but only with one interference source.
- G41 uses one QoS target and fixed local P_FA for the theoretical formula.
- G42 still uses the Gaussian approximation and averaged local probabilities.
- G43 is exact but evaluated on a discrete M grid.
- G44 shows information monotonicity within soft fusion; hard/consensus
  require threshold optimization to align their rho.
- G45 proves that a naive resource law is invalid; exact moment propagation
  is required.
- G46 replaces raw normalized information with the exact
  P_D-consistent `rho_exact`; it does not claim that any architecture gains
  performance, only that budget comparisons must use the exact coordinate.
- G47 mode switching is exact for the two audited branches but does not
  jointly optimize rate profiles, local thresholds, or a third architecture;
  the fixed threshold is audited on one operating profile.
- G48 keeps the centralized soft schedule fixed while switching some targets
  to peer; G49 adds a greedy reallocation of the freed bits, and joint
  schedule/mode optimality remains open.
- G49 uses a greedy exact-marginal reallocation and is certified only as a
  monotone ascent; a joint schedule/mode optimality certificate remains
  open.
- G50 switches only limiting peer targets and only when the switch strictly
  improves the worst target; a global certificate over all report
  reallocations and mode sequences remains open.
- G51 uses a declared AR(1) frame model, not a continuous-time SDR
  trajectory; G52 covers one-step conditional-mean prediction, and
  multi-step/covariance-aware stochastic-optimal control remain open.
- G52/G53 use conditional-mean prediction only; covariance-aware phase
  design and control-aware RIS remain open.
- G53 assumes the AR(1) correlation is known and does not optimize the
  reconfiguration horizon jointly with the report budget.
- G54 is a negative result: expected-gain covariance-aware phase design does
  not transfer to quantized exact P_D; quantization-aware robust design
  remains open.
- Rayleigh time diversity does not recover the integration gain in the toy
  front end.
- Strong FWER under mixed nulls requires closed testing.
- Same angle-DD collision decomposition remains a separate gate.
- Front end is toy-resolution, not bandwidth-consistent SDR.

## 7. Remaining paper-engineering tasks

1. Formal proof appendix is now drafted in `FORMAL_PROOFS.md`; needs to be
   rendered into the paper appendix and cross-checked against the code.
2. Submission-grade unified tables and figures: a draft CSV/Markdown table
   and draft PNG figures now exist; finish paper layout and any required
   journal-specific styling.
3. Full paper prose is drafted in `PAPER_DRAFT.md`; polish it into
   submission style and verify every number against the result JSONs.
4. Multi-seed epsilon-closed deployment certificate: the current 3-seed
   objective is epsilon-closed inside a 2 m local box and bounded at 0.16pp
   over the original local box; closing the original-box gap is the remaining
   open item.
5. Continuous projected gradient for RIS phase and placement, jointly with a
   differentiable/soft report-allocation relaxation, as an alternative to
   forward-only greedy.
6. Subarray-based shared-phase RIS design that approaches the per-target
   ideal-phase upper bound without time multiplexing.
7. G8-K/G8-M/G8-S now render the knapsack certificates into the paper table
   and figures with 20-seed means/std, paired t-tests, and bootstrap CIs;
   remaining engineering is to extend the
   Pareto frontier bookkeeping and the exact certificate to larger target
   counts.
8. Interleaved or adaptive subarray allocation and joint
   RIS-placement/subarray optimization.
9. Global steering-certificate search and joint
   allocation-steering-placement optimization.
10. Joint `(N, phase_bits, coherence_frames, allocation)` optimization with
    a proven feasibility frontier under the control-overhead identity.
11. Extend the G12 first-order derivation to include target-specific
    allocation water-filling and cross-block interference.
12. Include cross-block interference in the G13 surrogate so the allocation
    first-order condition reflects the exact array factor.
13. Derive a system-level (greedy-aware) first-order condition instead of a
    surrogate-only condition for allocation.
14. Add finer and multi-block moves plus a local-optimality certificate to
    the G15 system ascent.
15. Extend the G16 certificate to multi-block and joint
    placement-allocation moves.
16. Increase `T` in the G17 certificate and add joint placement-allocation
    moves.
17. Globalize the G18 joint certificate or combine it with a Lipschitz
    placement bound.
18. Optimize report bit length jointly with the decentralization stage,
    especially in the 4-16 bit report-budget regime.
19. Optimize local thresholds jointly with vote schedules and topology in the
    distributed branch.
20. Extend G21 to multi-hop consensus and partial local observability.
21. Add correlated link failures and heterogeneous per-UAV observability to
    the G22 consensus model.
22. Add regional failure groups and time-varying topology to the G23 model.
23. Derive the minimum M/Q ratio required for consensus to match centralized
    soft fusion as Q grows.
24. Extend the G25 scaled architecture to a polynomial allocation
    certificate for Q>3.
25. Optimize UAV trajectories jointly with RIS allocation under stochastic
    blockage.
26. Optimize total-aperture split and RIS positions jointly for multi-RIS.
27. Add a Lipschitz or exhaustive local certificate for the G28 split/place
    search.
28. Optimize the global variable-rate profile jointly with the schedule.
29. Allow per-target rate profiles and soft/hard hybrid fusion within one
    target.
30. Optimize the soft/hard hybrid schedule jointly with the rate profile.
31. Model spatial interference and interference suppression in the INR
    sweep.
32. Add multiple interference sources and RIS null-steering.
33. Implement RIS phase nulling toward interference directions.
34. Use the UPA model with elevation diversity and null-steering.
35. Optimize quantized null-steering phases and shared multi-target nulling.
36. Joint quantized nulling and RIS placement.
37. Shared multi-target quantized nulling with one physical phase vector.
38. Test distributed branches in very low budget/SNR regimes.
39. Extend the low-budget distributed comparison to very low SNR and
    multiple interference sources.
40. Derive a closed-form `(M,Q,B,SNR)` parity surface including optimized
    local thresholds.
41. Remove the Gaussian approximation with exact Poisson-binomial parity
    equations.
42. G43-B now computes the exact `M_min` by exhaustive prefix evaluation and
    audits monotonicity; the binary-search shortcut is documented as invalid
    without a monotonicity certificate because homogeneous majority
    feasibility can be non-monotone in `M`.
43. Build a closed-form `P_D(rho, B, M, N)` resource-information law.
44. Derive a corrected law that includes quantization loss and correlation.
45. Fit or derive a monotone correction between raw and exact `rho` so that
    the exact information coordinate can be predicted without rerunning the
    full moment-propagation chain.
46. Jointly optimize the centralized/consensus mode switch with report-bit
    granularity and local-decision thresholds.
47. Certify the joint schedule/mode optimum after reallocation, or extend the
    ascent to a two-sided mode update.
48. Derive a local-optimality certificate for the G50 mode ascent with
    respect to single-report additions and limiting-target switches.
49. Replace the AR(1) frame model with a continuous-time trajectory and a
    prediction-aware RIS phase policy.
50. Replace conditional-mean-only phase design with
    prediction-error-covariance-aware phase design.
51. Jointly select the RIS reconfiguration horizon and report/control
    budget under the G53 prediction-error frontier.
52. Design quantization-aware robust phase with the exact system P_D, not the
    expected-gain surrogate rejected by G54.

## 8. Current repository mapping

- `uav_otfs_isac/front_end.py`: G0-C front end.
- `uav_otfs_isac/evidence_calibration.py`: G1-A.
- `uav_otfs_isac/report_channel_calibration.py`: G1-B.
- `scripts/run_g1c_conditional_ranking_gate.py`: G1-C.
- `scripts/run_g1d_greedy_vs_oracle_gate.py`: G1-D.
- `scripts/run_g2_system_sweep.py`, `run_g2_algorithm_negative_gates.py`: G2.
- `uav_otfs_isac/fusion.py`, `scripts/run_pd_optimal_fusion_gate.py`: G3.
- `uav_otfs_isac/expected_pd.py`,
  `scripts/run_expected_pd_greedy_gate.py`: G4.
- `uav_otfs_isac/ris_scenario.py`,
  `scripts/run_ris_isac_gate.py`: G5.
- `SYSTEM_MODEL.md` -- unified notation and model used by G3-G5.
- `RELATED_WORK.md` -- related-work survey and gap positioning.
- `FORMAL_PROOFS.md` -- formal proof appendix for G3-G5.
- `G18_THEORY.md` -- G18 convergence, complexity, and explicit-information
  inventory.
- `SCENARIO_COMPLEXITY.md` -- scenario simplification audit and upgrade
  roadmap.
- `PAPER_DRAFT.md` -- full draft manuscript.
- `paper/submission.md` -- submission-oriented manuscript with formal
  sections, a selection-results table, and numbered references.
- `paper/submission.docx` -- Word version generated and structurally audited
  by `scripts/md_to_docx.py` / `scripts/audit_submission_docx.py`.
- `paper/main.tex` -- IEEE-style LaTeX version generated and structurally
  audited by `scripts/md_to_latex.py` / `scripts/audit_submission_latex.py`.
- `paper/main.pdf` -- compiled IEEEtran PDF (Tectonic 0.17.0).
- `paper/references.bib` -- BibTeX entries for the manuscript references.
- `scripts/audit_submission_completeness.py` -- completeness audit for the
  submission manuscript.
- `SUBMISSION_CHECKLIST.md` -- pre-submission checklist and environment
  follow-ups.
- `paper_figures/algorithm_evolution.png` -- algorithm-evolution diagram
  regenerated by `scripts/draw_algorithm_evolution.py`.
- `paper_figures/scenario_evolution.png` -- scenario-evolution diagram
  regenerated by `scripts/draw_scenario_evolution.py`.
- `scripts/run_ris_sensitivity_gate.py` -- G5-SEN parameter sweep.
- `scripts/run_sota_baseline_gate.py` -- G5-SOTA literature-style baseline
  comparison.
- `uav_otfs_isac/sota_baselines.py` -- static soft baselines and exact
  1-bit counting fusion.
- `uav_otfs_isac/discrete_descent.py` -- discrete add/remove/swap ascent.
- `scripts/run_budget_saturation_gate.py` -- G6 budget saturation frontier.
- `uav_otfs_isac/ris_optimization.py` -- shared-phase analytic gradient and
  system-level optimization.
- `scripts/run_ris_shared_phase_gate.py` -- G7 shared-phase comparison.
- `uav_otfs_isac/exact_quota_selection.py` -- exact equal-cost selection.
- `scripts/run_exact_quota_gate.py` -- G8 exact upper-bound comparison.
- `scripts/run_exact_budget_gate.py` -- G8-K heterogeneous-cost exact budget
  comparison against the exhaustive oracle and forward greedy.
- `scripts/run_exact_maxmin_gate.py` -- G8-M exact max-min comparison against
  the exhaustive oracle and forward greedy.
- `scripts/run_scaled_maxmin_gate.py` -- G8-S scaled max-min certificate and
  large-report-set branch-and-bound demonstration.
- `uav_otfs_isac/ris_subarray.py` -- aperture-conserved multi-beam phase and
  discrete aperture-gradient search.
- `scripts/run_ris_subarray_gate.py` -- G9 subarray multi-beam comparison.
- `scripts/run_ris_subarray_steering_gate.py` -- G10 steering optimization.
- `scripts/run_ris_aperture_scaling_gate.py` -- G11 aperture scaling.
- `uav_otfs_isac/architecture_objective.py` -- derived architecture
  objective and closed-form aperture condition.
- `scripts/run_derived_architecture_gate.py` -- G12 derived design.
- `scripts/run_waterfilling_architecture_gate.py` -- G13 water-filling
  allocation.
- `uav_otfs_isac/exact_allocation.py` -- exact array-factor surrogate.
- `scripts/run_exact_allocation_gate.py` -- G14 exact-array allocation.
- `scripts/run_system_allocation_gate.py` -- G15 greedy-aware allocation.
- `scripts/run_single_move_certificate_gate.py` -- G16 local certificate.
- `scripts/run_multi_move_certificate_gate.py` -- G17 multi-block
  certificate.
- `scripts/run_joint_placement_allocation_gate.py` -- G18 joint
  placement-allocation certificate.
- `scripts/run_progressive_decentralization_gate.py` -- G19 progressive
  decentralization ablation.
- `scripts/run_amplified_distributed_gate.py` -- G20 optimized distributed
  hard detection.
- `scripts/run_network_decentralization_gate.py` -- G21 network-level
  decentralization.
- `scripts/run_degraded_consensus_gate.py` -- G22 degraded consensus.
- `scripts/run_correlated_consensus_gate.py` -- G23 correlated consensus.
- `scripts/run_scalability_comparison_gate.py` -- G24 scalability
  comparison.
- `scripts/run_scaled_g18_scalability_gate.py` -- G25 scaled G18
  scalability.
- `scripts/run_mobility_blockage_gate.py` -- G26 mobility/blockage.
- `scripts/run_multi_ris_gate.py` -- G27 multi-RIS deployment.
- `scripts/run_multi_ris_split_optimization_gate.py` -- G28 split/placement
  optimization.
- `scripts/run_variable_rate_report_gate.py` -- G29 variable-rate reporting.
- `scripts/run_global_rate_optimization_gate.py` -- G30 global rate
  optimization.
- `scripts/run_exact_rate_certificate_gate.py` -- G30-E exact max-min
  re-certification of the G30 profile.
- `scripts/run_hybrid_fusion_gate.py` -- G31 hybrid soft/hard fusion.
- `scripts/run_interference_sensitivity_gate.py` -- G32 interference.
- `scripts/run_spatial_interference_placement_gate.py` -- G33 spatial
  interference placement.
- `scripts/run_multi_interference_placement_gate.py` -- G34 multi-source
  interference placement.
- `scripts/run_upd_vs_ula_gate.py` -- G35 ULA/UPA comparison.
- `scripts/run_null_steering_gate.py` -- G36 null-steering.
- `scripts/run_quantized_null_steering_gate.py` -- G37 quantized
  null-steering.
- `scripts/run_joint_null_placement_gate.py` -- G38 joint nulling/placement.
- `scripts/run_distributed_relaxation_gate.py` -- G39 relaxed distributed.
- `scripts/run_low_budget_snr_distributed_gate.py` -- G40 low-budget
  distributed.
- `scripts/run_consensus_parity_boundary_gate.py` -- G41 parity boundary.
- `scripts/run_optimized_parity_boundary_gate.py` -- G42 optimized boundary.
- `scripts/run_exact_parity_boundary_gate.py` -- G43 exact boundary.
- `scripts/run_exact_min_majority_gate.py` -- G43-B exact minimum majority
  count and monotonicity audit.
- `scripts/run_fundamental_information_gate.py` -- G44 information budget.
- `scripts/run_resource_information_law_gate.py` -- G45 resource law
  (negative).
- `scripts/run_exact_information_budget_gate.py` -- G46 exact
  effective-information budget.
- `scripts/run_architecture_switch_gate.py` -- G47 centralized/distributed
  architecture switch.
- `scripts/run_target_wise_architecture_switch_gate.py` -- G48 target-wise
  architecture switch.
- `scripts/run_soft_reallocation_gate.py` -- G49 soft-report reallocation.
- `scripts/run_mode_ascent_gate.py` -- G50 two-sided mode ascent.
- `scripts/run_stochastic_mobility_gate.py` -- G51 stochastic mobility and
  RIS latency.
- `scripts/run_prediction_aware_ris_gate.py` -- G52 MMSE prediction-aware
  RIS.
- `scripts/run_multi_step_prediction_gate.py` -- G53 multi-step MMSE
  prediction.
- `scripts/run_covariance_aware_ris_gate.py` -- G54 covariance-aware phase
  (negative).
- `uav_otfs_isac/covariance_aware_ris.py` -- expected-gain covariance-aware
  phase optimizer audited and rejected by G54.
- `uav_otfs_isac/stochastic_mobility.py` -- AR(1) stochastic trajectory
  model and MMSE prediction used by G51/G52/G53.
- `uav_otfs_isac/architecture_switch.py` -- exact/fixed and target-wise
  architecture mode selection, reallocation, and mode ascent used by
  G47/G48/G49/G50/G51.
- `FUNDAMENTAL_PRINCIPLE.md` -- unified principle document.
- `scripts/build_paper_tables.py` -- draft unified tables and figures.
