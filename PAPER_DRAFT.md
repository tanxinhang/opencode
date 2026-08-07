# Exact Budgeted Soft-Information Fusion for UAV-ISAC under Correlated Reporting Losses

Draft manuscript assembled from `PAPER_OUTLINE.md`, `SYSTEM_MODEL.md`,
`RELATED_WORK.md`, and `FORMAL_PROOFS.md`.  All quantities and numbers are
traced to the audited repository results; venue-specific formatting,
references, and any final wording remain open.

A submission-oriented version with formal sections, tables, an
algorithm-evolution figure, proof sketches, and a numbered reference list is
in `paper/submission.md` (BibTeX entries in `paper/references.bib`; a Word
conversion is at `paper/submission.docx` and an IEEE-style LaTeX source at
`paper/main.tex`, compiled to `paper/main.pdf` by Tectonic 0.17.0).

Pre-submission steps are tracked in `SUBMISSION_CHECKLIST.md`.

## Abstract

Multistatic UAV-ISAC sensing produces correlated soft evidence that must be
quantized, transmitted over error-prone reporting links, and selectively
fused under a finite bit budget.  We formulate the post-communication chain
as a moment-matched Gaussian detection problem whose H0/H1 statistics include
quantization, BSC, and correlated erasure effects.  A one-parameter linear
fusion family whose KKT-optimal member is the P_D-optimal linear score at
operating points with P_D > 0.5 is derived, and an exact heterogeneous-cost
budget/max-min certificate closes the greedy gap under nonuniform report
costs.  A geometry-aware normalized RIS power-gain model is instantiated as
an application, and the numerical audit separates combinatorial exactness,
value-oracle discretization, and statistical evidence.

## 1. Introduction

Integrated sensing and communication (ISAC) on UAV platforms must reconcile
three tensions: sensing creates correlated per-UAV evidence, reporting links
are finite-rate and error-prone, and the sensing and control planes compete
for the same resource budget.  Prior UAV-ISAC work emphasizes trajectory,
beamforming, and resource allocation; prior RIS-ISAC work optimizes phase
profiles or placement; prior distributed-detection work usually assumes
independent per-sensor observations or a fixed report channel.  The gap is
the selective fusion of post-communication correlated soft evidence under
heterogeneous report costs and correlated reporting losses, with exactness
certificates for the combinatorial selection layer.

The paper makes three contributions.

1. A communication-in-the-loop system model: quantization, BSC, detectable
   correlated erasures, and a random received set are propagated into H0/H1
   evidence moments before fusion.
2. A KKT-derived P_D-optimal linear fusion family with an independent
   zero-extension monotonicity argument, plus exact heterogeneous-cost
   budget/max-min selection certificates with a stated branch-and-bound
   pruning bound.
3. An audited, reproducible UAV-ISAC application study with a geometry-aware
   normalized RIS power-gain model, multi-seed paired comparisons, two-sided
   statistics with Holm correction, and explicit recording of where
   guarantees do and do not apply.

## 2. Related work

Full citations and gap statements are in `RELATED_WORK.md`.  The relevant
lines are:

- OTFS-ISAC waveform and receiver design, which does not model the
  post-communication selective fusion chain.
- RIS-assisted ISAC and RIS-assisted OTFS, which optimize phase/beamforming
  but not the joint control-bit/report-bit allocation.
- UAV-ISAC surveys, which emphasize trajectory and beamforming.
- Distributed detection with quantized soft fusion, which usually assumes
  independent observations and per-sensor channel models.
- Exact Poisson-binomial counting and consensus detection, which certify
  distributed majority but do not model the correlated post-communication
  evidence chain or the non-monotonicity of majority feasibility.
- Communication-constrained ISAC resource allocation, which allocates
  time/frequency/power but not per-report bits and RIS control/placement
  under an expected-P_D objective.

To the best of the draft's knowledge, no prior work combines all five lines
in one audited chain.

## 3. System model

The unified notation is in `SYSTEM_MODEL.md`; this section states the
minimal model needed for the proofs and gates.

Let `M` transmitting UAVs serve `Q` target hypotheses with one receive
array.  Per-UAV evidence under H0/H1 is moment-matched Gaussian
`(mu0, mu1, Sigma0, Sigma1)`.  A report `i` costs `b_i` bits, is corrupted by
an independent BSC with bit-flip probability `epsilon_i`, and may be erased.
The receiver fuses only the received reports.  The false-alarm point is
fixed at `P_FA`, and the primary metrics are mean and worst-target expected
`P_D` over the reception law.

The linear fusion score is `s = w^T x` with threshold
`tau = w^T mu0 + z sqrt(w^T Sigma0 w)`, where
`z = Phi^{-1}(1 - P_FA)`.  The detection probability is

`P_D(w) = Phi( (w^T delta - z sqrt(w^T Sigma0 w)) / sqrt(w^T Sigma1 w) )`

with `delta = mu1 - mu0`.

The RIS channel adds a controllable NLoS path.  The controlled model is
`gain_iq = 1 + (strength_q array_gain_iq)^2`; the physics-based model uses

`P_dir = 1 / (R_tx^2 R_rx^2)`,

`P_ris = N_ris^2 array_gain_iq^2 aperture_scale /
         (R_1^2 R_2^2 R_3^2)`,

and `gain_iq = 1 + P_ris / P_dir`, with an optional direct-path blockage
factor for the weak target.  Both models are additive in power, hence
monotone in array alignment and never reduce a link's evidence SNR.

The control/report resource identity is

`B_total = B_report + N_ris * phase_bits / coherence_frames`.

Under the conservative 1-symbol-per-bit convention, report and control bits
map to time-bandwidth symbols.

## 4. Method

### 4.1 P_D-optimal linear fusion

In whitened coordinates `y = L^T w`, `a = L^{-1} delta`,
`Q = L^{-1} Sigma1 L^{-T}`, the detection probability is

`P_D = Phi( (a^T y - z ||y||) / sqrt(y^T Q y) )`.

Maximizing `P_D` is equivalent to maximizing `F(y) = a^T y - z ||y||` over
the compact ellipsoid `y^T Q y = 1`.  At any global optimum with
`P_D > 0.5`, KKT conditions give

`w(mu) = L^{-T} (Q + mu I)^{-1} L^{-1} delta`, `mu >= 0`.

Optimizing this one-parameter family therefore attains the global linear
optimum, and zero extension makes the family-optimal P_D set-monotone at the
audited operating points.  In the proportional regime `Sigma1 = c Sigma0`,
the family degenerates to

`P_D = Phi( (sqrt(D(S)) - z) / sqrt(c) )`,

with `D(S) = delta_S^T Sigma0,SS^{-1} delta_S`.

### 4.2 Expected-P_D selection

For target `q` and schedule `S_q`, the objective is

`E_PD(q, S_q) = E_gamma[ P_D(owner union received(S_q, gamma)) ]`.

The two-stage greedy first minimizes normalized miss-deficit and then
maximizes expected-P_D gain per report bit.  Every fixed-pattern P_D is
set-monotone at audited operating points, so the expectation is monotone.
In the proportional-covariance, diagonal-Sigma0, strong-evidence regime the
per-pattern P_D is a concave function of a modular deflection, so the
expected objective is monotone submodular and equal-cost cardinality greedy
inherits the classical `1 - 1/e` bound.  The implemented bit-budgeted
greedy does not inherit that ratio and is not claimed to.

### 4.3 RIS phase, placement, and deployment search

The RIS phase profile is a target-aligned beam codebook with `b`-bit
uniform phase quantization.  The mean array-gain loss is
`sinc^2(1/2^b)` in the large-N limit, with the finite-N correction
`1/N + (1 - 1/N) sinc^2(1/2^b)`.  Placement is first searched over a
finite candidate set and then refined by coarse-to-fine multigrid and by a
Lipschitz-adaptive branch-and-bound search.  For an `L`-Lipschitz objective
and a grid with spacing `h`, the grid-search loss is bounded by
`L h sqrt(d) / 2`; branch-and-bound stops with an epsilon-optimality
certificate when the global upper bound is within epsilon of the best
evaluated value.

### 4.4 Exact selection under heterogeneous report costs

G29 makes the per-UAV quantizer bits a design variable, so report costs are
heterogeneous and the equal-cost quota certificate of G8 no longer covers
the feasible set.  Let `O_q` be the set of all `(cost(S), P_D(S))` pairs
obtained by enumerating every subset of non-owner reports for target `q`.
The global schedule is exact: choose one `O_q` option per target with total
cost at most `B`.  This is a multiple-choice knapsack, solved by a dynamic
program over targets and accumulated bits in which states are pruned only by
componentwise Pareto dominance.  The result is exact for the lexicographic
score `(QoS gap, weighted expected P_D, worst target)` (G8-K).

The system-level objective is worst-target expected `P_D`.  For a threshold
`t`, feasibility asks whether each target can choose a subset with value at
least `t` under the total budget; this is again a multiple-choice knapsack
feasibility problem.  Because feasibility is monotone in `t`, the exact
max-min value is found by binary search over the finite set of per-target
subset values, and the returned schedule is feasible at the optimal
threshold (G8-M).  Both selectors remain exact whenever the per-target
report set is small enough to enumerate.

For larger report sets, G8-S replaces the threshold search by
`m_q(t)`, the minimum bit cost for target `q` to reach `t`.  A schedule with
all values at least `t` exists if and only if `sum_q m_q(t) <= B`, because
the per-target minima are jointly attainable and every feasible subset costs
at least its minimum.  `minimum_cost_to_threshold` computes `m_q(t)` by
branch-and-bound with a closed-form Cauchy upper bound on the
`P_D`-optimal linear-score shift, valid at every operating point; pruning
occurs only when that bound is below `t` or when cost cannot beat the best
known feasible subset.  Small low-`P_D` models delegate to the exact
enumerator instead, so the scaled certificate never changes the objective
value on the audited small models.  The worst case remains exponential, and
the returned bound is reported as a pruning certificate, not a polynomial
guarantee.

### 4.5 Exact majority-count certificate

For the distributed branch, local 1-bit decisions with probabilities
`p0_i, p1_i` are fused by a counting rule.  The exact type-I/II
probabilities are Poisson-binomial tails, so the exact minimum number of
voters `M_min` is the first prefix of the voter sequence whose tail
constraints are feasible.  Feasibility is not monotone in the voter count in
general: the audit exhibits sequences with feasibility trace `[F,F,T,F,T]`.
Consequently `M_min` is computed by exhaustive prefix evaluation, and
binary search over `M` is used only when the per-sequence monotonicity
certificate is available.

## 5. Gates and results

The full rows are in `results/paper_results_table.md`; figures are in
`paper_figures/`.

### 5.1 G3: P_D-optimal linear fusion

On 1318 operating-point addition edges, the P_D-optimal family has zero
decreasing edges, while the deflection-optimal score decreases on 258 edges
(19.6%) with a maximum 16.1pp drop.  The mean gain over the deflection score
is +0.63pp and the maximum per-edge gain is +21.2pp.  In the proportional
regime, 960 checks match the closed form to `1.3e-15`.

### 5.2 G4: expected-P_D selection

At B=20 under correlated erasures, the expected-P_D greedy improves mean
expected P_D by +1.14pp (bootstrap CI [0.47, 1.74], 85% win) and worst-target
P_D by +7.56pp over the proposed selector; the hybrid policy gives +1.44pp
mean and +6.50pp worst-target.  At B=40 the expected-P_D greedy alone is
-0.91pp in mean, so the audited claim uses the hybrid and does not claim
universal dominance.  Submodularity audits found 0 diminishing-return
violations on 3040 edges, and the small-instance greedy ratio is 1.0.

### 5.3 G5: RIS-assisted 6G sensing channel

At B=20, aligned RIS plus expected-P_D greedy gives +12.3pp mean and +17.8pp
worst-target expected P_D over no RIS, with QoS feasibility rising from 0% to
95%; at B=30 the gains are +10.9pp and +15.2pp with 100% QoS feasibility.
With the physics-based channel, 1024 elements and aperture scale `1e-2`
raise mean and worst-target expected P_D by +16.3pp and +24.8pp at B=20.

The joint budget audit (G5-R) charges RIS control bits against the same
total budget.  At total B=40 and a 64-frame coherence block, the best
realizable 3-bit allocation leaves 28 report bits and is within 0.3pp of the
free-continuous upper bound.  The multigrid placement audit (G5-T) selects
`(0, 30, 6)` and improves worst-target expected P_D from 0.882 at the fixed
position to 0.980 (+9.8pp), and from 0.785 with no RIS (+19.5pp), using 34
deployment evaluations.  The adaptive search (G5-V) reaches 0.987 with a
bounded 1.03pp certificate under the used Lipschitz constant; a localized
single-seed search (G5-W) closes to 0.10pp.  With a coordinate-wise
Lipschitz box bound, the 3-seed objective is bounded at 0.16pp over the
original local box and closes to 0.09pp inside a 2 m box in 23 additional
evaluations.

The global resource ledger (G5-RF) shows the gain is not bought with extra
time-bandwidth: the RIS deployment uses 25 report + 12 control bits (4645 TB
symbols) versus 40 report bits (4648 symbols) for no RIS, while raising mean
expected P_D from 0.863 to 0.990 and worst-target from 0.785 to 0.980.

### 5.4 G5-SEN: sensitivity

At total B=40, the mean gain over no RIS rises from +1.3pp to +11.0pp as
aperture scale goes from `1e-3` to `3e-2`, and from +1.1pp to +13.3pp as RIS
elements go from 64 to 1024, even though control overhead lowers the report
budget from 39 to 28 bits.  Longer coherence blocks improve both the report
budget and the gain.  The RIS benefit is regime-specific: with direct-path
blockage `0.001/0.01/0.1/1.0`, mean gains are +10.1/+8.2/+3.5/+2.2pp, and the
worst-target gain CI crosses zero at blockage 1.0.  The paper therefore
claims RIS NLoS illumination for a blocked weak target, not universal
dominance.

### 5.5 G5-SOTA: literature-style baselines

At total B=40 and the G5-T deployment, the proposed chain beats the
strongest soft baseline, RIS plus static deflection Top-K, by +0.68pp mean
and +1.63pp worst-target expected P_D with a positive 12-seed bootstrap CI.
Against no-RIS deflection Top-K, random RIS deflection Top-K, and uniform
one-report soft allocation, the gains are +15.2pp/+27.5pp, +14.4pp/+25.4pp,
and +21.8pp/+46.1pp, respectively.  An exact 1-bit counting baseline, held
to a fusion-level P_FA near 0.008, is improved by +75.7pp/+79.9pp without RIS
and +52.1pp/+64.5pp with RIS.  On the same proposed schedule, the
P_D-optimal fusion family contributes +0.25pp mean and +0.45pp worst-target
over deflection fusion, isolating the fusion improvement from selection.

### 5.6 G6: budget saturation frontier

Without RIS, the worst-target expected P_D saturates near 0.788 and the 0.85
QoS target is not reached at any tested total budget up to 44.  With the
G5-T RIS deployment, total budget 20 (8 report bits after control overhead)
already gives 100% QoS feasibility across all audited seeds.  Discrete
coordinate ascent over add/remove/swap moves from the forward greedy
produces zero additional gain, so the forward greedy is a single-move local
optimum in this scenario and the binding constraint is the sensing
architecture, not the report-selection policy.  This motivates continuous
projected-gradient updates for RIS phase and placement as the next
architecture-level step.

### 5.7 G7: continuous shared-phase RIS

To test the architecture-level direction, one physical RIS phase profile is
parameterized by a single ULA steering cosine and optimized with analytic
array-power gradients.  The worst-array-power surrogate is a negative
result: it improves its own objective but degrades system expected P_D.
System-level grid-plus-refine optimization instead recovers the weak-target
steering direction.  At total budget 20, the shared system-optimized profile
raises worst-target expected P_D from 0.749 (no RIS) to 0.831 (+8.2pp) and
from 0.512 (random shared phase) by +31.9pp, while QoS feasibility reaches
50%.  Per-target ideal phase remains at 100% QoS and +12.3pp higher worst
P_D, so a single shared beam is physically limited; subarray-based
multi-beam RIS is the next design step.

### 5.8 G8: exact quota-constrained selection

Because all audited reports have equal cost, the report-budget constraint is
a cardinality constraint.  The exact selector evaluates every per-target
report subset under the exact reception law, retains the best subset of each
size, and searches all per-target report quotas globally.  In every audited
cell the exact result equals forward greedy to numerical precision, so the
greedy selection layer is already globally optimal for the audited
equal-cost model.  The remaining gap to all-scheduled is architectural:
3.6pp worst at no-RIS B=20 and 0.17pp worst at RIS B=40, versus selection
headroom of zero.

### 5.8.1 G8-K: exact budget selection under heterogeneous costs

The G8 exactness result holds only when every report costs the same number
of bits.  G8-K generalizes it to per-report heterogeneous costs with the
multiple-choice knapsack DP of Section 4.4, keeping the lexicographic
objective `(QoS gap, weighted expected P_D, worst target)`.  On controlled
3-target/4-report models, the DP matches an exhaustive global oracle in 100%
of 100 cells (20 seeds x 5 budgets) and never loses to forward greedy on
the lexicographic score.  On the variable-rate demo system, the exact
schedule is never worse than greedy on that score in all 100 cells.  Mean
worst-target expected P_D gains over greedy are +1.27pp at B=5
(paired one-sided t-test p=0.015; 95% bootstrap CI [0.35, 2.40] pp) and
+2.57pp at B=7 (p=0.009; CI [0.90, 4.65] pp), where greedy leaves cheaper
spare bits unused (exact uses 7.0 of 7 report bits versus greedy 6.6 on
average).  At B=11 the gain is +0.48pp (p=0.115; CI [-0.00, 1.32] pp).  At
B=9 the lexicographic selector is worse on worst-target P_D by -1.00pp
(p=0.895; CI [-2.72, 0.20] pp), and in controlled cells at B=11 the exact
lexicographic worst is 0.3745 versus 0.4081 for greedy.  The lexicographic
objective can therefore trade worst-target P_D for its primary QoS/mean
score, which is why the max-min variant is evaluated separately.

### 5.8.2 G8-M: exact max-min budget selection under heterogeneous costs

G8-M solves the system-level worst-target objective exactly with the
threshold-feasibility DP of Section 4.4.  On the same controlled models it
matches the exhaustive max-min oracle in 100% of 100 cells and is never
worse than greedy; the mean controlled worst-target gains are +5.37pp at
B=5 (p<1e-6), +8.24pp at B=7 (p<1e-6), +0.39pp at B=9 (p=0.083), and
+3.33pp at B=11 (p=3.9e-4).  On the variable-rate demo, the exact max-min
schedule is never worse than greedy in all 100 cells, with gains +1.27pp at
B=5 (p=0.015; CI [0.35, 2.40] pp), +2.57pp at B=7 (p=0.009; CI [0.90, 4.65]
pp), +0.28pp at B=9 (p=0.046; CI [0.00, 0.61] pp), and +1.09pp at B=11
(p=0.014; CI [0.31, 2.03] pp).  The same componentwise-Pareto dominance is
applied before the threshold search, so among schedules attaining the
optimal max-min threshold the returned one is also lexicographically best
in QoS gap, weighted mean, and worst target.

### 5.8.3 G8-S: scaled exact-threshold max-min selection

For a fixed threshold `t`, G8-S evaluates `sum_q m_q(t) <= B` with the
minimum-cost branch-and-bound certificate instead of per-target subset
enumeration.  On the 20-seed controlled set the scaled and exact selectors
agree to zero absolute error at every budget, and on a synthetic 12-report
model the certificate finds the minimum-cost subset (cost 2) without
enumerating all 4096 subsets.  A report-count benchmark covers
R=8/12/16/20/24/28/32/40 non-owner reports: exhaustive subset counts grow
from 256 to about $1.1\times 10^{12}$, while the branch-and-bound
certificate finishes in 24-60 ms and returns the exact minimum cost (1-2
bits) at every scale.  The worst case remains exponential; G8-S is an exact
pruning certificate, not a polynomial-time approximation.

### 5.8.4 G8-Target: exact selection across target count

The exact selectors are proved exact for any target count, but the DP state
set grows with Q.  The target-count audit re-runs the exhaustive-oracle
comparison at Q=3/4/5 and B=8/12/16 (3 seeds, grid 32).  Both selectors
match their exhaustive oracles in 100% of all 27 cells and are never worse
than forward greedy; mean wall time grows from about 180 ms at Q=3 to about
300-360 ms at Q=5, so the exactness certificate remains practical at larger
target counts.

### 5.9 G9: aperture-conserved subarray multi-beam RIS

The 256-element RIS aperture is partitioned into disjoint target-aligned
subarrays and optimized by discrete coordinate ascent over 32/16/8-element
transfers.  Total aperture, element count, and per-element phase bits are
unchanged, so the control overhead is identical to the G5 ledger.  The
optimized allocations are budget-dependent: `(6,85,165)`, `(6,149,101)`,
and `(6,173,77)` elements for budgets 20/28/40.  At total budget 28 the
multi-beam profile reaches 100% QoS feasibility and worst-target expected
P_D 0.913, which is +5.2pp over a single shared weak-aligned beam and
+13.7pp over no RIS.  The per-target ideal upper bound remains 6.7pp higher,
identifying aperture splitting as the remaining physical trade-off.

### 5.10 G10: per-subarray steering-cosine optimization

With the G9 aperture allocations fixed, each subarray steering cosine is
optimized by coordinate ascent over a bounded grid.  Total aperture and
control overhead are unchanged.  The optimized steering rotates the small
first block toward the strong target while keeping the weak block aligned,
improving worst-target expected P_D to 0.858/0.916/0.935 at total budgets
20/28/40, i.e. +0.41/+0.23/+0.14pp over G9.  QoS feasibility remains 50% at
B=20 and 100% at B=28/40; the per-target ideal gap is 9.6/6.5/4.8pp, showing
that aperture allocation is the dominant physical degree of freedom.

### 5.11 G11: fixed-budget RIS aperture scaling

The RIS path power scales as `N^2`, while the control overhead is
`N * phase_bits / coherence_frames`.  Under the exact ledger, increasing
aperture and amortizing phase bits over a longer coherence block closes the
B=20 QoS gap without increasing total budget: `N=1024`, 3-bit phase, and
`C=256` reach 100% QoS with only 8 report bits and worst-target expected P_D
0.982; `N=512` reaches 0.943.  The original `N=256`, `C=64` configuration
remains at 50%.  This confirms the proposed performance is architecture-
limited, not algorithm-limited.

### 5.12 G12: model-driven architecture derivation

The design is not a four-variable exhaustive search.  Under the subarray
approximation `G_q = a_q/N`, the RIS-to-direct power ratio is
`K_q a_q^2 sinc^2(1/2^b)`.  Because local deflection scales quadratically
with evidence SNR, the equal-allocation weak-target surrogate is
`J(N) = beta (1 + kappa N^2)^2 (R - LN)` with `kappa = K_weak
sinc^2(1/2^b)/9`, `L = b/C`, and `R = B_total`.  Its first-order condition
is the quadratic `5 kappa L N^2 - 4 kappa R N + L = 0`, so `N*` has a
closed form.  For B=20, `b=1, C=64` gives `N* = 1016` (rounded 1024) with
100% QoS, and `b=3, C=256` gives `N* = 1363` (rounded 1344) with worst P_D
0.974.  The exact system validates the derived operating point without
high-dimensional search.

### 5.13 G13: max-min deflection water-filling

The subarray allocation is derived from the monotone convex surrogate
`D_q(a_q) = beta_q (1 + kappa_q a_q^2)^2`.  Aperture is moved from the
current highest-D target to the lowest-D target until the minimum stops
improving.  A marginal-equalizing KKT variant was tested first and rejected
because it solves the wrong max-min condition.  With the correct
water-filling, exact worst-target P_D improves from 0.900 to 0.911
(`N=1024,b=1,C=64,B=20`), from 0.974 to 0.992
(`N=1344,b=3,C=256,B=20`), and from 0.999599 to 0.999995
(`N=2048,b=3,C=256,B=40`), all with 100% QoS.

### 5.14 G14: exact-array-factor allocation

The surrogate is upgraded to
`D_q(a) = beta_q (1 + K0_q N^2 G_q(a))^2`, where `G_q(a)` is the exact
squared array factor including cross-block interference and phase
quantization.  Max-min water-filling on this exact surrogate raises the
surrogate minimum in every tested configuration, but exact system P_D does
not consistently improve: at `N=1024,b=1,C=64,B=20` the exact allocation is
0.8pp worse than the separable allocation, while at
`N=2048,b=3,C=256,B=40` it is 0.00008pp better.  This is recorded as a
negative/equivocal result: a more accurate surrogate is necessary but not
sufficient for system-level optimality.

### 5.15 G15: greedy-aware system-level allocation

The allocation objective is upgraded to the exact system function
`F(a) = mean_seed min_q E_PD(q, S_q(a))`, where `S_q(a)` is the
expected-P_D greedy schedule under the allocation.  Coordinate ascent over
single-block aperture transfers accepts only moves that increase `F`, so the
stopping point is a local optimum of the true objective.  Exact validation
improves worst P_D from 0.911 to 0.924 (`N=1024,b=1,C=64`), 0.911 to 0.927
(`N=704,b=3,C=128`), and 0.981 to 0.985 (`N=960,b=3,C=128,B=28`).  At
`N=2048,b=3,C=256,B=40`, the coarse 8-element search ends 0.0018pp below the
exact-surrogate allocation, which is reported as a local-search limitation.

### 5.16 G16: single-element refinement and local certificate

Each G15 allocation is refined by 4/2/1-element coordinate ascent, and the
final allocation is certified by evaluating every one-element transfer of
the exact system objective.  All five configurations improve and satisfy
`local_optimal=true` with nonpositive maximum gradient.  The final values
are 0.924107 (`N=1024`), 0.927345 (`N=704`), 0.991896 (`N=1344`), 0.985738
(`N=960,B=28`), and 0.999986 (`N=2048,B=40`).  This provides the first
system-level local optimality certificate for the aperture allocation.

### 5.17 G17: bounded multi-block certificate

Every zero-sum reallocation moving at most three elements in total is
evaluated exactly on the system objective and iterated to the best point in
that neighborhood.  Four configurations are already multi-block local
optima, while `N=2048,B=40` improves from 0.999986 to 0.999988 in seven
rounds.  All five final allocations satisfy `local_optimal=true` with
respect to the `T<=3` neighborhood, extending the single-move certificate to
simultaneous multi-block moves.

### 5.18 G18: joint RIS placement and allocation

The exact system objective `F(s,a)` is optimized by alternating coordinate
ascent: allocation uses the T<=3 multi-block certificate and position uses
2/1/0.5-meter steps.  All three tested configurations improve and certify
both degrees of freedom: `N=1024,B=20` reaches 0.925224 at `(-2,30,6)`,
`N=1344,B=20` reaches 0.992907 at `(0.5,31,6)`, and `N=2048,B=40` reaches
0.999997 at `(6.5,34,5)`, all with `local_optimal=true` for allocation and
position.  The architecture is white-box: it uses exact system evaluations,
finite move-set gradients, and local certificates rather than a trained
surrogate; the full theory and explicit-information inventory are in
`G18_THEORY.md`.

### 5.19 G19: progressive decentralization

Decentralization is opened in stages: fair local scheduling, deflection
fusion, owner-only decisions, and 1-bit hard decisions with counting fusion.
At B=40/N=2048, local scheduling loses only 0.0013pp worst, deflection fusion
0.0026pp, owner-only 0.014pp, while 1-bit hard decisions lose 18.8pp and QoS
drops to 50%.  At B=20 with a 4-bit report budget, 5-bit soft reports are
infeasible, so centralized soft fusion equals owner-only; 1-bit hard
decisions can send three reports but cannot meet the global P_FA=0.05 with
one vote per target, so they are infeasible/worse (QoS 0%).  The earlier
+6.0pp claim was corrected after enforcing the global P_FA constraint.

### 5.20 G20: amplified distributed hard detection

The distributed branch is upgraded from a fixed baseline to a designed
detector: local 1-bit thresholds and the counting threshold are optimized per
target under the global P_FA constraint.  At B=40/N=2048, optimized 1-bit
detection raises worst P_D from 0.812 (fixed local P_FA) to 0.944 and QoS
from 0% to 100%, while centralized soft fusion remains at 0.999997.  At
B=20 with one vote per target, no counting rule meets global P_FA, so the
distributed branch is reported as infeasible rather than over-claimed.

### 5.21 G21: network-level decentralization

Report links and owner fusion are removed entirely.  Every UAV makes a local
1-bit decision and the target is declared by an optimized majority threshold
over all M=8 UAVs.  At B=20/N=1024, peer majority reaches worst P_D 0.955
(centralized soft 0.925); at B=20/N=1344 it reaches 0.998 (centralized
0.993); at B=40/N=2048 it reaches 0.9999977 (centralized 0.9999967), all
with 100% QoS and zero report bits.  This is the strongest distributed
result: with high local SNR, consensus voting can match or exceed centralized
soft fusion.

### 5.22 G22: degraded multi-hop consensus

Partial observability and per-hop erasure enter through the effective
participation `obs * (1 - (1 - r)^hops)`.  At B=40/N=2048, observability 0.75
gives worst P_D 0.966 and link reliability 0.8 gives 0.977, both below
centralized 0.999997; three hops at 0.8 recover to 0.9998, while severe
degradation drops to 0.877.  The distributed advantage is therefore
conditional on network quality.

### 5.23 G23: correlated failure and heterogeneous observability

Effective participation becomes
`obs_i * (1 - p_c) * (1 - (1 - r)^hops)`, where `p_c` is a network-wide
common failure and `obs_i` is geometry-derived per-UAV observability.  At
B=40/N=2048, common failure 0.2/0.4 gives 0.977/0.909, heterogeneous
observability gives 0.936, and the severe combination gives 0.858, all below
centralized 0.999997.  Correlated outages remove the distributed advantage
faster than independent degradation.

### 5.24 G24: scalability across target and UAV counts

For Q in {2,4,6} and M/Q in {1,2,3} with report budget `20*Q`, RIS ideal
phase is the most robust architecture and reaches 100% QoS in every tested
cell except Q=6,M=6.  Peer majority needs enough UAVs per target: it reaches
100% QoS at M/Q=3 for Q=2 and Q=6, while no-RIS is topology-sensitive and
drops to 0.460 at Q=2,M=4.  Consensus voting can compensate for losing soft
information only when the UAV-per-target ratio is sufficiently large.

### 5.25 G25: scaled white-box G18

For Q>3 the exhaustive multi-block certificate is replaced by the derived
max-min water-filling allocation and exact position ascent, keeping the same
exact system objective.  The scaled G18 keeps 100% QoS in every tested cell
except Q=6,M=6, where the ideal-phase upper bound also fails.  At
Q=6,M=12 it reaches worst P_D 0.922 versus 0.792 for peer majority and 0.934
for ideal phase; at Q=4,M=8 it reaches 0.964 versus 0.915 for peer majority.

### 5.26 G26: mobility and time-varying blockage

UAVs rotate along smooth trajectories, targets move on bounded paths, and
the weak-target direct-path blockage varies sinusoidally over frames.
Worst-over-time no-RIS QoS is 0%; RIS ideal remains 100%; static subarray
reaches 68.75%; adaptive subarray recomputed each frame reaches 81.25% with
worst P_D 0.847 versus 0.841 for static.  Adaptive allocation therefore
improves robustness under time-varying geometry.

### 5.27 G27: multi-RIS deployment

With total aperture fixed at 256 elements, one RIS reaches worst P_D
0.955/0.980/0.983 at B=20/28/40.  Splitting into two or three RISs lowers
performance (0.923/0.927 at B=28), because non-coherent power addition loses
the `N^2` coherent aperture; placement diversity only partially compensates.

### 5.28 G28: multi-RIS split and placement optimization

For a single target the reflected-power sum `sum N_r^2 / L_r` is convex, so
its maximum over a fixed total aperture occurs at an extreme split;
multi-RIS can only help through multi-target geometry differences.  Local
optimization on the exact system objective finds `(8,248)` elements with the
second RIS at `(4,42,2)`, reaching worst P_D 0.986 and slightly exceeding the
single-RIS 0.981, while the equal split gives only 0.924.

### 5.29 G29: variable-rate soft/hard reporting

`build_models` now supports per-UAV quantizer bits, making report cost and
quantization fidelity jointly variable.  Fixed 5-bit soft reporting is best
at B=20/28 (0.953/0.977), while the adaptive soft-rate profile outperforms
it at B=40 (0.988 versus 0.981).  Optimized 1-bit hard decisions remain the
weakest soft-information policy in this regime.

### 5.30 G30: global rate-profile optimization

Per-UAV quantizer bits are optimized by exact-system coordinate ascent.  At
B=28, the optimized profile reaches worst P_D 0.988 versus 0.981 for fixed
5-bit and 0.974 for the heuristic adaptive profile.  At B=40 it reaches 0.991
versus 0.987 for both fixed and heuristic adaptive profiles, with
single-rate-change local optimality certified.

### 5.30.1 G30-E: exact-objective rate certificate

The G30 certificate is stated for the greedy schedule.  Under heterogeneous
report costs the greedy schedule is not exact, so G30-E re-checks every
single-UAV quantizer-bit change with the exact max-min selector G8-M on the
same 2-seed/grid-256 audit as G30.  At B=28 the G30 profile remains a
single-rate exact local optimum at worst P_D 0.9879 with zero gain; the
greedy certificate is not false under the exact objective.  At B=40 the
greedy certificate is false under the exact objective: the G30 profile
reaches 0.9911 with the exact selector and has improving single-rate
changes.  Exact coordinate ascent finds a profile with worst P_D 0.9916
(+0.04pp) that is a single-rate exact local optimum.  The G30-E result is
therefore both a validation (B=28) and a correction (B=40) of the greedy
certificate, and it is reported with the exact objective rather than as an
extension of the greedy claim.

### 5.31 G31: exact soft/hard hybrid fusion

The combined score is a Gaussian soft score plus exact post-BSC hard
log-likelihood terms.  At B=28/40, hybrid fusion reaches 0.977/0.969 versus
0.977/0.981 for pure 5-bit soft and 0.843/0.736 for hard-only.  Hybrid fusion
is exact and P_FA-constrained, but it is not automatically better than
soft-only with the fixed schedule.

### 5.32 G32: interference sensitivity

Per-UAV INR is injected as `SINR = SNR / (1 + INR)`.  At INR=0 dB only RIS
ideal reaches 100% QoS; INR=3 dB makes all architectures fail; INR=10/20 dB
drives worst P_D below 0.2/0.06.

### 5.33 G33: spatial interference and RIS placement

Per-UAV INR follows free-space path loss from a fixed source.  No-RIS fails
all strengths, fixed RIS keeps 100% QoS, and optimizing RIS position adds
about +0.3pp worst P_D, showing that placement remains useful under spatial
interference.

### 5.34 G34: multiple interference sources

Per-UAV INR becomes the sum of path losses from three sources.  Mean INR
0.087 leaves no-RIS at 0.810 with 0% QoS; fixed RIS reaches 0.983 and
optimized placement 0.987, both with 100% QoS.  Direct multi-source
interference is modeled; RIS null-steering is not yet implemented.

### 5.35 G35: 1-D ULA versus 2-D UPA

With the same 256 elements, UPA is nearly identical to ULA in clean and
spatial interference scenarios.  The 2-D aperture does not add P_D in the
current geometry; its value would require elevation diversity or
null-steering.

### 5.36 G36: UPA null-steering

UPA phases are optimized to maximize target array gain while suppressing
interference directions.  Reflected INR falls from 0.0267 to 0.0106 (-60%)
while target gain drops from 1.000 to 0.984.  At B=40, worst P_D improves
from 0.98112 to 0.98216, showing that phase-domain interference suppression
is effective in the 2-D aperture.

### 5.37 G37: directly quantized null-steering

Discrete coordinate ascent over the `2^b` phase levels directly optimizes
the scalarized array power.  Reflected INR reaches 0.01052 and B=40 worst
P_D reaches 0.982166, slightly better than the continuous-then-quantized
approach, showing that quantization-aware design is worthwhile.

### 5.38 G38: joint quantized nulling and placement

Each candidate position redesigns the quantized null-steering phases.  B=40
worst P_D improves from 0.98217 (fixed) to 0.98481 (optimized), while
reflected INR rises from 0.0105 to 0.0296, an explicit target-gain versus
reflected-interference trade-off.

### 5.39 G39: distributed features under relaxed thresholds

With budgets 20/24/28 and QoS targets 0.70/0.75/0.80, centralized soft,
peer clean, peer multi-hop, and optimized hard decisions are all feasible.
Peer multi-hop stays at worst P_D 0.953 and optimized hard at 0.84-0.86,
showing that relaxing the threshold makes distributed viable, while actual
P_D remains dominated by the RIS/channel.

### 5.40 G40: low-budget/low-SNR distributed

With N=128 and spatial interference, B=12 makes centralized soft fusion drop
to 0.786, while peer clean 0.858 and peer multi-hop 0.855 outperform it;
optimized hard 0.765 remains feasible at QoS 0.70.  This shows that
distributed consensus wins when report bits are scarce.

### 5.41 G41: consensus parity boundary

The Gaussian approximation gives `M_min` around 14-17.  Empirically,
consensus wins at B=8 for M>=6 and at B=12 for M>=12, while centralized
regains the lead at B>=16.  The exact threshold optimization shifts the
empirical boundary below the fixed-local-P_FA formula.

### 5.42 G42: optimized-local-threshold boundary

Minimizing `M_min` over the local P_FA grid lowers the theoretical bound by
9-13%.  For M=16, the bound falls from 13.70 to 12.14, closer to the exact
wins observed at B=8/12.

### 5.43 G43: exact Poisson-binomial boundary

Exact majority feasibility uses Poisson-binomial tails and starts at M=6,
matching empirical wins, while the Gaussian approximation predicted
M_min=13.36.  Exact enumeration therefore closes the theory-empirics gap.

### 5.43.1 G43-B: exact minimum majority count and monotonicity audit

G43-B evaluates the exact Poisson-binomial feasibility on every prefix of
the voter sequence, so `M_min` is the first feasible prefix rather than a
coarse-grid estimate.  The audit checks monotonicity explicitly.  A
homogeneous example with `p0=0.1`, `p1=0.7`, `alpha=0.05`, `beta=0.7` has
feasibility trace `[F,F,T,F,T]`, so M=3 is feasible and M=4 is infeasible;
binary search over `M` is therefore not valid in general without a
per-sequence monotonicity certificate.  In the audited RIS-assisted system,
the exact minimum voter count is 14 at M=6, 17 at M=8, 16 at M=12, and 19 at
M=16, and the feasibility trace is non-monotone at M=8/12/16.  The gate
therefore reports the exact prefix result and the monotonicity audit
together; no monotonicity-based search is claimed.

### 5.44 G44: fundamental information budget

All methods are placed on the normalized information coordinate
`rho = J/D_full`.  Soft fusion P_D rises monotonically from 0.774 at
rho=0.507 to 0.933 at rho=0.946; consensus retains nonzero rho when soft
reports are unaffordable, and its optimized majority extracts more P_D per
KL unit than fixed-threshold hard fusion.  The unified framework is
documented in
`FUNDAMENTAL_PRINCIPLE.md`.

### 5.45 G45: closed-form resource law (negative)

A simple closed-form law
`P_D = Phi((sqrt(d0(1+n)g^2)-z)/sqrt(c))` is tested across N and B.  It
overestimates P_D by up to 30pp and saturates to 1 at N>=128.  The law is
rejected because quantization loss, correlation, and non-proportional H1
variance break its assumptions; exact moment propagation is required.

### 5.46 G46: exact information budget

The raw deflection/KL coordinate of G44 is not itself a performance law.
G46 therefore inverts the Gaussian relation under `Sigma1 = c Sigma0` with
`c=1`:

`D_eff = (Phi^{-1}(P_D) + z_FA)^2`,

and defines the exact normalized information budget
`rho_exact = D_eff / D_full`.  Every method is placed on this
P_D-consistent coordinate.  Under the same N=128, interference, 4-seed
audit, soft raw `rho` overestimates `rho_exact` by a factor 2.38-2.78;
soft P_D rises from 0.774 at `rho_exact=0.205` to 0.933 at
`rho_exact=0.351`, peer consensus has P_D 0.881 at `rho_exact=0.284`, and
optimized hard fusion has `rho_exact<=0.199`.  Hard and soft report budgets
are enforced consistently, so the peer advantage is confined to the
scarce-report regime where consensus spends zero report bits.

### 5.47 G47: centralized/distributed architecture switch

G46 identifies the regime where peer consensus has the larger exact
information coordinate.  G47 turns that comparison into a detector: for each
seed, compute the exact worst-target P_D of centralized soft fusion and peer
majority, and select the branch with the higher value.  Both branches are
calibrated to the same global false-alarm rate, so the switch is feasible by
construction.  At B=8/12, peer is selected and raises worst P_D from
0.774/0.824 to 0.881 (+10.68/+5.68pp), making the 0.85 QoS target feasible;
at B>=16 the exact switch returns to centralized soft.  A fixed
`report_budget < 10` policy reproduces the exact choices in this audit and
is reported as a design parameter, not as a universal crossover law.

### 5.48 G48: target-wise architecture switch

The global mode choice of G47 is refined to per-target selection.  For each
target `q`, the policy evaluates `soft_q` and `peer_q` and selects

`P_D,q = max(soft_q, peer_q)`.

The worst-target value is therefore
`min_q max(soft_q, peer_q)`, which satisfies

`min_q max(a_q,b_q) >= max(min_q a_q, min_q b_q)`,

so target-wise switching is never worse than the global switch.  In the
4-seed audit it adds +0.49pp at B=12 and +1.55pp at B=16/20, reaches 0.881
at B=8, and keeps the 0.85 QoS target feasible.  The peer-selection rate
falls from 92% at B=8 to 50% at B=16 and 25% at B=40, showing a gradual
per-target transition rather than a global cutoff.

### 5.49 G49: soft-report reallocation

When a target switches to peer consensus, its previously scheduled soft
reports are no longer transmitted.  G49 adds those freed bits back to the
remaining centralized targets with an exact expected-P_D greedy over report
costs.  The update is additive: no centralized schedule is shrunk, every
added report keeps the total used bits within the report budget, and every
per-target centralized P_D is therefore nondecreasing.  Consequently the
target-wise worst P_D cannot decrease.  In the 4-seed audit, reallocation
adds +0.75pp at B=16/20 over G48 (worst P_D 0.9250) and +1.55/+0.85pp at
B=28/40; at B=8 no soft report fits and peer consensus keeps the 0.85 QoS
target feasible.

### 5.50 G50: two-sided mode ascent

Reallocation alone never lets a peer target return to centralized soft.
G50 adds that second direction with a communication-efficient acceptance
rule: a peer target is considered only when it currently attains the worst
P_D, and the upgrade is accepted only if its upgraded soft P_D strictly
raises that worst value.  Failed attempts are discarded, so no report bit is
wasted on a non-improving switch.  In the 4-seed audit this adds +0.39pp at
B=12 over G48 (0.8858 -> 0.8898) with 3.75 report bits used on average; at
B=16-40 the ascent matches G49 and switches only when it improves the worst
target.

### 5.51 G51: stochastic mobility with RIS reconfiguration latency

G26 used deterministic trajectories.  G51 replaces them with an AR(1)
random model: UAV and target positions are perturbed with temporal
correlation, and the weak-target blockage is random over frames.  RIS phases
are evaluated as frozen, designed one frame earlier (latency-1), or ideal
per frame, and the target-wise switch and mode ascent run on every frame.
Under N=128 and total B=40, no-RIS worst-over-time P_D is 0.524; static RIS
mode ascent reaches 0.705, latency-1 RIS 0.722, ideal target-wise 0.847,
and ideal mode ascent 0.852.  Ideal mode ascent raises QoS over time from
81.25% to 90.625%, and latency-1 beats static by +1.64pp worst P_D.

### 5.52 G52: MMSE prediction-aware RIS

Latency-1 RIS uses the previous frame's true phase.  Under the AR(1)
trajectory, the conditional mean of the current target position is

`hat p_t = n_t + rho (p_{t-1} - n_{t-1})`,

where `n_t` is the deterministic nominal position.  For Gaussian AR(1)
innovations this predictor minimizes the mean squared position error, and
the RIS phase is designed from `hat p_t`.  In the 4-seed, 8-frame audit,
MMSE prediction improves latency-1 worst-over-time P_D from 0.7217 to
0.7283 (+0.65pp) and QoS from 43.75% to 46.875%; ideal per-frame phase
remains the upper bound.

### 5.53 G53: multi-step MMSE prediction

For a reconfiguration latency `h`, the AR(1) conditional-mean predictor is

`hat p_{t|t-h} = n_t + rho^h (p_{t-h} - n_{t-h})`,

and the prediction-error covariance is `(1 - rho^{2h}) sigma^2 I`.  In the
4-seed, 8-frame audit, MMSE over stale-phase worst P_D gains
+0.65/+3.24/+5.24pp for h=1/2/3, matching the qualitative growth of the
prediction-error covariance scale.  An exact per-frame selection over the
stale/MMSE designs raises the best fixed MMSE worst P_D from 0.7283 to
0.7369 (+0.86pp) and QoS from 46.875% to 53.125%; this is an oracle over six
feasible phase designs, not a joint exhaustive phase search.

To make the architecture adjustment practical, a hysteresis rule switches
only when the best candidate beats the incumbent architecture's current
frame value by more than `delta`.  For `delta=0.02`, QoS stays at 53.125%,
mean switches drop from 4.50 to 2.25 per seed, and the worst P_D loss versus
the oracle is 0.00104, below the `delta` bound.

When each architecture switch consumes control bits, the same hysteresis
family gives a cost-aware Pareto frontier.  Under a 6-bit control budget,
per-switch costs of 1/3/6 bits select `delta = 0.00/0.03/0.05` with worst
P_D `0.7369/0.7250/0.7217` and `4.50/1.50/0.75` switches per seed.

### 5.54 G54: covariance-aware phase (negative)

A covariance-aware phase maximizes the expected squared array gain under
the Gaussian AR(1) direction error, and the projected-gradient ascent is
monotone in that surrogate.  However, after 3-bit quantization and the exact
expected-P_D/mode-ascent chain at h=3, the robust phase degrades
worst-over-time P_D from 0.7200 (MMSE) to 0.6557 (-6.43pp) and QoS from
43.75% to 37.5%.  The surrogate is therefore not a system-level design
criterion, and MMSE phase is kept.  This is a negative result, not a
performance claim.

## 6. Discussion and limitations

- The moment-matched Gaussian score is a tractable detection model, not an
  exact discrete likelihood-ratio detector.
- The RIS channel is a controlled path-loss model; per-element mutual
  coupling, polarization, and waveform-level RIS responses are not modeled.
- The 1-symbol-per-bit ledger is conservative but not waveform-derived;
  sensing OTFS-grid scaling under fixed total time-bandwidth remains open.
- The formal KKT guarantee holds at `P_D > 0.5`; below that point the
  implementation does not claim global optimality.
- The empirical Lipschitz constant is valid for the certificate at the used
  constant, not a proven global constant for the seed-averaged greedy
  objective.
- The SOTA comparison is a 12-seed draft with re-implemented baseline
  components under identical channel and budget assumptions; external
  numerical results from other systems remain to be matched.
- Closure of the 3-seed certificate over the original local box is the
  remaining experimental gap.
- The G6 result is a negative result for single-report discrete ascent; a
  continuous relaxation of bit allocation and RIS phase/placement remains
  future work.
- The G7 result bounds what one shared ULA beam can do; multi-subarray and
  multi-RIS architectures are not yet implemented.
- The G8 equal-cost exactness result is closed; heterogeneous costs are
  handled exactly by G8-K/G8-M through subset enumeration plus a
  multiple-choice knapsack DP.  The per-target subset enumeration remains
  exponential, and G8-S provides an exact pruning certificate only in the
  regime where it can delegate small models to the exact enumerator.  The
  lexicographic G8-K objective can lower worst-target P_D relative to greedy
  in controlled cells; G8-M is the correct selector for the max-min
  objective and never loses to greedy in the audited cells.  The
  20-seed audit shows that G8-M's system worst-target gains are significant
  at B=5/7/9/11 (paired one-sided t-test p<0.05 with bootstrap CI excluding
  zero), while G8-K's lexicographic guarantee is kept separate from the
  max-min claim because its worst-target gain is negative at B=9.
- The G9 subarray search is a local integer coordinate ascent over contiguous
  1-D blocks; interleaved subarrays, 2-D apertures, and joint placement
  remain open.
- The G10 steering search is a local grid coordinate ascent without a global
  optimality certificate.
- The G11 aperture scaling uses fixed allocations and does not jointly
  optimize `(N, phase_bits, coherence, allocation)`.
- The G12 derivation uses equal allocation and ignores cross-block
  interference in the first-order condition.
- The G13 water-filling surrogate ignores cross-block interference and is
  validated by exact evaluation only at the tested configurations.
- The G14 result shows that surrogate exactness does not align perfectly
  with the greedy expected-P_D objective.
- The G15 result is a system-level local optimum, not a global certificate.
- The G16 certificate covers single-element transfers only.
- The G17 certificate is bounded to `T<=3` and excludes joint placement.
- The G18 joint certificate is local and uses 0.5m position granularity.
- The G19 ablation uses fixed report-bit lengths and a common fusion point
  rather than peer-to-peer consensus.
- The G20 optimization scans one-dimensional local P_FA and keeps fixed vote
  schedules.
- The G21 peer majority assumes full local observability and no report-link
  erasure.
- The G22 model assumes independent per-hop erasure and scalar
  observability.
- The G23 common-failure model is whole-network; regional groups and
  time-varying topology are not included.
- The G24 RIS branch uses per-target ideal phase and an empirical minimum
  M/Q ratio.
- The G25 scaled architecture does not claim the allocation certificate for
  Q>3.
- The G26 mobility model is deterministic; stochastic trajectories remain
  open.
- The G27 multi-RIS model is non-coherent and uses fixed RIS positions.
- The G28 split/placement result is a local optimum without a global
  certificate.
- The G29 adaptive rate profile is per-target equal budget, not a global
  knapsack optimization.
- The G30 optimization uses one rate per UAV and no hybrid soft/hard fusion.
  G30-E closes the greedy-certificate gap with the exact objective at
  B=28/40, but the exact local certificate is stated only for those two
  budgets and for single-rate changes.
- The G31 hybrid schedule is fixed heuristically.
- The G32 INR is uniform across UAVs and no suppression is modeled.
- The G33 model has one interference source and no RIS null-steering.
- The G34 model sums direct interference but does not null-steer.
- The G35 UPA model has no elevation separation in the target set.
- The G36 null-steering is per-target continuous-phase optimization.
- The G37 discrete search is a local coordinate ascent without a global
  certificate.
- The G38 joint search is local and still uses per-target phase vectors.
- The G39 relaxation does not test very low budget/SNR regimes.
- The G40 low-budget regime uses one interference source.
- The G41 theoretical formula uses fixed local P_FA.
- The G42 formula still relies on the Gaussian approximation.
- The G43 exact check is evaluated on a discrete M grid; G43-B removes the
  grid error by exact prefix evaluation, but the minimum voter count is
  reported only for the audited RIS-assisted sequence and is not monotone in
  M in general.
- The G44 information measure is deflection/KL based, not full mutual
  information.
- The G45 negative result justifies the exact moment-propagation model.
- The G46 `rho_exact` coordinate is exact only under the calibrated
  Gaussian inversion; it is a diagnostic budget, not a closed-form predictor
  for unseen quantization/correlation profiles.
- The G47 switch is exact only for the two audited architectures and one
  operating profile; joint rate/threshold optimization and re-estimation of
  the fixed threshold remain open.
- The G48 target-wise switch keeps the centralized soft schedule fixed;
  G49 adds a greedy reallocation of the freed bits, and joint
  schedule/mode optimality remains open.
- The G49 reallocation is a monotone greedy ascent, not a joint
  schedule/mode optimum; G50 covers limiting-target two-sided updates, and
  joint optimality remains open.
- The G50 ascent is monotone but greedy; a global certificate over all
  report additions and limiting-target switches remains open.
- The G51 AR(1) frame model is a declared stochastic abstraction, not a
  continuous-time SDR trajectory or a prediction-aware control law.
- The G52/G53 predictors assume the true AR(1) correlation and use
  conditional-mean-only design; model mismatch and covariance-aware phase
  shaping remain open.
- The G53 horizon sweep assumes known correlation and does not jointly
  optimize the reconfiguration horizon with the report/control budget.
- The G54 expected-gain surrogate is rejected after quantization-aware exact
  evaluation; a quantization-aware robust design remains open.

## 7. Conclusion

The paper provides an audited end-to-end chain for selective
soft-information fusion in RIS-assisted UAV-OTFS-ISAC: communication loss
enters the evidence moments, P_D-optimal monotone fusion is derived from a
KKT family, selection is an expected-P_D greedy with bounded-regime
submodularity, exact heterogeneous-cost budget/max-min certificates close
the greedy gap, exact Poisson-binomial majority counting replaces the
Gaussian boundary, and the RIS control/report/placement degrees of freedom
are optimized under one resource identity.  The numerical gates confirm the
tight-budget value of the method, include multi-seed paired comparisons and
formal exactness audits, and make the boundaries of each claim explicit.

## Appendix A. Formal proofs

All theorems and proofs are in `FORMAL_PROOFS.md`, including the KKT
representation, set monotonicity, closed-form proportional regime,
expectation monotonicity, concavity/submodularity lemmas, quantization gain
loss, path-loss monotonicity, grid-search bound, and branch-and-bound
certificate, the exact Poisson-binomial counting baseline used by the SOTA
comparison, the effective-deflection inversion, and the two-branch
architecture-switch lemma plus its target-wise order inequality and the
additive soft-report reallocation monotonicity certificate and the
limiting-target mode-ascent acceptance rule and the worst-over-time
monotonicity of per-frame mode ascent and the AR(1) conditional-mean
prediction lemma and its h-step covariance generalization and the
expected-gain surrogate negative result.

## Draft figures and tables

- Table: `results/paper_results_table.md` (2514 rows, regenerated by
  `scripts/build_paper_tables.py`).
- Figure: `paper_figures/g4_pd_vs_budget.png`.
- Figure: `paper_figures/g5_ris_pd_vs_budget.png`.
- Figure: `paper_figures/g5_phase_resolution_gain.png`.
- Figure: `paper_figures/g5_deployment_ci_forest.png`.
- Figure: `paper_figures/g5_resource_ledger.png`.
- Figure: `paper_figures/g5_sensitivity.png`.
- Figure: `paper_figures/g5_sota_baselines.png`.
- Figure: `paper_figures/g6_budget_saturation.png`.
- Figure: `paper_figures/g7_shared_phase.png`.
- Figure: `paper_figures/g8_exact_quota.png`.
- Figure: `paper_figures/g8k_exact_budget.png`.
- Figure: `paper_figures/g8m_exact_maxmin.png`.
- Figure: `paper_figures/g8s_scaled_maxmin.png`.
- Figure: `paper_figures/g8s_scalability_benchmark.png`.
- Figure: `paper_figures/g8_target_scalability.png`.
- Figure: `paper_figures/g9_subarray_multibeam.png`.
- Figure: `paper_figures/g10_subarray_steering.png`.
- Figure: `paper_figures/g11_aperture_scaling.png`.
- Figure: `paper_figures/g12_derived_architecture.png`.
- Figure: `paper_figures/g13_waterfilling_architecture.png`.
- Figure: `paper_figures/g14_exact_allocation.png`.
- Figure: `paper_figures/g15_system_allocation.png`.
- Figure: `paper_figures/g16_single_move_certificate.png`.
- Figure: `paper_figures/g17_multi_move_certificate.png`.
- Figure: `paper_figures/g18_joint_placement_allocation.png`.
- Figure: `paper_figures/g19_progressive_decentralization.png`.
- Figure: `paper_figures/g20_amplified_distributed.png`.
- Figure: `paper_figures/g21_network_decentralization.png`.
- Figure: `paper_figures/g22_degraded_consensus.png`.
- Figure: `paper_figures/g23_correlated_consensus.png`.
- Figure: `paper_figures/g24_scalability_comparison.png`.
- Figure: `paper_figures/g25_scaled_g18_scalability.png`.
- Figure: `paper_figures/g26_mobility_blockage.png`.
- Figure: `paper_figures/g27_multi_ris.png`.
- Figure: `paper_figures/g28_multi_ris_split.png`.
- Figure: `paper_figures/g29_variable_rate.png`.
- Figure: `paper_figures/g30_global_rate.png`.
- Figure: `paper_figures/g30e_exact_rate_certificate.png`.
- Figure: `paper_figures/g31_hybrid_fusion.png`.
- Figure: `paper_figures/g32_interference_sensitivity.png`.
- Figure: `paper_figures/g33_spatial_interference.png`.
- Figure: `paper_figures/g34_multi_interference.png`.
- Figure: `paper_figures/g35_ula_vs_upd.png`.
- Figure: `paper_figures/g36_null_steering.png`.
- Figure: `paper_figures/g37_quantized_null_steering.png`.
- Figure: `paper_figures/g38_joint_null_placement.png`.
- Figure: `paper_figures/g39_distributed_relaxation.png`.
- Figure: `paper_figures/g40_low_budget_snr_distributed.png`.
- Figure: `paper_figures/g41_consensus_parity.png`.
- Figure: `paper_figures/g42_optimized_parity.png`.
- Figure: `paper_figures/g43_exact_parity.png`.
- Figure: `paper_figures/g43b_exact_min_majority.png`.
- Figure: `paper_figures/g44_fundamental_information.png`.
- Figure: `paper_figures/g45_resource_information_law.png`.
- Figure: `paper_figures/g46_exact_information_budget.png`.
- Figure: `paper_figures/g47_architecture_switch.png`.
- Figure: `paper_figures/g48_target_wise_switch.png`.
- Figure: `paper_figures/g49_soft_reallocation.png`.
- Figure: `paper_figures/g50_mode_ascent.png`.
- Figure: `paper_figures/g51_stochastic_mobility.png`.
- Figure: `paper_figures/g52_prediction_aware_ris.png`.
- Figure: `paper_figures/g53_multi_step_prediction.png`.
- Figure: `paper_figures/g54_covariance_aware_negative.png`.
