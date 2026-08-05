# UAV-OTFS-ISAC selective fusion simulator

This repository implements the system model in
`UAV_OTFS_ISAC_论证与系统模型_revised_final.docx` as a reproducible Python
simulation. The current release focuses on the paper's minimum publishable
scope: fixed geometry/waveform parameters, correlated local evidence,
multi-bit reporting over error-prone links, detectable erasures, target-wise
report selection, and linear deflection-optimal fusion.

## Implemented model chain

1. Geometry and fractional-Doppler-dependent OTFS evidence moments.
2. Geometry-dependent air-to-air reporting reliability, low-bit scalar
   quantization, and exact binary symmetric channel transition.
3. Positive-definite cross-UAV covariance with shrinkage regularization.
4. Detectable packet erasures represented by a random received set.
5. Exact expected deflection by reception-pattern enumeration, with an SAA
   fallback for larger sets.
6. Schur-complement conditional marginal deflection.
7. Two-stage greedy scheduling: minimize normalized QoS shortfall, then improve
   expected total deflection within the bit budget.
8. Fusion weights recomputed from the actually received set.
9. Monte Carlo `P_D/P_FA`, weak-target performance, bits, and selection trace.
10. Exact reception-pattern loss distributions, upper-tail CVaR, and QoS
    violation probability.
11. Exact target-level evidence-set enumeration plus multiple-choice knapsack
    dynamic programming for mean-CVaR portfolio design.
12. Cross-resource-domain replication of critical reports with exact effective
    erasure distributions and equal-bit chance-constrained optimization.

## Novelty positioning

The contribution is the integrated scenario and its end-to-end validation
chain, not a new family of selection algorithms.  The conditional
marginal-deflection greedy is an adaptation of established deflection-optimal
linear fusion and greedy subset selection to communication-corrupted,
correlated OTFS evidence; it is explicitly not claimed as a new algorithm or
as universally better than Top-K.  What is new is the scenario: toy MF/CFAR
front end -> per-path Fisher covariance -> evidence moments ->
quantization/BSC/erasure reporting -> correlated selective fusion ->
system-level `P_D`, together with gates G1-A/B/C/D that make every link
auditable.  If a venue requires a new algorithm, the current greedy must be
replaced or upgraded, for example by a logit-`P_D`-gain greedy with a formal
selection property; the scenario alone does not supply algorithmic novelty.

## Run

```powershell
python scripts/run_demo.py --config config/demo.yaml
python scripts/run_benchmarks.py --config config/demo.yaml
python scripts/run_oracle_study.py --config config/oracle_small.yaml
python scripts/run_ablation_study.py --config config/demo.yaml
python scripts/run_sensitivity_study.py --config config/demo.yaml
python scripts/run_risk_portfolio_study.py --config config/demo.yaml
python scripts/run_chance_portfolio_study.py --config config/demo.yaml
python scripts/run_pd_proxy_study.py --config config/demo.yaml
python scripts/run_pd_diagnosis_study.py --config config/demo.yaml
python scripts/run_deflection_calibration_study.py --config config/demo.yaml
python scripts/run_correlated_erasure_study.py --config config/demo.yaml
python scripts/run_failure_diversity_audit.py
python scripts/run_real_network_headroom_study.py --config config/demo.yaml
python scripts/run_physical_failure_domain_study.py --config config/demo.yaml
python scripts/run_replication_repair_study.py --config config/demo.yaml
python scripts/run_replication_realism_study.py --config config/demo.yaml
python scripts/run_replication_access_study.py --config config/demo.yaml
python scripts/run_multistatic_front_end_gate.py
python scripts/run_multistatic_front_end_gate.py --integration-frames 4
python scripts/run_evidence_calibration_gate.py --trials 150 --gain-mode relative_deficit_reduction
python scripts/run_report_channel_calibration_gate.py --trials 50000
python scripts/run_g1c_conditional_ranking_gate.py
python scripts/run_g1d_greedy_vs_oracle_gate.py
python scripts/run_g2_system_sweep.py --seeds 5
python scripts/run_g2_algorithm_negative_gates.py --seeds 5
python -m pytest -q
```

The demo writes `results/demo_summary.json`.

## OTFS physical-layer direction (Gate 0)

The active physical-layer prototype is isolated in `otfs_physical.py` and
`dd_patterns.py`.  It implements the standard unitary symplectic convention
`X_TF = F_N^H X_DD F_M`, rectangular pulses, and a circular finite frame
(ideal periodic extension or sufficient cyclic prefix).  Integer and
fractional delay-Doppler paths are applied to time samples, and concurrent UAV
echoes are evaluated with waveform-level matched-filter maps.

Run `python scripts/run_dd_collision_gate0.py` for the current reproducible
smoke experiment.  It uses random path phases, common noise across methods,
cyclic non-maximum suppression, and one-to-one cyclic peak matching.  The
detector is deliberately favorable: it knows how many targets use each code.

Gate 0 confirms fractional leakage and concurrent waveform interference.
Gate 1 has **not** passed: in the audited smoke setting, minimizing pairwise
template collision does not improve localization over the same-code baseline.
This physical identity-code Gate 1 is distinct from the G1-A/B/C/D validation
gates described later in this document.
Cyclic shifts of one DD impulse are not distinct identity codes over an
unknown full delay-Doppler search region, so the candidate set now uses
unit-energy QPSK phase patterns.  Pairwise collision cost remains a diagnostic
surrogate, not a validated detection objective or a claimed contribution.

The subsequent Gate-1 audit fixes the frame-level false-alarm probability and
does not give the detector the target count.  It also distinguishes the
unconstrained detection Oracle from a three-code balanced baseline: with four
UAVs and three codes, the latter uses all three codes and permits exactly one
reuse pair.  This reduces, but does not eliminate, identity ambiguity for the
reused pair.  An unconstrained two-code solution benefits from a lower
multiple-testing threshold and cannot by itself establish UAV identity
separation.  Current low-ambiguity QPSK screening and separable CAZAC tests do
not deliver a five-percentage-point gain over the strongest balanced baseline;
Gate 1 therefore remains open.

A separate controlled three-dimensional prototype adds an eight-element ULA
and searches angle, integer delay, and integer Doppler jointly.  It shows that
known per-UAV spatial signatures can resolve targets occupying the same DD
cell when their angular separation is below the array-only resolution.  This
is currently a mechanism result, not a complete MIMO-OTFS claim: spatial
signatures require resolvable per-element or orthogonal probing observations,
their resource cost is not yet modeled, fractional refinement is absent, and
CAZAC signatures have not shown a material advantage over the strongest
random spatial-code baseline.

The next mechanism gate uses the normalized joint probe-angle-DD Gram matrix
to decide whether a coarse two-source collision cluster needs extra probing.
On an independent continuous validation set, the minimum eigenvalue predicts
conditional joint-LS resolution and sharply separates hard from easy cases.
Under sampled corners of a deterministic parameter-uncertainty box, the
minimum-over-samples Gram rule matches the scenario reliability of fixed
two-snapshot probing in the current stratified audit while reducing the mean
probe length.  This remains a known-coarse-support mechanism result: the
trigger has not yet been driven by estimated clusters from the full 3D
detector, and corner sampling is not a continuous-box robustness guarantee.

An initial end-to-end local-cluster audit now adds fine-grid off-grid
refinement and compares one- versus two-source joint-LS fits before deciding
whether to request a second orthogonal probe snapshot.  The current trigger
does not pass the joint gate: its default rule saves 34.4% of snapshots but
loses 8.17 percentage points of joint detection relative to fixed P=2.  Even a
label-aided threshold reachability sweep finds no raw-statistic threshold that
simultaneously saves 20% and limits loss to 1 percentage point (the nearest
points are 20% saving with 1.33 pp loss, or 18.5% saving within 1 pp).  This is
a negative result, not an end-to-end algorithm claim; support-confidence or
sequential evidence is still required.

A follow-up partial-confirmation gate replaces unreliable one-snapshot early
stopping with a full-energy sum observation and a sign-reversed difference
observation at energy fraction delta.  Decoding explicitly divides the
difference by sqrt(delta), so the corresponding noise amplification is
included.  On paired independent validation, delta=0.3 reaches 99.13% joint
detection versus 100% at full difference energy while saving 35% probing
energy.  A three-stage policy supplements the remaining energy for the lowest
40% confidence cases; on its held-out half it saves 20.86% energy and reaches
99.73% versus 100%.  These are energy—not latency or symbol-count—savings, the
coarse collision cluster is still supplied, and the thresholds impose a
conservative familywise false-alarm upper bound rather than exact equality.
The noise-only audit falls back in every frame, so the reported 20.86% saving
is conditional on a supplied two-source candidate cluster; network-wide
energy saving additionally depends on the prevalence of such H1 clusters.

The subsequent Gate-A fairness audit calibrates every complete policy to the
same empirical 1% false-alarm point and compares paired scenes at matched mean
energy.  It is negative for the present confidence rule: fixed equal energy
E=1.5828 reaches 99.93% conditional exact-support recovery, random fallback
99.73%, confidence fallback 99.60%, and the label-aided oracle 100%.  The
confidence-minus-random difference is -0.13 pp with a 95% interval crossing
zero.  Peak power, peak margin, residual ratio, and estimated-support Gram
diagnostics do not separate the remaining correct and incorrect supports.
Thus partial-energy confirmation remains a useful mechanism result, but the
current confidence-driven allocation is not an algorithmic contribution and
should not replace the stronger fixed-equal-energy baseline.

Gate B then introduces three physical incremental observations, target-wise
CFO, phase noise, and correlated complex fading in a non-saturated 5-degree
same-DD collision with 6 dB near-far imbalance.  The E=1.5828 scheme retains
loss within 2 pp on only 9.4%--23.4% of the tested mismatch grid, depending on
the assumed decoder.  Even at the ideal no-mismatch point it recovers 86.5%
versus 92.0% at E=2, a 5.5 pp loss.  Hence the failure is already caused by
insufficient energy in the non-saturated regime, not primarily by CFO or an
ill-conditioned inverse.  An exploratory energy curve suggests roughly
E=1.8 is needed at the ideal point, but its independently sampled 300-trial
points are not a precise minimum-energy estimate.  The physical robustness
gate therefore fails: the earlier 20.86% saving is limited to the saturated,
ideal-coherence experiment.

A known-state short-window extension was also implemented to test controlled
Markov pilot-signature hopping over normal OTFS frames.  It stacks the
pilot--angle--DD signature across frames, includes Doppler phase evolution and
optional Gauss--Markov complex gains, and keeps the state path known at the
receiver.  Gate M0 does not justify a Markov mainline: the strongest fixed
state pair reaches 92.23%, a simple cycle 91.97%, i.i.d. switching 89.01%, and
an independently validated sticky Markov policy 91.99%.  An exhaustively
coordinated deterministic T=3 path reaches 95.27%, only about 3 pp above the
strongest simple baseline and below the required 5 pp.  The effective gain is
therefore attributable to assigning distinguishable known signatures, not to
Markov randomization.  The temporal module is retained as an optional model
extension, but the physical core remains the frame-level pilot--angle--DD
receiver and no Markov-chain innovation is claimed.

The paper mainline is therefore fixed to a single normal OTFS frame with
known transmitter-specific identity signatures, array angle, and continuous
delay--Doppler structure.  Its general physical abstraction contains `M`
concurrent transmitting UAVs, `N` unknown physical targets, and one `L`-element
receive array.  Each target can induce up to `M` geometry-coupled bistatic
paths, so the receiver must map up to `MN` path-level components back to `N`
target-level groups.  Paths belonging to one target share its receive angle,
while their delays and Dopplers are constrained by the transmitter--target--
receiver geometry and target state.  `multistatic_targets.py` implements this
scene layer, including partial illumination; it is a truth-model foundation,
not yet an unknown-cardinality recovery algorithm.

Gate G0-B adds the first unknown-cardinality target-association back end.  It
converts each identity--angle--delay path candidate to a 2-D position through
the receive-angle ray/bistatic-range ellipse intersection, enforces distinct
transmitter identities within a target group, and checks a joint bistatic
Doppler velocity fit.  It is tested with 8% independent path misses, Poisson
0.4 false candidates per scene, and noisy angle/range/Doppler estimates.  In a
100-trial scan, `M=4` attains 98%--100% scene-exact recovery for `N=1,2,3`,
whereas `M=2` attains only 32%--64% because one missed path leaves fewer than
the required two distinct-UAV supports.  This is conditional evidence for the
geometry association mechanism and multistatic path redundancy.  The input
candidates already contain transmitter identity, angle, delay, and Doppler;
therefore G0-B is not end-to-end OTFS performance and does not yet establish a
fixed-codebook gain.  Gate G0-C below supplies a common toy MF/CFAR front end;
a bandwidth-consistent SDR-grade front end remains future work.

The first scalability audit extends G0-B to `M=8, N=6` (48 possible paths).
With 8% path misses and the same false-candidate model, 100 trials contain
44.66 candidates on average.  The indexed back end obtains 100% target recall,
83% target-count/scene-exact recovery, 98.74% identity-association accuracy,
1.06 m position RMSE, and 0.040 m/s mean velocity error.  All 17 failed scenes
over-estimate the count as seven; none under-estimates it.  A local fragment-
merge pass cannot repair them because the fragments already share transmitter
identities, identifying early greedy mis-association rather than simple group
splitting as the remaining algorithmic bottleneck.

The implementation now uses a closed-form receive-ray/ellipse intersection
and a 2-D spatial index before full identity, complete-link, and Doppler checks.
At `M=8, N=6`, mean association time is about 8.31 ms and the 95th percentile
is 11.36 ms on the current test machine, versus about 11.70 ms before these
changes.  Expected sparse-scene work is
`O(K log K + K C M + N^2 M^2)`, where `K` is the candidate count and `C` the
number of spatially nearby groups; the final term is the optional fragment
audit.  Worst-case complexity remains quadratic when all candidates occupy
one tolerance region.  These timings exclude the full SDR-grade OTFS path-
candidate front end; the toy Gate G0-C front end is timed separately below.

## Current multistatic receiver status

- G0-B: unknown-cardinality geometry--Doppler association back end with
  calibrated support gates, collision-triggered order selection, and
  velocity-observability checks.
- G0-C: toy waveform MF/CFAR front end feeding the G0-B back end, with
  per-view support calibration, sidelobe-aware CFAR, and per-peak Fisher-type
  covariance.
- Deep optimization: four-frame noncoherent integration plus sidelobe-aware
  CFAR raises separated-scene recovery to 96.7% (`N=1` and `N=2`) with near-zero
  H1 false candidates; see the Gate G0-C section below.
- Open items: equal bandwidth/frame-budget/communication-rate accounting,
  same-angle--DD collision recovery, strong FWER under mixed nulls, and a
  bandwidth-consistent OTFS front end.

## Gate G1 roadmap

- G1-A: evidence-moment calibration.  Export per-UAV raw matched-filter
  evidence `z_iq` under H0/H1, estimate `(mu_h, Sigma_h)` with shrinkage, and
  check whether predicted deflection orders actual fixed-`P_FA` `P_D`.  A
  smoke implementation is in `uav_otfs_isac/evidence_calibration.py` and
  `scripts/run_evidence_calibration_gate.py`.  Covariances are positive
  definite.  The 10 000-trial formal run (5000 train / 5000 test geometry)
  gives Spearman 0.588 for relative miss-deficit reduction and logit gain
  (bootstrap CI [0.23, 0.83] and [0.21, 0.84]); the point estimate is still
  below the 0.6 gate, so G1-A does not pass the formal gate with deflection
  as the predicted score.  A predicted-score ablation shows the failure is
  specific to deflection: using exact `P_D` gain as the predicted score gives
  held-out Spearman 0.996 (CI [0.98, 1.00]) and relative miss-deficit/logit
  gains give 0.994 in the formal 10 000-trial run.  G1-A therefore passes
  formally only when the selector is `P_D`-gain-based.  A grouped-consistency
  smoke across amplitudes 0.8/1.0/1.3 (60 train/60 test each) gives deflection
  Spearman 0.55/0.33/0.40 (all below 0.6) and `P_D`-gain predicted
  0.97/0.89/0.77 (all above 0.6), so the failure/pass pattern is consistent
  across SNR groups.
- G1-B: quantization/report-channel closure.  Verify Monte Carlo moments of
  quantized, erroneous, erased reports against the exact formulas, so
  communication loss is propagated into evidence moments rather than applied
  as a post-hoc reliability coefficient.  The smoke gate passes: across bits
  1--4, BER 0.01/0.08, erasure 0.9/0.7, and correlation 0/0.5/0.9, 50 000
  Monte Carlo trials give a maximum mean relative error of 4.08% and a maximum
  diagonal/main cross-covariance relative error of 8.51%, inside the 5%/10%
  targets.
- G1-C: conditional set-dependent ranking value.  The current method is a
  conditional-deflection greedy that re-ranks candidates as the selected set
  grows; it is *not* claimed as "better than Top-K".  Under diagonal
  covariance, equal bits, and equal success probability it degenerates to a
  static individual-deflection Top-K.  Baselines are named Static ID Top-K,
  Conditional-Deflection Greedy, and Exhaustive Oracle.  The smoke gate passes:
  the degeneracy test is identical, and in a high-SNR-but-correlated versus
  lower-SNR-independent scenario the greedy chooses the low-correlation report
  and achieves higher `P_D` than Static ID Top-K.
- G1-D: greedy approximation versus Oracle.  In an 8-configuration smoke
  (erasure, correlation, heterogeneous cost), the open-loop `pi*Delta-D/b`
  gain has Spearman 0.90 against the exact marginal expected-deflection gain.
  First-order, exact, and SAA greedy each match the exhaustive Oracle in 50%
  of configurations with a mean deflection gap of 0.161, so the first-order
  score ranks well but budget interactions still matter.
- G2: system-level sweep.  A 20-seed `N=8/12`, `Q=3/5`, `B_max=20/40` sweep
  gives proposed mean `P_D` 0.898 (worst 0.814) under a fair global budget,
  Sensing-SNR Top-K 0.898, Independent-Deflection Top-K 0.897, Communication
  Top-K 0.773, and All-scheduled 0.935.  Exact-`P_D`-gain greedy reaches 0.900
  mean and 0.814 worst, the best among greedy variants; a J-divergence
  surrogate was rejected because it does not align with `P_D` under
  heteroscedastic moments.  Under a strongly correlated model (top-SNR pair
  `rho=0.85`), the conditional greedy reaches mean `P_D` 0.870 versus 0.855
  for Independent-Deflection Top-K and wins in 83.1% of configurations, while
  exact-`P_D`-gain greedy reaches 0.880.  A multi-`rho` sweep
  (0/0.3/0.5/0.7/0.85) shows the conditional greedy beats Static ID Top-K for
  `rho>=0.3` with positive paired-diff CIs in most cells (e.g.,
  `rho=0.5`, `B=20`: `+0.0199`, CI `[0.013, 0.027]`); at `rho=0.85`, `B=20`
  the CI crosses zero.  Exact-`P_D`-gain greedy remains the strongest variant
  at every `rho`, supporting it as the main method.  A non-saturated stress
  gate (three targets, top-SNR pair `rho=0.9`, controlled moderate SNR) shows
  large significant gains: at `B=6`, conditional mean `P_D` 0.692 versus 0.520
  for Static ID Top-K (`+0.172`, CI [0.161, 0.181], win rate 100%); at `B=9`,
  `0.813` versus `0.699` (`+0.114`, CI [0.105, 0.123]).  Worst-target `P_D`
  improves from 0.471 to 0.569 (`B=6`) and 0.577 to 0.782 (`B=9`).  At
  `B=12` both methods saturate to All-scheduled, so the gain vanishes.

Current documentation files:

- `UAV_OTFS_ISAC_论证与系统模型_revised_final_G0C.docx` -- current full
  document with Appendices A/B/C.
- `UAV_OTFS_ISAC_论证与系统模型_revised_final.docx` -- synchronized Chinese
  document with the same Appendices A/B/C.
- `UAV_OTFS_ISAC_System_Model_revised.docx` -- synchronized revised system
  model with the same Appendices A/B/C.
- `PAPER_OUTLINE.md` -- paper skeleton with novelty positioning, gate results,
  and the three experiments still required before submission.
- `AUDIT.md` -- cross-document consistency audit report.
- `RUN_GUIDE.md` -- setup, smoke, and formal run commands for the target
  machine (7800X3D + 5070).
- `INNOVATION_AUDIT.md` -- assessment of whether current performance supports
  scenario/algorithm novelty claims.

The Word documents contain Appendices A (Gate G0-C and optimization results),
B (G1/G2 roadmap, results, and novelty positioning), and C (paper outline).
Obsolete
intermediate document versions were removed.  Regenerate all appendices with
`python scripts/update_system_model_doc.py` using the bundled document
runtime; the script removes and re-appends the appendices, so repeated runs
stay idempotent.

A paired baseline audit shows that the former geometry--Doppler insertion
greedy is not a suitable proposed algorithm.  On the separated `M=8, N=6`
scenes, position DBSCAN, angle--position DBSCAN, and identity-filtered DBSCAN
all reach 100% scene-exact recovery in about 1.2--1.4 ms, while the old greedy
reaches 83% in about 7.4 ms.  The old method is therefore retained only as an
ablation/negative result.

The replacement G0-B candidate is conflict-aware DBSCAN.  It first performs
fast density clustering on the reconstructed positions.  Components without a
repeated transmitter identity are accepted directly.  Only a component that
contains multiple paths from the same UAV triggers local splitting: the
maximum per-UAV path multiplicity gives a local target-count lower bound, and
paths from every UAV are assigned one-to-one to the local centers.  Thus the
extra combinatorial work is confined to collision regions.  It retains 100%
scene-exact recovery on the separated scenes at about 1.24 ms.  In a controlled
paired-collision stress test (three target pairs separated by about two degrees
and nearly equal range), ordinary DBSCAN variants merge each pair and obtain
0% scene-exact recovery; conflict-aware DBSCAN obtains 100%, 99.65% identity
association, and about 1.98 ms mean runtime.  The former greedy obtains only
19% at about 9.27 ms.

These are back-end mechanism results, not yet a final algorithm claim.  The
collision stress test is deliberately constructed and the current false paths
are spatially diffuse.  Correlated local false peaks from the same transmitter
could falsely increase the multiplicity-based target count and must be tested
before this method is promoted to the paper mainline.  A fair end-to-end study
uses the common Gate G0-C toy front end for all association methods; the
remaining fair comparison must add equal bandwidth, frame budget, and
communication-rate accounting.

The multiplicity-only splitter fails a necessary sensing robustness test: a
single target with several correlated local sidelobe/peak-splitting candidates
from the same UAV is falsely interpreted as multiple targets.  The current
candidate method therefore replaces fixed splitting with local physics-
constrained model-order selection.  For each density-connected component and
candidate order `q`, it alternates between (i) per-UAV Hungarian assignment to
`q` target states plus independent clutter dummy columns and (ii) joint
bistatic position/velocity fitting.  The assignment enforces at most one path
from one UAV to one physical target.  For calibrated path-existence probability
`p_k`, target assignment contributes normalized residual plus `-2 log p_k`,
whereas clutter assignment contributes `-2 log(1-p_k)` and the Gaussian-target
versus uniform-clutter density normalization. Peaks from one UAV are treated
as correlated, so the penalized fit uses the number of distinct UAV views
rather than the raw number of local peaks as its effective sample count.

The local order is not selected by comparing nonconvex BIC fits. To avoid
arbitrary combinatorial starts after the statistical gate admits an order, the solver
uses a small set of transmitter-anchored initial states: the strongest `q`
peaks from one UAV form a hypothesis, and all other UAVs validate it through
one-to-one assignment.  The former fixed three-UAV order rule has been removed.
For order `q`, view `m` now supplies one Bernoulli event when it contains at
least `q` candidates above the calibrated confidence threshold.  Under
`H_(q-1)`, its calibrated probability of one extra front-end peak is `pi_m`.
The receiver computes the Poisson--binomial tail
`Pr(sum_m Z_m >= r | H_(q-1))` by dynamic programming and chooses the smallest
support `r` whose tail is at most `alpha_col`.  Order `q` is opened only when
that many distinct UAV views support it.  Peaks within one UAV are never
counted as independent trials.  Sequential testing stops at the first rejected
order, so the probability of the first false order increase is bounded by
`alpha_col` under the calibrated conditional model; it is not multiplied by
the number of lower, true orders.  Every
reported 2-D velocity must also have a full-rank bistatic range-rate matrix
with bounded condition number; a numerically fitted but unobservable velocity
is not accepted as a complete target state.

The receiver uses adaptive complexity.  Components without independently
confirmed collision evidence use the strongest-per-UAV identity estimator.
Only confirmed collision components activate high-order fitting.  Orders that
fail the statistical gate are not searched, so weak same-UAV sidelobes cannot
inflate either target count or runtime.  Multiple
transmitter-anchored starts are screened with position association; only the
best start per order receives joint position--Doppler coordinate-descent
refinement, and an update is accepted only when it lowers the same objective.

With `pi_m=0.1`, `alpha_col=0.05`, 8% path misses, diffuse false candidates,
and correlated same-UAV sidelobe candidates, a 100-trial scale audit gives the
following scene-exact recovery.  At `N=6`, the proposed method obtains
96%/100%/97% for separated scenes at `M=4/6/8`, versus 97%/98%/98% for identity
DBSCAN: it has no general advantage when targets are already separable.  For
three controlled close target pairs, it obtains 45%/95%/99%, while identity
DBSCAN obtains 0% at every `M` because it merges each pair.  The `M=4` loss is
an explicit observability/power boundary: with path misses, too few independent
views remain to confirm every pair at the selected false-alarm level.  Relaxing
the gate merely to improve that number would invalidate the stated error
control.  Mean proposed runtimes are 1.26/2.07/2.51 ms in separated scenes and
15.42/24.13/31.70 ms in collision scenes for `M=4/6/8`, respectively; these
are serial measurements on the current machine and exclude the OTFS front end.
At `M=8`, target-count accuracy is 100%, identity-association accuracy
is 99.21%, position RMSE is 0.99 m, and mean velocity error is 0.204 m/s.
The defensible claim is false-alarm-controlled, collision-triggered recovery,
not universal dominance over DBSCAN.

The implementation retains the historical API label `bic_conflict` so older
audit scripts remain reproducible. It should now be read as a calibrated-order,
penalized-physics estimator: the Poisson--binomial gate chooses the order and
the BIC-like score only ranks initializations within that fixed admitted order.

For a local component, order `q`, `S` transmitter-anchored starts, and at most
`I` alternating iterations, the dominant assignment cost is approximately
`O(S I sum_m max(K_m,q)^3)` for per-UAV Hungarian solves, bounded to collision
components and with `S<=4`, `q<=4` in the current implementation.  Ordinary
components bypass high-order search.  The Poisson--binomial calibration costs
`O(M^2)` once per call and order support costs `O(M q_max)` per component,
which are lower order than collision fitting.  This is more expensive than
DBSCAN but avoids global enumeration of path partitions.

The scalability update adds likelihood-profiled multi-start screening. Before
iterative position--Doppler fitting, each transmitter-anchored center set is
scored by exactly minimizing its position/path-existence cost over per-UAV
partial target assignments and clutter. Velocity is omitted at this stage
because it is not identifiable before grouping. Only the best two starts are
fully fitted. Consequently the expensive term now has `S_refine<=2`; screening
cost is one assignment pass per raw start. This is a bounded-computation
approximation: the profile score is objective-consistent, but retaining two
starts is not claimed to guarantee the globally best nonconvex fit.

An exact target-occupancy subset DP was also derived. After subtracting the
all-clutter baseline, one UAV's assignment can be solved in `O(K q 2^q)` time
and `O(2^q)` value storage because every target has capacity one and clutter
has unlimited capacity. Exhaustive small-instance tests confirm equivalence to
the padded Hungarian formulation. It is not used on the production path:
despite the lower small-`q` operation structure, its Python implementation
increased the `M=8,N=6` collision runtime from 31.70 to 36.20 ms. The optimized
compiled Hungarian routine remains faster, and the DP becomes exponential if
large local collision order is allowed. This negative result is retained to
separate asymptotic structure from actual receiver latency.

With likelihood-profiled screening, 50-trial `M=8` target-count scaling gives
92%/94%/94% scene-exact recovery for separated `N=8/10/12` scenes at
2.99/3.48/5.10 ms. Identity DBSCAN gives 92%/92%/94% at lower latency. In the
paired-collision stress test, the proposed receiver gives 94% scene-exact and
94% target-count accuracy at every `N=8/10/12`, with 99.12%/99.06%/99.20%
identity-association accuracy and 30.61/39.44/48.13 ms runtime; identity DBSCAN
gives 0% scene-exact recovery. Runtime is approximately linear in the number
of separated local collision components in this controlled construction. This
does not imply linear worst-case complexity if many targets occupy one local
resolution cell; the current physical model caps local order at four.

The transmitter-count audit separates two statistical decisions that a fixed
`minimum_transmitters=2` rule conflates. The collision-order gate tests whether
an already detected local component contains an additional target. A new
target-existence gate tests whether a fitted group is more than a cross-UAV
false-path coincidence. For calibrated null probabilities `p_T,m`, it uses

`R_T(M)=min{r: Pr(sum_m F_m >= r | no target) <= alpha_T}`

and requires at least `R_T(M)` distinct UAV identities in every reported
target. Both tails are Poisson--binomial, but `p_T,m` and the collision
extra-peak probability are different physical events and must be calibrated
separately. This removes the observed `M=10,12` failure mode in which one
missed true path and one low-confidence false path formed a two-view phantom
target. It is also more defensible than increasing DBSCAN `min_samples` by an
arbitrary function of `M`.

At the conservative mechanism setting `p_T,m=0.1`, `alpha_T=0.05`, 50-trial
`N=6` scene-exact recovery for `M=4/6/8/10/12` is
80%/98%/100%/100%/100% in separated scenes and
52%/96%/100%/100%/100% in paired-collision scenes. The corresponding collision
runtimes are 16.78/22.64/28.97/36.67/41.72 ms. A 30-trial `N=12` cross-check
gives separated recovery 50%/100%/100%/100%/100% and collision recovery
30%/90%/96.7%/100%/100%. The low-`M` loss is under-counting caused by strict
support after 8% path misses; all over-counting is removed in this audit.

Sensitivity testing confirms that the null calibration, not `M` itself,
determines the support transition. With `alpha_T=0.05`, homogeneous
`p_T=0.02/0.05/0.10` gives required support across `M=4/6/8/10/12` of
`2/2/2/2/2`, `2/2/3/3/3`, and `3/3/3/4/4`, respectively. In 30-trial `N=6`
tests, `p_T=0.02` leaves 13% separated-scene over-counting at `M=12`, whereas
`p_T=0.10` causes 17% under-counting at `M=4`. The intermediate `p_T=0.05`
produces 100% separated and collision recovery for `M>=8` in this synthetic
candidate model, but it is reported as sensitivity evidence rather than a
tuned final value. A deployable receiver must estimate `p_T,m` on independent
target-free CFAR data and attach uncertainty bounds or use a conservative
upper confidence limit.

Increasing `M` has two opposing system effects. For independent, calibrated
multistatic measurements, Fisher information adds as
`J_M=sum_m H_m^T R_m^-1 H_m`; under persistently exciting geometry its
eigenvalues grow proportionally to `M`, so estimation standard deviations can
decrease approximately as `1/sqrt(M)`. In the separated `N=6` audit, mean
velocity error falls from 0.057 m/s at `M=4` to 0.033 m/s at `M=12`, closely
matching that scaling, while position RMSE falls from 1.43 m to 1.11 m. But
the raw candidate population grows approximately as `MN`, increasing false
group opportunities and computation. The adaptive existence gate is what
converts added views into a reliable gain instead of uncontrolled over-counting.

These `M` sweeps hold per-path miss probability and estimation variance fixed.
They therefore represent a back-end conditional-view experiment in which total
pilot energy/identity resources can grow with `M`; they are not a fixed-total-
energy communication comparison. Under equal total pilot energy, per-UAV
energy generally decreases with `M`, changing CFAR detection probability,
`p_T,m`, path-estimation covariance, and possibly the optimal `M`. That tradeoff
cannot be inferred honestly until the toy Gate G0-C front end couples energy
with bandwidth, frame duration, identity-code overhead, and
communication-rate loss.

The subsequent deep audit tightens both experimental design and metrics. UAV
counts are now compared on nested subsets of one fixed 12-UAV mother geometry,
so increasing `M` adds views without relocating existing transmitters. The
legacy `scene_exact_recovery` field is explicitly equivalent to
`position_set_exact_15m`: correct cardinality plus one-to-one position matches
within 15 m. It is not full state recovery. New metrics include position plus
velocity state-exact recovery (all matched velocity errors at most 1 m/s),
GOSPA (`p=2`, cutoff 15 m, `alpha=2`), and path-association precision, recall,
and F1 over all retained true candidates.

Under the original separated-confidence candidate model and nested geometry,
`M=8,N=6` paired-collision position-set recovery remains 100%, but strict
position--velocity state recovery is only 66%, path recall 91.6%, and path F1
94.6%. At `M=12` these become 100%, 74%, 92.5%, and 95.5%, respectively. Thus
the earlier 100% result is a valid position-cardinality mechanism result, not
complete recovery of every path and kinematic state.

A reproducible overlap-confidence stress model assigns true-path scores in
`[0.45,0.9]` and false/sidelobe scores in `[0.2,0.8]`. It deliberately removes
the original threshold separation, where every true score exceeded 0.7 and
every false score was at most 0.6 while the collision trigger was 0.6. With
the hard calibrated-null gate, paired-collision position-set recovery for
`M=4/6/8/10/12` falls to 0%/6%/56%/40%/56%; at `M=8`, strict state recovery is
26% and path F1 is 84.5%. Failure attribution shows that only about 66.8% of
true two-target components trigger second-order fitting at `M=8`. This is the
current primary algorithmic bottleneck.

An exploratory continuous posterior-support gate replaces the hard score
threshold by a Poisson--binomial tail using each view's q-th candidate score as
a Bernoulli probability. It is mathematically meaningful only for calibrated
probabilities. On the deliberately uncalibrated overlap scores it is negative:
paired-collision position recovery is 0% for `M=4,6,8,10` and 14% for `M=12`.
The branch is retained as an ablation and does not replace the default hard
gate. The result demonstrates that raw detector scores cannot be promoted to
probabilities merely by notation; an independent MF/CFAR calibration set or
cross-fitted probability model is required first.

Baseline fairness is also tightened with `gated_identity_dbscan`, which applies
the same Poisson--binomial target-existence support filter to identity DBSCAN.
For `M=8,N=6` overlap-confidence separated scenes it reaches 100% position and
state recovery versus 93% for the proposed method, confirming no advantage in
easy scenes. In paired collisions it remains at 0%, while the proposed hard-
gate method reaches 47% position-set recovery, 24% strict state recovery, and
82.1% path F1. Under the original separated-confidence collision model the
proposed method reaches 100% position-set recovery but only 70% strict state
recovery. The defensible algorithmic claim remains collision decomposition;
robust calibrated triggering is unresolved.

The next correction introduces a reusable weighted-PAV isotonic probability
calibrator. It minimizes weighted squared Bernoulli error subject to monotonicity,
clips only at the likelihood boundary, and losslessly merges adjacent equal-
probability steps. On 29,917 independent calibration candidates the final map
contains 13 steps. On 29,983 held-out validation candidates, score-only
calibration reduces Brier score from 0.1650 to 0.1234, ECE from 0.1578 to
0.0025, and maximum calibration error from 0.3509 to 0.0090. Calibration,
rank-event fitting, null calibration, null validation, and final evaluation
use distinct random seeds.

Directly feeding calibrated candidate probabilities into a local posterior
support gate improves the overlap-confidence `M=8,N=6` collision result from
47% to 64% position-set recovery and from 24% to 45% strict state recovery,
with path F1 increasing from 82.1% to 89.3%. However, that local gate does not
control frame-wide false triggers after scanning multiple selected components.
A selection-aware rank calibrator was therefore tested for the exact event
"the q-th ranked UAV candidate supports a q-th distinct target". Under a 95%
posterior-support requirement it is too conservative and gives 0% complete
collision recovery. This confirms that conditional calibration alone does not
remove cross-view and selection dependence.

The statistically valid correction calibrates the maximum collision statistic
over the complete data-selected frame. With six separated targets and a 1%
family-wise false-trigger target, the probability-only threshold rises from
0.391 under an invalid single-target null to 0.847. Independent 500-frame null
validation gives exactly 1%, but paired-collision position recovery falls to
6%. Thus score aggregation alone has insufficient power at strict frame-level
false alarm.

A physical generalized likelihood-ratio gate was then implemented. For every
selected component it compares the best constrained one-target and two-target
penalized likelihoods using identical calibrated path probabilities, clutter
density, per-UAV capacity constraints, bistatic position--Doppler fitting, and
velocity observability. Its frame-maximum threshold is independently calibrated
on separated six-target frames. At 1% target false trigger, 300 independent null
frames give 1%; on 100 overlap-confidence paired-collision frames it attains
52% position-set recovery, 37% strict state recovery, GOSPA 7.23 m, and path F1
87.1%, versus 0%/0%, GOSPA 18.75 m, and F1 48.6% for gated identity DBSCAN.
Mean proposed back-end runtime is 45.97 ms versus 1.29 ms. Expanding the GLRT
from four to eight deduplicated starts leaves recovery at 52% while increasing
runtime to 54.64 ms, so wider search is rejected as a production change.

An excess-peak-stratified conformal variant was also tested rather than merely
tuning the GLRT threshold. Component-wise conformal p-values lose tail
resolution: the low-excess strata contain only 64 and 291 components, whose
smallest possible p-values are respectively 1/65 and 1/292, both above the
calibrated frame threshold 0.00189. A finite-sample-valid frame-stratified
alternative therefore uses direct frame-maximum quantiles and returns an
infinite threshold when a stratum cannot support 1% resolution. In 2,000 null
frames only 35 usable frames occupy the low-excess stratum; 1,937 occupy the
high-excess stratum. The latter sets exactly the same threshold (131.51) as the
pooled test, so stratified and pooled methods both attain 30% position recovery,
25% strict state recovery, GOSPA 9.97 m, and F1 82.3% on the same 100 collision
frames. Stratification is consequently closed as a negative branch: it adds no
power under the observed nuisance distribution.

These results define the present limit honestly: calibrated physical order
testing has real collision-discrimination value, but complete-scene recovery
remains limited at a 1% frame-wide false trigger. A dedicated information audit
must distinguish detecting at least one collision component (a frame-maximum
question) from correctly opening and fitting every collision component (the
actual scene-recovery question). Further work should target the conditional
post-trigger partition/state fit or improve front-end information only after
that decomposition, rather than relax the false-alarm definition or enlarge
local search blindly.

The resulting 2,000-null/1,000-collision information audit makes this
distinction quantitative. The frame-maximum GLRT has empirical AUC 0.9991 and
detects at least one collision component in 97.3% of paired-collision frames at
the finite-sample 1% threshold. However, the number of components crossing that
same maximum-test threshold is 0/1/2/3 in 27/185/430/358 frames, respectively:
only 35.8% open all three true collision components. Complete position recovery
near 30% is therefore primarily a simultaneous multiple-testing power loss,
with a smaller post-trigger partition/state-fit loss. The mathematically
motivated next branch is a held-out, finite-sample step-down maxT test that
calibrates successive null order statistics and preserves frame-wise error
control; independent per-component threshold relaxation is not admissible.

That step-down gate is now implemented and independently tested. Four ordered
null thresholds are 131.09, 44.31, 28.57, and 18.55. Sequential testing stops
at the first failed rank, so under the separated-frame global null the event of
any rejection remains exactly the first frame-maximum event. Independent
validation gives 0.8% frame false triggers. On 100 paired-collision frames,
position-set recovery rises to 83%, strict position--velocity recovery to 69%,
GOSPA falls to 4.25 m, and path F1 reaches 91.9%; the corresponding single-
threshold same-partition result is 30%/25%, 9.97 m, and 82.3%. Mean proposed
back-end runtime is about 54 ms. This is a defensible algorithmic improvement,
not a stronger statistical claim: the current proof gives weak FWER control
under the global no-collision null. Strong FWER with a mixture of true and null
components would require subset pivotality or separately calibrated closed
testing and is explicitly not claimed.

Two follow-up audits sharpen both the estimator and that limitation. First,
after target order, paths, and position are fixed, bistatic Doppler is linear in
velocity. Replacing the final confidence-weighted least-squares velocity fit by
a confidence-weighted Huber IRLS fit therefore changes neither detection nor
association. Across three new paired seeds (300 frames), it leaves all 293/300
separated-scene position/state successes unchanged. In collision scenes it
raises strict state recovery from 62.0% to 77.7% (+15.7 pp), records 47 robust
wins and zero ordinary-LS wins, and reduces mean velocity error from 0.295 to
0.202 m/s. On the original 100 collision frames it preserves 83% position
recovery while increasing strict state recovery from 69% to 82%. This robust
post-fit is retained because it follows the physical bistatic observation model
and does not alter the calibrated collision decision.

Second, a mixed-null scene with one true collision pair and four separated
targets exposes the weak-FWER boundary. Across three paired 100-frame seeds,
the single maximum threshold gives 73.3% mean position/state recovery and no
over-counting. Weak-FWER step-down gives 70.0% and over-counts 1%--5% of frames.
Thus unrestricted step-down is beneficial in the three-collision stress scene
but is not yet a universal receiver. The next statistically defensible version
must calibrate remaining-null maxima under independently generated 0/1/2-pair
configurations (or use a valid closed-testing construction); rank-dependent
thresholds learned only from the global null cannot support a strong-FWER
claim.

Configuration-aware and density-triggered variants were then audited to address
that mixed-null failure. Offline truth labels were used only to estimate the
maximum GLRT statistic among normal components in independent 0/1/2-pair
calibration scenes; no labels enter the online receiver. In a shared-first-
threshold ablation, the configuration thresholds (74.28, 61.02, 66.38) remove
the 4%--6% over-counting of global-null step-down, but exactly reproduce the
single-threshold recovery in all 0/1/2/3-pair scenes. They are therefore a
negative result, not a useful new algorithm. Requiring two components above the
first threshold before activating low step-down thresholds restores 92% recovery
and zero over-counting for one pair, but still over-counts 8% with two pairs.
This exposes an identifiability limit of ranked scalar GLRTs: two strong true
collisions plus one normal component cannot be reliably distinguished from two
strong and one weak true collision by rank alone.

A separate implementation audit found a genuine complexity improvement. The
physical GLRT already refines four deduplicated two-target starts to compute its
decision statistic, but the estimator previously discarded that winning model,
screened only two starts, and solved the same local problem again. The GLRT now
returns its winning two-target model and the estimator uses it directly as the
initial point for joint refinement. On identical calibration, thresholds,
seeds, and 50-frame paired scenes, every recovery, over-counting, state, and
GOSPA result is unchanged. Mean back-end time falls by 6.2%, 21.5%, 35.8%, and
45.5% in the 0/1/2/3-pair scenes, respectively (three-pair: 68.96 to 37.58 ms).
This detection--estimation consistency change is retained; it reduces repeated
optimization without changing the test statistic or its error-control scope.

A refined-GLRT ablation then applied three monotone joint assignment--position--
Doppler coordinate-descent iterations under both the one- and two-target
hypotheses. Recalibrating each statistic independently at the same 1% global-
null level is essential: reusing coarse thresholds would be invalid. Full
refinement improves the three-pair scene from 64% to 78% recovery, but reduces
one-pair recovery from 68% to 56%, slightly reduces two-pair recovery, and adds
roughly 40%--50% runtime. It is therefore rejected as a universal replacement;
refinement makes weak true collisions easier to fit but also makes normal
components more compatible with a two-target explanation.

The retained performance branch is a configuration-calibrated coarse-to-refined
cascade. Its three finite-sample thresholds correspond directly to sequential
risks: (i) the maximum coarse GLRT in 500 separated frames, (ii) the maximum
normal-component coarse GLRT in 500 independent one-pair frames, and (iii) the
maximum normal-component three-iteration GLRT in 500 independent two-pair
frames. This gives thresholds 126.74, 64.92, and 55.96 from 464 and 391 finite
mixed-null maxima at stages two and three. Online operation uses no labels: it
tests the first two ranked components coarsely and computes the refined third
test only after both preceding stages pass.

On the initial paired 50-frame evaluation, cascade recovery is 100%, 68%, 86%,
and 88% for 0/1/2/3 collision pairs. It removes the coarse step-down over-count
in the two-pair case (6% to 0%) while improving recovery from 80% to 86%; in the
three-pair case it improves 86% to 88%. Three additional fixed-threshold seeds
(150 frames per configuration) give mean 0/1/2/3-pair recovery of 98.7%, 76.0%,
81.3%, and 81.3%, with corresponding over-count rates 0.7%, 0%, 0.7%, and 0%.
Mean two-/three-pair GOSPA is 4.41/4.16 m and path F1 is 91.5%/92.3%. These are
independent validation results: thresholds are not refitted per seed.

The cascade is a substantial and physically consistent performance improvement
over one frame-wide threshold, because expensive joint refinement is reserved
for the weak third collision for which it has demonstrated power. Its formal
scope remains limited: finite-sample calibration supports the explicitly
simulated N=6 configuration family under exchangeability. Arbitrary target
loads, channel mismatch, or mixed configurations outside that family do not
inherit a distribution-free strong-FWER guarantee and must be separately
validated.

Further optimization tested whether coarse and refined GLRT statistics contain
complementary information. On 904 independently generated true collision
components and 866 normal components, their Spearman correlation is 0.935.
Refined GLRT has a slightly higher component AUC (0.9247 versus 0.9160): 816
collisions pass both calibrated thresholds, 17 pass only refined, five pass
only coarse, and 66 pass neither. Each statistic produces two distinct normal-
component false triggers, with no common false trigger. Thus complementarity is
real but small and an uncalibrated OR rule is invalid.

A split empirical-CDF max fusion was therefore tested: one independent set
maps coarse/refined statistics to marginal empirical percentiles, and a second
set calibrates the frame maximum of their fused percentile. Although fusion AUC
rises to 0.9372, only 312 normal training components are available; upper-tail
percentiles saturate at one and the valid 1% frame threshold is also one. The
strict finite-sample test consequently has zero power. Changing `>` to `>=`
would reintroduce uncontrolled false alarms, so empirical-CDF fusion is closed
as a negative branch rather than patched heuristically. Continuous extreme-tail
modeling could avoid ties but would impose an additional distributional
assumption not justified by the current toy candidate front end.

Increasing the post-decision joint-refinement cap from three to ten iterations
was also evaluated on 100 paired frames per 0/1/2/3-pair configuration. Every
position, state, cardinality, GOSPA, F1, and velocity-error result is identical;
runtime is unchanged except for a small three-pair increase (44.18 to 45.10 ms).
The coordinate descent already converges within three iterations, so the higher
cap is rejected.

The remaining defensible performance opportunity lies upstream. The earlier
GLRT assigned every path the same 3 m position and 3 Hz Doppler scale. Gate
G0-C now exports per-peak angle/range/Doppler curvature scales and uses them
in post-decision GLS; moving those scales into assignment or GLRT scores would
alter the null statistic and require fresh held-out calibration, so that
heteroscedastic decision-stage use remains future work.

As a controlled intermediate step, the range--bearing inversion now propagates
the synthetic front-end range and receive-angle noise through its exact local
Jacobian.  For bistatic range \(\rho\), bearing \(\theta\), receiver-relative
transmitter baseline \(b\), and unit ray \(u(\theta)\), the ray distance is
\(r=(\rho^2-\lVert b\rVert^2)/(2(\rho-u^Tb))\).  The position covariance is
therefore the delta-method matrix
\(\Sigma_p=J_{\rho,\theta}\operatorname{diag}(\sigma_\rho^2,
\sigma_\theta^2)J_{\rho,\theta}^T\), with a small eigenvalue floor only for
numerical stability.  A 14,400-path geometry audit gives median minor/major
standard deviations of 0.775/1.765 m, a 90th-percentile major deviation of
7.99 m, and a median condition number of 5.11.  The Jacobian covariance agrees
with a 20,000-sample Monte Carlo calculation to 1.85% relative error.  Thus a
single isotropic 3 m position scale is physically misspecified even under the
current synthetic front end.

The covariance is currently used only after model order and path association
have been fixed.  Conditional on the retained paths, the final position is the
generalized least-squares estimate
\((\sum_i p_i\Sigma_i^{-1})^{-1}\sum_i p_i\Sigma_i^{-1}\hat p_i\), followed by
the existing Huber Doppler/velocity refit at that position.  Here \(p_i\) is an
existence weight and \(\Sigma_i\) is conditional measurement precision; these
two quantities are deliberately not identified with each other.  Across three
independent seeds and 300 frames per configuration, this post-decision update
leaves every seed's target count, strict-state recovery, and path F1 exactly
unchanged.  Mean GOSPA changes by -0.01%, -5.88%, -13.14%, and -20.49% for
zero, one, two, and three collision pairs, respectively; mean velocity error is
essentially unchanged and runtime rises by roughly 1.7--3.2 ms in collision
scenes.  The growing gain with collision density is consistent with replacing
an increasingly harmful equal-precision average, while the separated case
provides a useful no-regression control.

This is not yet a heteroscedastic detection claim.  Moving \(\Sigma_i\) into
assignment or GLRT scores would alter the null statistic and invalidate the
three existing cascade thresholds, requiring fresh held-out calibration and an
OTFS-derived peak covariance model.  The present result supports a modular,
geometry-aware state estimator without borrowing performance from target-count
selection or false-alarm control.

An exact nonlinear range--bearing Gaussian MLE was also tested after the same
fixed decision, initialized by GLS.  Across the same three seeds its GOSPA
differs from GLS by less than 0.008% in every configuration, while adding about
10.3 ms in the three-collision-pair case.  This agrees with the independent
Monte Carlo Jacobian check: at the simulated noise level the first-order model
is already accurate.  The nonlinear solver is therefore rejected from the
online mainline rather than retained merely for additional sophistication.

Gate G0-C replaces the synthetic path-candidate oracle with a toy
matched-filter/CFAR front end.  An eight-element ULA observes superposed
per-UAV unit-energy QPSK identity signatures on one DD grid; the receiver
computes separable angle--delay--Doppler matched-filter cubes, applies an
empirical max-map threshold for a 0.2% frame-level false-alarm bound, and
extracts unknown-count NMS peaks.  Peak scores are mapped to
path-existence probabilities by isotonic regression on held-out noise-only
and true-path frames, and the calibrated candidates are fed to the existing
physics-constrained association back end.  The toy grid uses explicitly
declared delay and Doppler resolutions and is not a bandwidth-consistent SDR.

On 50 independent separated `M=4, N=1` frames, path recall is 92%, scene-exact
recovery is 90%, mean matched position error is 1.37 m, and the mean GOSPA
(cutoff 15 m) is 2.46 m.  On 50 separated `N=2` frames, path recall is 92.5%,
scene-exact recovery is 82%, mean matched position error is 1.12 m, and mean
GOSPA is 3.47 m.  These use per-view false-target and false-extra
probabilities calibrated on held-out single-target components, together with
a two-view collision floor; the earlier conservative `p=0.1` setting reached
94% on both configurations but was not a measured error-control value.  Two
hundred noise-only validation frames contain no candidate at the calibrated
0.2% frame bound, while H1 sidelobe and cross-code leakage adds about 2.1
false candidates per frame for `N=1` and 5.7 for `N=2`; the association back
end absorbs this clutter through its calibrated target-existence gate.  Mean
front-end runtime is about 0.03 s per frame and association below 1 ms on the
current machine.  The gate is limited to separated scenes; resolving two
targets in one angle--DD cell remains a separate collision gate, and the grid
resolutions are not yet coupled to bandwidth, frame duration, or
communication-rate overhead.

The front end also exports per-peak angle, range, and Doppler scales from the
local curvature of the matched-filter cube.  For a locally quadratic
log-likelihood the negative second derivative is a Fisher-type precision
estimate, so each path carries its own measurement variance instead of one
fixed 3 m/3 Hz scale.  In a paired 50-frame comparison, post-decision GLS with
these per-path scales leaves path recall, target count, and scene-exact
recovery unchanged (90% for `N=1` and 82% for `N=2`) while reducing mean GOSPA
from 2.46 m to 1.88 m (`N=1`) and from 3.47 m to 2.82 m (`N=2`); mean matched
position error falls from 1.37 m to 0.81 m and from 1.12 m to 0.65 m.
Velocity error is essentially unchanged.  This is a state-estimation gain that
does not borrow from target-count selection or false-alarm control; the
curvature scales are a toy-Fisher approximation, not a calibrated CRB.

The support probabilities are estimated from held-out single-target spatial
components, so the Poisson-binomial null no longer hides an arbitrary 0.1.
The empirical grouped collision null returns support one, which is floored at
two because one view with an extra peak is not independent collision
evidence.  The resulting operating point is stricter than the earlier
conservative setting, and it quantifies the gap between mechanism performance
and a claimed calibrated error bound; a strong FWER guarantee under mixed
true/null components would still require closed testing.

The first deep front-end performance branch uses four-frame noncoherent
integration together with a sidelobe-aware CFAR threshold.  Integration
averages per-frame matched-filter energy, which under complex AWGN raises the
per-cell statistic toward a 2F-degree-of-freedom chi-square law; the CFAR
threshold is calibrated on held-out single-target frames after blanking the
true peak neighborhoods, so it sits above the deterministic waveform and
array sidelobe floor rather than the pure-noise floor.  On 30 independent
frames this raises `N=1` scene-exact recovery from 90% to 96.7% and `N=2` from
82% to 96.7%, while H1 false candidates fall from 2.1/5.7 per frame to
0.0/0.1.  Mean GOSPA falls to 1.05 m (`N=1`) and 1.44 m (`N=2`) with the
front-end covariance refit, and path recall is 90%/87.5%.  One hundred
noise-only validation frames still contain no candidate.  This mode spends
four frame durations per observation and has a front-end runtime of roughly
0.08--0.14 s per frame; equal-bandwidth, equal-frame-budget, and
communication-rate accounting remain future work.  A first equal-total-energy
check confirms the gain is resource-driven: halving per-frame amplitude to
keep total pilot energy fixed removes the recovery gain.  A first independent
Rayleigh-fading audit at the same average total energy also does not recover
the gain at the toy front-end's fractional-leakage threshold, so time
diversity remains exposed as an experimental script flag but is not claimed
as a demonstrated fair gain.

A first equal-total-pilot-energy audit fixes total pilot energy at `E=16`
and scales per-UAV amplitude as `sqrt(E/M)` for `M=2` and `M=4`.  With the
same target-existence gate (support two), 30 separated `N=2` trials give
scene-exact recovery 76.7% for `M=4` versus 6.7% for `M=2`, while path recall
is 89.6% versus 92.5%.  The `M=2` system actually detects paths slightly
better because each UAV carries more energy, but it loses enough independent
views to clear the calibrated existence gate.  The extra views therefore
convert into association reliability despite lower per-UAV pilot energy.
This remains a conditional mechanism audit: it does not yet account for
bandwidth, frame duration, or communication-rate overhead.

Important physical and communication qualifications remain.  Candidate
confidence must be calibrated by a common OTFS front end; raw correlation
amplitude is not automatically a path-existence probability.  The current
`pi_m=0.1` in the older synthetic audits is a conservative sensitivity
setting, not a measured property of an implemented detector; Gate G0-C now
estimates per-view false-target and false-extra probabilities on held-out
front-end components, and formal detector-level claims still require
independent target-free CFAR calibration without reusing evaluation scenes.
The present `alpha_col` controls one local sequential order-opening test; a
scene-wide family-wise guarantee over many data-dependent DBSCAN components
would additionally require an alpha-spending or multiplicity correction.
Conditional independence across UAV views can fail under common interference,
shared clutter, synchronization error, or code leakage; in that case the
Poisson--binomial threshold is not guaranteed and must be replaced by a grouped
or empirically calibrated null law.  The current
geometry also excludes rank-deficient velocity configurations, acceleration,
direct-path residuals, and array calibration error.  Transmitter identities
consume pilot dimensions or coding resources, so their net ISAC benefit must
later be evaluated under equal total pilot energy, bandwidth, frame duration,
and communication-rate overhead.  Until that end-to-end audit is complete,
these results support a structured association mechanism, not a deployable
receiver or a final paper claim.

Gate S0 remains a deliberately controlled `M=4, N=2`, known-path-count
mechanism experiment.  It compares shared pilots, a four-transmitter
two-feature fixed nonorthogonal identity codebook, and an orthogonal upper
bound using the same two-path joint-LS receiver and equal pilot energy.  Across
same-DD near-angle, fractional-leakage, and 6--10 dB near-far cases, position
set recovery is 53.78%, 65.89%, and 68.89%, respectively.  The realistic fixed
codebook gains 12.11 pp over shared pilots and passes the 10 pp mechanism gate.
This establishes the value of fixed transmitter identity structure only for
separating returns from different transmitters, not the value of a general
receiver.  It cannot resolve two returns with the same transmitter, angle, and
DD response.  Unknown target count, false peaks, path recovery, geometry-aware
target association, and joint continuous state refinement remain future Gate
G0/S1 work.  The old known-`Q=2` MF/OMP/SIC comparison can only serve as an
algorithm-isolation test; formal S1 must estimate path cardinality and group
paths into physical targets without being given `N`.
The benchmark script compares the proposed selector against no cooperation,
sensing-only Top-K, communication-only Top-K, independent post-report ranking,
random selection, and all-scheduled fusion. `exhaustive_oracle` is provided for
small candidate sets. It uses common random numbers across methods and reports
both the optimized weighted expected deflection and detection probabilities;
these metrics need not rank every method identically in a finite experiment.

`run_risk_portfolio_study.py` compares a risk-neutral portfolio, a mean-CVaR
portfolio, and the former marginal-deflection greedy solver. CVaR controls the
severity of upper-tail deficits; it is not presented as a substitute for a
chance constraint on the probability of violating the detection threshold.
`run_chance_portfolio_study.py` adds per-target chance constraints and returns
the minimum weighted reliability relaxation when the requested reliability is
not attainable under the available reporting budget.

The detection-level studies use a moment-matched Gaussian linear-score
probability of detection at fixed false-alarm rate. This is called
`gaussian_pd_chance`; it is not an exact discrete likelihood-ratio detector.
The diagnosis searches all deterministic received subsets and all scheduled
portfolios, so it does not assume that Gaussian-score P_D is set-monotone.
The calibration study checks the equal-covariance theoretical mapping and a
train/test-separated isotonic deflection calibration before comparing either
surrogate with direct Gaussian-P_D chance optimization. Proxy studies report
exact schedule matches separately from target and global-pair Jaccard overlap.

The correlated-erasure study supports global and grouped common-state link
failures while preserving every link's marginal success probability exactly.
This isolates dependence misspecification from average-link degradation. The
current experiments show that common failures make an independent-link model
optimistic; correlation-aware scheduling is evaluated without assuming that
it must recover the lost feasibility.

`run_failure_diversity_audit.py` is a controlled mechanism gate with explicit
failure-group labels. It verifies that equal-quality One-of-2 evidence should
diversify across groups, while a Two-of-2 threshold can favor positively
dependent reports. A quality-gap sweep identifies when sensing quality
outweighs recoverable failure-diversity headroom.

`run_real_network_headroom_study.py` attributes the small real-network gain
target by target. It reports the minimum number of successful reports needed
to clear the deterministic detection threshold, the number of distinct
failure domains containing a single-report substitute, the exact recoverable
headroom relative to a target-local correlated Oracle, and the fraction of
that headroom used by correlation-aware scheduling.
The Oracle is target-local and may spend up to the stated system budget on one
target; it diagnoses intrinsic target headroom, not a jointly attainable
multi-target allocation. Consequently, a target-level use ratio can be
negative when the shared-budget optimizer deliberately trades that target for
larger system-level benefit.
For system reporting, the study also provides a weighted aggregate headroom
capture rate, a nonnegative gain-capture rate, and the harmed-target fraction;
these are more stable than averaging ratios from targets with tiny headroom.
The physical-domain study compares result-independent, target-aware geometric
groupings while preserving each reporting link's marginal success probability.
Owner-centered angle/path clustering is the primary physical proxy. Formation
position and straight-link midpoint clustering are both reported, but their
equivalence is checked explicitly and they are not counted as independent
robustness evidence when they induce the same partition.

The original replication study is an ideal upper bound in which a second copy
can avoid the entire common failure state. The realism study supersedes that
interpretation by separating a physical-path availability shared by all copies
from schedulable resource-domain states. It applies equal per-domain capacity
limits and compares selection only, same-domain retransmission, and
cross-domain duplication under the same two-layer risk law. Replication is not
claimed to overcome physical-path failure: its benefit declines as the shared
path-risk fraction increases, and the exact optimizer is an offline small-scale
Oracle rather than a scalable online scheduler.

## What is deliberately not claimed

- The greedy score is not claimed to be submodular or to have a fixed
  approximation ratio.
- `pi * Delta-D / bits` is only the document's open-loop first-order score;
  selection uses the true expected-deflection difference when exact
  enumeration is tractable.
- The fusion direction is optimal only among linear fusion rules under the
  null-hypothesis deflection criterion.
- The built-in waveform calibration is a lightweight DD leakage model. It is
  not a complete SDR-grade OTFS transceiver.
- The Gate G0-C front end uses an abstract DD grid with declared delay and
  Doppler resolutions; it is not bandwidth-consistent and does not yet
  include equal-total-pilot-energy or communication-rate accounting.
- The per-peak covariance export is a local matched-filter curvature
  approximation of Fisher precision, not a calibrated CRB or a full
  heteroscedastic likelihood used inside target-count selection.
- The sidelobe-aware CFAR threshold is calibrated on held-out single-target
  frames and is not a universal clutter or mutual-interference model.

## GitHub reuse policy

See `THIRD_PARTY.md`. No third-party source code is copied into the core
package. A user may opt into the external GPL OTFS toolbox through
`ExternalOTFSBackend`.
