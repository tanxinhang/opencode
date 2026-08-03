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

## GitHub reuse policy

See `THIRD_PARTY.md`. No third-party source code is copied into the core
package. A user may opt into the external GPL OTFS toolbox through
`ExternalOTFSBackend`.
