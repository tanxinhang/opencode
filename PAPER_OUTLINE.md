# UAV-OTFS-ISAC Paper Outline

## Working title

Selective Soft-Information Fusion for UAV-OTFS-ISAC under
Communication-Corrupted Correlated Evidence

## Novelty positioning (must be explicit in the paper)

- New: the integrated scenario and its end-to-end validation chain.
  Toy MF/CFAR front end -> per-path Fisher covariance -> evidence moments ->
  quantization/BSC/erasure reporting -> correlated selective fusion ->
  system-level P_D.
- Not new: the conditional marginal-deflection greedy itself. It is an
  adaptation of established deflection-optimal linear fusion and greedy
  subset selection.
- Claim: conditional set-dependent ranking provides measurable gains under
  correlated evidence and reporting, not universal dominance over Top-K.

## Abstract bullets

- Multistatic UAV-OTFS sensing produces correlated soft evidence that is
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

## 1. Introduction

- Motivation: UAV-OTFS-ISAC reports are correlated, communication-corrupted,
  and budget-limited.
- Gap: existing work optimizes trajectories, beamforming, or bandwidth, but
  not selective fusion of post-communication correlated soft evidence.
- Scope: fixed geometry/waveform, aligned candidate targets, one fusion
  center.

## 2. System model and scenario

- OTFS DD evidence with fractional Doppler and leakage.
- Per-path Fisher-type covariance from matched-filter curvature.
- Quantization, BSC, detectable erasure, random effective received set.
- Moment-matched Gaussian fusion and expected deflection.
- Multi-target QoS and budget constraints.

## 3. Method

- Static Individual-Deflection Top-K (baseline).
- Conditional-Deflection Greedy (main adapted method).
- Exact-P_D-Gain Greedy (alternative, best in correlated smoke).
- Exhaustive Oracle (small instances).
- Not claimed: submodularity, approximation ratio, universal Top-K dominance.

## 4. Gates and results

### G0-C waveform front end

- Separated-scene recovery 96.7% with four-frame integration and
  sidelobe-aware CFAR.
- Per-path covariance reduces GOSPA by roughly 30%.
- Resource fairness (same-scale, 30 trials per column): fixed per-frame energy
  raises 86.7% to 100% with 4x energy; fixed total energy drops to 50%, so
  the gain is resource-driven.

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

## 5. Boundaries and open items

- Equal bandwidth/frame-budget/communication-rate accounting is not yet
  complete.
- Rayleigh time diversity does not recover the integration gain in the toy
  front end.
- Strong FWER under mixed nulls requires closed testing.
- Same angle-DD collision decomposition remains a separate gate.
- Front end is toy-resolution, not bandwidth-consistent SDR.

## 6. Required experiments before submission

1. G1-A formal run: 10 000 trials per hypothesis, report Spearman with
   bootstrap CI.
2. G2 correlated audit: 10-20 seeds, report win-rate CI for 77.5%.
3. One honest resource-accounting table: fixed per-frame energy, fixed total
   energy, fixed total frame budget.

## 7. Current repository mapping

- `uav_otfs_isac/front_end.py`: G0-C front end.
- `uav_otfs_isac/evidence_calibration.py`: G1-A.
- `uav_otfs_isac/report_channel_calibration.py`: G1-B.
- `scripts/run_g1c_conditional_ranking_gate.py`: G1-C.
- `scripts/run_g1d_greedy_vs_oracle_gate.py`: G1-D.
- `scripts/run_g2_system_sweep.py`, `run_g2_algorithm_negative_gates.py`: G2.
