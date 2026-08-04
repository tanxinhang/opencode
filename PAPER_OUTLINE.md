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

### G1-A evidence-moment calibration

- Positive-definite covariance after shrinkage.
- Formal 10k run (5000 train / 5000 test geometry): Spearman 0.588 for
  relative miss-deficit reduction and logit gain (CIs [0.23, 0.83] and
  [0.21, 0.84]); deflection does not pass the 0.6 gate.  With exact `P_D`
  gain or logit `P_D` gain as the predicted score, held-out Spearman is
  0.93/0.91 (300-trial smoke), so a `P_D`-gain selector passes G1-A.

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

- Fair global-budget comparison: proposed 0.886 vs Sensing 0.886,
  Independent 0.884, Communication 0.743, All-scheduled 0.933.
- Strong-correlation model: proposed 0.869 vs Independent 0.859, wins in
  77.5% of configurations; exact-P_D greedy 0.880.

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
