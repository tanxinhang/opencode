# Unified System Model and Notation

This document centralizes the formal model used by the G3-G5 results so the
paper can reference one consistent system model instead of scattered gate
descriptions.  All quantities are defined exactly as implemented in
`uav_otfs_isac/`.

## 1. Scope and assumptions

- Fixed geometry and waveform parameters; aligned candidate targets; one
  fusion center.
- `M` transmitting UAVs, `Q` target hypotheses, one `L`-element receive
  array.
- The sensing channel is a direct path plus an RIS-cascaded path whose phase
  profile is controllable.
- Reports are quantized, pass through a binary symmetric channel (BSC), and
  may be erased; the receiver fuses only the actually received reports.
- The operating point is fixed false-alarm rate `P_FA`; the primary metric
  is system-level detection probability `P_D` and its worst-target value.

## 2. Geometry and scenario

- UAV positions `p_m`, target positions `t_q`, receiver position `r`, RIS
  position `s`.
- Direct bistatic path for UAV `m` and target `q` has range sum
  `R_dir = |p_m - t_q| + |t_q - r|`.
- RIS cascaded path has legs `R_1 = |p_m - s|`, `R_2 = |s - t_q|`,
  `R_3 = |t_q - r|`.

## 3. RIS channel

The controlled additive-power gain used in G5 is

`gain_mq = 1 + (strength_q * array_gain(theta_q))^2`,

where `strength_q` is the RIS illumination strength for target `q` and
`array_gain(theta)` is the normalized phased-array response toward the target
direction.

The physics-based channel used in G5-P follows the two-way bistatic radar law
for the direct path,

`P_dir = 1 / (R_dir_tx^2 R_dir_rx^2)`,

and a three-leg cascaded loss for the RIS path,

`P_ris = N_ris^2 array_gain(theta)^2 aperture_scale /
         (R_1^2 R_2^2 R_3^2)`,

so the evidence SNR gain is `1 + P_ris / P_dir`, with an optional direct-path
blockage for the weak target.  Both models are monotone in array alignment
and never reduce a link's evidence SNR.

## 4. OTFS evidence moments

- DD grid with `N_d` Doppler bins and `N_l` delay bins; declared resolutions
  `Delta_f` and `Delta_tau`.
- Fractional Doppler leakage is modeled by `sinc^2(fractional_doppler)`.
- Per-UAV evidence under H0/H1 is moment-matched Gaussian
  `(mu0, mu1, Sigma0, Sigma1)` after non-coherent integration, quantization,
  and BSC propagation.
- Positive definiteness of `Sigma0` and `Sigma1` is enforced by shrinkage
  regularization.

## 5. Reporting channel

- Per-UAV report cost `b_i` bits; quantization levels `2^{b_i}`.
- BSC transition with bit-flip probability `epsilon_i`.
- Detectable erasure modeled by the reception law `gamma` over the scheduled
  report set, which may be independent, common-state, or grouped.

## 6. Fusion

For a received set `R`, the deflection-optimal linear score has weight
`w = Sigma0_R^{-1} delta_R`, where `delta = mu1 - mu0`.  The Gate G3
one-parameter family is

`w(mu) = L^{-T} (Q + mu I)^{-1} L^{-1} delta`,

with `Sigma0 = L L^T` and `Q = L^{-1} Sigma1 L^{-T}`.  The KKT-optimal member
maximizes `P_D` over linear scores at operating points with `P_D > 0.5`, and
the resulting set function is monotone under report addition.

## 7. Selection objective

The system selects, for each target `q`, a scheduled report set `S_q` under
the global bit budget.  The honest objective is expected `P_D` over the
reception law:

`E_PD(q, S_q) = E_gamma[ P_D(owner union received(S_q, gamma)) ]`.

The G4 selector is a two-stage greedy: minimize normalized miss-deficit, then
maximize expected-`P_D` gain per report bit.

## 8. Resource identities

- Report bits: `B_report = sum_q sum_{i in S_q} b_i`.
- RIS control bits (amortized): `B_control = N_ris * phase_bits /
  coherence_frames`.
- Total bits per frame: `B_total = B_report + B_control`.
- Time-bandwidth symbols (conservative 1-symbol-per-bit): report and control
  TB equal their bit counts; sensing TB is `frames * N_d * N_l`; identity TB
  is `M * N_d * N_l`.
- Sensing energy (unit amplitude): `E_sensing = frames * M * amplitude^2`;
  the passive RIS adds no transmit energy.

## 9. Performance metrics

- Mean expected `P_D`: `(1/Q) sum_q E_PD(q, S_q)`.
- Worst-target expected `P_D`: `min_q E_PD(q, S_q)`.
- QoS feasibility at threshold `tau`: all targets satisfy
  `E_PD(q, S_q) >= tau`.
- Paired bootstrap 95% CIs and win rates for gains over baselines.

## 10. Notation table

| Symbol | Meaning |
| --- | --- |
| `M`, `Q`, `L` | UAVs, target hypotheses, receive-array elements |
| `p_m`, `t_q`, `r`, `s` | UAV, target, receiver, RIS positions |
| `N_ris`, `phase_bits` | RIS elements and phase resolution |
| `P_FA`, `P_D` | false-alarm and detection probability |
| `mu0`, `mu1`, `Sigma0`, `Sigma1` | moment-matched evidence statistics |
| `delta` | `mu1 - mu0` |
| `b_i`, `epsilon_i` | report bits and BSC flip probability |
| `gamma` | reception law over scheduled reports |
| `B_report`, `B_control`, `B_total` | resource-bit ledger |
| `S_total` | total time-bandwidth symbols per frame |
| `E_PD(q, S_q)` | expected P_D of target q under schedule S_q |

## 11. Formal claims and proof sketches

### Claim 1 (G3, KKT family)

For a fixed received set, the `P_D`-optimal linear weight direction lies in
`{(Q + mu I)^{-1} a : mu >= 0}` whenever the optimum has `P_D > 0.5`.
Proof sketch: in whitened coordinates the shift is scale invariant; the KKT
stationarity condition gives `(Q + mu I)y = (R/f) a` with `mu >= 0`.

### Claim 2 (G3, set monotonicity)

At operating points with `P_D > 0.5`, the G3-optimal `P_D` is monotone in the
received set.  Proof sketch: the zero-extension of a subset-optimal weight is
feasible for a superset, and the superset family contains the global linear
optimum, so the max is nondecreasing.

### Claim 3 (G4, expectation preserves monotonicity)

`E_PD(q, S)` is monotone in `S` because every fixed-pattern `P_D` is
monotone and the expectation is a nonnegative mixture.

### Claim 4 (G4, bounded-regime submodularity)

If `Sigma1 = c Sigma0` and `Sigma0` is diagonal, `D(S)` is modular and
`P_D = Phi((sqrt(D) - z_FA)/sqrt(c))` is concave in `D` on the region
`c + D - z_FA sqrt(D) >= 0`; expectation over any reception law preserves
submodularity.  Cardinality-greedy then retains the classical `1 - 1/e`
property.

### Claim 5 (G5-U, grid-search bound)

For an `L`-Lipschitz deployment objective, a grid with spacing `h` has
deployment loss at most `L h sqrt(d)/2`.

### Claim 6 (G5-V/W, branch-and-bound certificate)

Each box upper bound `f(c) + L||h||_2` is valid for an `L`-Lipschitz
objective; splitting only boxes that can beat the current best and stopping
when `global_upper - best <= epsilon` yields an epsilon-optimality
certificate.

## 12. Open modeling boundaries

- The RIS channel is a controlled path-loss model; per-element mutual
  coupling, polarization, and waveform-level RIS responses are not modeled.
- The 1-symbol-per-bit ledger is conservative but not waveform-derived.
- Sensing OTFS-grid scaling under a fixed total time-bandwidth is still an
  open ledger path.
- Moment estimates are assumed exact in G3-G5; calibration error propagation
  is covered separately by G1-A/B.
