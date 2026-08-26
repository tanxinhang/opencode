# Unified System Model and Notation — CA-FRIDS Dual-Bus (P5 current)

> **当前状态 (2026-08-25, P5 / advice-020):** 本文档以 **当前注册主系统** 为正文：
> owner-directed evidence plane + detection-deficit task pricing + receiver-capacity
> dual bus + hard airtime admission + sequential owner detection + anytime-valid QoS
> 认证，即 `uav_otfs_isac/ca_frids.py` + `uav_otfs_isac/airtime.py` + `uav_otfs_isac/qos.py`
> 所实现并注册的模型（注册判决见 `results/p4_meta_cert.json`；P4.2b matched-QoS 前沿为
> 三状态分类）。旧时代模型（fixed geometry + one fusion center + BSC/erasure +
> expected-P_D fusion + RIS + global bit budget，即 G3-G5 链）已移入文末 **Legacy 附录**，
> 仅供历史参考，**不作为** 论文主模型。

This document centralizes the formal model used by the P5 results so the paper
can reference one consistent system model instead of scattered gate
descriptions.  All quantities are defined exactly as implemented in
`uav_otfs_isac/`.

## 0. Unified optimization formulation (P5)

The entire P5 system is **one hierarchical optimization** — not a bag of
heuristics.  It has a single system-level objective, a per-cycle scheduling
relaxation, a distributed dual solver, and a closed feasibility/QoS envelope.
The "series" of optimization objectives are a **tiered decomposition** in which
each level motivates the one below (Level 1 is the stopping-delay drift
surrogate of Level 0; Levels 2-3 are its dual decomposition; Level 4 closes the
QoS); the "pile" of constraints are exactly the stationarity, feasibility and
complementary-slackness conditions that make the deployed algorithm certified.
See §0.4 for the precise scope of the dual relationship (advice/001 P1-1).

### 0.1 Decision variables

- `x_{iqa} in {0,1}`: UAV `i` takes action `a` on target `q` (sense + report).
- `pi_i`: the local (owner-information) decision policy.
- `(A_q, B_q)`: per-target two-threshold stopping boundaries.

### 0.2 The series of optimization objectives (tiered)

**Level 0 — system objective** (the one that must ultimately be minimized):

```
min_{pi, A, B}   J = max_{q in Q} E[T_q | H1]
s.t.  C1 (QoS), C2 (capacity), C5 (information)
```

the worst-target stopping delay subject to the anytime-valid QoS spec.

**Level 1 — per-cycle scheduling relaxation** (the inner joint
sensing-communication LP that motivates the local index; the Dual-Bus does
NOT solve this LP — it runs its **distributed dual decomposition**, Level 2-3):

```
max_{x} z
s.t.  sum_{i,a} x_{iqa} g_{iqa}/(D_q+eps) >= z      (all q)  -> dual y_q
      sum_{q,a} x_{iqa} <= 1                        (all i)   (one action/UAV)
      sum_{i,q:o(q)=j,a} c_{iqa} x_{iqa} <= 1       (all j)  -> dual lambda_j
      x in [0,1]
```

This is the CONTINUOUS relaxation (`x in [0,1]`, an LP); the original is
integer (C3-C4).  The first row is the **detection-deficit coverage**
constraint (each target's reliable delivered information must clear its
residual deficit); the third row is the **receiver airtime capacity**
constraint.  The dual of the coverage row is the price weight `y_q`; the
deployed **task price** packages it with the deficit normalization,
`pi_q = y_q/(D_q+eps)` (exactly the code's definition).  The strong-dual
stationary point of this LP is the local index of Level 2; integrality is
restored by the DISCRETE per-UAV best response (C3) plus the hard pathwise
admission (C2).

**Level 2 — distributed local best response** (each UAV solves its own):

```
(i,q,a)^* = argmax_{q,a} [ pi_q g_{iqa} - lambda_{o(q)} c_{iqa} ]   (idle = 0)
```

with `pi_q = y_q/(D_q+eps)`, i.e. the index is `y_q g_{iqa}/(D_q+eps) -
lambda c` (sensing and communication enter the SAME score, not stitched) and
the idle action `score 0` (without it a purely additive price is a no-op,
Lemma 4.99).  In the lambda-free **normalization-free** form (B0-lite), the
OWNER packages its residual deficit into a single owner-local sufficient
control scalar `psi_q = theta_q - log(D_q+eps)` (`theta_q` the owner-local
log-price) and broadcasts the QUANTIZED `psi_q`:

```
q_i^* = argmax_q [ psi_q + log g_{iq} ],
```

(the common positive scale of the weights cancels, so the coverage-dual
`y_q` and the owner's private `D_q` are replaced by the owner-generated
`psi_q` — C5, no global reduction AND no uncharged global `D_q` channel,
advice/001 P0-3).

**Level 3 — price dynamics** (the dual update):

- deficit pricing (mirror descent): `y_q <- y_q exp(-mu r_q)/Z` (normalized)
  or the owner-local `theta_q <- theta_q - mu r_q` (normalization-free, no
  global reduction);
- capacity dual ascent: `lambda_j <- clamp(lambda_j + mu_c (rho_j - 1), 0, lam_cap)`.

**Level 4 — threshold / QoS frontier:** choose `(A_q, B_q)` so that C1 holds
with anytime-valid certification while minimizing `J` (three-state frontier:
CERTIFIED FEASIBLE / UNRESOLVED / CERTIFIED INFEASIBLE).

### 0.3 The constraints (a pile)

| id | constraint | role |
| --- | --- | --- |
| C1 | `P_FA,q <= alpha`, `P_MD,q <= beta` (all q), anytime-valid certified | QoS spec |
| C2 | `sum_{i,q:o(q)=j} x_{iq} tau_{ij} <= T_air` (all j), enforced PATHWISE by hard admission; `lambda_j` is the dual of the RELAXED offered-load (EMA) constraint it steers | receiver capacity |
| C3 | `sum_{q,a} x_{iqa} <= 1` (all i) | one action per UAV |
| C4 | `x_{iqa} in {0,1}`; idle score 0 allowed | feasibility / idle |
| C5 | `a_{i,t} = pi_i(I_{i,t})`: each local decision uses only local info + broadcast prices.  Normalized form needs ONE global scalar `Z` (spanning-tree/gossip); the normalization-free B0-lite removes even `Z` (owner-local `theta_q`, no global sum/max reduction) | information / distributed |
| C6 | broadcast `psi_q` quantized over `[psi_lo, psi_hi]`; action preserved iff `m_i > 2*eps_psi` while `psi` stays IN range (saturation gated) | finite-bit control |
| C7 | `L_q >= A_q => H1`, `L_q <= B_q => H0` | sequential detection |
| C8 | `lambda_j (rho_j - 1) = 0`, `lambda_j >= 0` (limit of the dual-ascent update) | capacity-regime KKT |

> **C2 two-layer note:** the LP (Level 1) relaxes C2 to the *expected/EMA*
> offered-load constraint whose dual `lambda_j` STEERS the best response;
> the deployed system then HARD-ADMITS a density-ranked subset under the
> pathwise budget `sum tau/T_air <= 1` (the fuse).  So `lambda_j` is the dual
> of the relaxed constraint (it sees the offered load that would overload),
> while the hard admission enforces the exact pathwise constraint every
> cycle.  C8 is the complementary-slackness LIMIT that the dual-ascent update
> `lambda_j <- clamp(lambda_j + mu_c (rho_j-1), 0, lam_cap)` approximates at
> convergence.

### 0.4 The unification (why the tiers are not ad hoc)

- Level 0 is the **true** objective; Level 1 is its **per-cycle relaxation**;
  Levels 2-3 are the **distributed dual solver** (local best response + price
  ascent); Level 4 **closes the QoS**.  The tiers form a single coherent
  optimization, not a sequence of disconnected objectives.
- **Scope of the "dual" claim (advice/001 P1-1):** Level 1 is NOT the exact
  Lagrangian of the Level-0 sequential-stopping problem (no proof that the
  stopping-delay Lagrangian is the per-cycle max-min).  The precise statement
  is: Level 1 is the **stopping-delay-oriented drift surrogate** -- under H1,
  ``E[Delta L_q | F_t] = sum x p^adm s I+`` is the expected
  detection-information drift, and with the residual-detection potential
  ``V(t) = sum_q y_q(t) log(D_q(t)+eps)`` the per-cycle objective
  ``max_x sum_q y_q g_q(x)/(D_q+eps)`` equals maximizing the expected decrease
  rate of ``V`` (Claim 0.4).  The dual of THAT drift surrogate motivates the
  local index and the prices -- not an exact dual of Level 0.
- The constraints are exactly the exactness conditions:
  - **C8** is the KKT complementary slackness that makes the `lambda` activation
    principled -- the capacity-slack / capacity-binding phase diagram is the
    solution of the Level-1 LP, not an empirical `(8,4)`-vs-`(16,8)` heuristic;
  - **C6** is the finite-bit action-error certificate that makes the deployed
    norm-free form exact-up-to-a-certified-approximation (quantize the
    owner-local `psi_q`, not the scale-dependent `pi_q`);
  - **C5** is the information constraint that the owner-local `psi_q` form
    satisfies with **no global reduction** and **no uncharged global `D_q`
    channel** (advice/001 P0-3).

### 0.5 What the advice/020 + advice/001 modifications "sublimate" into

1. **Normalization-free ψ-bus (owner-local `theta_q` -> broadcast
   `psi_q = theta_q - log(D_q+eps)`)** = the Level-2 index that satisfies C5
   (no global reduction, no uncharged global `D_q` channel) and makes C6 clean
   (quantize `psi_q`, not `pi_q`) — with the certified action-error bound of
   Claim 0.2 (valid only while `psi` stays in range; saturation gated,
   Claim 0.4).
2. **Capacity-regime gate** = C8 turned from an empirical scale observation
   into a principled complementary-slackness phase diagram (Claim 0.3), with
   the λ cap-hit / dual-residual / time-average diagnostics (advice/001 P1-2).
3. **matched-QoS gate** = C1 as the freeze condition (Level 4), now requiring
   HELD-OUT anytime-valid QoS PASS on both schedulers (advice/001 P0-1): the
   minimal B0-core is not certified slower than full C at matched certified
   QoS.

Hence the whole system reduces to: *minimize the worst-target sequential
stopping delay (Level 0) subject to the anytime-valid QoS (C1), the hard
receiver airtime capacity (C2), the per-UAV action and idle feasibility
(C3-C4), the distributed information constraint (C5), the finite-bit control
certificate (C6), the sequential detection rule (C7), and the KKT
capacity-regime condition (C8); solve it by the dual decomposition of the
per-cycle joint drift-surrogate LP (Level 1-3, the owner broadcasting
`psi_q`) and close the QoS threshold frontier (Level 4).*

## 1. Scope and assumptions

- `K` sensing UAVs, `Q` candidate targets, each target `q` assigned a fixed
  OWNER `o(q)` (cyclic: `o(q) = q mod K`).  There is NO single fusion center.
- Each UAV senses locally; each evidence token is reported ONLY to the target
  OWNER (owner-directed evidence plane, not full mesh).
- The system performs SEQUENTIAL detection: the owner accumulates its belief
  until a two-threshold stop rule fires, then declares H1/H0.
- The control plane is a real communication cost: `Q` task prices plus `K`
  receiver airtime prices (plus one global-simplex scalar in the normalized
  form) are broadcast every cycle.
- The operating point is fixed false-alarm `P_FA = alpha` and miss
  `P_MD = beta`; the primary metric is the worst-target stopping delay
  `E[T_q | H1]` (pooled, risk-adjusted), and QoS is certified with anytime-valid
  bounds.

## 2. Local sensing and reliable detection information

Each UAV `i` hosts one report kernel per target `q` per sensing-power lever.
The kernel is a quantized observation with H0/H1 distributions and an LLR
per observation outcome.  The scheduler-believed reliable post-communication
detection information of the deployed action is

```
g_{iqa} = I^+_{iqa} * rel_{i,o(q)},
rel_{i,j} = 1 if i == j else s_{i,j},
```

where `I^+_{iqa}` is the sensing information (post-communication reliable
information) of action `a` and `s_{i,j}` is the U2U link success probability
(owner-local evidence has `rel = 1`, so it is never taxed by a communication
price it does not pay).

## 3. Owner belief and sequential detection

The owner `o(q)` of target `q` accumulates the LLR of delivered evidence:

```
L_q(t+1) = L_q(t) + sum_i delta_{iq}(t) * ell_{iq}(t),
```

with `delta_{iq}` the delivered-token indicator (physical Bernoulli +
airtime-thinned) and `ell_{iq}` the quantized observation LLR.  The stopping
rule on the OWNER belief is the unchanged two-threshold rule

```
L_q >= A_q  => H1,
L_q <= B_q  => H0.
```

The residual detection deficit is

```
D_q(t) = [ A_q - L_q(t) ]_+.
```

## 4. Task-price plane (detection-deficit pricing)

Each owner maintains its own task price.  In the global-simplex form the
price is the entropic mirror descent

```
y_q^+ = y_q * exp(-mu * r_q) / Z,
r_q = S_q / (D_q + eps),
```

over the undecided targets, with `Z` the global scalar normalizer (a
spanning-tree/gossip reduction, the ONLY networked quantity).  In the
NORMALIZATION-FREE form (advice/019-020) the owner keeps an OWNER-LOCAL
log-price `theta_q` updated WITHOUT any global reduction:

```
theta_q(t+1) = theta_q(t) - mu * r_q(t),
```

and broadcasts the QUANTIZED `theta_q`; the local rule is

```
q_i^* = argmax_q [ theta_q + log g_{iq} - log(D_q + eps) ].
```

The broadcast price `pi_q` (normalized) or `psi_q` (norm-free, the
deficit-embedded log price) spans a pre-registered range and is quantized
with `pi_bits` / `psi_bits`.

## 5. Receiver-capacity plane (hard airtime admission)

The physical per-token airtime is `tau_{ij} = b_tok / R_{ij}` seconds, with
`R_{ij} = W_c log2(1 + gamma_{ij})` the Shannon link rate (an UPPER BOUND,
never a claimed throughput).  The per-receiver budget is `T_air`, so a report
consumes the fractional budget `c_{ij} = tau_{ij} / T_air`.

The receiver constraint is

```
sum_{i,q:o(q)=j} x_{iq} tau_{ij} <= T_air,
```

with dual price `lambda_j >= 0`.  KKT complementary slackness
`lambda_j (rho_j - 1) = 0` gives the capacity-regime phase diagram:
capacity-slack `rho_j < 1 => lambda_j = 0` (the B0-lite regime), capacity-
binding `rho_j ~ 1 => lambda_j > 0`.  The deployed system HARD-ADMITS a
density-ranked subset per receiver under the pathwise budget (the fuse), while
`lambda_j` is the dual-ascent STEERING price on the offered-load EMA.

`T_air` is either explicit, or derived from the full-mesh always-report ratio
`rho_target`, or from the owner-directed ratio `rho_owner` (advice/020
section 5) so the capacity regime is controlled across scales.

## 6. Local best response and idle gate

Each UAV computes the LOCAL best response over its own kernels and the
undecided targets,

```
(i, q, a)^* = argmax [ pi_q g_{iqa} - lambda_{o(q)} c_{iqa} ],
```

with the idle option `score 0` (do not report).  Without the idle action a
purely additive price is a no-op (Lemma 4.99); with it, the price decides
report vs silence AND reorders the sensing target.  In the norm-free form the
argmax is over `[ theta_q + log g_{iq} - log(D_q+eps) ]`.

## 7. Resource / control-bus ledger

Only ENABLED price planes are charged (advice/018 section 6):

```
B_ctrl = 1_task ( Q*pi_bits [+ Z]  |  Q*psi_bits if norm_free )
         + 1_lambda K*lam_bits
```

per cycle.  B0-lite (norm-free, lambda OFF) bills `Q*psi_bits` bits/cycle
(no global `Z`, no lambda bus, and no separate `D_q` channel — the owner
broadcasts the deficit-embedded `psi_q`, advice/001 P0-3) — at `(16,8),
b=10` this is 80 bits/cycle vs 250 for full C, a ~3x control-plane reduction
with NO global reduction on the control plane.

> **P1-4 physicalization note (advice/001 P1-4):** `B_ctrl` counts LOGICAL
> bits on an assumed **orthogonal, reliable control channel** — it does NOT
> charge the control plane's own airtime, link loss or staleness.  The
> evidence ledger is physical (Shannon-rate `tau_ij` + hard admission), but
> the control ledger is logical.  A fully physical control-plane ledger
> (control tokens over the same U2U physical channel, with their own
> airtime/outage/staleness) is an OPEN modeling boundary, not claimed here;
> the 80-vs-250 comparison is a logical-bits comparison only.

## 8. Performance metrics

- Pooled worst-target stopping delay `J = max_q sum_b S_bq / sum_b N_bq`
  (per-target H1 delay sums and counts pooled over blocks).
- Risk-aware delay `J_risk`: under H1, a run that declared H1 keeps its stop
  time; an MD error is charged `T_max` — a lower `J` can never be bought by
  earlier WRONG H0 stops.
- Anytime-valid QoS certificate: simultaneous Clopper-Pearson bounds on
  `P_FA` / `P_MD` across the family, reported per arm (three-state frontier:
  `CERTIFIED FEASIBLE / UNRESOLVED / CERTIFIED INFEASIBLE`).
- Paired per-block bootstrap 95% CIs (and, for cross-scenario claims, the
  hierarchical cell->seed->block bootstrap + per-cell sign consistency,
  advice/020 section 12).

## 9. Notation table

| Symbol | Meaning |
| --- | --- |
| `K`, `Q` | sensing UAVs, target hypotheses |
| `o(q)` | fixed owner of target `q` |
| `g_{iqa}` | reliable post-communication detection information of action |
| `rel_{i,j}` | owner-direct delivery success (1 on the diagonal) |
| `ell_{iq}` | quantized observation LLR |
| `L_q`, `A_q`, `B_q` | owner belief and upper/lower thresholds |
| `D_q` | residual detection deficit `[A_q - L_q]_+` |
| `pi_q` / `theta_q` | task price / owner-local log-price |
| `mu` | mirror-descent step |
| `tau_{ij}`, `T_air`, `c_{ij}` | token airtime, receiver budget, fractional budget |
| `lambda_j` | receiver airtime price (dual of capacity) |
| `rho` | receiver offered-load ratio `L/T_air` |
| `alpha`, `beta` | FA / MD QoS spec |
| `B_ctrl` | control-bus bits per cycle |
| `J`, `J_risk` | pooled / risk-aware worst-target stopping delay |

## 10. Formal claims and proof sketches (P5 current)

### Claim A (normalization-free scale invariance)

For `lambda = 0`, the local rule is unchanged by any common positive scale of
the price weights: `argmax_q (w_q g_{iq}/(D_q+eps)) = argmax_q [ theta_q +
log g_{iq} - log(D_q+eps) ]` where `theta_q = log w_q`.  Hence the global
sum-normalizer `Z` (and any global `rbar`) cancels and is neither computed nor
broadcast.  BEFORE finite-rate control quantization this is EXACTLY
scale-equivalent to the normalized B0; the deployed finite-bit implementation
is an APPROXIMATION whose action distortion must be certified (Claim B).

### Claim B (finite-bit action-error bound)

Only the owner-generated `psi_q = theta_q - log(D_q+eps)` is quantized over
the registered range `[psi_lo, psi_hi]` (mid-tread, `psi_lo` printed
exactly) with `bits`; `log g_{iq}` is an exact local quantity and the
owner's private `D_q` is never exposed to non-owner UAVs.  Let `m_i` be the
ideal top-1 vs top-2 margin in log space.  The action is preserved
whenever `m_i > 2*eps_psi`, `eps_psi = (psi_hi - psi_lo)/2^bits`
(`norm_free_action_error_bound`) AND `psi` stays IN range (the clipping
error is NOT covered -- saturation is gated via `psi_sat_rate`,
advice/001 P0-4).  This is a certified APPROXIMATION, NOT a strict
policy-equivalent reparameterization of the finite-bit normalized B0
(advice/020 section 2-3).

### Claim C (capacity-regime phase diagram)

With `lambda_j` the dual of the receiver constraint, complementary slackness
gives `rho_j < 1 => lambda_j = 0` (capacity-slack -> B0-lite) and
`rho_j ~ 1 => lambda_j > 0` (capacity-binding -> full CA).  This is the
KKT-grounded statement behind the `(8,4)` vs `(16,8)` empirical difference:
it is a CONGESTION-REGIME transition, not an intrinsic scale dependence
(advice/020 section 6-7).  A scale comparison must control the effective
receiver load (fixed `T_air` or matched `rho_owner`) and use a nested master
scenario (advice/020 section 8).

## 11. Open modeling boundaries

- The physical link rate is the Shannon upper bound; mutual coupling,
  polarization and waveform-level RIS responses are NOT in the current main
  model (they belong to the legacy RIS chain).
- The owner-directed load estimate assumes balanced reporting; a per-cycle
  realization may differ from the balanced `rho_owner` used for budgeting.
- Hierarchical bootstrap quantifies scenario variation but the pool is still a
  fixed registered-grid estimand unless the cell set is sampled as the
  population (advice/020 section 12).

---

# Legacy Appendix (G3-G5, historical — NOT the paper main model)

The following was the earlier paper-facing model (fixed geometry + one fusion
center + BSC/erasure + expected-P_D fusion + RIS + global bit budget).  It is
retained ONLY as history; the current model is the CA-FRIDS Dual-Bus system
above.

## L1. Legacy geometry / RIS channel

- One `L`-element receive array / fusion center `r`, `M` transmitting UAVs,
  `Q` targets, RIS at `s`.
- RIS additive-power gain `gain_mq = 1 + (strength_q * array_gain(theta_q))^2`;
  physics-based two-way bistatic `P_dir = 1/(R_dir_tx^2 R_dir_rx^2)` and
  three-leg cascaded `P_ris = N_ris^2 array_gain^2 aperture_scale/
  (R_1^2 R_2^2 R_3^2)`; `gain = 1 + P_ris/P_dir`; optional weak-target
  `direct_blockage`.

## L2. Legacy reporting / fusion / selection

- Report payload `p_i` bits, `b_i = p_i + 2` overhead, BSC flip `epsilon_i`,
  erasure reception law `gamma`.
- Deflection-optimal linear score `w = Sigma0_R^{-1} delta`; KKT family
  `w(mu) = L^{-T}(Q + mu I)^{-1}L^{-1} delta`.
- Global-bit-budget selection maximizing expected `P_D`
  `E_PD(q, S_q) = E_gamma[ P_D(owner union received(S_q, gamma)) ]`; G4
  two-stage greedy; G5-U grid-search and G5-V/W branch-and-bound Lipschitz
  deployment certificates.

## L3. Legacy claims (G3-G5)

- G3 KKT family / set monotonicity; G4 expectation-preserved monotonicity and
  bounded-regime submodularity (Sigma1 = c Sigma0, Sigma0 diagonal) with the
  `1 - 1/e` greedy property; G5 grid/branch-and-bound Lipschitz bounds.

These legacy mechanisms are superseded by the P5 owner-directed evidence +
deficit-pricing + capacity-dual + anytime-valid-QoS system.  Any residual
references to `expected-P_D`, central fusion, BSC/erasure, RIS, or the global
bit budget in the G3-G5 appendices are historical.
