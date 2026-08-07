# Fundamental Principle: Information Budget Governs Detection

## 1. The essence

The system is not a collection of isolated gates.  Its detection performance
is governed by one abstract quantity:

``J = sensing information + communication information - erasure/overhead loss``.

All audited architectures are projections of the same centralized likelihood
ratio onto different resource budgets:

- centralized soft fusion keeps the soft information but must pay report
  bits;
- 1-bit hard decisions keep only the binary decision information but pay one
  bit each;
- peer consensus keeps all local decisions and pays zero report bits, but
  loses soft information;
- RIS increases sensing information through `N^2` array gain while consuming
  control bits.

## 2. Information measures

For moment-matched Gaussian evidence:

- full-information deflection:
  `D_full = delta^T Sigma0^{-1} delta`;
- soft-schedule deflection:
  `D_soft = delta_S^T Sigma0,SS^{-1} delta_S`;
- 1-bit decision information:
  `I_i = KL( p1_i || p0_i )`;
- consensus information:
  `I_peer = sum_i I_i`.

The normalized budget is

`rho = J / D_full`.

## 3. What the gates show

- G44: within the soft family, worst P_D increases monotonically with
  `rho`; soft fusion reaches 0.933 at `rho=0.946`.
- Hard/consensus need an optimized threshold before their `rho` aligns with
  P_D; the raw KL at fixed local P_FA is a lower-order predictor.
- G46: the raw deflection coordinate is not a performance law.  Inverting
  the Gaussian detection relation gives `rho_exact`; soft raw rho
  overestimates it by 2.38-2.78x, while `rho_exact` orders soft P_D
  monotonically (0.774 at 0.205 -> 0.933 at 0.351).
- G47: when `rho_peer > rho_soft` in the scarce-report regime, the exact
  architecture switch selects peer consensus and raises worst P_D from
  0.774/0.824 to 0.881 at B=8/12; at B>=16 it returns to centralized soft.
- G48: per-target switching `max(soft_q, peer_q)` is never worse than the
  global switch and adds +0.49/+1.55/+1.55pp worst P_D at B=12/16/20.
- G49: soft-report bits freed by peer targets are reallocated to centralized
  targets with exact expected-P_D marginals, adding +0.75pp at B=16/20 over
  G48 and +1.55/+0.85pp at B=28/40 without changing the report budget.
- G50: a limiting peer target may switch back to centralized soft only when
  the switch strictly raises the worst P_D; this adds +0.39pp at B=12 over
  G48 (0.8858 -> 0.8898).
- G51: under AR(1) stochastic mobility and RIS latency, ideal mode ascent
  reaches 0.852 worst-over-time P_D and 90.625% QoS, versus 0.847 and
  81.25% for target-wise switching.
- G52: AR(1) conditional-mean RIS prediction raises latency-1
  worst-over-time P_D from 0.7217 to 0.7283 (+0.65pp) and QoS from 43.75%
  to 46.875%.
- G53: h-step MMSE prediction gains +0.65/+3.24/+5.24pp over stale-phase
  for h=1/2/3, with error covariance scale `1 - rho^{2h}` growing from 0.36
  to 0.74; exact per-frame horizon selection adds +0.86pp over the best
  fixed MMSE, and hysteresis delta=0.02 halves switches with loss bounded by
  delta; per-switch costs of 1/3/6 bits select delta 0.00/0.03/0.05 under a
  6-bit control budget.
- G54 (negative): covariance-aware expected-gain phase is monotone in its
  surrogate but degrades exact worst P_D from 0.7200 to 0.6557 under
  quantization at h=3; MMSE phase is kept.
- Exact Poisson-binomial parity (G43) is the correct feasibility check for
  consensus, and it starts at M=6.
- Low report budget (G40) is exactly the regime where soft `rho` collapses
  while consensus `rho` remains nonzero.

## 4. Fundamental trade-off

Let `c` be the cost of one soft report.  If `B < c`, centralized soft fusion
is constrained to `rho_soft ~ owner-only`, while peer consensus can achieve
`rho_peer = sum_i KL_i`.  Therefore:

- distributed wins when `B` is small enough that soft information cannot be
  transmitted;
- centralized wins when `B` is large enough to carry soft information;
- the crossover is determined by `B`, `c`, local SNR, and `M`, and can be
  computed exactly with Poisson-binomial tails.

The correct information coordinate is not `J_method / D_full` computed from
raw deflection/KL alone.  Because quantization and correlation change the
effective Gaussian moments, the P_D-consistent budget is

`rho_exact = (sqrt(c) Phi^{-1}(P_D) + z_FA)^2 / D_full`

under the calibrated `Sigma1 = c Sigma0` relation.  Raw `rho` remains a
convenient first-order surrogate but must be audited against `rho_exact`
before it is used to compare architectures.

The architecture switch follows directly from the budget coordinate: use the
architecture with the larger exact `rho` (equivalently larger calibrated
P_D).  Because each branch meets the same global false-alarm constraint,
selecting the higher-P_D branch is a feasible detector with no additional
communication budget.  The practical `report_budget < 10` crossover is a
design parameter estimated from the audited profile, not a law.

## 5. Implication for architecture

RIS is not a separate knob: it multiplies `D_full` through evidence SNR.
The control overhead consumes part of `B`, so the architecture question is
always:

``does the added sensing information justify the consumed control bits?``

This is the essential optimization that G12-G38 implement in increasing
detail.
