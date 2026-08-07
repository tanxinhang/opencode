# G18 Joint Placement-Allocation: Theory and Explicit Information

This document states what G18 optimizes, why it terminates, what information
it uses, and why it is not a black-box or neural-network surrogate.

## 1. Optimization problem

The exact system objective is

`F(s, a) = mean_seed min_q E_PD(q, S_q(s, a))`,

where `s in R^3` is the RIS position, `a in Z_+^Q` is the aperture
allocation with `sum a_q = N`, and `S_q(s, a)` is the expected-P_D greedy
schedule under the physics gain of `(s, a)`.

The search is constrained by:

- `s in [s_lo, s_hi]` (a bounded deployment box);
- `sum a_q = N`, `a_q >= 0`;
- `B_report = B_total - N * phase_bits / coherence_frames >= 0`.

## 2. Exact information used by G18

G18 does not estimate an unknown function from data.  Every quantity below
is computed from the model definition and evaluated exactly.

| Information | Source | Role in optimization |
| --- | --- | --- |
| UAV/target/receiver/RIS geometry | `uav_geometry`, `target_geometry` | path-loss constants and array phases |
| RIS path-loss constants `K_q` | `aperture_constants` | physics gain of subarray allocation |
| H0/H1 moment-matched evidence | `build_models` | expected P_D |
| Reception law | exact pattern/state probabilities | expected P_D over erasures/BSC |
| Resource identity | `B_report = B_total - N*b/C` | feasible report budget |
| Allocation move set | all zero-sum `T<=3` net vectors | exact multi-block gradient |
| Position move set | all six 0.5m coordinate neighbors | exact position gradient |
| Local optimality certificates | exact re-evaluation of `F` | termination condition |

There are no learned weights, no training set, and no hidden representation.
The "gradient" is not a stochastic estimator; it is the exact finite
difference of `F` on the searched move set.

## 3. Convergence and complexity

### Theorem 1 (finite termination)

`F(s,a)` is bounded above by 1.  G18 accepts only strict improvements, so
`F` is strictly increasing.  The allocation lattice with `sum a_q = N` is
finite, and the position grid with 0.5m steps inside the bounded box is
finite.  Therefore the alternating ascent terminates in finitely many
rounds.

### Theorem 2 (per-round complexity)

For `Q` targets and `T=3`:

- allocation certificate: at most
  `O(Q * (2T+1)^(Q-1))` exact `F` evaluations;
- position search: at most `O(3 * 2 * 3)` exact `F` evaluations per round;
- each `F` evaluation costs `O(seeds * C_greedy)`.

For the audited `Q=3` case the allocation certificate evaluates 36 candidate
vectors and the position search 18 candidates per full round, so the total
number of exact system evaluations is explicit and small.

### Theorem 3 (joint local optimality)

If G18 terminates, no `T<=3` allocation reallocation and no 0.5m
single-coordinate position move strictly increases `F`.  This is the
certificate stored in the G18 result (see `FORMAL_PROOFS.md`, Theorem 4.17).

## 4. Why this is not a neural network

A neural-network surrogate would replace the exact `F` with a trained
approximation.  It would introduce:

- training-data dependence;
- approximation error without a certificate;
- implicit feature representations;
- distribution shift risk when geometry or channel changes.

G18 uses the exact model instead.  Its "representation" is explicit:
geometry constants, physics path loss, moment-matched evidence, reception
law, resource identity, and exact finite gradients.  Every decision can be
traced to a model equation and re-evaluated exactly.

## 5. What G18 does not claim

- G18 is a local certificate, not a global optimum.
- The position certificate is 0.5m granular; a finer certificate requires
  additional evaluations.
- The allocation certificate is bounded by `T<=3`; larger `T` is a separate
  search.
- The greedy schedule is part of `F`; G18 does not alter the greedy rule.

## 6. Implications for replacing per-target ideal phase in G24

Replacing the per-target ideal phase with the G18 architecture is justified
only under the same explicit model.  The replacement should:

1. reuse the G18 final position and allocation;
2. keep the same resource identity and seed set;
3. report the joint local certificates;
4. treat the performance gap to per-target ideal phase as an explicit
   physical cost of using one shared aperture.

With those conditions, G18 provides a white-box architecture with exact
information and local guarantees, which is the theoretical basis needed
before scaling it to `Q > 3`.

## 7. Scaled extension for Q > 3

The full G18 allocation certificate evaluates all `(2T+1)^Q` net vectors,
which grows exponentially in `Q`.  For larger target counts the scaled
architecture uses:

- allocation: the derived max-min deflection water-filling of G13
  (`waterfilling_allocation`), which is polynomial and explicit;
- position: exact coordinate ascent over 2/1/0.5m steps, as in G18;
- certificates: allocation certificate is not claimed for `Q > 3`; position
  certificate is retained.

This trade-off is explicit: the scaled architecture keeps the same physical
model and exact system objective, but replaces the exhaustive multi-block
certificate with a derived allocation.  G25 evaluates this scaled
architecture at `Q in {2,4,6}` and `M/Q in {1,2,3}`.
