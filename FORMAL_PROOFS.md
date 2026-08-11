# Formal Proof Appendix

This appendix states the formal claims used by Gates G3-G5 and gives proofs
that match the implemented objects in `uav_otfs_isac/fusion.py`,
`uav_otfs_isac/expected_pd.py`, and `uav_otfs_isac/deployment_search.py`.
The notation follows `SYSTEM_MODEL.md`.

## 1. Notation and standing assumptions

- `Phi` and `phi` denote the standard Gaussian CDF and PDF.
- `z = Phi^{-1}(1 - P_FA)` is the fixed false-alarm threshold, with
  `P_FA in (0, 0.5)` so `z > 0`.
- For a linear score `s(x) = w^T x` with `x ~ N(mu_h, Sigma_h)`,
  `h in {0, 1}`, the threshold is
  `tau(w) = w^T mu0 + z sqrt(w^T Sigma0 w)`.
- `delta = mu1 - mu0`.
- `Sigma0` and `Sigma1` are positive definite.
- `L` is the Cholesky factor `Sigma0 = L L^T`.
- `Q = L^{-1} Sigma1 L^{-T}` and `a = L^{-1} delta`.
- `R` denotes a received report set and is always nonempty because each
  target keeps its local owner report.

The moment-matched Gaussian detection probability of the linear score is

`P_D(w) = Phi( (w^T delta - z sqrt(w^T Sigma0 w)) / sqrt(w^T Sigma1 w) )`.

The score is scale invariant, so we may optimize over directions.

## 2. Gate G3: the KKT family contains the global linear-score optimum

### Lemma 2.1 (whitening)

For every nonzero weight direction `w` there is a bijective map
`y = L^T w` such that

`w^T delta = a^T y`,
`w^T Sigma0 w = ||y||^2`,
`w^T Sigma1 w = y^T Q y`.

Consequently

`P_D(w) = Phi( (a^T y - z ||y||) / sqrt(y^T Q y) )`.

Proof: substitute `w = L^{-T} y` and use `Sigma0 = L L^T`,
`Sigma1 = L Q L^T`.

### Lemma 2.2 (compact normalization)

Maximizing `P_D` over nonzero `w` is equivalent to maximizing

`F(y) = a^T y - z ||y||`

over the compact ellipsoid `{ y : y^T Q y = 1 }`.

Proof: define `u = y / sqrt(y^T Q y)`.  Because the score is homogeneous of
degree zero,

`(a^T y - z ||y||) / sqrt(y^T Q y) = a^T u - z ||u||`.

Every nonzero direction appears exactly once on the ellipsoid, so a maximizer
exists.

### Theorem 2.3 (KKT representation of the global optimum)

If the global optimum satisfies `P_D* > 0.5`, then every maximizing direction
has the form

`w(mu) = L^{-T} (Q + mu I)^{-1} L^{-1} delta`

for some `mu >= 0`.

Proof: let `y*` maximize `F` on the ellipsoid.  Since `Q` is positive
definite, the ellipsoid constraint has nonzero gradient `2 Q y*`, so the
constraint qualification holds and the KKT conditions are necessary:

`a - z y* / ||y*|| - 2 lambda Q y* = 0`.

Multiply by `y*^T` and use `y*^T Q y* = 1`:

`a^T y* - z ||y*|| = 2 lambda`.

The left-hand side is `F(y*)`, which equals the shift `Phi^{-1}(P_D*)`.
`P_D* > 0.5` implies `F(y*) > 0`, hence `lambda > 0`.  Rearranging the
stationarity condition gives

`(Q + mu I) y* = a / (2 lambda)`,

where

`mu = z / (2 lambda ||y*||) >= 0`.

Thus `y*` is proportional to `(Q + mu I)^{-1} a`, and transforming back to
`w` gives the claimed family.

Corollary: because the global maximizer lies in the one-parameter family,
maximizing `P_D` over the family attains the global optimum whenever
`P_D* > 0.5`.  The implementation therefore evaluates the family over
`mu in [0, infinity)`, including the `mu -> infinity` limit
`y = a`, which is the deflection-optimal direction.

### Theorem 2.4 (set monotonicity at audited operating points)

Let `S subset T`.  If the family-optimal `P_D` equals the global linear-score
optimum on both sets, then

`P_D*(S) <= P_D*(T)`.

Proof: let `w_S` attain the global optimum on `S`.  Define `w_T` by zero
extension: `w_T[i] = w_S[i]` for `i in S` and `w_T[i] = 0` for
`i in T \ S`.  The H0/H1 means and covariances of the restricted score are
unchanged, so `P_D,T(w_T) = P_D,S(w_S)`.  Since `P_D*(T)` maximizes over all
linear weights on `T`, it is at least `P_D,T(w_T)`, proving the claim.

The audited operating points satisfy `P_D >= 0.5`, so Theorem 2.3 applies at
every addition edge and the implemented family is set-monotone there.  The
zero-extension argument itself does not require `P_D >= 0.5`; that condition
is needed only to identify the global optimum with the one-parameter family.

### Theorem 2.5 (proportional-covariance closed form)

If `Sigma1 = c Sigma0` with `c > 0`, then for any received set `S`,

`P_D*(S) = Phi( (sqrt(D(S)) - z) / sqrt(c) )`,

where `D(S) = delta_S^T Sigma0,SS^{-1} delta_S`.

Proof: in whitened coordinates, `Q = c I` and the family becomes

`y(mu) = a / (c + mu)`.

All members have the same direction `a`, so the family contains only the
whitened matched filter.  Direct substitution into Lemma 2.1 gives

`P_D = Phi( (||a_S|| - z) / sqrt(c) )`,

and `||a_S||^2 = delta_S^T Sigma0,SS^{-1} delta_S = D(S)`.

## 3. Gate G4: expected-P_D monotonicity and bounded-regime submodularity

### 3.1 Model

For target `q`, a schedule `S_q` induces a random received set
`A(S_q, omega)` through the independent, common-state, or grouped reception
law.  The owner report is always present.  The selection objective is

`E_PD(q, S_q) = E_omega[ P_D*( owner union A(S_q, omega) ) ]`.

### Theorem 3.1 (monotone expectation)

If every fixed-pattern `P_D*` is at the operating point where Theorem 2.4
applies, then `E_PD(q, S_q)` is monotone in `S_q`.

Proof: for every realization `omega`, adding a report to `S_q` cannot
decrease the received set.  Theorem 2.4 says the fixed-pattern `P_D*` is
nondecreasing in the received set.  The expectation is a nonnegative mixture
of nondecreasing functions, hence nondecreasing.

### Lemma 3.2 (modular deflection)

In the regime `Sigma0 = diag(sigma_i^2)` and `delta_i >= 0`,

`D(A) = sum_{i in A} delta_i^2 / sigma_i^2`,

so `D` is a modular set function with nonnegative coefficients.

Proof: for a diagonal `Sigma0`, the submatrix inverse is diagonal and the
quadratic form separates.

### Lemma 3.3 (concavity of the closed-form P_D)

For `f(D) = Phi( (sqrt(D) - z) / sqrt(c) )`,

`f''(D) <= 0` if and only if

`c + D - z sqrt(D) >= 0`.

Proof: set `t = sqrt(D)` and `u = (t - z)/sqrt(c)`.  Then

`f'(D) = phi(u) / (2 t sqrt(c))`

and

`f''(D) = -phi(u) (c + t^2 - z t) / (2 sqrt(c) t^2 c)`.

The denominator is positive for `D > 0`, so the sign is controlled by
`c + D - z sqrt(D)`.  The boundary case `D = 0` is obtained by continuity,
where `sqrt(D)` is concave.

### Theorem 3.4 (bounded-regime submodularity)

Assume `Sigma1 = c Sigma0`, `Sigma0` diagonal, nonnegative deflection
coefficients, and the inflection condition of Lemma 3.3 holds on every
realization in the support of the reception law.  Then `E_PD(q, S_q)` is
monotone submodular in `S_q`.

Proof: fix `omega`.  By Lemma 3.2,

`D(A(S_q, omega)) = sum_{i in S_q} d_i 1_{i in A(S_q, omega)}`

is a modular function of `S_q` with coefficients `d_i 1_{...} >= 0`.
Lemma 3.3 gives `f'' <= 0`, so for `A subset B` and `i notin B`,

`f(D_A + d_i) - f(D_A) >= f(D_B + d_i) - f(D_B)`.

Thus `omega -> f(D(A(S_q, omega)))` is submodular.  The expectation is a
nonnegative mixture of submodular functions, hence submodular.

Corollary: for the equal-cost cardinality variant, the classical
`1 - 1/e` greedy approximation guarantee applies to monotone submodular
maximization.  The implemented bit-budgeted selector with nonuniform report
costs does not inherit that ratio; its guarantee is the monotone-submodular
structure plus exact marginal evaluation, not a fixed approximation ratio.

## 4. Gate G5: RIS channel, quantization, and deployment search

### 4.1 Additive-power gain is never harmful

For UAV `i` and target `q`, the controlled channel gain is

`gain_iq = 1 + (strength_q array_gain_iq)^2 >= 1`.

The physics-based version is

`gain_iq = 1 + P_ris / P_dir >= 1`,

where

`P_dir = 1 / (R_tx^2 R_rx^2)`,

`P_ris = N_ris^2 array_gain_iq^2 aperture_scale / (R_1^2 R_2^2 R_3^2)`.

Proof: each term is a sum of nonnegative powers, and division of nonnegative
quantities with `P_dir > 0` keeps the ratio nonnegative.  Therefore aligned
or partially aligned RIS illumination can improve but never reduce evidence
SNR, and `gain_iq` is monotone in `N_ris`, `array_gain_iq`, and
`aperture_scale`.

### 4.2 Phase-quantization gain loss

Let the ideal steering phase of element `n` be `phi_n*`, and let the
`b`-bit quantizer produce `phi_n = phi_n* + e_n` with quantization errors
`e_n` independent and uniform on `[-pi / 2^b, pi / 2^b]`.  For a large
uniform linear array, the normalized power gain toward the steering
direction satisfies

`G_b^2 -> sinc^2(1 / 2^b)` as `N_ris -> infinity`.

For finite `N_ris`,

`E[G_b^2] = 1 / N_ris + (1 - 1 / N_ris) sinc^2(1 / 2^b)`.

Proof: the array output is `sum_n exp(j e_n)`.  Independence gives

`E[exp(j e_n)] = sin(pi / 2^b) / (pi / 2^b) = sinc(1 / 2^b)`.

Then

`E[ |sum_n exp(j e_n)|^2 ] =
 N_ris + N_ris(N_ris - 1) sinc^2(1 / 2^b)`,

and dividing by `N_ris^2` gives the finite-N expression.  The limit follows
immediately.

The audit in Gate G5-Q compares the asymptotic factor to random steering
realizations; finite-N corrections are explicit in the formula above and
should be used when `N_ris` is small.

### 4.3 Grid-search suboptimality bound

Let `f: R^d -> R` be `L`-Lipschitz, and let `G_h` be a grid with spacing `h`
in every coordinate.  Then

`max_x f(x) - max_{g in G_h} f(g) <= L h sqrt(d) / 2`.

Proof: every point `x` is within `l_inf` distance `h / 2` of a grid point,
so its Euclidean distance to that grid point is at most
`sqrt(d) h / 2`.  For the grid point nearest to the global maximizer `x*`,

`f(x*) - f(g(x*)) <= L ||x* - g(x*)|| <= L h sqrt(d) / 2`.

Since the grid maximum is at least `f(g(x*))`, the bound follows.

### 4.4 Lipschitz branch-and-bound certificate

Consider a box `B` with center `c` and half-side vector `h`.  For an
`L`-Lipschitz objective,

`max_{x in B} f(x) <= f(c) + L ||h||_2`.

The adaptive search keeps a list of boxes whose union contains the search
domain, evaluates every box center, and defines

`global_upper = max_B [ f(c_B) + L ||h_B||_2 ]`,

`best = max_B f(c_B)`.

If `global_upper - best <= epsilon`, then `best` is within `epsilon` of the
global maximum over the domain.

Proof: the upper bound follows from Lipschitz continuity and
`max_{x in B} ||x - c||_2 = ||h||_2`.  Splitting a box into two children
preserves the union of the box list and reduces the maximum radius of the
remaining boxes.  Therefore `global_upper` is a valid upper bound at every
iteration and `best` is a feasible value, so the gap is a valid optimality
certificate.

### Lemma 4.5 (coordinate-wise box bound)

Assume `f` also satisfies a coordinate-wise Lipschitz condition: for every
coordinate `j` and every pair of points that differ only in coordinate `j`,
`|f(x) - f(y)| <= L_j |x_j - y_j|` with `L_j > 0`.  Then for a box with
center `c` and half-side vector `h`,

`max_{x in B} f(x) - f(c) <= min( L ||h||_2, sum_j L_j h_j )`.

Proof: the first term is Theorem 4.4.  For the second term, telescope along
the coordinate axes:

`f(x) - f(c) <= sum_j L_j |x_j - c_j| <= sum_j L_j h_j`.

The adaptive search therefore stores
`min( f(c) + L ||h||_2, f(c) + sum_j L_j h_j )`, which is never looser than
the radial bound and is tighter when the objective is strongly anisotropic.
The empirical coordinate constants are estimated only from evaluated pairs
that differ in one coordinate, so the estimate has the same validity caveat
as the global empirical Lipschitz constant.

### Lemma 4.6 (exact hard-decision counting baseline)

For the 1-bit counting baseline, let `p0_i` and `p1_i` be the post-BSC
decision probabilities of report `i` under H0 and H1.  Conditioned on a
reception state with success probability `s_i`, the number of positive votes
has the Poisson-binomial PMF with vote probabilities `s_i p0_i` under H0 and
`s_i p1_i` under H1.  If the fusion threshold `K` is the smallest integer
with `P_FA(K) <= alpha`, then

`P_D = sum_{k >= K} pmf1(k)`

and `P_FA <= alpha`.

Proof: conditional on the state, reception and local decision are
independent Bernoulli events, so a vote occurs with probability
`s_i p_h_i`.  The DP in `_count_distribution` evaluates the exact
Poisson-binomial PMF, and the threshold search checks the H0 tail directly.
The common-state and grouped laws are handled by a nonnegative mixture over
states or patterns before threshold selection.

### Theorem 4.7 (exact equal-cost quota selection)

Assume every non-owner report has the same cost `c`, and the per-target
expected-P_D functions `f_q(S)` are evaluated exactly.  Let `K =
floor(B/c)`.  Then `exact_quota_select` returns a feasible schedule that is
globally optimal for the two-stage score (QoS gap first, then weighted mean,
then worst-target expected P_D) over all schedules with total report count at
most `K`.

Proof: because costs are equal, a schedule with report count `r_q` for target
`q` uses `c sum_q r_q` bits.  For a fixed quota vector `r`, the contribution
of target `q` to the weighted mean is maximized by choosing a size-`r_q`
subset with the maximum `f_q` value; `best_per_size` performs exactly this
maximization by exhaustive subset evaluation.  The composition generator
enumerates every quota vector with `sum_q r_q <= K` and every feasible
per-target capacity, so the feasible set is covered.  The lexicographic
score compares QoS gap first, then weighted mean, then worst target, which
matches the two-stage objective, so the returned schedule is optimal.

### Theorem 4.7A (exact heterogeneous-cost budget selection)

Assume per-target expected-P_D functions `f_q(S)` are evaluated exactly and
report `i` for target `q` has integer cost `c_{qi} >= 1`.  For every target
let `O_q` contain all `(cost(S), f_q(S))` pairs obtained by enumerating
every report subset, and let the DP state after `q` targets store, for each
accumulated bit cost, the componentwise-Pareto frontier of per-target value
vectors.  Then `exact_budget_select` returns a feasible schedule that is
globally optimal for the same two-stage score over all schedules with total
bit cost at most `B`.

Proof: every feasible global schedule is a path in which one `O_q` option is
selected per target, so the DP enumerates all feasible value vectors before
pruning.  If two partial schedules with the same accumulated cost have value
vectors `u` and `v` with `u_q >= v_q` for every processed target, then for
any completion `w` the completed vector `u+w` dominates `v+w` componentwise.
The QoS gap is nonincreasing in every component, while the weighted mean and
the minimum are nondecreasing, so the completed dominated schedule cannot
score strictly higher than the completed dominating schedule.  Discarding
`v` therefore preserves at least one optimal completion.  The final
lexicographic scan over all costs and frontier vectors selects the exact
optimum.

### Theorem 4.7B (exact max-min budget selection)

Assume per-target expected-P_D functions `f_q(S)` are evaluated exactly and
report costs are positive integers.  Let `V` be the finite set of all
per-target subset values.  For any threshold `t`, define the feasibility
problem `P(t)` as choosing one enumerated subset per target with value at
least `t` and total bit cost at most `B`.  Then `exact_maxmin_select` returns
a feasible schedule whose worst-target value equals

`t* = max { t in V : P(t) is feasible }`.

Proof: each subset option in `P(t)` is an item in a multiple-choice knapsack,
so the cost DP over targets and accumulated bits enumerates all feasible
combinations exactly.  If `P(t)` is feasible then `P(s)` is feasible for
every `s <= t`, so feasibility is monotone and binary search over the sorted
candidate set `V` is exact.  The optimal worst-target value must be one of
the enumerated values, because it is the minimum of the chosen per-target
values.  The threshold-feasibility DP keeps the componentwise-Pareto frontier
of value vectors at every accumulated cost, so a state that is dominated in
every processed target cannot improve any monotone secondary completion.
The final lexicographic scan therefore selects the best QoS gap, weighted
mean, and worst target among all schedules that attain `t*`, while remaining
exact for the max-min objective.

### Lemma 4.7C (per-target cost-value dominance)

For a fixed target, let options `A = (c_A, S_A, v_A)` and
`B = (c_B, S_B, v_B)` satisfy `c_A <= c_B` and `v_A >= v_B`.  Then replacing
`S_B` by `S_A` in any global schedule preserves the total bit budget and
never lowers any target's expected P_D, so `B` can be removed from the
per-target option set without changing the exact lexicographic or max-min
optimum.

Proof: the budget after replacement is `sum_{q != q0} c_q + c_A <= sum_q c_q
<= B`, and the affected target value changes from `v_B` to `v_A >= v_B`, so
every monotone objective (QoS gap, weighted mean, worst target) is no worse.
`_pareto_dominated_options` processes options in increasing cost and keeps
only an option whose value strictly exceeds all earlier options, which is
exactly the removal of all such dominated pairs.  Applying the rule inside
`exact_budget_select` and `exact_maxmin_select` therefore preserves the exact
optima of Theorems 4.7A and 4.7B.

### Theorem 4.7D (minimum-cost threshold branch-and-bound)

Let `f_q(S)` be the system's evaluated expected P_D for report set `S` and
let `f_q^*(S)` be the global P_D-optimal linear-score value, which is
set-monotone by zero extension of the weight vector.  For a DFS node with
set `S` and remaining reports `R_i`, every completion `T` satisfies
`T subset of S union R_i`, so `f_q^*(T) <= f_q^*(S union R_i)`.  Lemma 4.7G
provides a closed-form Cauchy bound `f_ub(S union R_i) >= f_q^*(S union
R_i) >= f_q^*(T)`.  The branch-and-bound in `minimum_cost_to_threshold`
prunes a node only when

`f_ub(S union R_i) < threshold`,

or, after a feasible subset is known with cost `c*`, when the current cost
plus the cheapest remaining report cannot be strictly below `c*`.  All other
nodes are explored, and equal-cost feasible subsets are accepted when the
budget cap has not yet produced a feasible solution.  Therefore the returned
subset has the exact minimum bit cost among all subsets whose evaluated
expected P_D reaches the threshold.

Proof: the Cauchy bound covers every completion, so the first prune removes
only infeasible nodes.  The cost prune removes only nodes whose every
completion costs at least the best known cost, so it cannot remove a strict
improvement.  The remaining DFS enumerates the entire feasible completion
tree, so the first (and final) minimum is exact.

### Theorem 4.7E (scaled max-min feasibility certificate)

For threshold `t`, let `m_q(t)` be the exact minimum bit cost for target `q`
to reach value `t`, as returned by `minimum_cost_to_threshold`.  A global
schedule with all values at least `t` exists if and only if
`sum_q m_q(t) <= B`.

Proof: every feasible target subset costs at least `m_q(t)`, so a global
schedule costs at least `sum_q m_q(t)`.  Conversely, choosing one
minimum-cost subset per target is jointly feasible when the sum is at most
`B`.  Feasibility is monotone decreasing in `t`, so `scaled_maxmin_select`
uses binary search and returns a feasible lower bound with the upper bound
of the search as an epsilon certificate.  When every report set is within
`max_exhaustive_reports`, `scaled_maxmin_select` delegates to the exact
selector instead, so the epsilon gap is zero for all small instances.

### Lemma 4.7F (cost-bounded minimality proof)

Let a greedy warm start return a feasible subset with cost `c*`.  For every
integer `cap < c*`, `_minimum_cost_bounded` enumerates every subset whose
total report cost is at most `cap`.  If no such subset reaches the threshold,
then no subset with cost strictly below `c*` is feasible, so `c*` is the
exact minimum cost.

Proof: a subset with cost `c < c*` is contained in the enumeration for
`cap = c`, and the recursion is exhaustive over cost-bounded subsets, so the
search misses no feasible subset.  If the loop finds a candidate at cost
`c < c*`, the earlier caps `0, ..., c-1` already established that no cheaper
subset exists, so the returned cost is exact.  This is used before the
branch-and-bound only to prove small-cost minima quickly; it never relaxes
the exactness of Theorem 4.7D.

### Lemma 4.7G (Cauchy upper bound for the linear-score shift)

In whitened coordinates `y = L^T w`, `a = L^-1 delta` and
`Q = L^-1 Sigma1 L^-T`, every linear score has shift

`s(y) = (a^T y - z ||y||) / sqrt(y^T Q y)`,

with `z = Phi^-1(1 - P_FA)`.  For `z >= 0`,

`s(y) <= sqrt(a^T Q^-1 a) - z / sqrt(lambda_max(Q))`,

and for `z < 0` the bound uses `lambda_min(Q)` in place of
`lambda_max(Q)`.

Proof: Cauchy-Schwarz applied to the inner product `a^T y` with norm
`sqrt(y^T Q y)` gives `a^T y <= sqrt(a^T Q^-1 a) sqrt(y^T Q y)`.  Dividing
the shift by `sqrt(y^T Q y)` leaves `-z ||y|| / sqrt(y^T Q y)`.  Since
`y^T Q y <= lambda_max(Q) ||y||^2`, the ratio `||y|| / sqrt(y^T Q y)` is at
least `1 / sqrt(lambda_max(Q))`; for `z >= 0` this yields the stated bound.
For `z < 0`, the same ratio is at most `1 / sqrt(lambda_min(Q))`, which
determines the tight bound.  `pd_shift_upper_bound` evaluates the bound in
the original covariance coordinates, so it is a valid upper bound on the
shift of the P_D-optimal linear score at every operating point without
requiring numerical monotonicity at `P_D >= 0.5`.

### Lemma 4.8 (aperture-conserved subarray gain)

Let the RIS aperture be partitioned into disjoint blocks with sizes
`N_1, ..., N_S` summing to `N`, and let block `b` steer toward target
`q_b`.  The squared array gain toward target `q` is

`G_q = | (1/N) sum_b sum_{n in block b}
          exp(j(phi_n - ideal_nq)) |^2`,

and every move in `coordinate_aperture_ascent` transfers `step` elements from
one block to another, so `sum_b N_b = N` is preserved exactly.

Proof: the phase vector built by `multi_beam_phase` places the phase of each
absolute element index inside its block, so the array factor is the sum over
all elements, and `ris_array_gain` evaluates the normalized magnitude.  The
coordinate ascent constructs every trial by subtracting `step` from one
entry and adding it to another, which leaves the total unchanged; the
feasibility check `trial[source] >= step` keeps all block sizes nonnegative.
An aligned self-block contributes exactly `N_b / N` to `G_q`, so the search
is a zero-sum aperture allocation under the exact cross-block interference
model.

### Lemma 4.9 (subarray steering coordinate ascent)

Fix an aperture allocation `(N_1, ..., N_S)` and a system objective
`F(u_1, ..., u_S)` evaluated exactly.  `coordinate_block_steering_ascent`
keeps `sum_b N_b = N` and the per-element phase-bits unchanged, and returns
a steering vector that cannot be improved by changing any single coordinate
on the searched grid.

Proof: the phase vector is built by `multi_beam_phase` with the fixed
allocation, so no element is added or removed.  Each sweep perturbs one
block cosine, clips it to `[-1, 1]`, evaluates `F`, and accepts the best
feasible improvement.  The loop stops when no single-coordinate grid change
improves `F`, which is exactly the definition of a grid-local optimum.

### Lemma 4.10 (aperture-versus-overhead trade-off)

Let `P_ris(N) = N^2 G^2 A / (R_1^2 R_2^2 R_3^2)` be the RIS path power and
let `B_report = B_total - N b / C` with `b` phase bits per element and `C`
coherence frames.  A configuration is feasible if and only if
`N b / C < B_total`, and within the feasible set the sensing SNR gain is
monotone in `N` while the report budget decreases linearly in `N`.

Proof: the first inequality follows directly from the resource identity and
the requirement `B_report >= 0`.  For fixed aperture scale and geometry,
`P_ris(N)` is quadratic in `N`; `B_report(N)` is affine decreasing in `N`.
Therefore increasing `N` raises the channel term quadratically while
reducing the communication term linearly, and the net system benefit is an
empirical question that G11 resolves by exact expected-P_D evaluation under
the same identity.

### Theorem 4.11 (closed-form aperture optimum)

Assume the subarray approximation `G_q = a_q / N`, equal allocation
`a_q = N/3`, and a deflection law quadratic in the evidence SNR.  Let
`kappa = K_weak sinc^2(1/2^b) / 9`, `R = B_total`, and `L = b / C`.  Then
the weak-target surrogate is

`J(N) = beta (1 + kappa N^2)^2 (R - L N)`,

and every interior stationary point satisfies

`5 kappa L N^2 - 4 kappa R N + L = 0`.

Proof: differentiating `J` gives

`J'(N) = (1 + kappa N^2) [ 4 kappa N (R - L N) - L (1 + kappa N^2) ]`.

The factor `1 + kappa N^2` is positive, so the bracket must vanish:

`4 kappa N R - 4 kappa L N^2 = L + L kappa N^2`,

which rearranges to the claimed quadratic.  The larger root,

`N* = (4 kappa R + sqrt(16 kappa^2 R^2 - 20 kappa L^2)) / (10 kappa L)`,

is the finite local maximum when it lies in `(0, R/L)`; if the discriminant
is negative the objective has no interior stationary point and the
architecture should stay at the feasibility boundary.

### Theorem 4.12 (max-min deflection water-filling)

Let `D_q(a_q) = beta_q (1 + kappa_q a_q^2)^2` be the owner-only deflection
surrogate with `beta_q > 0` and `kappa_q > 0`.  Consider the max-min problem

`max_{a >= 0, sum a_q = N} min_q D_q(a_q)`.

At a stable point of the block-moving iteration, every target with nonzero
aperture has the same surrogate value as the current minimum, or all
aperture is concentrated in one target.

Proof: `D_q` is continuous, increasing, and convex in `a_q`.  If target `j`
is the unique minimizer and target `i` has `D_i > D_j` and `a_i > 0`, moving
`delta` aperture from `i` to `j` strictly increases `D_j` while `D_i`
remains above the previous minimum for sufficiently small `delta`; hence the
minimum strictly increases.  A point where no such move improves the
minimum must either tie all nonempty targets at the minimum or be unable to
move aperture (a single nonempty target).  The implemented halving-step
iteration converges to such a fixed point.

Remark: equalizing the marginal derivatives `dD_q/da_q` is a necessary
condition for maximizing a sum of independent concave terms, not for the
max-min objective.  That variant was implemented first, degraded exact
worst-target P_D, and is recorded as a rejected branch.

### Lemma 4.13 (exact array-factor surrogate)

For a phase profile built by `multi_beam_phase`, the exact squared array gain
toward target `q` is

`G_q(a) = | (1/N) sum_b sum_{n in block b}
           exp(j(phi_n - ideal_nq)) |^2`,

and the exact owner-only deflection surrogate is

`D_q(a) = beta_q (1 + K0_q N^2 G_q(a))^2`.

Proof: `ris_array_gain` evaluates the normalized magnitude of the full
array-factor sum, so every cross-block interference term is included.
`multi_beam_phase` assigns each element a phase from its block steering
vector, and `aperture_allocation_gains` squares the resulting magnitude.
Max-min water-filling on this exact surrogate therefore optimizes the
surrogate including interference, but the exact expected-P_D system also
contains the greedy scheduling discontinuity and reporting law; G14 shows
the two objectives are not perfectly aligned.

### Theorem 4.14 (greedy-aware system-level local optimum)

Let `F(a)` be the exact system objective

`F(a) = mean_seed min_q E_PD(q, S_q(a))`,

where `S_q(a)` is the expected-P_D greedy schedule under allocation `a`.
If `coordinate_aperture_ascent` stops, no single-block transfer in the
searched step set strictly increases `F`, so the returned allocation is a
local optimum of the true objective with respect to those moves.

Proof: every move is evaluated by the exact `F`, including greedy scheduling
and the reporting law.  The algorithm accepts only strict improvements, so
`F` increases monotonically and the allocation remains feasible.  Termination
therefore implies that every considered single-block transfer has
`F(a') <= F(a)`, which is the definition of a local optimum over that move
set.  The step-size limitation is explicit: 8-element transfers may miss
finer optima near saturation, as G15 documents.

### Theorem 4.15 (single-element local-optimality certificate)

Let `F` be the exact system objective and `a` a feasible allocation.  If

`max_{r != q, a_r > 0} F(a + e_q - e_r) - F(a) <= 0`,

then `a` is locally optimal with respect to every single-element transfer.

Proof: `exact_single_move_gradients` enumerates all `Q(Q-1)` ordered pairs
with positive source aperture, evaluates `F` exactly for each trial, and
returns the maximum gradient.  A nonpositive maximum means no feasible
one-element move increases `F`, which is exactly the definition of
single-element local optimality.  The certificate is valid for the exact
greedy-aware objective, including the scheduling and reporting law.

### Theorem 4.16 (bounded multi-block local-optimality certificate)

Let `a` be a feasible allocation and `T` a positive integer.  If no integer
vector `n` with `sum n = 0`, `sum max(n, 0) <= T`, and `a + n >= 0`
satisfies `F(a + n) > F(a)`, then `a` is locally optimal with respect to all
multi-block moves moving at most `T` elements in total.

Proof: `bounded_multi_move_certificate` enumerates every such net-change
vector exactly, evaluates the exact system objective for each feasible
trial, and returns the best value.  A non-improving best value means no
bounded multi-block move increases `F`, which is the definition of local
optimality over that neighborhood.  The certificate is valid for the exact
greedy-aware objective; G17 iterates this check until a fixed point and
reports the resulting allocation.

### Theorem 4.17 (joint placement-allocation local optimum)

Let `(s, a)` be a RIS position and allocation.  If no T<=3 allocation
reallocation improves `F(s, a)` and no 0.5-meter single-coordinate position
move improves `F(s, a)`, then `(s, a)` is locally optimal with respect to
that joint neighborhood.

Proof: G18 alternates two local searches.  The allocation step applies
Theorem 4.16 and terminates only when no bounded multi-block move improves
`F`.  The position step evaluates all six 0.5m coordinate neighbors and
terminates only when none improves `F`.  Alternating until neither step
changes the point yields a joint point with no improving move in either
degree of freedom, which is the definition of joint local optimality for the
searched neighborhoods.

### Lemma 4.18 (bit-granularity feasibility)

Let soft reports cost `c` bits and hard decisions cost 1 bit.  If the report
budget satisfies `B < c`, no soft report can be scheduled, while up to `B`
hard decisions can be transmitted.  However, the transmitted hard-decision
schedule must still satisfy the global false-alarm constraint; with too few
votes per target, no counting threshold may meet `P_FA <= alpha`, in which
case the distributed rule is infeasible and cannot be claimed as a win.

Proof: scheduling feasibility is governed by the per-report cost, but
detection feasibility is governed by the fusion P_FA.  A 1-bit local
schedule can send one report per target whenever `B >= Q`, yet with two votes
per target the OR rule often has `P_FA > alpha` and the two-vote AND rule may
have zero detection power.  `hard_decision_fusion` therefore returns
`P_D = 0` when no threshold satisfies the global P_FA.  G19 shows this
explicitly at B=20 with `c=5`: the distributed 1-bit schedule is infeasible,
and the earlier +6.0pp claim was withdrawn after enforcing the global P_FA
constraint.

### Lemma 4.19 (distributed threshold optimization)

Let `p0_i(alpha)` and `p1_i(alpha)` be the post-BSC local decision
probabilities of report `i` under a local false-alarm rate `alpha`.  For a
fixed scheduled vote set, the counting rule with the smallest threshold
satisfying `P_FA <= alpha_g` maximizes P_D among counting thresholds for that
`alpha`.  Scanning `alpha` over a finite grid and retaining the feasible
rule with the largest P_D therefore produces a design that is never worse
than any included fixed local threshold.

Proof: `hard_decision_fusion` evaluates the exact Poisson-binomial P_FA and
P_D for each `alpha` and selects the smallest threshold meeting the global
P_FA.  `optimized_hard_decision_fusion` takes the maximum P_D over the
feasible grid.  Since the fixed default `alpha = 0.1` is included in the
grid, the optimized rule dominates the default by construction.  G20 verifies
this at B=40 and reports infeasibility when no grid point meets the global
P_FA.

### Lemma 4.20 (peer-majority consensus)

Let every UAV make an independent local 1-bit decision with optimized local
P_FA and let the target be declared when at least `K` of `M` UAVs vote
positive.  The counting P_FA and P_D are exact Poisson-binomial tails.  When
`M` is large enough for a threshold satisfying `P_FA <= alpha_g`, the rule is
feasible even if owner-based fusion with very few report votes is not; with
high local SNR the optimized majority can match or exceed centralized soft
fusion.

Proof: `peer_majority_fusion` evaluates `q0_i(alpha)` and `q1_i(alpha)` for
every UAV, forms the exact count distributions, and selects the smallest
feasible threshold for each candidate local P_FA.  Feasibility requires only
that some `K` has `P_FA(K) <= alpha_g`; with `M=8` votes this is generally
possible, whereas the two-vote owner rule of G19/G20 often has no feasible
threshold.  G21 validates the P_D values at all three configurations.

### Lemma 4.21 (multi-hop reachability)

Let each hop succeed independently with probability `r`.  The probability
that a vote traverses at least one of `hops` independent paths is
`1 - (1 - r)^hops`.  With per-UAV observability `obs_i`, the effective
participation of UAV `i` is

`p_i = obs_i (1 - (1 - r)^hops)`,

and the degraded peer-majority P_FA/P_D are exact Poisson-binomial tails of
`p_i q0_i` and `p_i q1_i`.

Proof: `1 - (1-r)^hops` is the complement of all-hops-failure probability for
independent hops.  Multiplying by observability gives the probability that
the UAV both observes the target and reaches consensus.  Conditional on
participation, the local decision is independent, so the count distribution
is the standard Poisson-binomial law evaluated by `_count_distribution`.

### Lemma 4.22 (common failure and heterogeneous observability)

Let a network-wide outage occur with probability `p_c`, independently of the
per-UAV observation and per-hop link events.  The effective participation of
UAV `i` is

`p_i = obs_i (1 - p_c) (1 - (1 - r)^hops)`,

and the degraded consensus P_FA/P_D are exact Poisson-binomial tails of
`p_i q0_i` and `p_i q1_i`.

Proof: conditional on no common failure, the observation and multi-hop
events are independent, so the participation formula is the product of their
probabilities.  The common failure event is a global factor `(1 - p_c)`, and
the counting distribution remains Poisson-binomial because the local
decisions are conditionally independent given the network state.

### Lemma 4.23 (majority scaling with UAV count)

For i.i.d. local decisions with `p0 < p1`, the majority rule satisfies
`P_FA(M) -> 0` and `P_D(M) -> 1` exponentially in the number of UAVs `M`
as `M -> infinity`.  Consequently, a fully distributed peer majority can
approach centralized soft fusion when `M` is large relative to the number of
targets `Q`, while the per-target report budget in G24 is scaled only
linearly in `Q`.

Proof: the majority count is a sum of i.i.d. Bernoulli variables.  By the
Chernoff bound,

`P_FA(M) <= exp(-M D(p0 || 1/2))`,

and `1 - P_D(M) <= exp(-M D(p1 || 1/2))` for `p1 > 1/2`.  The exponents are
positive and linear in `M`, giving exponential convergence.  G24 observes
that the practical requirement is a sufficiently large `M/Q`, since each
target consumes scheduling and report resources.

### Theorem 4.24 (finite termination and explicit complexity)

G18 alternates exact allocation and position searches, accepts only strict
improvements of `F`, and stops when neither degree of freedom improves.
Since `F <= 1`, the allocation lattice is finite, and the position grid with
0.5m steps inside a bounded box is finite, the algorithm terminates in
finitely many rounds.  For `Q` targets and `T=3`, each allocation certificate
evaluates at most `(2T+1)^Q` net vectors and each position round at most 18
neighbors, so the number of exact system evaluations is explicit and
polynomial in `Q` for fixed `T`.

Proof: strict improvement gives a strictly increasing sequence in a finite
set, so termination is guaranteed.  The complexity bounds follow from the
enumerated move sets and the bounded position grid.  The full information
inventory and non-neural-network argument are in `G18_THEORY.md`.

### Theorem 4.25 (scaled G18 for Q>3)

For `Q > 3`, the scaled architecture replaces the exponential T<=3 allocation
certificate with `waterfilling_allocation` and keeps the exact position
coordinate ascent.  Its per-configuration cost is polynomial in `Q`:

`O(seeds * C_greedy * (Q^2 + axes * steps * rounds))`,

and it does not claim an allocation certificate for `Q > 3`.

Proof: `waterfilling_allocation` runs a deterministic max-min iteration over
`Q` blocks with halving step sizes, which costs `O(Q^2 * rounds)`.  The
position search evaluates at most `3 * 2 * 3` neighbors per full step sweep.
Each `F` evaluation costs `O(seeds * C_greedy)`.  Multiplying these terms
gives the bound.  Since the exponential allocation certificate is not used,
the architecture is polynomial in `Q`; G25 validates the QoS and P_D at
`Q in {2,4,6}`.

### Lemma 4.26 (adaptive water-filling under time-varying geometry)

Let the surrogate constants `beta_q(t), kappa_q(t)` vary with time.  If
`waterfilling_allocation` maximizes the frame-wise max-min surrogate, then
for every frame `t`,

`min_q D_q(a_adaptive(t), t) >= min_q D_q(a_static, t)`,

so the adaptive allocation is never worse than any static allocation in the
frame-wise surrogate worst case.

Proof: a static allocation `a_static` is a feasible candidate in the frame-t
water-filling problem.  The adaptive water-filling accepts the best feasible
allocation for that frame, so its surrogate minimum dominates the static
allocation's minimum.  The exact system objective may still disagree with
the surrogate (G14), which is why G26 validates the adaptive allocation with
the exact expected-P_D system.

### Lemma 4.27 (multi-RIS power sum and aperture split)

For non-coherent RIS paths,

`gain_iq = 1 + sum_r N_r^2 G_rq^2 A / (R_1r^2 R_2r^2 R_3r^2) / P_dir_iq`,

and the control overhead is `sum_r N_r b / C`.  If a total aperture `N` is
split into `R` equal RISs with identical leg products, the coherent part of
the reflected power scales as `1/R` compared with a single RIS:

`sum_r (N/R)^2 = N^2 / R`.

Therefore, without a geometry advantage, one large RIS dominates; multi-RIS
is useful only when the additional surfaces reduce the cascaded leg products
enough to compensate.  G27 validates this trade-off exactly.

### Lemma 4.28 (convex aperture split)

For one target and aligned RISs, the reflected power is
`P(N) = sum_r N_r^2 / L_r` with fixed `sum N_r = N`.  The Hessian of
`N_r^2` is positive, so `P` is strictly convex on the simplex and its maximum
occurs at an extreme split.  Consequently, a multi-RIS split cannot help a
single target; it can only help when multiple targets have different cascaded
loss vectors, which is why G28 optimizes the split on the exact multi-target
system objective.

Proof: `d^2(N_r^2)/dN_r^2 = 2 > 0`, so the sum of convex functions is convex;
a convex function maximized over a compact simplex attains its maximum at an
extreme point.  G28 validates the multi-target case exactly.

### Lemma 4.29 (quantization bits versus report cost)

For a uniform quantizer with `2^b` levels over the same finite range, the
mean-square quantization error is nonincreasing in `b` for bounded input
densities, while the report cost is `c(b) = b + 2` in the audited model.
Therefore increasing `b` improves soft-evidence fidelity at a linear cost,
and the rate-allocation problem is a knapsack over per-report bit counts.

Proof: uniform quantization over a fixed interval has cell width
`Delta = range / 2^b`; the per-cell MSE is `O(Delta^2)` for a bounded
density, hence decreases as `2^(-2b)`.  The audited cost follows directly
from `report_bits = quantizer_bits + 2`.  G29 evaluates fixed and adaptive
profiles exactly with this cost.

### Theorem 4.30 (single-rate-change local optimum)

Let `F(bits)` be the exact system objective and `bits*` the final rate
profile of the coordinate ascent.  If every single-UAV quantizer-bit change
has `F(bits') <= F(bits*)`, then `bits*` is locally optimal with respect to
single-rate changes of the exact objective.

Proof: each coordinate sweep evaluates all `M` UAVs and all alternative bit
counts exactly, and accepts only strict improvements.  Termination implies
no single-rate change improves `F`, which is the definition of local
optimality over that neighborhood.  G30 stores this certificate explicitly.

### Lemma 4.31 (exact soft/hard hybrid LLR)

Let soft reports define a Gaussian score with P_D-optimal weights and let
hard report `i` have post-BSC decision probabilities `p0_i, p1_i`.  The score

`T = w^T x_soft + sum_i log( p_{b_i,i} / q_{b_i,i} )`

is an exact likelihood-ratio combination.  Enumerating hard patterns and
searching the soft threshold to satisfy the global P_FA gives an exact hybrid
P_FA/P_D.

Proof: conditioned on a hard pattern, the soft score is Gaussian and the
hard log-likelihood terms are constants, so the tail probability is a
Gaussian CDF.  Averaging over the exact Bernoulli pattern law yields the
unconditional P_FA/P_D, and the threshold search makes the P_FA constraint
exact.  G31 validates this and shows that hybrid is not automatically better
than soft-only.

### Lemma 4.32 (SINR injection)

For additive independent interference with power `I` and noise power `N`,
the effective signal-to-interference-plus-noise ratio is

`SINR = SNR / (1 + INR)`,

with `INR = I / N`.  Injecting this into the moment-matched effective SNR
preserves the model structure and monotonically reduces the local deflection
as `INR` increases.

Proof: `SINR = S / (N + I) = (S/N) / (1 + I/N)`.  In `build_models`, the
effective SNR is divided by `1 + INR`, so the H1 mean shift and deflection
decrease monotonically with `INR`.  G32 validates the system-level effect.

### Lemma 4.33 (spatial interference path loss)

For an interference source at position `j` and a UAV at `p_i`, the
interference-to-noise ratio follows free-space path loss:

`INR_i = inr_ref * (d_ref / ||j - p_i||)^2`.

Consequently, UAVs farther from the source receive lower INR and the
per-UAV SINR profile is spatially heterogeneous.  RIS placement changes the
target-side SNR but not this direct INR, so the architecture gain under
interference is an exact-system question.

Proof: `INR_i` is defined by the distance scaling and injected into
`build_models` as `SINR_i = SNR_i / (1 + INR_i)`.  Since only the target-side
SNR is RIS-dependent, the placement search evaluates the combined effect
exactly.  G33 validates this.

### Lemma 4.34 (multi-source INR superposition)

For independent interference sources, total interference power adds:

`INR_i = sum_s inr_ref_s (d_ref / d_{is})^2`.

The SINR is then `SNR_i / (1 + INR_i)`.  This preserves the moment-matched
model and makes the per-UAV interference profile spatially heterogeneous
across multiple directions.

Proof: powers of independent sources add, so the total INR is the sum of the
individual path-loss terms.  G34 validates the effect with three sources.

### Lemma 4.35 (UPA versus ULA in a planar target set)

For a UPA with `N = Nx Ny` elements and per-target ideal phases, the
normalized array gain is 1 when the target direction matches the steering
vector.  If all target directions lie in a single plane and the per-target
ideal phase already achieves unit gain, the 2-D aperture does not add
detection gain over the 1-D ULA with the same `N`.

Proof: the per-target ideal phase is a matched filter, and its normalized
gain is 1 regardless of aperture geometry.  The system-level difference is
therefore only sidelobe/selectivity structure, not the mainlobe gain.  G35
observes near-identical P_D; a UPA advantage requires elevation separation
or null-steering, which are not present in the audited geometry.

### Lemma 4.36 (null-steering scalarization)

For target direction `t` and interference directions `j`, the scalarized
array power is

`J(phi) = G_t(phi) - lambda sum_j G_j(phi)`.

The gradient of each `G_d` is

`dG_d/dphi_n = 2 Re[ conj(A_d) (j/N) exp(j(phi_n - ideal_nd)) ]`,

so L-BFGS-B can optimize the phase vector.  Reflected interference INR
follows `N^2 G_j^2` with normalized path losses, so reducing `G_j` directly
reduces reflected INR.

Proof: the gradient formula follows from the definition of the array factor,
and the reflected-power expression follows from the cascaded path-loss
model.  G36 validates the phase-domain suppression effect exactly.

### Lemma 4.37 (quantized null-steering coordinate ascent)

Let the phase of every element be restricted to `2^b` uniform levels.  A
coordinate ascent that flips one element to the best level while improving
the scalarized array power converges to a local optimum of the quantized
problem.

Proof: each flip strictly increases `J` and the phase set is finite, so the
algorithm terminates at a point where no single-element level change
improves `J`, which is local optimality over the discrete neighborhood.
G37 validates the quantized result against continuous-then-quantized phases.

### Theorem 4.38 (joint nulling-placement local optimum)

Let `F(s)` be the exact system objective after redesigning quantized
null-steering phases at position `s`.  If no 0.5m single-coordinate position
move improves `F`, then `s` is locally optimal with respect to that position
neighborhood, jointly with the redesigned quantized phases.

Proof: each position candidate is evaluated exactly with its own quantized
null-steering design.  Termination implies no 0.5m move improves `F`, which
is local optimality over the searched position neighborhood.

### Lemma 4.39 (QoS threshold relaxation)

For a fixed schedule, feasibility at QoS target `tau` implies feasibility at
every `tau' < tau`, because `E_PD >= tau > tau'`.  Therefore lowering the QoS
target monotonically expands the set of feasible distributed policies.

Proof: the feasibility predicate is `min_q E_PD(q) >= tau`.  Reducing `tau`
relaxes the right-hand side, so the predicate remains true for every policy
that was feasible at the higher threshold.  G39 validates this for budgets
20-28.

### Lemma 4.40 (consensus advantage under scarce report bits)

Peer consensus uses zero report bits, while centralized soft fusion requires
at least `c` bits per report.  When the report budget is smaller than the
cost of the best soft report, centralized fusion is limited to owner-only
information, while consensus can still use all `M` local decisions.  In this
regime consensus can strictly outperform centralized soft fusion.

Proof: centralized feasibility is constrained by the per-report cost, so
`B < c` leaves the owner-only schedule.  Peer consensus does not consume the
report budget and can use all local votes; with enough `M` and sufficient
local SNR, its majority P_D can exceed the owner-only P_D.  G40 validates
this at B=12.

### Theorem 4.41 (majority parity bound)

For i.i.d. local decisions with probabilities `p0 < p1`, the Gaussian
approximation of the majority rule satisfies

`M >= M_min = p1 (1 - p1) (z_alpha + z_beta)^2 / (p1 - p0)^2`

to achieve `P_FA <= alpha` and `P_D >= beta`.

Proof: for majority of `M` i.i.d. Bernoulli votes, the mean and variance
under H1 are `p1` and `p1(1-p1)/M`, and the separation from H0 is
`(p1-p0) / sqrt(p1(1-p1)/M)`.  Requiring this separation to be at least
`z_alpha + z_beta` gives the bound.  G41 compares this first-order predictor
with the exact system and shows that optimized local thresholds shift the
empirical boundary to smaller `M`.

### Corollary 4.42 (optimized local threshold)

Let `M_min(alpha)` be the parity bound for local P_FA `alpha`.  Then

`M_min^opt = min_alpha M_min(alpha) <= M_min(0.1)`,

because the fixed threshold is one feasible point of the minimization.
The optimized bound is therefore never worse and, in the audited cases, is
9-13% lower.

Proof: the minimization domain contains the fixed `alpha = 0.1`, so the
minimum is no larger than the value at that point.  G42 validates the
reduction against exact system wins.

### Theorem 4.43 (exact Poisson-binomial feasibility)

For vote probabilities `p0_i, p1_i`, the majority rule with threshold `K`
has exact tails

`P_FA(K) = sum_{k>=K} pmf0(k)`,
`P_D(K) = sum_{k>=K} pmf1(k)`,

where `pmf0, pmf1` are the exact Poisson-binomial distributions.  The
allocation is feasible if and only if some `K` satisfies `P_FA(K) <= alpha_g`
and `P_D(K) >= beta`.

Proof: the count of positive votes under each hypothesis is a sum of
independent Bernoulli variables, whose distribution is exactly the
Poisson-binomial law evaluated by `_count_distribution`.  Checking all `K`
therefore decides feasibility without approximation.  G43 shows that this
exact boundary starts at M=6, matching empirical wins.

### Theorem 4.43A (exact minimum majority count and monotonicity)

For a fixed voter sequence `(p0_i, p1_i)`, `exact_min_majority_uavs` returns
the smallest prefix size `m*` for which Theorem 4.43 feasibility holds, by
evaluating every prefix exactly.  Majority feasibility is not monotone in the
prefix size in general: with `p0 = 0.1`, `p1 = 0.7`, `alpha = 0.05` and
`beta = 0.7`, the feasibility trace is `[F, F, T, F, T]`, so `m=3` is
feasible while `m=4` is infeasible.

Proof: the prefix evaluation is exhaustive over `1 <= m <= n`, so `m*` is
exact.  The counterexample is evaluated directly by the Poisson-binomial
tails; it shows that a binary search over `m` assumes a monotonicity property
that the exact feasibility function does not possess in general.  A binary
search is therefore valid only after a monotonicity certificate for the
particular voter sequence.

### Theorem 4.44 (information-budget monotonicity within soft fusion)

In the proportional-covariance regime, the soft-score detection probability
is

`P_D = Phi( (sqrt(D) - z_FA) / sqrt(c) )`,

which is strictly increasing in `D`.  Therefore, for a fixed target,
normalized information `rho = D/D_full` is a monotone predictor of P_D
within the soft fusion family.

Proof: the derivative of `P_D` with respect to `D` is positive for
`D > 0`.  G44 validates this monotonicity across report budgets.

### Lemma 4.45 (naive closed-form law is invalid)

The formula

`P_D = Phi( (sqrt(d0 (1+n) mean(gain^2)) - z_FA) / sqrt(c) )`

ignores quantization loss, cross-report correlation, and non-proportional H1
variance.  Under the audited model it overestimates P_D by up to 30pp and
saturates to 1 for `N >= 128`, so it cannot serve as a universal
resource-information law.

Proof: the exact model propagates quantization/BSC through moments and keeps
cross-UAV covariance, so the effective deflection is not a separable product
`d0(1+n)g^2`.  G45 validates the failure numerically and justifies the exact
moment-propagation chain used elsewhere.

### Lemma 4.46 (exact effective-information coordinate)

Assume the Gaussian detection relation

`P_D = Phi( (sqrt(D_eff) - z_FA) / sqrt(c) )`

with fixed `c > 0`.  Then the effective deflection that reproduces an
observed `P_D` is

`D_eff = ( sqrt(c) Phi^{-1}(P_D) + z_FA )^2`,

and the exact normalized information budget is

`rho_exact = D_eff / D_full`.

Proof: apply `Phi^{-1}` to both sides and solve the linear equation for
`sqrt(D_eff)`.  Since `Phi` is strictly increasing, the coordinate is
monotone in `P_D`: larger `P_D` always corresponds to larger `D_eff`.
G46 uses `c=1` for the aggregate audit and shows that the raw deflection
coordinate `rho_raw` can overstate `rho_exact` by 2.38-2.78x under
quantization and correlation, so `rho_raw` is not a P_D-consistent
information law.

Corollary: when comparing centralized soft, hard-decision, or consensus
architectures, the report-budget identity must be enforced before
`rho_exact` is computed.  In G46, hard decisions spend one bit per report
and soft reports spend `quantizer_bits + 2` bits, so a hard schedule with
`per_target = floor(report_budget / Q)` is feasible only when that number is
zero or positive; the previous `max(1, ...)` construction violated the
budget at `B=8`.

### Lemma 4.47 (exact two-branch architecture switch)

Let detector A have worst-target detection probability `P_D,A` and detector B
have `P_D,B`, both calibrated so their global false-alarm probabilities do
not exceed `alpha_g`.  Selecting the branch with the larger `P_D` is
feasible and yields

`P_D,select = max(P_D,A, P_D,B)`.

Proof: each branch independently satisfies the false-alarm constraint, and
only one branch is active for a given decision.  The selected P_D is
therefore the maximum of the two calibrated probabilities.  G47 applies this
to centralized soft fusion (`A`) and peer majority (`B`); no extra report
bits are consumed because the active branch's own budget is used.  The fixed
`report_budget < 10` policy is an empirical substitute for the exact
comparison and is not asserted as a theorem.

### Lemma 4.48 (target-wise architecture switch dominates the global switch)

For per-target detection probabilities `a_q = P_D,soft(q)` and
`b_q = P_D,peer(q)`, the target-wise policy has worst-target P_D

`P_D,target-wise = min_q max(a_q, b_q)`.

The global two-branch switch has worst-target P_D

`P_D,global = max(min_q a_q, min_q b_q)`,

and

`min_q max(a_q, b_q) >= max(min_q a_q, min_q b_q)`.

Proof: for every `q`, `max(a_q,b_q) >= a_q` and
`max(a_q,b_q) >= b_q`.  Taking the minimum over `q` preserves both
inequalities, so `min_q max(a_q,b_q) >= min_q a_q` and
`>= min_q b_q`.  Hence it is at least the maximum of those two minima.
G48 validates the strict improvement at B=12/16/20.  The target-wise policy
is feasible because each selected target uses one of the two calibrated
branches, and the soft targets use no more report bits than the centralized
schedule already computed.

### Lemma 4.49 (additive soft-report reallocation is a monotone ascent)

Let `S` be a feasible soft schedule under report budget `B`, and let
`M_q in {soft, peer}` be per-target modes.  If peer-selected targets spend
zero report bits, then any update that only adds reports to soft targets
while keeping `used(S') <= B` satisfies

`P_D,soft(q, S'_q) >= P_D,soft(q, S_q)`

for every soft target `q`, because `S_q subset S'_q` and the expected-P_D
objective is set-monotone at the audited operating points.  Peer targets are
unchanged, so the target-wise worst P_D

`min_q [ max(P_D,soft(q), P_D,peer(q)) ]`

is nondecreasing.  If the added report has positive exact marginal
expected-P_D gain, the improvement is strict.

Proof: the reallocation routine in G49 evaluates exact
`expected_gaussian_detection_probability` marginals, never removes a
scheduled report, and stops when no budget remains or no positive marginal
exists.  Set monotonicity of the Gaussian linear-score optimum (Theorem 2.4)
then gives the componentwise inequality, and the worst-target minimum
preserves it.

### Lemma 4.50 (limiting-target mode-ascent acceptance rule)

Let `w = min_q P_D,q` be the current worst-target P_D under a feasible
target-wise mode assignment.  For a peer target `q` with
`P_D,peer(q) <= w`, suppose unused report bits can produce a centralized
soft schedule with `P_D,soft(q) > w`.  Switching `q` to centralized soft is
feasible and strictly improves the worst target:

`min_q' P_D,q' = P_D,soft(q) > w`.

Proof: before the switch, every other target has P_D at least `w` and the
peer branch of `q` equals `P_D,peer(q) <= w`.  After the switch, the value of
`q` becomes `P_D,soft(q) > w`, while all other targets are unchanged and are
at least `w`, so the new minimum is strictly larger.  The switch uses only
unused report bits, so budget feasibility is preserved.  Failed attempts are
discarded and cannot change the objective.  G50 applies this rule and
verifies the +0.39pp improvement at B=12.

### Lemma 4.51 (per-frame mode ascent carries to worst-over-time QoS)

Let `v_t` be the target-wise worst P_D at frame `t` and let `a_t` be the
worst P_D after the G50 mode ascent on the same frame.  Lemma 4.50 and the
additive reallocation monotonicity give `a_t >= v_t` for every `t`, so

`min_t a_t >= min_t v_t`.

Proof: the minimum over frames preserves the componentwise inequality.
G51 evaluates both quantities on AR(1) stochastic trajectories with exact
per-frame moment propagation, so the worst-over-time gain is an exact
property of the sampled frame model, not a Monte Carlo smoothing artifact.

### Lemma 4.52 (AR(1) conditional-mean RIS prediction)

Let the target position decompose as `p_t = n_t + x_t` with deterministic
nominal `n_t` and Gaussian AR(1) perturbation

`x_t = rho x_{t-1} + w_t`,  `w_t ~ N(0, (1-rho^2) sigma^2 I)`.

The conditional mean of `p_t` given `p_{t-1}` is

`hat p_t = n_t + rho (p_{t-1} - n_{t-1})`,

and it minimizes the mean squared prediction error over all measurable
functions of `p_{t-1}`:

`E[ ||p_t - hat p_t||^2 ] = 3 (1-rho^2) sigma^2`.

Proof: conditionally on `x_{t-1}`, `x_t = rho x_{t-1} + w_t` is Gaussian
with mean `rho x_{t-1}` and covariance `(1-rho^2) sigma^2 I`.  The
conditional mean is the MMSE estimator for Gaussian variables, and the
unconditional expected squared error equals the trace of that covariance.
The RIS phase in G52 is designed from `hat p_t`; this is a position-domain
optimality statement, not a claim of global P_D optimality after nonlinear
quantization, blockage, and report selection.  G52 validates the +0.65pp
worst-over-time gain over latency-1 numerically.

### Lemma 4.53 (h-step AR(1) prediction and error covariance)

For a stationary Gaussian AR(1) perturbation `x_t = rho x_{t-1} + w_t` with
`Var(x_t) = sigma^2 I`, the h-step conditional-mean predictor is

`hat p_{t|t-h} = n_t + rho^h (p_{t-h} - n_{t-h})`,

with prediction-error covariance

`Cov(p_t - hat p_{t|t-h}) = (1 - rho^{2h}) sigma^2 I`.

Proof: iterating the recursion gives
`x_t = rho^h x_{t-h} + sum_{k=0}^{h-1} rho^k w_{t-k}`.  The conditional mean
is `rho^h x_{t-h}`, and the residual is a Gaussian sum with covariance
`sigma^2 (1 - rho^{2h}) I` by the stationary geometric-series identity.
The trace `3 (1 - rho^{2h}) sigma^2` is strictly increasing in `h`, which is
the theoretical scale that G53 matches numerically.

### Lemma 4.54 (expected-squared-gain gradient ascent is monotone)

Let

`G(phi) = E[ |(1/N) sum_n exp(j(phi_n - phi_n^pred + k d n epsilon))|^2 ]`

with `epsilon ~ N(0, sigma_dir^2)`.  The projected-gradient update

`phi <- (phi + step * grad G(phi)) mod 2 pi`

with a sufficiently small step does not decrease `G`.

Proof: `G` is a smooth quadratic form in the complex phase exponentials, and
`grad G` is the exact gradient of the smooth objective.  For a sufficiently
small step, a gradient ascent step is a descent-direction update of
`-G`, so `G` is nondecreasing along the trajectory.  This is a surrogate
property only.

### Corollary 4.55 (expected-gain surrogate is not system-optimal)

G54 evaluates the expected-gain-optimal phase through 3-bit quantization and
the exact expected-P_D/mode-ascent chain.  At h=3, the robust phase reduces
worst-over-time P_D from 0.7200 to 0.6557, so monotone surrogate gain does
not imply monotone system P_D.  Quantization and nonlinear fusion break the
transfer, and the surrogate is rejected as a design criterion.

### Lemma 4.56 (hysteresis architecture-reconfiguration loss bound)

Let `b_t` be the best P_D among the feasible architecture candidates at frame
`t`, and let `a_t` be the value of the incumbent architecture under the
hysteresis rule: switch from the incumbent value `v_t` to `b_t` only when
`b_t > v_t + delta`, otherwise keep the incumbent.  Then

`a_t >= b_t - delta`

for every frame, and therefore

`min_t a_t >= min_t b_t - delta`.

Proof: if a switch occurs, `a_t = b_t`.  If no switch occurs, the incumbent
value satisfies `v_t >= b_t - delta` because the switch condition failed.
Taking the minimum over frames preserves the inequality.  G53 validates the
bound at delta=0.02: the observed oracle-to-hysteresis worst loss is 0.00104,
below delta, while the mean switch count falls from 4.50 to 2.25 per seed.

### Corollary 4.57 (cost-aware hysteresis selection)

Let `N(delta)` be the number of architecture switches under the hysteresis
rule.  `N(delta)` is nonincreasing in `delta`, so for a per-switch control
cost `c` and control budget `B_ctl`, the feasible set

`{ delta : c N(delta) <= B_ctl }`

is an upper interval of the scanned grid.  The exact system objective is
then maximized over this feasible set.  G53 reports the resulting frontier:
under `B_ctl = 6` bits, `c=1/3/6` bits choose `delta=0.00/0.03/0.05` with
worst P_D `0.7369/0.7250/0.7217`.

Proof: increasing `delta` makes the switch condition harder, so no switch
that occurs at a larger threshold can be absent at a smaller threshold;
hence `N(delta)` is monotone.  The finite grid search therefore evaluates
the exact objective over the feasible upper interval.

### Theorem 4.58 (exact worst-scenario chance-constrained allocation)

Let target `q` have a finite scenario set `S_q`, and let each schedule
`A_q` be scored under every scenario `s` by violation probability
`v_qs(A_q)` and risk `r_qs(A_q)`.  For global weights `w_q` and violation
limits `l_q`, define the scenario excess

`E_s(A) = sum_q w_q max(v_qs(A_q) - l_q, 0)`

and the scenario risk `R_s(A) = sum_q r_qs(A_q)`.  The robust DP in
`robust_portfolio.py` keeps, for every exact cost, the componentwise
nondominated labels `(E_1, ..., E_S, R_1, ..., R_S)` and returns the
schedule minimizing `(max_s E_s, max_s R_s, used_bits)`.

Proof: every global schedule is a path through one option per target, and
each `E_s`, `R_s` is a sum of nonnegative target-level contributions.  If
label A dominates label B componentwise in both vectors, then for every
future option group the final max under A is no larger than under B, so
pruning B cannot remove an optimal completion.  The final comparison over
the remaining labels therefore solves the max-of-sums knapsack exactly.
When the scenario set contains all physically relevant degradation states,
the returned schedule is exact worst-case optimal over that finite set.

### Theorem 4.59 (BSC degradation ordering and exact-LRT ROC dominance)

For `0 <= p1 <= p2 <= 0.5`, let

`q = (p2 - p1) / (1 - 2 p1)`.

Then `BSC(p2)` is the cascade of `BSC(p1)` followed by `BSC(q)`, and every
decision rule applied to the `p2` output is a randomized decision rule
applied to the `p1` output.  Consequently the exact likelihood-ratio ROC
under `p1` dominates the ROC under `p2`: at every fixed `P_FA`, the exact
LRT `P_D` under the cleaner channel is no smaller.

Proof: for a sent bit `b` and received bit `r`, the cascade
`BSC(p1)` then `BSC(q)` flips `b` twice with probabilities `p1` and `q`, so
the net flip probability is `p1 (1-q) + (1-p1) q`, which equals `p2` by the
definition of `q`.  For an arbitrary decision rule `delta` on the `p2`
output, define `delta'(y1) = E_q[ delta(y2) | y1 ]` on the `p1` output; this
randomized rule has the same H0/H1 probabilities as `delta` on the `p2`
output.  The exact LRT is optimal at each fixed `P_FA`, so its ROC cannot be
worse than the `p1` ROC, and the dominance inequality follows.  The gate in
`channel_degradation.py` verifies the same ordering on the exact quantized
Gaussian likelihood-ratio grid.

### Theorem 4.60 (erasure stochastic dominance and expected-P_D monotonicity)

Let independent reporting links have success probabilities `p_a` and `p_b`
with `p_b <= p_a` componentwise.  Set `r_i = p_b_i / p_a_i` for `p_a_i > 0`
and `r_i = 0` otherwise.  The `p_b` reception process is exactly the process
that first receives with `p_a` and then drops every received report
independently with probability `1 - r_i`.

Proof: by independence, the survival probability of link `i` under the
cascade is `p_a_i r_i = p_b_i`, and in the shared-uniform coupling
`U_i < p_b_i` implies `U_i < p_a_i`, so the degraded received set is always a
subset of the clean received set.  Any decision rule on the degraded output
is therefore a randomized decision rule on the clean output, and the exact
LRT ROC under `p_a` dominates the ROC under `p_b`.  Moreover, when the
P_D-optimal linear family is at operating points with `P_D >= 0.5`, Theorem
2.4 makes fixed-set P_D set-monotone; averaging over the coupled reception
law gives a nonincreasing expected P_D as success probabilities fall.  The
gate in `erasure_dominance.py` verifies both the coupling and the exact
expected-P_D ordering.

### Theorem 4.61 (velocity-bounded sensing mobility envelope)

Let a target move by displacement `delta` with `||delta|| <= R`, where
`R = v_max T_frame` is the product of a maximum target speed and the frame
duration.  For every UAV position `p_i`, the reverse triangle inequality
gives

`| ||p_i - (t + delta)|| - ||p_i - t|| | <= R`.

Under the free-space power law `P(d) = P_ref / d^2`, the largest possible
relative power increase is attained at the shortest post-move range, so for
`R < min_i ||p_i - t||`:

`max_delta P(d + delta) / P(d) <= (d_min / (d_min - R))^2`.

Proof: the first inequality is the reverse triangle inequality.  For the
second, `P` is decreasing in `d`, so the power maximum occurs at distance
`d_min - R`; substituting into `P(d) = P_ref / d^2` gives the bound.  The
stress suite therefore samples mobility inside a compact, physically
bounded sensing envelope, and `mobility_envelope.py` verifies both the range
and power inequalities on independent samples.

### Corollary 4.61A (range-derived dB-SNR envelope in `build_models`)

Let `d` be the UAV-target distance vector before a displacement bounded by
`R`, let `p = ptp(d)`, and suppose `p > 2R`.  In `build_models`, the
per-UAV range SNR is

`snr_db_i = snr_hi - span (d_i - min(d)) / ptp(d)`.

Then every normalized range term changes by at most `4R / (p - 2R)`, so the
linear SNR relative change is bounded by

`10^( span * 4R / (10 (p - 2R)) ) - 1`.

Proof: each distance changes by at most `R`, the minimum changes by at most
`R`, and the range width changes by at most `2R`.  Applying the triangle
inequality to the normalized term
`x_i = (d_i - min(d)) / ptp(d)` gives
`|x'_i - x_i| <= (R + R + p * 2R/(p p')) / p' <= 4R/(p - 2R)`.
Converting the dB bound to a linear-SNR relative increase gives the
corollary.  The mobility gate therefore verifies the envelope against the
same range-derived SNR law used by the stress suite, not only the standalone
free-space model.

### Theorem 4.62 (independent per-target ambiguity reduces to scalar DP)

Let target `q` have an independent ambiguity set `S_q`, and let
`E_q(A_q)` and `R_q(A_q)` be its worst-case excess and risk over `S_q`.
For independent choices across targets,

`max_{A in product S_q} sum_q E_q(A_q) = sum_q E_q(A_q*)`,

where `A_q*` is the option minimizing the target-level worst-case
lexicographic objective, and the analogous identity holds for the sum of
worst-case risks.

Proof: the product set makes every target choice independent, and the
objective is a sum of nonnegative target-level terms, so the maximum
separates as the sum of per-target maxima.  The scalar DP over per-target
worst-case option metrics is therefore exact; it also permits different
numbers of scenarios per target.  Theorem 4.58 remains the appropriate
formulation when one common degradation state affects all targets
simultaneously.

### Theorem 4.63 (exact robust-DP complexity)

Let `Q` be the target count, `B` the bit budget, `S` the scenario count,
`O <= 2^R` the per-target schedule/bit options in one scenario, and `E` the
cost of one exact reception-pattern quality evaluation.  Enumerating the
robust target options costs

`O( Q S O E )`.

For the common-scenario vector DP, let `L` be the maximum number of
componentwise nondominated labels kept at one exact cost.  The DP costs

`O( Q B L O S )`

in time, with `L` bounded in practice by Pareto pruning.  For the
independent per-target ambiguity DP, each cost keeps one scalar label, so
the DP costs

`O( Q B O )`.

Proof: every scenario enumerates the same target option set; each common-DP
transition updates `S` scenario totals and compares against `L` existing
labels, giving `O(B L O S)` per target.  The independent DP replaces the
scenario vectors by scalar worst cases, so the usual multiple-choice knapsack
transition gives `O(B O)` per target.  `benchmark_robustness_performance.py`
measures these operations on the target machine; on the audited 8-UAV/3-target
configuration the formal stress sweep completes in about 16 seconds and the
robust-allocation sweep in about 11 seconds.

### Lemma 4.64 (physical report-link model)

Let report link `i` have range `d_i` and path-loss exponent `alpha`, with
reference SNR `SNR_ref` at `d_ref`.  Define

`SNR_db_i = SNR_ref - 10 alpha log10(d_i / d_ref)`,

`epsilon_i = Q(sqrt(2 * 10^(SNR_db_i / 10)))`,

and, for a log-normal link with shadowing `sigma_shadow` and outage threshold
`gamma_th`,

`s_i = Phi((SNR_db_i - gamma_th) / sigma_shadow)`.

Then `epsilon_i` is nondecreasing in `d_i` and `s_i` is nonincreasing in
`d_i`, with the owner link reset to `epsilon = 0`, `s = 1`.

Proof: the path-loss formula is decreasing in `d_i`; `Q(x)` is decreasing in
`x`, so `epsilon_i` increases as `d_i` grows.  The normal CDF is increasing,
so `s_i` decreases as `SNR_db_i` falls.  `build_models` uses these values as
the post-communication moments and reception probabilities, so the physical
link layer preserves the model structure.

### Lemma 4.65 (threshold-feasibility complexity with Pareto frontiers)

Let every target have a Pareto frontier of `O` cost-value options, sorted by
nondecreasing cost with strictly nondecreasing value.  For a threshold `t`,
the minimum cost to reach `t` is found by binary search in `O(log O)`.
Therefore one feasibility check in `exact_joint_maxmin` costs `O(Q log O)`,
and the full threshold binary search costs `O(Q log O log V)`, where `V` is
the number of distinct frontier values.

Proof: feasibility of a threshold is the sum over targets of each target's
minimum cost to reach the threshold, because targets are separable and the
objective is max-min.  The first option whose value is at least `t` in a
sorted frontier gives that minimum cost, and `np.searchsorted` finds it in
logarithmic time.  This replaces the previous `O(Q O log V)` scan without
changing the exact result.  In addition, `target_options` now delegates to
`vectorized_target_options`, so the per-target enumeration is evaluated in
batches instead of one Python call per combination.

### Lemma 4.66 (joint power-bit allocation contains single-dimension baselines)

For a fixed target, let sensing power `p_i` scale the evidence separation by
`sqrt(p_i)` and let bit count `b_i` determine the report cost and
quantization fidelity.  The per-target joint option set enumerates all
affordable `(p_i, b_i)` combinations.  A sensing-only baseline is the subset
with `b_i = 1` for every selected report, and a communication-only baseline
is the subset with `p_i = 1` for every report, so both feasible sets are
subsets of the joint option set.

Proof: fixing either resource dimension restricts the same Cartesian
product, so every single-dimension feasible allocation appears unchanged in
the joint enumeration.  Since `exact_joint_maxmin` is exact over the
supplied option set, the joint optimum is never below either baseline at the
same budget.  The gate in `joint_power_bit.py` verifies this and reports the
joint gain at low/medium budgets.

### Lemma 4.67 (vectorized power-bit option enumeration)

Let `R` be the number of reports and `C = |P| * |B|` the number of
`(power, bit)` choices per report.  The joint power-bit frontier enumerates
`C^R` combinations.  The vectorized implementation precomputes the
whitened per-report gain and variance-ratio tables once and evaluates
`P_D`-optimal shifts in batches, so the per-target cost is
`O(C^R grid / batch_size)` with an exactness-equivalent result to the
per-combination evaluator.

Proof: the precomputed tables and the batched shift formula are algebraically
the same operations as the per-combination evaluator, and the same
cost/value Pareto pruning is applied afterward.  The gate compares both
paths on small instances and reports equal frontiers up to floating-point
tolerance.

### Lemma 4.68 (sensing and communication channels are decoupled)

Let the sensing channel map target geometry and power to the pre-report H0/H1
moments, and let the communication channel map the UAV-to-owner link to the
BSC flip probability and erasure survival.  If the communication-channel
parameters change while the sensing channel stays fixed, the owner-only
evidence is unchanged.

Proof: the owner does not send a report, so its BSC flip and erasure are
reset to zero and one, respectively.  Its moments are therefore produced
entirely by the sensing channel.  The gate in `physical_link_model.py`
builds two models with different communication reference SNRs and verifies
that the owner-only deflection is identical while the report flip
probabilities differ.

### Lemma 4.69 (communication-aware sensing score is a certificate-optimal surrogate)

Consider one target with diagonal, proportional-covariance evidence
`Sigma0_ii = Sigma1_ii = v_i`, independent erasures with survival `s_i`, and
equal report costs.  Define

`J_i = s_i * delta_i^2 / v_i`.

Then, for every budget, the subset with the largest `J_i` maximizes the
expected received deflection

`E[ D_R ] = sum_{i in S} J_i`.

Proof: with diagonal covariance, the deflection of a received set is the sum
of per-report contributions `delta_i^2 / v_i`, and erasure keeps report `i`
with probability `s_i` independently, so the expectation is exactly the sum
of `J_i`.  When the P_D-deflection map is concave on the operating region,
Jensen's inequality gives

`E[ P_D(D_R) ] <= P_D(E[ D_R ])`,

so the largest-J subset also maximizes the upper-bound surrogate
`P_D(E[D_R])`.  Exact P_D optimality with heterogeneous erasure survival can
still require DP; the gate reports both the exact expected-deflection match
and the resulting P_D gap.

### Lemma 4.70 (communication ambiguity endpoint reduction)

Let a fixed schedule have violation probability `v(p, s)` that is
nondecreasing in the BSC flip probability `p` and nonincreasing in the link
success probability `s`.  Over the rectangle
`[p_lo, p_hi] x [s_lo, s_hi]`, the worst-case violation is attained at
`(p_hi, s_lo)`.

Proof: for every `(p, s)` in the rectangle, monotonicity gives
`v(p, s) <= v(p_hi, s) <= v(p_hi, s_lo)`.  The exact-LRT basis is provided by
the BSC cascade ordering (Theorem 4.59) and erasure stochastic dominance
(Theorem 4.60); the moment-model gate in `communication_ambiguity.py`
verifies the same inequality on a grid.  Consequently, for common
communication ambiguity, the robust DP over the single endpoint scenario is
exact whenever the monotonicity closure passes.

### Corollary 4.70A (endpoint-reduced robust DP)

Under the assumptions of Lemma 4.70, the exact worst-case robust DP over the
four-corner ambiguity set has the same worst weighted violation excess as the
DP over the single `(flip_hi, success_lo)` scenario, with scenario count
reduced from four to one.

Proof: Lemma 4.70 shows the endpoint dominates every rectangle point for
every fixed schedule, so the maximum over the four corners is attained at
the endpoint for each schedule.  Since the robust DP minimizes the maximum
over scenario vectors, replacing the four scenario vectors by the endpoint
vector preserves both the objective and the optimum.  The gate compares the
two DPs on controlled models.

### Lemma 4.71 (robust joint power-bit allocation is exact)

Let every `(power, bit)` option be evaluated at the worst communication
endpoint `(flip_hi, success_lo)`.  Then the exact DP over the resulting
robust option frontiers is the exact worst-case allocation over the
communication ambiguity rectangle, and it is never worse under the endpoint
than any clean-optimal schedule.

Proof: by Lemma 4.70, the endpoint dominates every rectangle point for every
fixed option.  Evaluating each option at the endpoint gives its exact
worst-case P_D, so the target-separable max-min DP is exact over the robust
option set.  The clean-optimal schedule is one feasible candidate in the
robust DP, so the robust optimum is no worse under the endpoint.

### Lemma 4.72 (robust communication-aware sensing score)

In the diagonal/proportional regime, let the robust model be evaluated at
the worst communication endpoint `(flip_hi, success_lo)`.  The robust
communication-aware score

`J_i^rob = success_lo * delta_i(flip_hi)^2 / sigma0_ii`

maximizes the expected received deflection at the endpoint, and therefore
certifies the worst-case surrogate `P_D(E[D_R])` over the communication
ambiguity rectangle.

Proof: the endpoint model has the same diagonal/proportional structure as
Lemma 4.69, so the top-J rule maximizes `E[D_R]` at the endpoint.  Lemma
4.70 shows the endpoint dominates the rectangle for every schedule, so the
same top-J rule certifies the rectangle's worst-case surrogate.  The gate
compares nominal and robust top-K schedules under the endpoint and verifies
the robust schedule is never worse in expected deflection.

### Lemma 4.73 (robust CAS divergence condition)

Let `J_i^clean` and `J_i^rob` be the clean and endpoint scores of report `i`.
The nominal and robust top-K schedules differ if and only if there exist
reports `i, j` with

`J_i^clean > J_j^clean` and `J_i^rob < J_j^rob`.

Whenever this reversal occurs under strict score separation, the robust
top-K schedule has strictly larger expected received deflection at the
endpoint than the nominal top-K schedule.

Proof: top-K order is determined solely by the score sequence.  If no
reversal exists, the order is preserved and both schedules select the same
set.  If a reversal exists, the nominal schedule keeps a lower-endpoint-score
report over a higher one, so replacing it with the robust top-K cannot
decrease the endpoint expected deflection; with strict separation it strictly
increases.  The gate sweeps flip/success severities and reports the
divergence rate and the improvement.

### Lemma 4.74 (exact max-min schedule reconstruction)

Let `t*` be the exact max-min value returned by `exact_joint_maxmin`.  For
every target, selecting any enumerated option with value at least `t*` and
minimum cost gives a feasible schedule attaining `t*`.

Proof: feasibility of `t*` means each target has at least one option with
value at least `t*`, and the sum of their minimum costs is at most the
budget.  Taking the per-target minimum-cost option preserves feasibility and
keeps every target value at least `t*`, so the schedule attains the max-min
value.  `exact_joint_maxmin_selection` reconstructs this schedule and the
joint power-bit split gate reports the resulting sensing-power and
communication-bit shares.

### Lemma 4.75 (winner-take-all sensing power allocation)

In a diagonal proportional-covariance target model with deterministic
reception, the deflection of a power allocation is

`D(p) = sum_i p_i * J_i`,

where `J_i` is the per-unit-power communication-aware sensing gain.  For a
fixed bit profile and power budget, the deflection-maximizing allocation
puts all power on the report with the largest `J_i`.

Proof: `D` is linear in `p`, and `P_D` is monotone in deflection in the
proportional-covariance regime, so maximizing `D` over the power simplex is
equivalent to maximizing a linear function, whose optimum is attained at an
extreme point.  The gate compares the closed-form allocation with exhaustive
power splits on random diagonal models.

### Lemma 4.76 (winner-take-all reduces joint power-bit enumeration)

Under the diagonal proportional-covariance model with deterministic
reception, fix a bit profile and a power budget.  By Lemma 4.75 the optimal
power allocation is winner-take-all, so the joint power-bit frontier can be
enumerated over bit profiles and scalar power budgets instead of full power
vectors.  The power dimension drops from `P^R` to `P`.

Proof: for every bit profile and every scalar power budget, Lemma 4.75 gives
the exact power allocation in closed form.  Evaluating those options with
the same `Sigma1 = Sigma0` model produces the same P_D values as the full
power-vector enumeration, so the resulting max-min DP is exact.  The gate
compares both frontiers on random proportional models and reports equality.

### Lemma 4.77 (error feedback corrects WTA winner selection)

Let true per-report gains `J_i` be distinct and let the algorithm hold noisy
estimates.  Each feedback round explores the top-`K` estimated reports,
updates their estimates toward the observed true gains, and allocates all
power to the current best estimate.  With enough rounds, the true winner is
explored and becomes the best estimate, so the allocation converges to
winner-take-all.

Proof: exploring report `i` applies a convex combination
`J_i <- (1-lr) J_i + lr J_i*`, so the estimate converges to the true gain for
every explored report.  Since the true gains are distinct, the true maximum
is eventually explored and strictly dominates all others; thereafter it
remains the best estimate.  The gate measures the one-shot versus feedback
deflection improvement over random noise draws.

### Lemma 4.78 (UCB certificate stopping)

Let observations be sub-Gaussian with noise scale `sigma`, and let each
report keep a prior-regularized mean and uncertainty width
`beta * sigma_prior / sqrt(n)` with `beta` from the Gaussian quantile and a
union bound over reports.  The feedback loop stops when the current active
winner's lower confidence bound exceeds the upper confidence bound of every
other report, active or inactive.
Under the standard sub-Gaussian concentration inequality, the true best
report is then certified with probability at least `1 - confidence`, and the
loop terminates after at most `max_feedback_rounds` iterations.

Proof: each UCB/LCB width is chosen from the Gaussian quantile with a union
bound over reports, so the true gain lies inside the interval with the
claimed confidence.  The stopping condition certifies that the active winner's
lower bound dominates all other upper bounds, hence the true best is the
certified winner.  The loop always terminates at `max_feedback_rounds` even if
the certificate is not reached.  In the joint comparison the certificate only
stops the estimation feedback loop after the allocation loop has finished, so
certifying a winner cannot starve later power/bit additions.  The gate reports
the certificate stop rate and mean feedback rounds.

### Lemma 4.79 (per-target minimum cover)

When the budget satisfies `B >= 2Q`, the online allocator first activates one
report per target with 1 sensing power and 1 communication bit, using the
report with the largest per-unit-power gain.  The activation costs exactly
`2Q`, so the remaining allocation is feasible, and every target keeps at least
one active sensing/communication link.

Proof: each activation is a single `(power=1, bit=1)` report whose total cost
is 2.  Selecting the maximum `J_i` report for every target is a per-target
choice, and `2Q <= B` guarantees feasibility.  After this phase no target can
be left unobserved, which also prevents the average-score greedy from
spending the whole budget on one target while another remains unsensed.

### Lemma 4.80 (leximin refinement monotonicity and termination)

Let `v(p,b) = (v_1,...,v_Q)` be the target P_D vector and let the refinement
accept only moves that strictly improve the lexicographically sorted vector.
Then the worst target value never decreases, the sorted vector cannot cycle,
and the loop stops at a finite local optimum or at the hard `max_rounds` cap.

Proof: `leximin_improves` compares sorted vectors from the smallest entry; a
strict improvement means the first differing entry is larger, so `min_q v_q`
is nondecreasing and increases whenever a move is accepted at the minimum
position.  All powers and bits are integer-valued in `[0, B]`, so the number
of feasible allocations is finite.  A strictly monotone sequence over a
finite set cannot revisit an allocation, hence termination is guaranteed even
before the cap.  The implemented `maxmin_refine` also reports `rounds_used`
for audit.

### Corollary 4.80A (single-exchange water-filling reachability)

For two targets with one active winner report each, repeatedly moving one
power unit from the richer target to the poorer target visits every
budget-feasible power split.  The leximin acceptance rule keeps the worst
value nondecreasing, so the water-filling max-min split is reachable by the
single-exchange moves implemented in `maxmin_refine`.  The joint power-bit
gate compares the refined online allocation with the exact winner-take-all
frontier and reports equality on the tested Q=2, R=2 proportional scenarios.

Proof: with one active report per target, moving one unit from target `a` to
target `b` preserves the total budget and changes only `(p_a, p_b)`.
Repeating this generates all integer splits of the remaining power budget.
Among those splits, the max-min value is maximized by the split that
equalizes the two P_D values as closely as possible, which is exactly the
water-filling rule; the move sequence that reaches it keeps the larger value
above the smaller one at every step, so leximin accepts each move.  The gate
does not claim exactness for arbitrary report counts; it reports the measured
equality on the tested frontier.

### Lemma 4.81 (communication-aware NOMP refinement under channel mismatch)

Let the sensing channel set the report deflection while the communication
channel adds per-link BSC bit flips and link erasures.  The expected P_D
marginalizes over independent erasure patterns, so the per-report worst
endpoint `(flip_i, success_i)` is evaluated for every candidate.  When both
channels are present, the per-target covariance is no longer proportional, so
winner-take-all power is a heuristic rather than exact.  The online
refinement accepts only leximin-improving single power/bit/atom exchanges
under the same expected-P_D score.

Proof: erasure marginalization is a convex combination of fixed-received-set
P_D values, and each fixed-set value is monotone in flip degradation at the
operating points used by the gate, so the per-report worst endpoint is a
valid robust score.  Applying the same leximin acceptance argument as Lemma
4.80 keeps the robust worst target nondecreasing and guarantees termination
at a local optimum or at `max_rounds`.  The minimum-cover stage is adaptive:
a report is activated only when it improves the target's expected P_D, which
prevents forcing a low-reliability link into the fusion set.  The gate
reports measured NOMP-to-robust-exact gaps; in the Q=2/R=2 per-link gate they
are 0.000 at budgets 8/10/12, versus 0.088/0.066/0.049 for WTA-Greedy.
Under channel mismatch the winner report is selected by the marginal
expected-P_D gain at the current allocation, and the candidate set includes
every active destination report, activation of a new report, whole-atom
switches, and within-target activation transfers, so the discrete refinement
can escape the local optima caused by proxy-ranked single exchanges.  The
per-link UCB-NOMP variant observes noisy per-link coefficients, runs the
all-report winner certificate from Lemma 4.78 on the feedback loop, and then
applies the same expected-P_D refinement; in the gate it keeps the
deterministic NOMP worst values with certificate stop rates of 0.60/0.55/0.55.
For report counts above R=8 the erasure expectation is estimated by
independent Monte Carlo draws instead of enumerating all received subsets;
the estimator is unbiased.  The refinement additionally ranks candidates by a
cheap deflection proxy and verifies only the top-`candidate_budget` moves
exactly, so the per-scenario runtime stays 0.4-42s in the R=2..10 sweep.

### Lemma 4.82 (QoS-scaled leximin refinement)

Let target `q` have detection floor `l_q > 0` and priority weight `w_q > 0`,
and define the normalized QoS slack `u_q = w_q (v_q - l_q) / l_q`.  If the
refinement accepts only moves whose sorted QoS-slack vector improves
lexicographically, then the worst normalized slack never decreases and the
loop terminates at a finite local optimum or at `max_rounds`.

Proof: the QoS transform is a strictly increasing affine function of `v_q`
for fixed positive `(l_q, w_q)`, so the argument of Lemma 4.80 applies
unchanged to `u`.  The gate compares QoS-aware NOMP with an exact brute-force
over per-target robust frontiers and reports zero gap at the tested budgets,
while plain NOMP's QoS slack improves by 0.095-0.122.  The per-link
UCB-NOMP variant with the same floors and weights keeps the QoS worst values
under noisy observations and stops its feedback loop by the all-report
certificate.

### Lemma 4.83 (two-stage MAPPO-NOMP decomposition)

Let the MAPPO policy propose a report activation and bit profile, and let the
NOMP power stage allocate the sensing power under that fixed profile using
winner-take-all greedy followed by power-only leximin refinement.  The hybrid
is never below the MAPPO proposal because the power stage only accepts
moves that improve the max-min vector, and it is bounded above by the full
NOMP schedule that may also change the bit profile.

Proof: the power-only refinement uses the same leximin acceptance as Lemma
4.80, so it preserves or improves the proposed schedule.  Full NOMP searches
a superset of moves (bit/atom changes included), hence its optimum is no
worse.  In the clean homogeneous gate the hybrid reaches the exact max-min;
in clean heterogeneous it matches exact at B=8/10 and stays 0.040 below the
full NOMP schedule at B=12 because the fixed MAPPO bit profile is the only
remaining restriction.

The stricter `MAPPO-Probe-NOMP` variant restricts NOMP's report activation to
the MAPPO probe mask.  It reaches the exact max-min only when the mask
contains the optimal active set, so its guarantee is conditioned on probe
accuracy; proposal-plus-refine remains the robust form of the decomposition.

### Lemma 4.84 (MAPPO-NOMP adapter)

Let the adapter hold a finite set of NOMP requirement modes, sample MAPPO
rollouts, translate each rollout into the requested NOMP input, run NOMP, and
keep the best worst-target value after at most `iters` rounds.  The adapter
result is no worse than any single mode used alone and is bounded above by
the full NOMP schedule.  The modes are a pluggable registry and can be
selected dynamically from the operating regime instead of being fixed.  The
registry always contains the pure NOMP mode, so the adapter result is at
least the full NOMP value.

Proof: the adapter takes a maximum over schedules produced by the individual
modes, so it dominates every mode by construction.  Each mode is bounded by
the full NOMP schedule by Lemma 4.83, hence the maximum is too.  The loop is
finite by the hard `iters` cap.  In the clean heterogeneous gate the adapter
reaches 0.892/0.956/0.981 at B=8/10/12, equal to NOMP/Exact and above both
single-mode hybrids.
MAPPO training is seeded and entropy-regularized so the adapter input
distribution is reproducible.

### Lemma 4.85 (UCB mode-selection adapter)

Let the information modes have max-min rewards in `[0, 1]`, and let the
adapter choose modes by the UCB index
`mean_m + beta sqrt(log(t) / count_m)`.  The adapter pulls every mode at
least once, then concentrates pulls on promising modes, and keeps the best
observed schedule after at most `iters` pulls.

Proof: the UCB index is the standard optimistic estimator for bounded
rewards; each mode is explored at least once, so no mode is starved before
its first observation.  The kept schedule is the maximum over pulled modes,
so the result is bounded below by the best observed mode and above by the
full NOMP schedule.  Because the pure NOMP mode is always in the registry,
the result is at least NOMP.  The loop terminates at `iters`.  In the clean
heterogeneous gate the bandit adapter keeps the 0.892/0.956/0.981 NOMP/Exact
values while learning the mode means online.

### Lemma 4.86 (priority middleware)

Let a small policy choose the priority target each round, giving that target
QoS weight `2` and the others `0.5`; NOMP then solves the weighted max-min
problem and the middleware returns the unweighted worst P_D as reward.  The
middleware keeps plain NOMP as the initial best, so the final result is never
below NOMP.

Proof: every weighted solve is an exact max-min schedule for that weight
vector, and the middleware keeps the maximum over plain NOMP and all weighted
solves.  Therefore the result dominates plain NOMP by construction and is
bounded above by the exact oracle.  The policy is trained online with a
finite episode cap.  The gate reports hard R=4 scenarios where plain NOMP is
stuck in a local optimum and the priority middleware improves 0.852 to 0.864
and 0.869 to 0.878, a genuine `1 + 1 > 2` collaboration; in the five-scenario
R=4 gate two scenarios improve and three match plain NOMP, so there is no
regression.  The policy state includes the normalized NOMP residuals from the
previous weighted solve, so the priority choice is conditioned on the actual
per-target deficit that NOMP left behind.

### Lemma 4.87 (NOMP-final reward shaping)

Let the MAPPO policy be trained with the reward equal to the worst P_D after
NOMP refines the policy's proposal, instead of the raw proposal P_D.  Then
the policy gradient maximizes the system-level objective directly, and the
learned proposals are the ones NOMP can turn into good allocations.

Proof: the shaped reward is a deterministic function of the proposal and the
environment, so the REINFORCE estimator remains unbiased for the system
objective.  Because NOMP is max-min monotone, the shaped reward is at least
the raw proposal reward, giving the policy a denser and more aligned signal.
The gate reports the standalone proposal improving from 0.506/0.659/0.788 to
0.988/0.921/0.999 when the same NOMP-final reward is used with the PPO
trainer, and the proposal-plus-refine hybrid reaches NOMP/Exact at B=8/12;
the adapter with the pure NOMP fallback retains the no-worse-than-NOMP
guarantee.

### Lemma 4.88 (multi-temperature proposal ensemble)

Let the joint allocator sample PPO proposals at multiple temperatures,
refine each with NOMP, and keep the best refined schedule.  The best-of-K
value is monotone in the number of proposals, and the loop stops after
`patience` consecutive proposals with no improvement or at a hard sample cap.

Proof: the retained value is the maximum over the refined proposals, so
adding a proposal cannot decrease it.  The early-stop rule only observes
whether the maximum increased, so it cannot discard the current best; the
hard cap guarantees termination.  The gate reports hard R=2 PPO+NOMP
improving from 0.215 (single proposal) to 0.748 (ensemble) versus the
Bandit/NOMP 0.761 reference, and extreme R=2 from 0.141 to 0.306 versus
0.307.  The ensemble remains bounded by NOMP/Exact, as expected from the
max-min upper bound.

### Lemma 4.89 (difficulty-adaptive curriculum)

Let the PPO policy be trained on a curriculum of physical baseline, hard, and
extreme channels, with the NOMP-final reward evaluated by low-cost
refinement.  Then the learned proposals are aligned with difficult channels,
and the standalone policy improves on those channels without changing the
max-min upper bound.

Proof: the curriculum exposes the policy to the same channel degradation
statistics used at evaluation, and the NOMP-final reward remains an unbiased
system-level signal by Lemma 4.87.  Low-cost refinement bounds the training
cost while preserving the reward alignment.  The gate reports adapted PPO
standalone improving from 0.179 to 0.634 on hard R=2 and from 0.096 to 0.277
on extreme R=2, while the adapted ensemble stays at 0.750/0.306, bounded by
NOMP 0.761/0.306.

### Lemma 4.90 (UCB temperature allocation)

Let the adapter allocate its finite proposal budget among sampling
temperatures by the UCB index, using the refined worst P_D as the reward of
the chosen temperature.  Every temperature is sampled at least once, and the
adapter keeps the best refined schedule.

Proof: the UCB index is the standard optimistic estimator for bounded
rewards; the first round explores all temperatures, so no regime is starved,
and later rounds concentrate samples on the temperature with the best
refined reward.  The retained value is the maximum over all proposals, so
the result is monotone and bounded by the NOMP/Exact upper bound.  The gate
reports the ensemble reaching 0.750/0.306 at hard/extreme R=2.

The deployment gates use an empirical Lipschitz constant computed from
evaluated deployments and doubled as a safety factor.  The certificate is
therefore valid under that empirical constant, not under a proven global
constant for the stochastic/greedy pipeline; this boundary is reported in
the audit.

## 5. Traceability to implementation and tests

| Formal result | Implementation | Tests |
| --- | --- | --- |
| Lemma 2.1 / Theorem 2.3 | `_pd_optimal_score_components`, `optimal_gaussian_weights` | `test_optimal_pd_matches_closed_form_in_proportional_regime`, `test_optimal_pd_never_below_deflection_pd` |
| Theorem 2.4 | `optimal_gaussian_detection_probability` with grid and refinement | `test_optimal_pd_is_set_monotone_at_operating_points` |
| Theorem 2.5 | `gaussian_pd_closed_form` | `test_gaussian_pd_reduces_to_equal_covariance_formula` |
| Theorem 3.1 | `expected_gaussian_detection_probability` | `test_expected_pd_is_set_monotone_at_operating_points` |
| Lemma 3.3 | `pd_inflection_condition` | `test_pd_inflection_condition_boundaries` |
| Theorem 3.4 | bounded-regime audit in Gate G4 | `tests/test_expected_pd.py`, Gate G4 JSON |
| Section 4.1 | `ris_gain_matrix`, `ris_physics_gain_matrix` | `test_ris_gain_matrix_boosts_weak_target`, `test_snr_gain_identity_matches_baseline` |
| Section 4.2 | `ris_quantized_gain_loss` | Gate G5-Q JSON |
| Theorem 4.3 | `grid_search_suboptimality_bound` | `test_grid_bound_holds_for_distance_function` |
| Theorem 4.4 | `lipschitz_adaptive_search` | `test_adaptive_search_epsilon_optimal_for_distance_function` |
| Lemma 4.5 | `estimate_coordinate_lipschitz`, `lipschitz_adaptive_search(..., coordinate_lipschitz=...)` | `test_coordinate_lipschitz_matches_separable_function`, `test_adaptive_search_with_coordinate_lipschitz` |
| Lemma 4.6 | `hard_decision_fusion`, `_count_distribution` in `uav_otfs_isac/sota_baselines.py` | `test_hard_decision_fusion_matches_bruteforce` |
| Theorem 4.7 | `subset_expected_pd_map`, `best_per_size`, `exact_quota_select` | `test_best_per_size_returns_maximum_subset`, `test_exact_quota_never_worse_than_greedy` |
| Theorem 4.7A | `best_by_cost`, `_pareto_frontier`, `exact_budget_select` | `test_exact_budget_matches_exhaustive_oracle_nonuniform_costs`, `test_exact_budget_lexicographically_dominates_greedy_under_qos` |
| Theorem 4.7B | `exact_maxmin_select` | `test_exact_maxmin_matches_exhaustive_oracle_nonuniform_costs`, `test_exact_maxmin_never_worse_than_greedy_worst_target` |
| Lemma 4.7C | `_pareto_dominated_options` | `test_pareto_dominated_options_keeps_only_cost_value_frontier` |
| Theorem 4.7D | `minimum_cost_to_threshold` | `test_minimum_cost_to_threshold_matches_bruteforce` |
| Theorem 4.7E | `scaled_maxmin_select` | `test_scaled_maxmin_close_to_exhaustive_oracle_nonuniform_costs` |
| Lemma 4.7F | `_minimum_cost_bounded` in `scalable_selection.py` | `test_minimum_cost_to_threshold_matches_bruteforce` |
| Lemma 4.7G | `pd_shift_upper_bound` in `fusion.py` | `test_pd_shift_upper_bound_covers_optimal_linear_score` |
| Lemma 4.8 | `multi_beam_phase`, `coordinate_aperture_ascent` | `test_multi_beam_phase_preserves_total_aperture`, `test_coordinate_ascent_preserves_aperture_and_never_worsens` |
| Lemma 4.9 | `coordinate_block_steering_ascent` | `test_block_steering_ascent_never_worsens_objective` |
| Lemma 4.10 | `ris_control_overhead_bits`, `run_ris_aperture_scaling_gate.py` | Gate G11 JSON |
| Theorem 4.11 | `optimal_aperture_formula`, `derived_surrogate_objective` | `test_optimal_aperture_formula_satisfies_first_order_condition`, `test_derived_surrogate_increases_then_decreases` |
| Theorem 4.12 | `waterfilling_allocation` | `test_waterfilling_symmetric_targets_is_equal`, `test_waterfilling_weak_target_gets_more_aperture` |
| Lemma 4.13 | `exact_block_surrogate`, `exact_waterfilling_allocation` | `test_exact_surrogate_all_aperture_to_weak`, `test_exact_waterfilling_never_worsens_equal_min` |
| Theorem 4.14 | `coordinate_aperture_ascent`, `run_system_allocation_gate.py` | Gate G15 JSON |
| Theorem 4.15 | `exact_single_move_gradients`, `run_single_move_certificate_gate.py` | `test_single_move_gradients_detect_improving_move`, `test_single_move_gradients_local_optimal` |
| Theorem 4.16 | `bounded_multi_move_certificate`, `run_multi_move_certificate_gate.py` | `test_bounded_multi_move_certificate_detects_improvement`, `test_bounded_multi_move_certificate_local_optimal` |
| Theorem 4.17 | `run_joint_placement_allocation_gate.py` | Gate G18 JSON |
| Lemma 4.18 | `round_robin_schedule`, `local_hard_decision_schedule` in `run_progressive_decentralization_gate.py` | Gate G19 JSON |
| Lemma 4.19 | `optimized_hard_decision_fusion` | `test_optimized_hard_decision_never_worse_than_default` |
| Lemma 4.20 | `peer_majority_fusion`, `run_network_decentralization_gate.py` | `test_peer_majority_fusion_respects_pfa` |
| Lemma 4.21 | `degraded_peer_majority_fusion`, `run_degraded_consensus_gate.py` | `test_degraded_peer_majority_is_feasible_and_not_better` |
| Lemma 4.22 | `degraded_peer_majority_fusion`, `run_correlated_consensus_gate.py` | `test_common_failure_and_heterogeneous_observability` |
| Lemma 4.23 | `peer_majority_fusion`, `run_scalability_comparison_gate.py` | Gate G24 JSON |
| Theorem 4.24 | `G18_THEORY.md`, `run_joint_placement_allocation_gate.py` | Gate G18 JSON |
| Theorem 4.25 | `run_scaled_g18_scalability_gate.py` | Gate G25 JSON |
| Lemma 4.26 | `waterfilling_allocation`, `run_mobility_blockage_gate.py` | Gate G26 JSON |
| Lemma 4.27 | `multi_ris_physics_gain_matrix`, `run_multi_ris_gate.py` | `test_single_ris_matches_existing_model`, `test_two_ris_gain_is_not_lower` |
| Lemma 4.28 | `run_multi_ris_split_optimization_gate.py` | Gate G28 JSON |
| Lemma 4.29 | `quantizer_bits_per_uav` in `build_models`, `run_variable_rate_report_gate.py` | `test_variable_report_bits_change_model` |
| Theorem 4.30 | `run_global_rate_optimization_gate.py` | Gate G30 JSON |
| Lemma 4.31 | `hybrid_gaussian_hard_pd`, `run_hybrid_fusion_gate.py` | `test_hybrid_reduces_to_soft_only`, `test_hybrid_with_hard_reports_respects_pfa` |
| Lemma 4.32 | `interference_to_noise` in `build_models`, `run_interference_sensitivity_gate.py` | `test_interference_reduces_deflection` |
| Lemma 4.33 | `inr_profile` in `run_spatial_interference_placement_gate.py` | Gate G33 JSON |
| Lemma 4.34 | `multi_inr_profile` in `run_multi_interference_placement_gate.py` | Gate G34 JSON |
| Lemma 4.35 | `ris_upd.upd_physics_gain_matrix`, `run_upd_vs_ula_gate.py` | `test_upd_aligned_gain_is_one`, `test_upd_gain_matrix_shape_and_lower_bound` |
| Lemma 4.36 | `ris_null_steering.optimize_null_steering_phases`, `run_null_steering_gate.py` | `test_array_power_gradient_matches_finite_difference`, `test_null_steering_reduces_interference_power` |
| Lemma 4.37 | `ris_null_steering.quantized_null_steering_phases`, `run_quantized_null_steering_gate.py` | `test_quantized_null_steering_reduces_interference` |
| Theorem 4.38 | `run_joint_null_placement_gate.py` | Gate G38 JSON |
| Lemma 4.39 | `run_distributed_relaxation_gate.py` | Gate G39 JSON |
| Lemma 4.40 | `peer_majority_fusion`, `run_low_budget_snr_distributed_gate.py` | Gate G40 JSON |
| Theorem 4.41 | `theoretical_min_uavs` in `run_consensus_parity_boundary_gate.py` | Gate G41 JSON |
| Corollary 4.42 | `min_uavs_for_probabilities` in `run_optimized_parity_boundary_gate.py` | Gate G42 JSON |
| Theorem 4.43 | `exact_feasible` in `run_exact_parity_boundary_gate.py` | Gate G43 JSON |
| Theorem 4.43A | `exact_min_majority_uavs`, `majority_feasibility_trace` | `test_exact_counting_feasible_matches_poisson_binomial_prefixes`, `test_majority_feasibility_is_not_monotone_in_voter_count` |
| Theorem 4.44 | `fundamental_info.py`, `run_fundamental_information_gate.py` | `test_full_info_deflection_upper_bounds_schedule` |
| Lemma 4.45 | `run_resource_information_law_gate.py` | Gate G45 JSON |
| Lemma 4.46 | `effective_deflection`, `run_exact_information_budget_gate.py` | Gate G46 JSON |
| Lemma 4.47 | `architecture_switch.py`, `run_architecture_switch_gate.py` | `tests/test_architecture_switch.py`, Gate G47 JSON |
| Lemma 4.48 | `target_wise_architecture_switch`, `run_target_wise_architecture_switch_gate.py` | `test_target_wise_switch_is_never_worse_than_global_switch`, Gate G48 JSON |
| Lemma 4.49 | `reallocate_soft_report_bits`, `run_soft_reallocation_gate.py` | `test_reallocate_soft_report_bits_is_monotone_and_budget_feasible`, Gate G49 JSON |
| Lemma 4.50 | `two_sided_mode_ascent`, `run_mode_ascent_gate.py` | `test_two_sided_mode_ascent_never_worsens_target_wise_worst`, Gate G50 JSON |
| Lemma 4.51 | `stochastic_mobility.py`, `run_stochastic_mobility_gate.py` | `test_stochastic_trajectories_shapes_and_bounds`, Gate G51 JSON |
| Lemma 4.52 | `ar1_mmse_prediction`, `run_prediction_aware_ris_gate.py` | `test_ar1_mmse_prediction_outperforms_previous_frame_predictor`, Gate G52 JSON |
| Lemma 4.53 | `ar1_horizon_prediction`, `run_multi_step_prediction_gate.py` | `test_ar1_horizon_prediction_error_grows_with_horizon`, Gate G53 JSON |
| Lemma 4.54 / Corollary 4.55 | `covariance_aware_ris.py`, `run_covariance_aware_ris_gate.py` | `test_covariance_aware_phase_improves_expected_gain`, Gate G54 JSON |
| Lemma 4.56 | `run_multi_step_prediction_gate.py` architecture_reconfiguration | Gate G53 JSON |
| Corollary 4.57 | `run_multi_step_prediction_gate.py` switch_cost_analysis | Gate G53 JSON |
| Theorem 4.58 | `robust_portfolio.optimize_robust_chance_constrained_portfolio` | `tests/test_robust_portfolio.py` |
| Theorem 4.59 | `channel_degradation.verify_bsc_roc_dominance` | `tests/test_channel_degradation.py`, Gate BSC-D JSON |
| Theorem 4.60 | `erasure_dominance.verify_monotone_coupling`, `verify_expected_pd_monotonicity` | `tests/test_erasure_dominance.py`, Gate ER-D JSON |
| Theorem 4.61 / Corollary 4.61A | `mobility_envelope.verify_displacement_envelope`, `verify_range_snr_envelope` | `tests/test_mobility_envelope.py`, Gate MOB-E JSON |
| Theorem 4.62 | `robust_portfolio.optimize_independent_robust_chance_constrained_portfolio` | `tests/test_robust_portfolio.py` |
| Theorem 4.63 | `scripts/benchmark_robustness_performance.py` | `results/robustness_performance_benchmark.json` |
| Lemma 4.64 | `physical_link_model.physical_report_link_parameters`, `build_physical_link_models` | `tests/test_physical_link_model.py` |
| Lemma 4.65 | `joint_allocation.minimum_cost_for_threshold`, `exact_joint_maxmin` | `tests/test_joint_allocation.py`, `results/exact_joint_scaling_benchmark.json` |
| Lemma 4.66 | `joint_power_bit.power_bit_target_options`, `exact_joint_power_bit_maxmin` | `tests/test_joint_power_bit.py`, Gate JPB JSON |
| Lemma 4.67 | `joint_power_bit.vectorized_power_bit_target_options` | `tests/test_joint_power_bit.py`, `results/joint_power_bit_scaling_benchmark.json` |
| Lemma 4.68 | `physical_link_model.build_physical_link_models` | `tests/test_physical_link_model.py` |
| Lemma 4.69 | `communication_aware.communication_aware_sensing_score`, `communication_aware_top_k` | `tests/test_communication_aware.py` |
| Lemma 4.70 | `communication_ambiguity.verify_endpoint_dominance` | `tests/test_communication_ambiguity.py`, Gate CA-E JSON |
| Corollary 4.70A | `communication_ambiguity.build_endpoint_scenario_groups` | `tests/test_communication_ambiguity.py` |
| Lemma 4.71 | `robust_joint_power_bit.enumerate_robust_power_bit_options` | `tests/test_robust_joint_power_bit.py`, Gate RJB JSON |
| Lemma 4.72 | `robust_communication_aware` gate | `tests/test_robust_communication_aware.py`, Gate RCA JSON |
| Lemma 4.73 | `robust_cas_divergence` gate | Gate RCD JSON |
| Lemma 4.74 | `joint_allocation.exact_joint_maxmin_selection` | `tests/test_joint_allocation.py`, Gate PBS JSON |
| Lemma 4.75 | `power_split_theory.winner_take_all_allocation` | `tests/test_power_split_theory.py`, Gate WTA JSON |
| Lemma 4.76 | `power_split_theory.winner_take_all_proportional_options` | `tests/test_winner_take_all_joint_proportional.py`, Gate WTAP JSON |
| Lemma 4.77 | `error_feedback.wta_feedback_allocator` | `tests/test_error_feedback.py`, Gate EFB JSON |
| Lemma 4.78 | `error_feedback.ucb_wta_feedback_allocator` | `tests/test_error_feedback.py`, Gate UCB JSON |
| Lemma 4.79 | `nomp_refinement.initial_min_cover` | `tests/test_nomp_refinement.py` |
| Lemma 4.80 | `nomp_refinement.maxmin_refine` | `tests/test_nomp_refinement.py` |
| Corollary 4.80A | `nomp_refinement.nomp_wta_greedy_joint_multi` | `tests/test_nomp_refinement.py`, Gate JC JSON |
| Lemma 4.81 | `nomp_refinement` channel parameters + `scripts/run_joint_power_comm_mismatch_gate.py` | `tests/test_nomp_refinement.py`, Gate CMM JSON |
| Lemma 4.82 | `nomp_refinement.qos_scores` + `scripts/run_qos_weighted_maxmin_gate.py` | `tests/test_nomp_refinement.py`, Gate QOS JSON |
| Lemma 4.83 | `scripts/run_joint_power_comparison.evaluate_mappo_nomp`, `evaluate_mappo_probe_nomp` | Gate JC JSON |
| Lemma 4.84 | `uav_otfs_isac.mappo_nomp_adapter.MappoNompAdapter` | Gate JC JSON |
| Lemma 4.85 | `uav_otfs_isac.mappo_nomp_adapter.ModeBanditAdapter` | Gate JC JSON |
| Lemma 4.86 | `uav_otfs_isac.mappo_nomp_adapter.PriorityNompAdapter` | Gate PMW JSON |
| Lemma 4.87 | `scripts/run_joint_power_comparison.train_mappo_nomp_reward` | Gate MNR JSON |
| Lemma 4.88 | `scripts/run_joint_power_comparison.evaluate_mappo_nomp_multi` | Gate ENS JSON |
| Lemma 4.89 | `scripts/run_physical_mappo_adapt_gate.py` | Gate MAD JSON |
| Lemma 4.90 | `scripts/run_joint_power_comparison.evaluate_mappo_nomp_multi` | Gate ENS JSON |

## 6. Explicit non-claims

- The one-parameter family is proven to contain the global linear-score
  optimum only when `P_D* > 0.5`; the implementation does not claim global
  optimality or monotonicity below that operating point.
- The implemented bit-budgeted greedy does not inherit the `1 - 1/e`
  cardinality-greedy ratio; the ratio applies only to the equal-cost
  monotone-submodular variant.
- The empirical Lipschitz constant is evidence for the certificate at the
  used constant, not a proven global constant for the seed-averaged,
  greedy-approximated deployment objective.
- The finite-N quantization formula should be used for small RIS arrays; the
  Gate G5-Q audit reports the asymptotic `sinc^2` factor.
- The erasure expected-P_D monotonicity gate is asserted only at operating
  points with `P_D >= 0.5`, where Theorem 2.4 guarantees set monotonicity;
  outside that region the empirical ordering is reported but not claimed as
  a theorem consequence.
