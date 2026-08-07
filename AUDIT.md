# Documentation Consistency Audit

Audit date: 2026-08-04.

## Scope

- README.md
- PAPER_OUTLINE.md
- SYSTEM_MODEL.md
- RELATED_WORK.md
- FORMAL_PROOFS.md
- G18_THEORY.md
- SCENARIO_COMPLEXITY.md
- PAPER_DRAFT.md
- THIRD_PARTY.md
- results/paper_results_table.csv, results/paper_results_table.md
- paper_figures/
- uav_otfs_isac/sota_baselines.py, tests/test_sota_baselines.py
- results/sota_baseline_gate.json
- uav_otfs_isac/discrete_descent.py, tests/test_discrete_descent.py
- results/budget_saturation_gate.json
- uav_otfs_isac/ris_optimization.py, tests/test_ris_optimization.py
- results/ris_shared_phase_gate.json
- uav_otfs_isac/exact_quota_selection.py,
  tests/test_exact_quota_selection.py
- results/exact_quota_gate.json
- uav_otfs_isac/ris_subarray.py, tests/test_ris_subarray.py
- results/ris_subarray_gate.json
- results/ris_subarray_steering_gate.json
- results/ris_aperture_scaling_gate.json
- uav_otfs_isac/architecture_objective.py,
  tests/test_architecture_objective.py
- results/derived_architecture_gate.json
- results/waterfilling_architecture_gate.json
- uav_otfs_isac/exact_allocation.py, tests/test_exact_allocation.py
- results/exact_allocation_gate.json
- results/system_allocation_gate.json
- results/single_move_certificate_gate.json
- results/multi_move_certificate_gate.json
- results/joint_placement_allocation_gate.json
- results/progressive_decentralization_gate.json
- results/amplified_distributed_gate.json
- results/network_decentralization_gate.json
- results/degraded_consensus_gate.json
- results/correlated_consensus_gate.json
- results/scalability_comparison_gate.json
- results/scaled_g18_scalability_gate.json
- results/mobility_blockage_gate.json
- results/multi_ris_gate.json
- results/multi_ris_split_optimization_gate.json
- results/variable_rate_report_gate.json
- results/global_rate_optimization_gate.json
- results/hybrid_fusion_gate.json
- results/interference_sensitivity_gate.json
- results/spatial_interference_placement_gate.json
- results/multi_interference_placement_gate.json
- results/upd_vs_ula_gate.json
- results/null_steering_gate.json
- results/quantized_null_steering_gate.json
- results/joint_null_placement_gate.json
- results/distributed_relaxation_gate.json
- results/low_budget_snr_distributed_gate.json
- results/consensus_parity_boundary_gate.json
- results/optimized_parity_boundary_gate.json
- results/exact_parity_boundary_gate.json
- results/fundamental_information_gate.json
- results/resource_information_law_gate.json
- UAV_OTFS_ISAC_论证与系统模型_revised_final.docx
- UAV_OTFS_ISAC_论证与系统模型_revised_final_G0C.docx
- UAV_OTFS_ISAC_System_Model_revised.docx
- Relevant code and scripts under `uav_otfs_isac/`, `scripts/`, `tests/`

## What was checked

- Key quantitative claims across README, paper outline, and Word appendices:
  `96.7%`, `0.886/0.885/0.884/0.743/0.933`, `0.869/0.859/77.5%/0.880`,
  `0.62/0.61/0.58`, `4.08%/8.51%`, `0.90/0.161`.
- Status wording for G0-C, G1-A/B/C/D, G2, and the older physical Gate 1.
- File references and documentation-file lists.
- Third-party reuse boundaries.
- Idempotency of the Word appendix generator.

## Issues found and fixed

1. README listed Word appendices as A and B only, while the generated
   documents contain A/B/C.  Fixed to A/B/C.
2. The older physical "Gate 1" (identity-code collision) was not explicitly
   distinguished from the G1-A/B/C/D validation gates.  Added a clarifying
   note.
3. THIRD_PARTY.md did not list the newer independently implemented receiver
   components.  Extended the original-implementation section.
4. README said "unimplemented OTFS path-candidate front end" even though the
   toy Gate G0-C front end exists.  Changed to "full SDR-grade OTFS
   path-candidate front end".
5. README said G2 runs "only after G1 passes" while a G2 smoke already
   exists.  Clarified that the formal sweep follows G1 and the smoke is
   included to expose fairness/metric issues.

## Verified consistent

- G0-C separated-scene recovery `96.7%` (README, paper outline, Word A.3).
- G1-A saturation-robust Spearman `0.62` was a same-sample smoke; the 10k
  formal run with train/test geometry separation gives `0.588` (CIs
  `[0.23,0.83]` / `[0.21,0.84]`), below the `0.6` gate; predicted-score
  ablation with exact `P_D` gain gives formal 10k `0.996` (CI
  `[0.98,1.00]`) and logit/relative-deficit `0.994`, so the deflection proxy
  is the failure point and a `P_D`-gain selector passes G1-A; grouped
  consistency across amplitudes 0.8/1.0/1.3 confirms deflection
  `0.55/0.33/0.40` (<0.6) and P_D-gain `0.97/0.89/0.77` (>0.6) (README,
  paper outline, Word B.1).
- G1-B mean/covariance errors `4.08%` / `8.51%` (README, paper outline,
  Word B.2).
- G1-D first-order vs exact Spearman `0.90` and Oracle match rate `50%`
  (README, paper outline, Word B.4).
- G2 20-seed fair-budget numbers `0.898/0.898/0.897/0.773/0.935` with
  exact-PD greedy `0.900`, and correlated-model
  `0.870/0.855/83.1%/0.880`, plus a multi-rho sweep
  `0/0.3/0.5/0.7/0.85` with positive paired-diff CIs for `rho>=0.3`, and a
  non-saturated stress gate with gains `+0.172`/`+0.114` at `B=6/9` (README,
  paper outline, Word B.5), and a scaling sweep where Exact-PD gains grow
  from `+0.006` to `+0.067` as `(M,Q)` goes from `(8,3)` to `(16,8)`, plus a
  non-saturated scaling-stress gate with `+0.114` mean gain at `Q=3/5/8` and
  worst-target gains `+0.205` to `+0.272`.
- Resource fairness same-scale table: fixed per-frame energy
  `86.7% -> 100%` with 4x energy; fixed total energy `86.7% -> 50%`;
  time-bandwidth `2575` vs `4111` symbols; fixed-TB path requires grid scaling
  (README, paper outline, Word A.4).
- Gate G3 P_D-optimal fusion family: 1318 operating-point edges with 0
  decreasing edges under the optimal rule versus 258 (19.6%) and a maximum
  16.1pp drop under the deflection rule; mean 0.63pp / maximum 21.2pp P_D
  gain per addition edge; closed-form proportional-regime identity with max
  absolute error `1.3e-15` over 960 checks; greedy fusion gain +0.83pp and
  total mean gain +0.99pp (README, paper outline, results JSON).
- Gate G4 expected-P_D greedy: at B=20, +1.14pp mean expected P_D (bootstrap
  CI [0.47, 1.74], 85% win) and +7.56pp worst-target over the proposed
  selector under correlated erasures; the two-candidate hybrid is never
  worse and reaches +1.44pp mean / +6.50pp worst-target; at B=30 gains are
  +0.04pp / +3.05pp, and at B=40 the expected-P_D greedy alone is -0.91pp in
  mean while the hybrid is +0.02pp; bounded-regime submodularity has 0
  violations on 3040 edges and the small-instance greedy ratio is 1.0
  (README, paper outline, results JSON).
- Gate G5 RIS-assisted 6G channel: aligned RIS plus expected-P_D greedy
  raises mean expected P_D by +12.3pp / +10.9pp at B=20/30 over no RIS, with
  worst-target gains +17.8pp / +15.2pp and QoS feasibility at the 0.85 target
  rising from 0% to 95% / 100%; aligned beats random RIS phase by +9.0pp /
  +8.0pp mean (README, paper outline, results JSON).
- Gate G5-Q RIS phase resolution: 1/2/3-bit quantization retains +10.8 /
  +11.9 / +12.2pp mean expected P_D over no RIS at B=20 (worst-target
  +15.1 / +17.1 / +17.6pp), matching the closed-form `sinc^2(1/2^b)`
  array-gain loss, with amortized control overhead 0.16/0.32/0.48 bits per
  frame (README, paper outline, results JSON).
- Gate G5-P physics-based RIS channel: with a 1024-element RIS and aperture
  scale `1e-2`, aligned RIS gives +16.3pp mean and +24.8pp worst-target
  expected P_D over no RIS at B=20 (100% QoS feasibility) and +13.9pp /
  +21.8pp at B=30; a 256-element RIS retains +10.8pp / +13.3pp at B=20
  (README, paper outline, results JSON).
- Gate G5-R joint control/report budget: charging
  `N * phase_bits / coherence_frames` against the same total budget, 3-bit
  phase with remaining bits on reports gives +9.3pp mean and +12.9pp
  worst-target expected P_D over no RIS at total budget 40 (within 0.3pp of
  the free-continuous upper bound), and +8.6pp / +13.4pp at total budget 60
  (README, paper outline, results JSON).
- Gate G5-S joint RIS placement: a finite deployment search selects
  `(0, 20, 8)`, raising worst-target expected P_D from 0.882 at the fixed
  position to 0.952 at total budget 40 (+7.1pp over fixed, +16.7pp over no
  RIS) and +6.6pp over fixed at total budget 60 (README, paper outline,
  results JSON).
- Gate G5-T multigrid placement: one local refinement selects `(0, 30, 6)`,
  raising worst-target expected P_D from 0.952 (coarse) to 0.980 at total
  budget 40 (+2.8pp over coarse, +9.8pp over fixed, +19.5pp over no RIS)
  with 34 deployment evaluations (README, paper outline, results JSON).
- Gate G5-U Lipschitz certificate: empirical `L = 2.97e-3` gives a 1.29pp
  suboptimality bound at spacing 5; the second refinement improves
  worst-target expected P_D by +0.46pp, inside the bound (README, paper
  outline, results JSON).
- Gate G5-V adaptive deployment search: after 251 evaluations it finds
  `(6.25, 39.375, 4.5)` with worst-target expected P_D 0.987 and certifies
  the deployment optimum within +1.03pp under the used Lipschitz constant
  `3.43e-3`; the certificate is reported as bounded, not epsilon-closed
  (README, paper outline, results JSON).
- Gate G5-W epsilon-closed deployment: the localized search closes the
  certificate to 0.10pp in 111 evaluations on a single-seed objective
  (`epsilon_closed = true`).  The 3-seed averaged objective is bounded at
  0.16pp over the original local box after 3001 main-search and 400
  corner-refinement evaluations, and closes to 0.09pp inside a 2 m local box
  in 23 additional evaluations (`local_epsilon_closed = true`); both results
  are stored separately (README, paper outline, results JSON).
- Gate G5-CI paired bootstrap: aligned RIS vs no RIS at B=20 gives +12.33pp
  mean (CI [11.66, 13.02]) and +17.79pp worst-target (CI [15.73, 20.03])
  with 100% win rate; physics 1024-element RIS gives +16.26pp mean (CI
  [14.82, 17.77]) and +24.78pp worst-target (CI [23.24, 26.42]); best
  placement vs fixed gives +7.06pp worst-target (CI [5.76, 8.33]) (README,
  paper outline, results JSON).
- Gate G5-DCI deployment paired bootstrap: G5-T vs fixed gives +4.42pp mean
  (CI [3.09, 5.85]) and +9.85pp worst-target (CI [7.52, 12.08]); G5-V vs
  fixed +10.45pp worst-target (CI [7.78, 13.08]); G5-W vs fixed +9.10pp
  worst-target (CI [7.20, 10.93]), all with 100% win rate (README, paper
  outline, results JSON).
- Gate G5-RF global resource ledger: no-RIS uses 40 report bits / 4648 TB
  symbols, while the G5-T RIS deployment uses 25 report + 12 control bits /
  4645 TB symbols and raises mean expected P_D from 0.863 to 0.990 and
  worst-target from 0.785 to 0.980 (README, paper outline, results JSON).
- Gate G5-SEN sensitivity: at total B=40, mean gain over no RIS rises from
  +1.3pp to +11.0pp as aperture scale goes from 1e-3 to 3e-2, and from
  +1.1pp to +13.3pp as RIS elements go from 64 to 1024 despite report budget
  falling from 39 to 28 bits; larger coherence frames improve both control
  overhead and report budget, while direct-path blockage 0.001/0.01/0.1/1.0
  gives mean gains +10.1/+8.2/+3.5/+2.2pp and worst-target gains
  +10.8/+9.9/+3.2/-0.2pp (README, paper outline, results JSON).
- Gate G5-SOTA baselines: at total B=40 with 12 seeds, the proposed chain
  beats RIS deflection Top-K by +0.68pp mean and +1.63pp worst-target
  expected P_D, no-RIS deflection Top-K by +15.2pp/+27.5pp, random RIS by
  +14.4pp/+25.4pp, uniform soft by +21.8pp/+46.1pp, and exact 1-bit counting
  fusion by +75.7pp/+79.9pp (no RIS) and +52.1pp/+64.5pp (RIS), all with
  100% win rate and positive bootstrap CIs; the fusion-only gain on the same
  proposed schedule is +0.25pp/+0.45pp (README, paper outline, results
  JSON).
- Gate G6 budget saturation: without RIS the worst-target expected P_D
  saturates near 0.788 and no budget up to 44 reaches the 0.85 QoS target;
  with G5-T RIS, total budget 20 (8 report bits) gives 100% QoS feasibility,
  and discrete add/remove/swap ascent from forward greedy gives zero gain in
  every audited cell, showing the limiting constraint is sensing
  architecture rather than selection local optimality (README, paper
  outline, results JSON).
- Gate G7 shared-phase gradient: the worst-array-power surrogate improves
  the surrogate but degrades system P_D, so it is reported as a negative
  result; system-level grid-plus-refine optimization recovers the weak-target
  cosine and gives +8.2pp worst-target P_D over no RIS at B=20, while a
  single shared beam remains 12.3pp below per-target ideal phase and reaches
  only 50% QoS at B=20 (README, paper outline, results JSON).
- Gate G8 exact quota selection: with equal report costs, exhaustive
  per-target subset evaluation plus global quota search exactly matches
  forward greedy in every audited cell (0.0pp gain), so the selection layer
  has no remaining headroom and the gap to all-scheduled is architectural
  (README, paper outline, results JSON).
- Gate G9 subarray multi-beam: at B=28 the optimized aperture allocation
  `(6,149,101)` raises worst-target expected P_D to 0.913 with 100% QoS,
  +5.2pp over single shared weak-aligned and +13.7pp over no-RIS, while
  remaining 6.7pp below the per-target ideal upper bound (README, paper
  outline, results JSON).
- Gate G10 steering optimization: with G9 allocations fixed, coordinate
  ascent over block steering cosines improves worst-target expected P_D to
  0.858/0.916/0.935 at B=20/28/40, +0.41/+0.23/+0.14pp over G9, while the
  per-target ideal gap is 9.6/6.5/4.8pp (README, paper outline, results
  JSON).
- Gate G11 aperture scaling: under the exact control-overhead identity,
  `N=1024`, 3-bit phase, `C=256`, and equal subarray allocation reach 100%
  QoS at B=20 with only 8 report bits and worst-target expected P_D 0.982;
  `N=512` reaches 0.943, while the original N=256 configuration remains at
  50%, confirming the architecture-limited conclusion (README, paper
  outline, results JSON).
- Gate G12 derived architecture: the weak-target surrogate
  `J(N) = beta (1 + kappa N^2)^2 (R - LN)` and its quadratic first-order
  condition give `N* = 1016` for B=20, b=1, C=64 (rounded 1024, 100% QoS)
  and `N* = 1363` for b=3, C=256 (rounded 1344, worst P_D 0.974), so the
  architecture variables are determined by the objective, not by exhaustive
  search (README, paper outline, results JSON).
- Gate G13 water-filling allocation: the max-min deflection surrogate
  `D_q = beta_q (1 + kappa_q a_q^2)^2` drives aperture from highest-D to
  lowest-D targets; exact validation raises worst P_D from 0.900 to 0.911
  (`N=1024,b=1,C=64`), 0.974 to 0.992 (`N=1344,b=3,C=256`), and 0.999599 to
  0.999995 (`N=2048,b=3,C=256,B=40`), all 100% QoS; the marginal-equalizing
  KKT variant was tested and rejected (README, paper outline, results JSON).
- Gate G14 exact-array allocation: the exact surrogate
  `D_q = beta_q (1 + K0_q N^2 G_q(a))^2` raises the surrogate minimum in all
  tested cells, but exact system P_D does not consistently improve: 0.8pp
  worse at N=1024, statistically unchanged at N=1344, and 0.00008pp better
  at N=2048; this is recorded as a negative/equivocal result (README, paper
  outline, results JSON).
- Gate G15 greedy-aware allocation: coordinate ascent on the exact
  expected-P_D objective `F(a)` improves worst P_D from 0.911 to 0.924
  (`N=1024`), 0.911 to 0.927 (`N=704`), and 0.981 to 0.985 (`N=960,B=28`),
  while the coarse 8-element search at `N=2048,B=40` ends 0.0018pp below the
  exact-surrogate allocation (README, paper outline, results JSON).
- Gate G16 single-move certificate: 4/2/1-element refinement improves every
  configuration and `exact_single_move_gradients` verifies
  `local_optimal=true` with nonpositive maximum gradient in all five cells
  (README, paper outline, results JSON).
- Gate G17 multi-block certificate: all zero-sum reallocations moving up to
  three elements are evaluated exactly; four cells are already local optima
  and `N=2048,B=40` improves from 0.999986 to 0.999988 in 7 rounds, with all
  cells certified `local_optimal=true` for the T<=3 neighborhood (README,
  paper outline, results JSON).
- Gate G18 joint placement-allocation: alternating exact-system coordinate
  ascent over position and T<=3 allocation improves all three configurations
  to 0.925224 (`N=1024`), 0.992907 (`N=1344`), and 0.999997
  (`N=2048,B=40`), with both allocation and 0.5m-position local certificates
  true (README, paper outline, results JSON).
- Gate G19 progressive decentralization: at B=40/N=2048 the loss from global
  scheduling to owner-only is only 0.014pp worst, while 1-bit hard decisions
  lose 18.8pp; at B=20 with 4 report bits, centralized soft fusion equals
  owner-only because 5-bit reports are infeasible, and 1-bit hard decisions
  cannot meet the global P_FA=0.05 with one vote per target (QoS 0%); the
  earlier +6.0pp claim was corrected after enforcing P_FA (README, paper
  outline, results JSON).
- Gate G20 amplified distributed detection: optimizing local P_FA and the
  counting threshold raises 1-bit worst P_D from 0.812 to 0.944 at
  B=40/N=2048 (QoS 0% to 100%), still below centralized 0.999997; at B=20
  with one vote per target the counting rule is P_FA-infeasible (README,
  paper outline, results JSON).
- Gate G21 network decentralization: peer majority over all M=8 UAVs with
  zero report bits reaches worst P_D 0.955/0.998/0.9999977 at the three
  tested configurations, matching or exceeding centralized soft fusion with
  100% QoS (README, paper outline, results JSON).
- Gate G22 degraded consensus: observability 0.75 or link reliability 0.8
  drop peer majority below centralized soft, while three-hop 0.8 recovery
  reaches 0.9998 and severe degradation drops to 0.877 at B=40/N=2048
  (README, paper outline, results JSON).
- Gate G23 correlated consensus: common failure 0.2/0.4 gives 0.977/0.909,
  heterogeneous observability gives 0.936, and severe combination gives
  0.858 at B=40/N=2048, all below centralized 0.999997 (README, paper
  outline, results JSON).
- Gate G24 scalability: for Q=2/4/6 and M/Q=1/2/3, RIS ideal phase is the
  most robust (100% QoS except Q=6,M=6), peer majority needs M/Q>=3 for
  high QoS, and no-RIS is topology-sensitive with Q=2,M=4 dropping to 0.460
  (README, paper outline, results JSON).
- Gate G25 scaled G18: water-filling allocation plus exact position ascent
  keeps 100% QoS except Q=6,M=6, reaches 0.922 at Q=6,M=12 (peer 0.792,
  ideal 0.934), and 0.964 at Q=4,M=8 (peer 0.915) (README, paper outline,
  results JSON).
- Gate G26 mobility/blockage: rotating UAVs and sinusoidal weak-target
  blockage give no-RIS worst QoS 0%, RIS ideal 100%, static subarray
  68.75%, and adaptive subarray 81.25% with worst P_D 0.847 (README, paper
  outline, results JSON).
- Gate G27 multi-RIS: with total aperture 256, one RIS reaches worst P_D
  0.980 at B=28 versus 0.923/0.927 for two/three RISs, so splitting the
  aperture loses coherent gain and diversity only partially compensates
  (README, paper outline, results JSON).
- Gate G28 split/placement: equal two-RIS split gives 0.924; local
  optimization finds `(8,248)` at `(4,42,2)` and reaches 0.986, slightly
  exceeding single-RIS 0.981 (README, paper outline, results JSON).
- Gate G29 variable-rate reporting: fixed 5-bit soft is best at B=20/28
  (0.953/0.977), adaptive soft beats it at B=40 (0.988 vs 0.981), and 1-bit
  hard remains weakest (README, paper outline, results JSON).
- Gate G30 global rate optimization: coordinate ascent over per-UAV
  quantizer bits raises worst P_D to 0.988 at B=28 and 0.991 at B=40, with
  single-rate-change local optimality true (README, paper outline, results
  JSON).
- Gate G31 hybrid fusion: exact Gaussian-plus-hard LLR fusion reaches
  0.977/0.969 at B=28/40, below pure soft 5-bit 0.977/0.981 but above
  hard-only 0.843/0.736 (README, paper outline, results JSON).
- Gate G32 interference: INR=0 dB only RIS ideal reaches QoS 100%; INR=3 dB
  fails all architectures; INR=10/20 dB drops worst P_D below 0.2/0.06
  (README, paper outline, results JSON).
- Gate G33 spatial interference: per-UAV INR follows free-space path loss;
  no-RIS fails all strengths, fixed RIS keeps 100% QoS, and optimized RIS
  position adds +0.3pp worst P_D (README, paper outline, results JSON).
- Gate G34 multi-interference: INR is a three-source path-loss sum (mean
  0.087); no-RIS fails, fixed RIS 0.983, optimized placement 0.987, all with
  100% QoS for RIS (README, paper outline, results JSON).
- Gate G35 ULA/UPA: with 256 elements, UPA is nearly identical to ULA in
  clean and interference scenarios (e.g., B=40 interference 0.98691 vs
  0.98691); 2-D benefit requires elevation or null-steering (README, paper
  outline, results JSON).
- Gate G36 null-steering: optimized UPA phases reduce reflected INR from
  0.0267 to 0.0106 (-60%) with target gain 1.000 to 0.984; B=40 worst P_D
  improves from 0.98112 to 0.98216 (README, paper outline, results JSON).
- Gate G37 quantized null-steering: direct discrete coordinate ascent reaches
  reflected INR 0.01052 and B=40 P_D 0.982166, slightly better than
  continuous-then-quantized (README, paper outline, results JSON).
- Gate G38 joint nulling/placement: optimized position raises B=40 worst P_D
  from 0.98217 to 0.98481 with reflected INR rising 0.0105 to 0.0296
  (README, paper outline, results JSON).
- Gate G39 relaxed distributed: at QoS 0.70-0.80 and budgets 20-28, peer
  clean/multi-hop and optimized hard are feasible; peer multi-hop stays at
  0.953 and hard around 0.84-0.86 (README, paper outline, results JSON).
- Gate G40 low-budget distributed: with N=128, interference, and B=12,
  centralized drops to 0.786 while peer clean/multi-hop reach 0.858/0.855;
  hard optimized 0.765 remains feasible (README, paper outline, results
  JSON).
- Gate G41 parity boundary: theoretical M_min 14-17; consensus wins at B=8/12
  for M>=8 and B=8 for M=6, while centralized leads at B>=16 (README, paper
  outline, results JSON).
- Gate G42 optimized boundary: minimizing M_min over local P_FA lowers it by
  9-13% (e.g., M=16 13.70 to 12.14), closer to exact wins (README, paper
  outline, results JSON).
- Gate G43 exact boundary: exact Poisson-binomial feasibility starts at M=6,
  matching empirical wins, while Gaussian M_min was 13.36 (README, paper
  outline, results JSON).
- Gate G44 information budget: soft P_D rises monotonically with normalized
  information from 0.774 at 0.507 to 0.933 at 0.946; consensus retains nonzero
  information when soft reports are unaffordable (README, paper outline,
  results JSON).
- Gate G45 resource law (negative): the closed-form
  `Phi((sqrt(d0(1+n)g^2)-z)/sqrt(c))` overestimates P_D by up to 30pp and
  saturates to 1 at N>=128, so it is rejected (README, paper outline, results
  JSON).
- Gate G46 exact information budget: soft raw rho overestimates
  P_D-consistent `rho_exact` by 2.38-2.78x; under enforced report budgets,
  soft P_D rises 0.774->0.933 as `rho_exact` rises 0.205->0.351, peer
  consensus is 0.881 at `rho_exact=0.284`, and hard fusion is at most
  `rho_exact=0.199` (README, paper outline, results JSON).
- Gate G47 architecture switch: exact mode selection picks peer at B=8/12
  (+10.68/+5.68pp worst P_D, 0.85 QoS becomes feasible) and centralized soft
  at B>=16; the fixed `report_budget < 10` policy matches exact choices in
  the audit (README, paper outline, results JSON).
- Gate G48 target-wise switch: per-target `max(soft_q, peer_q)` is never
  worse than the global switch and adds +0.49/+1.55/+1.55pp worst P_D at
  B=12/16/20, with peer selected on 92%/83%/50% of targets at B=8/12/16
  (README, paper outline, results JSON).
- Gate G49 soft-report reallocation: freed peer-target soft bits are added
  back to centralized targets with exact expected-P_D marginals; the update
  is additive and nondecreasing, adding +0.75pp at B=16/20 over G48 and
  +1.55/+0.85pp at B=28/40 (README, paper outline, results JSON).
- Gate G50 two-sided mode ascent: a limiting peer target may switch back to
  centralized soft only when the switch strictly raises the worst P_D; this
  adds +0.39pp at B=12 over G48 (0.8858 -> 0.8898) with 3.75 used report
  bits on average, and matches G49 at B>=16 (README, paper outline, results
  JSON).
- Gate G51 stochastic mobility: AR(1) random trajectories and blockage give
  no-RIS worst-over-time P_D 0.524, static RIS mode ascent 0.705, latency-1
  RIS 0.722, ideal target-wise 0.847, and ideal mode ascent 0.852 with
  90.625% QoS; latency-1 beats static by +1.64pp (README, paper outline,
  results JSON).
- Gate G52 prediction-aware RIS: conditional-mean AR(1) prediction raises
  latency-1 worst-over-time P_D from 0.7217 to 0.7283 (+0.65pp) and QoS from
  43.75% to 46.875% (README, paper outline, results JSON).
- Gate G53 multi-step prediction: h-step MMSE over stale-phase worst P_D
  gains +0.65/+3.24/+5.24pp for h=1/2/3, with error covariance scale
  `1-rho^{2h}`; exact per-frame horizon selection adds +0.86pp over the best
  fixed MMSE, and hysteresis delta=0.02 halves switches with bounded loss
  (README, paper outline, results JSON); under a 6-bit control budget,
  per-switch costs of 1/3/6 bits select delta 0.00/0.03/0.05.
- Gate G54 covariance-aware phase (negative): expected-gain robust phase is
  monotone in its surrogate but degrades exact worst P_D from 0.7200 to
  0.6557 and QoS from 43.75% to 37.5% at h=3, so MMSE phase is kept
  (README, paper outline, results JSON).
- Documentation file list now includes PAPER_OUTLINE.md, SYSTEM_MODEL.md,
  RELATED_WORK.md, FORMAL_PROOFS.md, PAPER_DRAFT.md, and all three Word
  appendices.

## Residual open items (not contradictions)

- Equal bandwidth/frame-budget/communication-rate accounting now has a
  same-scale resource table; full time-bandwidth and latency accounting is
  still incomplete.
- Formal G1-A run at 10 000 trials per hypothesis with bootstrap CI.
- G2 correlated audit at 10-20 seeds with win-rate CI.
- Same angle-DD collision decomposition and strong FWER.
- Bandwidth-consistent SDR front end.

## Deep audit: Gate G3 P_D-optimal linear fusion (2026-08-05)

### Independent numerical verification

- Random-direction global search on 828 operating-point cases (`P_D >= 0.5`)
  found no case where a sampled linear direction beats the one-parameter
  family by more than `1e-6`; the maximum observed gap was `8.7e-8`.
- Grid-resolution audit at 1024/2048/4096 points with local refinement found
  0 decreasing `P_D` edges at `1e-9` tolerance over roughly 1100-1900
  addition edges per run.
- Independent seed `20260806` with 40 instances gave 0 decreasing edges on
  409 operating-point edges, 17.1% decreasing edges and a 16.5pp maximum
  drop for the deflection rule, and the same qualitative gain pattern.
- `optimal_gaussian_weights` reproduces the returned `P_D` to `7.2e-16`
  maximum error over 200 proportional-regime checks.

### Issues found and fixed during this audit

1. The README/PAPER weight formula wrote
   `Sigma0^-1/2 (Q + mu I)^-1 Sigma0^-1/2 delta`, which is not the
   Cholesky-whitened direction used by the implementation.  Corrected to
   `w(mu) = L^-T (Q + mu I)^-1 L^-1 delta` with `Sigma0 = L L^T` and
   `Q = L^-1 Sigma1 L^-T`.
2. A grid-only search showed a `7.5e-7` numerical `P_D` drop at `grid=1024`.
   Added bounded local refinement around the best grid point; the drop
   disappears at `1e-9` tolerance.
3. The first refinement version returned the grid `P_D` while the returned
   weights used the refined `mu`, so the two could disagree.  The function
   now returns the refined shift so weights and `P_D` are consistent.
4. The gate was tied to one fixed RNG stream.  Added `--seed` and validated
   an independent seed run.

### Residual limitations

- The formal KKT guarantee is exact for `P_D > 0.5`; `P_D = 0.5` is audited
  numerically as the inclusive boundary.
- Monotonicity is proven for deterministic received report sets.  Expected
  `P_D` under random erasures/BSC is not claimed by this gate.
- The greedy-level numbers come from a controlled deterministic-report model
  (`success_prob = 1`, unit report cost); the formal contribution is the
  monotone fusion rule, not a universal scheduling gain.
- The gate assumes exact H0/H1 moment estimates.  Propagation of
  calibration/report-channel error into the new weights is not re-audited by
  G3; G1-A/B remain the evidence-calibration gates.
- The implementation is a numerical optimizer over the family, so the
  guarantee holds up to optimizer accuracy; the random-direction and
  grid-resolution audits bound that error.

## Deep audit: Gate G4 expected-P_D selection (2026-08-05)

### What was checked

- Objective construction: expected `P_D` over the model's exact independent,
  common-state, and grouped reception laws, with the Gate G3 monotone fusion
  family used for every received set.
- Monotonicity: every fixed-pattern `P_D` is set-monotone at operating
  points, so the expectation is set-monotone; this is inherited from Gate G3
  and covered by `tests/test_expected_pd.py`.
- Bounded-regime submodularity: in the proportional-covariance,
  strong-evidence regime the per-pattern `P_D` is concave in a modular
  deflection, so the expected objective is monotone submodular.  The audit
  found 0 diminishing-return violations on 1520 edges for both
  `Sigma1 = Sigma0` and `Sigma1 = 0.5 Sigma0` (3040 edges total).
- Approximation ratio: on 20 small single-target instances the expected-P_D
  greedy matches the exhaustive oracle (ratio 1.0), above the `1 - 1/e`
  threshold.
- Budget behavior: at B=20 the new selector gives +1.14pp mean and +7.56pp
  worst-target over the proposed selector with a positive bootstrap CI; the
  hybrid policy (keep the better of proposed and expected-P_D greedy by
  expected P_D) is never worse and gives +1.44pp mean at B=20.  At B=40 the
  expected-P_D greedy alone is -0.91pp in mean, which is why the audited
  system claim uses the hybrid and does not claim universal dominance.

### Residual limitations

- Expected-P_D greedy is strongest in tight-budget/non-saturated regimes; at
  saturated budgets it is only a candidate and must be combined with the
  existing selector.
- The submodularity guarantee is bounded to the proportional-covariance,
  strong-evidence regime; arbitrary H1 covariance is audited only
  empirically.
- The gate assumes exact H0/H1 moments and exact reception-law parameters;
  estimation error is not re-audited here.
- Exact pattern enumeration is used up to 14 reports per target; larger
  systems use sample-average approximation, whose tolerance is covered by
  `test_saa_matches_exact_for_small_set`.

## Deep audit: Gate G5 RIS-assisted 6G channel (2026-08-05)

### What was checked

- Channel construction: the RIS phase profile is a deterministic beam toward
  each target; array gain is 1 when aligned and decreases smoothly with
  mismatch.  The gain rule is additive in SNR power, `1 + (strength * array
  gain)^2`, so alignment is monotone in link quality and cannot degrade a
  link.
- Injection path: the gain matrix multiplies pre-quantization effective SNR
  in `build_models`; an identity gain reproduces the baseline models exactly
  (`test_snr_gain_identity_matches_baseline`).
- Geometry/phase sanity: aligned beam array gain is 1
  (`test_ris_beam_aligned_gain_is_one`); the weak-target gain exceeds
  strong-target gains (`test_ris_gain_matrix_boosts_weak_target`).
- 20-seed audit: aligned RIS versus no RIS gives +12.3pp/+10.9pp mean
  expected P_D at B=20/30, +17.8pp/+15.2pp worst-target, and QoS feasibility
  rising from 0% to 95%/100%; aligned beats random phase by +9.0pp/+8.0pp
  mean.

### Issues found and fixed during this audit

1. A first coherent-combining gain rule
   `|h_dir + h_ris|^2 / |h_dir|^2` could boost the weak target by more than
   1000x and, because the moment-matched Gaussian detector is not monotone
   in coherent amplitude, could lower the P_D of strong targets.  This
   violates the basic sensing principle that a better channel should not
   hurt detection and was replaced by the additive-power rule.

### Residual limitations

- The RIS channel is controlled, not a full cascaded-channel SDR: per-element
  phase quantization, RIS control overhead, and explicit path-loss physics
  are not modeled.
- The gate uses one RIS position and a per-target aligned beam codebook;
  joint RIS placement and phase optimization remain open.
- Paired bootstrap CIs for the aligned-versus-no-RIS gains are listed as a
  required experiment before submission.

## Deep audit: Gate G5-Q RIS phase resolution (2026-08-05)

### What was checked

- Closed-form phase-quantization loss: with b-bit uniform phase quantization,
  the mean power gain scales by `sinc^2(1/2^b)`.  The test averages 2000
  random steering realizations and matches the formula within 0.05 for
  1/2/3/4 bits.
- Performance retention: at B=20, 1/2/3-bit RIS still gives +10.8/+11.9/
  +12.2pp mean expected P_D over no RIS and +15.1/+17.1/+17.6pp worst-target,
  versus +12.3/+17.8pp for ideal continuous phase.
- Control-overhead ledger: `N * phase_bits / coherence_frames` bits per
  frame; for 16 elements over 100 frames this is 0.16/0.32/0.48 bits per
  frame, negligible relative to the sensing report budget.

### Residual limitations

- The quantization formula assumes uniformly distributed phase errors and
  ideal per-element independence; finite-N and correlated element behavior
  are covered only empirically.
- Control overhead is amortized over a fixed 100-frame coherence block and
  is not yet coupled to the sensing report budget in a joint resource
  optimization.

## Deep audit: Gate G5-P physics-based RIS channel (2026-08-05)

### What was checked

- Direct path uses the two-way bistatic radar law
  `1 / (R_tx^2 R_rx^2)`; the RIS path uses
  `N^2 array_gain^2 aperture_scale / (R_1^2 R_2^2 R_3^2)`, so the model
  follows standard propagation loss rather than a free multiplicative gain.
- Monotonicity in array alignment and elements: tests verify the weak-target
  gain increases with `N` and aligned phase beats random phase; all gains are
  at least 1, so the channel cannot degrade a link.
- 12-seed audit: at B=20, `N=1024`, aperture scale `1e-2` gives +16.3pp mean
  and +24.8pp worst-target expected P_D over no RIS with 100% QoS
  feasibility; `N=256` retains +10.8pp / +13.3pp.

### Residual limitations

- The model uses scalar path losses with an ideal point-target and omits
  per-element mutual coupling, polarization, and waveform-level RIS
  responses.
- The direct-path blockage and aperture scale are controlled parameters;
  their physical calibration against a real RIS testbed is future work.
- The gate sweeps element count and aperture scale but not RIS placement;
  joint placement/phase optimization remains open.

## Deep audit: Gate G5-R joint control/report budget (2026-08-05)

### What was checked

- Budget identity: `B_total = B_report + N * phase_bits / coherence_frames`;
  phase options include the free-continuous upper bound and realizable 1/2/3
  bit quantizers.
- 12-seed physics-channel audit: the best realizable allocation is 3-bit
  phase with the remaining budget on reports.  At total budget 40, 64-frame
  coherence, it gives +9.3pp mean and +12.9pp worst-target expected P_D over
  no RIS, within 0.3pp of the free-continuous upper bound; at total budget
  60, +8.6pp / +13.4pp.
- All phase resolutions and report budgets remain feasible, and the selected
  policy always uses the exact expected-P_D objective with the monotone
  fusion family.

### Residual limitations

- The control overhead is amortized over a fixed coherence block and assumes
  one phase configuration per block; dynamic reconfiguration cost is not
  modeled.
- The joint search is a finite sweep over phase resolution and report
  schedules; an analytical allocation rule or convex relaxation is future
  work.
- Paired bootstrap CIs for G5-P/G5-R gains are still required before
  submission.

## Deep audit: Gate G5-S joint RIS placement (2026-08-05)

### What was checked

- Candidate deployment set of seven positions spanning near/away from the
  blocked weak target; each position uses the target-aligned phase codebook,
  the G5-P physics channel, and the G5-R budget identity.
- Summary aggregation was corrected during the audit: the best position is
  selected by mean worst-target expected P_D over all 12 seeds, not by a
  single seed's best row.
- 12-seed audit: `(0, 20, 8)` gives worst-target expected P_D 0.952 at total
  budget 40 (64-frame coherence), +7.1pp over the fixed position and +16.7pp
  over no RIS; at total budget 60 the worst-target gain over fixed is +6.6pp.

### Residual limitations

- The placement search is a finite candidate set, not continuous
  optimization; a gradient or grid-refinement method is future work.
- The best position depends on the fixed weak-target geometry; dynamic
  target motion and RIS repositioning cost are not modeled.
- Paired bootstrap CIs for G5-S gains are still required before submission.

## Deep audit: Gate G5-T multigrid RIS placement (2026-08-05)

### What was checked

- Coarse-to-fine structure: 7 coarse deployments are evaluated over the seed
  set, the best coarse position is refined with a 27-point local grid, and
  the best fine deployment is selected by mean worst-target expected P_D.
- 12-seed audit: `(0, 30, 6)` gives worst-target expected P_D 0.980 at total
  budget 40 (64-frame coherence), +2.8pp over the G5-S coarse optimum,
  +9.8pp over the fixed position, and +19.5pp over no RIS.
- Evaluation cost: 34 deployments total (7 coarse + 27 fine), each followed
  by the expected-P_D greedy; the audit records this count in the JSON.

### Residual limitations

- The multigrid method is a finite refinement, not a gradient or exact
  continuous optimizer; a convergence guarantee under a Lipschitz condition
  is not yet written.
- The fine grid is axis-aligned with fixed spacing; rotated or adaptive
  refinement is future work.
- Paired bootstrap CIs for G5-T gains are still required before submission.

## Deep audit: Gate G5-U Lipschitz deployment certificate (2026-08-05)

### What was checked

- Lemma statement: for an `L`-Lipschitz objective, a grid with spacing `h`
  has suboptimality at most `L h sqrt(d)/2`.  The proof follows from
  `||x - grid(x)||_2 <= h sqrt(d)/2` plus Lipschitz continuity.
- Unit tests verify the lemma on `f(x) = ||x - c||_2` with `L = 1` and a
  2-D grid, and verify the empirical Lipschitz estimator returns 1.
- 6-seed physics-channel audit: empirical `L = 2.97e-3`; the second
  refinement to `(0, 35, 5)` improves worst-target expected P_D from 0.983 to
  0.988 (+0.46pp), inside the 1.29pp bound at spacing 5 and the 2.57pp bound
  at spacing 10.

### Residual limitations

- The empirical Lipschitz constant is computed from evaluated deployments and
  is a valid but possibly loose bound; it is not a proven global constant.
- The certificate bounds grid-search loss only within the refined box; global
  placement search is not certified.
- The lemma assumes an exact Lipschitz objective, while the pipeline uses
  greedy approximations and finite seeds, so the empirical check is evidence
  rather than a formal guarantee.

## Deep audit: Gate G5-V adaptive deployment search (2026-08-05)

### What was checked

- Branch-and-bound correctness: each box upper bound is `f(c) + L||h||_2`,
  the box with the largest upper bound is split along its longest axis, and
  both child centers are evaluated; the global upper bound is the maximum
  over all remaining boxes.
- Unit tests verify epsilon-optimality on `f(x) = ||x - c||_2` with `L = 1`,
  including input validation.
- 3-seed physics-channel audit: after 251 evaluations, best deployment
  `(6.25, 39.375, 4.5)` has worst-target expected P_D 0.987 and certificate
  upper bound 0.997 (+1.03pp) under the used Lipschitz constant `3.43e-3`.

### Residual limitations

- The certificate did not close to the requested `1e-3` within the
  evaluation budget; the gate reports the bounded gap honestly instead of
  claiming convergence.
- The used Lipschitz constant is a doubled empirical estimate, so the
  certificate is valid under that constant, not under a proven global one.
- The search is restricted to a local deployment box around the known good
  region; a global adaptive search with the same budget is future work.

## Deep audit: Gate G5-W epsilon-closed deployment (2026-08-05)

### What was checked

- Two-phase structure: the finite candidate search localizes, then the
  adaptive search runs inside a small box around the known good region.
- Single-seed objective: certificate closes to 0.10pp within 111 evaluations
  (`epsilon_closed = true`), best deployment `(11.875, 34.21875, 6.5)` with
  worst-target expected P_D 0.983.
- 3-seed averaged objective: certificate gap 0.16pp after 3001 main-search
  evaluations plus 400 corner-refinement evaluations, using a
  coordinate-wise Lipschitz box bound that is never looser than the radial
  bound.  A second branch-and-bound run inside a 2 m box around the best
  point closes to 0.09pp in 23 evaluations (`local_epsilon_closed = true`);
  the original-box gap is stored separately so the closure claim is not
  over-stated.

### Residual limitations

- The single-seed certificate is closed; the multi-seed certificate is
  closed only inside a 2 m local box, while the original local-box gap
  remains 0.16pp.
- The local box is chosen from prior G5-T/G5-V audits; the procedure is not
  fully autonomous from box to box.
- The Lipschitz constant remains an empirical (doubled) estimate, not a
  proven global constant.

## Deep audit: Gate G5-CI paired bootstrap (2026-08-05)

### What was checked

- Paired per-seed differences are computed from the stored G5/G5-P/G5-S/G5-R
  rows, and 5000-replicate bootstrap 95% CIs and win rates are reported for
  mean and worst-target expected P_D.
- All primary comparisons have positive CIs and 100% win rates: aligned vs
  no RIS (B=20), physics 1024-element RIS, best placement vs fixed, and
  joint 3-bit allocation vs no RIS.
- One secondary cell (physics N=256, aperture `1e-3`, B=30 mean) has win
  rate 0.917 and a positive CI, which is reported without over-claiming.

### Residual limitations

- Bootstrap intervals assume the stored seeds are independent; the rows are
  reused from earlier gates rather than freshly resampled.
- The deployment CIs are computed by re-running the physics objective for
  the fixed deployments rather than by re-reading the original G5-T/U/V/W
  per-seed files.

## Deep audit: Gate G5-DCI deployment paired bootstrap (2026-08-05)

### What was checked

- Per-seed expected-P_D rows are generated for no-RIS, fixed, G5-S, G5-T,
  G5-V, and G5-W deployments under the same joint control/report budget
  (total 40, coherence 64, N=256, 3-bit phase).
- Paired bootstrap 95% CIs and win rates are computed for each deployment
  versus no-RIS and versus the fixed position.
- All deployment gains are positive with 100% win rate; G5-T vs fixed gives
  +9.85pp worst-target (CI [7.52, 12.08]), G5-V +10.45pp (CI [7.78, 13.08]),
  G5-W +9.10pp (CI [7.20, 10.93]).

### Residual limitations

- The deployment CIs reuse the same seed stream as other gates and are not
  freshly resampled scenes.
- The G5-W best deployment used here is the single-seed epsilon-closed
  point; a multi-seed deployment optimum would need a separate search.

## Deep audit: Gate G5-RF global resource ledger (2026-08-05)

### What was checked

- Resource identities: `B_total = B_report + N * phase_bits /
  coherence_frames`; report/control bits map to time-bandwidth symbols at a
  conservative 1-symbol-per-bit rate; sensing and identity TB reuse the
  G0-C ledger values (`frames * 512` and `num_uavs * 512`).
- 12-seed audit: no-RIS uses 40 report bits / 4648 TB symbols and reaches
  mean expected P_D 0.863 with QoS feasibility 0%; the G5-T RIS deployment
  uses 25 report + 12 control bits / 4645 TB symbols and reaches mean 0.990,
  worst 0.980, QoS feasibility 100%.
- The RIS gain is not resource-driven in reverse: total time-bandwidth and
  total occupation are slightly lower for RIS.

### Residual limitations

- The 1-symbol-per-bit mapping is conservative but not waveform-derived; a
  full SDR-grade report/control symbol mapping is future work.
- Sensing OTFS-grid scaling under a fixed total TB remains open; only the
  report/control plane is closed by this ledger.
- The ledger uses the G5-T deployment; other deployments may have different
  exact bit counts, though the qualitative conclusion is unchanged.

## Deep audit: Gate G5-SEN RIS parameter sensitivity (2026-08-05)

### What was checked

- One sweep per parameter with all other parameters fixed at the mainline:
  `aperture_scale in {1e-3, 3e-3, 1e-2, 3e-2}`, `N_ris in {64,128,256,512,
  1024}`, `coherence_frames in {16,32,64,128,256}`, and
  `direct_blockage in {0.001,0.01,0.1,1.0}`.
- Every cell uses `B_total = 40`, 3-bit phase, the G5-P physics channel, the
  expected-P_D greedy selector, and exact `B_report = B_total -
  N_ris * phase_bits / coherence_frames`.
- Six seeds per cell with paired bootstrap 95% CIs on aligned-versus-no-RIS
  mean and worst-target gains.

### Findings

- The gain is monotone in aperture scale and element count.  At
  `aperture_scale = 3e-2` the mean gain is +11.04pp and worst-target +14.07pp;
  at `N_ris = 1024` (with report budget reduced to 28 bits by control
  overhead) the mean gain is +13.35pp and worst-target +20.67pp.
- Coherence frames trade control overhead against report budget.  `C=16` is
  infeasible under the ledger; increasing `C` from 32 to 256 raises mean gain
  from +7.27pp to +8.70pp and worst-target from +8.89pp to +10.49pp.
- Direct-path blockage is the regime condition for the RIS claim: mean gain
  falls from +10.06pp at blockage 0.001 to +2.16pp at blockage 1.0, and the
  worst-target gain CI crosses zero at blockage 1.0.  The RIS mechanism is
  therefore claimed for a blocked weak target, not as a universal channel
  improvement.

### Residual limitations

- The sweep is a 6-seed draft; the paper should either use the 12-seed
  mainline convention or state the smaller seed count explicitly.
- The `coherence_frames = 16` cell is recorded as infeasible rather than
  allocated by dropping phase bits; a joint phase/coherence search is still
  open.
- The negative worst-target gain at blockage 1.0 combines the controlled
  report-budget reduction with greedy scheduling, so it is not a channel
  degradation claim.

## Deep audit: Gate G5-SOTA literature-style baselines (2026-08-05)

### What was checked

- All methods share the G5-T deployment `(0, 30, 6)`, 256 RIS elements, 3-bit
  phase, total budget 40, coherence 64, and 28 report bits.
- Soft baselines are static deflection Top-K and uniform one-report soft
  allocation with deflection-optimal fusion.
- The 1-bit counting baseline uses a local per-report false-alarm rate of
  0.1 and an exact Poisson-binomial count distribution over the same
  reception law; the fusion threshold is the smallest vote count with
  `P_FA <= 0.05`, giving measured fusion-level P_FA near 0.008.
- Paired bootstrap 95% CIs and win rates are computed over 12 seeds.

### Findings

- The proposed chain dominates every baseline on mean and worst-target
  expected P_D with 100% win rate and positive CIs.
- The smallest margin is against the strongest soft baseline (RIS plus
  deflection Top-K): +0.68pp mean and +1.63pp worst-target, consistent with
  the G3/G4 mechanism claims rather than with an inflated universal gain.
- Replacing deflection fusion with the P_D-optimal family on the same
  proposed schedule gives +0.25pp mean and +0.45pp worst-target, isolating
  the fusion contribution from selection.

### Residual limitations

- The 1-bit baseline uses post-report moment-matched thresholds and sends
  21 of the 28 available bits (7 reports per target); it is a fair but
  conservative literature-style baseline, not an optimized hard-decision
  scheme.
- External numerical results from other systems are not reproduced here;
  the comparison is against re-implemented method components under the same
  channel and budget assumptions.

## Deep audit: Gate G6 budget saturation frontier (2026-08-05)

### What was checked

- Total budgets 20/24/28/32/36/40/44 with and without the G5-T RIS
  deployment, 6 seeds, and the exact joint control/report identity.
- Three selection states per cell: forward expected-P_D greedy, discrete
  coordinate ascent initialized from that greedy, and the all-scheduled
  upper bound.
- QoS target 0.85 on worst-target expected P_D.

### Findings

- No-RIS never reaches the QoS target in the audited budget range; its
  worst-target value saturates near 0.788.
- RIS reaches 100% QoS feasibility at total budget 20, i.e. 8 report bits
  after 12 control bits, with all 6 seeds achieving the target at that
  budget.
- Discrete ascent made zero improvements, so the forward greedy is already a
  single-move local optimum under the exact expected-P_D objective in these
  cells.  This is a negative result for selection refinement and a positive
  pointer to continuous RIS phase/placement optimization.

### Residual limitations

- The sweep uses 6 seeds and one RIS deployment; a 12-seed run and a wider
  placement set should confirm the minimum-budget claim before submission.
- Discrete ascent evaluates only single-report moves; larger multi-report
  moves or a continuous relaxation could still improve on the local optimum.

## Deep audit: Gate G7 continuous shared-phase RIS (2026-08-05)

### What was checked

- Shared-phase parameterization: one ULA steering cosine, analytic squared
  array-gain gradients verified against central finite differences.
- Three total budgets (20/28/40) and six seeds under the exact
  control/report budget identity.
- Comparison set: no RIS, random shared phase, per-target ideal phase,
  shared weak-aligned phase, shared surrogate-gradient phase, and shared
  system-optimized phase (101-point grid plus bounded refinement).

### Findings

- The surrogate gradient (worst squared array power) converges to
  `u ~ 0.0005`, which is a local surrogate optimum but a poor system
  operating point; it is kept as a negative result.
- System-level optimization finds `u ~ -0.99` for all budgets, matching the
  weak-target direction.  At B=20 it gives worst-target expected P_D 0.831,
  +8.2pp over no RIS and +31.9pp over random phase, with QoS feasibility
  50%.
- Per-target ideal phase remains 12.3pp better in worst-target P_D at B=20
  and reaches 100% QoS, so the single shared beam is physically limited.

### Residual limitations

- The shared phase is a 1-D ULA parameterization; a 2-D aperture or
  subarray-based multi-beam design is not yet implemented.
- System-level optimization uses a finite grid plus local refinement and is
  not a proven global certificate for the expected-P_D objective.

## Deep audit: Gate G8 exact quota-constrained selection (2026-08-05)

### What was checked

- Equal report costs are verified programmatically; the report budget maps to
  a cardinality quota.
- For each target, all `2^(num_uavs-1)` report subsets are evaluated with the
  exact expected-P_D reception law, and the best subset of each size is
  retained.
- All per-target quota compositions with total quota not exceeding the budget
  are searched; the score is QoS gap, then weighted mean, then worst-target
  expected P_D.
- Four seeds, budgets 20/28/40, with and without the G5-T RIS deployment.

### Findings

- Exact equals greedy in every audited cell: mean and worst gains are exactly
  0.0.  The forward expected-P_D greedy is therefore globally optimal for
  the audited equal-cost selection problem.
- The remaining gap to all-scheduled is 3.59pp worst (no-RIS B=20), 1.08pp
  (no-RIS B=28), 0.04pp (no-RIS B=40), and 3.11/0.43/0.17pp (RIS), showing
  that architecture, not selection, is the binding resource.

### Residual limitations

- The exact search is exponential in reports per target and is capped at 10
  non-owner reports; larger UAV counts require a knapsack or branch-and-bound
  variant.
- The guarantee assumes equal report costs; the general nonuniform-cost case
  is now covered by G8-K/G8-M/G8-S below.

## Deep audit: Gate G8-K/G8-M/G8-S exact selection (2026-08-06)

### What was checked

- G8-K exact lexicographic budget selection and G8-M exact max-min selection
  are audited on controlled 3-target/4-report models against an exhaustive
  global oracle (all subset combinations of all targets) and on the
  variable-rate demo system against forward greedy.
- Formal run: 20 seeds x 5 report budgets `{3,5,7,9,11}`, grid 64, for both
  selectors; G8-S uses the same 20 seeds at budgets `{5,9}` and a synthetic
  12-report set with 4096 subsets.

### Findings

- G8-K oracle match rate 100% (100/100); system never worse than greedy on
  the lexicographic score 100% (100/100).  Mean worst-target system gains
  +1.27pp at B=5 (p=0.015) and +2.57pp at B=7 (p=0.009); exact uses 7.0/7
  report bits versus greedy 6.6 on average at B=7.  At B=9 the worst-target
  gain is -1.00pp (p=0.895), recording the lexicographic trade-off.
- G8-K's lexicographic objective can lower worst-target P_D relative to
  greedy in controlled cells (B=11: 0.3745 versus 0.4081), motivating G8-M.
- G8-M oracle match rate 100% (100/100); system never worse than greedy 100%
  (100/100).  Controlled gains +5.37pp at B=5 (p<1e-6), +8.24pp at B=7
  (p<1e-6), +0.39pp at B=9 (p=0.083), +3.33pp at B=11 (p=3.9e-4); system
  gains +1.27pp (p=0.015), +2.57pp (p=0.009), +0.28pp (p=0.046), +1.09pp
  (p=0.014) at B=5/7/9/11, with 95% bootstrap CIs excluding zero at the
  significant cells.
- G8-S matches the exact selector with zero absolute error on all 20-seed
  cells and finds the minimum-cost subset of the 12-report model (cost 2)
  without enumerating all 4096 subsets.
- The G8-S report-count benchmark (R=8/12/16/20/24/28/32/40) returns exact
  minimum costs of 1-2 bits in 24-60 ms while the exhaustive baseline grows
  to about $1.1\times 10^{12}$ subsets.
- The target-count audit (Q=3/4/5, B=8/12/16, 3 seeds, grid 32) keeps the
  exhaustive-oracle match and never-worse-than-greedy rates at 100% across
  all 27 cells, with max-min wall time below 360 ms at Q=5.

### Residual limitations

- The per-target subset enumeration remains exponential; G8-S is an exact
  pruning certificate, not a polynomial-time guarantee.
- The formal selection audit uses budgets 3-11 report bits; the
  high-budget saturation regime is covered by the equal-cost G8 audit.
- At 20 seeds, G8-M system gains are significant at B=5/7/9/11 (p<0.05);
  G8-K is significant at B=5/7 but negative at B=9, so the lexicographic
  and max-min claims are reported separately.

## Deep audit: Gate G9 aperture-conserved subarray multi-beam (2026-08-05)

### What was checked

- The 256-element ULA is partitioned into three contiguous target-aligned
  subarrays; total elements and phase-bits per element are unchanged, so the
  control overhead identity is identical to G5.
- Discrete coordinate ascent over 32/16/8-element aperture transfers,
  maximizing mean worst-target expected P_D over six seeds.
- Baselines: no RIS, random shared phase, single shared weak-aligned phase,
  and per-target ideal phase.

### Findings

- Optimized allocations move most aperture toward the targets that bind the
  worst expected P_D: `(6,85,165)` at B=20, `(6,149,101)` at B=28, and
  `(6,173,77)` at B=40.
- At B=28, subarray multi-beam reaches 100% QoS feasibility with worst-target
  expected P_D 0.913, +5.2pp over shared weak-aligned and +13.7pp over
  no-RIS.
- The per-target ideal phase remains 6.7pp higher at B=28, so the remaining
  gap is the aperture-splitting/cross-beam trade-off, not selection.

### Residual limitations

- The search uses contiguous blocks and a 1-D ULA; interleaved subarrays and
  2-D apertures are not covered.
- The coordinate-ascent result is a local optimum over integer transfers,
  not a global certificate for the aperture allocation.

## Deep audit: Gate G10 per-subarray steering optimization (2026-08-05)

### What was checked

- G9 aperture allocations are loaded from the stored JSON, so the comparison
  uses identical allocations and seeds.
- Coordinate ascent perturbs each block steering cosine on a 7-point grid
  with step 0.1 for up to two rounds, maximizing mean worst-target expected
  P_D over six seeds.
- Total aperture, phase bits, and control overhead are unchanged.

### Findings

- Steering optimization improves worst-target expected P_D by 0.41pp (B=20),
  0.23pp (B=28), and 0.14pp (B=40) over G9, with positive bootstrap CIs.
- The useful change is rotating the small first block toward the strong
  target (`u ~ 0.96-0.99`); the weak block remains aligned to the weak
  target.
- QoS feasibility is unchanged (50% at B=20, 100% at B=28/40), and the
  per-target ideal gap remains 9.6/6.5/4.8pp, so aperture allocation is the
  primary physical degree of freedom.

### Residual limitations

- The steering search is a local grid coordinate ascent; no global
  optimality certificate is provided.
- The one-dimensional cosine parameterization limits the search to the ULA
  axis; a 2-D aperture would require two steering parameters per block.

## Deep audit: Gate G11 fixed-budget aperture scaling (2026-08-05)

### What was checked

- Total budgets 20/28/40, RIS elements 128/256/512/1024, phase bits 1/3,
  coherence frames 64/256, and three fixed subarray allocations.
- Every configuration uses `B_report = B_total - N * phase_bits /
  coherence_frames`; infeasible configurations are skipped.
- Four seeds per cell with the expected-P_D greedy selector.

### Findings

- Increasing aperture alone can close the B=20 QoS gap if the control
  overhead is amortized: `N=1024`, 3-bit, `C=256` reaches 100% QoS with 8
  report bits and worst P_D 0.982; `N=512` reaches 0.943.
- Equal allocation beats weak-biased allocation at large `N`, because the
  `N^2` coherent gain is large enough to support all targets simultaneously.
- At small aperture (`N=128`), no tested allocation reaches 100% QoS at
  B=20, so aperture, not allocation, is the binding constraint.

### Residual limitations

- The gate uses three fixed allocations rather than joint
  `(N, phase_bits, coherence, allocation)` optimization.
- Longer coherence frames are assumed cost-free beyond the amortization
  identity; channel time-variation within the coherence block is not modeled.

## Deep audit: Gate G12 model-driven architecture derivation (2026-08-05)

### What was checked

- The surrogate starts from the subarray approximation
  `G_q = a_q/N`, the RIS/direct power ratio `K_q a_q^2`, phase-quantization
  loss `sinc^2(1/2^b)`, and the quadratic deflection-in-SNR relation.
- The equal-allocation weak-target objective is
  `J(N) = beta (1 + kappa N^2)^2 (R - LN)`; its derivative yields the
  quadratic `5 kappa L N^2 - 4 kappa R N + L = 0`.
- The closed-form `N*` is rounded to a 64-element grid and validated by the
  exact expected-P_D system at `N*-64`, `N*`, `N*+64`.

### Findings

- B=20, b=1, C=64: `N* = 1016`, rounded to 1024, exact worst P_D 0.900 and
  100% QoS.
- B=20, b=3, C=128: `N* = 678`, rounded to 704, exact worst P_D 0.910 and
  100% QoS.
- B=20, b=3, C=256: `N* = 1363`, rounded to 1344, exact worst P_D 0.974 and
  100% QoS.
- The surrogate correctly ranks the bad configuration `b=3, C=64` as
  non-beneficial, matching the exact 0% QoS at N=256/320.

### Residual limitations

- The derivation uses equal allocation and ignores cross-block interference
  in the first-order condition.
- The quadratic-in-SNR deflection law is an asymptotic moment approximation;
  the exact validation is performed by the full expected-P_D model.

## Deep audit: Gate G13 max-min deflection water-filling (2026-08-05)

### What was checked

- Surrogate `D_q(a_q) = beta_q (1 + kappa_q a_q^2)^2`, with `beta_q` from
  averaged owner-only deflection and `kappa_q` from geometry and phase
  quantization.
- A first implementation equalized `dD_q/da_q`; its exact evaluation
  degraded worst P_D and was rejected as the wrong KKT for max-min.
- Correct implementation moves aperture from the current max-D target to the
  current min-D target with halving step sizes until the minimum stops
  improving.

### Findings

- At `N=1024,b=1,C=64,B=20`: water-filling `(382,388,254)` gives exact worst
  P_D 0.911 versus 0.900 for equal allocation.
- At `N=1344,b=3,C=256,B=20`: `(656,444,244)` gives worst P_D 0.992 versus
  0.974.
- At `N=2048,b=3,C=256,B=40`: `(1046,661,341)` gives worst P_D 0.999995
  versus 0.999599.

### Residual limitations

- The surrogate ignores cross-block interference and report-budget effects
  on allocation.
- The fixed point is a deterministic max-min iteration, not a proven global
  certificate over the exact system objective.

## Deep audit: Gate G14 exact-array-factor allocation (2026-08-05)

### What was checked

- The surrogate uses the exact squared array factor
  `G_q(a) = |(1/N) sum_b sum_{n in block b} exp(j(phi_n - ideal_nq))|^2`,
  including cross-block interference and phase quantization.
- Max-min water-filling runs on the exact surrogate; equal, separable, and
  exact allocations are each evaluated by the full expected-P_D system.

### Findings

- Exact surrogate minimum is highest for exact allocation in all five tested
  configurations.
- Exact system worst P_D is not consistently best: `N=1024,b=1,C=64` exact
  allocation is 0.8pp worse than separable; `N=1344,b=3,C=256` is unchanged
  within 0.004pp; `N=2048,b=3,C=256` is 0.00008pp better.
- The greedy expected-P_D objective and the surrogate are not perfectly
  aligned, so surrogate exactness alone does not certify system optimality.

### Residual limitations

- The gate uses four seeds and fixed G12-derived apertures.
- A system-level first-order condition that accounts for the greedy
  scheduling discontinuity remains open.

## Deep audit: Gate G15 greedy-aware system-level allocation (2026-08-05)

### What was checked

- The objective is the exact system `F(a) = mean_seed min_q E_PD(q, S_q(a))`,
  including the expected-P_D greedy schedule and reporting law.
- Coordinate ascent over single-block transfers uses 32/16/8-element steps
  and accepts only strict improvements of `F`.
- Equal, separable, exact-surrogate, and system-ascent allocations are all
  evaluated with the same seeds.

### Findings

- System ascent dominates all surrogate allocations in four of five tested
  configurations.
- Improvements: `N=1024,b=1,C=64` +1.31pp, `N=704,b=3,C=128` +1.63pp,
  `N=960,b=3,C=128,B=28` +0.48pp, `N=1344,b=3,C=256` +0.03pp.
- At `N=2048,b=3,C=256,B=40`, system ascent ends 0.0018pp below exact
  surrogate, showing the 8-element local step is not fine enough near
  saturation.

### Residual limitations

- The search is a deterministic coordinate ascent without a global
  optimality certificate.
- Step sizes stop at 8 elements; a finer or multi-block variant is future
  work.

## Deep audit: Gate G16 single-element local certificate (2026-08-05)

### What was checked

- Each G15 allocation is refined by 4/2/1-element coordinate ascent on the
  exact system objective.
- The final allocation is certified by `exact_single_move_gradients`, which
  evaluates all ordered one-element transfers and checks whether any has a
  positive objective gradient.

### Findings

- Refinement improves every configuration: 0.923826 to 0.924107
  (`N=1024`), 0.927291 to 0.927345 (`N=704`), 0.991872 to 0.991896
  (`N=1344`), 0.985459 to 0.985738 (`N=960,B=28`), and 0.999978 to 0.999986
  (`N=2048,B=40`).
- All five final allocations satisfy `local_optimal=true`; maximum gradients
  are `-1.9e-4`, `0.0`, `0.0`, `-5.7e-6`, and `-9.2e-8`.

### Residual limitations

- The certificate is only for single-element transfers of the exact
  objective; multi-block and joint placement-allocation moves are not
  covered.
- G16/G17 must use the same seed set as the upstream G15 allocation.  A
  1-seed smoke run on a 4-seed allocation converged to a 1-seed local
  optimum that is worse for the 4-seed objective, so cross-seed chain
  reuse is invalid.

## Deep audit: Gate G17 bounded multi-block certificate (2026-08-05)

### What was checked

- Starting from each G16 allocation, all zero-sum net-change vectors with
  positive mass at most 3 are evaluated exactly on the system objective.
- The search iterates to the best allocation in that bounded neighborhood
  until no move improves, then certifies the final point.

### Findings

- Four configurations are already multi-block local optima (0 rounds).
- `N=2048,B=40` improves from 0.999986 to 0.999988 in 7 rounds.
- All five final allocations satisfy `local_optimal=true` with 36 evaluated
  candidate moves each.

### Residual limitations

- The certificate is bounded to `T<=3` simultaneously moved elements.
- Joint RIS placement and allocation moves are not yet included.
- The same-seed requirement from G16 applies: upstream allocations must not
  be refined with a different seed count.

## Deep audit: Gate G18 joint placement-allocation (2026-08-05)

### What was checked

- Starting from G17 allocations at `(0,30,6)`, alternating coordinate ascent
  optimizes the exact system objective over allocation (T<=3) and position
  (2/1/0.5m steps).
- Final points are certified for allocation with `bounded_multi_move_certificate`
  and for position by all six 0.5m neighbors.

### Findings

- `N=1024,B=20`: `(-2,30,6)`, `(271,479,274)`, value 0.925224.
- `N=1344,B=20`: `(0.5,31,6)`, `(649,443,252)`, value 0.992907.
- `N=2048,B=40`: `(6.5,34,5)`, `(956,619,473)`, value 0.999997.
- All three have allocation and position certificates `local_optimal=true`.

### Residual limitations

- The joint result is local with 0.5m position granularity and T<=3
  allocation moves.
- Global placement certificates and larger T remain open.

## Deep audit: Gate G19 progressive decentralization (2026-08-05)

### What was checked

- Four decentralization stages: local round-robin scheduling, deflection
  fusion, owner-only, and local 1-bit hard decisions with counting fusion.
- Three G18 final configurations and four seeds.

### Findings

- B=40/N=2048: centralized 0.999997; local schedule 0.999984 (-0.0013pp),
  deflection 0.999971 (-0.0026pp), owner-only 0.999856 (-0.014pp), hard
  0.81197 (-18.8pp, QoS 50%).
- B=20/N=1024: report budget 4 makes 5-bit soft reports infeasible, so
  centralized equals owner-only; 1-bit hard decisions use 3 bits but cannot
  meet the global P_FA=0.05 with one vote per target, giving worst P_D 0.623
  and QoS 0%.

### Residual limitations

- Report bit length is fixed at 5 for soft reports; variable-length
  soft/hard hybrid reporting is not modeled.
- The decentralized stages still assume a common fusion point for soft and
  counting fusion; true peer-to-peer consensus is not implemented.

## Deep audit: Gate G20 amplified distributed hard detection (2026-08-05)

### What was checked

- Local 1-bit thresholds are optimized over a geometric P_FA grid including
  0.1; the exact counting P_FA/P_D is evaluated for each candidate.
- Per target, the optimized rule chooses the local P_FA and vote threshold
  that maximize P_D while satisfying global P_FA.

### Findings

- B=40/N=2048: hard default worst P_D 0.812 (QoS 0%); optimized hard 0.944
  (QoS 100%); centralized 0.999997.
- B=20 with one vote per target: no counting threshold satisfies global
  P_FA, so the distributed branch returns infeasible (P_D 0).
- Optimized local P_FA values are around 0.032-0.055, showing the fixed 0.1
  default was a poor design point.

### Residual limitations

- The optimization scans a one-dimensional local P_FA grid and uses fixed
  per-target vote schedules.
- Topology and peer-to-peer consensus are not modeled.

## Deep audit: Gate G21 network-level decentralization (2026-08-05)

### What was checked

- Stages: centralized soft, optimized hard with full/top-K links, and peer
  majority with all M UAV votes and no report links.
- Peer majority optimizes local P_FA and the global vote threshold under the
  P_FA constraint.

### Findings

- B=20/N=1024: centralized 0.925, peer majority 0.955.
- B=20/N=1344: centralized 0.993, peer majority 0.998.
- B=40/N=2048: centralized 0.9999967, peer majority 0.9999977.
- All peer-majority cells have 100% QoS and use 0 report bits.

### Residual limitations

- Peer majority assumes every UAV observes the target and votes without
  report-link erasure; partial observability and multi-hop consensus are not
  modeled.

## Deep audit: Gate G22 degraded multi-hop consensus (2026-08-05)

### What was checked

- Effective participation `obs * (1 - (1 - r)^hops)` is derived from
  observability and independent per-hop reliability.
- Local P_FA and majority threshold are optimized exactly under the global
  P_FA constraint for each degradation stage.

### Findings

- B=40/N=2048: peer clean 0.9999977; obs 0.75 -> 0.966; link 0.8 -> 0.977;
  three-hop 0.8 -> 0.9998; severe -> 0.877.
- B=20/N=1024: obs 0.75 -> QoS 0%; link 0.8 -> 0.864 (QoS 100%); three-hop
  -> 0.952.
- The consensus advantage is network-quality-dependent.

### Residual limitations

- Link failures are independent and observability is scalar; correlated
  failures and per-UAV heterogeneity are not modeled.

## Deep audit: Gate G23 correlated failure and heterogeneous observability (2026-08-05)

### What was checked

- Common network failure probability `p_c` enters the participation as
  `obs_i * (1 - p_c) * (1 - (1 - r)^hops)`.
- Per-UAV observability is derived from target distance with
  `clip(1 - 0.5 * normalized_distance, 0.2, 1.0)`.

### Findings

- B=40/N=2048: peer clean 0.9999977; common failure 0.2 -> 0.977; 0.4 ->
  0.909; heterogeneous obs -> 0.936; severe combined -> 0.858.
- B=20/N=1024: common failure 0.4 and heterogeneous obs both drop QoS to 0%.
- Correlated outages remove the distributed advantage faster than
  independent degradation.

### Residual limitations

- A single common failure probability models whole-network outage only;
  regional groups and time-varying topology are not included.

## Deep audit: Gate G24 scalability across target/UAV counts (2026-08-05)

### What was checked

- Q in {2,4,6}, M/Q in {1,2,3}, three seeds, per-target report budget
  `20*Q`, RIS control overhead 12 bits.
- Methods: no-RIS centralized, RIS per-target ideal phase centralized,
  fully distributed peer majority.

### Findings

- RIS ideal phase reaches QoS 100% in all tested cells except Q=6,M=6
  (0.762).
- Peer majority reaches QoS 100% for Q=2 at all ratios, Q=4 at all ratios,
  and Q=6 at M/Q=3 (0.919).
- No-RIS is not monotone in M: Q=2,M=4 gives 0.460 while Q=2,M=2 gives
  0.915, due to topology/reporting geometry.

### Residual limitations

- RIS uses per-target ideal phase (upper bound) rather than a real shared
  subarray design at scale.
- The minimum M/Q ratio for consensus parity is empirical, not derived.

## Deep audit: Gate G25 scaled white-box G18 (2026-08-05)

### What was checked

- Scaled architecture uses `waterfilling_allocation` (derived max-min
  surrogate) and exact 2/1/0.5m position coordinate ascent.
- Same Q/M matrix and resource ledger as G24; two seeds.

### Findings

- Scaled G18 keeps 100% QoS in all cells except Q=6,M=6, matching ideal-phase
  feasibility behavior.
- It always beats peer majority in the tested cells and stays close to the
  ideal-phase upper bound.
- Allocations are explicit and budget-dependent; positions move away from
  `(0,30,6)` for most configurations.

### Residual limitations

- Allocation certificate is not claimed for Q>3.
- Only two seeds were used; a formal seed count should match G24 before
  submission.

## Deep audit: Gate G26 mobility and time-varying blockage (2026-08-05)

### What was checked

- `build_models` accepts explicit transmitter/target positions, keeping the
  moment-matched and reporting chain unchanged.
- UAVs rotate, targets move on bounded circular paths, and weak-target
  blockage varies sinusoidally over 8 frames.
- Static allocation is designed at t=0; adaptive allocation recomputes the
  derived water-filling each frame.

### Findings

- no-RIS worst-over-time P_D 0.489, QoS 0%.
- RIS ideal worst-over-time P_D 0.935, QoS 100%.
- Static subarray worst 0.841, QoS 68.75%; adaptive subarray worst 0.847,
  QoS 81.25%.

### Residual limitations

- The original G26 trajectory is deterministic; G51 now adds AR(1)
  stochastic mobility and one-frame RIS latency.
- Trajectory optimization and prediction-aware RIS control remain open.

## Deep audit: Gate G27 multi-RIS deployment (2026-08-05)

### What was checked

- Total aperture fixed at 256, phase bits 3, coherence 64; report budget is
  `B_total - control_overhead`.
- One/two/three RIS configurations use per-target ideal phase and
  non-coherent power summation.

### Findings

- One RIS is best at every budget: 0.955/0.980/0.983.
- Two RIS: 0.868/0.923/0.926; three RIS: 0.863/0.927/0.939.
- Three RIS partially recovers at B=40 but remains below one RIS.

### Residual limitations

- RIS positions are fixed manually; joint aperture-split and placement
  optimization is open.
- Non-coherent summation is conservative; synchronized coherent multi-RIS
  combining would require phase alignment and is not claimed.

## Deep audit: Gate G28 multi-RIS split/placement optimization (2026-08-05)

### What was checked

- Total aperture fixed at 256; first RIS fixed at `(0,30,6)`; second RIS
  position and first-RIS element count optimized on the exact system
  objective with 16/8-element and 16/8/4/2/1-meter? position steps.
- Compared with one RIS and equal split.

### Findings

- One RIS: 0.981; equal split: 0.924.
- Optimized: `(8,248)` at `(4,42,2)`, value 0.986.
- The optimized two-RIS configuration slightly exceeds the single-RIS
  baseline, showing placement diversity can compensate for aperture split.

### Residual limitations

- Search is local with 8-element and 2m/1m position steps; no global
  certificate.
- Both RISs use per-target ideal phase; shared-phase multi-RIS is open.

## Deep audit: Gate G29 variable-rate reporting (2026-08-05)

### What was checked

- `build_models` accepts `quantizer_bits_per_uav`; report cost is
  `bits + 2`.
- Policies: fixed 5-bit, fixed 3-bit, adaptive per-target equal-budget
  high/low-rate profile, optimized 1-bit hard decisions.

### Findings

- B=20: soft5 0.953, soft3 0.951, adaptive 0.951, hard1 0.845.
- B=28: soft5 0.977, adaptive 0.974, hard1 0.936.
- B=40: adaptive 0.988, soft5 0.981, soft3 0.978, hard1 0.858.

### Residual limitations

- Adaptive profile is per-target equal budget and quality-ranked, not a
  global knapsack optimization.
- Hard/soft hybrid fusion within one target is not modeled.

## Deep audit: Gate G30 global rate-profile optimization (2026-08-05)

### What was checked

- Objective is exact `F(bits) = mean_seed min_q E_PD(q, S_q(bits))`.
- Coordinate ascent changes one UAV quantizer bit at a time; final point is
  certified over all single-rate changes.

### Findings

- B=28: fixed3 0.969, fixed5 0.981, adaptive 0.974, optimized 0.988.
- B=40: fixed3 0.978, fixed5 0.987, adaptive 0.987, optimized 0.991.
- Optimized profiles mix 1/2/3-bit rates and pass the local certificate.

### Residual limitations

- One rate per UAV across all targets; per-target rate profiles are not yet
  optimized.
- Hybrid soft/hard fusion inside a target is not modeled.

### Formal exact-objective re-certification (G30-E, 2026-08-06)

- G30-E re-checks the G30 profile with the exact max-min selector G8-M on
  the same 2-seed/grid-256 audit as G30, at B=28 and B=40.
- B=28: the G30 profile remains a single-rate exact local optimum at
  0.9879; `greedy_certificate_false_under_exact=false`.
- B=40: the greedy certificate is false under the exact objective
  (`true`); exact coordinate ascent improves worst P_D from 0.9911 to
  0.9916 and certifies the result as a single-rate exact local optimum.
- The paper therefore reports the exact-objective certificate, and the
  B=40 correction is a positive audit outcome: the greedy certificate was
  not silently extended to the exact objective.

## Deep audit: Gate G31 exact soft/hard hybrid fusion (2026-08-05)

### What was checked

- Soft score uses P_D-optimal Gaussian weights; hard reports contribute exact
  post-BSC log-likelihood terms.
- Hard-decision patterns are enumerated and the soft threshold is found by
  binary search to meet global P_FA.

### Findings

- B=28: soft5 0.977, hybrid 0.977, hard1 0.843.
- B=40: soft5 0.981, hybrid 0.969, hard1 0.736.
- Hybrid fusion is exact and P_FA-constrained but not automatically better
  than soft-only with the fixed schedule.

### Residual limitations

- Hybrid schedule is fixed heuristically; joint schedule/rate optimization is
  open.
- Hard LLRs use post-soft moments as an approximation of pre-quantization
  local decisions.

## Deep audit: Gate G32 interference sensitivity (2026-08-05)

### What was checked

- `build_models` accepts per-UAV `interference_to_noise`; effective SINR is
  `SNR / (1 + INR)`.
- INR sweep 0/3/10/20 dB with no-RIS, RIS ideal, and peer majority.

### Findings

- INR=0 dB: RIS ideal 0.861 QoS 100%; no-RIS 0.525, peer 0.739.
- INR=3 dB: RIS ideal 0.691, all QoS 0%.
- INR=10/20 dB: worst P_D below 0.2/0.06 for all architectures.

### Residual limitations

- INR is uniform across UAVs; spatial patterns and suppression are not
  modeled.

## Deep audit: Gate G33 spatial interference and RIS placement (2026-08-05)

### What was checked

- INR profile `inr_ref * (d_ref/d_i)^2` from a fixed source at `(60,-20,0)`.
- No-RIS, fixed RIS, and RIS position optimized by exact-system coordinate
  ascent.

### Findings

- Mean INR 0.0024/0.0244/0.2444: no-RIS worst P_D 0.828/0.824/0.768, QoS 0%.
- Fixed RIS 0.988/0.987/0.969, QoS 100%; optimized position
  0.991/0.990/0.973.

### Residual limitations

- Single interference source; no RIS null-steering is implemented.

## Deep audit: Gate G34 multiple interference sources (2026-08-05)

### What was checked

- INR is the sum of three free-space source terms.
- No-RIS, fixed RIS, and RIS position optimized by exact-system coordinate
  ascent.

### Findings

- Mean INR 0.087, max INR 0.131.
- No-RIS 0.810 QoS 0%; fixed RIS 0.983 QoS 100%; optimized 0.987 QoS 100%.

### Residual limitations

- Direct interference only; RIS null-steering toward sources is not modeled.

## Deep audit: Gate G35 1-D ULA versus 2-D UPA (2026-08-05)

### What was checked

- `RisConfig.aperture_shape` and `ris_upd.upd_physics_gain_matrix` are
  implemented and unit-tested.
- Same 256 elements, budgets 20/28/40, clean and spatial interference.

### Findings

- Clean B=40: ULA 0.981009, UPA 0.981003.
- Interference B=40: ULA 0.986915, UPA 0.986912.
- UPA does not add P_D in the current target geometry.

### Residual limitations

- The model has no elevation separation in the target set; null-steering is
  not implemented.

## Deep audit: Gate G36 UPA null-steering (2026-08-05)

### What was checked

- Phases optimized on scalarized array power with analytic gradient,
  L-BFGS-B, 256 elements.
- Reflected interference INR evaluated with aligned and designed phases.

### Findings

- Reflected INR 0.0267 -> 0.0106; target array gain 1.000 -> 0.984.
- B=28: aligned 0.97715, null 0.97842; B=40: aligned 0.98112, null 0.98216.
- Null-steering improves P_D while keeping QoS 100%.

### Residual limitations

- Continuous phases are optimized then quantized at evaluation; quantized
  nulling is not directly optimized.
- One phase vector per target; shared multi-target nulling is open.

## Deep audit: Gate G37 directly quantized null-steering (2026-08-05)

### What was checked

- `quantized_null_steering_phases` runs coordinate ascent over the `2^b`
  discrete phase levels.
- Compared with aligned quantized and continuous-then-quantized phases.

### Findings

- Reflected INR: aligned 0.02670, continuous-quantized 0.01056, quantized
  optimized 0.01052.
- B=40 P_D: aligned 0.981119, continuous-quantized 0.982165, quantized
  optimized 0.982166.

### Residual limitations

- Coordinate ascent is a local search over discrete phases; global phase
  certificate is open.
- Still one phase vector per target.

## Deep audit: Gate G38 joint quantized nulling and placement (2026-08-05)

### What was checked

- For each candidate position, quantized null-steering phases are redesigned
  for all targets.
- Position coordinate ascent maximizes exact system P_D.

### Findings

- No RIS 0.810; fixed nulling 0.98217; joint optimized 0.98481 at
  `(-2.5,32.5,1.0)`.
- Reflected INR increases from 0.0105 to 0.0296, showing the placement moves
  to improve target SNR despite higher reflected interference.

### Residual limitations

- Local search only; no joint certificate for phase+position.
- One phase vector per target; shared multi-target nulling is open.

## Deep audit: Gate G39 distributed relaxation (2026-08-05)

### What was checked

- Budgets 20/24/28 and QoS targets 0.70/0.75/0.80.
- Centralized soft, peer clean, peer multi-hop (0.8 per hop, 3 hops), and
  optimized hard decisions.

### Findings

- All four methods become QoS-feasible in every tested cell.
- Peer multi-hop worst P_D is 0.953; optimized hard 0.842-0.864.
- Lowering the QoS threshold is sufficient to make distributed viable, but
  actual P_D remains high because the channel/RIS still dominates.

### Residual limitations

- Very low budget/SNR regimes are not tested.
- Distributed features are still simplified consensus models.

## Deep audit: Gate G40 low-budget/low-SNR distributed (2026-08-05)

### What was checked

- N=128, mean INR 0.122, budgets 12/16/20, QoS target 0.70.
- Centralized soft, peer clean, peer multi-hop, optimized hard.

### Findings

- B=12: centralized 0.786; peer clean 0.858; peer multi-hop 0.855; hard
  0.765.
- B=16/20: centralized 0.891; peer stays 0.858/0.855; hard 0.758/0.739.
- Peer consensus outperforms centralized at the lowest budget because it
  needs zero report bits.

### Residual limitations

- Single interference source; very low SNR sweeps remain open.

## Deep audit: Gate G41 consensus parity boundary (2026-08-05)

### What was checked

- Gaussian approximation `M_min = p1(1-p1)(z_alpha+z_beta)^2/(p1-p0)^2`.
- Exact system sweep over M in {3,6,8,12,16} and B in {8,12,16,20}.

### Findings

- Theoretical M_min 13.7-16.8.
- Empirical consensus wins at B=8 for M>=6, B=12 for M>=12; centralized wins
  at B>=16.
- The formula is a useful lower-order predictor but the exact threshold
  optimization shifts the empirical boundary to smaller M.

### Residual limitations

- Formula uses fixed local P_FA 0.1; optimized local thresholds are not in
  the closed form.

## Deep audit: Gate G42 optimized-local-threshold boundary (2026-08-05)

### What was checked

- `M_min` is minimized over a geometric local P_FA grid.
- Same exact M/B sweep as G41.

### Findings

- Optimized M_min is 9-13% lower than fixed-P_FA M_min.
- Exact wins remain at B=8 for M>=6 and B=12 for M>=12, closer to the
  optimized formula.

### Residual limitations

- Formula still uses Gaussian approximation and averaged probabilities.
- Exact Poisson-binomial parity equations are not solved in closed form.

## Deep audit: Gate G43 exact Poisson-binomial boundary (2026-08-05)

### What was checked

- For each M, exact `_count_distribution` tails are used to test whether a
  threshold satisfies global P_FA and QoS.
- Local P_FA is optimized over the same grid.

### Findings

- Exact feasibility starts at M=6; M=3 is infeasible.
- Empirical wins also start at M=6 (B=8), closing the gap left by the
  Gaussian approximation.

### Residual limitations

- Feasibility is checked only at the discrete M grid; binary-search refinement
  is open.

## Deep audit: Gate G44 fundamental information budget (2026-08-05)

### What was checked

- Normalized information `rho = J/D_full` for soft, hard, and peer methods.
- Exact P_D computed with the same greedy/consensus chain.

### Findings

- Soft: 0.774 at rho 0.507, 0.823 at 0.656, 0.902 at 0.731, 0.923 at 0.895,
  0.933 at 0.946.
- Hard rho is 0.00-0.09 after budget enforcement; at B=8 no hard report
  fits the 2-bit budget, so owner-only optimized hard P_D is 0.763.
- Peer rho 0.09 with P_D 0.86 shows optimized consensus extracts more P_D
  per KL unit than raw fixed-threshold hard fusion.

### Residual limitations

- The information measure is deflection/KL-based, not full mutual
  information.
- A closed-form `P_D(rho, B, M, N)` law is not yet derived.

## Deep audit: Gate G45 closed-form resource law (2026-08-05)

### What was checked

- Law: `D_pred = d0(1+n)mean(gain^2)`, `P_D=Phi((sqrt(D)-z)/sqrt(c))`.
- c calibrated from owner-only exact P_D.

### Findings

- N=64: error 12-30pp; N=128/256: prediction saturates to 1 while exact is
  0.79-0.98.
- The law fails because it ignores quantization loss, correlation, and
  non-proportional H1 variance.

### Residual limitations

- The negative result is the contribution: it justifies the exact
  moment-propagation model used throughout the paper.

## Deep audit: Gate G46 exact information budget (2026-08-06)

### What was checked

- Effective deflection inversion
  `D_eff=(Phi^{-1}(P_D)+z_FA)^2` and normalization by `D_full`.
- Hard-decision report budget consistency at every total budget, including
  the previously unguarded B=8 case where `report_budget=2` cannot pay for
  one 1-bit hard report per target.
- Soft report bits, hard report bits, and peer zero-report bits stored in
  the result JSON.

### Findings

- Soft raw `rho` is 2.38-2.78x larger than `rho_exact`; raw deflection
  overstates the usable information because quantization and correlation
  are not included.
- Soft P_D is monotone in `rho_exact` (0.774 at 0.205 -> 0.933 at 0.351).
- Peer consensus has `rho_exact=0.284` at P_D 0.881; it remains competitive
  only in the scarce-report regime, where it pays zero report bits.
- Optimized hard fusion has `rho_exact<=0.199` after budget correction.

### Residual limitations

- The inversion assumes the calibrated Gaussian detection relation and
  `Sigma1 = c Sigma0` with `c=1`; it is not a universal closed-form law.
- The raw-to-exact inflation factor is audited at one operating profile
  (`N=128`, interference, four seeds) and should not be extrapolated.

## Deep audit: Gate G47 architecture switch (2026-08-06)

### What was checked

- Exact per-seed worst P_D for centralized soft fusion and peer majority.
- Exact mode selection: choose peer only when its worst P_D is strictly
  larger; otherwise keep centralized soft.
- A fixed `report_budget < 10` threshold as the practical non-oracle policy.
- Report-bit accounting: selected soft branch spends at most the report
  budget, peer branch spends zero report bits.

### Findings

- At B=8/12, peer is selected in 100% of seeds; worst P_D rises from
  0.774/0.824 to 0.881 (+10.68/+5.68pp) and the 0.85 QoS target becomes
  feasible.
- At B>=16, exact selection returns to centralized soft; no loss is
  observed at the tested budgets.
- The fixed threshold reproduces the exact mode choice in this audit and is
  stored as a design parameter, not claimed as a universal crossover.

### Residual limitations

- The audit uses four seeds and one operating profile; the threshold should
  be re-estimated for other geometries, RIS apertures, and target counts.
- The switch is between two complete detection architectures; joint
  optimization with per-target report rates and local thresholds is open.

## Deep audit: Gate G48 target-wise architecture switch (2026-08-06)

### What was checked

- Per-target soft P_D from the global centralized greedy schedule and
  per-target peer majority P_D.
- Per-target mode selection `max(soft_q, peer_q)` and its worst-target
  aggregate.
- The order inequality
  `min_q max(a_q,b_q) >= max(min_q a_q, min_q b_q)` as the certificate.
- Report-bit accounting: selected soft targets use the already-computed
  centralized schedule; peer targets spend zero report bits.

### Findings

- B=12: target-wise worst P_D 0.8858 vs global-switch 0.8810 (+0.49pp);
  B=16/20: 0.9174 vs 0.9019 (+1.55pp); B=28/40 no additional gain because
  the worst target already prefers soft.
- Peer target-selection rate is 92% at B=8, 83% at B=12, 50% at B=16, and
  drops to 33%/25% at B=28/40, confirming a gradual rather than binary
  transition.
- B=8 target-wise worst P_D 0.881 keeps the 0.85 QoS target feasible.

### Residual limitations

- G49 covers greedy reallocation of freed peer-target soft bits; joint
  schedule/mode optimality remains open.
- Per-target mode decisions use the exact soft schedule from the full
  centralized greedy run; an online implementation would need the same
  model information.

## Deep audit: Gate G49 soft-report reallocation (2026-08-06)

### What was checked

- Additive reallocation from peer-selected targets to centralized targets
  using exact expected-P_D marginal gains per report bit.
- Budget feasibility: `reallocation_used_bits <= report_budget` at every
  total budget.
- Monotonicity: centralized schedules are only extended, never shrunk, so
  per-target soft P_D and the target-wise worst P_D are nondecreasing.

### Findings

- B=16/20: worst P_D rises from 0.9174 (G48) to 0.9250 (+0.75pp); B=28/40
  rise to 0.9384/0.9412 (+1.55/+0.85pp over G48).
- B=8: no 5-bit soft report fits the 2-bit report budget, so reallocation
  has no gain; peer consensus still keeps 0.85 QoS feasible.
- Reallocation uses the freed bits: average used bits rise from 1.25 to 2.5
  at B=12, 6.25 to 10 at B=16, 10 to 20 at B=28, and 20 to 30 at B=40.
- B=12: reallocation uses only the reports that fit and leaves worst P_D at
  0.8858, unchanged from G48.

### Residual limitations

- The greedy is a monotone ascent, not a joint mode/schedule optimum.
- Reallocation does not allow a peer target to return to soft after new
  reports are added to other targets; G50 covers limiting-target two-sided
  updates, and joint optimality remains open.

## Deep audit: Gate G50 two-sided mode ascent (2026-08-06)

### What was checked

- Additive reallocation to centralized targets followed by limiting-target
  peer-to-soft upgrades.
- Peer-to-soft acceptance rule: switch only when the target currently
  attains the worst P_D and the upgraded soft P_D strictly raises the worst
  value.
- Budget feasibility after every accepted switch and report-bit usage.

### Findings

- B=12: worst P_D rises from 0.8858 (G48) to 0.8898 (+0.39pp); the accepted
  switch uses 3.75 report bits on average and only happens when it is the
  limiting target with no tied worst target remaining.
- B=16/20/28/40: the ascent matches G49 (0.9250/0.9250/0.9384/0.9412), and
  no non-improving peer-to-soft switch is accepted.
- B=8: no report budget for a soft upgrade; peer consensus keeps 0.85 QoS
  feasible.

### Residual limitations

- The ascent certifies monotone improvement, not global optimality.
- The limiting-target rule is greedy; a global certificate over report
  additions and mode sequences remains open.

## Deep audit: Gate G51 stochastic mobility (2026-08-06)

### What was checked

- AR(1)-correlated random UAV/target positions and random time-varying
  blockage over eight frames and four seeds.
- RIS designs: frozen from frame 0, designed from the previous frame
  (latency-1), or ideal per frame.
- Target-wise switching and mode ascent evaluated on every frame with the
  same report budget; worst-over-time and QoS-over-time aggregated per seed.

### Findings

- No-RIS soft/mode ascent worst-over-time P_D is 0.524 with 0% QoS; mode
  ascent cannot compensate for the absent RIS.
- Static RIS mode ascent reaches 0.705/0.822 worst/mean and 40.625% QoS;
  latency-1 RIS reaches 0.722/0.830 and 43.75% QoS, a +1.64pp worst gain.
- Ideal target-wise reaches 0.847/0.884 and 81.25% QoS; ideal mode ascent
  reaches 0.852/0.890 and 90.625% QoS (+0.49pp worst, +9.38pp QoS).

### Residual limitations

- The frame model is a declared AR(1) process, not a continuous-time SDR
  trajectory or a stochastic optimal-control problem.
- RIS latency is modeled as one-frame phase staleness; scheduling latency
  and control-bit timing are not yet included.

## Deep audit: Gate G52 MMSE prediction-aware RIS (2026-08-06)

### What was checked

- AR(1) conditional-mean predictor
  `hat p_t = n_t + rho (p_{t-1} - n_{t-1})` used to design per-target RIS
  phase one frame ahead.
- Comparison against no-RIS, static RIS, latency-1 RIS, ideal target-wise,
  and ideal mode ascent on the same four seeds and eight frames.

### Findings

- MMSE prediction improves latency-1 worst-over-time P_D from 0.7217 to
  0.7283 (+0.65pp) and QoS from 43.75% to 46.875%.
- The ideal per-frame phase remains the upper bound at 0.852 worst P_D and
  90.625% QoS; MMSE closes a meaningful but not complete part of the gap.

### Residual limitations

- The predictor uses the true AR(1) correlation; model mismatch and
  estimated correlation are not audited.
- G53 covers multi-step prediction; prediction-error covariance shaping and
  control-aware phase design remain open.

## Deep audit: Gate G53 multi-step MMSE prediction (2026-08-06)

### What was checked

- h-step AR(1) predictor
  `hat p_{t|t-h} = n_t + rho^h (p_{t-h} - n_{t-h})` for h=1/2/3.
- Theoretical prediction-error covariance scale `1 - rho^{2h}` stored next
  to the numerically evaluated stale and MMSE worst-over-time P_D.
- Same four seeds, eight frames, N=128, total B=40.

### Findings

- h=1: stale 0.7217 vs MMSE 0.7283 (+0.65pp).
- h=2: stale 0.6726 vs MMSE 0.7051 (+3.24pp).
- h=3: stale 0.6676 vs MMSE 0.7200 (+5.24pp).
- The MMSE gain grows with horizon as the error covariance scale rises from
  0.36 to 0.59 to 0.74.
- Exact per-frame selection over all stale/MMSE h=1/2/3 reaches 0.7369
  worst P_D and 53.125% QoS, versus 0.7283 and 46.875% for the best fixed
  MMSE.
- Hysteresis with delta=0.02 keeps 0.7358 worst P_D (loss 0.00104), QoS
  53.125%, and cuts mean switches from 4.50 to 2.25 per seed.
- Cost-aware selection under a 6-bit budget picks delta=0.00 for 1-bit
  switching (0.7369 worst), delta=0.03 for 3-bit switching (0.7250), and
  delta=0.05 for 6-bit switching (0.7217), with 4.50/1.50/0.75 switches.

### Residual limitations

- The correlation coefficient is assumed known; estimation error is not
  audited.
- Horizon and report/control budget are not optimized jointly.

## Deep audit: Gate G54 covariance-aware phase (2026-08-06)

### What was checked

- Covariance-aware phase maximizes the expected squared array gain under the
  h=3 AR(1) direction-error distribution, starting from the MMSE phase.
- The optimized phase is evaluated through 3-bit quantization and the exact
  expected-P_D/mode-ascent chain on the same four seeds and eight frames.

### Findings

- The robust phase is monotone in the expected-gain surrogate.
- Exact worst-over-time P_D drops from 0.7200 (MMSE) to 0.6557
  (-6.43pp); QoS drops from 43.75% to 37.5%.
- The expected-gain surrogate does not transfer to the quantized system, so
  the design is rejected and MMSE phase remains the practical choice.

### Residual limitations

- The negative result is for expected-squared-gain optimization; a
  quantization-aware robust design with the exact system P_D remains open.
- The beam broadening may become useful for very small RIS apertures, which
  are not covered here.
