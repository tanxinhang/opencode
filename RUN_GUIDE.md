# Run Guide

Target machine: Ryzen 7 7800X3D + RTX 5070.  The current experiments are
CPU-bound (NumPy/SciPy); the GPU is not used.  The 7800X3D is enough for the
formal runs, but the scripts are single-process, so run independent gates in
separate terminals to use multiple cores.

## Setup

```powershell
git clone https://gitee.com/xinhangtan/yc_codex.git
cd yc_codex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Smoke checks (fast, run first)

```powershell
python scripts/run_evidence_calibration_gate.py --trials 200 --amplitude 1.0 --gain-mode relative_deficit_reduction
python scripts/run_report_channel_calibration_gate.py --trials 20000
python scripts/run_g1c_conditional_ranking_gate.py
python scripts/run_g1d_greedy_vs_oracle_gate.py
python scripts/run_g2_system_sweep.py --seeds 5
python scripts/run_g2_algorithm_negative_gates.py --seeds 5
python scripts/run_pd_optimal_fusion_gate.py --seeds 10 --grid 1024 --greedy-instances 10
python scripts/run_expected_pd_greedy_gate.py --seeds 5 --grid 256 --audit-instances 4
python scripts/run_ris_isac_gate.py --seeds 5 --budgets 20 30 --grid 256
python scripts/run_ris_phase_resolution_gate.py --seeds 5 --budgets 20 30 --grid 256
python scripts/run_ris_physics_gate.py --seeds 5 --grid 256
python scripts/run_ris_joint_budget_gate.py --seeds 5 --grid 256
python scripts/run_ris_placement_gate.py --seeds 5 --grid 256
python scripts/run_ris_multigrid_gate.py --seeds 5 --grid 256
python scripts/run_deployment_theory_gate.py --seeds 5 --grid 256
python scripts/run_lipschitz_adaptive_deployment_gate.py --seeds 3 --grid 128
python scripts/run_epsilon_closed_deployment_gate.py --seeds 1 --grid 128
python scripts/run_g5_bootstrap_ci_gate.py
python scripts/run_g5_deployment_ci_gate.py
python scripts/run_global_resource_fairness_gate.py
python scripts/run_ris_sensitivity_gate.py --seeds 6 --grid 256
python scripts/run_sota_baseline_gate.py --seeds 6 --grid 256
python scripts/run_budget_saturation_gate.py --seeds 6 --grid 256
python scripts/run_ris_shared_phase_gate.py --seeds 6 --grid 256
python scripts/run_exact_quota_gate.py --seeds 4 --grid 256
python scripts/run_exact_budget_gate.py --seeds 1 --budgets 5 9 --grid 64
python scripts/run_exact_maxmin_gate.py --seeds 1 --budgets 5 7 --grid 64
python scripts/run_scaled_maxmin_gate.py --seeds 1 --budgets 5 --grid 64
python scripts/run_ris_subarray_gate.py --seeds 6 --grid 256
python scripts/run_ris_subarray_steering_gate.py --seeds 6 --grid 256
python scripts/run_ris_aperture_scaling_gate.py --seeds 4 --grid 256
python scripts/run_derived_architecture_gate.py --seeds 4 --grid 256
python scripts/run_waterfilling_architecture_gate.py --seeds 4 --grid 256
python scripts/run_exact_allocation_gate.py --seeds 4 --grid 256
python scripts/run_system_allocation_gate.py --seeds 4 --grid 256
python scripts/run_single_move_certificate_gate.py --seeds 4 --grid 256
python scripts/run_multi_move_certificate_gate.py --seeds 4 --grid 256
python scripts/run_joint_placement_allocation_gate.py --seeds 4 --grid 256
python scripts/run_progressive_decentralization_gate.py --seeds 4 --grid 256
python scripts/run_amplified_distributed_gate.py --seeds 4 --grid 256
python scripts/run_network_decentralization_gate.py --seeds 4 --grid 256
python scripts/run_degraded_consensus_gate.py --seeds 4 --grid 256
python scripts/run_correlated_consensus_gate.py --seeds 4 --grid 256
python scripts/run_scalability_comparison_gate.py --seeds 3 --grid 256
python scripts/run_scaled_g18_scalability_gate.py --seeds 2 --grid 256
python scripts/run_mobility_blockage_gate.py --seeds 2 --frames 8 --grid 256
python scripts/run_multi_ris_gate.py --seeds 3 --grid 256
python scripts/run_multi_ris_split_optimization_gate.py --seeds 2 --grid 256
python scripts/run_variable_rate_report_gate.py --seeds 2 --grid 256
python scripts/run_global_rate_optimization_gate.py --seeds 2 --grid 256
python scripts/run_exact_rate_certificate_gate.py --seeds 1 --budgets 28 --grid 32
python scripts/run_hybrid_fusion_gate.py --seeds 2 --grid 256
python scripts/run_interference_sensitivity_gate.py --seeds 2 --grid 256
python scripts/run_spatial_interference_placement_gate.py --seeds 2 --grid 256
python scripts/run_multi_interference_placement_gate.py --seeds 2 --grid 256
python scripts/run_upd_vs_ula_gate.py --seeds 2 --grid 256
python scripts/run_null_steering_gate.py --seeds 2 --grid 256
python scripts/run_quantized_null_steering_gate.py --seeds 2 --grid 256
python scripts/run_joint_null_placement_gate.py --seeds 2 --grid 256
python scripts/run_distributed_relaxation_gate.py --seeds 2 --grid 256
python scripts/run_low_budget_snr_distributed_gate.py --seeds 2 --grid 256
python scripts/run_consensus_parity_boundary_gate.py --seeds 2 --grid 256
python scripts/run_optimized_parity_boundary_gate.py --seeds 2 --grid 256
python scripts/run_exact_parity_boundary_gate.py --seeds 2 --grid 256
python scripts/run_exact_min_majority_gate.py --uav-counts 3 6 8
python scripts/run_fundamental_information_gate.py --seeds 2 --grid 256
python scripts/run_resource_information_law_gate.py --seeds 2 --grid 256
python scripts/run_exact_information_budget_gate.py --seeds 2 --grid 256
python scripts/run_architecture_switch_gate.py --seeds 2 --grid 256
python scripts/run_target_wise_architecture_switch_gate.py --seeds 2 --grid 256
python scripts/run_soft_reallocation_gate.py --seeds 2 --grid 256
python scripts/run_mode_ascent_gate.py --seeds 2 --grid 256
python scripts/run_stochastic_mobility_gate.py --seeds 2 --frames 4 --grid 256
python scripts/run_prediction_aware_ris_gate.py --seeds 2 --frames 4 --grid 256
python scripts/run_multi_step_prediction_gate.py --seeds 2 --frames 4 --grid 256
python scripts/run_covariance_aware_ris_gate.py --seeds 2 --frames 4 --grid 256 --horizon 3
python scripts/run_robustness_stress_suite.py --seeds 1 --grid 32 --budgets 20
python scripts/run_robust_stress_allocation.py --seeds 1 --budgets 16 20 24
python scripts/run_bsc_degradation_roc_gate.py --bits 1 2 3 --deltas 1.0 1.5 2.0 --lo 0.0 0.1 0.2 --hi 0.3 0.4 0.45 --pfa-grid 0.01 0.05 0.1 0.2
python scripts/run_erasure_dominance_gate.py --samples 50000 --grid 128
python scripts/run_sensing_mobility_envelope_gate.py --samples 5000 --max-displacement 8.0
python scripts/benchmark_robustness_performance.py
python scripts/benchmark_robustness_performance.py --formal
python scripts/benchmark_exact_joint_scaling.py --targets 2 4 8 16
python scripts/run_joint_power_bit_gate.py --budgets 8 12 16 --grid 32
python scripts/benchmark_joint_power_bit_scaling.py --reports 2 4 6 --grid 32
python scripts/run_communication_aware_gate.py --seeds 10 --budgets 2 4 6 8
python scripts/run_communication_ambiguity_gate.py --seeds 5
python scripts/run_robust_joint_power_bit_gate.py --budgets 8 12 16 --grid 32
python scripts/run_robust_communication_aware_gate.py --seeds 10 --budgets 4 6 8
python scripts/run_robust_cas_divergence_gate.py --seeds 10 --budget 6
python scripts/run_joint_power_bit_split_gate.py --budgets 8 12 16 --grid 32
python scripts/run_mappo_greedy_scaling.py --targets 2 4 6 8 --train-seeds 20 --test-seeds 20 --episodes 800 --budget-multiplier 8
python scripts/build_paper_tables.py
```

If the runtime is `E:\anaconda\conda\python.exe`, use
`scripts/run_experiment_matrix.ps1` or the commands in `RUN_MATRIX.md`; the
matrix covers MAPPO/Greedy/Exact comparisons, weakened Exact Joint, robust
allocation baselines, joint power-bit allocation, and performance
benchmarks.

The robustness stress suite is the incremental entry point for the
progressive resistance gates: stage 1 covers spatial INR via free-space path
loss, BSC flip probability, link success scaling, and bounded target
mobility.  Add each future axis with a focused test in
`tests/test_robustness_stress.py` before enabling it in
`uav_otfs_isac/robustness_stress.py`.

The robust allocation script applies the same physical stress as a finite
scenario set and solves the exact worst-case chance-constrained allocation
from `uav_otfs_isac/robust_portfolio.py`; see Theorem 4.58 in
`FORMAL_PROOFS.md`.  It also reports worst-case sensing Top-K,
communication Top-K, independent post-report Top-K, deterministic random
Top-K, no cooperation, and worst-case marginal greedy baselines from
`uav_otfs_isac/robust_baselines.py` on the same worst-excess scale, together
with per-target worst violation probabilities and their mean in `[0,1]`.

For independent per-target degradation ambiguity, use
`optimize_independent_robust_chance_constrained_portfolio`; Theorem 4.62
shows the worst-case total separates as the sum of per-target worst cases.

For a geometry-derived reporting channel, use
`build_physical_link_models`; Lemma 4.64 records the BPSK/outage formulas.

Run `scripts/benchmark_robustness_performance.py` for smoke timing and
`--formal` for the submitted-scale stress/allocation sweeps.  The current
machine baseline is about 0.6 s smoke, 16 s for the stress sweep, and 11 s
for the robust-allocation sweep; see Theorem 4.63 for the complexity
statement.

Exact Joint scaling is measured with
`scripts/benchmark_exact_joint_scaling.py`; Lemma 4.65 explains the
`O(Q log O log V)` threshold-feasibility improvement.

Joint sensing-communication allocation is checked with
`scripts/run_joint_power_bit_gate.py`; Lemma 4.66 proves the joint option set
contains both sensing-only and communication-only baselines.

The BSC degradation ROC gate checks Theorem 4.59: cleaner BSC channels
dominate degraded ones on the exact quantized likelihood-ratio ROC, so the
endpoints of a flip-probability interval are the natural communication
scenario extremes.

The erasure dominance gate checks Theorem 4.60: lower link success
probabilities are a subset of the clean received set in a monotone coupling,
so expected P_D cannot increase under higher erasure at set-monotone
operating points.

The sensing mobility envelope gate checks Theorem 4.61: with displacement
`R = v_max T_frame`, every UAV-target range changes by at most `R`, and the
free-space power change is bounded by `(d_min/(d_min-R))^2`.  Corollary
4.61A adds the same envelope for the range-derived dB-SNR law in
`build_models`, so the gate covers the actual stress-suite model rather than
only the standalone free-space law.

## Formal runs

G1-A formal evidence calibration (10 000 trials per hypothesis, train/test
geometry separated):

```powershell
python scripts/run_evidence_calibration_gate.py `
  --trials 10000 `
  --amplitude 1.0 `
  --gain-mode relative_deficit_reduction `
  --output results/evidence_calibration_10k.json

python scripts/run_evidence_calibration_gate.py `
  --trials 10000 `
  --amplitude 1.0 `
  --gain-mode relative_deficit_reduction `
  --predicted-mode pd_gain `
  --output results/evidence_calibration_10k_pd_gain.json

python scripts/run_g1a_grouped_consistency_gate.py --trials-per-group 120
```

G1-B report-channel closure (50 000 Monte Carlo trials):

```powershell
python scripts/run_report_channel_calibration_gate.py `
  --trials 50000 `
  --output results/report_channel_calibration.json
```

G2 system sweep and algorithm negative audit (20 seeds):

```powershell
python scripts/run_g2_system_sweep.py --seeds 20
python scripts/run_g2_algorithm_negative_gates.py --seeds 20
python scripts/run_g2_correlation_sweep_gate.py --seeds 20 --oracle-seeds 10
python scripts/run_g2_nonsaturated_stress_gate.py --seeds 20
python scripts/run_resource_fairness_gate.py
python scripts/run_g2_scaling_gate.py --seeds 10
python scripts/run_g2_scaling_stress_gate.py --seeds 20
python scripts/run_pd_optimal_fusion_gate.py --seeds 80 --grid 2048 --greedy-instances 30
python scripts/run_expected_pd_greedy_gate.py --seeds 20 --budgets 20 30 40 --grid 512 --audit-instances 20
python scripts/run_ris_isac_gate.py --seeds 20 --budgets 20 30 --grid 512
python scripts/run_ris_phase_resolution_gate.py --seeds 20 --budgets 20 30 --grid 512
python scripts/run_ris_physics_gate.py --seeds 12 --grid 512
python scripts/run_ris_joint_budget_gate.py --seeds 12 --grid 512
python scripts/run_ris_placement_gate.py --seeds 12 --grid 512
python scripts/run_ris_multigrid_gate.py --seeds 12 --grid 512
python scripts/run_deployment_theory_gate.py --seeds 6 --grid 256
python scripts/run_lipschitz_adaptive_deployment_gate.py --seeds 3 --grid 128 --max-evaluations 250
python scripts/run_epsilon_closed_deployment_gate.py --seeds 1 --grid 128 --max-evaluations 250
python scripts/run_epsilon_closed_deployment_gate.py --seeds 3 --grid 128 --max-evaluations 3000 --local-evaluations 600 --output results/epsilon_closed_deployment_gate_3seeds.json
python scripts/run_g5_bootstrap_ci_gate.py --output results/g5_bootstrap_ci_gate.json
python scripts/run_g5_deployment_ci_gate.py --output results/g5_deployment_ci_gate.json
python scripts/run_global_resource_fairness_gate.py --output results/global_resource_fairness_gate.json
python scripts/run_ris_sensitivity_gate.py --seeds 12 --grid 512 --output results/ris_sensitivity_gate.json
python scripts/run_sota_baseline_gate.py --seeds 12 --grid 512 --output results/sota_baseline_gate.json
python scripts/run_budget_saturation_gate.py --seeds 12 --grid 512 --output results/budget_saturation_gate.json
python scripts/run_ris_shared_phase_gate.py --seeds 12 --grid 512 --output results/ris_shared_phase_gate.json
python scripts/run_exact_quota_gate.py --seeds 4 --grid 512 --output results/exact_quota_gate.json
python scripts/run_exact_budget_gate.py --seeds 20 --budgets 3 5 7 9 11 --grid 64 --output results/exact_budget_gate.json
python scripts/run_exact_maxmin_gate.py --seeds 20 --budgets 3 5 7 9 11 --grid 64 --output results/exact_maxmin_gate.json
python scripts/run_scaled_maxmin_gate.py --seeds 20 --budgets 5 9 --grid 64 --output results/scaled_maxmin_gate.json
python scripts/run_exact_selection_target_scalability.py --seeds 3 --budgets 8 12 16 --grid 32 --output results/exact_selection_target_scalability.json
python scripts/run_ris_subarray_gate.py --seeds 12 --grid 512 --output results/ris_subarray_gate.json
python scripts/run_ris_subarray_steering_gate.py --seeds 12 --grid 512 --g9-result results/ris_subarray_gate.json --output results/ris_subarray_steering_gate.json
python scripts/run_ris_aperture_scaling_gate.py --seeds 8 --grid 512 --output results/ris_aperture_scaling_gate.json
python scripts/run_derived_architecture_gate.py --seeds 8 --grid 512 --output results/derived_architecture_gate.json
python scripts/run_waterfilling_architecture_gate.py --seeds 8 --grid 512 --output results/waterfilling_architecture_gate.json
python scripts/run_exact_allocation_gate.py --seeds 8 --grid 512 --output results/exact_allocation_gate.json
python scripts/run_system_allocation_gate.py --seeds 8 --grid 512 --g14-result results/exact_allocation_gate.json --output results/system_allocation_gate.json
python scripts/run_single_move_certificate_gate.py --seeds 8 --grid 512 --g15-result results/system_allocation_gate.json --output results/single_move_certificate_gate.json
python scripts/run_multi_move_certificate_gate.py --seeds 8 --grid 512 --g16-result results/single_move_certificate_gate.json --output results/multi_move_certificate_gate.json
python scripts/run_joint_placement_allocation_gate.py --seeds 8 --grid 512 --g17-result results/multi_move_certificate_gate.json --output results/joint_placement_allocation_gate.json
python scripts/run_progressive_decentralization_gate.py --seeds 8 --grid 512 --g18-result results/joint_placement_allocation_gate.json --output results/progressive_decentralization_gate.json
python scripts/run_amplified_distributed_gate.py --seeds 8 --grid 512 --g18-result results/joint_placement_allocation_gate.json --output results/amplified_distributed_gate.json
python scripts/run_network_decentralization_gate.py --seeds 8 --grid 512 --g18-result results/joint_placement_allocation_gate.json --output results/network_decentralization_gate.json
python scripts/run_degraded_consensus_gate.py --seeds 8 --grid 512 --g18-result results/joint_placement_allocation_gate.json --output results/degraded_consensus_gate.json
python scripts/run_correlated_consensus_gate.py --seeds 8 --grid 512 --g18-result results/joint_placement_allocation_gate.json --output results/correlated_consensus_gate.json
python scripts/run_scalability_comparison_gate.py --seeds 5 --grid 512 --output results/scalability_comparison_gate.json
python scripts/run_scaled_g18_scalability_gate.py --seeds 3 --grid 512 --output results/scaled_g18_scalability_gate.json
python scripts/run_mobility_blockage_gate.py --seeds 5 --frames 12 --grid 512 --output results/mobility_blockage_gate.json
python scripts/run_multi_ris_gate.py --seeds 8 --grid 512 --output results/multi_ris_gate.json
python scripts/run_multi_ris_split_optimization_gate.py --seeds 5 --grid 512 --output results/multi_ris_split_optimization_gate.json
python scripts/run_variable_rate_report_gate.py --seeds 8 --grid 512 --output results/variable_rate_report_gate.json
python scripts/run_global_rate_optimization_gate.py --seeds 2 --grid 256 --output results/global_rate_optimization_gate.json
python scripts/run_exact_rate_certificate_gate.py --seeds 2 --budgets 28 40 --grid 256 --output results/exact_rate_certificate_gate.json
python scripts/run_hybrid_fusion_gate.py --seeds 5 --grid 512 --output results/hybrid_fusion_gate.json
python scripts/run_interference_sensitivity_gate.py --seeds 8 --grid 512 --output results/interference_sensitivity_gate.json
python scripts/run_spatial_interference_placement_gate.py --seeds 5 --grid 512 --output results/spatial_interference_placement_gate.json
python scripts/run_multi_interference_placement_gate.py --seeds 5 --grid 512 --output results/multi_interference_placement_gate.json
python scripts/run_upd_vs_ula_gate.py --seeds 5 --grid 512 --output results/upd_vs_ula_gate.json
python scripts/run_null_steering_gate.py --seeds 5 --grid 512 --output results/null_steering_gate.json
python scripts/run_quantized_null_steering_gate.py --seeds 5 --grid 512 --output results/quantized_null_steering_gate.json
python scripts/run_joint_null_placement_gate.py --seeds 5 --grid 512 --output results/joint_null_placement_gate.json
python scripts/run_distributed_relaxation_gate.py --seeds 5 --grid 512 --output results/distributed_relaxation_gate.json
python scripts/run_low_budget_snr_distributed_gate.py --seeds 5 --grid 512 --output results/low_budget_snr_distributed_gate.json
python scripts/run_consensus_parity_boundary_gate.py --seeds 5 --grid 512 --output results/consensus_parity_boundary_gate.json
python scripts/run_optimized_parity_boundary_gate.py --seeds 5 --grid 512 --output results/optimized_parity_boundary_gate.json
python scripts/run_exact_parity_boundary_gate.py --seeds 5 --grid 512 --output results/exact_parity_boundary_gate.json
python scripts/run_exact_min_majority_gate.py --uav-counts 3 6 8 12 16 --output results/exact_min_majority_gate.json
python scripts/run_fundamental_information_gate.py --seeds 5 --grid 512 --output results/fundamental_information_gate.json
python scripts/run_resource_information_law_gate.py --seeds 5 --grid 512 --output results/resource_information_law_gate.json
python scripts/run_exact_information_budget_gate.py --seeds 5 --grid 512 --output results/exact_information_budget_gate.json
python scripts/run_architecture_switch_gate.py --seeds 5 --grid 512 --output results/architecture_switch_gate.json
python scripts/run_target_wise_architecture_switch_gate.py --seeds 5 --grid 512 --output results/target_wise_architecture_switch_gate.json
python scripts/run_soft_reallocation_gate.py --seeds 5 --grid 512 --output results/soft_reallocation_gate.json
python scripts/run_mode_ascent_gate.py --seeds 5 --grid 512 --output results/mode_ascent_gate.json
python scripts/run_stochastic_mobility_gate.py --seeds 5 --frames 8 --grid 512 --output results/stochastic_mobility_gate.json
python scripts/run_prediction_aware_ris_gate.py --seeds 5 --frames 8 --grid 512 --output results/prediction_aware_ris_gate.json
python scripts/run_multi_step_prediction_gate.py --seeds 5 --frames 8 --grid 512 --output results/multi_step_prediction_gate.json
python scripts/run_covariance_aware_ris_gate.py --seeds 5 --frames 8 --grid 512 --output results/covariance_aware_ris_gate.json --horizon 3
python scripts/build_paper_tables.py
python scripts/draw_algorithm_evolution.py
python scripts/draw_scenario_evolution.py
python scripts/md_to_docx.py
python scripts/audit_submission_docx.py
python scripts/md_to_latex.py
python scripts/audit_submission_latex.py
python scripts/audit_submission_completeness.py
```

## Paper PDF build

From `paper/`:

```powershell
..\.tools\tectonic\tectonic.exe --keep-logs main.tex
```

## Regression tests

```powershell
python -m pytest -q
```

If the sandbox blocks the default temp/cache paths, use the workspace-local
fallback:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp .pytest_tmp_run
```

Use a fresh basetemp path if a previous sandbox run leaves its temp directory
locked (e.g. `.pytest_tmp_final`).

## Paper-number verification

```powershell
python scripts/verify_paper_numbers.py
```

The verifier cross-checks every key number used in `paper/submission.md`
against the audited result JSONs (G3-G5 including G5-SOTA, G8-K/M/S and
G8-target, G25, G43/G43-B, G47-G49, and G30-E).

## Notes

- G0-C front-end runs are the slowest; use `--integration-frames 4` only after
  the smoke passes.
- `--rayleigh-fading` is experimental and currently shows a negative
  equal-energy result; it is not part of the mainline claim.
- The G15 -> G16 -> G17 allocation chain must use the same seed count at
  every stage; refining a 4-seed allocation with a 1-seed objective produces
  a valid 1-seed local optimum but can be worse for the original 4-seed
  objective.
- Results are written under `results/` as JSON.
