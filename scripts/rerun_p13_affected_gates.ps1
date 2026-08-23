# P1-1/P1-3 model-layer rerun (full affected-gate matrix).
#
# The P1-1 RIS direct-blockage fix and the P1-3 post-communication
# covariance attenuation clamp change every build_models output, so every
# gate whose results JSON embeds build_models / RIS physics must be rerun
# before the paper tables are rebuilt.  Gates that use only the toy
# payload-bit oracle (run_quantization_*, run_joint_multi_gate) or the
# pure toy exact-joint models are NOT affected and are excluded.
#
# Usage (powershell):
#   .\scripts\rerun_p13_affected_gates.ps1            # run everything
#   .\scripts\rerun_p13_affected_gates.ps1 -Only <id> # run one cell
#   .\scripts\rerun_p13_affected_gates.ps1 -DryRun    # print the matrix
#
# Every command writes its normal results/*.json output (fixed model).

param(
  [string]$Py = "E:\anaconda\conda\python.exe",
  [switch]$DryRun = $false,
  [string]$Only = ""
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot\..   # repo root

$cells = @(
  @{ id="demo";          cmd="scripts/run_demo.py --config config/demo.yaml" },
  @{ id="benchmarks";    cmd="scripts/run_benchmarks.py --config config/demo.yaml" },
  @{ id="oracle";        cmd="scripts/run_oracle_study.py --config config/oracle_small.yaml" },
  @{ id="ablation";      cmd="scripts/run_ablation_study.py --config config/demo.yaml" },
  @{ id="maxmin";        cmd="scripts/run_exact_maxmin_gate.py --seeds 500 --budgets 3 5 7 9 11 --grid 64" },
  @{ id="pd_greedy";     cmd="scripts/run_expected_pd_greedy_gate.py --seeds 5" },
  @{ id="pd_optimal";    cmd="scripts/run_pd_optimal_fusion_gate.py" },
  @{ id="ris_control";   cmd="scripts/run_ris_isac_gate.py --seeds 5" },
  @{ id="ris_phase";     cmd="scripts/run_ris_phase_resolution_gate.py --seeds 5" },
  @{ id="ris_physics";   cmd="scripts/run_ris_physics_gate.py --seeds 5" },
  @{ id="ris_joint";     cmd="scripts/run_ris_joint_budget_gate.py --seeds 5" },
  @{ id="ris_place";     cmd="scripts/run_ris_placement_gate.py --seeds 5" },
  @{ id="ris_mgrid";     cmd="scripts/run_ris_multigrid_gate.py --seeds 5" },
  @{ id="ris_shared";    cmd="scripts/run_ris_shared_phase_gate.py --seeds 6" },
  @{ id="ris_subarray";  cmd="scripts/run_ris_subarray_gate.py --seeds 6" },
  @{ id="ris_substeer";  cmd="scripts/run_ris_subarray_steering_gate.py --seeds 6" },
  @{ id="ris_aperture";  cmd="scripts/run_ris_aperture_scaling_gate.py --seeds 4" },
  @{ id="ris_sens";      cmd="scripts/run_ris_sensitivity_gate.py --seeds 6" },
  @{ id="ris_upd";       cmd="scripts/run_upd_vs_ula_gate.py --seeds 2" },
  @{ id="ris_null";      cmd="scripts/run_null_steering_gate.py --seeds 2" },
  @{ id="ris_nullq";     cmd="scripts/run_quantized_null_steering_gate.py --seeds 2" },
  @{ id="ris_jointplace"; cmd="scripts/run_joint_null_placement_gate.py --seeds 2" },
  @{ id="sota";          cmd="scripts/run_sota_baseline_gate.py --seeds 12" },
  @{ id="budget_sat";    cmd="scripts/run_budget_saturation_gate.py --seeds 6" },
  @{ id="fairness";      cmd="scripts/run_global_resource_fairness_gate.py" },
  @{ id="derived_arch";  cmd="scripts/run_derived_architecture_gate.py --seeds 4" },
  @{ id="waterfill";     cmd="scripts/run_waterfilling_architecture_gate.py --seeds 4" },
  @{ id="exact_alloc";   cmd="scripts/run_exact_allocation_gate.py --seeds 4" },
  @{ id="system_alloc";  cmd="scripts/run_system_allocation_gate.py --seeds 4" },
  @{ id="joint_place_alloc"; cmd="scripts/run_joint_placement_allocation_gate.py --seeds 4" },
  @{ id="g5_boot";       cmd="scripts/run_g5_bootstrap_ci_gate.py" },
  @{ id="g5_deploy_ci";  cmd="scripts/run_g5_deployment_ci_gate.py" },
  @{ id="consensus";     cmd="scripts/run_progressive_decentralization_gate.py --seeds 4" },
  @{ id="amplified";     cmd="scripts/run_amplified_distributed_gate.py --seeds 4" },
  @{ id="network_dec";   cmd="scripts/run_network_decentralization_gate.py --seeds 4" },
  @{ id="degraded_cons"; cmd="scripts/run_degraded_consensus_gate.py --seeds 4" },
  @{ id="corr_cons";     cmd="scripts/run_correlated_consensus_gate.py --seeds 4" },
  @{ id="scal_compar";   cmd="scripts/run_scalability_comparison_gate.py --seeds 3" },
  @{ id="g18_scal";      cmd="scripts/run_scaled_g18_scalability_gate.py --seeds 2" },
  @{ id="mob_block";     cmd="scripts/run_mobility_blockage_gate.py --seeds 2 --frames 8" },
  @{ id="multi_ris";     cmd="scripts/run_multi_ris_gate.py --seeds 3" },
  @{ id="multi_ris_split"; cmd="scripts/run_multi_ris_split_optimization_gate.py --seeds 2" },
  @{ id="rate_report";   cmd="scripts/run_variable_rate_report_gate.py --seeds 2" },
  @{ id="global_rate";   cmd="scripts/run_global_rate_optimization_gate.py --seeds 2" },
  @{ id="hybrid_fus";    cmd="scripts/run_hybrid_fusion_gate.py --seeds 2" },
  @{ id="interf_sens";   cmd="scripts/run_interference_sensitivity_gate.py --seeds 2" },
  @{ id="spat interf";   cmd="scripts/run_spatial_interference_placement_gate.py --seeds 2" },
  @{ id="multi_interf";  cmd="scripts/run_multi_interference_placement_gate.py --seeds 2" },
  @{ id="cov_ris";       cmd="scripts/run_covariance_aware_ris_gate.py --seeds 4 --frames 8" },
  @{ id="pred_ris";      cmd="scripts/run_prediction_aware_ris_gate.py --seeds 4 --frames 8" },
  @{ id="arch_switch";   cmd="scripts/run_architecture_switch_gate.py --seeds 4" },
  @{ id="target_switch"; cmd="scripts/run_target_wise_architecture_switch_gate.py --seeds 4" },
  @{ id="soft_realloc";  cmd="scripts/run_soft_reallocation_gate.py --seeds 4" },
  @{ id="mode_ascent";   cmd="scripts/run_mode_ascent_gate.py --seeds 4" },
  @{ id="stoch_mob";     cmd="scripts/run_stochastic_mobility_gate.py --seeds 4 --frames 8" },
  @{ id="multi_step";    cmd="scripts/run_multi_step_prediction_gate.py --seeds 4 --frames 8" }
)

if ($DryRun) {
  foreach ($c in $cells) { Write-Output ("{0,-18} {1}" -f $c.id, $c.cmd) }
  exit 0
}
$fail = @()
foreach ($c in $cells) {
  if ($Only -ne "" -and $c.id -ne $Only) { continue }
  Write-Output ("== [" + $c.id + "] " + $c.cmd)
  & $Py $c.cmd.Split(" ") ; if ($LASTEXITCODE -ne 0) { $fail += $c.id }
}
if ($fail.Count) {
  Write-Error ("FAILED cells: " + ($fail -join ", ")); exit 1
}
Write-Output ("ALL OK (" + ($cells.Count) + " cells)")