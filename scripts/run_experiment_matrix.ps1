param(
    [string]$Python = "E:\anaconda\conda\python.exe",
    [string[]]$Only = @(),
    [switch]$DryRun
)

# Anaconda on Windows often loads libiomp5md.dll from both NumPy and PyTorch;
# this silences the duplicate-OpenMP-runtime error before launching Python.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "results\run_matrix_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$cells = @(
    [pscustomobject]@{
        Name = "mappo-q2"
        Args = @(
            "scripts\run_mappo_baseline.py",
            "--train-seeds", "50",
            "--test-seeds", "30",
            "--episodes", "5000",
            "--budgets", "14", "16", "18",
            "--output", "results\mappo_baseline.json"
        )
    },
    [pscustomobject]@{
        Name = "mappo-q4"
        Args = @(
            "scripts\run_mappo_baseline.py",
            "--targets", "4",
            "--reports", "4",
            "--train-seeds", "50",
            "--test-seeds", "30",
            "--episodes", "5000",
            "--budgets", "28", "32", "36",
            "--output", "results\mappo_q4_baseline.json"
        )
    },
    [pscustomobject]@{
        Name = "mappo-q4-limited"
        Args = @(
            "scripts\run_mappo_baseline.py",
            "--targets", "4",
            "--reports", "4",
            "--train-seeds", "50",
            "--test-seeds", "30",
            "--episodes", "5000",
            "--budgets", "28", "32", "36",
            "--exact-max-bits", "3",
            "--output", "results\mappo_q4_limited_bits_baseline.json"
        )
    },
    [pscustomobject]@{
        Name = "mappo-scaling"
        Args = @(
            "scripts\run_mappo_greedy_scaling.py",
            "--targets", "2", "4", "6", "8",
            "--train-seeds", "50",
            "--test-seeds", "30",
            "--episodes", "5000",
            "--budget-multiplier", "8",
            "--output", "results\mappo_greedy_scaling.json",
            "--figure", "paper_figures\mappo_greedy_scaling.png"
        )
    },
    [pscustomobject]@{
        Name = "robust-allocation"
        Args = @(
            "scripts\run_robust_stress_allocation.py",
            "--config", "config\demo.yaml",
            "--seeds", "5",
            "--budgets", "16", "20", "24",
            "--qos-target", "0.7",
            "--violation-limit", "0.2",
            "--output", "results\robust_stress_allocation_comparison.json"
        )
    },
    [pscustomobject]@{
        Name = "joint-power-bit"
        Args = @(
            "scripts\run_joint_power_bit_gate.py",
            "--budgets", "8", "12", "16",
            "--grid", "32",
            "--output", "results\joint_power_bit_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "joint-power-bit-scaling"
        Args = @(
            "scripts\benchmark_joint_power_bit_scaling.py",
            "--reports", "2", "4", "6",
            "--grid", "32",
            "--output", "results\joint_power_bit_scaling_benchmark.json"
        )
    },
    [pscustomobject]@{
        Name = "communication-aware"
        Args = @(
            "scripts\run_communication_aware_gate.py",
            "--seeds", "10",
            "--budgets", "2", "4", "6", "8",
            "--output", "results\communication_aware_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "communication-ambiguity"
        Args = @(
            "scripts\run_communication_ambiguity_gate.py",
            "--seeds", "5",
            "--output", "results\communication_ambiguity_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "robust-joint-power-bit"
        Args = @(
            "scripts\run_robust_joint_power_bit_gate.py",
            "--budgets", "8", "12", "16",
            "--grid", "32",
            "--output", "results\robust_joint_power_bit_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "robust-communication-aware"
        Args = @(
            "scripts\run_robust_communication_aware_gate.py",
            "--seeds", "10",
            "--budgets", "4", "6", "8",
            "--output", "results\robust_communication_aware_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "robust-cas-divergence"
        Args = @(
            "scripts\run_robust_cas_divergence_gate.py",
            "--seeds", "10",
            "--budget", "6",
            "--output", "results\robust_cas_divergence_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "power-bit-split"
        Args = @(
            "scripts\run_joint_power_bit_split_gate.py",
            "--budgets", "8", "12", "16",
            "--grid", "32",
            "--output", "results\joint_power_bit_split_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "exact-vs-greedy-config"
        Args = @(
            "scripts\run_exact_vs_greedy_config_gate.py",
            "--seeds", "20",
            "--targets", "2", "4",
            "--budget-multiplier", "8",
            "--output", "results\exact_vs_greedy_config_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "winner-take-all-power"
        Args = @(
            "scripts\run_winner_take_all_power_gate.py",
            "--seeds", "10",
            "--output", "results\winner_take_all_power_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "winner-take-all-joint"
        Args = @(
            "scripts\run_winner_take_all_joint_proportional_gate.py",
            "--seeds", "10",
            "--budget", "4",
            "--output", "results\winner_take_all_joint_proportional_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "winner-take-all-scaling"
        Args = @(
            "scripts\benchmark_winner_take_all_scaling.py",
            "--reports", "2", "3", "4",
            "--budget", "4",
            "--grid", "16",
            "--output", "results\winner_take_all_scaling_benchmark.json"
        )
    },
    [pscustomobject]@{
        Name = "joint-power-comparison"
        Args = @(
            "scripts\run_joint_power_comparison.py",
            "--reports", "2",
            "--targets", "2",
            "--budgets", "8", "10", "12",
            "--episodes", "300",
            "--train-seeds", "30",
            "--test-seeds", "20",
            "--output", "results\joint_power_comparison.json"
        )
    },
    [pscustomobject]@{
        Name = "joint-power-comparison-hetero"
        Args = @(
            "scripts\run_joint_power_comparison.py",
            "--mode", "heterogeneous",
            "--reports", "2",
            "--targets", "2",
            "--budgets", "8", "10", "12",
            "--episodes", "300",
            "--train-seeds", "30",
            "--test-seeds", "20",
            "--output", "results\joint_power_comparison_heterogeneous.json"
        )
    },
    [pscustomobject]@{
        Name = "joint-power-scaling"
        Args = @(
            "scripts\run_joint_power_scaling.py",
            "--targets", "2", "4", "6", "8",
            "--train-seeds", "20",
            "--test-seeds", "20",
            "--episodes", "200",
            "--output", "results\joint_power_scaling.json",
            "--figure", "paper_figures\joint_power_scaling.png"
        )
    },
    [pscustomobject]@{
        Name = "joint-power-comm-mismatch"
        Args = @(
            "scripts\run_joint_power_comm_mismatch_gate.py",
            "--seeds", "20",
            "--budgets", "8", "10", "12",
            "--output", "results\joint_power_comm_mismatch_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "joint-power-summary"
        Args = @(
            "scripts\summarize_joint_power_results.py"
        )
    },
    [pscustomobject]@{
        Name = "nomp-report-scaling"
        Args = @(
            "scripts\run_nomp_report_scaling_gate.py",
            "--reports", "2", "4", "6",
            "--seeds", "10",
            "--output", "results\nomp_report_scaling_gate.json",
            "--figure", "paper_figures\nomp_report_scaling.png"
        )
    },
    [pscustomobject]@{
        Name = "qos-weighted-maxmin"
        Args = @(
            "scripts\run_qos_weighted_maxmin_gate.py",
            "--seeds", "10",
            "--budgets", "8", "10", "12",
            "--output", "results\qos_weighted_maxmin_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "qr-scenario-comparison"
        Args = @(
            "scripts\run_qr_scenario_comparison.py",
            "--modes", "homogeneous", "heterogeneous", "comm_mismatch",
            "--targets", "2", "4", "6",
            "--reports", "2", "3", "4",
            "--seeds", "5",
            "--output", "results\qr_scenario_comparison.json",
            "--figure", "paper_figures\qr_scenario_comparison.png"
        )
    },
    [pscustomobject]@{
        Name = "error-feedback"
        Args = @(
            "scripts\run_error_feedback_gate.py",
            "--seeds", "20",
            "--rounds", "1", "3", "10", "30",
            "--noise", "0.8",
            "--output", "results\error_feedback_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "ucb-error-feedback"
        Args = @(
            "scripts\run_ucb_error_feedback_gate.py",
            "--seeds", "20",
            "--max-rounds", "50",
            "--noise", "0.2",
            "--output", "results\ucb_error_feedback_gate.json"
        )
    },
    [pscustomobject]@{
        Name = "exact-joint-scaling"
        Args = @(
            "scripts\benchmark_exact_joint_scaling.py",
            "--targets", "2", "4", "8", "16",
            "--reports", "4",
            "--grid", "16",
            "--output", "results\exact_joint_scaling_benchmark.json"
        )
    },
    [pscustomobject]@{
        Name = "benchmark-smoke"
        Args = @(
            "scripts\benchmark_robustness_performance.py",
            "--output", "results\robustness_performance_benchmark.json"
        )
    },
    [pscustomobject]@{
        Name = "benchmark-formal"
        Args = @(
            "scripts\benchmark_robustness_performance.py",
            "--formal",
            "--output", "results\robustness_performance_benchmark.json"
        )
    },
    [pscustomobject]@{
        Name = "verify-tests"
        Args = @(
            "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--basetemp", ".pytest_tmp_run_matrix"
        )
    }
)

foreach ($cell in $cells) {
    if ($Only.Count -gt 0 -and $Only -notcontains $cell.Name) {
        continue
    }
    $commandText = (@($Python) + $cell.Args) -join " "
    Write-Host "==> $($cell.Name): $commandText"
    if ($DryRun) {
        continue
    }
    $logFile = Join-Path $logDir "$($cell.Name).log"
    & $Python @($cell.Args) 2>&1 | Tee-Object -FilePath $logFile
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        throw "Cell '$($cell.Name)' failed with exit code $exit"
    }
}

Write-Host "Run matrix finished."
