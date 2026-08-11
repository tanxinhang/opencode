# Run Matrix

Target runtime for the remote machine:

```text
E:\anaconda\conda\python.exe
```

All commands below use that interpreter.  The PowerShell batch runner is
`scripts/run_experiment_matrix.ps1`; it runs every cell sequentially and
writes logs to `results/run_matrix_logs/`.

On Windows with Anaconda, the runner sets `KMP_DUPLICATE_LIB_OK=TRUE` before
launching Python to avoid the duplicate `libiomp5md.dll` OpenMP runtime
error.

## Matrix

| ID | Experiment | Command | Output |
| --- | --- | --- | --- |
| `mappo-q2` | Q=2 MAPPO / Greedy / Exact Joint | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_baseline.py --train-seeds 40 --test-seeds 20 --episodes 3000 --budgets 14 16 18 --output results\mappo_baseline.json` | `results/mappo_baseline.json` |
| `mappo-q4` | Q=4 MAPPO / Greedy / Exact Joint | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_baseline.py --targets 4 --reports 4 --train-seeds 40 --test-seeds 20 --episodes 3000 --budgets 28 32 36 --output results\mappo_q4_baseline.json` | `results/mappo_q4_baseline.json` |
| `mappo-q4-limited` | Q=4 weakened Exact Joint (max 3 bits) | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_baseline.py --targets 4 --reports 4 --train-seeds 40 --test-seeds 20 --episodes 3000 --budgets 28 32 36 --exact-max-bits 3 --output results\mappo_q4_limited_bits_baseline.json` | `results/mappo_q4_limited_bits_baseline.json` |
| `mappo-scaling` | Cross-target scaling figure | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_greedy_scaling.py --targets 2 4 6 8 --train-seeds 20 --test-seeds 20 --episodes 800 --budget-multiplier 8 --output results\mappo_greedy_scaling.json --figure paper_figures\mappo_greedy_scaling.png` | `results/mappo_greedy_scaling.json`, `paper_figures/mappo_greedy_scaling.png` |
| `robust-allocation` | Robust multi-baseline allocation | `& "E:\anaconda\conda\python.exe" scripts\run_robust_stress_allocation.py --config config\demo.yaml --seeds 5 --budgets 16 20 24 --qos-target 0.7 --violation-limit 0.2 --output results\robust_stress_allocation_comparison.json` | `results/robust_stress_allocation_comparison.json` |
| `joint-power-bit` | Joint sensing-power and communication-bit allocation | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_bit_gate.py --budgets 8 12 16 --grid 32 --output results\joint_power_bit_gate.json` | `results/joint_power_bit_gate.json` |
| `joint-power-bit-scaling` | Joint power-bit enumeration scaling | `& "E:\anaconda\conda\python.exe" scripts\benchmark_joint_power_bit_scaling.py --reports 2 4 6 --grid 32 --output results\joint_power_bit_scaling_benchmark.json` | `results/joint_power_bit_scaling_benchmark.json` |
| `communication-aware` | Communication-aware sensing score certificate | `& "E:\anaconda\conda\python.exe" scripts\run_communication_aware_gate.py --seeds 10 --budgets 2 4 6 8 --output results\communication_aware_gate.json` | `results/communication_aware_gate.json` |
| `communication-ambiguity` | Communication ambiguity endpoint reduction | `& "E:\anaconda\conda\python.exe" scripts\run_communication_ambiguity_gate.py --seeds 5 --output results\communication_ambiguity_gate.json` | `results/communication_ambiguity_gate.json` |
| `robust-joint-power-bit` | Robust joint power-bit allocation | `& "E:\anaconda\conda\python.exe" scripts\run_robust_joint_power_bit_gate.py --budgets 8 12 16 --grid 32 --output results\robust_joint_power_bit_gate.json` | `results/robust_joint_power_bit_gate.json` |
| `robust-communication-aware` | Robust communication-aware sensing score | `& "E:\anaconda\conda\python.exe" scripts\run_robust_communication_aware_gate.py --seeds 10 --budgets 4 6 8 --output results\robust_communication_aware_gate.json` | `results/robust_communication_aware_gate.json` |
| `robust-cas-divergence` | When robust CAS differs from nominal CAS | `& "E:\anaconda\conda\python.exe" scripts\run_robust_cas_divergence_gate.py --seeds 10 --budget 6 --output results\robust_cas_divergence_gate.json` | `results/robust_cas_divergence_gate.json` |
| `power-bit-split` | Optimal sensing-power vs communication-bit split | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_bit_split_gate.py --budgets 8 12 16 --grid 32 --output results\joint_power_bit_split_gate.json` | `results/joint_power_bit_split_gate.json` |
| `exact-vs-greedy-config` | Where Exact Joint beats Greedy | `& "E:\anaconda\conda\python.exe" scripts\run_exact_vs_greedy_config_gate.py --seeds 20 --targets 2 4 --budget-multiplier 8 --output results\exact_vs_greedy_config_gate.json` | `results/exact_vs_greedy_config_gate.json` |
| `winner-take-all-power` | Winner-take-all sensing power allocation | `& "E:\anaconda\conda\python.exe" scripts\run_winner_take_all_power_gate.py --seeds 10 --output results\winner_take_all_power_gate.json` | `results/winner_take_all_power_gate.json` |
| `winner-take-all-joint` | Joint power-bit with winner-take-all power | `& "E:\anaconda\conda\python.exe" scripts\run_winner_take_all_joint_proportional_gate.py --seeds 10 --budget 4 --output results\winner_take_all_joint_proportional_gate.json` | `results/winner_take_all_joint_proportional_gate.json` |
| `winner-take-all-scaling` | Winner-take-all enumeration scaling | `& "E:\anaconda\conda\python.exe" scripts\benchmark_winner_take_all_scaling.py --reports 2 3 4 --budget 4 --grid 16 --output results\winner_take_all_scaling_benchmark.json` | `results/winner_take_all_scaling_benchmark.json` |
| `joint-power-comparison` | MAPPO vs Greedy vs WTA-Greedy vs UCB-WTA vs UCB-NOMP vs NOMP-Greedy vs WTA-Exact | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_comparison.py --reports 2 --targets 2 --budgets 8 10 12 --episodes 300 --train-seeds 30 --test-seeds 20 --output results\joint_power_comparison.json` | `results/joint_power_comparison.json` |
| `joint-power-comparison-hetero` | Same comparison in heterogeneous target/report channels | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_comparison.py --mode heterogeneous --reports 2 --targets 2 --budgets 8 10 12 --episodes 300 --train-seeds 30 --test-seeds 20 --output results\joint_power_comparison_heterogeneous.json` | `results/joint_power_comparison_heterogeneous.json` |
| `joint-power-scaling` | Worst P_D versus Q=2/4/6/8 with per-target budget 4Q | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_scaling.py --targets 2 4 6 8 --train-seeds 20 --test-seeds 20 --episodes 200 --output results\joint_power_scaling.json --figure paper_figures\joint_power_scaling.png` | `results/joint_power_scaling.json`, `paper_figures/joint_power_scaling.png` |
| `joint-power-comm-mismatch` | Per-link sensing/communication mismatch with robust expected-P_D oracle | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_comm_mismatch_gate.py --seeds 20 --budgets 8 10 12 --output results\joint_power_comm_mismatch_gate.json` | `results/joint_power_comm_mismatch_gate.json` |
| `joint-power-summary` | Consolidated tables and overall comparison figure | `& "E:\anaconda\conda\python.exe" scripts\summarize_joint_power_results.py` | `paper_figures/joint_power_overall_comparison.png`, console tables |
| `nomp-report-scaling` | NOMP vs WTA versus report count under per-link channels | `& "E:\anaconda\conda\python.exe" scripts\run_nomp_report_scaling_gate.py --reports 2 4 6 8 10 --seeds 5 --samples 512 --candidate-budget 32 --output results\nomp_report_scaling_gate.json --figure paper_figures\nomp_report_scaling.png` | `results/nomp_report_scaling_gate.json`, `paper_figures/nomp_report_scaling.png` |
| `qos-weighted-maxmin` | QoS floors/weights multi-target max-min | `& "E:\anaconda\conda\python.exe" scripts\run_qos_weighted_maxmin_gate.py --seeds 10 --budgets 8 10 12 --output results\qos_weighted_maxmin_gate.json` | `results/qos_weighted_maxmin_gate.json` |
| `qr-scenario-comparison` | Q x R x scenario comparison grid | `& "E:\anaconda\conda\python.exe" scripts\run_qr_scenario_comparison.py --modes homogeneous heterogeneous comm_mismatch --targets 2 4 6 --reports 2 3 4 --seeds 5 --output results\qr_scenario_comparison.json --figure paper_figures\qr_scenario_comparison.png` | `results/qr_scenario_comparison.json`, `paper_figures/qr_scenario_comparison.png` |
| `unknown-environment` | Generalization under unseen channels and targets | `& "E:\anaconda\conda\python.exe" scripts\run_unknown_environment_gate.py --budgets 8 10 12 --episodes 300 --train-seeds 30 --test-seeds 10 --output results\unknown_environment_gate.json --figure paper_figures\unknown_environment.png` | `results/unknown_environment_gate.json`, `paper_figures/unknown_environment.png` |
| `robust-curriculum` | Channel-aware MAPPO curriculum plus NOMP robustness | `& "E:\anaconda\conda\python.exe" scripts\run_robust_curriculum_gate.py --budgets 8 10 12 --episodes 300 --train-seeds 20 --test-seeds 10 --output results\robust_curriculum_gate.json --figure paper_figures\robust_curriculum.png` | `results/robust_curriculum_gate.json`, `paper_figures/robust_curriculum.png` |
| `priority-middleware` | MAPPO-guided QoS weights inside NOMP solving | `& "E:\anaconda\conda\python.exe" scripts\run_priority_middleware_gate.py --episodes 12 --samples 512 --candidate-budget 8 --output results\priority_middleware_gate.json --figure paper_figures\priority_middleware.png` | `results/priority_middleware_gate.json`, `paper_figures/priority_middleware.png` |
| `mappo-nomp-reward` | MAPPO trained with NOMP-final P_D reward | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_nomp_reward_gate.py --budgets 8 10 12 --episodes 300 --train-seeds 30 --test-seeds 10 --output results\mappo_nomp_reward_gate.json --figure paper_figures\mappo_nomp_reward.png` | `results/mappo_nomp_reward_gate.json`, `paper_figures/mappo_nomp_reward.png` |
| `hard-channel` | Robustness under increasing channel difficulty | `& "E:\anaconda\conda\python.exe" scripts\run_hard_channel_gate.py --budgets 8 10 12 --episodes 300 --train-seeds 20 --test-seeds 10 --output results\hard_channel_gate.json --figure paper_figures\hard_channel.png` | `results/hard_channel_gate.json`, `paper_figures/hard_channel.png` |
| `ppo-nomp-reward-robustness` | PPO + NOMP-final reward under unseen channels/targets | `& "E:\anaconda\conda\python.exe" scripts\run_ppo_nomp_reward_robustness_gate.py --budgets 8 10 12 --episodes 300 --train-seeds 30 --test-seeds 10 --output results\ppo_nomp_reward_robustness_gate.json --figure paper_figures\ppo_nomp_reward_robustness.png` | `results/ppo_nomp_reward_robustness_gate.json`, `paper_figures/ppo_nomp_reward_robustness.png` |
| `physical-hard-channel` | SNR-derived channel difficulty with R=2/4/6 | `& "E:\anaconda\conda\python.exe" scripts\run_physical_hard_channel_gate.py --reports 2 4 6 --seeds 10 --output results\physical_hard_channel_gate.json --figure paper_figures\physical_hard_channel.png` | `results/physical_hard_channel_gate.json`, `paper_figures/physical_hard_channel.png` |
| `physical-ppo-nomp` | PPO + NOMP (multi-proposal) under physical hard channels | `& "E:\anaconda\conda\python.exe" scripts\run_physical_ppo_nomp_gate.py --reports 2 4 --episodes 100 --train-seeds 20 --test-seeds 3 --output results\physical_ppo_nomp_gate.json --figure paper_figures\physical_ppo_nomp.png` | `results/physical_ppo_nomp_gate.json`, `paper_figures/physical_ppo_nomp.png` |
| `mappo-nomp-ensemble` | Multi-temperature PPO+NOMP proposal ensemble | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_nomp_ensemble_gate.py --episodes 80 --train-seeds 15 --test-seeds 3 --budget 16 --output results\mappo_nomp_ensemble_gate.json` | `results/mappo_nomp_ensemble_gate.json` |
| `error-feedback` | WTA allocation under coefficient error with feedback | `& "E:\anaconda\conda\python.exe" scripts\run_error_feedback_gate.py --seeds 20 --rounds 1 3 10 30 --noise 0.8 --output results\error_feedback_gate.json` | `results/error_feedback_gate.json` |
| `ucb-error-feedback` | UCB WTA feedback with certificate stopping | `& "E:\anaconda\conda\python.exe" scripts\run_ucb_error_feedback_gate.py --seeds 20 --max-rounds 50 --noise 0.2 --output results\ucb_error_feedback_gate.json` | `results/ucb_error_feedback_gate.json` |
| `exact-joint-scaling` | Exact Joint target-count benchmark | `& "E:\anaconda\conda\python.exe" scripts\benchmark_exact_joint_scaling.py --targets 2 4 8 16 --reports 4 --grid 16 --output results\exact_joint_scaling_benchmark.json` | `results/exact_joint_scaling_benchmark.json` |
| `benchmark-smoke` | Robustness performance smoke | `& "E:\anaconda\conda\python.exe" scripts\benchmark_robustness_performance.py --output results\robustness_performance_benchmark.json` | `results/robustness_performance_benchmark.json` |
| `benchmark-formal` | Robustness performance formal sweep | `& "E:\anaconda\conda\python.exe" scripts\benchmark_robustness_performance.py --formal --output results\robustness_performance_benchmark.json` | `results/robustness_performance_benchmark.json` |
| `verify-tests` | Full regression tests | `& "E:\anaconda\conda\python.exe" -m pytest -q -p no:cacheprovider --basetemp .pytest_tmp_run_matrix` | console |

## Batch runner

Run all cells:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiment_matrix.ps1
```

Run only one cell, for example the scaling figure:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiment_matrix.ps1 -Only mappo-scaling
```

Dry-run to print commands without executing:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiment_matrix.ps1 -DryRun
```
