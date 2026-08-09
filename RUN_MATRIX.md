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
| `joint-power-comparison` | MAPPO vs Greedy vs WTA-Greedy vs WTA-Exact | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_comparison.py --reports 2 --targets 2 --budgets 8 10 12 --episodes 300 --train-seeds 30 --test-seeds 20 --output results\joint_power_comparison.json` | `results/joint_power_comparison.json` |
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
