# Run Matrix

Target runtime for the remote machine:

```text
E:\anaconda\conda\python.exe
```

All commands below use that interpreter.  The PowerShell batch runner is
`scripts/run_experiment_matrix.ps1`; it runs every cell sequentially and
writes logs to `results/run_matrix_logs/`.

## Matrix

| ID | Experiment | Command | Output |
| --- | --- | --- | --- |
| `mappo-q2` | Q=2 MAPPO / Greedy / Exact Joint | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_baseline.py --train-seeds 40 --test-seeds 20 --episodes 3000 --budgets 14 16 18 --output results\mappo_baseline.json` | `results/mappo_baseline.json` |
| `mappo-q4` | Q=4 MAPPO / Greedy / Exact Joint | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_baseline.py --targets 4 --reports 4 --train-seeds 40 --test-seeds 20 --episodes 3000 --budgets 28 32 36 --output results\mappo_q4_baseline.json` | `results/mappo_q4_baseline.json` |
| `mappo-q4-limited` | Q=4 weakened Exact Joint (max 3 bits) | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_baseline.py --targets 4 --reports 4 --train-seeds 40 --test-seeds 20 --episodes 3000 --budgets 28 32 36 --exact-max-bits 3 --output results\mappo_q4_limited_bits_baseline.json` | `results/mappo_q4_limited_bits_baseline.json` |
| `mappo-scaling` | Cross-target scaling figure | `& "E:\anaconda\conda\python.exe" scripts\run_mappo_greedy_scaling.py --targets 2 4 6 8 --train-seeds 20 --test-seeds 20 --episodes 800 --budget-multiplier 8 --output results\mappo_greedy_scaling.json --figure paper_figures\mappo_greedy_scaling.png` | `results/mappo_greedy_scaling.json`, `paper_figures/mappo_greedy_scaling.png` |
| `robust-allocation` | Robust multi-baseline allocation | `& "E:\anaconda\conda\python.exe" scripts\run_robust_stress_allocation.py --config config\demo.yaml --seeds 5 --budgets 16 20 24 --qos-target 0.7 --violation-limit 0.2 --output results\robust_stress_allocation_comparison.json` | `results/robust_stress_allocation_comparison.json` |
| `joint-power-bit` | Joint sensing-power and communication-bit allocation | `& "E:\anaconda\conda\python.exe" scripts\run_joint_power_bit_gate.py --budgets 8 12 16 --grid 32 --output results\joint_power_bit_gate.json` | `results/joint_power_bit_gate.json` |
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
