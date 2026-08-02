# UAV-OTFS-ISAC selective fusion simulator

This repository implements the system model in
`UAV_OTFS_ISAC_论证与系统模型_revised_final.docx` as a reproducible Python
simulation. The current release focuses on the paper's minimum publishable
scope: fixed geometry/waveform parameters, correlated local evidence,
multi-bit reporting over error-prone links, detectable erasures, target-wise
report selection, and linear deflection-optimal fusion.

## Implemented model chain

1. Geometry and fractional-Doppler-dependent OTFS evidence moments.
2. Geometry-dependent air-to-air reporting reliability, low-bit scalar
   quantization, and exact binary symmetric channel transition.
3. Positive-definite cross-UAV covariance with shrinkage regularization.
4. Detectable packet erasures represented by a random received set.
5. Exact expected deflection by reception-pattern enumeration, with an SAA
   fallback for larger sets.
6. Schur-complement conditional marginal deflection.
7. Two-stage greedy scheduling: minimize normalized QoS shortfall, then improve
   expected total deflection within the bit budget.
8. Fusion weights recomputed from the actually received set.
9. Monte Carlo `P_D/P_FA`, weak-target performance, bits, and selection trace.
10. Exact reception-pattern loss distributions, upper-tail CVaR, and QoS
    violation probability.
11. Exact target-level evidence-set enumeration plus multiple-choice knapsack
    dynamic programming for mean-CVaR portfolio design.

## Run

```powershell
python scripts/run_demo.py --config config/demo.yaml
python scripts/run_benchmarks.py --config config/demo.yaml
python scripts/run_oracle_study.py --config config/oracle_small.yaml
python scripts/run_ablation_study.py --config config/demo.yaml
python scripts/run_sensitivity_study.py --config config/demo.yaml
python scripts/run_risk_portfolio_study.py --config config/demo.yaml
python scripts/run_chance_portfolio_study.py --config config/demo.yaml
python -m pytest -q
```

The demo writes `results/demo_summary.json`.
The benchmark script compares the proposed selector against no cooperation,
sensing-only Top-K, communication-only Top-K, independent post-report ranking,
random selection, and all-scheduled fusion. `exhaustive_oracle` is provided for
small candidate sets. It uses common random numbers across methods and reports
both the optimized weighted expected deflection and detection probabilities;
these metrics need not rank every method identically in a finite experiment.

`run_risk_portfolio_study.py` compares a risk-neutral portfolio, a mean-CVaR
portfolio, and the former marginal-deflection greedy solver. CVaR controls the
severity of upper-tail deficits; it is not presented as a substitute for a
chance constraint on the probability of violating the detection threshold.
`run_chance_portfolio_study.py` adds per-target chance constraints and returns
the minimum weighted reliability relaxation when the requested reliability is
not attainable under the available reporting budget.

## What is deliberately not claimed

- The greedy score is not claimed to be submodular or to have a fixed
  approximation ratio.
- `pi * Delta-D / bits` is only the document's open-loop first-order score;
  selection uses the true expected-deflection difference when exact
  enumeration is tractable.
- The fusion direction is optimal only among linear fusion rules under the
  null-hypothesis deflection criterion.
- The built-in waveform calibration is a lightweight DD leakage model. It is
  not a complete SDR-grade OTFS transceiver.

## GitHub reuse policy

See `THIRD_PARTY.md`. No third-party source code is copied into the core
package. A user may opt into the external GPL OTFS toolbox through
`ExternalOTFSBackend`.
