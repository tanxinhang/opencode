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
```

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
```

## Regression tests

```powershell
python -m pytest -q
```

## Notes

- G0-C front-end runs are the slowest; use `--integration-frames 4` only after
  the smoke passes.
- `--rayleigh-fading` is experimental and currently shows a negative
  equal-energy result; it is not part of the mainline claim.
- Results are written under `results/` as JSON.
