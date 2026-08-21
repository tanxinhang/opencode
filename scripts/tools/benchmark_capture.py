"""Capture benchmark outputs with wall time, for before/after comparison.

Usage: python scripts/benchmark_capture.py <out_json> <cmd_args...>
Runs each benchmark command, times it, and stores {name, seconds, payload}.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

PY = r"E:\anaconda\conda\python.exe"

BENCHMARKS = [
    ("rob_smoke", [PY, r"scripts\benchmark_robustness_performance.py"]),
    ("rob_formal", [PY, r"scripts\benchmark_robustness_performance.py", "--formal"]),
    ("exact_joint", [PY, r"scripts\benchmark_exact_joint_scaling.py"]),
    ("powerbit", [PY, r"scripts\benchmark_joint_power_bit_scaling.py", "--reports", "2", "4", "6"]),
    ("wta", [PY, r"scripts\benchmark_winner_take_all_scaling.py", "--reports", "2", "3", "4"]),
]

out_path = sys.argv[1]
results = []
for name, cmd in BENCHMARKS:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - start
    payload = None
    try:
        payload = json.loads(proc.stdout)
    except (ValueError, TypeError):
        payload = {"stderr_tail": (proc.stderr or "")[-500:]}
    results.append({"name": name, "seconds": round(elapsed, 3), "returncode": proc.returncode, "payload": payload})
    print(f"{name}: {elapsed:.3f}s rc={proc.returncode}")

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2, default=str)
print(f"written: {out_path}")
