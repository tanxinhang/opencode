"""Pytest bootstrap: avoid the torch/MKL duplicate libiomp5md.dll crash.

On Windows with Anaconda, importing torch (used by the MAPPO/PPO modules)
after MKL has initialized raises OMP Error #15 and aborts the interpreter
unless KMP_DUPLICATE_LIB_OK is set.  This must be set before any OpenMP
runtime loads, so it happens at the very top of collection.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
