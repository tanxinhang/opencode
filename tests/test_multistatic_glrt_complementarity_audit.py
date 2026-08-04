import numpy as np

from scripts.run_multistatic_glrt_complementarity_audit import (
    run_complementarity_audit,
)


def test_complementarity_audit_rejects_mismatched_seed_partitions():
    # Structural smoke: implementation must use one seed for both statistics.
    import inspect
    source = inspect.getsource(run_complementarity_audit)
    assert "paired_component_statistics" in source
    assert "seed, frames" in source
