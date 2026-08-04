import inspect

from scripts.run_multistatic_robust_velocity_audit import _evaluate


def test_robust_velocity_audit_exposes_explicit_toggle():
    assert "robust" in inspect.signature(_evaluate).parameters
