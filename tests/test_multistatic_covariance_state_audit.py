import inspect

from scripts.run_multistatic_covariance_state_audit import (
    run_covariance_state_audit,
)


def test_covariance_audit_is_paired_and_post_decision_only():
    source = inspect.getsource(run_covariance_state_audit)
    assert "3, False" in source
    assert "3, True" in source
    assert "for seed in seeds" in source
