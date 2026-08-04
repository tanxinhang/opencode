import inspect

from scripts.run_multistatic_cascade_multiseed_audit import run_multiseed_audit


def test_multiseed_audit_does_not_refit_thresholds():
    source = inspect.getsource(run_multiseed_audit)
    assert 'gate_payload["cascade_thresholds"]' in source
    assert "finite_sample_upper_threshold" not in source
