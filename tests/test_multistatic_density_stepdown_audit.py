import inspect

from scripts.run_multistatic_density_stepdown_audit import run_density_audit


def test_density_audit_has_fixed_activation_count_in_source():
    assert "activation_count=2" in inspect.getsource(run_density_audit)
