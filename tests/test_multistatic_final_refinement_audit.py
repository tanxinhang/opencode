import inspect

from scripts.run_multistatic_final_refinement_audit import (
    run_final_refinement_audit,
)


def test_final_refinement_audit_uses_fixed_limits():
    assert "for limit in (3, 10)" in inspect.getsource(
        run_final_refinement_audit
    )
