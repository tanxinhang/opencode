import inspect

from scripts.run_multistatic_cascade_glrt_gate import run_cascade_gate


def test_cascade_uses_independent_two_pair_normal_maximum():
    source = inspect.getsource(run_cascade_gate)
    assert '"two_pair_collision", 8, 3' in source
    assert "normal_component_maxima" in source
    assert "finite_sample_upper_threshold" in source
    assert '"single_pair_collision", 8, 0' in source
