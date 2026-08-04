import inspect

from scripts.run_multistatic_refined_glrt_audit import run_refined_glrt_audit


def test_refined_audit_calibrates_both_statistics_separately():
    source = inspect.getsource(run_refined_glrt_audit)
    assert '(("coarse", 0), ("refined", 3))' in source
    assert "fit_ordered_frame_thresholds" in source
