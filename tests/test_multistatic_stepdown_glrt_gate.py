from scripts.run_multistatic_physics_glrt_gate import (
    fit_ordered_frame_thresholds,
)
from scripts.run_multistatic_stepdown_glrt_gate import rejection_count


def test_ordered_thresholds_use_successive_frame_statistics():
    frames = [[(3.0, 1, 1), (1.0, 1, 1)],
              [(4.0, 1, 1), (2.0, 1, 1)]] * 100
    thresholds = fit_ordered_frame_thresholds(frames, 2, 0.05)
    assert thresholds == (4.0, 2.0)


def test_stepdown_stops_at_first_failed_rank():
    frame = [(10.0, 1, 1), (8.0, 1, 1), (7.0, 1, 1)]
    assert rejection_count(frame, (9.0, 9.0, 1.0)) == 1
    assert rejection_count(frame, (9.0, 7.0, 6.0)) == 3
