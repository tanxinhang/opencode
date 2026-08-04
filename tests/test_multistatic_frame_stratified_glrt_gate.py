import numpy as np

from scripts.run_multistatic_frame_stratified_glrt_gate import (
    run_frame_stratified_gate,
)
from scripts.run_multistatic_physics_glrt_gate import (
    finite_sample_upper_threshold, fit_frame_stratified_thresholds,
)


def test_finite_sample_threshold_refuses_unsupported_one_percent_resolution():
    assert np.isinf(finite_sample_upper_threshold(np.arange(98.0), 0.01))
    assert finite_sample_upper_threshold(np.arange(99.0), 0.01) == 98.0


def test_frame_thresholds_use_maximum_and_predecision_excess_stratum():
    frames = [[(1.0, 2, 2), (3.0, 4, 2)], [(2.0, 3, 3)]] * 100
    thresholds, counts = fit_frame_stratified_thresholds(frames, 0.05)
    assert counts == {0: 100, 1: 100}
    assert thresholds == {0: 2.0, 1: 3.0}


def test_frame_stratified_gate_uses_disjoint_partitions():
    result = run_frame_stratified_gate(
        probability_scenes=8, null_frames=8, validation_frames=4,
        evaluation_trials=2, transmitters=4,
        probability_seed=41, null_seed=42,
        validation_seed=43, evaluation_seed=44,
    )
    assert len(set(result["seeds"].values())) == 4
    assert result["gate"]["frame_false_trigger_target"] == 0.01
