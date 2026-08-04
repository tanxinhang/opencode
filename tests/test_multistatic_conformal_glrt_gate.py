from scripts.run_multistatic_conformal_glrt_gate import run_conformal_gate


def test_conformal_gate_uses_five_disjoint_partitions():
    result = run_conformal_gate(
        probability_scenes=8, component_null_frames=8,
        frame_threshold_frames=8, validation_frames=4,
        evaluation_trials=2, transmitters=4,
        probability_seed=31, component_seed=32, threshold_seed=33,
        validation_seed=34, evaluation_seed=35,
    )
    assert len(set(result["seeds"].values())) == 5
    assert result["gate"]["frame_false_trigger_target"] == 0.01
