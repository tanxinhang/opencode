from scripts.run_multistatic_physics_glrt_gate import run_physics_gate


def test_physics_gate_uses_disjoint_partitions():
    result = run_physics_gate(
        calibration_scenes=8, null_scenes=8, validation_scenes=4,
        evaluation_trials=2, transmitters=4,
        calibration_seed=21, null_seed=22,
        validation_seed=23, evaluation_seed=24,
    )
    assert len(set(result["seeds"].values())) == 4
    assert result["gate"]["frame_false_trigger_target"] == 0.01
