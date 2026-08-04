from scripts.run_multistatic_calibrated_gate import run_calibrated_gate


def test_calibrated_gate_separates_calibration_validation_and_evaluation():
    result = run_calibrated_gate(
        calibration_scenes=8, validation_scenes=4, evaluation_trials=2,
        rank_calibration_scenes=8, null_calibration_scenes=10,
        calibration_seed=11, rank_calibration_seed=14,
        null_calibration_seed=15, validation_seed=12, evaluation_seed=13,
        transmitters=4,
    )
    assert len(set(result["seeds"].values())) == 5
    assert result["sample_sizes"]["calibration_candidates"] > 0
    assert result["sample_sizes"]["validation_candidates"] > 0
    assert "paired_collision" in result
