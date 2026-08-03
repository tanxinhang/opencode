from scripts.run_robust_minimal_probe_gate import (
    LAMBDA_THRESHOLD,
    nominal_minimum_length,
    uncertainty_grams_by_length,
)
from uav_otfs_isac.identifiability import minimum_probe_length
from uav_otfs_isac.otfs_physical import (
    otfs_modulate,
    qpsk_phase_pattern,
)


def test_zero_radius_robust_and_nominal_lengths_agree():
    reference = otfs_modulate(qpsk_phase_pattern(8, 16, 11))
    estimate = {
        "angle_gap": 4.0, "delay_gap": 0.1, "doppler_gap": 0.1,
        "phase_step": 0.1, "cfo": 0.01,
    }
    zero = {key: 0.0 for key in ("angle", "delay", "doppler", "phase", "cfo")}
    robust, _ = minimum_probe_length(
        uncertainty_grams_by_length(reference, estimate, zero),
        LAMBDA_THRESHOLD,
    )
    assert robust == nominal_minimum_length(reference, estimate)
