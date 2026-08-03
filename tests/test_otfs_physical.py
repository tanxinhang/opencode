import numpy as np

from uav_otfs_isac.otfs_physical import (
    heisenberg,
    apply_delay_doppler_channel,
    cyclic_impulse_pattern,
    cyclic_nms_peaks,
    dd_cross_ambiguity,
    matched_filter_map,
    matched_filter_cell_threshold,
    qpsk_phase_pattern,
    threshold_cyclic_nms_peaks,
    superpose_uav_echoes,
    isfft,
    otfs_demodulate,
    otfs_modulate,
    sfft,
    separable_cazac_pattern,
    cazac_sequence,
    spatial_matched_filter_map,
    spatial_otfs_template,
    ula_steering_vector,
    wigner,
    delay_doppler_path,
)


def test_isfft_sfft_are_unitary_inverses():
    rng = np.random.default_rng(20260803)
    dd = rng.standard_normal((8, 16)) + 1j * rng.standard_normal((8, 16))
    tf = isfft(dd)
    assert np.allclose(sfft(tf), dd, atol=1e-12)
    assert np.isclose(np.vdot(tf, tf).real, np.vdot(dd, dd).real)


def test_isfft_matches_explicit_symplectic_dft_matrices():
    rng = np.random.default_rng(20260807)
    dd = rng.standard_normal((3, 5)) + 1j * rng.standard_normal((3, 5))
    doppler_dft = np.fft.fft(np.eye(3), axis=0, norm="ortho")
    delay_dft = np.fft.fft(np.eye(5), axis=0, norm="ortho")
    expected = doppler_dft.conj().T @ dd @ delay_dft
    assert np.allclose(isfft(dd), expected, atol=1e-12)


def test_heisenberg_wigner_are_unitary_inverses():
    rng = np.random.default_rng(20260804)
    tf = rng.standard_normal((8, 16)) + 1j * rng.standard_normal((8, 16))
    samples = heisenberg(tf)
    assert np.allclose(wigner(samples, 8, 16), tf, atol=1e-12)
    assert np.isclose(np.vdot(samples, samples).real, np.vdot(tf, tf).real)


def test_otfs_modulation_round_trip_and_energy():
    rng = np.random.default_rng(20260805)
    dd = rng.standard_normal((8, 16)) + 1j * rng.standard_normal((8, 16))
    samples = otfs_modulate(dd)
    recovered = otfs_demodulate(samples, 8, 16)
    assert np.allclose(recovered, dd, atol=1e-12)
    assert np.isclose(np.vdot(samples, samples).real, np.vdot(dd, dd).real)


def test_integer_delay_doppler_path_produces_single_dd_peak():
    n_doppler, n_delay = 8, 16
    dd = np.zeros((n_doppler, n_delay), dtype=complex)
    dd[0, 0] = 1.0
    samples = otfs_modulate(dd)
    received = apply_delay_doppler_channel(
        samples, [(1.0, 3.0, 2.0)], n_doppler
    )
    recovered = otfs_demodulate(received, n_doppler, n_delay)
    peak = np.unravel_index(np.argmax(np.abs(recovered)), recovered.shape)
    assert peak == (2, 3)
    assert np.isclose(np.abs(recovered[peak]), 1.0, atol=1e-12)
    assert np.count_nonzero(np.abs(recovered) > 1e-10) == 1


def test_fractional_doppler_causes_leakage_and_preserves_energy():
    n_doppler, n_delay = 16, 32
    dd = np.zeros((n_doppler, n_delay), dtype=complex)
    dd[0, 0] = 1.0
    samples = otfs_modulate(dd)
    integer = otfs_demodulate(
        apply_delay_doppler_channel(samples, [(1.0, 0.0, 2.0)], n_doppler),
        n_doppler, n_delay,
    )
    fractional = otfs_demodulate(
        apply_delay_doppler_channel(samples, [(1.0, 0.0, 2.37)], n_doppler),
        n_doppler, n_delay,
    )
    assert np.count_nonzero(np.abs(integer) > 1e-10) == 1
    assert np.count_nonzero(np.abs(fractional) > 1e-3) > 1
    assert np.max(np.abs(fractional)) < np.max(np.abs(integer))
    assert np.isclose(np.vdot(fractional, fractional).real, 1.0, atol=1e-12)


def test_fractional_path_energy_scales_with_complex_gain():
    dd = cyclic_impulse_pattern(8, 16, 0, 0)
    samples = otfs_modulate(dd)
    gain = 0.3 + 0.4j
    received = apply_delay_doppler_channel(
        samples, [(gain, 1.25, -2.4)], 8
    )
    assert np.isclose(
        np.vdot(received, received).real,
        abs(gain) ** 2 * np.vdot(samples, samples).real,
        atol=1e-12,
    )


def test_shifted_patterns_have_different_fractional_cross_ambiguity():
    first = cyclic_impulse_pattern(8, 16, 0, 0)
    same = cyclic_impulse_pattern(8, 16, 0, 0)
    shifted = cyclic_impulse_pattern(8, 16, 3, 5)
    delays = np.array([-0.4, 0.0, 0.4])
    dopplers = np.array([-0.4, 0.0, 0.4])
    same_ambiguity = np.abs(dd_cross_ambiguity(
        first, same, delays, dopplers
    )) ** 2
    shifted_ambiguity = np.abs(dd_cross_ambiguity(
        first, shifted, delays, dopplers
    )) ** 2
    assert np.max(same_ambiguity) > 0.99
    assert 0.0 < np.max(shifted_ambiguity) < 1e-3
    assert np.max(shifted_ambiguity) < np.max(same_ambiguity) / 100.0


def test_same_pattern_concurrent_echoes_create_indistinguishable_collision():
    rng = np.random.default_rng(20260806)
    pattern = cyclic_impulse_pattern(8, 16, 0, 0)
    received = superpose_uav_echoes(
        [pattern, pattern],
        [[(1.0, 2.0, 1.0)], [(1.0, 2.0, 1.0)]],
        0.0, rng,
    )
    detection = matched_filter_map(received, pattern)
    assert np.unravel_index(np.argmax(detection), detection.shape) == (1, 2)
    assert np.isclose(detection[1, 2], 4.0, atol=1e-12)


def test_qpsk_phase_patterns_are_unit_energy_and_reproducible():
    first = qpsk_phase_pattern(8, 16, 17)
    repeated = qpsk_phase_pattern(8, 16, 17)
    other = qpsk_phase_pattern(8, 16, 18)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert np.isclose(np.vdot(first, first).real, 1.0)
    assert np.allclose(np.abs(first), 1.0 / np.sqrt(8 * 16))


def test_cyclic_nms_extracts_multiple_separated_targets():
    energy = np.zeros((4, 8))
    energy[0, 0] = 4.0
    energy[2, 4] = 3.0
    energy[0, 1] = 2.0
    assert cyclic_nms_peaks(energy, 2, guard_radius=1) == [(0, 0), (2, 4)]


def test_delay_doppler_path_rejects_invalid_physical_parameters():
    samples = np.ones(16, dtype=complex)
    with np.testing.assert_raises_regex(ValueError, "positively divide"):
        delay_doppler_path(samples, 0.0, 0.0, 3)
    with np.testing.assert_raises_regex(ValueError, "finite"):
        delay_doppler_path(samples, np.nan, 0.0, 4)
    with np.testing.assert_raises_regex(ValueError, "finite"):
        delay_doppler_path(samples, 0.0, np.inf, 4)


def test_threshold_nms_does_not_require_target_count():
    energy = np.zeros((4, 8))
    energy[0, 0] = 4.0
    energy[0, 1] = 3.0
    energy[2, 4] = 2.0
    assert threshold_cyclic_nms_peaks(energy, 1.0, 1) == [(0, 0), (2, 4)]
    assert threshold_cyclic_nms_peaks(energy, 5.0, 1) == []


def test_fixed_pfa_threshold_matches_complex_awgn_monte_carlo():
    noise_variance = 0.2
    p_false_alarm = 0.01
    threshold = matched_filter_cell_threshold(noise_variance, p_false_alarm)
    rng = np.random.default_rng(20260808)
    samples = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)
    )
    empirical = np.mean(np.abs(samples) ** 2 >= threshold)
    assert abs(empirical - p_false_alarm) < 0.001


def test_separable_cazac_dd_array_has_zero_cyclic_correlation_sidelobes():
    pattern = separable_cazac_pattern(8, 16, 1, 1)
    correlations = np.empty(pattern.shape)
    for k in range(pattern.shape[0]):
        for l in range(pattern.shape[1]):
            shifted = np.roll(np.roll(pattern, k, axis=0), l, axis=1)
            correlations[k, l] = abs(np.vdot(pattern, shifted)) ** 2
    correlations[0, 0] = 0.0
    assert np.max(correlations) < 1e-20
    assert np.isclose(np.vdot(pattern, pattern).real, 1.0)
    with np.testing.assert_raises_regex(ValueError, "coprime"):
        separable_cazac_pattern(8, 16, 2, 1)


def test_dd_cazac_does_not_imply_zero_otfs_waveform_ambiguity():
    pattern = separable_cazac_pattern(8, 16, 1, 1)
    ambiguity = np.abs(dd_cross_ambiguity(
        pattern, pattern, np.arange(16), np.arange(8)
    )) ** 2
    ambiguity[0, 0] = 0.0
    assert np.max(ambiguity) > 0.01


def test_ula_and_spatial_otfs_template_are_unit_energy():
    pattern = qpsk_phase_pattern(8, 16, 11)
    steering = ula_steering_vector(8, 20.0)
    code = cazac_sequence(8, 1)
    template = spatial_otfs_template(
        pattern, 2.25, 1.4, 20.0, 8, code
    )
    assert np.isclose(np.vdot(steering, steering).real, 1.0)
    assert template.shape == (8, 128)
    assert np.isclose(np.vdot(template, template).real, 1.0)


def test_spatial_cube_resolves_angle_for_one_integer_dd_path():
    pattern = qpsk_phase_pattern(4, 8, 11)
    template = spatial_otfs_template(pattern, 2.0, 1.0, 30.0, 8)
    angles = np.arange(-60.0, 61.0, 5.0)
    cube = spatial_matched_filter_map(template, pattern, angles)
    peak = np.unravel_index(np.argmax(cube), cube.shape)
    assert angles[peak[0]] == 30.0
    assert peak[1:] == (1, 2)
