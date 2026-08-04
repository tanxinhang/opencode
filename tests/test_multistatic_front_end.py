import numpy as np

from uav_otfs_isac.front_end import (
    FrontEndConfig,
    calibrate_confidence,
    calibrate_frame_threshold,
    calibrate_sidelobe_aware_threshold,
    detection_cubes,
    extract_peaks,
    identity_patterns,
    integrated_detection_cubes,
    peak_measurement_sigmas,
    peak_estimate,
    peak_matches_path,
    peaks_to_candidates,
    precompute_templates,
    simulate_received,
)
from uav_otfs_isac.multistatic_association import PathCandidate
from uav_otfs_isac.multistatic_model_selection import bic_conflict_association
from uav_otfs_isac.multistatic_targets import (
    KinematicNode,
    PhysicalTarget,
    generate_bistatic_paths,
)


CARRIER = 5.9e9


def geometry(count=4):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return tuple(KinematicNode(
        (100.0 * np.cos(angle), 100.0 * np.sin(angle)), (0.0, 0.0)
    ) for angle in angles)


def single_target_scene():
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    target = PhysicalTarget(0, (55.0, 30.0), (0.5, 0.2))
    paths = generate_bistatic_paths(
        geometry(), [target], receiver, CARRIER
    )
    return receiver, target, paths


def test_peak_extraction_recovers_single_true_path():
    config = FrontEndConfig(noise_variance=0.0)
    patterns = identity_patterns(config, 4)
    receiver, target, paths = single_target_scene()
    rng = np.random.default_rng(7)
    gains = [1.8 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)) for _ in paths]
    received = simulate_received(config, patterns, paths, gains, rng)
    peaks = extract_peaks(
        config, received, patterns, 0.2, dd_guard=1
    )
    assert any(
        peak_matches_path(config, peak, path, angle_tolerance_degrees=6.0,
                          range_tolerance_m=15.0,
                          doppler_tolerance_hz=25.0)
        for peak in peaks for path in paths
    )


def test_peak_estimate_maps_bins_back_to_si_units():
    config = FrontEndConfig()
    receiver, _, paths = single_target_scene()
    path = paths[0]
    delay_bin = config.delay_bin(path.delay_s)
    doppler_bin = config.wrapped_doppler_bin(path.doppler_hz)
    delay_s, doppler_hz = config.estimate_from_bins(delay_bin, doppler_bin)
    assert abs(delay_s - path.delay_s) < 2.0 * config.delay_resolution_m / (
        299_792_458.0
    )
    assert abs(doppler_hz - path.doppler_hz) <= config.doppler_resolution_hz


def test_cazac_identity_patterns_are_unit_energy():
    config = FrontEndConfig()
    patterns = identity_patterns(config, 4, kind="cazac")
    assert all(
        abs(np.linalg.norm(pattern) - 1.0) < 1e-12 for pattern in patterns
    )


def test_integrated_cubes_match_single_frame_detection():
    config = FrontEndConfig(noise_variance=0.0)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    receiver, _, paths = single_target_scene()
    rng = np.random.default_rng(3)
    gains = [1.8 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)) for _ in paths]
    received = simulate_received(config, patterns, paths, gains, rng)
    single = detection_cubes(config, received, patterns, templates=templates)
    integrated = integrated_detection_cubes(
        config, received, patterns, templates=templates
    )
    assert all(
        np.allclose(left, right)
        for left, right in zip(single, integrated)
    )


def test_sidelobe_aware_threshold_exceeds_noise_threshold():
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    receiver, _, paths = single_target_scene()
    rng = np.random.default_rng(5)
    frames = []
    for _ in range(6):
        gains = [2.0 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
                 for _ in paths]
        received = simulate_received(config, patterns, paths, gains, rng)
        frames.append((received, paths))
    noise = calibrate_frame_threshold(
        config, patterns, trials=100,
        frame_false_alarm_probability=0.05,
        templates=templates, batch_size=50, seed=7,
    )
    sidelobe = calibrate_sidelobe_aware_threshold(
        config, patterns, frames,
        templates=templates,
        frame_false_alarm_probability=0.05,
    )
    assert sidelobe > noise


def test_confidence_calibration_orders_true_above_noise_peaks():
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    receiver, _, paths = single_target_scene()
    rng = np.random.default_rng(11)
    frames = []
    for _ in range(8):
        gains = [2.0 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
                 for _ in paths]
        received = simulate_received(config, patterns, paths, gains, rng)
        frames.append((received, paths))
    for _ in range(8):
        frames.append((
            simulate_received(config, patterns, (), (), rng), None,
        ))
    calibrator = calibrate_confidence(
        config, patterns, frames, collect_threshold=0.1,
        templates=templates,
    )
    true_scores = []
    noise_scores = []
    for received, true_paths in frames:
        for transmitter_id, cube in enumerate(
            detection_cubes(config, received, patterns, templates=templates)
        ):
            score = float(np.max(cube))
            if true_paths is None:
                noise_scores.append(score)
            else:
                true_scores.append(score)
    true_probability = float(np.mean([
        calibrator(score) for score in true_scores
    ]))
    noise_probability = float(np.mean([
        calibrator(score) for score in noise_scores
    ]))
    assert true_probability > noise_probability


def test_end_to_end_single_target_association_recovers_position():
    config = FrontEndConfig(noise_variance=0.0)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    receiver, target, paths = single_target_scene()
    rng = np.random.default_rng(13)
    gains = [1.8 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)) for _ in paths]
    received = simulate_received(config, patterns, paths, gains, rng)
    peaks = extract_peaks(config, received, patterns, 0.2, templates=templates)
    cubes = detection_cubes(config, received, patterns, templates=templates)

    class FixedCalibrator:
        def __call__(self, score):
            return float(np.clip(score / 3.0, 1e-4, 1.0 - 1e-4))

    candidates = peaks_to_candidates(config, peaks, cubes, FixedCalibrator())
    groups = bic_conflict_association(
        candidates, geometry(), receiver, CARRIER,
        position_tolerance_m=12.0,
        position_sigma_m=6.0,
        doppler_sigma_hz=15.0,
        clutter_doppler_span_hz=160.0,
        view_false_target_probability=0.1,
        order_confidence_threshold=0.4,
        maximum_local_targets=2,
        final_joint_refinement_iterations=0,
    )
    assert len(groups) == 1
    assert np.linalg.norm(groups[0].position - target.position) <= 12.0


def test_explicit_support_overrides_are_accepted_by_association():
    config = FrontEndConfig(noise_variance=0.0)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    receiver, target, paths = single_target_scene()
    rng = np.random.default_rng(17)
    gains = [1.8 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)) for _ in paths]
    received = simulate_received(config, patterns, paths, gains, rng)
    peaks = extract_peaks(config, received, patterns, 0.2, templates=templates)
    cubes = detection_cubes(config, received, patterns, templates=templates)

    class FixedCalibrator:
        def __call__(self, score):
            return float(np.clip(score / 3.0, 1e-4, 1.0 - 1e-4))

    candidates = peaks_to_candidates(config, peaks, cubes, FixedCalibrator())
    groups = bic_conflict_association(
        candidates, geometry(), receiver, CARRIER,
        position_tolerance_m=12.0,
        position_sigma_m=6.0,
        doppler_sigma_hz=15.0,
        clutter_doppler_span_hz=160.0,
        required_target_support_override=2,
        required_collision_support_override=2,
        order_confidence_threshold=0.4,
        maximum_local_targets=2,
        final_joint_refinement_iterations=0,
    )
    assert len(groups) >= 1


def test_calibrated_frame_threshold_limits_noise_false_alarms():
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 2)
    threshold = calibrate_frame_threshold(
        config, patterns, trials=200,
        frame_false_alarm_probability=0.05,
        batch_size=100, seed=19,
    )
    rng = np.random.default_rng(23)
    alarms = 0
    for _ in range(100):
        received = simulate_received(config, patterns, (), (), rng)
        peaks = extract_peaks(config, received, patterns, threshold)
        alarms += int(len(peaks) > 0)
    assert alarms / 100.0 <= 0.2


def test_peak_measurement_sigmas_reflect_local_curvature():
    config = FrontEndConfig()
    shape = (config.angle_grid_degrees.size, config.doppler_bins,
             config.delay_bins)
    center = (config.angle_grid_degrees.size // 2,
              config.doppler_bins // 2, config.delay_bins // 2)
    broad = np.zeros(shape)
    sharp = np.zeros(shape)
    for a in range(shape[0]):
        for k in range(shape[1]):
            for l in range(shape[2]):
                broad[a, k, l] = np.exp(-(
                    (a - center[0]) ** 2 / 8.0
                    + (k - center[1]) ** 2 / 4.0
                    + (l - center[2]) ** 2 / 4.0
                ))
                sharp[a, k, l] = np.exp(-(
                    (a - center[0]) ** 2 / 2.0
                    + (k - center[1]) ** 2
                    + (l - center[2]) ** 2
                ))
    broad_sigmas = peak_measurement_sigmas(config, broad, center)
    sharp_sigmas = peak_measurement_sigmas(config, sharp, center)
    assert all(0.0 < value < 50.0 for value in broad_sigmas)
    assert all(0.0 < value < 50.0 for value in sharp_sigmas)
    assert sharp_sigmas[0] < broad_sigmas[0]
    assert sharp_sigmas[1] < broad_sigmas[1]
    assert sharp_sigmas[2] < broad_sigmas[2]


def test_path_candidate_accepts_front_end_precision_fields():
    candidate = PathCandidate(
        0, 1.0e-6, 20.0, 0.1, 0.9,
        range_sigma_m=2.0,
        angle_sigma_rad=0.01,
        doppler_sigma_hz=4.0,
    )
    assert candidate.range_sigma_m == 2.0
    assert candidate.doppler_sigma_hz == 4.0
    with np.testing.assert_raises_regex(ValueError, "positive and finite"):
        PathCandidate(0, 1.0e-6, 20.0, 0.1, 0.9, doppler_sigma_hz=-1.0)
