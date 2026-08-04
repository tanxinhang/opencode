import numpy as np

from uav_otfs_isac.multistatic_association import PathCandidate
from uav_otfs_isac.multistatic_model_selection import (
    _clutter_baseline_assignment,
    _initial_center_score,
    _initial_center_profile,
    _screen_distinct_initializations,
    bic_conflict_association,
    collision_support_threshold,
    poisson_binomial_tail,
)
from uav_otfs_isac.multistatic_targets import KinematicNode, PhysicalTarget, generate_bistatic_paths


def node(position):
    return KinematicNode(position, (0.0, 0.0))


def candidates(paths):
    return [PathCandidate(
        path.transmitter_id, path.delay_s, path.doppler_hz,
        path.receive_azimuth_rad, 0.9,
    ) for path in paths]


def geometry():
    transmitters = tuple(node((300 * np.cos(angle), 300 * np.sin(angle)))
                         for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False))
    return transmitters, node((0.0, 0.0))


def test_bic_rejects_same_transmitter_local_sidelobes_as_clutter():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (2.0, -1.0))
    true_candidates = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))
    augmented = list(true_candidates)
    for candidate in true_candidates[:4]:
        augmented.append(PathCandidate(
            candidate.transmitter_id,
            candidate.delay_s + 4.0 / 299_792_458.0,
            candidate.doppler_hz + 12.0,
            candidate.receive_azimuth_rad + np.deg2rad(0.5),
            0.2,
        ))
    groups = bic_conflict_association(
        augmented, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0,
    )
    assert len(groups) == 1
    assert len(groups[0].paths) == 8


def test_bic_selects_two_close_targets_with_cross_uav_support():
    transmitters, receiver = geometry()
    targets = (
        PhysicalTarget(0, (-3.0, 180.0), (3.0, -1.0)),
        PhysicalTarget(1, (3.0, 180.0), (-2.0, 2.0)),
    )
    groups = bic_conflict_association(
        candidates(generate_bistatic_paths(
            transmitters, targets, receiver, 5.9e9
        )),
        transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0,
    )
    assert len(groups) == 2
    assert all(len(group.paths) == 8 for group in groups)


def test_rank_deficient_velocity_geometry_is_not_reported_as_observable_target():
    receiver = node((0.0, 0.0))
    transmitters = (
        node((-300.0, 0.0)),
        node((-200.0, 0.0)),
    )
    target = PhysicalTarget(0, (100.0, 0.0), (3.0, 2.0))
    groups = bic_conflict_association(
        candidates(generate_bistatic_paths(
            transmitters, [target], receiver, 5.9e9
        )),
        transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0,
    )
    assert groups == ()


def test_single_order_fast_path_keeps_only_strongest_candidate_per_uav():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (40.0, 170.0), (1.0, -2.0))
    true_candidates = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))
    augmented = list(true_candidates)
    candidate = true_candidates[0]
    augmented.append(PathCandidate(
        candidate.transmitter_id,
        candidate.delay_s + 3.0 / 299_792_458.0,
        candidate.doppler_hz + 10.0,
        candidate.receive_azimuth_rad + np.deg2rad(0.3),
        0.2,
    ))
    groups = bic_conflict_association(
        augmented, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0,
    )
    assert len(groups) == 1
    assert len(groups[0].paths) == 8


def test_calibrated_probability_has_monotone_target_vs_clutter_log_odds():
    probabilities = np.array([0.2, 0.5, 0.9])
    target_cost = -2.0 * np.log(probabilities)
    clutter_cost = -2.0 * np.log1p(-probabilities)
    relative_target_cost = target_cost - clutter_cost
    assert np.all(np.diff(relative_target_cost) < 0.0)


def test_poisson_binomial_tail_matches_exact_enumeration():
    probabilities = (0.1, 0.2, 0.35, 0.05)
    exact = 0.0
    for mask in range(1 << len(probabilities)):
        successes = sum((mask >> index) & 1 for index in range(len(probabilities)))
        probability = np.prod([
            value if (mask >> index) & 1 else 1.0 - value
            for index, value in enumerate(probabilities)
        ])
        exact += probability * (successes >= 2)
    assert np.isclose(poisson_binomial_tail(probabilities, 2), exact)


def test_poisson_binomial_tail_is_monotone_in_required_support():
    tails = [poisson_binomial_tail((0.1,) * 8, support)
             for support in range(1, 9)]
    assert np.all(np.diff(tails) <= 0.0)
    assert collision_support_threshold((0.1,) * 8, 0.05) == 3


def test_collision_threshold_is_defined_for_m4_m6_m8():
    assert [collision_support_threshold((0.1,) * count, 0.05)
            for count in (4, 6, 8)] == [3, 3, 3]


def test_target_support_threshold_adapts_to_m_and_null_probability():
    transmitter_counts = (4, 6, 8, 10, 12)
    assert [collision_support_threshold((0.02,) * count, 0.05)
            for count in transmitter_counts] == [2, 2, 2, 2, 2]
    assert [collision_support_threshold((0.05,) * count, 0.05)
            for count in transmitter_counts] == [2, 2, 3, 3, 3]
    assert [collision_support_threshold((0.10,) * count, 0.05)
            for count in transmitter_counts] == [3, 3, 3, 4, 4]


def test_collision_threshold_respects_calibration_and_false_alarm_level():
    assert collision_support_threshold((0.05,) * 8, 0.05) <= (
        collision_support_threshold((0.2,) * 8, 0.05)
    )
    assert collision_support_threshold((0.1,) * 8, 0.01) >= (
        collision_support_threshold((0.1,) * 8, 0.1)
    )


def test_identical_view_probabilities_match_binomial_tail():
    from scipy.stats import binom

    assert np.isclose(
        poisson_binomial_tail((0.12,) * 6, 3), binom.sf(2, 6, 0.12)
    )


def test_fixed_support_is_less_significant_with_more_false_extra_peaks():
    assert poisson_binomial_tail((0.05,) * 6, 3) < (
        poisson_binomial_tail((0.2,) * 6, 3)
    )


def test_collision_is_not_confirmed_when_null_always_produces_extra_peak():
    assert collision_support_threshold((1.0,) * 4, 0.05) is None


def test_view_probability_vector_must_match_transmitter_count():
    transmitters, receiver = geometry()
    with np.testing.assert_raises(ValueError):
        bic_conflict_association(
            [], transmitters, receiver, 5.9e9,
            position_tolerance_m=14.0,
            view_false_extra_probability=(0.1, 0.1),
        )
    with np.testing.assert_raises(ValueError):
        bic_conflict_association(
            [], transmitters, receiver, 5.9e9,
            position_tolerance_m=14.0,
            view_false_target_probability=(0.1, 0.1),
        )


def test_target_existence_gate_rejects_two_view_false_fragment_at_large_m():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (2.0, -1.0))
    sparse = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))[:2]
    groups = bic_conflict_association(
        sparse, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0,
    )
    assert groups == ()


def test_target_existence_gate_keeps_cross_view_supported_target():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (2.0, -1.0))
    supported = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))[:3]
    groups = bic_conflict_association(
        supported, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0,
    )
    assert len(groups) == 1


def test_subset_assignment_matches_exhaustive_partial_matching():
    import itertools

    rng = np.random.default_rng(17)
    for candidate_count in range(1, 6):
        for target_count in range(1, 4):
            target_costs = rng.uniform(-1.0, 6.0, (candidate_count, target_count))
            clutter_costs = rng.uniform(0.0, 4.0, candidate_count)
            assignment, cost = _clutter_baseline_assignment(
                target_costs, clutter_costs
            )
            exhaustive = np.inf
            for choices in itertools.product(
                range(-1, target_count), repeat=candidate_count
            ):
                occupied = [choice for choice in choices if choice >= 0]
                if len(occupied) != len(set(occupied)):
                    continue
                value = sum(
                    clutter_costs[index] if choice < 0
                    else target_costs[index, choice]
                    for index, choice in enumerate(choices)
                )
                exhaustive = min(exhaustive, value)
            assert np.isclose(cost, exhaustive)
            assert len(set(assignment[assignment >= 0])) == np.sum(assignment >= 0)


def test_subset_assignment_never_uses_costlier_target_than_clutter():
    assignment, cost = _clutter_baseline_assignment(
        np.asarray([[3.0, 4.0], [5.0, 6.0]]), np.asarray([1.0, 2.0])
    )
    assert np.array_equal(assignment, (-1, -1))
    assert cost == 3.0


def test_profile_start_score_prefers_centers_near_supported_paths():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (0.0, 0.0))
    paths = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))
    from uav_otfs_isac.multistatic_baselines import _project

    projected = _project(paths, transmitters, receiver, 299_792_458.0)
    good = np.asarray([[20.0, 180.0]])
    bad = np.asarray([[80.0, 100.0]])
    assert _initial_center_score(projected, good, 3.0, 5.0) < (
        _initial_center_score(projected, bad, 3.0, 5.0)
    )


def test_profile_signature_is_invariant_to_target_label_permutation():
    transmitters, receiver = geometry()
    targets = (
        PhysicalTarget(0, (-3.0, 180.0), (0.0, 0.0)),
        PhysicalTarget(1, (3.0, 180.0), (0.0, 0.0)),
    )
    from uav_otfs_isac.multistatic_baselines import _project

    projected = _project(candidates(generate_bistatic_paths(
        transmitters, targets, receiver, 5.9e9
    )), transmitters, receiver, 299_792_458.0)
    centers = np.asarray([target.position for target in targets])
    forward = _initial_center_profile(projected, centers, 3.0, 5.0)
    reverse = _initial_center_profile(projected, centers[::-1], 3.0, 5.0)
    assert np.isclose(forward[0], reverse[0])
    assert forward[1] == reverse[1]
    screened = _screen_distinct_initializations(
        projected, (centers, centers[::-1]), 3.0, 5.0
    )
    assert len(screened) == 1


def test_posterior_collision_gate_uses_subthreshold_calibrated_evidence():
    transmitters, receiver = geometry()
    targets = (
        PhysicalTarget(0, (-3.0, 180.0), (3.0, -1.0)),
        PhysicalTarget(1, (3.0, 180.0), (-2.0, 2.0)),
    )
    paths = [PathCandidate(
        path.transmitter_id, path.delay_s, path.doppler_hz,
        path.receive_azimuth_rad, 0.65,
    ) for path in generate_bistatic_paths(
        transmitters, targets, receiver, 5.9e9
    )]
    hard = bic_conflict_association(
        paths, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0, collision_gate_mode="hard_null",
        order_confidence_threshold=0.7,
    )
    posterior = bic_conflict_association(
        paths, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0, collision_gate_mode="posterior_support",
        order_confidence_threshold=0.7,
    )
    assert len(hard) == 1
    assert len(posterior) == 2


def test_infeasible_high_order_returns_without_empty_state_failure():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (0.0, 0.0))
    paths = [PathCandidate(
        path.transmitter_id, path.delay_s, path.doppler_hz,
        path.receive_azimuth_rad, 0.99,
    ) for path in generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    )]
    groups = bic_conflict_association(
        paths, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0, collision_gate_mode="posterior_support",
    )
    assert len(groups) == 1


def test_robust_velocity_refinement_limits_one_doppler_outlier():
    from uav_otfs_isac.multistatic_association import _fit_group
    from uav_otfs_isac.multistatic_baselines import _project
    from uav_otfs_isac.multistatic_model_selection import _robust_velocity_refinement

    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (4.0, -2.0))
    paths = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))
    contaminated = list(paths)
    bad = contaminated[-1]
    contaminated[-1] = PathCandidate(
        bad.transmitter_id, bad.delay_s, bad.doppler_hz + 40.0,
        bad.receive_azimuth_rad, bad.confidence,
    )
    projected = _project(contaminated, transmitters, receiver, 299_792_458.0)
    ordinary = _fit_group(
        tuple(item[0] for item in projected), tuple(item[1] for item in projected),
        transmitters, receiver, 5.9e9, 299_792_458.0,
    )
    robust = _robust_velocity_refinement(
        ordinary, transmitters, receiver, 5.9e9, 299_792_458.0, 3.0,
    )
    assert np.linalg.norm(robust.velocity - target.velocity) < np.linalg.norm(
        ordinary.velocity - target.velocity
    )


def test_stepdown_activation_count_must_be_positive():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (0.0, 0.0))
    paths = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))
    import pytest
    with pytest.raises(ValueError, match="activation count"):
        bic_conflict_association(
            paths, transmitters, receiver, 5.9e9,
            position_tolerance_m=14.0,
            collision_gate_mode="physics_stepdown",
            physics_stepdown_thresholds=(1.0,),
            physics_stepdown_activation_count=0,
        )


def test_physics_order_evidence_returns_detecting_two_target_model():
    from uav_otfs_isac.multistatic_baselines import _project
    from uav_otfs_isac.multistatic_model_selection import physics_order_evidence

    transmitters, receiver = geometry()
    targets = (
        PhysicalTarget(0, (-3.0, 180.0), (3.0, -1.0)),
        PhysicalTarget(1, (3.0, 180.0), (-2.0, 2.0)),
    )
    projected = _project(candidates(generate_bistatic_paths(
        transmitters, targets, receiver, 5.9e9
    )), transmitters, receiver, 299_792_458.0)
    gain, model = physics_order_evidence(
        projected, transmitters, receiver, 5.9e9, 3.0, 3.0, 10.0,
        2, 100.0, 20, 299_792_458.0,
    )
    assert np.isfinite(gain)
    assert model is not None
    assert len(model.groups) == 2


def test_physics_order_evidence_rejects_negative_refinement_count():
    import pytest
    from uav_otfs_isac.multistatic_model_selection import physics_order_evidence
    with pytest.raises(ValueError, match="cannot be negative"):
        physics_order_evidence(
            [], (), KinematicNode((0.0, 0.0), (0.0, 0.0)), 5.9e9,
            3.0, 3.0, 10.0, 2, 100.0, 20, 299_792_458.0, -1,
        )


def test_cascade_requires_three_thresholds():
    import pytest
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (20.0, 180.0), (0.0, 0.0))
    paths = candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    ))
    with pytest.raises(ValueError, match="three cascade thresholds"):
        bic_conflict_association(
            paths, transmitters, receiver, 5.9e9,
            position_tolerance_m=14.0, collision_gate_mode="physics_cascade",
            physics_cascade_thresholds=(1.0, 2.0),
        )


def test_final_refinement_iterations_cannot_be_negative():
    import pytest
    transmitters, receiver = geometry()
    with pytest.raises(ValueError, match="final joint refinement"):
        bic_conflict_association(
            [], transmitters, receiver, 5.9e9,
            position_tolerance_m=14.0,
            final_joint_refinement_iterations=-1,
        )


def test_covariance_weighted_state_preserves_target_cardinality():
    transmitters, receiver = geometry()
    targets = (
        PhysicalTarget(0, (-3.0, 180.0), (3.0, -1.0)),
        PhysicalTarget(1, (3.0, 180.0), (-2.0, 2.0)),
    )
    paths = candidates(generate_bistatic_paths(
        transmitters, targets, receiver, 5.9e9
    ))
    ordinary = bic_conflict_association(
        paths, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0, collision_gate_mode="posterior_support",
    )
    weighted = bic_conflict_association(
        paths, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0, collision_gate_mode="posterior_support",
        covariance_weighted_final_state=True,
    )
    assert len(weighted) == len(ordinary)


def test_covariance_weighted_state_is_applied_to_single_target_branch():
    transmitters, receiver = geometry()
    target = PhysicalTarget(0, (35.0, 185.0), (2.0, -1.0))
    clean = list(candidates(generate_bistatic_paths(
        transmitters, [target], receiver, 5.9e9
    )))
    # Deterministic unequal range perturbations make equal and geometry-aware
    # weighting distinct without changing the accepted path set.
    perturbed = [PathCandidate(
        path.transmitter_id,
        path.delay_s + (index - 1.5) * 0.8 / 299_792_458.0,
        path.doppler_hz,
        path.receive_azimuth_rad,
        path.confidence,
    ) for index, path in enumerate(clean)]
    ordinary = bic_conflict_association(
        perturbed, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0, collision_gate_mode="posterior_support",
    )
    weighted = bic_conflict_association(
        perturbed, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0, collision_gate_mode="posterior_support",
        covariance_weighted_final_state=True,
    )
    assert len(ordinary) == len(weighted) == 1
    assert not np.allclose(ordinary[0].position, weighted[0].position)
