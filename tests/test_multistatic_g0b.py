import numpy as np

from scripts.run_multistatic_g0b import (
    draw_targets, gospa_distance, imperfect_candidates,
    nested_transmitter_geometry, run_study,
)
from uav_otfs_isac.multistatic_targets import (
    KinematicNode, PhysicalTarget, generate_bistatic_paths,
)


def test_g0b_smoke_is_reproducible_and_unknown_cardinality():
    first = run_study(
        trials=3, seed=19, transmitter_counts=(2,), target_counts=(1, 2)
    )
    second = run_study(
        trials=3, seed=19, transmitter_counts=(2,), target_counts=(1, 2)
    )
    assert first.keys() == second.keys()
    for first_row, second_row in zip(first["rows"], second["rows"]):
        deterministic_keys = set(first_row) - {
            "mean_association_time_ms", "p95_association_time_ms"
        }
        assert {key: first_row[key] for key in deterministic_keys} == {
            key: second_row[key] for key in deterministic_keys
        }
    assert first["receiver"]["known_target_count"] is False
    assert len(first["rows"]) == 2
    for row in first["rows"]:
        assert 0.0 <= row["target_count_accuracy"] <= 1.0
        assert 0.0 <= row["target_recall"] <= 1.0
        assert 0.0 <= row["identity_association_accuracy"] <= 1.0


def test_separated_generator_scales_without_rejection_bias():
    targets = draw_targets(np.random.default_rng(3), 12, "separated")
    angles = np.sort([np.arctan2(
        target.position[1], target.position[0]
    ) for target in targets])
    assert len(targets) == 12
    assert np.min(np.diff(angles)) >= np.deg2rad(6.0) - 1e-12


def test_nested_transmitter_geometry_preserves_smaller_deployment():
    small = nested_transmitter_geometry(4)
    large = nested_transmitter_geometry(12)
    assert all(np.allclose(left.position, right.position)
               for left, right in zip(small, large))


def test_gospa_penalizes_missed_and_false_targets():
    truth = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    exact = gospa_distance(truth, truth)
    missed = gospa_distance(truth, truth[:1])
    false = gospa_distance(truth[:1], truth)
    assert exact == 0.0
    assert missed > 0.0
    assert false > 0.0


def test_overlap_confidence_model_has_nontrivial_score_overlap():
    transmitters = nested_transmitter_geometry(4)
    receiver = KinematicNode((0.0, 0.0), (0.0, 0.0))
    paths = generate_bistatic_paths(
        transmitters,
        (PhysicalTarget(0, (20.0, 180.0), (0.0, 0.0)),),
        receiver, 5.9e9,
    )
    candidates, truth, _ = imperfect_candidates(
        np.random.default_rng(5), paths, 4, miss_probability=0.0,
        false_mean=20.0, clutter_model="correlated_sidelobes",
        confidence_model="overlap",
    )
    true_scores = [c.confidence for c in candidates if truth[id(c)] is not None]
    false_scores = [c.confidence for c in candidates if truth[id(c)] is None]
    assert max(false_scores) > min(true_scores)
