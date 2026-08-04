import numpy as np

from scripts.run_multistatic_g0b import draw_targets


def test_single_pair_collision_has_one_close_pair_and_six_targets():
    targets = draw_targets(np.random.default_rng(1), 6, "single_pair_collision")
    assert len(targets) == 6
    positions = np.asarray([target.position for target in targets])
    assert np.linalg.norm(positions[0] - positions[1]) < 14.0
    assert np.all(np.linalg.norm(positions[2:] - positions[0], axis=1) > 14.0)


def test_two_pair_collision_has_two_close_pairs():
    targets = draw_targets(np.random.default_rng(2), 6, "two_pair_collision")
    positions = np.asarray([target.position for target in targets])
    assert np.linalg.norm(positions[0] - positions[1]) < 14.0
    assert np.linalg.norm(positions[2] - positions[3]) < 14.0
