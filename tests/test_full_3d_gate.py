import numpy as np

from scripts.run_full_3d_gate import (
    cyclic_dd_distance,
    match_full_3d_detections,
)


def test_full_3d_cyclic_dd_distance_wraps_both_axes():
    assert cyclic_dd_distance((0, 0), (3, 7), (4, 8)) == (1, 1)


def test_full_3d_matching_is_one_to_one_and_identity_aware():
    matched, identity, used = match_full_3d_detections(
        [(0, 1, 1, 3), (1, 3, 1, 3)],
        [-5.0, 5.0], [(1, 3), (1, 3)],
        np.array([-10.0, -5.0, 0.0, 5.0, 10.0]), (4, 8), True,
    )
    assert matched == 2
    assert identity == 2
    assert used == {0, 1}
