import numpy as np
import pytest

from uav_otfs_isac.multistatic_targets import (
    KinematicNode,
    PhysicalTarget,
    bistatic_delay,
    bistatic_doppler,
    generate_bistatic_paths,
    group_paths_by_target,
)


def node(position, velocity=(0.0, 0.0)):
    return KinematicNode(position=np.asarray(position), velocity=np.asarray(velocity))


def test_general_scene_has_mn_paths_and_m_paths_per_target():
    transmitters = [node((-100.0, 0.0)), node((0.0, 100.0)), node((100.0, 0.0))]
    receiver = node((0.0, 0.0))
    targets = [
        PhysicalTarget(7, (20.0, 30.0), (2.0, -1.0)),
        PhysicalTarget(11, (-40.0, 50.0), (0.0, 3.0)),
    ]
    paths = generate_bistatic_paths(transmitters, targets, receiver, 5.9e9)
    groups = group_paths_by_target(paths)

    assert len(paths) == 3 * 2
    assert set(groups) == {7, 11}
    assert all(len(group) == 3 for group in groups.values())
    assert {path.transmitter_id for path in groups[7]} == {0, 1, 2}


def test_same_target_paths_share_receive_angle_but_follow_bistatic_ranges():
    transmitters = [node((-30.0, 0.0)), node((50.0, -20.0))]
    receiver = node((0.0, 0.0))
    target = PhysicalTarget(0, (30.0, 40.0), (0.0, 0.0))
    paths = generate_bistatic_paths(transmitters, [target], receiver, 3.5e9)

    assert np.isclose(paths[0].receive_azimuth_rad, np.arctan2(40.0, 30.0))
    assert np.isclose(paths[0].receive_azimuth_rad, paths[1].receive_azimuth_rad)
    for transmitter, path in zip(transmitters, paths):
        expected = (
            np.linalg.norm(target.position - transmitter.position)
            + np.linalg.norm(target.position - receiver.position)
        )
        assert np.isclose(path.bistatic_range_m, expected)
        assert np.isclose(path.delay_s, bistatic_delay(
            transmitter.position, target.position, receiver.position
        ))


def test_doppler_uses_bistatic_range_rate_and_static_scene_is_zero():
    transmitter = node((-100.0, 0.0))
    receiver = node((100.0, 0.0))
    static = PhysicalTarget(0, (0.0, 100.0), (0.0, 0.0))
    moving = PhysicalTarget(1, (0.0, 100.0), (0.0, 10.0))
    carrier = 6.0e9
    speed = 3.0e8

    assert bistatic_doppler(transmitter, static, receiver, carrier, speed) == 0.0
    expected_rate = 2.0 * 10.0 / np.sqrt(2.0)
    assert np.isclose(
        bistatic_doppler(transmitter, moving, receiver, carrier, speed),
        carrier * expected_rate / speed,
    )


def test_visibility_supports_fewer_than_mn_paths():
    transmitters = [node((-10.0, 0.0)), node((10.0, 0.0))]
    targets = [
        PhysicalTarget(0, (0.0, 10.0), (0.0, 0.0)),
        PhysicalTarget(1, (0.0, 20.0), (0.0, 0.0)),
    ]
    visibility = np.array([[True, False], [True, True]])
    paths = generate_bistatic_paths(
        transmitters, targets, node((0.0, 0.0)), 5.0e9, visibility
    )

    assert [(path.transmitter_id, path.target_id) for path in paths] == [
        (0, 0), (1, 0), (1, 1)
    ]


def test_invalid_geometry_and_duplicate_target_ids_are_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        generate_bistatic_paths(
            [node((0.0, 0.0))],
            [PhysicalTarget(0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0))],
            node((2.0, 0.0)),
            5.0e9,
        )
    with pytest.raises(ValueError, match="unique"):
        generate_bistatic_paths(
            [node((0.0, 0.0))],
            [
                PhysicalTarget(2, (1.0, 1.0), (0.0, 0.0)),
                PhysicalTarget(2, (2.0, 2.0), (0.0, 0.0)),
            ],
            node((3.0, 0.0)),
            5.0e9,
        )
