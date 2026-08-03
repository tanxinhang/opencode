import numpy as np

from uav_otfs_isac.otfs_physical import qpsk_phase_pattern, spatial_otfs_template
from uav_otfs_isac.spatial_detection import (
    decode_probe_code,
    probe_coded_spatial_template,
    separable_detection_cube,
    spatial_dictionary,
    threshold_nms_3d,
    waveform_dictionary,
)


def test_separable_cube_finds_known_angle_and_dd_cell():
    pattern = qpsk_phase_pattern(4, 8, 11)
    received = spatial_otfs_template(pattern, 2.0, 1.0, 20.0, 8)
    angles = np.arange(-40.0, 41.0, 5.0)
    cube = separable_detection_cube(
        received, waveform_dictionary(pattern),
        spatial_dictionary(angles, 8),
    )
    peak = np.unravel_index(np.argmax(cube), cube.shape)
    assert angles[peak[0]] == 20.0
    assert peak[1] == 1 * 8 + 2


def test_threshold_nms_3d_uses_cyclic_dd_and_noncyclic_angle_guards():
    cube = np.zeros((3, 4 * 8))
    cube[0, 0] = 5.0
    cube[0, 1] = 4.0
    cube[2, 2 * 8 + 4] = 3.0
    peaks = threshold_nms_3d(cube, 1.0, (4, 8), 1, 1)
    assert peaks == [(0, 0, 0), (2, 2, 4)]


def test_probe_code_round_trip_preserves_template_and_total_energy():
    template = np.arange(12, dtype=float).reshape(3, 4).astype(complex)
    code = np.exp(2j * np.pi * np.arange(8) / 8) / np.sqrt(8)
    lifted = probe_coded_spatial_template(template, code)
    assert np.isclose(np.vdot(lifted, lifted).real,
                      np.vdot(template, template).real)
    assert np.allclose(decode_probe_code(lifted, code), template)
