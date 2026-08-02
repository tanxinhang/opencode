from pathlib import Path

import numpy as np

from uav_otfs_isac.waveform import (
    ExternalOTFSBackend,
    normalized_matched_energy,
    shift_dd,
)


def test_fractional_dd_shift_preserves_energy_and_causes_bin_leakage():
    grid = np.zeros((16, 32), dtype=complex)
    grid[0, 0] = 1.0
    shifted = shift_dd(grid, delay_shift=0.0, doppler_shift=0.37)
    assert np.isclose(np.vdot(shifted, shifted).real, 1.0, atol=1e-12)
    integer_template_energy = normalized_matched_energy(shifted, grid, 1.0)
    matched_template_energy = normalized_matched_energy(shifted, shifted, 1.0)
    assert integer_template_energy < matched_template_energy


def test_external_backend_rejects_non_checkout(tmp_path: Path):
    backend = ExternalOTFSBackend(tmp_path)
    try:
        backend.load()
    except FileNotFoundError as exc:
        assert "OTFS.py" in str(exc)
    else:
        raise AssertionError("invalid checkout should be rejected")
