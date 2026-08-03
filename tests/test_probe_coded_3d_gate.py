import numpy as np

from scripts.run_probe_coded_3d_gate import make_probe_codes


def test_cazac_and_dft_probe_pairs_are_orthogonal():
    for length in (2, 4, 8):
        for kind in ("cazac", "dft"):
            first, second = make_probe_codes(length, kind)
            assert np.isclose(np.linalg.norm(first), 1.0)
            assert np.isclose(np.linalg.norm(second), 1.0)
            assert abs(np.vdot(first, second)) < 1e-12
