import numpy as np

from scripts.run_fixed_identity_gate import identity_codebook, joint_pair_ls


def test_identity_codebooks_have_equal_unit_energy():
    for mode in ("shared", "fixed_nonorthogonal", "ideal_orthogonal"):
        codes = identity_codebook(mode)
        assert codes.shape[1] == 2
        assert np.allclose(np.linalg.norm(codes, axis=1), 1.0)
    ideal = identity_codebook("ideal_orthogonal")
    assert np.isclose(abs(np.vdot(ideal[0], ideal[1])), 0.0)


def test_joint_pair_ls_recovers_orthogonal_identity_atoms_without_noise():
    dictionary = np.eye(3, dtype=complex)
    parameters = np.array([
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]
    ])
    codes = identity_codebook("ideal_orthogonal")
    observation = (
        np.outer(codes[0], dictionary[:, 0])
        + np.outer(codes[1], dictionary[:, 2])
    )
    estimate = joint_pair_ls(
        observation, dictionary, parameters, codes, shortlist=2
    )
    assert np.array_equal(estimate, parameters[[0, 2]])
