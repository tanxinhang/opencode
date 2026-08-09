from dataclasses import replace

import numpy as np

from uav_otfs_isac.communication_aware import (
    communication_aware_top_k,
    expected_received_deflection,
)
from uav_otfs_isac.joint_allocation import model_from_bits


def models_for_seed(seed: int):
    rng = np.random.default_rng(seed)
    deltas = np.concatenate(([0.4], rng.uniform(0.8, 2.0, 4)))
    bits = np.array([0, 2, 2, 2, 2])
    clean = model_from_bits(deltas, bits, bit_flip_probability=0.0)
    clean = replace(
        clean,
        success_prob=np.array([1.0, 0.9, 0.9, 0.9, 0.9]),
        sigma1=clean.sigma0,
    )
    robust = model_from_bits(deltas, bits, bit_flip_probability=0.2)
    robust = replace(
        robust,
        success_prob=np.array([1.0, 0.5, 0.5, 0.5, 0.5]),
        sigma1=robust.sigma0,
    )
    return clean, robust


def test_robust_cas_top_k_improves_endpoint_expected_deflection():
    for seed in range(5):
        clean, robust = models_for_seed(seed)
        nominal = communication_aware_top_k(clean, 6)
        robust_top = communication_aware_top_k(robust, 6)
        nominal_deflection = expected_received_deflection(robust, nominal)
        robust_deflection = expected_received_deflection(robust, robust_top)
        assert robust_deflection >= nominal_deflection - 1e-12
