from dataclasses import replace
from itertools import combinations

import numpy as np

from uav_otfs_isac.communication_aware import (
    communication_aware_sensing_score,
    communication_aware_top_k,
    expected_received_deflection,
)
from uav_otfs_isac.joint_allocation import model_from_bits


def diagonal_model(seed: int = 0):
    rng = np.random.default_rng(seed)
    deltas = np.concatenate(([0.4], rng.uniform(0.8, 2.0, 4)))
    bits = np.array([0, 2, 2, 2, 2])
    model = model_from_bits(deltas, bits, bit_flip_probability=0.05)
    success = np.array([1.0, 0.9, 0.7, 0.8, 0.6])
    return replace(model, success_prob=success, sigma1=model.sigma0)


def test_communication_aware_top_k_maximizes_expected_deflection():
    model = diagonal_model()
    candidates = [
        i for i in range(model.num_uavs)
        if i != model.owner and int(model.report_bits[i]) > 0
    ]
    for budget in (2, 4, 6, 8):
        top = communication_aware_top_k(model, budget)
        best = None
        best_value = -1.0
        for count in range(len(candidates) + 1):
            for subset in combinations(candidates, count):
                scheduled = {model.owner, *subset}
                cost = sum(int(model.report_bits[i]) for i in subset)
                if cost > budget:
                    continue
                value = expected_received_deflection(
                    model, scheduled,
                )
                if value > best_value + 1e-12:
                    best_value = value
                    best = frozenset(scheduled)
        assert top == best


def test_communication_aware_score_matches_formula():
    model = diagonal_model()
    for i in range(model.num_uavs):
        if i == model.owner:
            continue
        expected = (
            model.success_prob[i]
            * model.delta[i] ** 2
            / model.sigma0[i, i]
        )
        assert np.isclose(
            communication_aware_sensing_score(model, i), expected
        )
