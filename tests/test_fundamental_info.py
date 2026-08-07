import numpy as np
from scipy.stats import norm

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fundamental_info import (
    effective_deflection,
    full_info_deflection,
    hard_consensus_information,
    hard_kl_information,
    schedule_deflection,
)


def _model():
    return symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
    )


def test_full_info_deflection_upper_bounds_schedule():
    model = _model()
    full = full_info_deflection([model])
    owner = schedule_deflection([model], [{model.owner}])
    assert full[0] > owner[0]


def test_hard_kl_information_positive():
    model = _model()
    values = [
        hard_kl_information(model, uav) for uav in range(model.num_uavs)
        if uav != model.owner
    ]
    assert all(value > 0.0 for value in values)
    consensus = hard_consensus_information([model])
    assert consensus[0] > 0.0


def test_effective_deflection_inverts_gaussian_detection_probability():
    false_alarm_rate = 0.05
    z = norm.ppf(1.0 - false_alarm_rate)
    for pd in (0.6, 0.75, 0.9, 0.99):
        d_eff = effective_deflection(pd, false_alarm_rate)
        round_trip = norm.cdf(np.sqrt(d_eff) - z)
        assert np.isclose(round_trip, pd, atol=1e-9)


def test_effective_deflection_is_strictly_monotone_in_pd():
    false_alarm_rate = 0.05
    values = [
        effective_deflection(pd, false_alarm_rate)
        for pd in (0.55, 0.7, 0.85, 0.99)
    ]
    assert all(a < b for a, b in zip(values, values[1:]))
