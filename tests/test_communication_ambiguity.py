import numpy as np

from uav_otfs_isac.communication_ambiguity import (
    build_endpoint_models,
    verify_endpoint_dominance,
)


def test_endpoint_models_cover_four_corners():
    models = build_endpoint_models(
        0.4,
        np.array([1.2, 1.5, 1.8]),
        np.array([2, 3, 2]),
        (0.0, 0.2),
        (0.5, 1.0),
    )
    assert len(models) == 4
    for model in models:
        assert model.num_uavs == 4
        assert np.isclose(model.bit_flip_prob[model.owner], 0.0)
        assert np.isclose(model.success_prob[model.owner], 1.0)


def test_endpoint_dominance_holds_on_grid():
    result = verify_endpoint_dominance(
        0.4,
        np.array([1.2, 1.5, 1.8, 2.0]),
        np.array([2, 3, 2, 3]),
        (0.0, 0.2),
        (0.5, 1.0),
        scheduled=set(range(5)),
        minimum_pd=0.2,
        false_alarm_rate=0.05,
        grid=32,
        p_steps=5,
        s_steps=5,
    )
    assert result["passed"]
    assert result["grid_worst_at"] is None
