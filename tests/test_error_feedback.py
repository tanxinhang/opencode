import numpy as np

from uav_otfs_isac.error_feedback import (
    evaluate_feedback_gain,
    one_shot_wta,
    wta_feedback_allocator,
)


def test_feedback_corrects_noisy_winner_choice():
    true = np.array([1.0, 2.0, 0.5])
    noisy = np.array([2.0, 0.5, 1.0])
    one = one_shot_wta(true, noisy, budget=5.0)
    feedback = wta_feedback_allocator(
        true, noisy, 5.0,
        rounds=20, learning_rate=0.5, explore=2,
    )
    assert one["best_report"] != 1
    assert feedback["best_report"] == 1
    assert feedback["true_deflection"] > one["true_deflection"]


def test_feedback_never_uses_more_than_budget():
    true = np.array([1.0, 2.0, 3.0])
    result = wta_feedback_allocator(
        true, true + 0.1, 4.0,
        rounds=5, learning_rate=0.5, explore=1,
    )
    assert np.isclose(sum(result["allocation"]), 4.0)


def test_feedback_gain_positive_on_noisy_scenarios():
    true = np.array([1.0, 2.0, 1.5, 0.8])
    gains = [
        evaluate_feedback_gain(
            true, 0.8, budget=4.0, rounds=10,
            learning_rate=0.5, explore=2, seed=seed,
        )["feedback_improvement"]
        for seed in range(5)
    ]
    assert np.mean(gains) >= 0.0
