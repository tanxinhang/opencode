import numpy as np

from uav_otfs_isac.architecture_switch import (
    exact_architecture_switch,
    fixed_budget_architecture_switch,
    reallocate_soft_report_bits,
    selected_architecture_pd,
    target_wise_architecture_switch,
    two_sided_mode_ascent,
)
from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.expected_pd import expected_gaussian_detection_probability


def test_exact_switch_picks_higher_pd():
    assert exact_architecture_switch(0.80, 0.90) == "peer"
    assert exact_architecture_switch(0.90, 0.80) == "soft"
    assert exact_architecture_switch(0.90, 0.90) == "soft"


def test_fixed_budget_switch_uses_threshold():
    assert fixed_budget_architecture_switch(6, threshold_bits=10) == "peer"
    assert fixed_budget_architecture_switch(10, threshold_bits=10) == "soft"
    assert fixed_budget_architecture_switch(22, threshold_bits=10) == "soft"


def test_selected_architecture_pd_matches_mode():
    assert selected_architecture_pd(0.8, 0.9, "peer") == 0.9
    assert selected_architecture_pd(0.8, 0.9, "soft") == 0.8


def test_target_wise_switch_is_never_worse_than_global_switch():
    soft_pds = [0.90, 0.70, 0.85]
    peer_pds = [0.80, 0.92, 0.75]
    _, values = target_wise_architecture_switch(soft_pds, peer_pds)
    target_wise_worst = min(values)
    global_soft_worst = min(soft_pds)
    global_peer_worst = min(peer_pds)
    assert target_wise_worst >= max(global_soft_worst, global_peer_worst)


def test_reallocate_soft_report_bits_is_monotone_and_budget_feasible():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
    )
    before = expected_gaussian_detection_probability(
        model, {model.owner}, 0.05, grid=64
    )
    current, quality, used = reallocate_soft_report_bits(
        [model], ["soft"], [{model.owner}], 4, 0.05, grid=64
    )
    assert quality[0] >= before
    assert used <= 4
    assert len(current[0]) > 1


def test_two_sided_mode_ascent_never_worsens_target_wise_worst():
    models = [
        symmetric_diversity_model(
            np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
        ),
        symmetric_diversity_model(
            np.array([0.6, 0.8, 1.0, 1.2]), success_probability=0.8
        ),
    ]
    peer_pds = [0.75, 0.98]
    scheduled = [{model.owner} for model in models]
    soft_pds = [
        expected_gaussian_detection_probability(
            model, scheduled[q], 0.05, grid=64
        )
        for q, model in enumerate(models)
    ]
    modes, values, _, used = two_sided_mode_ascent(
        models, peer_pds, scheduled, 8, 0.05, grid=64
    )
    target_wise_before = min([
        max(soft_pds[q], peer_pds[q]) for q in range(len(models))
    ])
    target_wise_after = min([
        values[q] if modes[q] == "soft" else peer_pds[q]
        for q in range(len(models))
    ])
    assert target_wise_after >= target_wise_before - 1e-12
    assert used <= 8
