import numpy as np

from uav_otfs_isac.config import load_config
from uav_otfs_isac.robustness_stress import (
    StressProfile,
    build_stress_models,
    evaluate_stress_profile,
    inr_from_sources,
    stress_target_positions,
    survival_envelope,
)
from uav_otfs_isac.scenario import target_geometry


CLEAN = StressProfile("clean")
COMBINED = StressProfile(
    "combined",
    interference_to_noise=20.0,
    bit_flip_probability=0.15,
    success_probability_scale=0.6,
    mobility_std=6.0,
)


def test_stress_models_apply_channel_degradation_and_stay_valid():
    cfg = load_config("config/demo.yaml")
    clean = build_stress_models(cfg, cfg.seed, CLEAN)
    stressed = build_stress_models(cfg, cfg.seed, COMBINED)
    assert np.all(
        stressed[0].success_prob <= clean[0].success_prob + 1e-12
    )
    assert np.allclose(
        stressed[0].bit_flip_prob,
        [0.0] + [0.15] * (stressed[0].num_uavs - 1),
    )
    assert np.isclose(stressed[0].success_prob[stressed[0].owner], 1.0)
    assert np.isclose(stressed[0].bit_flip_prob[stressed[0].owner], 0.0)


def test_combined_stress_reduces_worst_expected_pd():
    cfg = load_config("config/demo.yaml")
    clean = evaluate_stress_profile(
        cfg, cfg.seed, CLEAN, budget_bits=20, grid=64
    )
    combined = evaluate_stress_profile(
        cfg, cfg.seed, COMBINED, budget_bits=20, grid=64
    )
    assert combined["worst_pd"] < clean["worst_pd"] - 1e-6
    assert combined["used_bits"] <= 20
    assert np.isfinite(combined["worst_pd"])


def test_bsc_stress_changes_evidence_and_reduces_worst_pd():
    cfg = load_config("config/demo.yaml")
    clean = evaluate_stress_profile(
        cfg, cfg.seed, CLEAN, budget_bits=20, grid=64
    )
    hard_channel = evaluate_stress_profile(
        cfg,
        cfg.seed,
        StressProfile("hard_channel", bit_flip_probability=0.15),
        budget_bits=20,
        grid=64,
    )
    clean_models = build_stress_models(cfg, cfg.seed, CLEAN)
    hard_models = build_stress_models(
        cfg, cfg.seed, StressProfile("hard_channel", bit_flip_probability=0.15)
    )
    assert not np.allclose(hard_models[0].mu1, clean_models[0].mu1)
    assert hard_channel["worst_pd"] < clean["worst_pd"] - 1e-6
    assert np.allclose(
        hard_models[0].bit_flip_prob,
        [0.0] + [0.15] * (hard_models[0].num_uavs - 1),
    )


def test_inr_from_sources_follows_free_space_path_loss():
    positions = np.array([
        [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
    ])
    profile = inr_from_sources(
        [(0.0, 0.0, 0.0)],
        [0.4],
        positions,
        reference_distance=100.0,
    )
    assert np.allclose(profile, [0.4, 0.1])
    multi = inr_from_sources(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 100.0)],
        [0.4, 0.1],
        positions,
        reference_distance=100.0,
    )
    assert np.all(multi > profile)


def test_stress_target_displacement_is_bounded():
    cfg = load_config("config/demo.yaml")
    profile = StressProfile(
        "bounded_mobility", mobility_std=4.0, max_displacement_std=3.0
    )
    targets = stress_target_positions(cfg, cfg.seed, profile)
    base = [target_geometry(q) for q in range(cfg.num_targets)]
    for perturbed, original in zip(targets, base):
        displacement = np.linalg.norm(perturbed - original)
        assert displacement <= 3.0 * 4.0 + 1e-12


def test_survival_envelope_records_finite_monotone_rows():
    cfg = load_config("config/demo.yaml")
    interference = StressProfile(
        "interference", interference_to_noise=10.0
    )
    rows = survival_envelope(
        cfg,
        [CLEAN, interference, COMBINED],
        seeds=2,
        budget_bits=20,
        grid=32,
        qos_target=0.7,
    )
    assert len(rows) == 3
    assert [row["label"] for row in rows] == [
        "clean", "interference", "combined"
    ]
    clean_worst = rows[0]["worst_pd_mean"]
    assert all(np.isfinite(row["worst_pd_mean"]) for row in rows)
    assert rows[1]["worst_pd_mean"] <= clean_worst + 1e-9
    assert rows[2]["worst_pd_mean"] <= clean_worst + 1e-9
    assert all(row["qos_rate"] is not None for row in rows)
    assert all(
        cell["used_bits"] <= 20 for row in rows for cell in row["cells"]
    )
