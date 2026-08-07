import numpy as np

from uav_otfs_isac.config import load_config
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    build_ris_models,
    quantize_phase,
    ris_array_gain,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_gain_matrix,
    ris_quantized_gain_loss,
    ris_physics_gain_matrix,
)


def test_ris_beam_aligned_gain_is_one():
    config = RisConfig(
        position=np.array([55.0, 15.0, 12.0]),
        num_elements=16,
    )
    target = np.array([20.0, 30.0, 0.0])
    phase = ris_beam_phase(target, config)
    assert np.isclose(ris_array_gain(phase, target, config), 1.0, atol=1e-10)


def test_ris_gain_matrix_boosts_weak_target():
    config = RisConfig(
        position=np.array([55.0, 15.0, 12.0]),
        num_elements=16,
        weak_target_id=2,
        ris_strength_weak=3.0,
        ris_strength_strong=0.5,
    )
    targets = [np.array([40.0, 20.0, 0.0]), np.array([20.0, 40.0, 0.0]), np.array([10.0, 35.0, 0.0])]
    phases = [ris_beam_phase(target, config) for target in targets]
    gains = ris_gain_matrix(config, targets, num_uavs=6, phase_per_target=phases)
    assert np.all(gains >= 0.0)
    assert float(np.mean(gains[2])) > 2.0
    assert float(np.mean(gains[2])) > float(np.mean(gains[0]))
    assert float(np.mean(gains[2])) > float(np.mean(gains[1]))


def test_build_ris_models_validate():
    cfg = load_config("config/demo.yaml")
    config = RisConfig(
        position=np.array([55.0, 15.0, 12.0]),
        num_elements=16,
        weak_target_id=2,
    )
    rng = np.random.default_rng(cfg.seed)
    phases = [
        ris_beam_phase(
            np.array([45.0 * np.cos(1.7 * q), 55.0 * np.sin(1.3 * q), 0.0]),
            config,
        )
        for q in range(cfg.num_targets)
    ]
    models = build_ris_models(cfg, rng, config, phases)
    assert len(models) == cfg.num_targets
    for model in models:
        model.validate()


def test_snr_gain_identity_matches_baseline():
    cfg = load_config("config/demo.yaml")
    baseline = build_models(cfg, np.random.default_rng(cfg.seed))
    gain_models = build_models(
        cfg,
        np.random.default_rng(cfg.seed),
        snr_gain=np.ones((cfg.num_targets, cfg.num_uavs)),
    )
    for left, right in zip(baseline, gain_models):
        assert np.allclose(left.mu0, right.mu0)
        assert np.allclose(left.mu1, right.mu1)
        assert np.allclose(left.sigma0, right.sigma0)


def test_quantize_phase_resolution_and_idempotence():
    phase = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    quantized = quantize_phase(phase, 2)
    assert len(np.unique(np.round(quantized, 10))) == 4
    assert np.allclose(quantize_phase(quantized, 2), quantized)


def test_quantized_gain_loss_matches_simulation():
    rng = np.random.default_rng(7)
    n = 16
    for bits in (1, 2, 3, 4):
        theory = ris_quantized_gain_loss(bits)
        values = []
        for _ in range(2000):
            aligned = rng.uniform(0.0, 2.0 * np.pi, n)
            quantized = quantize_phase(aligned, bits)
            values.append(abs(np.mean(np.exp(1j * (quantized - aligned)))) ** 2)
        empirical = float(np.mean(values))
        assert abs(empirical - theory) < 0.05


def test_ris_control_overhead_ledger():
    continuous = RisConfig(position=np.array([0.0, 0.0]), num_elements=16)
    assert ris_control_overhead_bits(continuous) == 0.0
    quantized = RisConfig(
        position=np.array([0.0, 0.0]),
        num_elements=16,
        phase_bits=2,
    )
    assert ris_control_overhead_bits(quantized, coherence_frames=100) == 0.32


def _physics_geometry():
    transmitters = [
        np.array([180.0, 0.0, 100.0]),
        np.array([-100.0, 150.0, 110.0]),
        np.array([-80.0, -160.0, 100.0]),
    ]
    targets = [
        np.array([40.0, 20.0, 0.0]),
        np.array([20.0, 40.0, 0.0]),
        np.array([10.0, 35.0, 0.0]),
    ]
    receiver = np.array([0.0, 0.0, 0.0])
    return transmitters, targets, receiver


def test_ris_physics_gain_increases_with_elements():
    transmitters, targets, receiver = _physics_geometry()
    means = []
    for elements in (64, 256, 1024):
        config = RisConfig(
            position=np.array([55.0, 15.0, 12.0]),
            num_elements=elements,
            weak_target_id=2,
        )
        phases = [ris_beam_phase(target, config) for target in targets]
        gains = ris_physics_gain_matrix(
            config, transmitters, targets, receiver,
            aperture_scale=1e-2, direct_blockage=0.01,
            phase_per_target=phases,
        )
        means.append(float(np.mean(gains[2])))
    assert means[0] < means[1] < means[2]
    assert means[2] > 5.0


def test_ris_physics_aligned_beats_random_phase():
    transmitters, targets, receiver = _physics_geometry()
    rng = np.random.default_rng(20260805)
    config = RisConfig(
        position=np.array([55.0, 15.0, 12.0]),
        num_elements=256,
        weak_target_id=2,
    )
    aligned = [ris_beam_phase(target, config) for target in targets]
    random_phases = [
        rng.uniform(0.0, 2.0 * np.pi, 256) for _ in targets
    ]
    aligned_gain = ris_physics_gain_matrix(
        config, transmitters, targets, receiver,
        aperture_scale=1e-2, direct_blockage=0.01,
        phase_per_target=aligned,
    )
    random_gain = ris_physics_gain_matrix(
        config, transmitters, targets, receiver,
        aperture_scale=1e-2, direct_blockage=0.01,
        phase_per_target=random_phases,
    )
    assert float(np.mean(aligned_gain[2])) > float(np.mean(random_gain[2]))
    assert np.all(aligned_gain >= 1.0)
