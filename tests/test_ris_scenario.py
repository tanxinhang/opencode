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
    # P1-1: the weak-target gain is ``direct_blockage + P_ris/P_dir``: it
    # grows with aperture (the N^2 RIS lift), never falls below blockage,
    # and the direct term never carries the old ~100x ``1/blockage`` boost.
    assert means[0] < means[1] < means[2]
    assert all(value >= 0.01 for value in means)


def test_blockage_attenuates_only_the_direct_term():
    """P1-1 regression: for the weak target the gain is
    ``direct_blockage + P_ris/P_dir`` (unblocked reference), never
    ``1 + P_ris/(blockage*P_dir)``."""
    tx = np.array([180.0, 0.0, 100.0])
    targets = [np.array([40.0, 20.0, 0.0]), np.array([10.0, 35.0, 0.0])]
    receiver = np.array([0.0, 0.0, 0.0])
    config = RisConfig(
        position=np.array([55.0, 15.0, 12.0]),
        num_elements=256,
        weak_target_id=1,
    )
    phases = [ris_beam_phase(target, config) for target in targets]
    gains = ris_physics_gain_matrix(
        config, [tx], targets, receiver,
        aperture_scale=1e-2, direct_blockage=0.01, phase_per_target=phases,
    )
    weak = gains[1, 0]
    # P_dir = 1/(R_tx^2 R_rx^2); P_ris = N^2 G^2 A/(R1^2 R2^2 R3^2)
    tx_ris = float(np.linalg.norm(tx - config.position))
    ris_target = float(np.linalg.norm(config.position - targets[1]))
    target_rx = float(np.linalg.norm(targets[1] - receiver))
    tx_target = float(np.linalg.norm(tx - targets[1]))
    array_gain = ris_array_gain(phases[1], targets[1], config)
    direct = 1.0 / (tx_target**2 * target_rx**2)
    ris = (
        config.num_elements**2 * array_gain**2 * 1e-2
        / (tx_ris**2 * ris_target**2 * target_rx**2)
    )
    assert np.isclose(weak, 0.01 + ris / direct, atol=1e-12)
    assert weak < 1.0  # blocked weak target cannot reach the clean baseline
    # strong target stays at the clean 1 + P_ris/P_dir floor
    array_gain_strong = ris_array_gain(phases[0], targets[0], config)
    tx_ris_s = float(np.linalg.norm(tx - config.position))
    ris_t_s = float(np.linalg.norm(config.position - targets[0]))
    rx_s = float(np.linalg.norm(targets[0] - receiver))
    tx_s = float(np.linalg.norm(tx - targets[0]))
    direct_s = 1.0 / (tx_s**2 * rx_s**2)
    ris_s = (
        config.num_elements**2 * array_gain_strong**2 * 1e-2
        / (tx_ris_s**2 * ris_t_s**2 * rx_s**2)
    )
    assert np.isclose(gains[0, 0], 1.0 + ris_s / direct_s, atol=1e-12)


def test_ris_physics_aligned_beats_random():
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
    # P1-1: strong targets keep the clean ``1 + P_ris/P_dir`` floor; the
    # blocked weak target never falls below ``direct_blockage`` (alignment
    # only ever adds RIS power on top of it).
    strong = np.delete(aligned_gain, 2, axis=0)
    assert np.all(strong >= 1.0)
    assert np.all(aligned_gain[2] >= 0.01)
