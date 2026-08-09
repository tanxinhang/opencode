import numpy as np

from uav_otfs_isac.config import load_config
from uav_otfs_isac.fusion import optimal_deflection
from uav_otfs_isac.physical_link_model import (
    bpsk_bit_flip_probability,
    build_physical_link_models,
    lognormal_outage_success,
    physical_report_link_parameters,
    report_link_snr_db,
)
from uav_otfs_isac.scenario import uav_geometry


def test_bpsk_flip_probability_increases_as_snr_falls():
    low = bpsk_bit_flip_probability(20.0)
    high = bpsk_bit_flip_probability(5.0)
    assert low > 0.0
    assert low < high < 0.5


def test_lognormal_outage_success_is_monotone_in_threshold():
    easy = lognormal_outage_success(20.0, threshold_db=5.0, shadowing_db=3.0)
    hard = lognormal_outage_success(20.0, threshold_db=15.0, shadowing_db=3.0)
    assert hard < easy


def test_physical_parameters_match_closed_formulas():
    cfg = load_config("config/demo.yaml")
    positions = uav_geometry(cfg.num_uavs)
    flip, success = physical_report_link_parameters(
        cfg,
        reference_snr_db=20.0,
        threshold_db=5.0,
        shadowing_db=3.0,
    )
    owner = cfg.owners[0]
    snr_db = report_link_snr_db(
        positions, owner, reference_snr_db=20.0
    )
    for i in range(cfg.num_uavs):
        if i == owner:
            assert np.isclose(flip[0, i], 0.0)
            assert np.isclose(success[0, i], 1.0)
        else:
            assert np.isclose(
                flip[0, i], bpsk_bit_flip_probability(snr_db[i])
            )
            assert np.isclose(
                success[0, i],
                lognormal_outage_success(snr_db[i], 5.0, 3.0),
            )


def test_physical_link_models_are_valid_and_use_derived_links():
    cfg = load_config("config/demo.yaml")
    models = build_physical_link_models(
        cfg,
        cfg.seed,
        reference_snr_db=20.0,
        threshold_db=5.0,
        shadowing_db=3.0,
    )
    assert len(models) == cfg.num_targets
    assert np.isclose(models[0].bit_flip_prob[models[0].owner], 0.0)
    assert np.isclose(models[0].success_prob[models[0].owner], 1.0)


def test_sensing_and_communication_channels_are_decoupled():
    cfg = load_config("config/demo.yaml")
    high_comm = build_physical_link_models(
        cfg,
        cfg.seed,
        reference_snr_db=30.0,
        threshold_db=5.0,
        shadowing_db=3.0,
    )
    low_comm = build_physical_link_models(
        cfg,
        cfg.seed,
        reference_snr_db=5.0,
        threshold_db=5.0,
        shadowing_db=3.0,
    )
    for high, low in zip(high_comm, low_comm):
        # Owner-only evidence uses the sensing channel only, so it must be
        # invariant to the communication-channel reference SNR.
        high_owner_pd = optimal_deflection(
            high.delta, high.sigma0, {high.owner}
        )
        low_owner_pd = optimal_deflection(
            low.delta, low.sigma0, {low.owner}
        )
        assert np.isclose(high_owner_pd, low_owner_pd)
        # Report links must differ because the communication channel differs.
        assert np.any(high.bit_flip_prob != low.bit_flip_prob)
