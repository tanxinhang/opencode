import numpy as np
import pytest

from scripts.run_partial_confirmation_gate import (
    combine_incremental_difference,
    decoded_sum_difference,
    wilson_interval,
)


def test_partial_difference_decoding_recovers_two_noiseless_sources():
    first = np.array([1.0 + 2.0j, 3.0 - 1.0j])
    second = np.array([-2.0 + 1.0j, 0.5 + 0.5j])
    fraction = 0.25
    decoded = decoded_sum_difference(
        first + second, np.sqrt(fraction) * (first - second), fraction
    )
    assert np.allclose(decoded[0], first)
    assert np.allclose(decoded[1], second)


def test_partial_difference_fraction_is_validated():
    with pytest.raises(ValueError):
        decoded_sum_difference(np.ones(2), np.ones(2), 0.0)


def test_incremental_energy_combination_recovers_full_difference_signal():
    fraction = 0.3
    signal = np.array([1.0 + 1.0j, -2.0j])
    partial = np.sqrt(fraction) * signal
    supplemental = np.sqrt(1.0 - fraction) * signal
    assert np.allclose(
        combine_incremental_difference(partial, supplemental, fraction), signal
    )


def test_wilson_interval_handles_zero_alarms():
    lower, upper = wilson_interval(0, 100)
    assert lower == 0.0
    assert 0.0 < upper < 0.05
