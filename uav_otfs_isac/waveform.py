from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray


def fractional_shift_kernel(size: int, shift: float) -> NDArray[np.complex128]:
    """Unitary circular fractional shift via the DFT shift theorem."""
    frequencies = np.fft.fftfreq(size)
    impulse = np.zeros(size, dtype=complex); impulse[0] = 1.0
    return np.fft.ifft(np.fft.fft(impulse) * np.exp(-2j * np.pi * frequencies * shift))


def shift_dd(
    grid: NDArray[np.complex128], delay_shift: float, doppler_shift: float
) -> NDArray[np.complex128]:
    """Apply a separable fractional circular shift in the DD grid."""
    kd = fractional_shift_kernel(grid.shape[0], doppler_shift)
    ld = fractional_shift_kernel(grid.shape[1], delay_shift)
    result = np.empty_like(grid, dtype=complex)
    for col in range(grid.shape[1]):
        result[:, col] = np.fft.ifft(np.fft.fft(grid[:, col]) * np.fft.fft(kd))
    shifted = np.empty_like(result)
    for row in range(result.shape[0]):
        shifted[row, :] = np.fft.ifft(np.fft.fft(result[row, :]) * np.fft.fft(ld))
    return shifted


def dd_observation(
    transmit_grid: NDArray[np.complex128],
    paths: list[tuple[complex, float, float]],
    noise_variance: float,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    received = np.zeros_like(transmit_grid, dtype=complex)
    for gain, delay, doppler in paths:
        received += gain * shift_dd(transmit_grid, delay, doppler)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(received.shape) + 1j * rng.standard_normal(received.shape)
    )
    return received + noise


def normalized_matched_energy(
    observation: NDArray[np.complex128],
    template: NDArray[np.complex128],
    noise_variance: float,
) -> float:
    denominator = max(noise_variance * float(np.vdot(template, template).real), 1e-14)
    return float(abs(np.vdot(template, observation)) ** 2 / denominator)


@dataclass(frozen=True)
class ExternalOTFSBackend:
    """Optional adapter to the GPL-3.0 whatshow/Phy_Mod_OTFS checkout.

    The project does not copy that implementation. Users opt in by pointing
    this adapter at an independently obtained checkout and must comply with its
    license when redistributing a combined work.
    """

    checkout: Path

    def load(self):
        checkout = self.checkout.resolve()
        if not (checkout / "OTFS.py").exists():
            raise FileNotFoundError(f"OTFS.py not found in {checkout}")
        sys.path.insert(0, str(checkout))
        try:
            from OTFS import OTFS  # type: ignore
        finally:
            sys.path.pop(0)
        return OTFS

