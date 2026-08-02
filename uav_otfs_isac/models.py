from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TargetEvidenceModel:
    """Post-report moment model for one aligned candidate target."""

    target_id: int
    owner: int
    mu0: NDArray[np.float64]
    mu1: NDArray[np.float64]
    sigma0: NDArray[np.float64]
    sigma1: NDArray[np.float64]
    success_prob: NDArray[np.float64]
    report_bits: NDArray[np.int64]
    bit_flip_prob: NDArray[np.float64]
    quantizer_edges: NDArray[np.float64]
    quantizer_values: NDArray[np.float64]

    @property
    def num_uavs(self) -> int:
        return int(self.mu0.size)

    @property
    def delta(self) -> NDArray[np.float64]:
        return self.mu1 - self.mu0

    def validate(self) -> None:
        n = self.num_uavs
        for array in (
            self.mu1,
            self.success_prob,
            self.report_bits,
            self.bit_flip_prob,
        ):
            if array.size != n:
                raise ValueError("per-UAV arrays have inconsistent sizes")
        if self.sigma0.shape != (n, n) or self.sigma1.shape != (n, n):
            raise ValueError("covariance matrices must be square num_uavs matrices")
        if not np.allclose(self.sigma0, self.sigma0.T):
            raise ValueError("sigma0 must be symmetric")
        if np.linalg.eigvalsh(self.sigma0).min() <= 0:
            raise ValueError("sigma0 must be positive definite after regularization")


@dataclass(frozen=True)
class SelectionResult:
    scheduled: tuple[frozenset[int], ...]
    expected_deflection: NDArray[np.float64]
    used_bits: int
    normalized_qos_gap: float
    trace: tuple[dict[str, float | int | str], ...]
    quality_mode: str = "deflection"
    expected_quality: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class DetectionMetrics:
    pd_per_target: NDArray[np.float64]
    pfa_per_target: NDArray[np.float64]
    mean_pd: float
    worst_pd: float
    used_bits: int
    selected_reports: int
