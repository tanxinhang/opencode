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
    reception_patterns: NDArray[np.int8] | None = None
    pattern_probabilities: NDArray[np.float64] | None = None
    reception_state_probabilities: NDArray[np.float64] | None = None
    conditional_success_probabilities: NDArray[np.float64] | None = None

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
        if (self.reception_patterns is None) != (self.pattern_probabilities is None):
            raise ValueError("reception patterns and probabilities must be provided together")
        if self.reception_patterns is not None:
            patterns = np.asarray(self.reception_patterns)
            probabilities = np.asarray(self.pattern_probabilities)
            if patterns.ndim != 2 or patterns.shape[1] != n:
                raise ValueError("reception_patterns must have shape [patterns, num_uavs]")
            if probabilities.shape != (patterns.shape[0],):
                raise ValueError("pattern_probabilities has an incompatible shape")
            if np.any((patterns != 0) & (patterns != 1)):
                raise ValueError("reception patterns must be binary")
            if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
                raise ValueError("pattern probabilities must be nonnegative and sum to one")
            marginals = probabilities @ patterns
            if not np.allclose(marginals, self.success_prob, atol=1e-10):
                raise ValueError("reception-pattern marginals must match success_prob")
        if ((self.reception_state_probabilities is None) !=
                (self.conditional_success_probabilities is None)):
            raise ValueError("state probabilities and conditional successes must be provided together")
        if self.reception_state_probabilities is not None:
            state_probabilities = np.asarray(self.reception_state_probabilities)
            conditional = np.asarray(self.conditional_success_probabilities)
            if conditional.shape != (state_probabilities.size, n):
                raise ValueError("conditional success probabilities have an incompatible shape")
            if np.any(state_probabilities < 0) or not np.isclose(state_probabilities.sum(), 1.0):
                raise ValueError("state probabilities must be nonnegative and sum to one")
            if np.any((conditional < 0.0) | (conditional > 1.0)):
                raise ValueError("conditional success probabilities must lie in [0, 1]")
            if not np.allclose(state_probabilities @ conditional, self.success_prob, atol=1e-10):
                raise ValueError("conditional-state marginals must match success_prob")


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
class ExpectedPdSelectionResult:
    scheduled: tuple[frozenset[int], ...]
    expected_pd: NDArray[np.float64]
    used_bits: int
    normalized_qos_gap: float
    trace: tuple[dict[str, float | int | str], ...]
    certificate_upper_bound: float | None = None


@dataclass(frozen=True)
class DetectionMetrics:
    pd_per_target: NDArray[np.float64]
    pfa_per_target: NDArray[np.float64]
    mean_pd: float
    worst_pd: float
    used_bits: int
    selected_reports: int
