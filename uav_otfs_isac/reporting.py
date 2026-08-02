from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


def quantizer_from_gaussian_range(
    mu0: ArrayLike,
    sigma0: ArrayLike,
    mu1: ArrayLike,
    sigma1: ArrayLike,
    bits: int,
    span_std: float = 4.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    low = min(float(np.min(mu0 - span_std * np.sqrt(np.diag(sigma0)))),
              float(np.min(mu1 - span_std * np.sqrt(np.diag(sigma1)))))
    high = max(float(np.max(mu0 + span_std * np.sqrt(np.diag(sigma0)))),
               float(np.max(mu1 + span_std * np.sqrt(np.diag(sigma1)))))
    levels = 2**bits
    boundaries = np.linspace(low, high, levels + 1)
    edges = np.concatenate(([-np.inf], boundaries[1:-1], [np.inf]))
    values = 0.5 * (boundaries[:-1] + boundaries[1:])
    return edges, values


def quantize(values: ArrayLike, edges: ArrayLike) -> NDArray[np.int64]:
    x = np.asarray(values)
    return np.searchsorted(np.asarray(edges)[1:-1], x, side="right").astype(int)


def bsc_transition(bits: int, bit_flip_probability: float) -> NDArray[np.float64]:
    levels = 2**bits
    indices = np.arange(levels, dtype=np.uint64)
    xor = np.bitwise_xor(indices[:, None], indices[None, :])
    distances = np.zeros_like(xor, dtype=int)
    work = xor.copy()
    for _ in range(bits):
        distances += (work & 1).astype(int)
        work >>= 1
    p = float(bit_flip_probability)
    return (p**distances) * ((1.0 - p) ** (bits - distances))


def transmit_indices(
    indices: ArrayLike,
    bits: int,
    bit_flip_probability: ArrayLike,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    sent = np.asarray(indices, dtype=np.int64)
    p = np.asarray(bit_flip_probability, dtype=float)
    received = sent.copy()
    for bit in range(bits):
        flips = rng.random(sent.shape) < p
        received ^= flips.astype(np.int64) << bit
    return received


def gaussian_bin_probabilities(mu: float, variance: float, edges: ArrayLike) -> NDArray[np.float64]:
    std = max(float(np.sqrt(variance)), 1e-12)
    cdf = norm.cdf((np.asarray(edges, dtype=float) - mu) / std)
    return np.diff(cdf)


def post_bsc_moments(
    mu: float,
    variance: float,
    edges: ArrayLike,
    reconstruction_values: ArrayLike,
    bits: int,
    bit_flip_probability: float,
) -> tuple[float, float]:
    source = gaussian_bin_probabilities(mu, variance, edges)
    received = source @ bsc_transition(bits, bit_flip_probability)
    values = np.asarray(reconstruction_values, dtype=float)
    mean = float(received @ values)
    var = float(received @ (values - mean) ** 2)
    return mean, max(var, 1e-12)

