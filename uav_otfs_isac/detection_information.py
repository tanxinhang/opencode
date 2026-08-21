"""Post-communication detection information and sequential evidence accumulation.

The module implements the detection-information layer proposed for the
active-detection upgrade: every quantity downstream of sensing (quantization,
BSC, erasure) is summarized by the post-communication likelihood pair
``(p0_y, p1_y)``, and the *detection information* of a report is measured by
the KL divergences

    I^+ = KL(p1 || p0) = E_{H1}[ LLR ],      I^- = KL(p0 || p1) = -E_{H0}[ LLR ],

which are the average decision-evidence growth rates of the accumulated
log-likelihood ratio (Wald).  The data-processing principle gives the
contraction chain ``I_post <= I_quant <= I_sensing``, and detectable erasure
scales the information linearly: ``I^+ = s * KL(p1_rec || p0_rec)`` with
``s`` the link success probability.

Sequential detection is evaluated *exactly*: the LLR PMF of each observation
is binned on a uniform grid and the accumulated-LLR PMF is obtained by FFT
convolution, so ``P_D(n)`` for a fixed P_FA constraint needs no Gaussian
approximation and heterogeneous observation sequences (different reports /
power levels per cycle) are handled by convolving their PMFs in order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .reporting import bsc_transition


def _bin_probabilities(mu: float, variance: float, edges) -> NDArray[np.float64]:
    """Gaussian bin masses, cancellation-free in the tails.

    ``scipy.stats.norm.cdf`` rounds to exactly 1.0 (or 0.0) once the
    tail mass drops below half an ulp, and subtracting two such values
    gives exact zeros that would artificially disjoint the supports
    (infinite KL) even though the true masses are strictly positive.
    Bins entirely on one side of the mean are computed through the small
    tail value (``Phi(-a) - Phi(-b)`` on the right), and the remaining
    underflow floor is ``1e-300``, below every true mass in the system's
    design family (tails down to ``Phi(-8) ~ 6e-16``), so the quantized KL
    is not inflated and the data-processing chain ``I_post <= I_quant <=
    I_sensing`` holds numerically for every link.
    """
    std = max(float(np.sqrt(variance)), 1e-12)
    a = (np.asarray(edges, dtype=float)[:-1] - mu) / std
    b = (np.asarray(edges, dtype=float)[1:] - mu) / std
    right = a >= 0.0
    left = b <= 0.0
    mid = ~(right | left)
    masses = np.zeros_like(a)
    masses[left] = norm.cdf(b[left]) - norm.cdf(a[left])
    masses[right] = norm.cdf(-a[right]) - norm.cdf(-b[right])
    masses[mid] = norm.cdf(b[mid]) - norm.cdf(a[mid])
    masses = np.maximum(masses, 1e-300)
    return masses / masses.sum()


def discrete_kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Discrete KL divergence ``sum p log(p/q)`` with zero-mass handling.

    Returns ``inf`` when ``p`` has mass where ``q`` is zero.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any(p < 0.0) or np.any(q < 0.0):
        raise ValueError("probability mass functions must be nonnegative")
    if not np.isclose(p.sum(), 1.0, atol=1e-8):
        raise ValueError("p must sum to one")
    if not np.isclose(q.sum(), 1.0, atol=1e-8):
        raise ValueError("q must sum to one")
    positive = p > 0.0
    if np.any(positive & (q == 0.0)):
        return float("inf")
    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def gaussian_kl(
    mu1: float, var1: float, mu0: float, var0: float
) -> float:
    """Closed-form KL(N(mu1, var1) || N(mu0, var0))."""
    var0 = max(float(var0), 1e-12)
    var1 = max(float(var1), 1e-12)
    return float(
        np.log(np.sqrt(var0 / var1))
        + (var1 + (mu1 - mu0) ** 2) / (2.0 * var0)
        - 0.5
    )


def chernoff_information(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    grid: int = 513,
) -> float:
    """Chernoff information ``C = max_s -log sum_y p(y)^s q(y)^(1-s)``.

    Computed by a vectorized grid scan over ``s in [0, 1]`` using the
    log-sum-exp form for numerical stability.  ``C`` bounds the best error
    exponent of a Neyman-Pearson test on repeated observations.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    log_p = np.log(p[p > 0.0])
    log_q = np.log(q[p > 0.0])
    if log_p.size == 0:
        return 0.0
    s_grid = np.linspace(0.0, 1.0, grid)
    exponents = s_grid[:, None] * log_p[None, :] + (1.0 - s_grid)[:, None] * log_q[None, :]
    log_z = np.logaddexp.reduce(exponents, axis=1)
    return float(np.max(-log_z))


def post_communication_likelihoods(
    mu0: float,
    var0: float,
    mu1: float,
    var1: float,
    edges: NDArray[np.float64],
    values: NDArray[np.float64],
    bits: int,
    flip_probability: float,
    success_probability: float,
) -> dict[str, float | NDArray[np.float64]]:
    """Post-communication H0/H1 likelihoods of one report link.

    Builds the quantized level PMFs ``p0_q``/``p1_q`` from the scalar
    Gaussian evidence, propagates them through the BSC, and appends the
    detectable-erasure atom (LLR contribution zero, mass ``1 - s`` under
    both hypotheses).

    Returns (among others) ``p0_y``/``p1_y`` over ``{levels} U {erased}``,
    the detection information ``kl_plus``/``kl_minus``, ``chernoff``, and
    the contraction-chain quantities ``kl_quant``/``kl_sensing``.
    """
    var0 = max(float(var0), 1e-12)
    var1 = max(float(var1), 1e-12)
    p0_q = _bin_probabilities(mu0, var0, edges)
    p1_q = _bin_probabilities(mu1, var1, edges)
    transition = bsc_transition(int(bits), float(flip_probability))
    p0_rec = p0_q @ transition
    p1_rec = p1_q @ transition
    s = float(success_probability)
    p0_y = np.concatenate((s * p0_rec, np.array([1.0 - s])))
    p1_y = np.concatenate((s * p1_rec, np.array([1.0 - s])))
    kl_quant = discrete_kl(p1_q, p0_q)
    kl_sensing = gaussian_kl(mu1, var1, mu0, var0)
    return {
        "p0_q": p0_q,
        "p1_q": p1_q,
        "p0_rec": p0_rec,
        "p1_rec": p1_rec,
        "p0_y": p0_y,
        "p1_y": p1_y,
        "kl_plus": s * discrete_kl(p1_rec, p0_rec),
        "kl_minus": s * discrete_kl(p0_rec, p1_rec),
        "kl_rec_plus": discrete_kl(p1_rec, p0_rec),
        "kl_quant": kl_quant,
        "kl_sensing": kl_sensing,
        "chernoff": chernoff_information(p1_y, p0_y),
    }


def llr_pmf(
    p1_y: NDArray[np.float64],
    p0_y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """LLR values with their H1/H0 probabilities.

    Returns ``(llr, pmf1, pmf0)`` where ``pmf1[k] = P_{H1}(LLR = llr[k])``.
    Symbols with zero mass under both hypotheses are dropped; symbols with
    equal H0/H1 mass (e.g. the erasure atom) contribute an LLR of zero.

    Raises when a symbol has positive mass under exactly one hypothesis:
    its LLR is infinite and the uniform-grid convolution machinery cannot
    represent it.  Every symbol produced by ``post_communication_likelihoods``
    satisfies the contract (bin-mass floors are strictly positive), so the
    raise guards the public API against silently corrupted results rather
    than triggering in the system's call paths.
    """
    p1_y = np.asarray(p1_y, dtype=float)
    p0_y = np.asarray(p0_y, dtype=float)
    keep = (p1_y > 0.0) | (p0_y > 0.0)
    p1 = p1_y[keep]
    p0 = p0_y[keep]
    one_sided = (p1 > 0.0) != (p0 > 0.0)
    if np.any(one_sided):
        raise ValueError(
            "symbols with positive mass under exactly one hypothesis are "
            "not supported (infinite LLR); post_communication_likelihoods "
            "never produces them"
        )
    llr = np.log(p1 / p0)
    return llr, p1, p0


def _bin_pmf(
    llr: NDArray[np.float64],
    pmf1: NDArray[np.float64],
    pmf0: NDArray[np.float64],
    grid_step: float,
    size: int,
    base: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate an observation's LLR PMF onto a uniform integer grid."""
    indices = np.rint(llr / grid_step).astype(np.int64)
    out1 = np.zeros(size, dtype=float)
    out0 = np.zeros(size, dtype=float)
    np.add.at(out1, base + indices, pmf1)
    np.add.at(out0, base + indices, pmf0)
    return out1, out0


def _fft_convolve_centered(
    acc: NDArray[np.float64],
    kernel: NDArray[np.float64],
    base: int,
) -> NDArray[np.float64]:
    """Full convolution of two PMFs on a uniform grid with re-centering.

    ``acc`` and ``kernel`` both represent probabilities as functions of
    an LLR offset measured from index ``base``.  The ordinary convolution
    shifts the zero point to ``2*base``; this function re-centers the
    result so that the accumulated-LLR zero stays at index ``base``.
    """
    total = len(acc) + len(kernel) - 1
    n = int(2 ** np.ceil(np.log2(total)))
    conv = np.fft.irfft(
        np.fft.rfft(acc, n) * np.fft.rfft(kernel, n), n,
    )[:total]
    conv = np.maximum(conv, 0.0)
    out = np.zeros(len(acc), dtype=float)
    crop = conv[base: base + len(acc)]
    out[: len(crop)] = crop
    mass = out.sum()
    if mass > 0.0:
        out /= mass
    return out


def sequential_pd(
    p1_y: NDArray[np.float64],
    p0_y: NDArray[np.float64],
    n: int,
    alpha: float = 0.05,
    grid_step: float = 0.01,
) -> dict[str, float | int]:
    """Exact P_FA/P_D of the NP test after ``n`` i.i.d. post-communication
    observations of one report.

    The accumulated LLR PMF under H0/H1 is computed by iterated FFT
    convolution; the threshold is the largest grid point whose H0
    upper tail is at most ``alpha``.  Returns the achieved P_FA, P_D,
    the threshold, and ``n``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    llr, pmf1, pmf0 = llr_pmf(p1_y, p0_y)
    span = max(np.abs(llr).max(), 1e-9) + grid_step
    size = int(2 * (n * span / grid_step) + 8)
    base = size // 2
    acc1, acc0 = _bin_pmf(llr, pmf1, pmf0, grid_step, size, base)
    for _ in range(n - 1):
        k1, k0 = _bin_pmf(llr, pmf1, pmf0, grid_step, size, base)
        acc1 = _fft_convolve_centered(acc1, k1, base)
        acc0 = _fft_convolve_centered(acc0, k0, base)
    cdf0 = np.cumsum(acc0)
    cdf1 = np.cumsum(acc1)
    j = int(np.searchsorted(cdf0, 1.0 - alpha))
    j = min(max(j, 0), size - 1)
    pfa = float(np.clip(1.0 - cdf0[j], 0.0, 1.0))
    pd = float(np.clip(1.0 - cdf1[j], 0.0, 1.0))
    return {"n": n, "alpha": alpha, "pfa": pfa, "pd": pd,
            "threshold": float((j - base) * grid_step)}


def sequential_pd_sequence(
    observations: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    alpha: float = 0.05,
    grid_step: float = 0.01,
) -> dict[str, float | int]:
    """Exact P_FA/P_D after a *heterogeneous* observation sequence.

    ``observations`` is a list of ``(p1_y, p0_y)`` pairs, one per received
    observation in order (different reports / power levels are allowed).
    The accumulated LLR is the convolution of the per-observation LLR PMFs.
    """
    if not observations:
        raise ValueError("at least one observation is required")
    n = len(observations)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    lr_list = [llr_pmf(p1, p0) for p1, p0 in observations]
    max_abs = max(np.abs(lr).max() for lr, _, _ in lr_list)
    span = max(max_abs, 1e-9) + grid_step
    size = int(2 * (n * span / grid_step) + 8)
    base = size // 2
    first_lr, first_p1, first_p0 = lr_list[0]
    acc1, acc0 = _bin_pmf(first_lr, first_p1, first_p0, grid_step, size, base)
    for lr, p1, p0 in lr_list[1:]:
        k1, k0 = _bin_pmf(lr, p1, p0, grid_step, size, base)
        acc1 = _fft_convolve_centered(acc1, k1, base)
        acc0 = _fft_convolve_centered(acc0, k0, base)
    cdf0 = np.cumsum(acc0)
    cdf1 = np.cumsum(acc1)
    j = int(np.searchsorted(cdf0, 1.0 - alpha))
    j = min(max(j, 0), size - 1)
    pfa = float(np.clip(1.0 - cdf0[j], 0.0, 1.0))
    pd = float(np.clip(1.0 - cdf1[j], 0.0, 1.0))
    return {"n": n, "alpha": alpha, "pfa": pfa, "pd": pd,
            "threshold": float((j - base) * grid_step)}


def cycles_to_pd(
    p1_y: NDArray[np.float64],
    p0_y: NDArray[np.float64],
    alpha: float,
    pd_star: float,
    max_n: int = 16,
    grid_step: float = 0.01,
) -> int | None:
    """Minimal cycle count ``n`` with exact ``P_D(n) >= pd_star``.

    Returns ``None`` when the target is not reachable within ``max_n``
    cycles (or when even ``P_D(1)`` saturates below ``pd_star`` with
    ``n -> inf``; the caller may treat a non-None result of ``max_n``
    as an upper bound).
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < pd_star < 1.0:
        raise ValueError("alpha and pd_star must lie in (0, 1)")
    for n in range(1, max_n + 1):
        result = sequential_pd(p1_y, p0_y, n, alpha, grid_step)
        if float(result["pd"]) >= pd_star:
            return n
    return None


def contraction_chain(
    mu0: float, var0: float, mu1: float, var1: float,
    edges: NDArray[np.float64], values: NDArray[np.float64],
    bits: int, flip_probability: float, success_probability: float,
) -> dict[str, float | bool]:
    """Detection-information contraction ``I_post <= I_quant <= I_sensing``.

    Returns the three information values and whether the data-processing
    chain holds for this link.  The chain is exact for the scalar-Gaussian
    quantization model (quantization is a deterministic function, BSC is a
    channel, erasure is a detectable-output operation).
    """
    info = post_communication_likelihoods(
        mu0, var0, mu1, var1, edges, values,
        bits, flip_probability, success_probability,
    )
    i_sensing = float(info["kl_sensing"])
    i_quant = float(info["kl_quant"])
    i_post = float(info["kl_plus"])
    held = i_post <= i_quant + 1e-12 and i_quant <= i_sensing + 1e-12
    return {
        "i_sensing": i_sensing,
        "i_quant": i_quant,
        "i_post": i_post,
        "contraction_holds": held,
    }


def predicted_remaining_cycles(
    n_so_far: int,
    i_plus: float,
    threshold_next: float,
) -> float:
    """Predicted cycles until the decision boundary under H1.

    Wald-style first-order estimate ``tau ~= (eta(n+1) - n * I^+) / I^+``
    with ``eta`` the NP threshold of the next checkpoint and ``I^+`` the
    per-observation expected LLR drift under H1.  Nonpositive values mean
    the boundary is expected to be crossed within the next cycle.
    """
    if i_plus <= 1e-12:
        return float("inf")
    return float((threshold_next - n_so_far * i_plus) / i_plus)


def threshold_curve(
    p1_y: NDArray[np.float64],
    p0_y: NDArray[np.float64],
    alpha: float,
    max_n: int,
    grid_step: float = 0.01,
) -> NDArray[np.float64]:
    """NP threshold ``eta(n)`` for ``n = 1..max_n`` i.i.d. observations."""
    eta = np.zeros(max_n, dtype=float)
    for n in range(1, max_n + 1):
        result = sequential_pd(p1_y, p0_y, n, alpha, grid_step)
        eta[n - 1] = float(result["threshold"])
    return eta