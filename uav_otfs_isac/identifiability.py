"""Joint probe-angle-delay-Doppler identifiability diagnostics."""

from __future__ import annotations

import numpy as np


def normalized_columns(signatures):
    """Normalize a [feature, source] signature matrix column by column."""
    matrix = np.asarray(signatures, dtype=complex)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("signatures must be a nonempty matrix")
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("signature columns must have finite nonzero energy")
    return matrix / norms


def normalized_gram(signatures):
    """Return the Hermitian normalized Gram matrix of joint signatures."""
    columns = normalized_columns(signatures)
    gram = columns.conj().T @ columns
    return 0.5 * (gram + gram.conj().T)


def gram_identifiability_metrics(gram):
    """Compute minimum eigenvalue, condition number, and maximum coherence."""
    matrix = np.asarray(gram, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("gram must be a nonempty square matrix")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
        raise ValueError("gram must be Hermitian")
    eigenvalues = np.linalg.eigvalsh(matrix).real
    minimum = max(float(eigenvalues[0]), 0.0)
    maximum = max(float(eigenvalues[-1]), 0.0)
    condition = float(np.inf if minimum <= 1e-12 else maximum / minimum)
    off_diagonal = np.abs(matrix - np.diag(np.diag(matrix)))
    return {
        "lambda_min": minimum,
        "lambda_max": maximum,
        "condition_number": condition,
        "max_effective_coherence": float(np.max(off_diagonal)),
    }


def joint_signature(probe, steering, waveform):
    """Construct one normalized probe-angle-DD Kronecker signature."""
    vectors = [np.asarray(value, dtype=complex) for value in
               (probe, steering, waveform)]
    if any(vector.ndim != 1 or vector.size == 0 for vector in vectors):
        raise ValueError("probe, steering, and waveform must be nonempty vectors")
    result = np.kron(np.kron(vectors[0], vectors[1]), vectors[2])
    norm = np.linalg.norm(result)
    if norm <= 0.0:
        raise ValueError("joint signature must have nonzero energy")
    return result / norm


def factorized_joint_gram(probes, steerings, waveforms):
    """Form the joint Gram as the Hadamard product of factor Grams."""
    factors = []
    for values in (probes, steerings, waveforms):
        columns = np.column_stack([np.asarray(value, dtype=complex) for value in values])
        factors.append(normalized_gram(columns))
    if not all(factor.shape == factors[0].shape for factor in factors):
        raise ValueError("all factor lists must contain the same source count")
    gram = factors[0] * factors[1] * factors[2]
    return 0.5 * (gram + gram.conj().T)


def worst_case_gram_metrics(grams):
    """Return worst identifiability metrics over a nonempty Gram collection."""
    metrics = [gram_identifiability_metrics(gram) for gram in grams]
    if not metrics:
        raise ValueError("at least one Gram matrix is required")
    return {
        "worst_lambda_min": min(item["lambda_min"] for item in metrics),
        "worst_condition_number": max(
            item["condition_number"] for item in metrics
        ),
        "worst_effective_coherence": max(
            item["max_effective_coherence"] for item in metrics
        ),
        "evaluated_grams": len(metrics),
    }


def minimum_probe_length(grams_by_length, lambda_threshold):
    """Choose the shortest length whose worst-case minimum eigenvalue passes."""
    if not np.isfinite(lambda_threshold) or lambda_threshold < 0.0:
        raise ValueError("lambda_threshold must be finite and nonnegative")
    decisions = {}
    for length in sorted(grams_by_length):
        if int(length) != length or length <= 0:
            raise ValueError("probe lengths must be positive integers")
        metrics = worst_case_gram_metrics(grams_by_length[length])
        decisions[int(length)] = metrics
        if metrics["worst_lambda_min"] >= lambda_threshold:
            return int(length), decisions
    return None, decisions
