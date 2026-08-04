"""Monotone calibration of front-end path-candidate scores."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsotonicProbabilityCalibrator:
    thresholds: np.ndarray
    probabilities: np.ndarray
    clip: float = 1e-4

    def __post_init__(self) -> None:
        thresholds = np.asarray(self.thresholds, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if thresholds.ndim != 1 or thresholds.shape != probabilities.shape:
            raise ValueError("thresholds and probabilities must be equal vectors")
        if len(thresholds) == 0 or np.any(~np.isfinite(thresholds)):
            raise ValueError("calibrator must contain finite thresholds")
        if np.any(np.diff(thresholds) <= 0):
            raise ValueError("thresholds must be strictly increasing")
        if np.any(np.diff(probabilities) < -1e-12):
            raise ValueError("probabilities must be nondecreasing")
        if not 0 < self.clip < 0.5:
            raise ValueError("clip must lie in (0, 0.5)")
        object.__setattr__(self, "thresholds", thresholds.copy())
        object.__setattr__(self, "probabilities", probabilities.copy())

    def predict(self, scores) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("scores must be finite")
        # Thresholds are the inclusive upper endpoints of PAV blocks.
        indices = np.searchsorted(self.thresholds, values, side="left")
        indices = np.clip(indices, 0, len(self.thresholds) - 1)
        return np.clip(
            self.probabilities[indices], self.clip, 1.0 - self.clip
        )

    def __call__(self, score: float) -> float:
        return float(self.predict(np.asarray([score]))[0])

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds.tolist(),
            "probabilities": self.probabilities.tolist(),
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(
            np.asarray(payload["thresholds"], dtype=float),
            np.asarray(payload["probabilities"], dtype=float),
            float(payload.get("clip", 1e-4)),
        )


def fit_isotonic_probability(scores, labels, weights=None, clip=1e-4):
    """Fit weighted Bernoulli probabilities by the PAV algorithm.

    This minimizes weighted squared probability error over all nondecreasing
    functions of the supplied score. Equal scores are aggregated before PAV,
    so the fitted mapping is invariant to input ordering.
    """
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) == 0:
        raise ValueError("scores and labels must be nonempty equal vectors")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("scores and labels must be finite")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("labels must lie in [0, 1]")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != x.shape or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and positive")
    order = np.argsort(x, kind="stable")
    x, y, w = x[order], y[order], w[order]
    unique, starts = np.unique(x, return_index=True)
    block_weight = np.add.reduceat(w, starts)
    block_sum = np.add.reduceat(w * y, starts)
    blocks = []
    for threshold, weight, total in zip(unique, block_weight, block_sum):
        blocks.append([float(threshold), float(weight), float(total)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[1] <= right[2] / right[1] + 1e-15:
                break
            blocks[-2:] = [[
                right[0], left[1] + right[1], left[2] + right[2]
            ]]
    thresholds = np.asarray([block[0] for block in blocks])
    probabilities = np.asarray([block[2] / block[1] for block in blocks])
    # Adjacent PAV blocks can have exactly the same fitted mean without
    # violating monotonicity. They define one identical step and are losslessly
    # represented by the last (inclusive upper) endpoint.
    compressed_thresholds = []
    compressed_probabilities = []
    for threshold, probability in zip(thresholds, probabilities):
        if (compressed_probabilities and
                abs(probability - compressed_probabilities[-1]) <= 1e-15):
            compressed_thresholds[-1] = threshold
        else:
            compressed_thresholds.append(threshold)
            compressed_probabilities.append(probability)
    return IsotonicProbabilityCalibrator(
        np.asarray(compressed_thresholds),
        np.asarray(compressed_probabilities), clip,
    )


def probability_metrics(probabilities, labels, bins=10) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probabilities.shape != labels.shape or probabilities.ndim != 1:
        raise ValueError("probabilities and labels must be equal vectors")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")
    brier = float(np.mean((probabilities - labels) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1,
                         bins - 1)
    indices = np.maximum(indices, 0)
    ece = 0.0
    maximum_error = 0.0
    for index in range(bins):
        selected = indices == index
        if not np.any(selected):
            continue
        error = abs(float(np.mean(probabilities[selected]) - np.mean(labels[selected])))
        ece += np.mean(selected) * error
        maximum_error = max(maximum_error, error)
    return {"brier_score": brier, "ece": float(ece),
            "maximum_calibration_error": float(maximum_error)}


@dataclass(frozen=True)
class ExcessPeakConformalNull:
    """Empirical GLRT null survival stratified by same-UAV excess peaks."""

    strata: dict[int, np.ndarray]

    def __post_init__(self):
        cleaned = {}
        for key, values in self.strata.items():
            array = np.sort(np.asarray(values, dtype=float))
            if int(key) not in (0, 1, 2) or array.ndim != 1:
                raise ValueError("strata keys must be 0, 1, or 2+")
            if np.any(~np.isfinite(array)):
                raise ValueError("null statistics must be finite")
            cleaned[int(key)] = array
        if set(cleaned) != {0, 1, 2} or not any(len(x) for x in cleaned.values()):
            raise ValueError("all excess-peak strata are required")
        object.__setattr__(self, "strata", cleaned)

    @staticmethod
    def stratum(candidate_count: int, distinct_views: int) -> int:
        return min(max(int(candidate_count) - int(distinct_views), 0), 2)

    def p_value(self, statistic: float, candidate_count: int,
                distinct_views: int) -> float:
        values = self.strata[self.stratum(candidate_count, distinct_views)]
        if len(values) == 0:
            values = np.sort(np.concatenate([
                array for array in self.strata.values() if len(array)
            ]))
        exceedances = len(values) - int(np.searchsorted(
            values, float(statistic), side="left"
        ))
        return float((1 + exceedances) / (len(values) + 1))

    def to_dict(self) -> dict:
        return {str(key): values.tolist() for key, values in self.strata.items()}
