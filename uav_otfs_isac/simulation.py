from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from .fusion import optimal_weights
from .models import DetectionMetrics, SelectionResult, TargetEvidenceModel


def evaluate_detection(
    models: Sequence[TargetEvidenceModel],
    selection: SelectionResult,
    false_alarm_rate: float,
    trials: int,
    rng: np.random.Generator,
) -> DetectionMetrics:
    pd = np.zeros(len(models)); pfa = np.zeros(len(models))
    for q, model in enumerate(models):
        scheduled = sorted(selection.scheduled[q] - {model.owner})
        h0_detect = 0; h1_detect = 0
        for _ in range(trials):
            received = {model.owner}
            for i in scheduled:
                if rng.random() < model.success_prob[i]:
                    received.add(i)
            idx = np.asarray(sorted(received), dtype=int)
            weights = optimal_weights(model.delta, model.sigma0, idx)
            mu_t0 = float(weights @ model.mu0[idx])
            var_t0 = float(weights @ model.sigma0[np.ix_(idx, idx)] @ weights)
            threshold = mu_t0 + np.sqrt(max(var_t0, 1e-12)) * norm.ppf(1.0 - false_alarm_rate)
            z0 = rng.multivariate_normal(model.mu0[idx], model.sigma0[np.ix_(idx, idx)])
            z1 = rng.multivariate_normal(model.mu1[idx], model.sigma1[np.ix_(idx, idx)])
            h0_detect += float(weights @ z0 > threshold)
            h1_detect += float(weights @ z1 > threshold)
        pfa[q] = h0_detect / trials
        pd[q] = h1_detect / trials
    return DetectionMetrics(
        pd_per_target=pd,
        pfa_per_target=pfa,
        mean_pd=float(pd.mean()),
        worst_pd=float(pd.min()),
        used_bits=selection.used_bits,
        selected_reports=sum(len(x) - 1 for x in selection.scheduled),
    )

