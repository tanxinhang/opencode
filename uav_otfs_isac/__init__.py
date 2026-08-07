"""Distributed UAV-OTFS-ISAC selective fusion simulator."""

from .config import ExperimentConfig, load_config
from .expected_pd import (
    expected_gaussian_detection_probability,
    expected_pd_greedy_select,
)
from .exact_quota_selection import exact_budget_select, exact_maxmin_select
from .fusion import (
    gaussian_pd_closed_form,
    optimal_deflection,
    optimal_gaussian_detection_probability,
    optimal_gaussian_weights,
    optimal_weights,
    pd_shift_upper_bound,
)
from .selection import greedy_select
from .models import ExpectedPdSelectionResult
from .scalable_selection import minimum_cost_to_threshold, scaled_maxmin_select

__all__ = [
    "ExperimentConfig",
    "load_config",
    "ExpectedPdSelectionResult",
    "expected_gaussian_detection_probability",
    "expected_pd_greedy_select",
    "exact_budget_select",
    "exact_maxmin_select",
    "minimum_cost_to_threshold",
    "scaled_maxmin_select",
    "gaussian_pd_closed_form",
    "optimal_deflection",
    "optimal_gaussian_detection_probability",
    "optimal_gaussian_weights",
    "optimal_weights",
    "pd_shift_upper_bound",
    "greedy_select",
]
