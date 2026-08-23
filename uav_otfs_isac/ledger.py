"""Unified bit-budget accountant (advice/011 P4 semantic freeze).

One denomination for every selector, gate, and ledger:

- ``model.report_bits[i]`` is the TRANSMITTED cost of report ``i``
  (``payload quantizer bits + 2`` packet overhead; the owner entry is 0)
  -- SYSTEM_MODEL sections 5/8.  Do NOT mix this with the toy exact-joint
  gates that price payload bits only.
- ``B_report = sum`` of ``report_bits`` over the scheduled non-owner
  reports.
- ``B_control = N_ris * phase_bits / coherence_frames`` (RIS control
  plane, amortized per frame); ``B_total = B_report + B_control`` is the
  accounting identity.
- Fractional control overhead must NOT be silently truncated out of the
  ledger: ``report_budget_from_total`` returns the residual side
  explicitly so ``B_total = B_report + overhead + residual`` holds.

``report_cost_bits``/``scheduled_report_bits`` are the single source of
truth for every selector cost so the mainline, replication, fairness, and
RIS ledgers stay byte-consistent with scenario.py's ``report_bits``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TargetEvidenceModel


def report_cost_bits(model: "TargetEvidenceModel", uav: int) -> int:
    """Transmitted report cost of one report (the owner is free)."""
    return int(model.report_bits[int(uav)])


def scheduled_report_bits(model: "TargetEvidenceModel", scheduled) -> int:
    """``B_report`` contribution of a scheduled report set of one target."""
    owner = int(model.owner)
    return sum(
        int(model.report_bits[int(i)]) for i in scheduled if int(i) != owner
    )


def ris_control_overhead_bits(
    num_elements: int,
    phase_bits: int | None,
    coherence_frames: int,
) -> float:
    """``B_control = N_ris * phase_bits / coherence_frames``."""
    if phase_bits is None or int(phase_bits) <= 0:
        return 0.0
    return float(num_elements) * int(phase_bits) / max(int(coherence_frames), 1)


def report_budget_from_total(
    total_budget_bits: float,
    control_overhead: float,
) -> tuple[int, float, float]:
    """Split ``B_total`` into (integerized report budget, residual bits,
    control-overhead bits) so ``B_total = B_report + B_control + residual``
    holds exactly; the fractional part of the overhead is never dropped."""
    overhead = max(float(control_overhead), 0.0)
    total = max(float(total_budget_bits), 0.0)
    report = int(max(total - overhead, 0.0))
    residual = max(total - overhead - report, 0.0)
    return report, float(residual), float(overhead)