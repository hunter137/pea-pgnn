"""Numerical audits for the point predictor's structural properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ConstraintReport:
    """Summary of a numerical trajectory-constraint audit."""

    nonnegative: bool
    monotonic: bool
    bounded: Optional[bool]
    minimum: float
    maximum: float
    largest_drop: float
    largest_upper_excess: Optional[float]

    @property
    def passed(self) -> bool:
        """Whether every requested constraint passed."""

        return self.nonnegative and self.monotonic and self.bounded is not False


def audit_trajectory(
    prediction: ArrayLike,
    *,
    magnitude: Optional[ArrayLike] = None,
    axis: int = -1,
    atol: float = 1e-6,
) -> ConstraintReport:
    """Audit non-negativity, monotonicity, and optional upper boundedness.

    Values must already be ordered by increasing time along ``axis``.
    """

    values = np.asarray(prediction, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("prediction must be a non-empty finite array")
    differences = np.diff(values, axis=axis)
    largest_drop = float(max(0.0, -float(np.min(differences)))) if differences.size else 0.0
    nonnegative = bool(np.all(values >= -atol))
    monotonic = bool(np.all(differences >= -atol))

    bounded: Optional[bool] = None
    largest_upper_excess: Optional[float] = None
    if magnitude is not None:
        upper = np.asarray(magnitude, dtype=float)
        if not np.all(np.isfinite(upper)) or np.any(upper <= 0.0):
            raise ValueError("magnitude must contain finite, positive values")
        try:
            excess = values - upper
        except ValueError as exc:
            raise ValueError("magnitude cannot be broadcast against prediction") from exc
        largest_upper_excess = float(max(0.0, float(np.max(excess))))
        bounded = bool(np.all(excess <= atol))

    return ConstraintReport(
        nonnegative=nonnegative,
        monotonic=monotonic,
        bounded=bounded,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        largest_drop=largest_drop,
        largest_upper_excess=largest_upper_excess,
    )
