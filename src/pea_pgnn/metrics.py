"""Small dependency-light regression metric utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def regression_metrics(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    mape_floor: float = 1.0,
) -> dict[str, float]:
    """Return R-squared, RMSE, MAE, MAPE, and sample count."""

    truth = np.ravel(np.asarray(target, dtype=float))
    estimate = np.ravel(np.asarray(prediction, dtype=float))
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("target and prediction must be non-empty arrays with equal shape")
    mask = np.isfinite(truth) & np.isfinite(estimate)
    if not np.any(mask):
        raise ValueError("target and prediction have no jointly finite observations")
    truth = truth[mask]
    estimate = estimate[mask]
    residual = truth - estimate
    residual_sum = float(np.sum(residual**2))
    total_sum = float(np.sum((truth - np.mean(truth)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else float("nan")
    return {
        "r2": r_squared,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mape": float(np.mean(np.abs(residual) / np.maximum(np.abs(truth), mape_floor)) * 100.0),
        "n": int(truth.size),
    }
