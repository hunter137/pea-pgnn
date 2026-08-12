"""Normalized, structure-preserving candidate laws for temporal evolution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

N_BASIS = 4


def candidate_time_laws(
    time: ArrayLike,
    timescale: ArrayLike,
    alpha: ArrayLike = 0.5,
) -> NDArray[np.float64]:
    """Evaluate the four normalized candidate time laws.

    Parameters
    ----------
    time:
        Non-negative query time.
    timescale:
        Strictly positive characteristic timescale.
    alpha:
        Strictly positive exponent used by the rational-power law.

    Returns
    -------
    numpy.ndarray
        Broadcast input shape followed by a final basis dimension of length 4.
        Each value lies in ``[0, 1]`` and is non-decreasing with time for fixed
        positive parameters.
    """

    time_array, timescale_array, alpha_array = np.broadcast_arrays(
        np.asarray(time, dtype=float),
        np.asarray(timescale, dtype=float),
        np.asarray(alpha, dtype=float),
    )
    if not np.all(np.isfinite(time_array)) or np.any(time_array < 0.0):
        raise ValueError("time must contain finite, non-negative values")
    if not np.all(np.isfinite(timescale_array)) or np.any(timescale_array <= 0.0):
        raise ValueError("timescale must contain finite, positive values")
    if not np.all(np.isfinite(alpha_array)) or np.any(alpha_array <= 0.0):
        raise ValueError("alpha must contain finite, positive values")

    ratio = time_array / timescale_array
    law_1 = np.tanh(np.sqrt(ratio))
    law_2 = (time_array / (time_array + timescale_array)) ** alpha_array
    law_3 = np.sqrt(time_array / (time_array + timescale_array**2 / 100.0))
    law_4 = np.clip(np.log1p(ratio) / np.log1p(1.0e4), 0.0, 1.0)
    return np.stack((law_1, law_2, law_3, law_4), axis=-1)


def convex_time_evolution(
    time: ArrayLike,
    timescale: ArrayLike,
    weights: Sequence[float] | NDArray[np.float64],
    alpha: ArrayLike = 0.5,
    *,
    atol: float = 1e-7,
) -> NDArray[np.float64]:
    """Combine candidate laws using non-negative weights that sum to one."""

    laws = candidate_time_laws(time=time, timescale=timescale, alpha=alpha)
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim == 0 or weight_array.shape[-1] != N_BASIS:
        raise ValueError(f"weights must have a final dimension of length {N_BASIS}")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.allclose(weight_array.sum(axis=-1), 1.0, atol=atol, rtol=0.0):
        raise ValueError("weights must sum to one along the final dimension")
    try:
        return np.sum(laws * weight_array, axis=-1)
    except ValueError as exc:
        raise ValueError("weights cannot be broadcast against the candidate laws") from exc
