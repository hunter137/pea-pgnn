"""Concrete drying-shrinkage empirical priors used by the paper implementation.

The functions reproduce the algebra and microstrain conversion used in the
research code. They are empirical utilities, not substitutes for checking the
scope, units, or applicability of the underlying design formulations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _broadcast_validated(**values: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    names = tuple(values)
    arrays = np.broadcast_arrays(*(np.asarray(values[name], dtype=float) for name in names))
    for name, array in zip(names, arrays):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values")
    return tuple(arrays)


def _safe_microstrain(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    return np.clip(
        np.nan_to_num(np.abs(array), nan=0.0, posinf=3000.0, neginf=0.0),
        0.0,
        3000.0,
    )


def _validate_conditions(
    loading_age: NDArray[np.float64],
    relative_humidity: NDArray[np.float64],
    volume_surface_ratio: NDArray[np.float64],
    compressive_strength: NDArray[np.float64],
    water_content: NDArray[np.float64] | None = None,
) -> None:
    if np.any(loading_age <= 0.0):
        raise ValueError("loading_age must be positive")
    if np.any((relative_humidity < 0.0) | (relative_humidity > 100.0)):
        raise ValueError("relative_humidity must be in [0, 100]")
    if np.any(volume_surface_ratio <= 0.0):
        raise ValueError("volume_surface_ratio must be positive")
    if np.any(compressive_strength <= 0.0):
        raise ValueError("compressive_strength must be positive")
    if water_content is not None and np.any(water_content <= 0.0):
        raise ValueError("water_content must be positive")


def b3_timescale(
    loading_age: ArrayLike,
    volume_surface_ratio: ArrayLike,
    compressive_strength: ArrayLike,
) -> NDArray[np.float64]:
    """Return the B3-inspired characteristic time in days."""

    loading_age, volume_surface_ratio, compressive_strength = _broadcast_validated(
        loading_age=loading_age,
        volume_surface_ratio=volume_surface_ratio,
        compressive_strength=compressive_strength,
    )
    dummy_humidity = np.full_like(loading_age, 50.0)
    _validate_conditions(
        loading_age,
        dummy_humidity,
        volume_surface_ratio,
        compressive_strength,
    )
    return np.maximum(
        0.085
        * loading_age ** (-0.08)
        * compressive_strength ** (-0.25)
        * (2.0 * volume_surface_ratio) ** 2,
        1.0,
    )


def b3_ultimate_shrinkage(
    loading_age: ArrayLike,
    relative_humidity: ArrayLike,
    volume_surface_ratio: ArrayLike,
    water_content: ArrayLike,
    compressive_strength: ArrayLike,
) -> NDArray[np.float64]:
    """Return the B3-inspired ultimate drying shrinkage in microstrain."""

    loading_age, relative_humidity, volume_surface_ratio, water_content, strength = (
        _broadcast_validated(
            loading_age=loading_age,
            relative_humidity=relative_humidity,
            volume_surface_ratio=volume_surface_ratio,
            water_content=water_content,
            compressive_strength=compressive_strength,
        )
    )
    _validate_conditions(
        loading_age, relative_humidity, volume_surface_ratio, strength, water_content
    )
    humidity = relative_humidity / 100.0
    ultimate = (0.019 * water_content**2.1 * strength ** (-0.28) + 270.0) * 1e-6
    timescale = b3_timescale(loading_age, volume_surface_ratio, strength)
    shifted_time = np.clip(loading_age + timescale, 1.0, 1e6)
    correction = np.clip(
        np.sqrt((607.0 * (4.0 + 0.85 * shifted_time)) / (shifted_time * (4.0 + 0.85 * 607.0))),
        0.5,
        2.0,
    )
    return _safe_microstrain(ultimate * correction * (1.0 - humidity**3) * 1e6)


def gl2000_ultimate_shrinkage(
    relative_humidity: ArrayLike,
    volume_surface_ratio: ArrayLike,
    compressive_strength: ArrayLike,
) -> NDArray[np.float64]:
    """Return the GL2000-inspired ultimate drying shrinkage in microstrain."""

    relative_humidity, volume_surface_ratio, strength = _broadcast_validated(
        relative_humidity=relative_humidity,
        volume_surface_ratio=volume_surface_ratio,
        compressive_strength=compressive_strength,
    )
    dummy_loading_age = np.ones_like(relative_humidity)
    _validate_conditions(
        dummy_loading_age, relative_humidity, volume_surface_ratio, strength
    )
    humidity = relative_humidity / 100.0
    return _safe_microstrain(
        900.0 * np.sqrt(30.0 / strength) * (1.0 - 1.18 * humidity**4)
    )


def aci209_ultimate_shrinkage(
    relative_humidity: ArrayLike,
    volume_surface_ratio: ArrayLike,
) -> NDArray[np.float64]:
    """Return the ACI209-inspired ultimate drying shrinkage in microstrain."""

    relative_humidity, volume_surface_ratio = _broadcast_validated(
        relative_humidity=relative_humidity,
        volume_surface_ratio=volume_surface_ratio,
    )
    if np.any((relative_humidity < 0.0) | (relative_humidity > 100.0)):
        raise ValueError("relative_humidity must be in [0, 100]")
    if np.any(volume_surface_ratio <= 0.0):
        raise ValueError("volume_surface_ratio must be positive")
    humidity_factor = np.where(
        relative_humidity <= 80.0,
        1.4 - 0.01 * relative_humidity,
        3.0 - 0.03 * relative_humidity,
    )
    return _safe_microstrain(
        780.0
        * np.maximum(humidity_factor, 0.01)
        * 1.2
        * np.exp(-0.0047 * volume_surface_ratio)
    )


def b3_shrinkage(
    time: ArrayLike,
    loading_age: ArrayLike,
    relative_humidity: ArrayLike,
    volume_surface_ratio: ArrayLike,
    water_content: ArrayLike,
    compressive_strength: ArrayLike,
) -> NDArray[np.float64]:
    """Return the B3-inspired shrinkage trajectory in microstrain."""

    time, loading_age, humidity, ratio, water, strength = _broadcast_validated(
        time=time,
        loading_age=loading_age,
        relative_humidity=relative_humidity,
        volume_surface_ratio=volume_surface_ratio,
        water_content=water_content,
        compressive_strength=compressive_strength,
    )
    if np.any(time < 0.0):
        raise ValueError("time must be non-negative")
    ultimate = b3_ultimate_shrinkage(loading_age, humidity, ratio, water, strength)
    timescale = b3_timescale(loading_age, ratio, strength)
    return _safe_microstrain(ultimate * np.tanh(np.sqrt(time / timescale)))


def gl2000_shrinkage(
    time: ArrayLike,
    relative_humidity: ArrayLike,
    volume_surface_ratio: ArrayLike,
    compressive_strength: ArrayLike,
) -> NDArray[np.float64]:
    """Return the GL2000-inspired shrinkage trajectory in microstrain."""

    time, humidity, ratio, strength = _broadcast_validated(
        time=time,
        relative_humidity=relative_humidity,
        volume_surface_ratio=volume_surface_ratio,
        compressive_strength=compressive_strength,
    )
    if np.any(time < 0.0):
        raise ValueError("time must be non-negative")
    ultimate = gl2000_ultimate_shrinkage(humidity, ratio, strength)
    return _safe_microstrain(ultimate * np.sqrt(time / (time + 0.15 * ratio**2)))


def aci209_shrinkage(
    time: ArrayLike,
    relative_humidity: ArrayLike,
    volume_surface_ratio: ArrayLike,
) -> NDArray[np.float64]:
    """Return the ACI209-inspired shrinkage trajectory in microstrain."""

    time, humidity, ratio = _broadcast_validated(
        time=time,
        relative_humidity=relative_humidity,
        volume_surface_ratio=volume_surface_ratio,
    )
    if np.any(time < 0.0):
        raise ValueError("time must be non-negative")
    ultimate = aci209_ultimate_shrinkage(humidity, ratio)
    return _safe_microstrain(ultimate * time / (35.0 + time))


def concrete_prior_anchors(
    loading_age: ArrayLike,
    relative_humidity: ArrayLike,
    volume_surface_ratio: ArrayLike,
    water_content: ArrayLike,
    compressive_strength: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Build the paper implementation's magnitude and timescale anchors.

    Magnitude is the arithmetic mean of the B3-, GL2000-, and ACI209-inspired
    ultimate estimates. Timescale is the B3-inspired estimate.
    """

    b3_value = b3_ultimate_shrinkage(
        loading_age,
        relative_humidity,
        volume_surface_ratio,
        water_content,
        compressive_strength,
    )
    gl_value = gl2000_ultimate_shrinkage(
        relative_humidity, volume_surface_ratio, compressive_strength
    )
    aci_value = aci209_ultimate_shrinkage(relative_humidity, volume_surface_ratio)
    timescale = b3_timescale(loading_age, volume_surface_ratio, compressive_strength)
    return {
        "magnitude": (b3_value + gl_value + aci_value) / 3.0,
        "timescale": np.clip(timescale, 1.0, 5000.0),
        "b3_magnitude": b3_value,
        "gl2000_magnitude": gl_value,
        "aci209_magnitude": aci_value,
    }


__all__ = [
    "aci209_shrinkage",
    "aci209_ultimate_shrinkage",
    "b3_shrinkage",
    "b3_timescale",
    "b3_ultimate_shrinkage",
    "concrete_prior_anchors",
    "gl2000_shrinkage",
    "gl2000_ultimate_shrinkage",
]
