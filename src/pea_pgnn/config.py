"""Configuration objects for the model and its training loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and admissible-domain settings.

    The relative correction bounds must contain zero. They apply to the
    empirical magnitude and timescale anchors before absolute-domain clipping.
    """

    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.03
    magnitude_relative_bounds: tuple[float, float] = (-0.5, 1.2)
    timescale_relative_bounds: tuple[float, float] = (-0.8, 2.0)
    magnitude_bounds: tuple[float, float] = (100.0, 1500.0)
    timescale_bounds: tuple[float, float] = (5.0, 1000.0)
    additive_magnitude_scale: float = 200.0
    alpha_bounds: tuple[float, float] = (0.1, 0.9)

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name, bounds in (
            ("magnitude_relative_bounds", self.magnitude_relative_bounds),
            ("timescale_relative_bounds", self.timescale_relative_bounds),
        ):
            low, high = bounds
            if not low < 0.0 < high:
                raise ValueError(f"{name} must strictly contain zero")
        for name, bounds in (
            ("magnitude_bounds", self.magnitude_bounds),
            ("timescale_bounds", self.timescale_bounds),
            ("alpha_bounds", self.alpha_bounds),
        ):
            low, high = bounds
            if not 0.0 < low < high:
                raise ValueError(f"{name} must be positive and increasing")
        if self.additive_magnitude_scale < 0.0:
            raise ValueError("additive_magnitude_scale must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings for :class:`PriorAnchoredRegressor`."""

    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 75
    validation_fraction: float = 0.15
    prior_regularization: float = 0.01
    gradient_clip: float = 5.0
    seed: int = 42
    device: str = "auto"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid learning rate or weight decay")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.prior_regularization < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("regularization must be non-negative and gradient_clip positive")
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device string")
