"""Low-level PyTorch implementation of the prior-anchored predictor."""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor, nn

from .config import ModelConfig


def _zero_correction_bias(bounds: tuple[float, float]) -> float:
    """Return the sigmoid-logit bias that maps asymmetric bounds to zero."""

    low, high = bounds
    probability = -low / (high - low)
    return math.log(probability / (1.0 - probability))


def _bounded_sigmoid(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
    low, high = bounds
    return low + (high - low) * torch.sigmoid(raw)


def _as_batch_vector(value: Tensor, batch_size: int, name: str) -> Tensor:
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim != 1:
        raise ValueError(f"{name} must be a scalar, (n,), or (n, 1) tensor")
    if value.numel() == 1 and batch_size != 1:
        value = value.expand(batch_size)
    if value.shape[0] != batch_size:
        raise ValueError(f"{name} has {value.shape[0]} rows; expected {batch_size}")
    return value


class ResidualBlock(nn.Module):
    """Linear-normalized GELU block with a learned residual projection."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")

    def forward(self, inputs: Tensor) -> Tensor:
        transformed = self.dropout(self.activation(self.normalization(self.linear(inputs))))
        return transformed + self.residual(inputs)


class PriorAnchoredTemporalModel(nn.Module):
    """Prior-anchored structured predictor for a non-decreasing response.

    Context is intentionally separated from query time. For a fixed context,
    the learned magnitude, timescale, exponent, and mixture weights therefore
    remain constant along the queried trajectory.
    """

    basis_names = ("tanh_sqrt", "rational_power", "sqrt_rational", "logarithmic")

    def __init__(self, context_dim: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.context_dim = int(context_dim)
        self.config = config or ModelConfig()

        layers = []
        current_dim = self.context_dim
        for width in self.config.hidden_dims:
            layers.append(ResidualBlock(current_dim, width, self.config.dropout))
            current_dim = width
        self.backbone = nn.Sequential(*layers)

        self.magnitude_relative_head = nn.Linear(current_dim, 1)
        self.magnitude_additive_head = nn.Linear(current_dim, 1)
        self.timescale_relative_head = nn.Linear(current_dim, 1)
        self.alpha_head = nn.Linear(current_dim, 1)
        self.weight_head = nn.Linear(current_dim, len(self.basis_names))

        correction_heads = (
            self.magnitude_relative_head,
            self.magnitude_additive_head,
            self.timescale_relative_head,
        )
        for head in correction_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(
            self.magnitude_relative_head.bias,
            _zero_correction_bias(self.config.magnitude_relative_bounds),
        )
        nn.init.constant_(
            self.timescale_relative_head.bias,
            _zero_correction_bias(self.config.timescale_relative_bounds),
        )
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    @staticmethod
    def candidate_laws(time: Tensor, timescale: Tensor, alpha: Tensor) -> Tensor:
        """Evaluate the four normalized candidate laws as a ``(n, 4)`` tensor."""

        if torch.any(time < 0.0):
            raise ValueError("time must be non-negative")
        if torch.any(timescale <= 0.0) or torch.any(alpha <= 0.0):
            raise ValueError("timescale and alpha must be positive")
        ratio = time / timescale
        law_1 = torch.tanh(torch.sqrt(ratio))
        law_2 = (time / (time + timescale)).pow(alpha)
        law_3 = torch.sqrt(time / (time + timescale.square() / 100.0))
        denominator = torch.log1p(time.new_tensor(1.0e4))
        law_4 = torch.clamp(torch.log1p(ratio) / denominator, 0.0, 1.0)
        return torch.stack((law_1, law_2, law_3, law_4), dim=-1)

    def forward(
        self,
        context: Tensor,
        time: Tensor,
        magnitude_prior: Tensor,
        timescale_prior: Tensor,
        *,
        return_details: bool = False,
    ) -> Union[Tensor, tuple[Tensor, dict[str, Tensor]]]:
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"context must have shape (n, {self.context_dim}); got {tuple(context.shape)}"
            )
        batch_size = context.shape[0]
        time = _as_batch_vector(time, batch_size, "time")
        magnitude_prior = _as_batch_vector(magnitude_prior, batch_size, "magnitude_prior")
        timescale_prior = _as_batch_vector(timescale_prior, batch_size, "timescale_prior")
        if not torch.isfinite(context).all():
            raise ValueError("context must contain finite values")
        if not torch.isfinite(time).all() or torch.any(time < 0.0):
            raise ValueError("time must contain finite, non-negative values")
        if not torch.isfinite(magnitude_prior).all() or torch.any(magnitude_prior <= 0.0):
            raise ValueError("magnitude_prior must contain finite, positive values")
        if not torch.isfinite(timescale_prior).all() or torch.any(timescale_prior <= 0.0):
            raise ValueError("timescale_prior must contain finite, positive values")

        hidden = self.backbone(context)
        magnitude_relative = _bounded_sigmoid(
            self.magnitude_relative_head(hidden).squeeze(-1),
            self.config.magnitude_relative_bounds,
        )
        magnitude_additive = (
            torch.tanh(self.magnitude_additive_head(hidden).squeeze(-1))
            * self.config.additive_magnitude_scale
        )
        magnitude_low, magnitude_high = self.config.magnitude_bounds
        magnitude = torch.clamp(
            magnitude_prior * (1.0 + magnitude_relative) + magnitude_additive,
            magnitude_low,
            magnitude_high,
        )

        timescale_relative = _bounded_sigmoid(
            self.timescale_relative_head(hidden).squeeze(-1),
            self.config.timescale_relative_bounds,
        )
        timescale_low, timescale_high = self.config.timescale_bounds
        timescale = torch.clamp(
            timescale_prior * (1.0 + timescale_relative),
            timescale_low,
            timescale_high,
        )

        alpha_low, alpha_high = self.config.alpha_bounds
        alpha = alpha_low + (alpha_high - alpha_low) * torch.sigmoid(
            self.alpha_head(hidden).squeeze(-1)
        )
        weights = torch.softmax(self.weight_head(hidden), dim=-1)
        laws = self.candidate_laws(time, timescale, alpha)
        evolution = torch.sum(weights * laws, dim=-1)
        prediction = magnitude * evolution

        if not return_details:
            return prediction
        details = {
            "magnitude": magnitude,
            "timescale": timescale,
            "alpha": alpha,
            "weights": weights,
            "candidate_laws": laws,
            "evolution": evolution,
            "magnitude_relative_correction": magnitude_relative,
            "magnitude_additive_correction": magnitude_additive,
            "timescale_relative_correction": timescale_relative,
        }
        return prediction, details

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
