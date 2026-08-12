"""A NumPy-facing training and inference wrapper around the PyTorch model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import RobustScaler

from .config import ModelConfig, TrainingConfig
from .model import PriorAnchoredTemporalModel

ValidationData = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]


def _as_context(values: ArrayLike, name: str = "context") -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _as_vector(values: ArrayLike, n_rows: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(n_rows, float(array))
    else:
        array = np.ravel(array)
        if array.size == 1 and n_rows != 1:
            array = np.full(n_rows, float(array[0]))
    if array.shape != (n_rows,):
        raise ValueError(f"{name} must be scalar or have {n_rows} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PriorAnchoredRegressor:
    """Train and apply a prior-anchored temporal predictor.

    Scaling is fitted only on the training portion. A caller may pass explicit
    validation data to implement a domain-specific split; otherwise a seeded
    random validation split is used for optimization only.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.model_: Optional[PriorAnchoredTemporalModel] = None
        self.scaler_: Optional[RobustScaler] = None
        self.history_: dict[str, list] = {"train_loss": [], "validation_loss": []}
        self.context_dim_: Optional[int] = None
        self.is_fitted_ = False

    def _device(self) -> torch.device:
        requested = self.training_config.device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
        return torch.device(requested)

    @staticmethod
    def _tensor(values: NDArray[np.float64], device: torch.device) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def fit(
        self,
        context: ArrayLike,
        time: ArrayLike,
        target: ArrayLike,
        magnitude_prior: ArrayLike,
        timescale_prior: ArrayLike,
        *,
        validation_data: Optional[ValidationData] = None,
    ) -> PriorAnchoredRegressor:
        """Fit the model and restore the best validation checkpoint."""

        context_array = _as_context(context)
        n_rows = context_array.shape[0]
        if n_rows < 5:
            raise ValueError("at least five observations are required")
        time_array = _as_vector(time, n_rows, "time")
        target_array = _as_vector(target, n_rows, "target")
        magnitude_array = _as_vector(magnitude_prior, n_rows, "magnitude_prior")
        timescale_array = _as_vector(timescale_prior, n_rows, "timescale_prior")
        if np.any(time_array < 0.0):
            raise ValueError("time must be non-negative")
        if np.any(target_array < 0.0):
            raise ValueError("target must be non-negative")
        if np.any(magnitude_array <= 0.0) or np.any(timescale_array <= 0.0):
            raise ValueError("both prior arrays must be positive")

        _set_seed(self.training_config.seed)
        rng = np.random.default_rng(self.training_config.seed)
        if validation_data is None:
            indices = rng.permutation(n_rows)
            validation_size = max(1, int(round(n_rows * self.training_config.validation_fraction)))
            validation_indices = indices[:validation_size]
            training_indices = indices[validation_size:]
            arrays = (context_array, time_array, target_array, magnitude_array, timescale_array)
            train = tuple(array[training_indices] for array in arrays)
            validation = tuple(array[validation_indices] for array in arrays)
        else:
            train = (context_array, time_array, target_array, magnitude_array, timescale_array)
            validation_context = _as_context(validation_data[0], "validation context")
            n_validation = validation_context.shape[0]
            if validation_context.shape[1] != context_array.shape[1]:
                raise ValueError("training and validation context dimensions differ")
            validation = (
                validation_context,
                _as_vector(validation_data[1], n_validation, "validation time"),
                _as_vector(validation_data[2], n_validation, "validation target"),
                _as_vector(
                    validation_data[3], n_validation, "validation magnitude_prior"
                ),
                _as_vector(
                    validation_data[4], n_validation, "validation timescale_prior"
                ),
            )
            if np.any(validation[1] < 0.0):
                raise ValueError("validation time must be non-negative")
            if np.any(validation[2] < 0.0):
                raise ValueError("validation target must be non-negative")
            if np.any(validation[3] <= 0.0) or np.any(validation[4] <= 0.0):
                raise ValueError("validation prior arrays must be positive")

        train_context, train_time, train_target, train_magnitude, train_timescale = train
        val_context, val_time, val_target, val_magnitude, val_timescale = validation
        self.scaler_ = RobustScaler().fit(train_context)
        train_context_scaled = self.scaler_.transform(train_context)
        val_context_scaled = self.scaler_.transform(val_context)

        device = self._device()
        self.context_dim_ = context_array.shape[1]
        self.model_ = PriorAnchoredTemporalModel(self.context_dim_, self.model_config).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        train_tensors = tuple(
            self._tensor(array, device)
            for array in (
                train_context_scaled,
                train_time,
                train_target,
                train_magnitude,
                train_timescale,
            )
        )
        validation_tensors = tuple(
            self._tensor(array, device)
            for array in (
                val_context_scaled,
                val_time,
                val_target,
                val_magnitude,
                val_timescale,
            )
        )
        target_scale = max(float(np.std(train_target)), 1.0)
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.history_ = {"train_loss": [], "validation_loss": []}

        for epoch in range(self.training_config.epochs):
            self.model_.train()
            permutation = torch.randperm(len(train_context), device=device)
            batch_losses = []
            for start in range(0, len(train_context), self.training_config.batch_size):
                index = permutation[start : start + self.training_config.batch_size]
                batch_context, batch_time, batch_target, batch_magnitude, batch_timescale = (
                    tensor[index] for tensor in train_tensors
                )
                optimizer.zero_grad(set_to_none=True)
                prediction, details = self.model_(
                    batch_context,
                    batch_time,
                    batch_magnitude,
                    batch_timescale,
                    return_details=True,
                )
                data_loss = torch.mean(((prediction - batch_target) / target_scale) ** 2)
                regularization = (
                    torch.mean(details["magnitude_relative_correction"] ** 2)
                    + torch.mean(details["timescale_relative_correction"] ** 2)
                    + torch.mean(
                        (
                            details["magnitude_additive_correction"]
                            / max(self.model_config.additive_magnitude_scale, 1.0)
                        )
                        ** 2
                    )
                )
                loss = data_loss + self.training_config.prior_regularization * regularization
                if not torch.isfinite(loss):
                    raise FloatingPointError("training produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(), self.training_config.gradient_clip
                )
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            self.model_.eval()
            with torch.no_grad():
                val_prediction = self.model_(
                    validation_tensors[0],
                    validation_tensors[1],
                    validation_tensors[3],
                    validation_tensors[4],
                )
                validation_loss = torch.mean(
                    ((val_prediction - validation_tensors[2]) / target_scale) ** 2
                )
            validation_value = float(validation_loss.cpu())
            train_value = float(np.mean(batch_losses))
            self.history_["train_loss"].append(train_value)
            self.history_["validation_loss"].append(validation_value)
            if self.training_config.verbose and (
                epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == self.training_config.epochs
            ):
                print(
                    f"epoch={epoch + 1:04d} train_loss={train_value:.6f} "
                    f"validation_loss={validation_value:.6f}"
                )

            if validation_value < best_loss:
                best_loss = validation_value
                best_state = deepcopy(self.model_.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.training_config.patience:
                    break

        if best_state is None:
            raise RuntimeError("training did not produce a valid model checkpoint")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.is_fitted_ = True
        return self

    def _prepare_prediction(
        self,
        context: ArrayLike,
        time: ArrayLike,
        magnitude_prior: ArrayLike,
        timescale_prior: ArrayLike,
    ) -> tuple[torch.device, tuple[torch.Tensor, ...]]:
        if not self.is_fitted_ or self.model_ is None or self.scaler_ is None:
            raise RuntimeError("the regressor must be fitted before prediction")
        context_array = _as_context(context)
        if context_array.shape[1] != self.context_dim_:
            raise ValueError(
                f"context has {context_array.shape[1]} features; expected {self.context_dim_}"
            )
        n_rows = context_array.shape[0]
        arrays = (
            self.scaler_.transform(context_array),
            _as_vector(time, n_rows, "time"),
            _as_vector(magnitude_prior, n_rows, "magnitude_prior"),
            _as_vector(timescale_prior, n_rows, "timescale_prior"),
        )
        if np.any(arrays[1] < 0.0):
            raise ValueError("time must be non-negative")
        if np.any(arrays[2] <= 0.0) or np.any(arrays[3] <= 0.0):
            raise ValueError("both prior arrays must be positive")
        device = next(self.model_.parameters()).device
        return device, tuple(self._tensor(array, device) for array in arrays)

    def predict(
        self,
        context: ArrayLike,
        time: ArrayLike,
        magnitude_prior: ArrayLike,
        timescale_prior: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return non-negative point predictions."""

        _, tensors = self._prepare_prediction(
            context, time, magnitude_prior, timescale_prior
        )
        assert self.model_ is not None
        self.model_.eval()
        with torch.no_grad():
            prediction = self.model_(tensors[0], tensors[1], tensors[2], tensors[3])
        return prediction.detach().cpu().numpy().astype(float)

    def predict_details(
        self,
        context: ArrayLike,
        time: ArrayLike,
        magnitude_prior: ArrayLike,
        timescale_prior: ArrayLike,
    ) -> dict[str, NDArray[np.float64]]:
        """Return predictions and interpretable latent parameters."""

        _, tensors = self._prepare_prediction(
            context, time, magnitude_prior, timescale_prior
        )
        assert self.model_ is not None
        self.model_.eval()
        with torch.no_grad():
            prediction, details = self.model_(
                tensors[0], tensors[1], tensors[2], tensors[3], return_details=True
            )
        output = {
            name: value.detach().cpu().numpy().astype(float)
            for name, value in details.items()
        }
        output["prediction"] = prediction.detach().cpu().numpy().astype(float)
        return output

    def save(self, path: str | Path) -> None:
        """Save a fitted checkpoint. Load checkpoints only from trusted sources."""

        if not self.is_fitted_ or self.model_ is None or self.scaler_ is None:
            raise RuntimeError("only a fitted regressor can be saved")
        payload = {
            "format_version": 1,
            "package_version": "0.1.0",
            "context_dim": self.context_dim_,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "model_state": self.model_.state_dict(),
            "scaler_center": self.scaler_.center_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> PriorAnchoredRegressor:
        """Load a checkpoint created by :meth:`save` from a trusted source."""

        map_location = "cpu" if device == "auto" else device
        try:
            payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        except TypeError:  # Compatibility with older supported PyTorch releases.
            payload = torch.load(Path(path), map_location=map_location)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        training_values = dict(payload["training_config"])
        training_values["device"] = device
        estimator = cls(
            model_config=ModelConfig(**payload["model_config"]),
            training_config=TrainingConfig(**training_values),
        )
        estimator.context_dim_ = int(payload["context_dim"])
        estimator.scaler_ = RobustScaler()
        estimator.scaler_.center_ = np.asarray(payload["scaler_center"], dtype=float)
        estimator.scaler_.scale_ = np.asarray(payload["scaler_scale"], dtype=float)
        estimator.scaler_.n_features_in_ = estimator.context_dim_
        target_device = estimator._device()
        estimator.model_ = PriorAnchoredTemporalModel(
            estimator.context_dim_, estimator.model_config
        ).to(target_device)
        estimator.model_.load_state_dict(payload["model_state"])
        estimator.model_.eval()
        estimator.is_fitted_ = True
        return estimator
