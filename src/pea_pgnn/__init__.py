"""PEA-PGNN public package interface."""

from .config import ModelConfig, TrainingConfig
from .constraints import ConstraintReport, audit_trajectory
from .estimator import PriorAnchoredRegressor
from .metrics import regression_metrics
from .model import PriorAnchoredTemporalModel
from .temporal import candidate_time_laws, convex_time_evolution

__all__ = [
    "ConstraintReport",
    "ModelConfig",
    "PriorAnchoredRegressor",
    "PriorAnchoredTemporalModel",
    "TrainingConfig",
    "audit_trajectory",
    "candidate_time_laws",
    "convex_time_evolution",
    "regression_metrics",
]

__version__ = "0.1.0"

