"""PEA-PGNN public package interface."""

from ._version import __version__
from .config import ModelConfig, TrainingConfig
from .constraints import ConstraintReport, audit_trajectory
from .estimator import PriorAnchoredRegressor
from .metrics import regression_metrics
from .model import PriorAnchoredTemporalModel
from .splitting import GroupedSplit, grouped_train_validation_test_split
from .temporal import candidate_time_laws, convex_time_evolution

__all__ = [
    "ConstraintReport",
    "GroupedSplit",
    "ModelConfig",
    "PriorAnchoredRegressor",
    "PriorAnchoredTemporalModel",
    "TrainingConfig",
    "audit_trajectory",
    "candidate_time_laws",
    "convex_time_evolution",
    "grouped_train_validation_test_split",
    "regression_metrics",
    "__version__",
]
