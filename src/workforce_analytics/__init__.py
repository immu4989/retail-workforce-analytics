"""Turnover prediction, headcount forecasting and attrition-driver analysis
for hourly retail workforces, with a fully synthetic data generator whose
ground truth is known."""

from .config import GroundTruth, SimulationConfig
from .generator import WorkforceSimulator, generate
from .snapshots import ALL_FEATURES, build_snapshots, time_split
from .turnover import TurnoverModel
from .evaluation import evaluate_model, evaluate_horizon
from .oracle import ceiling_auc, evaluate_with_ceiling, oracle_log_odds
from .headcount import build_hiring_plan, validate_expected_attrition
from .drivers import (
    InterventionSimulator,
    driver_importance,
    ground_truth_comparison,
    partial_dependence_curve,
    raise_pay_floor,
    stabilize_schedules,
    close_hours_gap,
)
from .cost_model import CostModel, targeting_roi, turnover_cost_summary
from .explain import reason_codes, shap_importance, shap_matrix
from .survival_nn import SurvivalNN

__version__ = "0.1.0"

__all__ = [
    "GroundTruth",
    "SimulationConfig",
    "WorkforceSimulator",
    "generate",
    "ALL_FEATURES",
    "build_snapshots",
    "time_split",
    "TurnoverModel",
    "evaluate_model",
    "evaluate_horizon",
    "ceiling_auc",
    "evaluate_with_ceiling",
    "oracle_log_odds",
    "build_hiring_plan",
    "validate_expected_attrition",
    "InterventionSimulator",
    "driver_importance",
    "ground_truth_comparison",
    "partial_dependence_curve",
    "raise_pay_floor",
    "stabilize_schedules",
    "close_hours_gap",
    "CostModel",
    "targeting_roi",
    "turnover_cost_summary",
    "reason_codes",
    "shap_importance",
    "shap_matrix",
    "SurvivalNN",
]
