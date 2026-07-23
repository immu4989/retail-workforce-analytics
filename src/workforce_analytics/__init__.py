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
from .demand import (
    LaborDemandForecaster,
    TrafficConfig,
    TrafficSimulator,
    build_week_schedule,
    required_staff,
    schedule_stability,
)
from .absence import (
    AbsenceGroundTruth,
    CalloutModel,
    build_callout_panel,
    reserve_staffing_plan,
    simulate_absences,
)
from .funnel import FunnelGroundTruth, funnel_report, req_timing, simulate_funnel
from .mobility import (
    PromotionModel,
    bench_strength,
    build_promotion_panel,
    promotion_events,
)
from .contagion import contagion_analysis, peer_exit_exposure
from .config import WageProgram
from .compensation import event_study, pay_elasticity, pay_shift, run_wage_experiment
from .callcenter import (
    CallCenterGroundTruth,
    CallTopicModel,
    operational_linkage,
    simulate_calls,
    topic_trends,
)
from .realdata import (
    Finding,
    MessyExtract,
    ValidationReport,
    audit_split,
    make_messy_extract,
    validate_person_months,
)
from .elasticity import (
    ServiceConfig,
    derive_understaffing_cost,
    service_loss_curve,
    service_outcome,
    staffing_sales_elasticity,
)

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
    "LaborDemandForecaster",
    "TrafficConfig",
    "TrafficSimulator",
    "build_week_schedule",
    "required_staff",
    "schedule_stability",
    "AbsenceGroundTruth",
    "CalloutModel",
    "build_callout_panel",
    "reserve_staffing_plan",
    "simulate_absences",
    "FunnelGroundTruth",
    "funnel_report",
    "req_timing",
    "simulate_funnel",
    "PromotionModel",
    "bench_strength",
    "build_promotion_panel",
    "promotion_events",
    "contagion_analysis",
    "peer_exit_exposure",
    "WageProgram",
    "event_study",
    "pay_elasticity",
    "pay_shift",
    "run_wage_experiment",
    "CallCenterGroundTruth",
    "CallTopicModel",
    "operational_linkage",
    "simulate_calls",
    "topic_trends",
    "Finding",
    "MessyExtract",
    "ValidationReport",
    "audit_split",
    "make_messy_extract",
    "validate_person_months",
    "ServiceConfig",
    "derive_understaffing_cost",
    "service_loss_curve",
    "service_outcome",
    "staffing_sales_elasticity",
]
