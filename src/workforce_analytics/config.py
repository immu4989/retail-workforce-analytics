"""Configuration and ground-truth parameters for the workforce simulator.

The simulator draws monthly termination events from a discrete-time hazard
model. Every coefficient that shapes the hazard is declared here, in one
place, so that downstream driver-analysis results can be checked against the
data-generating process. ``GroundTruth.as_dict()`` serialises the true effects
alongside any generated dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

HOURLY_ROLES = ("barista", "shift_supervisor")
SALARIED_ROLES = ("assistant_store_manager", "store_manager")
ALL_ROLES = HOURLY_ROLES + SALARIED_ROLES

AGE_BANDS = ("16-20", "21-25", "26-35", "36-50", "50+")


@dataclass
class GroundTruth:
    """True log-odds effects used by the simulator's monthly hazard model.

    Positive values increase the chance an employee leaves in a given month.
    Units are noted per field; continuous effects are applied to centred or
    thresholded versions of the raw feature as documented in
    ``generator._monthly_hazard``.
    """

    # Baseline monthly termination log-odds by role (before tenure decay).
    base_log_odds: dict[str, float] = field(default_factory=lambda: {
        "barista": -2.70,
        "shift_supervisor": -2.90,
        "assistant_store_manager": -3.90,
        "store_manager": -4.25,
    })

    # Additive log-odds by tenure bucket (new-hire washout, then settling).
    tenure_log_odds: dict[str, float] = field(default_factory=lambda: {
        "0-2": 0.00,
        "3-5": -0.35,
        "6-11": -0.60,
        "12-23": -0.95,
        "24+": -1.25,
    })
    # Salaried tenure effects are milder than hourly ones.
    tenure_log_odds_salaried: dict[str, float] = field(default_factory=lambda: {
        "0-2": 0.40,
        "3-5": 0.25,
        "6-11": 0.10,
        "12-23": -0.10,
        "24+": -0.30,
    })

    # Compensation: applied to (pay_ratio - 1), i.e. pay relative to the
    # local market median for the role. Being 10% under market adds
    # 0.25 (hourly) / 0.35 (salaried) to the log-odds.
    pay_ratio_hourly: float = -2.5
    pay_ratio_salaried: float = -3.5
    # High performers who are paid under market get poached.
    underpaid_high_performer_salaried: float = 0.50

    # Scheduling (hourly only). Volatility is the std-dev of weekly scheduled
    # hours; the effect applies per hour of std-dev above 3.
    schedule_volatility_per_hour: float = 0.12
    # Underemployment: per desired-minus-scheduled weekly hour (floored at 0).
    hours_gap_per_hour: float = 0.05

    # Commute distance, per km beyond the threshold, capped.
    commute_per_km_hourly: float = 0.04
    commute_threshold_km_hourly: float = 5.0
    commute_cap_hourly: float = 0.60
    commute_per_km_salaried: float = 0.02
    commute_threshold_km_salaried: float = 10.0
    commute_cap_salaried: float = 0.40

    # Management stability.
    manager_change_recent_hourly: float = 0.30      # store manager changed <=3 months ago
    manager_change_recent_salaried: float = 0.25

    # Seasonality.
    student_back_to_school: float = 0.90            # students in Aug/Sep
    post_holiday_january_hourly: float = 0.25

    # Store operations: staffed below 85% of target.
    understaffed_store_hourly: float = 0.25
    district_understaffing_salaried: float = 1.00   # x share of understaffed stores

    # Performance (mix of voluntary and managed exits).
    low_performance_hourly: float = 0.50            # rating <= 2
    low_performance_salaried: float = 0.60

    # Growth and recognition.
    recent_promotion_hourly: float = -0.60          # promoted <=6 months ago
    stagnation_salaried: float = 0.50               # >36 months since promotion/hire

    # Demographics and life context.
    age_16_20: float = 0.25
    age_50_plus: float = -0.30
    second_job: float = 0.15

    # Share of low-performer terminations that are involuntary.
    involuntary_share_low_perf_hourly: float = 0.40
    involuntary_share_low_perf_salaried: float = 0.60

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationConfig:
    """Size, horizon and labour-market settings for a simulated company."""

    n_districts: int = 12
    stores_per_district_min: int = 8
    stores_per_district_max: int = 14
    n_months: int = 60                  # month 0 = start_year-01
    start_year: int = 2021
    seed: int = 7

    # Store composition: probability of volume tier 1/2/3 and the staffing
    # targets for each tier.
    tier_probs: tuple[float, float, float] = (0.30, 0.45, 0.25)
    target_baristas: dict[int, int] = field(default_factory=lambda: {1: 10, 2: 14, 3: 19})
    target_shift_supervisors: dict[int, int] = field(default_factory=lambda: {1: 3, 2: 4, 3: 5})
    # Tier 2-3 stores carry an assistant store manager; every store has one SM.

    # Districts that keep opening new stores (growth markets).
    n_growth_districts: int = 3
    new_stores_per_growth_district: int = 3

    # Market pay medians (hourly $/hr, salaried $k/yr) before district index.
    market_pay: dict[str, float] = field(default_factory=lambda: {
        "barista": 16.0,
        "shift_supervisor": 20.0,
        "assistant_store_manager": 55.0,
        "store_manager": 70.0,
    })
    market_drift_monthly: float = 0.0025    # market pay rises ~3%/yr
    merit_raise_mean: float = 0.030          # annual merit increase
    merit_raise_sd: float = 0.010

    ground_truth: GroundTruth = field(default_factory=GroundTruth)
