"""Oracle ceiling: how well could a *perfect* model do on this data?

Turnover is a low-signal problem — most of the month-to-month outcome is
irreducible noise, and an AUC of 0.67 can be close to optimal. With real HR
data there is no way to know how much signal exists, so teams burn quarters
chasing accuracy that is not there.

Synthetic data with a known generating process removes that excuse. This
module recomputes each snapshot's *true* termination log-odds from the same
ground-truth coefficients the simulator used, then scores them against the
realised labels. The resulting AUC is (a close proxy for) the ceiling any
model can reach; compare your model against it with
:func:`evaluate_with_ceiling`.

Caveat: the oracle uses the hazard *at the snapshot month*, while a k-month
label integrates the hazard over the window (schedules change, raises land).
The proxy therefore slightly understates the exact ceiling, but it is within
a point or two — close enough to tell "weak model" from "weak signal".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import HOURLY_ROLES, GroundTruth


def oracle_log_odds(snapshots: pd.DataFrame, gt: GroundTruth | None = None) -> np.ndarray:
    """True monthly termination log-odds for each snapshot row."""
    gt = gt or GroundTruth()
    s = snapshots
    role = s["role"].astype(str).to_numpy()
    hourly = np.isin(role, HOURLY_ROLES)
    ten = s["tenure_months"].to_numpy()
    month_of_year = s["month_of_year"].astype(str).to_numpy()

    lo = np.array([gt.base_log_odds[r] for r in role])
    bucket = np.select([ten < 3, ten < 6, ten < 12, ten < 24],
                       ["0-2", "3-5", "6-11", "12-23"], "24+")
    lo += np.where(hourly,
                   np.array([gt.tenure_log_odds[b] for b in bucket]),
                   np.array([gt.tenure_log_odds_salaried[b] for b in bucket]))

    pc = s["pay_ratio"].to_numpy() - 1.0
    lo += np.where(hourly, gt.pay_ratio_hourly, gt.pay_ratio_salaried) * pc
    rating = s["performance_rating"].to_numpy()
    lo += (~hourly) * (rating >= 4) * (pc < -0.03) * gt.underpaid_high_performer_salaried
    lo += (rating <= 2) * np.where(hourly, gt.low_performance_hourly,
                                   gt.low_performance_salaried)

    vol = s["schedule_volatility_3m"].to_numpy()
    lo += hourly * gt.schedule_volatility_per_hour * np.maximum(vol - 3.0, 0)
    lo += hourly * gt.hours_gap_per_hour * s["hours_gap"].to_numpy()

    commute = s["commute_km"].to_numpy()
    lo += hourly * np.minimum(
        gt.commute_per_km_hourly * np.maximum(commute - gt.commute_threshold_km_hourly, 0),
        gt.commute_cap_hourly)
    lo += (~hourly) * np.minimum(
        gt.commute_per_km_salaried * np.maximum(commute - gt.commute_threshold_km_salaried, 0),
        gt.commute_cap_salaried)

    mgr_recent = s["months_since_mgr_change"].to_numpy() <= 3
    lo += mgr_recent * (role != "store_manager") * np.where(
        hourly, gt.manager_change_recent_hourly, gt.manager_change_recent_salaried)

    student = s["is_student"].to_numpy().astype(bool)
    lo += student * np.isin(month_of_year, ["8", "9"]) * gt.student_back_to_school
    lo += hourly * (month_of_year == "1") * gt.post_holiday_january_hourly

    lo += hourly * (s["store_staffing_ratio"].to_numpy() < 0.85) * gt.understaffed_store_hourly
    lo += ((~hourly) * gt.district_understaffing_salaried
           * s["district_understaffed_share"].to_numpy())

    since_promo = s["months_since_promotion"].to_numpy()
    promo_recent = (since_promo <= 6) & (since_promo < ten)
    lo += hourly * promo_recent * gt.recent_promotion_hourly
    lo += (~hourly) * (since_promo > 36) * gt.stagnation_salaried

    age = s["age_band"].astype(str).to_numpy()
    lo += (age == "16-20") * gt.age_16_20
    lo += (age == "50+") * gt.age_50_plus
    lo += s["second_job"].to_numpy() * gt.second_job
    return lo


def ceiling_auc(snapshots: pd.DataFrame, horizon: int,
                gt: GroundTruth | None = None) -> float:
    """AUC of the true hazard against realised labels — the signal ceiling."""
    rows = snapshots[snapshots[f"label_{horizon}m"].notna()]
    lo = oracle_log_odds(rows, gt)
    return float(roc_auc_score(rows[f"label_{horizon}m"], lo))


def evaluate_with_ceiling(model, test: pd.DataFrame,
                          gt: GroundTruth | None = None) -> pd.DataFrame:
    """Model metrics side by side with the oracle ceiling per horizon."""
    from .evaluation import evaluate_model

    metrics = evaluate_model(model, test)
    rows = test[test["role"].isin(model.roles)]
    metrics["ceiling_auc"] = [
        ceiling_auc(rows, h, gt) for h in model.horizons]
    metrics["signal_captured"] = ((metrics["roc_auc"] - 0.5)
                                  / (metrics["ceiling_auc"] - 0.5)).round(3)
    return metrics
