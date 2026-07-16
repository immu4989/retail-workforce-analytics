"""Use case 6: call-out (unplanned absence) prediction and reserve staffing.

A call-out is an unplanned absence from a scheduled shift: the 6am text that
leaves a store two people down through the morning rush. QSR operators run
5-10% shift no-show rates, and the standard mitigation — predict expected
absences, then staff reserve/float capacity against them — is a
predict-then-optimize pattern from the rostering literature.

Same discipline as the rest of the repo: absences are simulated from a
published ground-truth rate model (:class:`AbsenceGroundTruth`), the
predictor is evaluated out-of-time against a base-rate yardstick, and the
reserve plan is sized from the predicted distribution, not the point
estimate.

The absence process is deliberately independent of the termination hazard
(no feedback loop in v1); the docs note this simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import HOURLY_ROLES
from .snapshots import ALL_FEATURES

FLU_MONTHS = {12, 1, 2}
EXAM_MONTHS = {5, 12}


@dataclass
class AbsenceGroundTruth:
    """True log-rate effects for monthly call-out counts (Poisson)."""

    base_log_rate_hourly: float = -0.75      # ~0.47 call-outs/month
    base_log_rate_salaried: float = -2.30
    second_job: float = 0.35
    volatility_per_hour: float = 0.06        # per hour of std-dev above 3
    commute_per_km: float = 0.02             # per km beyond 5, capped
    commute_cap: float = 0.40
    new_hire: float = 0.25                   # tenure < 3 months
    student: float = 0.15
    student_exam_season: float = 0.40        # May / December
    flu_season: float = 0.30                 # December - February
    low_performance: float = 0.30            # rating <= 2

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def simulate_absences(person_months: pd.DataFrame,
                      gt: AbsenceGroundTruth | None = None,
                      seed: int = 23) -> pd.DataFrame:
    """Monthly call-out counts per active employee-month."""
    gt = gt or AbsenceGroundTruth()
    rng = np.random.default_rng(seed)
    pm = person_months
    hourly = pm["role"].isin(HOURLY_ROLES).to_numpy()
    month_of_year = (pm["month"] % 12 + 1).to_numpy()

    log_rate = np.where(hourly, gt.base_log_rate_hourly, gt.base_log_rate_salaried)
    log_rate = log_rate + pm["second_job"].to_numpy() * gt.second_job
    log_rate = log_rate + hourly * gt.volatility_per_hour * np.maximum(
        pm["schedule_volatility"].to_numpy() - 3.0, 0)
    log_rate = log_rate + np.minimum(
        gt.commute_per_km * np.maximum(pm["commute_km"].to_numpy() - 5, 0), gt.commute_cap)
    log_rate = log_rate + (pm["tenure_months"].to_numpy() < 3) * gt.new_hire
    student = pm["is_student"].to_numpy().astype(bool)
    log_rate = log_rate + student * gt.student
    log_rate = log_rate + student * np.isin(month_of_year, list(EXAM_MONTHS)) \
        * gt.student_exam_season
    log_rate = log_rate + np.isin(month_of_year, list(FLU_MONTHS)) * gt.flu_season
    log_rate = log_rate + (pm["performance_rating"].to_numpy() <= 2) * gt.low_performance

    out = pm[["employee_id", "month", "store_id", "district_id", "role"]].copy()
    out["callouts"] = rng.poisson(np.exp(log_rate))
    return out


def build_callout_panel(snapshots: pd.DataFrame,
                        absences: pd.DataFrame) -> pd.DataFrame:
    """Snapshot features at month t + call-out count in month t+1 as target."""
    nxt = absences[["employee_id", "month", "callouts"]].copy()
    nxt["month"] = nxt["month"] - 1          # align t+1 outcome onto month t
    panel = snapshots.merge(nxt, on=["employee_id", "month"], how="inner")
    return panel.rename(columns={"callouts": "callouts_next_month"})


class CalloutModel:
    """Poisson gradient boosting on the same point-in-time snapshot features."""

    def __init__(self, roles: list[str] | None = None, **gbm_params):
        self.roles = roles or list(HOURLY_ROLES)
        self.features = list(ALL_FEATURES)
        self.params = {"loss": "poisson", "max_iter": 250, "learning_rate": 0.06,
                       "max_leaf_nodes": 31, "min_samples_leaf": 40,
                       "random_state": 0, **gbm_params}
        self.model: HistGradientBoostingRegressor | None = None
        self._categories: dict[str, dict[str, float]] = {}

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        from .snapshots import CATEGORICAL_FEATURES
        X = df[self.features].copy()
        for c in CATEGORICAL_FEATURES:
            if not self._categories.get(c):
                cats = sorted(df[c].astype(str).unique())
                self._categories[c] = {v: float(i) for i, v in enumerate(cats)}
            X[c] = X[c].astype(str).map(self._categories[c])
        return X.astype(float)

    def fit(self, panel: pd.DataFrame) -> "CalloutModel":
        rows = panel[panel["role"].isin(self.roles)]
        self.model = HistGradientBoostingRegressor(**self.params)
        self.model.fit(self._encode(rows), rows["callouts_next_month"])
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        rows = panel[panel["role"].isin(self.roles)]
        out = rows[["employee_id", "store_id", "district_id", "month", "role"]].copy()
        out["expected_callouts"] = np.maximum(
            self.model.predict(self._encode(rows)), 0)
        return out.reset_index(drop=True)

    def evaluate(self, panel: pd.DataFrame) -> dict:
        """Deviance vs base-rate, plus top-decile concentration of absences."""
        from sklearn.metrics import mean_poisson_deviance

        rows = panel[panel["role"].isin(self.roles)].reset_index(drop=True)
        y = rows["callouts_next_month"].to_numpy(dtype=float)
        mu = np.maximum(self.model.predict(self._encode(rows)), 1e-6)
        base = np.full_like(mu, y.mean())
        n_top = max(int(len(rows) * 0.10), 1)
        top = np.argsort(-mu)[:n_top]
        return {
            "n": int(len(rows)),
            "mean_callouts": round(float(y.mean()), 3),
            "poisson_deviance_model": round(float(mean_poisson_deviance(y, mu)), 4),
            "poisson_deviance_baseline": round(float(mean_poisson_deviance(y, base)), 4),
            "top_decile_share_of_callouts": round(float(y[top].sum() / max(y.sum(), 1)), 3),
            "predicted_total": round(float(mu.sum()), 1),
            "actual_total": int(y.sum()),
        }


def reserve_staffing_plan(predictions: pd.DataFrame,
                          service_quantile: float = 0.90) -> pd.DataFrame:
    """Float-pool sizing per district from predicted call-out volume.

    Treats each store-month's total expected call-outs as Poisson and sizes
    reserve shifts at the requested service level: covering the mean is a
    coin flip, so the plan holds ``q``-quantile - mean extra shifts ready.
    """
    from scipy import stats as _stats  # optional; fall back to normal approx
    per_store = predictions.groupby(["district_id", "store_id"], observed=True)[
        "expected_callouts"].sum()
    try:
        q = _stats.poisson.ppf(service_quantile, per_store.to_numpy())
    except Exception:  # pragma: no cover
        mu = per_store.to_numpy()
        q = mu + 1.2816 * np.sqrt(mu)
    plan = per_store.to_frame("expected_callout_shifts")
    plan["reserve_shifts_needed"] = np.maximum(
        np.ceil(q) - np.round(plan["expected_callout_shifts"]), 0).astype(int)
    out = plan.groupby("district_id").agg(
        expected_callout_shifts=("expected_callout_shifts", "sum"),
        reserve_shifts_needed=("reserve_shifts_needed", "sum"),
    ).reset_index()
    out["expected_callout_shifts"] = out["expected_callout_shifts"].round(1)
    out["service_level"] = service_quantile
    return out
