"""Why people leave: driver analysis and what-if intervention estimates.

Two complementary tools:

* :func:`driver_importance` — permutation importance on held-out, out-of-time
  data, with each feature tagged *actionable* (operations can move it: pay
  positioning, schedule stability, staffing, promotion velocity) or
  *contextual* (useful for targeting but not a lever: commute, age, tenure,
  season). Presenting the split this way is what makes the analysis land
  with operators instead of reading as a feature-importance dump.

* :class:`InterventionSimulator` — rescore the current workforce under a
  hypothetical policy ("bring everyone to at least 95% of market pay",
  "cap schedule volatility at the median") and report the change in expected
  exits, overall and per district.

A note on causality: permutation importance and what-if rescoring describe
the *model*, and the model learned associations. On this repo's synthetic
data the features cause attrition by construction, so the estimates are
causal and can be checked against the generator's coefficients
(:func:`ground_truth_comparison`). On real HR data, treat these outputs as
hypothesis generators and validate levers with experiments or quasi-
experimental designs before spending money on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ACTIONABLE = {
    "pay_ratio": "compensation vs local market",
    "schedule_volatility_3m": "week-to-week schedule consistency",
    "hours_gap": "under-scheduling vs desired hours",
    "scheduled_hours": "scheduled weekly hours",
    "months_since_promotion": "promotion velocity",
    "months_since_raise": "raise recency",
    "months_since_mgr_change": "management stability",
    "store_staffing_ratio": "store staffing level",
    "district_understaffed_share": "district staffing level",
}
CONTEXTUAL = {
    "tenure_months": "tenure",
    "commute_km": "home-to-store distance",
    "performance_rating": "performance rating",
    "district_unemployment": "local labour market",
    "role": "role",
    "age_band": "age band",
    "month_of_year": "season",
    "is_student": "student status",
    "second_job": "second job",
}


def driver_importance(
    model,
    test: pd.DataFrame,
    horizon: int | None = None,
    n_repeats: int = 5,
    max_rows: int = 20000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Permutation importance (AUC drop) on out-of-time data.

    ``model`` is a fitted :class:`~workforce_analytics.turnover.TurnoverModel`;
    importance is computed for the requested horizon's calibrated classifier.
    """
    horizon = horizon or model.horizons[0]
    rows = test[test["role"].isin(model.roles)]
    rows = rows[rows[f"label_{horizon}m"].notna()]
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=random_state)
    X, y = rows[model.features], rows[f"label_{horizon}m"]

    result = permutation_importance(
        model.models_[horizon], X, y, scoring="roc_auc",
        n_repeats=n_repeats, random_state=random_state)

    out = pd.DataFrame({
        "feature": model.features,
        "auc_drop_mean": result.importances_mean,
        "auc_drop_std": result.importances_std,
    })
    out["lever"] = np.where(out["feature"].isin(ACTIONABLE), "actionable", "contextual")
    out["description"] = out["feature"].map({**ACTIONABLE, **CONTEXTUAL})
    return out.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def partial_dependence_curve(
    model,
    test: pd.DataFrame,
    feature: str,
    horizon: int | None = None,
    grid_size: int = 12,
    max_rows: int = 5000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Average predicted probability while sweeping one feature.

    Manual partial dependence over the calibrated model, so the y-axis is a
    real probability, directly readable as "expected turnover at this value".
    """
    horizon = horizon or model.horizons[0]
    rows = test[test["role"].isin(model.roles)]
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=random_state)
    grid = np.quantile(rows[feature].dropna(), np.linspace(0.02, 0.98, grid_size))
    grid = np.unique(np.round(grid, 3))
    curve = []
    X = rows[model.features].copy()
    for v in grid:
        X[feature] = v
        curve.append(float(model.models_[horizon].predict_proba(X)[:, 1].mean()))
    return pd.DataFrame({feature: grid, f"mean_p_{horizon}m": curve})


class InterventionSimulator:
    """Rescore the workforce under hypothetical policy changes."""

    def __init__(self, model, snapshots: pd.DataFrame):
        self.model = model
        rows = snapshots[snapshots["role"].isin(model.roles)]
        self.baseline = rows.reset_index(drop=True)

    def run(self, transform, name: str, horizon: int | None = None) -> pd.DataFrame:
        """Apply ``transform(df) -> df`` to a copy of the snapshot and rescore.

        Returns per-district expected exits before/after and the reduction.
        """
        horizon = horizon or self.model.horizons[0]
        col = f"p_{horizon}m"
        before = self.model.predict(self.baseline)
        after_rows = transform(self.baseline.copy())
        after = self.model.predict(after_rows)

        cmp = before[["district_id"]].copy()
        cmp["p_before"], cmp["p_after"] = before[col], after[col]
        out = cmp.groupby("district_id", observed=True).agg(
            n_employees=("p_before", "size"),
            expected_exits_before=("p_before", "sum"),
            expected_exits_after=("p_after", "sum"),
        ).reset_index()
        total = pd.DataFrame([{
            "district_id": "ALL",
            "n_employees": len(cmp),
            "expected_exits_before": cmp["p_before"].sum(),
            "expected_exits_after": cmp["p_after"].sum(),
        }])
        out = pd.concat([out, total], ignore_index=True)
        out["exits_avoided"] = out["expected_exits_before"] - out["expected_exits_after"]
        out["reduction_pct"] = (out["exits_avoided"] / out["expected_exits_before"] * 100)
        for c in ["expected_exits_before", "expected_exits_after", "exits_avoided",
                  "reduction_pct"]:
            out[c] = out[c].round(1)
        out.insert(0, "intervention", name)
        out.insert(1, "horizon_months", horizon)
        return out


# Ready-made interventions ------------------------------------------------

def raise_pay_floor(min_ratio: float = 0.95):
    """Bring everyone paid below ``min_ratio`` of market up to it."""
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df["pay_ratio"] = df["pay_ratio"].clip(lower=min_ratio)
        return df
    return transform


def stabilize_schedules(max_volatility: float | None = None):
    """Cap 3-month schedule volatility (defaults to the population median)."""
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        cap = max_volatility if max_volatility is not None \
            else float(df["schedule_volatility_3m"].median())
        df["schedule_volatility_3m"] = df["schedule_volatility_3m"].clip(upper=cap)
        return df
    return transform


def close_hours_gap():
    """Schedule hourly employees for the hours they want."""
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df["scheduled_hours"] = df["scheduled_hours"] + df["hours_gap"]
        df["hours_gap"] = 0.0
        return df
    return transform


def ground_truth_comparison(importance: pd.DataFrame, ground_truth: dict) -> pd.DataFrame:
    """Line up learned importance with the simulator's true coefficients.

    Maps each model feature to the ground-truth parameter(s) that drive it, so
    users can confirm the pipeline recovers the data-generating process — the
    validation step real HR data never allows.
    """
    mapping = {
        "pay_ratio": abs(ground_truth["pay_ratio_hourly"]),
        "schedule_volatility_3m": ground_truth["schedule_volatility_per_hour"] * 4,
        "hours_gap": ground_truth["hours_gap_per_hour"] * 6,
        "commute_km": ground_truth["commute_cap_hourly"],
        "months_since_mgr_change": ground_truth["manager_change_recent_hourly"],
        "store_staffing_ratio": ground_truth["understaffed_store_hourly"],
        "performance_rating": ground_truth["low_performance_hourly"],
        "is_student": ground_truth["student_back_to_school"],
        "second_job": ground_truth["second_job"],
        "months_since_promotion": abs(ground_truth["recent_promotion_hourly"]),
    }
    out = importance.copy()
    out["true_effect_scale"] = out["feature"].map(mapping)
    return out
