"""Headcount planning: turn calibrated attrition probabilities into hiring plans.

The question a district manager actually asks is not "who is at risk" but
"how many baristas do I need to hire in the next three months". With
calibrated per-employee exit probabilities, expected losses are additive:

    expected exits in district d, role r, horizon h  =  Σ p_i(h)

over active employees. The hiring need adds two more terms that live in the
operational data, not the model:

    hires_needed = expected exits + current open positions + planned growth

``validate_expected_attrition`` closes the loop by comparing the summed
probabilities against realised exits per district on a held-out window —
group-level calibration, the property this whole plan depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLE_TARGET_COLUMNS = {
    "barista": "target_baristas",
    "shift_supervisor": "target_shift_supervisors",
}


def _store_targets_long(stores: pd.DataFrame, month: int) -> pd.DataFrame:
    """Per-store role targets for stores open at ``month``, in long format."""
    open_now = stores[stores["open_month"] <= month].copy()
    frames = []
    for role, col in ROLE_TARGET_COLUMNS.items():
        f = open_now[["store_id", "district_id", col]].rename(columns={col: "target"})
        f["role"] = role
        frames.append(f)
    asm = open_now[open_now["has_asm"]][["store_id", "district_id"]].copy()
    asm["target"], asm["role"] = 1, "assistant_store_manager"
    sm = open_now[["store_id", "district_id"]].copy()
    sm["target"], sm["role"] = 1, "store_manager"
    return pd.concat(frames + [asm, sm], ignore_index=True)


def build_hiring_plan(
    predictions: pd.DataFrame,
    stores: pd.DataFrame,
    horizon: int,
    month: int | None = None,
) -> pd.DataFrame:
    """Hiring plan per district and role for the next ``horizon`` months.

    Parameters
    ----------
    predictions : scored snapshot for a single month — output of
        ``TurnoverModel.predict`` (concatenate hourly and salaried outputs to
        plan for all roles). Must contain ``p_{horizon}m``.
    stores : store table with staffing targets and ``open_month``.
    horizon : planning window in months.
    month : the snapshot month (defaults to the month in ``predictions``).
    """
    col = f"p_{horizon}m"
    if col not in predictions.columns:
        raise ValueError(f"predictions must contain {col!r}")
    if month is None:
        months = predictions["month"].unique()
        if len(months) != 1:
            raise ValueError("predictions span multiple months; pass month= explicitly")
        month = int(months[0])
    preds = predictions[predictions["month"] == month]

    grouped = preds.groupby(["district_id", "role"], observed=True).agg(
        active_headcount=("employee_id", "size"),
        expected_attrition=(col, "sum"),
    ).reset_index()

    targets_now = (_store_targets_long(stores, month)
                   .groupby(["district_id", "role"])["target"].sum())
    targets_future = (_store_targets_long(stores, month + horizon)
                      .groupby(["district_id", "role"])["target"].sum())

    grouped = grouped.set_index(["district_id", "role"])
    grouped["target_headcount"] = targets_now
    grouped["current_gap"] = (grouped["target_headcount"]
                              - grouped["active_headcount"]).clip(lower=0)
    grouped["growth_positions"] = (targets_future.reindex(grouped.index).fillna(0)
                                   - targets_now.reindex(grouped.index).fillna(0)).clip(lower=0)
    grouped["hires_needed"] = np.ceil(
        grouped["expected_attrition"] + grouped["current_gap"] + grouped["growth_positions"]
    ).astype(int)
    grouped["horizon_months"] = horizon

    out = grouped.reset_index()
    out["expected_attrition"] = out["expected_attrition"].round(1)
    return out.sort_values(["district_id", "role"]).reset_index(drop=True)


def validate_expected_attrition(
    predictions: pd.DataFrame,
    person_months: pd.DataFrame,
    horizon: int,
    by: str = "district_id",
) -> pd.DataFrame:
    """Compare predicted vs realised exits per group over the horizon.

    Uses the scored month in ``predictions`` and counts actual terminations of
    those same employees within the following ``horizon`` months. Returns one
    row per group with predicted, actual, and percentage error — the
    group-level calibration check for the hiring plan.
    """
    col = f"p_{horizon}m"
    months = predictions["month"].unique()
    if len(months) != 1:
        raise ValueError("pass predictions for a single snapshot month")
    month = int(months[0])

    terms = person_months[(person_months["terminated"] == 1)
                          & (person_months["month"] > month)
                          & (person_months["month"] <= month + horizon)]
    actual = (predictions.merge(terms[["employee_id"]], on="employee_id", how="left",
                                indicator=True)
              .assign(left=lambda d: (d["_merge"] == "both").astype(int)))

    out = actual.groupby(by, observed=True).agg(
        n_employees=("employee_id", "size"),
        predicted_exits=(col, "sum"),
        actual_exits=("left", "sum"),
    ).reset_index()
    out["predicted_exits"] = out["predicted_exits"].round(1)
    out["pct_error"] = ((out["predicted_exits"] - out["actual_exits"])
                        / out["actual_exits"].replace(0, np.nan) * 100).round(1)
    return out
