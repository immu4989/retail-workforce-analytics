"""Point-in-time snapshots: the supervised-learning view of the workforce.

Turnover prediction in production is a *snapshot* problem: on a given date,
for every active employee, predict the probability of exit within the next
``h`` months, using only information available on that date. This module
builds those training tables from the person-month history:

* one row per (employee, snapshot month)
* features computed from data at or before the snapshot month
* one binary label per horizon: did the employee leave in ``(t, t+h]``?
* snapshots too close to the end of the observation window are dropped for
  a given horizon (their label would be censored)

Splitting is **out-of-time** (train on earlier snapshot months, validate and
test on later ones), which is the honest way to evaluate a model that will be
scoring future months in production. Random row-level splits leak: the same
employee's adjacent months land on both sides of the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "tenure_months",
    "pay_ratio",
    "schedule_volatility_3m",
    "scheduled_hours",
    "hours_gap",
    "commute_km",
    "months_since_mgr_change",
    "months_since_promotion",
    "months_since_raise",
    "performance_rating",
    "store_staffing_ratio",
    "district_understaffed_share",
    "district_unemployment",
]
CATEGORICAL_FEATURES = ["role", "age_band", "month_of_year"]
BINARY_FEATURES = ["is_student", "second_job"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

ID_COLUMNS = ["employee_id", "store_id", "district_id", "month"]


def build_snapshots(
    person_months: pd.DataFrame,
    horizons: tuple[int, ...] = (3, 6, 12),
    snapshot_months: list[int] | None = None,
    min_month: int = 6,
    voluntary_only: bool = False,
) -> pd.DataFrame:
    """Build a snapshot table with one label column per horizon.

    Parameters
    ----------
    person_months : output of the simulator (or a real HRIS extract shaped
        the same way — see ``docs/adapting_to_real_data.md``).
    horizons : label windows in months, e.g. ``(3, 6, 12)``.
    snapshot_months : which months to snapshot; defaults to every month from
        ``min_month`` onward. Snapshots whose window is censored for a given
        horizon get ``NaN`` for that label.
    min_month : skip the first months so rolling features have history.
    voluntary_only : label only voluntary exits (regrettable-attrition view).
    """
    pm = person_months.sort_values(["employee_id", "month"]).copy()
    max_month = int(pm["month"].max())
    if snapshot_months is None:
        snapshot_months = list(range(min_month, max_month))

    # Exit month per employee (single employment spell per employee id).
    terms = pm[pm["terminated"] == 1]
    if voluntary_only:
        terms = terms[terms["termination_type"] == "voluntary"]
    term_month = terms.set_index("employee_id")["month"]

    # Rolling 3-month mean of schedule volatility, point-in-time by design:
    # the window at month t covers t-2..t only.
    pm["schedule_volatility_3m"] = (
        pm.groupby("employee_id")["schedule_volatility"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
        .round(2)
    )

    snaps = pm[pm["month"].isin(snapshot_months) & (pm["terminated"] == 0)].copy()
    snaps["month_of_year"] = (snaps["month"] % 12 + 1).astype(str)

    emp_term = snaps["employee_id"].map(term_month)
    for h in horizons:
        label = ((emp_term > snaps["month"]) & (emp_term <= snaps["month"] + h))
        label = label.fillna(False).astype(float)
        # Censored: the full window extends past the observed history.
        label[snaps["month"] + h > max_month] = np.nan
        snaps[f"label_{h}m"] = label

    keep = ID_COLUMNS + ALL_FEATURES + [f"label_{h}m" for h in horizons]
    out = snaps[keep].reset_index(drop=True)
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype("category")
    for c in BINARY_FEATURES:
        out[c] = out[c].astype(int)
    return out


def time_split(
    snapshots: pd.DataFrame,
    train_end: int,
    val_end: int,
    train_stride: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Out-of-time train/validation/test split on snapshot month.

    ``train_stride`` keeps every k-th snapshot month in train, which reduces
    the near-duplicate rows produced by the same employee appearing in many
    consecutive months. Validation (used for probability calibration) and
    test keep all months in their range.
    """
    m = snapshots["month"]
    train = snapshots[(m <= train_end) & (m % train_stride == 0)]
    val = snapshots[(m > train_end) & (m <= val_end)]
    test = snapshots[m > val_end]
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))
