"""Use case 9: turnover contagion — and why naive estimates of it mislead.

"Turnover is contagious" is one of people analytics' most repeated findings:
employees whose coworkers just quit are far more likely to quit themselves.
The effect is real in the raw numbers almost everywhere, including here. The
harder question — the one this module exists to teach — is how much of it is
*contagion* (peer exits causing exits) versus *common causes* (a bad store
burns everyone at once: understaffing, a manager change, a weak market
position hit every employee's hazard simultaneously).

This simulator plants **no direct peer-exit effect** in the ground truth.
Any exposure gradient in the raw data is therefore confounding by
construction, which makes this a rare controlled testbed:

* :func:`exposure_table` — monthly exit rate by "share of my store's team
  that exited in the last 3 months", raw. The gradient appears and looks
  exactly like the published contagion findings.
* :func:`adjusted_exposure_table` — the same gradient after stratifying on
  the store conditions the ground truth actually uses (staffing level,
  recent manager change, tenure mix). Watch the gradient shrink.

On real data the honest conclusion usually lands in between: some genuine
contagion (quitting coworkers transmit information about outside options and
lower the social cost of leaving) plus a lot of common cause. The method
here — build the naive table, then see what stratification survives — is the
first thing to run before anyone budgets for a "contagion intervention".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HOURLY_ROLES

# Exposure is the SHARE of the store's hourly team that exited in the
# trailing window — raw counts mostly proxy store size.
EXPOSURE_BUCKETS = [(0.0, 0.05, "<5%"), (0.05, 0.15, "5-15%"),
                    (0.15, 0.30, "15-30%"), (0.30, 10.0, "30%+")]
BASE_BUCKET = "<5%"


def _bucketize(x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=object)
    for lo, hi, label in EXPOSURE_BUCKETS:
        out[(x >= lo) & (x < hi)] = label
    return out


def peer_exit_exposure(person_months: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Per (employee, month): share of hourly peers who exited, trailing window.

    Returns hourly person_months plus ``peer_exit_rate_{window}m`` (trailing
    exits over current team size) and its bucket. The window covers months
    t-window..t-1 only, so an employee's own same-month exit is never part of
    their exposure.
    """
    pm = person_months[person_months["role"].isin(HOURLY_ROLES)].copy()
    exits = pm[pm["terminated"] == 1].groupby(
        ["store_id", "month"], observed=True).size()
    heads = pm.groupby(["store_id", "month"], observed=True).size()

    months = pm["month"].to_numpy()
    stores_ = pm["store_id"].to_numpy()
    trailing = np.zeros(len(pm))
    for k in range(1, window + 1):
        keys = pd.MultiIndex.from_arrays([stores_, months - k])
        trailing += exits.reindex(keys).fillna(0).to_numpy()
    team = heads.reindex(pd.MultiIndex.from_arrays([stores_, months])).to_numpy()

    col = f"peer_exit_rate_{window}m"
    pm[col] = trailing / np.maximum(team, 1)
    pm[f"{col}_bucket"] = _bucketize(pm[col].to_numpy())
    return pm


def exposure_table(pm_exposed: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Raw monthly exit rate by peer-exit exposure bucket."""
    col = f"peer_exit_rate_{window}m_bucket"
    order = [b[2] for b in EXPOSURE_BUCKETS]
    out = (pm_exposed.groupby(col, observed=True)
           .agg(n_person_months=("terminated", "size"),
                exit_rate=("terminated", "mean"))
           .reindex(order).reset_index())
    base = out.loc[out[col] == BASE_BUCKET, "exit_rate"].iloc[0]
    out["relative_risk"] = (out["exit_rate"] / base).round(2)
    out["exit_rate"] = out["exit_rate"].round(4)
    return out


def adjusted_exposure_table(pm_exposed: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Exposure gradient after stratifying on the true common causes.

    Strata: store staffing (under/ok), recent manager change (yes/no), and
    tenure bucket. Within-stratum relative risks vs the lowest-exposure
    bucket are combined with person-month weights.
    """
    col = f"peer_exit_rate_{window}m_bucket"
    df = pm_exposed.copy()
    df["understaffed"] = (df["store_staffing_ratio"] < 0.85)
    df["mgr_recent"] = (df["months_since_mgr_change"] <= 3)
    df["tenure_bucket"] = pd.cut(df["tenure_months"], [-1, 2, 11, 200],
                                 labels=["0-2", "3-11", "12+"])
    strata = ["understaffed", "mgr_recent", "tenure_bucket"]

    grouped = df.groupby(strata + [col], observed=True)["terminated"] \
        .agg(["size", "mean"]).reset_index()
    zero = grouped[grouped[col] == BASE_BUCKET]
    rows = []
    for bucket in [b[2] for b in EXPOSURE_BUCKETS]:
        b = grouped[grouped[col] == bucket]
        merged = b.merge(zero, on=strata, suffixes=("", "_base"))
        merged = merged[(merged["size"] >= 30) & (merged["size_base"] >= 30)
                        & (merged["mean_base"] > 0)]
        if len(merged) == 0:
            continue
        w = merged["size"]
        rows.append({
            col: bucket,
            "n_person_months": int(w.sum()),
            "exit_rate": round(float(np.average(merged["mean"], weights=w)), 4),
            "adjusted_relative_risk": round(
                float(np.average(merged["mean"] / merged["mean_base"], weights=w)), 2),
        })
    return pd.DataFrame(rows)


def contagion_analysis(person_months: pd.DataFrame, window: int = 3) -> dict:
    """Raw vs adjusted exposure gradients, ready for the docs figure."""
    exposed = peer_exit_exposure(person_months, window)
    raw = exposure_table(exposed, window)
    adj = adjusted_exposure_table(exposed, window)
    return {"raw": raw, "adjusted": adj, "window_months": window}
