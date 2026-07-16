"""SHAP explanations: global driver views and per-employee reason codes.

Requires the optional ``shap`` dependency (``pip install shap``).

Two consumption modes, matching how these models get used in practice:

* **Global** (:func:`shap_matrix`, :func:`shap_importance`) — which features
  move predictions across the whole population. Complements the permutation
  importance in :mod:`~workforce_analytics.drivers`: permutation measures
  *predictive reliance* (AUC lost when a feature is destroyed), SHAP
  decomposes *individual predictions*. When both agree on the top drivers,
  you can present either with confidence.

* **Per-employee** (:func:`reason_codes`) — the top factors pushing one
  employee's risk up, phrased in plain language. This is what an HR partner
  actually reads; nobody actions a bare probability. Attributions come from
  the uncalibrated booster margin (log-odds); isotonic calibration is
  monotone, so "what pushes risk up" is unchanged by it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _explainer(model, horizon: int):
    try:
        import shap
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SHAP support needs the optional dependency: pip install shap") from e
    return shap.TreeExplainer(model.raw_models_[horizon])


def shap_matrix(model, snapshots: pd.DataFrame, horizon: int | None = None,
                max_rows: int = 5000, random_state: int = 0):
    """SHAP values (log-odds units) for a sample of snapshot rows.

    Returns ``(shap_df, rows)`` where ``shap_df`` has one column per feature
    and ``rows`` are the corresponding raw snapshot rows (same order), for
    colouring plots by feature value.
    """
    horizon = horizon or model.horizons[0]
    rows = snapshots[snapshots["role"].isin(model.roles)]
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=random_state)
    rows = rows.reset_index(drop=True)
    sv = _explainer(model, horizon).shap_values(model.encode(rows))
    return pd.DataFrame(sv, columns=model.features), rows


def shap_importance(model, snapshots: pd.DataFrame, horizon: int | None = None,
                    max_rows: int = 5000) -> pd.DataFrame:
    """Global mean |SHAP| per feature, in log-odds units."""
    shap_df, _ = shap_matrix(model, snapshots, horizon, max_rows)
    out = (shap_df.abs().mean().rename("mean_abs_shap").reset_index()
           .rename(columns={"index": "feature"}))
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

def _phrase(feature: str, value) -> str:
    """Human wording for one contributing feature at its value."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = None
    match feature:
        case "pay_ratio":
            return f"paid {abs(1 - v) * 100:.0f}% {'below' if v < 1 else 'above'} local market"
        case "tenure_months":
            return f"{v:.0f} months of tenure"
        case "schedule_volatility_3m":
            return f"weekly schedule swings +/-{v:.1f} hours"
        case "hours_gap":
            return f"scheduled {v:.0f} hours/week under desired"
        case "scheduled_hours":
            return f"scheduled {v:.0f} hours/week"
        case "commute_km":
            return f"{v:.0f} km commute"
        case "months_since_mgr_change":
            return f"store manager changed {v:.0f} months ago"
        case "months_since_promotion":
            return f"{v:.0f} months since last promotion"
        case "months_since_raise":
            return f"{v:.0f} months since last raise"
        case "performance_rating":
            return f"performance rating {v:.0f}/5"
        case "store_staffing_ratio":
            return f"store staffed at {v * 100:.0f}% of target"
        case "district_understaffed_share":
            return f"{v * 100:.0f}% of district stores understaffed"
        case "district_unemployment":
            return f"local unemployment {v:.1f}%"
        case "is_student":
            return "student" if v else "not a student"
        case "second_job":
            return "works a second job" if v else "no second job"
        case "month_of_year":
            names = {1: "January", 2: "February", 3: "March", 4: "April",
                     5: "May", 6: "June", 7: "July", 8: "August",
                     9: "September", 10: "October", 11: "November",
                     12: "December"}
            return f"{names.get(int(v), value)} seasonality"
        case "age_band":
            return f"age {value}"
        case "role":
            return str(value).replace("_", " ")
        case _:
            return f"{feature} = {value}"


def reason_codes(model, snapshots: pd.DataFrame, horizon: int | None = None,
                 top_n: int = 3, max_rows: int = 5000) -> pd.DataFrame:
    """Top risk-increasing factors per employee, in plain language.

    Returns one row per employee with the predicted probability and
    ``reason_1..reason_n`` columns, sorted highest risk first.
    """
    horizon = horizon or model.horizons[0]
    shap_df, rows = shap_matrix(model, snapshots, horizon, max_rows)
    preds = model.predict(rows)

    out = rows[["employee_id", "store_id", "district_id", "role"]].copy()
    out[f"p_{horizon}m"] = preds[f"p_{horizon}m"].to_numpy()

    sv = shap_df.to_numpy()
    order = np.argsort(-sv, axis=1)[:, :top_n]
    for k in range(top_n):
        feats = [model.features[j] for j in order[:, k]]
        vals = [rows.iloc[i][f] for i, f in enumerate(feats)]
        contribs = sv[np.arange(len(rows)), order[:, k]]
        # Only phrase factors that actually push risk up.
        out[f"reason_{k + 1}"] = [
            _phrase(f, v) if c > 0 else ""
            for f, v, c in zip(feats, vals, contribs)]
    return out.sort_values(f"p_{horizon}m", ascending=False).reset_index(drop=True)
