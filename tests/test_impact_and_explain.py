import numpy as np
import pandas as pd
import pytest

from workforce_analytics import (
    CostModel,
    InterventionSimulator,
    raise_pay_floor,
    targeting_roi,
    turnover_cost_summary,
)

shap = pytest.importorskip("shap")


def test_turnover_cost_summary_accounting(sim):
    out = turnover_cost_summary(sim.person_months)
    total_row = out[out["role"] == "TOTAL"].iloc[0]
    per_role = out[out["role"] != "TOTAL"]
    assert total_row["annual_cost"] == per_role["annual_cost"].sum()
    n_years = (sim.person_months["month"].max() + 1) / 12
    assert total_row["total_exits"] == sim.person_months["terminated"].sum()
    assert total_row["exits_per_year"] == pytest.approx(
        total_row["total_exits"] / n_years, abs=0.5)


def test_intervention_reports_dollars(snapshots, hourly_model):
    sim_tool = InterventionSimulator(hourly_model, snapshots[snapshots["month"] == 36])
    out = sim_tool.run(raise_pay_floor(1.0), "pay to market", horizon=6,
                       cost_model=CostModel())
    total = out[out["district_id"] == "ALL"].iloc[0]
    assert total["dollars_saved"] > 0
    # Dollars must be bounded by exits avoided x the priciest role.
    assert total["dollars_saved"] <= total["exits_avoided"] * 7500 + 1


def test_targeting_roi_prefers_targeted(splits, hourly_model):
    _, _, test = splits
    rows = test[test["role"].isin(hourly_model.roles)]
    rows = rows[rows["label_6m"].notna()].reset_index(drop=True)
    report = hourly_model.predict(rows)
    roi = targeting_roi(report, rows["label_6m"], horizon=6)
    # Top-decile targeting concentrates leavers, so its ROI multiple must
    # beat treating everyone.
    assert roi["roi_multiple_targeted"] > roi["roi_multiple_untargeted"]
    assert roi["targeted"]["treated"] < roi["untargeted_everyone"]["treated"]


def test_shap_importance_agrees_with_ground_truth(splits, hourly_model):
    from workforce_analytics import shap_importance

    _, _, test = splits
    imp = shap_importance(hourly_model, test, horizon=3, max_rows=2000)
    assert set(imp["feature"]) == set(hourly_model.features)
    assert (imp["mean_abs_shap"] >= 0).all()
    # Tenure and the scheduling levers carry the largest true effects. SHAP
    # (mean |contribution|) and permutation importance (AUC reliance) rank
    # differently by construction, so only pin the coarse structure.
    top5 = set(imp.head(5)["feature"])
    assert "tenure_months" in top5
    assert top5 & {"scheduled_hours", "hours_gap", "schedule_volatility_3m",
                   "pay_ratio"}


def test_shap_is_additive_and_directionally_correct(splits, hourly_model):
    """Guards the ordinal-encoding choice: with HGB categorical bitset splits
    TreeExplainer silently returns non-additive garbage. Additivity plus
    ground-truth directions prove the explanations are real."""
    from workforce_analytics import shap_matrix

    _, _, test = splits
    h = hourly_model.horizons[0]
    sv, rows = shap_matrix(hourly_model, test, horizon=h, max_rows=3000)
    X = hourly_model.encode(rows)
    ex = shap.TreeExplainer(hourly_model.raw_models_[h])
    recon = ex.expected_value + ex.shap_values(X).sum(axis=1)
    margin = hourly_model.raw_models_[h].decision_function(X)
    np.testing.assert_allclose(recon, margin, atol=1e-4)

    pay = rows["pay_ratio"].to_numpy()
    assert sv.loc[pay < 0.95, "pay_ratio"].mean() > sv.loc[pay > 1.05, "pay_ratio"].mean()
    ten = rows["tenure_months"].to_numpy()
    assert sv.loc[ten < 3, "tenure_months"].mean() > sv.loc[ten >= 24, "tenure_months"].mean()


def test_reason_codes_are_phrased(splits, hourly_model):
    from workforce_analytics import reason_codes

    _, _, test = splits
    out = reason_codes(hourly_model, test, horizon=3, top_n=3, max_rows=1000)
    assert {"reason_1", "reason_2", "reason_3"} <= set(out.columns)
    # Sorted by risk, and the top employee has at least one real reason.
    assert out["p_3m"].is_monotonic_decreasing
    assert out.iloc[0]["reason_1"] != ""
    # No raw feature names should leak into the phrasing.
    joined = " ".join(out["reason_1"].head(50))
    assert "pay_ratio" not in joined and "tenure_months" not in joined


def test_survival_nn_optional():
    torch = pytest.importorskip("torch")
    from workforce_analytics import SurvivalNN
    from workforce_analytics import SimulationConfig, generate, build_snapshots, time_split
    from workforce_analytics.config import HOURLY_ROLES
    from sklearn.metrics import roc_auc_score

    res = generate(SimulationConfig(n_districts=3, n_months=36, seed=5))
    snaps = build_snapshots(res.person_months, horizons=(1, 3))
    train, val, test = time_split(snaps, 24, 28, train_stride=1)
    model = SurvivalNN(roles=list(HOURLY_ROLES), epochs=3).fit(
        pd.concat([train, val]))

    rows = test[test["role"].isin(HOURLY_ROLES)].reset_index(drop=True)
    preds = model.predict(rows, horizons=(3,))
    mask = rows["label_3m"].notna()
    auc = roc_auc_score(rows.loc[mask, "label_3m"], preds.loc[mask.to_numpy(), "p_3m"])
    assert auc > 0.60

    curves = model.survival_curves(rows.head(50), horizon=6)
    s = curves[[f"S_{k}m" for k in range(1, 7)]].to_numpy()
    assert ((np.diff(s, axis=1) <= 1e-9).all()), "survival must be non-increasing"
    assert ((s > 0) & (s <= 1)).all()
