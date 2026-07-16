import numpy as np
import pandas as pd

from workforce_analytics import (
    InterventionSimulator,
    TurnoverModel,
    build_hiring_plan,
    ceiling_auc,
    driver_importance,
    evaluate_model,
    raise_pay_floor,
    validate_expected_attrition,
)


def test_model_beats_chance_and_baseline(hourly_model, splits):
    _, _, test = splits
    metrics = evaluate_model(hourly_model, test)
    assert (metrics["roc_auc"] > 0.60).all()
    # Calibrated probabilities should beat the predict-the-base-rate Brier.
    assert (metrics["brier"] < metrics["brier_baseline"]).all()
    assert (metrics["ece"] < 0.06).all()


def test_model_approaches_oracle_ceiling(hourly_model, splits):
    """On synthetic data the model should capture most of the true signal."""
    _, _, test = splits
    metrics = evaluate_model(hourly_model, test)
    rows = test[test["role"].isin(hourly_model.roles)]
    for h, auc in zip(hourly_model.horizons, metrics["roc_auc"]):
        ceiling = ceiling_auc(rows, h)
        assert auc > 0.5 + 0.75 * (ceiling - 0.5), (
            f"{h}m AUC {auc:.3f} captures <75% of ceiling {ceiling:.3f}")


def test_predictions_are_probabilities(hourly_model, splits):
    _, _, test = splits
    preds = hourly_model.predict(test)
    for h in hourly_model.horizons:
        p = preds[f"p_{h}m"]
        assert p.between(0, 1).all()
    # Longer horizons should predict more attrition on average.
    assert preds["p_6m"].mean() > preds["p_3m"].mean()


def test_salaried_population_is_disjoint(splits):
    train, val, _ = splits
    m = TurnoverModel("salaried", horizons=(6,)).fit(train, val)
    preds = m.predict(val)
    assert set(preds["role"].astype(str)) <= {"assistant_store_manager", "store_manager"}


def test_hiring_plan_accounting(sim, snapshots, hourly_model):
    month = 36
    preds = hourly_model.predict(snapshots[snapshots["month"] == month])
    plan = build_hiring_plan(preds, sim.stores, horizon=6, month=month)
    assert (plan["hires_needed"] >= 0).all()
    # hires = ceil(expected attrition + gap + growth), so never less than parts.
    parts = (plan["expected_attrition"] + plan["current_gap"]
             + plan["growth_positions"])
    assert (plan["hires_needed"] >= np.floor(parts)).all()
    assert set(plan["role"].astype(str)) == {"barista", "shift_supervisor"}


def test_group_level_calibration(sim, snapshots, hourly_model):
    """Summed probabilities should track realised exits within 25% overall."""
    month = 36
    preds = hourly_model.predict(snapshots[snapshots["month"] == month])
    check = validate_expected_attrition(preds, sim.person_months, horizon=6)
    total_pred = check["predicted_exits"].sum()
    total_actual = check["actual_exits"].sum()
    assert abs(total_pred - total_actual) / total_actual < 0.25


def test_driver_importance_finds_tenure_and_levers(hourly_model, splits):
    _, _, test = splits
    imp = driver_importance(hourly_model, test, horizon=3, n_repeats=3,
                            max_rows=8000)
    assert imp.iloc[0]["feature"] == "tenure_months"  # strongest true effect
    assert set(imp["lever"]) == {"actionable", "contextual"}
    assert imp["description"].notna().all()


def test_pay_intervention_reduces_expected_exits(snapshots, hourly_model):
    sim_tool = InterventionSimulator(hourly_model, snapshots[snapshots["month"] == 36])
    out = sim_tool.run(raise_pay_floor(1.00), "pay to market", horizon=6)
    total = out[out["district_id"] == "ALL"].iloc[0]
    assert total["exits_avoided"] > 0
    assert total["expected_exits_after"] < total["expected_exits_before"]


def test_score_report_deciles(hourly_model, splits):
    _, _, test = splits
    report = hourly_model.score_report(test, horizon=3)
    assert report["risk_decile"].between(1, 10).all()
    top = report[report["risk_decile"] == 10]["p_3m"].mean()
    bottom = report[report["risk_decile"] == 1]["p_3m"].mean()
    assert top > bottom


def test_multiple_scoring_months_rejected():
    preds = pd.DataFrame({
        "employee_id": [1, 2], "month": [3, 4], "p_6m": [0.1, 0.2],
        "district_id": ["D00", "D00"], "role": ["barista", "barista"],
        "store_id": ["S000", "S000"],
    })
    import pytest
    with pytest.raises(ValueError):
        build_hiring_plan(preds, pd.DataFrame(), horizon=6)
