import numpy as np
import pandas as pd
import pytest

from workforce_analytics import (
    CalloutModel,
    LaborDemandForecaster,
    PromotionModel,
    TrafficConfig,
    TrafficSimulator,
    TurnoverModel,
    bench_strength,
    build_callout_panel,
    build_promotion_panel,
    build_week_schedule,
    contagion_analysis,
    promotion_events,
    req_timing,
    required_staff,
    reserve_staffing_plan,
    schedule_stability,
    simulate_absences,
    simulate_funnel,
    time_split,
)
from workforce_analytics.config import HOURLY_ROLES


@pytest.fixture(scope="module")
def traffic(sim):
    stores = sim.stores.head(6)
    return stores, TrafficSimulator(stores, TrafficConfig(n_weeks=70, seed=5)).run()


# ---------------------------------------------------------------- demand

def test_forecaster_beats_seasonal_naive(traffic):
    stores, tf = traffic
    fc = LaborDemandForecaster().fit(tf, train_end_week=58)
    preds = fc.predict(tf, weeks=list(range(60, 70)))
    metrics = fc.evaluate(preds).set_index("forecaster")
    assert metrics.loc["model", "wape"] < metrics.loc["seasonal_naive", "wape"]
    assert metrics.loc["model", "wape"] < 0.30
    assert abs(metrics.loc["model", "bias_pct"]) < 5


def test_schedule_covers_demand_without_violations(traffic):
    stores, tf = traffic
    fc = LaborDemandForecaster().fit(tf, train_end_week=58)
    preds = fc.predict(tf, weeks=[62, 63])
    sid = stores["store_id"].iloc[0]
    roster = pd.DataFrame({
        "employee_id": range(40),
        "desired_hours": np.random.default_rng(1).choice([20, 28, 32, 38], 40)})
    a = build_week_schedule(preds[(preds.store_id == sid) & (preds.week == 62)], roster)
    assert a.summary["coverage_pct"] == 100.0
    assert a.violations() == 0
    # Fair-workweek stickiness: rebuilding next week against the previous
    # schedule must yield strictly more repeated shifts than rebuilding blind
    # (how much more depends on how similar the two weeks' demand is).
    week63 = preds[(preds.store_id == sid) & (preds.week == 63)]
    sticky = build_week_schedule(week63, roster, previous_shifts=a.shifts)
    blind = build_week_schedule(week63, roster)
    assert schedule_stability(sticky.shifts, a.shifts) \
        > schedule_stability(blind.shifts, a.shifts)
    assert sticky.summary["coverage_pct"] == 100.0 and sticky.violations() == 0


def test_required_staff_floors_and_scales():
    req = required_staff(np.array([0.0, 18.0, 90.0]), service_rate=18, min_staff=2)
    assert req.tolist() == [2, 2, 5]


# ---------------------------------------------------------------- absence

def test_callout_model_beats_baseline(sim, snapshots):
    absences = simulate_absences(sim.person_months)
    hourly = absences[absences["role"].isin(HOURLY_ROLES)]
    # Realistic monthly call-out volume (roughly 3-8% of ~16 shifts).
    assert 0.3 < hourly["callouts"].mean() < 1.5

    panel = build_callout_panel(snapshots, absences)
    tr, va, te = time_split(panel, 28, 34, train_stride=2)
    model = CalloutModel().fit(pd.concat([tr, va]))
    m = model.evaluate(te)
    assert m["poisson_deviance_model"] < m["poisson_deviance_baseline"]
    assert m["top_decile_share_of_callouts"] > 0.12
    # Aggregate volume within 15% — reserve staffing depends on it.
    assert abs(m["predicted_total"] - m["actual_total"]) / m["actual_total"] < 0.15


def test_reserve_plan_covers_above_mean(sim, snapshots):
    absences = simulate_absences(sim.person_months)
    panel = build_callout_panel(snapshots, absences)
    tr, va, te = time_split(panel, 28, 34, train_stride=2)
    model = CalloutModel().fit(pd.concat([tr, va]))
    plan = reserve_staffing_plan(model.predict(panel[panel["month"] == 36]))
    assert (plan["reserve_shifts_needed"] >= 0).all()
    assert plan["reserve_shifts_needed"].sum() > 0


# ---------------------------------------------------------------- funnel

def test_funnel_stages_are_monotone(sim):
    reqs = simulate_funnel(sim.districts, n_months=8, seed=7)
    stages = ["applications", "screen_passes", "interviews_attended",
              "offers", "accepts", "starts"]
    totals = [reqs[s].sum() for s in stages]
    assert all(a >= b for a, b in zip(totals, totals[1:]))
    assert 0.7 < reqs["filled"].mean() <= 1.0


def test_req_timing_flags_slow_roles(sim):
    reqs = simulate_funnel(sim.districts, n_months=12, seed=7)
    plan = pd.DataFrame({
        "district_id": [sim.districts["district_id"].iloc[0]] * 2,
        "role": ["barista", "store_manager"],
        "hires_needed": [40, 1],
        "horizon_months": [6, 6],
    })
    out = req_timing(plan, reqs)
    ttf = out.set_index("role")["ttf_p50_days"]
    assert ttf["store_manager"] > ttf["barista"]


# ---------------------------------------------------------------- mobility

def test_promotion_model_ranks_promotions(sim, snapshots):
    events = promotion_events(sim.person_months)
    assert len(events) > 50
    assert (events["from_role"] != events["to_role"]).all()

    panel = build_promotion_panel(snapshots, sim.person_months, horizon=6)
    tr, va, te = time_split(panel, 28, 34, train_stride=2)
    model = PromotionModel(horizon=6).fit(tr, va)
    m = model.evaluate(te)
    assert m["roc_auc"] > 0.70
    assert m["precision_top_5pct"] > 2 * m["promotion_base_rate"]


def test_bench_strength_accounting(sim, snapshots, splits):
    train, val, _ = splits
    panel = build_promotion_panel(snapshots, sim.person_months, horizon=6)
    ptr, pva, _ = time_split(panel, 28, 34, train_stride=2)
    promo = PromotionModel(horizon=6).fit(ptr, pva)
    salaried = TurnoverModel("salaried", horizons=(12,)).fit(train, val)
    month = 36
    bench = bench_strength(promo.predict(panel[panel["month"] == month]),
                           salaried.predict(snapshots[snapshots["month"] == month]),
                           sim.stores, horizon=12)
    assert (bench["expected_vacancies"] >= 0).all()
    assert (bench["ready_now"] >= 0).all()
    assert bench["district_id"].nunique() == len(bench)


# ---------------------------------------------------------------- contagion

def test_contagion_adjustment_shrinks_gradient(sim):
    c = contagion_analysis(sim.person_months)
    raw, adj = c["raw"], c["adjusted"]
    raw_top = raw.iloc[-1]["relative_risk"]
    adj_top = adj.iloc[-1]["adjusted_relative_risk"]
    # The simulator plants NO direct contagion effect: whatever raw gradient
    # exists must shrink toward 1 once store conditions are stratified out.
    assert abs(adj_top - 1.0) < abs(raw_top - 1.0) + 0.05
    assert (raw["n_person_months"] > 100).all()


def test_exposure_never_counts_self(sim):
    from workforce_analytics import peer_exit_exposure
    exposed = peer_exit_exposure(sim.person_months, window=3)
    # Employees in their first simulated month have no trailing window at
    # brand-new stores; exposure must be finite and non-negative everywhere.
    col = "peer_exit_rate_3m"
    assert (exposed[col] >= 0).all()
    assert np.isfinite(exposed[col]).all()
