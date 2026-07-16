# Use case 5: Demand-driven labor forecasting and scheduling

Predict transactions per store per hour, convert them to required staff, and
build next week's schedule against the curve — under labor-law and
fair-workweek constraints.

## The problem

Labor is a QSR chain's largest controllable cost, and most stores still staff
it to an average day. The result is the familiar double failure: overstaffed
Tuesday afternoons paying wages nobody's sales support, and understaffed
Friday rushes that burn customers and baristas at once. Workforce-management
vendors report high-single-digit to 12% labor-cost reductions when chains
move from template staffing to demand-driven schedules, which is why every
large operator (Starbucks, McDonald's, Chipotle among them) runs some version
of this system.

The second half of the problem is legal and human. Predictive-scheduling
("fair workweek") laws in Seattle, NYC, Chicago and Oregon penalise late
schedule changes, and use case 4 already showed schedule volatility is a
turnover driver. A scheduler that reoptimises from scratch every week is
technically optimal and operationally corrosive.

## What the module does

```python
traffic = TrafficSimulator(stores).run()                     # known ground truth
fc = LaborDemandForecaster().fit(traffic, train_end_week=90)
preds = fc.predict(traffic, weeks=test_weeks)
schedule = build_week_schedule(preds_one_store_week, roster,
                               previous_shifts=last_week.shifts)
```

The forecaster is a pooled gradient-boosted regressor on calendar structure,
planned promotions and lag features. The scheduler is a deliberately readable
greedy reference implementation with the constraints production systems
enforce: 4-8 hour shifts, one shift a day, five days a week, hours near each
person's desired hours, 10-hour minimum rest (no clopening), and stickiness
to last week's schedule.

## Results

| Forecaster | WAPE (hourly, 12 held-out weeks) | Bias |
|------------|----------------------------------|------|
| Gradient boosting | **17.8%** | -0.4% |
| Seasonal naive (same hour last week) | 24.1% | -0.5% |

![Demand forecast](../figures/demand_forecast.png)

Hourly Poisson noise puts a hard floor under WAPE at this granularity;
beating seasonal-naive by a quarter is what good looks like. The scheduler
then covers the curve completely and legally — one example store-week: 234
required person-hours, 100% coverage, zero clopening or overwork violations,
23 overstaffed hours from shift granularity — and with last week's schedule
passed in, **94% of shifts repeat week over week**, against 17% when
rebuilding blind.

## What it saves

Scoring both strategies against realised demand across all 139 stores and 12
held-out weeks, at a $21 loaded wage and $35 per understaffed person-hour
(lost transactions plus service decay — adjust both in the script):

| Strategy | Staffing cost per store-week |
|----------|------------------------------|
| Staff to the average day | $1,270 |
| Staff to the forecast | $942 |

That is **$328 per store-week, or $2.4M a year on this simulated 139-store
chain** — roughly $17k per store per year. At the scale of a 10,000-store
operator, the same per-store economics land in the **$170M a year** range,
which is why this use case gets funded before any of the others. The honest
caveats: the simulator's demand is friendlier than real life (no weather
feeds, no local events calendar), and the savings figure prices the *plan*,
not execution slippage.

## Run it

```bash
python examples/run_operations.py
pytest tests/test_operations.py -k "forecaster or schedule" -q
```
