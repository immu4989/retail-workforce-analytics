# Use case 6: Call-out prediction and reserve staffing

Predict unplanned absences a month ahead and size float coverage so a 6am
"can't make it" text doesn't wreck the morning rush.

## The problem

QSR operators run 5-10% shift no-show rates. A call-out is worse than a
planned absence in every way: it is discovered hours before the shift, it is
covered by whoever answers the phone (at overtime rates and with resentment),
and on a understaffed day it simply isn't covered, which loops back into the
schedule-volatility-drives-turnover finding from use case 4. The rostering
literature's standard answer is predict-then-optimize: forecast absence
volume, then hold reserve shifts against the forecast at a chosen service
level, instead of scrambling.

## What the module does

```python
absences = simulate_absences(person_months)          # published ground truth
panel = build_callout_panel(snapshots, absences)     # features at t, call-outs in t+1
model = CalloutModel().fit(train)
reserve = reserve_staffing_plan(model.predict(current_month))
```

The simulator plants realistic drivers — second jobs, long commutes, chaotic
schedules, new hires, students in exam season, flu season — and the model is
a Poisson gradient boosting on the same point-in-time snapshot features the
turnover models use. Reserve sizing treats each store-month's predicted
call-out volume as Poisson and staffs floats to the 90th percentile: covering
the mean is a coin flip.

## Results (out-of-time test window)

| Metric | Value |
|--------|-------|
| Call-outs per hourly employee-month | 0.73 (≈4.5% of shifts) |
| Poisson deviance, model vs predict-the-mean | **1.095 vs 1.154** |
| Share of all call-outs in the model's top decile | 17.1% (1.7x concentration) |
| Predicted vs actual total volume | 23,293 vs 24,233 (-3.9%) |

![Call-out seasonality](../figures/callout_seasonality.png)

Individual call-outs are close to irreducibly random — the deviance gain is
modest and anyone claiming to predict *which* employee will call out
*which* day is overfitting. The valuable signal is in the aggregate: the
model tracks the December-January flu spike, the store-level volume is
accurate within 4%, and that is exactly what reserve sizing needs.

## What it saves

An uncovered call-out costs an overtime backfill premium (half wage on an
8-hour shift ≈ $85) when someone answers, and degraded service when nobody
does. This chain generates about 1,700 hourly call-outs a month; converting
even the flu-season excess from scramble to planned float coverage is worth
**$400-700k a year at chain scale**, before counting the turnover effect of
no longer leaning on the same reliable people every time. The per-district
reserve plan (`reports/reserve_staffing.csv`) is the deliverable: expected
call-out shifts and the float count that covers them at 90% service.

## Run it

```bash
python examples/run_operations.py
pytest tests/test_operations.py -k callout -q
```
