# Use case 3: Headcount forecasting

Answer the question a district manager actually asks: *"How many baristas do
I need to hire in district D06 in the next six months?"* — before the holes
appear on the schedule.

## The problem

Hiring in hourly retail is reactive almost everywhere. A store discovers it
is short, posts a requisition, and eats 4 to 8 weeks of understaffing while
the pipeline fills. Understaffing is not a neutral state: remaining staff get
volatile schedules and extra hours, which raises *their* exit probability
(this feedback loop is in the simulator's ground truth because it is one of
the most reliable findings in workforce research). One unplanned exit becomes
two. Districts that hire ahead of attrition break the spiral; districts that
hire behind it pay for the same position twice.

## What the model does

Calibrated probabilities are additive. Summing every employee's 6-month exit
probability gives expected attrition per district and role, and the hiring
plan is then plain accounting:

```
hires_needed = expected exits + open positions today + planned new-store growth
```

```python
preds = pd.concat([hourly.predict(month), salaried.predict(month)])
plan = build_hiring_plan(preds, stores, horizon=6)
```

No separate forecasting model, no aggregate time series that goes blind when
one district's tenure mix shifts. The forecast inherits every feature the
turnover models see, so a district full of new hires forecasts high, a
district that just absorbed a minimum-wage increase in its market pay
forecasts low, and the "why" is inspectable per employee.

## Results

The entire plan rests on one property: summed probabilities must match
realised exits at the group level. Held-out validation, one 6-month window:

![Predicted vs realised exits](../figures/headcount_validation.png)

Company-wide: **699 predicted vs 710 actual exits (1.6% under)**. Ten of
twelve districts land within 11%; a test enforces 25% at company level so
regressions cannot ship quietly. The output is a per-district, per-role
table, which for one simulated planning cycle totals 1,060 hires over six
months, decomposed into what drives it:

![Hiring plan](../figures/hiring_plan.png)

| district | role | active | expected attrition | open today | growth | hire |
|----------|------|--------|--------------------|------------|--------|------|
| D00 | barista | 161 | 48.8 | 24 | 0 | 73 |
| D00 | shift supervisor | 50 | 7.0 | 2 | 0 | 9 |
| D00 | assistant store manager | 10 | 1.5 | 0 | 0 | 2 |
| D00 | store manager | 13 | 1.0 | 0 | 0 | 1 |

## What it saves

Three cost lines move when hiring goes proactive:

**Vacancy and overtime cost.** Every month a barista position sits open, its
hours are covered by overtime (1.5x) or lost sales. At even $1,500 of
friction per vacancy-month, the ~350 predicted exits per quarter, hired
against in advance rather than 6 weeks late, avoid roughly **$800k of
vacancy-months a year** in this 2,900-person company.

**The understaffing spiral.** Use case 4 prices the attrition caused by
understaffing itself; hiring ahead of attrition is the intervention that
addresses it at the root.

**Recruiter efficiency.** A 6-month district-level plan converts hiring from
129 stores firing requisitions ad hoc into batched, schedulable pipelines,
which is where recruiting teams get their capacity back.

The group-level calibration check (`validate_expected_attrition`) is the
quarterly health metric: when it drifts, retrain before the plan misleads
anyone.

## Run it

```bash
python examples/run_pipeline.py     # writes reports/hiring_plan_6m.csv
pytest tests/test_models.py::test_group_level_calibration -q
```
