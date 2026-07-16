# Use case 1: Turnover risk for hourly store employees

Predict, for every barista and shift supervisor, the probability of leaving
within the next 3, 6 and 12 months.

## The problem

Hourly retail turnover runs 60 to 100% annualised. On the simulated company in
this repo (about 2,400 hourly employees across 130 stores), baristas churn at
63% and shift supervisors at 33% per year, which matches published
quick-service benchmarks. At a fully-loaded replacement cost of $5,000 per
barista and $7,500 per shift supervisor (Cornell CHR estimates recruiting,
onboarding, training and the productivity ramp at roughly $5,900 per hourly
service employee), attrition burns **$10.9M per year in this simulated
company alone** before a single manager position is counted.

The operational pain is concentrated in two places. Districts discover
staffing holes after they happen, then hire in a panic against a 4 to 8 week
lead time. And retention budgets get spread evenly over everyone, which means
most of the money lands on people who were never going to leave.

## What the model does

`TurnoverModel("hourly")` trains one gradient-boosted classifier per horizon
on point-in-time monthly snapshots and calibrates each on a later,
non-overlapping window. Output is a probability per employee per horizon,
plus a 1-10 risk decile for consumption by HR partners:

```python
model = TurnoverModel("hourly", horizons=(3, 6, 12)).fit(train, val)
report = model.score_report(current_month, horizon=6)
```

## Results (out-of-time test window)

| Horizon | Base rate | ROC-AUC | Oracle ceiling | Signal captured | ECE | Precision in top decile | Lift |
|---------|-----------|---------|----------------|-----------------|------|--------------------------|------|
| 3 mo | 17% | 0.667 | 0.668 | 99% | 0.9% | 35% | 2.1x |
| 6 mo | 29% | 0.658 | 0.663 | 97% | 2.2% | 52% | 1.8x |
| 12 mo | 45% | 0.653 | 0.666 | 92% | 4.5% | 72% | 1.6x |

Read the 6-month row as an operator: the top-decile list is right about one
employee in two, in a population where the background rate is one in 3.5.

![AUC vs ceiling](../figures/auc_vs_ceiling.png)

An AUC of 0.66 looks unimpressive next to Kaggle leaderboards, and that
instinct is exactly what the oracle ceiling exists to correct. Scoring the
*true* hazard that generated the data yields 0.663 to 0.668; the model is
within a point or two of perfect. Turnover is mostly irreducible noise, and
teams that expect 0.85 on real HR data either leak labels or quit in month
three. This repo makes the ceiling measurable.

Probabilities are calibrated, not just ranked, which is what makes use case 3
(headcount) possible at all:

![Calibration](../figures/calibration.png)

## The deep-learning alternative

`SurvivalNN` (PyTorch, optional) trains a single discrete-time hazard network
on person-months instead of one classifier per horizon. Tenure and calendar
month advance deterministically inside the prediction window, and every
horizon comes from one model:

| Horizon | NN AUC | GBM AUC | NN ECE | GBM ECE |
|---------|--------|---------|--------|---------|
| 3 mo | 0.661 | 0.667 | 2.4% | 0.9% |
| 6 mo | 0.656 | 0.658 | 4.0% | 2.2% |
| 12 mo | 0.656 | 0.653 | 8.2% | 4.5% |

![Survival curves](../figures/survival_curves.png)

The honest summary: the network matches the GBM on ranking, loses on
calibration, and wins on product surface (full retention curves, any horizon
from one artifact). On tabular HR data, gradient boosting remains the right
default; the survival network earns its keep when the UI needs S(t) curves or
when horizons change often. That comparison is in the repo so nobody has to
rediscover it.

## What it saves

A retention program (stay interviews plus targeted schedule and pay fixes,
assumed 20% effective at $300 per participant, both parameters adjustable in
`targeting_roi`) pointed at the model's top decile:

| Strategy | Treated | Leavers reached | Gross savings | Program cost | Net | ROI |
|----------|---------|-----------------|---------------|--------------|-----|-----|
| Top decile by model | 2,146 | 1,125 | $1.13M | $0.64M | **+$487k** | 1.76x |
| Everyone (no model) | 21,460 | 6,209 | $6.66M | $6.44M | +$225k | 1.03x |

Same program, same assumptions: targeting turns a break-even initiative into
one that returns $1.76 per dollar. On a real chain with 200,000 hourly
employees (roughly Starbucks' US retail scale), the same per-employee
economics put the net value of targeting in the **$40M per year** range.
Every input to that number is a parameter you can change, and should, with
your own costs.

## Run it

```bash
python examples/run_pipeline.py     # trains, evaluates, writes reports/ and figures
pytest tests/test_models.py -q      # calibration, ceiling and leakage guarantees
```
