# Retail Industry Workforce Analytics

Production-style people analytics for hourly retail workforces: multi-horizon
turnover prediction, headcount forecasting, and attrition-driver analysis,
built the way these systems are actually built inside large retailers.

I spent years building these models for one of the largest retail workforces in
the world. Real HR data can never be shared, so this repo does the next best
thing: it ships a workforce simulator with a **documented ground-truth hazard
model**, and builds the full modeling stack on top of it. Everything is
reproducible, every claim about "what drives turnover" can be checked against
the data-generating process, and the whole pipeline runs on a laptop in about
a minute.

```
pip install -e ".[dev]"
python examples/run_pipeline.py
```

## The four use cases

| # | Use case | Module | Detailed writeup |
|---|----------|--------|------------------|
| 1 | Hourly turnover risk (baristas, shift supervisors) at 3/6/12 months | `TurnoverModel("hourly")` | [docs/use_cases/01_hourly_turnover.md](docs/use_cases/01_hourly_turnover.md) |
| 2 | Salaried turnover risk (store managers, assistants) at 6/12 months | `TurnoverModel("salaried")` | [docs/use_cases/02_salaried_turnover.md](docs/use_cases/02_salaried_turnover.md) |
| 3 | Headcount forecasting: "hire N baristas in district D in the next 3 months" | `build_hiring_plan` | [docs/use_cases/03_headcount_forecasting.md](docs/use_cases/03_headcount_forecasting.md) |
| 4 | Turnover drivers and what-if interventions | `driver_importance`, `InterventionSimulator` | [docs/use_cases/04_turnover_drivers.md](docs/use_cases/04_turnover_drivers.md) |

## What makes this different from the usual attrition demo

Most public attrition projects train a classifier on a static cross-section
(one row per employee) and report an inflated AUC. Turnover does not work like
that in production. This repo gets four things right:

**Point-in-time snapshots.** Training data is one row per employee per month,
with features as they were that month and a label for "left within the next
h months". No information from the future leaks into the features; there is a
test asserting it.

**Out-of-time evaluation.** Models train on early months and are evaluated on
later months, because that is what scoring next quarter actually looks like.
Random row splits put the same employee's adjacent months on both sides of the
split and overstate performance.

**Calibrated probabilities.** Classifiers are calibrated on a held-out later
window, so probabilities can be summed into expected headcount losses. That is
what turns a risk score into a hiring plan.

**A known answer key.** Because the simulator's hazard model is published
(`ground_truth.json` ships with every dataset), you can compute the *oracle
ceiling*: the AUC of the true hazard itself. Turnover is a low-signal problem
and this makes that concrete instead of leaving you to wonder whether 0.67 is
a weak model or a hard problem.

![Model AUC vs oracle ceiling](docs/figures/auc_vs_ceiling.png)

The hourly model captures 92 to 99% of the signal that exists; the salaried
model 75 to 94%. On real data you never get to know this. Here it is a unit
test.

## Results at a glance

Out-of-time test metrics from `examples/run_pipeline.py` (13,285 simulated
employees across 12 districts, 60 months):

| Population | Horizon | Base rate | ROC-AUC | Ceiling | Calibration (ECE) | Lift @ top decile |
|------------|---------|-----------|---------|---------|-------------------|-------------------|
| Hourly | 3 mo | 0.17 | 0.668 | 0.668 | 0.010 | 2.1x |
| Hourly | 6 mo | 0.29 | 0.655 | 0.663 | 0.023 | 1.8x |
| Hourly | 12 mo | 0.45 | 0.652 | 0.666 | 0.045 | 1.6x |
| Salaried | 6 mo | 0.11 | 0.643 | 0.691 | 0.033 | 1.8x |
| Salaried | 12 mo | 0.21 | 0.655 | 0.664 | 0.061 | 2.7x |

The simulated workforce reproduces realistic dynamics, including the new-hire
washout that dominates hourly retail attrition:

![Monthly exit rate by tenure](docs/figures/tenure_hazard.png)

Summed probabilities track realised exits closely enough to hire against
(691 predicted vs 710 actual over one 6-month window, company-wide):

![Headcount validation](docs/figures/headcount_validation.png)

And the driver analysis separates levers operations can pull from context that
is only useful for targeting:

![Turnover drivers](docs/figures/drivers.png)

## Quick tour

```python
from workforce_analytics import (
    SimulationConfig, generate, build_snapshots, time_split,
    TurnoverModel, evaluate_with_ceiling, build_hiring_plan,
    InterventionSimulator, stabilize_schedules,
)

# 1. Simulate a company (or load your own HRIS extract in the same shape).
result = generate(SimulationConfig())

# 2. Point-in-time snapshots with 3/6/12-month labels.
snaps = build_snapshots(result.person_months, horizons=(3, 6, 12))
train, val, test = time_split(snaps, train_end=36, val_end=44)

# 3. Calibrated multi-horizon turnover model.
model = TurnoverModel("hourly").fit(train, val)
print(evaluate_with_ceiling(model, test))

# 4. "How many baristas does each district need to hire this half?"
preds = model.predict(snaps[snaps["month"] == 48])
plan = build_hiring_plan(preds, result.stores, horizon=6)

# 5. "What would stabilising schedules buy us?"
sim = InterventionSimulator(model, snaps[snaps["month"] == 48])
print(sim.run(stabilize_schedules(), "cap schedule volatility", horizon=6))
```

## Repository layout

```
src/workforce_analytics/
    config.py      simulation settings + the ground-truth hazard coefficients
    generator.py   month-by-month workforce simulator (hiring, promotion, attrition)
    snapshots.py   point-in-time feature/label construction, out-of-time splits
    turnover.py    calibrated multi-horizon gradient-boosted turnover models
    evaluation.py  discrimination, calibration and lift metrics
    oracle.py      the oracle ceiling: how well could a perfect model do?
    headcount.py   hiring plans from expected attrition + growth + vacancies
    drivers.py     permutation importance, PDPs, what-if intervention simulator
examples/          end-to-end pipeline producing reports/ and figures
tests/             23 tests: realism, leakage, calibration, accounting
docs/              methodology and per-use-case writeups
```

## Using this with real data

Shape your HRIS extract like `person_months` (one row per employee per month;
see `docs/adapting_to_real_data.md`) and everything downstream of the
generator works unchanged. Two cautions that matter more than any modeling
choice: keep features point-in-time (no post-termination edits, no backfilled
ratings), and treat driver analysis as hypothesis generation until a lever has
been validated with an experiment.

No real employee data was used anywhere in this project. The simulator's
parameters were chosen to match publicly documented retail benchmarks
(annualised turnover of 60 to 70% for quick-service hourly roles, 20 to 30%
for store management).

## License

MIT
