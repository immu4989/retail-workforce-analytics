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
python examples/run_pipeline.py      # use cases 1-4: turnover, headcount, drivers
python examples/run_operations.py    # use cases 5-9: scheduling, call-outs, funnel, bench
```

## The nine use cases

| # | Use case | Module | Detailed writeup |
|---|----------|--------|------------------|
| 1 | Hourly turnover risk (baristas, shift supervisors) at 3/6/12 months | `TurnoverModel("hourly")` | [docs/use_cases/01_hourly_turnover.md](docs/use_cases/01_hourly_turnover.md) |
| 2 | Salaried turnover risk (store managers, assistants) at 6/12 months | `TurnoverModel("salaried")` | [docs/use_cases/02_salaried_turnover.md](docs/use_cases/02_salaried_turnover.md) |
| 3 | Headcount forecasting: "hire N baristas in district D in the next 3 months" | `build_hiring_plan` | [docs/use_cases/03_headcount_forecasting.md](docs/use_cases/03_headcount_forecasting.md) |
| 4 | Turnover drivers and what-if interventions | `driver_importance`, `InterventionSimulator` | [docs/use_cases/04_turnover_drivers.md](docs/use_cases/04_turnover_drivers.md) |
| 5 | Demand-driven labor forecasting + fair-workweek scheduling | `LaborDemandForecaster`, `build_week_schedule` | [docs/use_cases/05_demand_scheduling.md](docs/use_cases/05_demand_scheduling.md) |
| 6 | Call-out (unplanned absence) prediction + reserve staffing | `CalloutModel`, `reserve_staffing_plan` | [docs/use_cases/06_absenteeism.md](docs/use_cases/06_absenteeism.md) |
| 7 | Hiring funnel analytics + requisition timing | `simulate_funnel`, `req_timing` | [docs/use_cases/07_hiring_funnel.md](docs/use_cases/07_hiring_funnel.md) |
| 8 | Promotion readiness + leadership bench strength | `PromotionModel`, `bench_strength` | [docs/use_cases/08_internal_mobility.md](docs/use_cases/08_internal_mobility.md) |
| 9 | Turnover contagion, and why the naive estimate misleads | `contagion_analysis` | [docs/use_cases/09_turnover_contagion.md](docs/use_cases/09_turnover_contagion.md) |

Use cases 1-4 are the people-side stack (`examples/run_pipeline.py`);
5-9 are the operations stack (`examples/run_operations.py`). What large
operators run beyond these, and what each would need to be added with the
same rigour, is in [docs/ROADMAP.md](docs/ROADMAP.md).

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
| Hourly | 3 mo | 0.17 | 0.667 | 0.668 | 0.009 | 2.1x |
| Hourly | 6 mo | 0.29 | 0.658 | 0.663 | 0.022 | 1.8x |
| Hourly | 12 mo | 0.45 | 0.653 | 0.666 | 0.045 | 1.6x |
| Salaried | 6 mo | 0.11 | 0.632 | 0.691 | 0.038 | 1.9x |
| Salaried | 12 mo | 0.21 | 0.655 | 0.664 | 0.065 | 2.4x |

A deep discrete-time survival network (`SurvivalNN`, PyTorch, optional)
matches the gradient-boosted models on ranking from a single trained
artifact and produces full 12-month retention curves per employee; the GBM
stays the recommended default. The comparison table ships in
`reports/nn_vs_gbm_hourly.csv` so nobody has to rediscover it.

The simulated workforce reproduces realistic dynamics, including the new-hire
washout that dominates hourly retail attrition:

![Monthly exit rate by tenure](docs/figures/tenure_hazard.png)

Summed probabilities track realised exits closely enough to hire against
(699 predicted vs 710 actual over one 6-month window, company-wide):

![Headcount validation](docs/figures/headcount_validation.png)

And the driver analysis separates levers operations can pull from context that
is only useful for targeting:

![Turnover drivers](docs/figures/drivers.png)

SHAP decomposes individual predictions (with a test asserting the
attributions are additive and directionally correct against the ground
truth), and per-employee reason codes phrase them for HR partners:

![SHAP beeswarm](docs/figures/shap_beeswarm.png)

## What it is worth in dollars

The cost model (`CostModel`, all parameters explicit and adjustable) prices
the outputs. On the simulated company, roughly 2,900 employees:

| Question | Answer |
|----------|--------|
| Baseline attrition burn | $12.9M per year (2,126 exits, mostly baristas) |
| Best single retention lever found | Scheduling people their desired hours: $240k/yr |
| Targeted retention program (top decile) | 1.76x ROI vs 1.03x untargeted, ~$487k net per cycle |
| Headcount plan accuracy | 699 predicted vs 710 actual exits over 6 months |
| Demand-driven scheduling vs staff-to-average | $328 per store-week, $2.4M/yr chain-wide |
| Hourly demand forecast | WAPE 17.8% vs 24.1% seasonal-naive, at 94% week-over-week schedule stability |

![Intervention value](docs/figures/intervention_value.png)

These are exact computations under stated assumptions on synthetic data, not
promises. The per-use-case writeups show the arithmetic and how it scales
with workforce size; swap in your own replacement costs before quoting any of
it internally.

## The operations stack

Use case 5 forecasts every store's hourly demand (12 held-out weeks shown
company-wide in the metrics; one store-week below) and builds legal,
fair-workweek-stable schedules against it:

![Demand forecast](docs/figures/demand_forecast.png)

Use case 9 uses the known ground truth for a methods lesson real data can't
deliver: the simulator plants **no** turnover contagion, yet the raw
peer-exit gradient shows the +19% "contagion effect" the literature reports.
Stratify on store conditions and it vanishes — the naive estimate was
confounding, and the testbed proves it:

![Contagion](docs/figures/contagion.png)

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

# 5. "What would stabilising schedules buy us, in dollars?"
from workforce_analytics import CostModel
sim = InterventionSimulator(model, snaps[snaps["month"] == 48])
print(sim.run(stabilize_schedules(), "cap schedule volatility", horizon=6,
              cost_model=CostModel()))

# 6. "Why is this employee at risk?" (needs the shap extra)
from workforce_analytics import reason_codes
print(reason_codes(model, snaps[snaps["month"] == 48], horizon=6).head())

# 7. Full retention curves from one deep survival model (needs the torch extra)
from workforce_analytics import SurvivalNN
from workforce_analytics.config import HOURLY_ROLES
snaps1 = build_snapshots(result.person_months, horizons=(1,))
nn = SurvivalNN(roles=list(HOURLY_ROLES)).fit(snaps1[snaps1["month"] <= 44])
curves = nn.survival_curves(snaps[snaps["month"] == 48], horizon=12)
```

## Repository layout

```
src/workforce_analytics/
    config.py       simulation settings + the ground-truth hazard coefficients
    generator.py    month-by-month workforce simulator (hiring, promotion, attrition)
    snapshots.py    point-in-time feature/label construction, out-of-time splits
    turnover.py     calibrated multi-horizon gradient-boosted turnover models
    survival_nn.py  deep discrete-time survival model (optional, torch)
    evaluation.py   discrimination, calibration and lift metrics
    oracle.py       the oracle ceiling: how well could a perfect model do?
    headcount.py    hiring plans from expected attrition + growth + vacancies
    drivers.py      permutation importance, PDPs, what-if intervention simulator
    explain.py      SHAP: global importance + per-employee reason codes (optional, shap)
    cost_model.py   dollars: baseline burn, intervention value, targeting ROI
    demand.py       hourly traffic simulator, labor forecaster, fair-workweek scheduler
    absence.py      call-out simulator, Poisson prediction, reserve staffing
    funnel.py       hiring funnel simulator, stage conversion, requisition timing
    mobility.py     promotion events, readiness model, bench strength
    contagion.py    peer-exit exposure analysis, raw vs stratified
examples/           two pipelines (people + operations) producing reports/ and figures
tests/              41 tests: realism, leakage, calibration, SHAP additivity, accounting
docs/               per-use-case writeups, roadmap, guide to adapting real HRIS data
```

Optional extras: `pip install -e ".[explain]"` for SHAP, `".[deep]"` for the
survival network, `".[dev]"` for tests and plotting.

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
