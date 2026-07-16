<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img alt="Retail Industry Workforce Analytics" src="docs/assets/banner-light.svg" width="100%">
</picture>

<p align="center">
  <a href="https://github.com/immu4989/retail-workforce-analytics/actions/workflows/ci.yml"><img src="https://github.com/immu4989/retail-workforce-analytics/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-2a78d6" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/tests-50%20passing-008300" alt="50 tests">
  <img src="https://img.shields.io/badge/HR%20data-100%25%20synthetic-184f95" alt="100% synthetic data">
  <img src="https://img.shields.io/badge/license-MIT-52514e" alt="MIT license">
</p>

<p align="center">
  <a href="#the-eleven-use-cases">Use cases</a> ·
  <a href="#see-it-work">Results</a> ·
  <a href="#what-it-is-worth-in-dollars">Dollars</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a>
</p>

I spent years building these systems for one of the largest retail
workforces in the world. Real HR data can never be shared, so this repo does
the next best thing: it ships a workforce simulator with a **documented
ground-truth hazard model** and builds the full modelling stack on top of it.
Every claim about "what drives turnover" or "what a raise buys" can be
checked against the process that generated the data — something no project
built on real HR data, and no static demo dataset, can offer.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/stats-dark.svg">
  <img alt="11 decision systems, 50 tests in CI, 3 one-command pipelines, 0 rows of real HR data" src="docs/assets/stats-light.svg" width="100%">
</picture>

> [!TIP]
> **The oracle ceiling is the repo's signature move.** Because the simulator's
> hazard coefficients are published, you can score the *true* risk model and
> measure the best AUC any model could reach. Turnover turns out to be a
> low-signal problem: the ceiling is ~0.67, and the models here capture
> **92-99% of it**. On real data you never know if 0.67 is a weak model or a
> hard problem. Here it is a unit test.

## How it fits together

```mermaid
flowchart LR
    GT["ground-truth<br/>coefficients"] --> SIM["workforce simulator<br/>139 stores · 60 months"]
    SIM --> PM[("person-month<br/>history")]
    PM --> SNAP["point-in-time<br/>snapshots"]
    SNAP --> MODELS["calibrated models<br/>turnover · promotion · call-outs"]
    SIM --> TRAFFIC["hourly traffic"] --> SCHED["demand forecasts<br/>+ legal schedules"]
    MODELS --> OUT["hiring plans · watchlists<br/>reserves · bench strength"]
    GT -. "oracle ceiling" .-> MODELS
    SIM -. "paired A/B reruns" .-> WAGE["wage-program<br/>experiments"]
```

## The eleven use cases

| # | | Use case | Writeup |
|---|---|----------|---------|
| 1 | 🚪 | Hourly turnover risk (baristas, shift supervisors) at 3/6/12 months | [docs](docs/use_cases/01_hourly_turnover.md) |
| 2 | 👔 | Salaried turnover risk (store managers, assistants) at 6/12 months | [docs](docs/use_cases/02_salaried_turnover.md) |
| 3 | 🧮 | Headcount forecasting: "hire N baristas in district D this quarter" | [docs](docs/use_cases/03_headcount_forecasting.md) |
| 4 | 🎛️ | Turnover drivers, SHAP reason codes, what-if interventions | [docs](docs/use_cases/04_turnover_drivers.md) |
| 5 | 📅 | Demand-driven labor forecasting + fair-workweek scheduling | [docs](docs/use_cases/05_demand_scheduling.md) |
| 6 | 🤒 | Call-out (unplanned absence) prediction + reserve staffing | [docs](docs/use_cases/06_absenteeism.md) |
| 7 | 🧲 | Hiring funnel analytics + requisition timing | [docs](docs/use_cases/07_hiring_funnel.md) |
| 8 | 🪜 | Promotion readiness + leadership bench strength | [docs](docs/use_cases/08_internal_mobility.md) |
| 9 | 🧪 | Turnover contagion, and why the naive estimate misleads | [docs](docs/use_cases/09_turnover_contagion.md) |
| 10 | 💵 | Compensation: raises, floors and freezes priced by true experiments | [docs](docs/use_cases/10_compensation.md) |
| 11 | 🎧 | Employee call-center topic mining (voice of the frontline) | [docs](docs/use_cases/11_call_center_topics.md) |

Use cases 1-4 are the people stack, 5-9 the operations stack, 10-11
compensation and voice of the frontline — one runnable pipeline each (see
[Quickstart](#quickstart)). What large operators run beyond these, and what
each would need to be added with the same rigour, is in
[docs/ROADMAP.md](docs/ROADMAP.md).

## See it work

**Models against the best possible model.** Out-of-time AUC per horizon,
with the oracle ceiling drawn on top — and calibration tight enough
(ECE 1-4%) to sum probabilities into hiring plans:

<table>
  <tr>
    <td width="50%"><img src="docs/figures/auc_vs_ceiling.png" alt="Model AUC vs oracle ceiling"></td>
    <td width="50%"><img src="docs/figures/calibration.png" alt="Calibration, predicted vs observed"></td>
  </tr>
</table>

**A true experiment on pay.** Use case 10 doesn't stop at what-if rescoring:
it reruns the *world* with and without a wage program on paired seeds. The
two arms are the same company — same people, same shocks — until the policy
lands:

<table>
  <tr>
    <td width="50%"><img src="docs/figures/wage_event_study.png" alt="Wage program event study"></td>
    <td width="50%"><img src="docs/figures/wage_programs_roi.png" alt="Wage program ROI with confidence intervals"></td>
  </tr>
</table>

**A methods lesson no real dataset can teach.** The simulator plants **no**
turnover contagion, yet the raw data reproduces the +19% "contagious
turnover" effect from the literature. Stratify on store conditions and it
vanishes — the naive estimate was confounding, and here that is provable:

<table>
  <tr>
    <td width="50%"><img src="docs/figures/contagion.png" alt="Contagion, raw vs adjusted"></td>
    <td width="50%"><img src="docs/figures/shap_beeswarm.png" alt="SHAP beeswarm"></td>
  </tr>
</table>

**The frontline tells you what is broken, if you listen.** An unsupervised
topic model recovers the planted call-center topics (NMI 0.88), and call
volumes carry operational signal — stores that call about scheduling are
the stores with chaotic schedules (r = 0.71):

<table>
  <tr>
    <td width="50%"><img src="docs/figures/callcenter_topics.png" alt="Call center topics by month"></td>
    <td width="50%"><img src="docs/figures/callcenter_linkage.png" alt="Scheduling calls track schedule volatility"></td>
  </tr>
</table>

<details>
<summary><b>📊 Full figure gallery</b> (9 more: demand forecasts, hiring plans, survival curves, funnel, drivers...)</summary>
<br>
<table>
  <tr>
    <td width="50%"><img src="docs/figures/demand_forecast.png" alt="Hourly demand forecast"></td>
    <td width="50%"><img src="docs/figures/tenure_hazard.png" alt="Exit rate by tenure"></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/figures/hiring_plan.png" alt="Hiring plan by district"></td>
    <td width="50%"><img src="docs/figures/headcount_validation.png" alt="Predicted vs actual exits"></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/figures/drivers.png" alt="Turnover drivers"></td>
    <td width="50%"><img src="docs/figures/intervention_value.png" alt="Intervention value"></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/figures/survival_curves.png" alt="Deep survival model retention curves"></td>
    <td width="50%"><img src="docs/figures/hiring_funnel.png" alt="Hiring funnel"></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/figures/callout_seasonality.png" alt="Call-out seasonality"></td>
    <td width="50%"></td>
  </tr>
</table>
</details>

## What it is worth in dollars

Every number is computed under stated, adjustable assumptions (`CostModel`),
on the simulated ~2,900-person company. The per-use-case writeups show the
arithmetic and how it scales with workforce size.

| Question | Answer |
|----------|--------|
| Baseline attrition burn | **$12.9M per year** (2,126 exits, mostly baristas) |
| Demand-driven scheduling vs staff-to-average | **$328 per store-week**, $2.4M/yr chain-wide |
| Best single retention lever found | Scheduling people their desired hours: $240k/yr |
| Targeted retention program (top decile) | **1.76x ROI** vs 1.03x untargeted |
| Targeted 95% pay floor vs blanket +5% raise | **~2.5x more retention per wage dollar** |
| Headcount plan accuracy | 699 predicted vs 710 actual exits over 6 months |
| Merit freeze | "saves" $1.2M on replacement-cost accounting — the trap use case 10 dissects |

> [!IMPORTANT]
> These are exact computations on synthetic data, not promises. Swap in your
> own replacement costs and wage assumptions before quoting any of it
> internally — every parameter is explicit, and the writeups flag what each
> ledger deliberately leaves out.

## What makes this different from the usual attrition demo

Most public attrition projects train a classifier on a static cross-section
and report an inflated AUC. Turnover does not work like that in production.

- **Point-in-time snapshots.** One row per employee per month, features as
  they were known that month, labels from the future window. No leakage;
  there is a test asserting it.
- **Out-of-time evaluation.** Train on early months, test on later months —
  what scoring next quarter actually looks like. Random row splits memorise
  employee identity and overstate everything.
- **Calibrated probabilities.** Isotonic calibration on a later window, so
  probabilities sum into expected headcount losses. That is what turns a
  risk score into a hiring plan.
- **A known answer key.** `ground_truth.json` ships with every dataset;
  driver recovery, SHAP direction, and the oracle ceiling are all tested
  against it.

A deep discrete-time survival network (`SurvivalNN`, PyTorch, optional)
matches the gradient-boosted models on ranking from a single trained
artifact and produces full 12-month retention curves; the GBM stays the
recommended default, and the honest comparison ships in
`reports/nn_vs_gbm_hourly.csv`.

<details>
<summary><b>📈 Headline metrics table</b> (out-of-time test window)</summary>
<br>

| Population | Horizon | Base rate | ROC-AUC | Ceiling | Calibration (ECE) | Lift @ top decile |
|------------|---------|-----------|---------|---------|-------------------|-------------------|
| Hourly | 3 mo | 0.17 | 0.667 | 0.668 | 0.009 | 2.1x |
| Hourly | 6 mo | 0.29 | 0.658 | 0.663 | 0.022 | 1.8x |
| Hourly | 12 mo | 0.45 | 0.653 | 0.666 | 0.045 | 1.6x |
| Salaried | 6 mo | 0.11 | 0.632 | 0.691 | 0.038 | 1.9x |
| Salaried | 12 mo | 0.21 | 0.655 | 0.664 | 0.065 | 2.4x |

Plus: demand forecast WAPE 17.8% vs 24.1% seasonal-naive · schedule
stability 94% week-over-week with zero legal violations · promotion
readiness AUC 0.845 · call-topic recovery NMI 0.88 · pay elasticity
estimates bracket the true coefficient.
</details>

## Quickstart

```bash
git clone https://github.com/immu4989/retail-workforce-analytics
cd retail-workforce-analytics
pip install -e ".[dev]"

python examples/run_pipeline.py        # 1-4: turnover, headcount, drivers      (~2 min)
python examples/run_operations.py      # 5-9: scheduling, call-outs, funnel     (~3 min)
python examples/run_comp_and_voice.py  # 10-11: wage experiments, call topics   (~4 min)
```

Everything lands in `reports/` (CSV/JSON) and `docs/figures/` (PNG).
Optional extras: `".[explain]"` for SHAP, `".[deep]"` for the survival
network.

<details>
<summary><b>🐍 Python quick tour</b></summary>

```python
from workforce_analytics import (
    SimulationConfig, generate, build_snapshots, time_split,
    TurnoverModel, evaluate_with_ceiling, build_hiring_plan,
    InterventionSimulator, stabilize_schedules, CostModel,
)

# 1. Simulate a company (or load your own HRIS extract in the same shape).
result = generate(SimulationConfig())

# 2. Point-in-time snapshots with 3/6/12-month labels.
snaps = build_snapshots(result.person_months, horizons=(3, 6, 12))
train, val, test = time_split(snaps, train_end=36, val_end=44)

# 3. Calibrated multi-horizon turnover model, scored against the ceiling.
model = TurnoverModel("hourly").fit(train, val)
print(evaluate_with_ceiling(model, test))

# 4. "How many baristas does each district need to hire this half?"
preds = model.predict(snaps[snaps["month"] == 48])
plan = build_hiring_plan(preds, result.stores, horizon=6)

# 5. "What would stabilising schedules buy us, in dollars?"
sim = InterventionSimulator(model, snaps[snaps["month"] == 48])
print(sim.run(stabilize_schedules(), "cap schedule volatility", horizon=6,
              cost_model=CostModel()))

# 6. "Why is this employee at risk?" (shap extra)
from workforce_analytics import reason_codes
print(reason_codes(model, snaps[snaps["month"] == 48], horizon=6).head())

# 7. A true experiment on pay (reruns the simulator, paired seeds).
from workforce_analytics import WageProgram, run_wage_experiment
floor = WageProgram("95% pay floor", kind="floor", floor_ratio=0.95, start_month=36)
print(run_wage_experiment(floor, seeds=(0, 1, 2)))
```
</details>

<details>
<summary><b>🗂 Repository layout</b></summary>

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
    compensation.py pay elasticity, wage-program experiments (raise/floor/freeze)
    callcenter.py   synthetic transcripts with hidden topics, NMF topic model
examples/           three pipelines (people, operations, comp & voice)
tests/              50 tests: realism, leakage, calibration, SHAP additivity, accounting
docs/               per-use-case writeups, roadmap, guide to adapting real HRIS data
```
</details>

## Using this with real data

Shape your HRIS extract like `person_months` (one row per employee per
month; the full contract is in
[docs/adapting_to_real_data.md](docs/adapting_to_real_data.md)) and
everything downstream of the generator works unchanged. Two cautions that
matter more than any modelling choice: keep features point-in-time (no
backfilled ratings, no post-termination edits), and treat driver analysis as
hypothesis generation until a lever has been validated with an experiment.

The same document covers the ethical guardrails: aggregate before sharing,
no protected attributes as features, retention actions that are
positive-sum, and the regulatory context for automated employment decisions.
No real employee data was used anywhere in this project; simulator
parameters were set from public retail benchmarks.

## License

MIT. If this repo is useful to you, a ⭐ helps other people find it.
