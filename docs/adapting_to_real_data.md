# Adapting this pipeline to real HRIS data

The simulator exists because real HR data cannot be shared. Everything
downstream of it (snapshots, models, headcount plans, driver analysis) is
plain pandas + scikit-learn and works on any table with the right shape.

## The contract: a person-month table

Build one row per employee per month they were active, with these columns:

| Column | Type | Notes |
|--------|------|-------|
| `month` | int | months since the start of your extract (0, 1, 2, ...) |
| `employee_id` | any | one employment spell per id (give rehires a new id) |
| `store_id`, `district_id` | str | or whatever your org units are |
| `role` | str | used to route employees to the hourly vs salaried model |
| `tenure_months` | int | as of that month |
| `age_band` | str | banded, never raw date of birth |
| `is_student`, `second_job` | 0/1 | drop if you don't have them |
| `pay_rate` | float | that month's rate |
| `pay_ratio` | float | pay vs the local market median for the role. This is the single most valuable engineered feature; source market data from BLS OES or a comp vendor |
| `schedule_volatility` | float | std-dev of weekly scheduled hours, last 4 weeks |
| `scheduled_hours`, `hours_gap` | float | `hours_gap` = desired minus scheduled, floored at 0. Desired hours come from availability forms or scheduling systems |
| `commute_km` | float | geocode home and store zip centroids; never store addresses |
| `months_since_mgr_change` | int | months since the store's manager changed |
| `months_since_promotion`, `months_since_raise` | int | from job-change history |
| `performance_rating` | float | most recent rating *as of that month* |
| `store_staffing_ratio` | float | active headcount / target, that month |
| `district_understaffed_share` | float | share of stores in the district below 85% staffing |
| `district_unemployment` | float | BLS LAUS by county, lagged one month |
| `terminated` | 0/1 | 1 only in the employee's final month |
| `termination_type` | str | `voluntary` / `involuntary`; lets you model regrettable attrition separately |

Then:

```python
snaps = build_snapshots(your_person_months, horizons=(3, 6, 12))
```

## The mistakes that actually sink these projects

**Backfilled features.** HRIS systems silently overwrite history: a rating
entered in March is stamped onto January, a final pay adjustment lands on the
termination row. Every feature must be reconstructed from effective-dated
records as it was known at snapshot time. This is where most of the
engineering effort goes, and it is the difference between a real 0.67 AUC and
a fake 0.85 that collapses in production.

**Random splits.** Use `time_split`. A model that looks great on a random
split is memorising employee identity through their repeated monthly rows.

**Uncalibrated scores.** If probabilities feed a hiring plan, calibrate on a
later window (the `TurnoverModel.fit` signature forces this) and check
group-level calibration with `validate_expected_attrition` every quarter.

**Label leakage through terminations in progress.** Exclude employees who have
already given notice at the snapshot date, or the model learns to predict
paperwork.

**Treating drivers as causal.** Permutation importance on observational data
gives you ranked hypotheses. Validate a lever with a pilot (a scheduling
policy in 50 stores, a targeted comp adjustment) before scaling spend.

## Ethical guardrails

These models score people. Treat that with respect:

- Aggregate before sharing: district and store rollups for planning; individual
  scores only to HR partners with a defined retention playbook, never to the
  employee's own manager as a raw list.
- Never use protected attributes (or proxies like exact age) as features.
  `age_band` is included here because the simulator uses it as a ground-truth
  driver; consider dropping it in a real deployment and audit error rates
  across demographic groups either way.
- Retention actions should be positive-sum (better schedules, market pay,
  career paths), not pre-emptive termination of "flight risks".
- Check your jurisdiction's rules on automated employment decisions (e.g.
  NYC Local Law 144, EU AI Act high-risk category) before deploying.
