# Use case 8: Internal mobility — promotion readiness and bench strength

Know who is ready to move up, and whether each district's internal bench can
cover the leadership vacancies the turnover models say are coming.

## The problem

Retail promotes internally at around 9%, far below other industries, even
though the internal path is cheaper and stickier on every axis this repo can
measure: an externally hired store manager costs ~$35k to land (use case 2),
takes a median 83 days to fill (use case 7), and arrives not knowing the
store — while the promotion itself *reduces* the promoted employee's own
exit hazard (a ground-truth effect here, and a robust real-world finding).
Districts that discover a thin bench the week a store manager resigns end up
buying the expensive option by default.

## What the module does

```python
events = promotion_events(person_months)              # role transitions, observed
panel = build_promotion_panel(snaps, person_months)   # promoted within 6m label
model = PromotionModel(horizon=6).fit(train, val)
bench = bench_strength(model.predict(now), salaried_model.predict(now), stores)
```

One framing note that matters: the model predicts *who the current process
will promote*. That makes it two tools in one — a planning tool (readiness
scores, bench counts) and an **audit tool**: if the top decile of predicted
promotions skews on anything other than performance and tenure-in-role, the
promotion process itself does, and that is worth knowing before any
fairness review does it for you.

## Results (out-of-time test window)

| Metric | Value |
|--------|-------|
| Promotion base rate (6 months, hourly) | 3.5% |
| ROC-AUC | **0.845** |
| Precision in top 5% | 16.7% (4.8x base rate) |

Promotion is far more predictable than turnover (compare AUC 0.85 vs the
0.67 ceiling-limited turnover models) because it is process-driven: the
simulator promotes on tenure gates, performance and vacancy timing, exactly
like a real chain, and the model recovers that process.

Bench strength, per district, at the current scoring month: every district
holds 21-43 ready-now shift supervisors against 2-6 expected leadership
vacancies over the next year — **coverage ratios of 5-11x**, a healthy
pipeline. The table's job is the day that changes: a district drifting
toward coverage < 1 has locked in slow, expensive external fills months
before anyone feels it.

## What it saves

Each leadership vacancy filled internally instead of externally avoids most
of the $27.5-35k replacement cost and roughly two months of vacancy
(use case 7's fill times), plus the hourly-team destabilisation that a
manager gap causes (use case 2). For this simulated chain's ~66 leadership
exits a year, shifting the internal-fill share from the simulator's ~55% to
75% is worth roughly **$400k a year**, and the bench table shows exactly
which districts have the depth to do it.

## Run it

```bash
python examples/run_operations.py
pytest tests/test_operations.py -k "promotion or bench" -q
```
