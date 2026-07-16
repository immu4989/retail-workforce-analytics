# Use case 7: Hiring funnel analytics and requisition timing

Find where the hiring pipeline leaks, know how long each role really takes to
fill, and post requisitions early enough that the headcount plan from use
case 3 actually lands.

## The problem

Frontline hiring is a funnel with brutal leaks. In this simulation's ground
truth (set to published frontline benchmarks): roughly half of applications
survive screening, **only ~62% of scheduled interviews are attended** — the
single biggest leak, and a notorious one in hourly hiring — half of attended
interviews convert to offers, ~18% of offers are declined, and 15% of
accepted candidates never show up for day one.

The reference point for what fixing this is worth is public: Chipotle's
conversational-AI funnel work cut candidate time-to-start from 12 days to 4
and nearly doubled application completion, which is what let hiring keep
pace with 150%+ hourly turnover. You don't need their bot to get the first
win; you need to know which stage of *your* funnel is leaking and how long
fills really take.

## What the module does

```python
reqs = simulate_funnel(districts, n_months=24)   # req-level, ground truth published
funnel_report(reqs, by="district_id")            # stage conversions + fill times
timing = req_timing(hiring_plan, reqs)           # when to post, per district-role
```

## Results

![Hiring funnel](../figures/hiring_funnel.png)

31,548 applications become 4,123 day-one starts: **13% end to end**, meaning
every barista seat costs ~8 applications and every stage-rate point moves
real money. Fill times differ by an order of magnitude across roles, and
that spread is the actionable output:

| Role | Median days to fill | p90 |
|------|---------------------|-----|
| Barista | 30 | 91 |
| Shift supervisor | 41 | 135 |
| Assistant store manager | 63 | 190 |
| Store manager | 83 | 224 |

`req_timing` joins these onto the 6-month hiring plan: barista reqs can be
posted in waves through the half, while **41 of the 48 district-role batches
containing any leadership role are flagged post-now** — a p90
store-manager fill takes most of the planning horizon by itself.

## What it saves

Three levers, priced with the other use cases' outputs:

* **Interview no-shows.** Raising the 62% show rate to 75% (reminder
  cadence, self-scheduling, shorter apply-to-interview gaps) yields ~20%
  more starts from the same application volume — recruiting spend you don't
  have to make.
* **Posting on time.** Every leadership vacancy-month avoided saves the
  interim-coverage scramble that use case 2 showed destabilises a store's
  hourly team (the manager-change hazard bump), on top of lost-productivity
  costs.
* **Funnel-aware headcount planning.** The use-case-3 plan says D06 needs 102
  baristas this half; this module converts that into "post ~14 reqs by week
  2", which is the sentence a recruiting team can execute.

## Run it

```bash
python examples/run_operations.py
pytest tests/test_operations.py -k "funnel or timing" -q
```
