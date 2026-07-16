# Use case 9: Turnover contagion, and why the naive estimate misleads

"Turnover is contagious" is one of people analytics' most repeated claims,
and the raw numbers back it up almost everywhere — including here. This use
case is different from the others: it is a *methods lesson* built on the one
thing this repo has that real HR data never does — a known ground truth.

## The setup

The simulator plants **no peer-exit effect whatsoever**. No employee's hazard
depends on coworkers leaving. Whatever "contagion" appears in the data is,
by construction, something else.

Exposure is measured the way the published studies measure it: the share of
an employee's store team that exited in the trailing 3 months, versus their
own exit in the current month.

## What the data shows

![Contagion, raw vs adjusted](../figures/contagion.png)

| Teammates lost, prior 3 months | Raw relative risk | Adjusted for store conditions |
|--------------------------------|-------------------|-------------------------------|
| <5% | 1.00 | 1.00 |
| 5-15% | 0.91 | 0.92 |
| 15-30% | 1.01 | 0.92 |
| 30%+ | **1.19** | **0.93** |

Raw: employees who just watched a third of their team leave are 19% more
likely to exit this month. Presented alone, that row funds a "contagion
intervention". But stratify on the store conditions the ground truth
actually uses — understaffing, a recent manager change, tenure mix — and the
gradient doesn't just shrink, it disappears. The high-churn store was a bad
store *before* the exits; the exits and the elevated risk share causes
(the understaffing spiral and manager-change destabilisation are both real
mechanisms in this simulator, as in real stores).

## Why this matters beyond the simulator

On real data, genuine contagion probably exists — departing coworkers
transmit information about outside options and lower the social cost of
leaving, and well-identified studies find effects. The lesson is not
"contagion is fake"; it is that **the naive exposure gradient is not
evidence of it**, because common causes produce the identical signature.
Run `contagion_analysis` before budgeting for peer-exit-triggered
interventions:

```python
from workforce_analytics import contagion_analysis
tables = contagion_analysis(person_months)   # raw + stratified gradients
```

If the adjusted gradient survives stratification on everything your data
knows about store conditions, you may have contagion. If it collapses — as
it does here, where we know the answer — the money belongs on the store
conditions instead: staffing, manager stability, schedules. Conveniently,
those are use cases 3, 4 and 5.

## Run it

```bash
python examples/run_operations.py
pytest tests/test_operations.py -k contagion -q
```
