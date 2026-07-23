# Use case 12: Staffing-to-sales elasticity — deriving the cost of an understaffed hour

Replace the flat "$35 per understaffed person-hour" that use cases 5 and 6
assume with a service-loss mechanism, so the number is derived from primitives
and the sales cost of short-staffing can be measured rather than asserted.

## The problem

Every scheduling and reserve-staffing dollar figure in this repo rests on one
assumption: that a person-hour of understaffing costs about $35 in lost
transactions and service decay. It is a reasonable industry number, but it is
still a number typed into a config. When a store manager pushes back — "prove
that pulling one barista at 12pm costs us anything" — an assumed constant is
not an answer, and when finance asks a workforce team to justify labor budget,
"$35 because we said so" does not survive the meeting.

The honest version derives the cost from things you can defend: how many
transactions a worker can process, how much margin a transaction carries, and
what happens to a queue when you take a head off the floor.

## The mechanism (published ground truth)

Each worker can process up to `capacity_per_head` transactions an hour (22 by
default), comfortably above the 18/hour planning standard that
`required_staff` uses — so a store that staffs to the standard has headroom
and loses nothing. Cut below it and utilisation climbs past a threshold
(0.85); a queue forms, a share of arriving customers balk or renege, and each
lost transaction costs `gross_margin_per_txn` in contribution ($4.00 — roughly
a $9 ticket at ~45% margin). Abandonment rises linearly with utilisation to a
cap, a readable stand-in for the convex delay curve of an M/M/c queue.

```python
from workforce_analytics import derive_understaffing_cost, service_loss_curve

derive_understaffing_cost(traffic)
# {'derived_cost_first_head_short': 34.71,   # vs the assumed $35
#  'avg_marginal_cost_3_heads_short': 63.06, # service loss is convex
#  'loss_rate_at_required_staffing': 0.0,    # meeting the standard loses nothing
#  'ratio_derived_to_assumed': 0.992, ...}
```

## Results

The service-loss curve across all 139 stores and two years of hourly traffic —
mean margin lost per store-hour as the floor is cut below the requirement:

| Heads short | Lost margin / hour | Marginal cost of that head | Store-hours affected |
|-------------|-------------------|----------------------------|----------------------|
| 0 (at standard) | $0.00 | — | 0% |
| 1 | $28.98 | **$34.71** | 84% |
| 2 | $74.32 | $85.83 | 90% |
| 3 | $89.76 | $68.63 | 90% |

Two things matter here. First, the marginal cost of the **first** head short
is **$34.71** — the mechanism reproduces the scheduler's assumed $35 to within
1%, so use case 5's savings figures stand on a derived number now, not a
guess. Second, the cost is **convex**: the second head short costs far more
than the first (the store is already queuing), and the marginal only turns
over once abandonment saturates its cap. A flat per-hour cost understates the
damage of deep short-staffing, which is exactly why targeted coverage of the
worst hours beats spreading labor evenly.

## Elasticity

The same curve gives the number finance asks for — percent change in sales per
percent change in labor — evaluated where labor actually moves sales, at a
short-staffed operating point:

| Operating point | Sales elasticity w.r.t. labor |
|-----------------|-------------------------------|
| One head short of standard | 0.67 |
| Two heads short of standard | 0.82 |

At or above the labor standard the store is on the flat of the curve and the
elasticity is ~0 — adding staff to an already-adequate hour buys no sales,
only cost. The elasticity is meaningful precisely when a store is short, and
it climbs as the shortfall deepens. That asymmetry is the argument for
demand-driven scheduling: labor has high marginal sales value in the hours
you are short and none in the hours you are long, so the win is moving hours
between them, not adding them.

## Why a simulator makes this honest

On real POS data you can only observe sales at the staffing levels that
actually occurred, and staffing correlates with everything (busy stores staff
up), so a naive sales-on-labor regression is hopelessly confounded — the same
trap use case 10 shows for pay. Here the service-loss mechanism is ground
truth, so the derived cost and elasticity are exact by construction, and a
downstream estimator built on observational store-hours can be scored against
them. The point is not that $35 is right; it is that the number is now
derived, adjustable, and checkable.

## Run it

```bash
python examples/run_operations.py    # prints the derived cost + elasticity in step 2
pytest tests/test_elasticity.py -q
```

The derived cost, its ratio to the assumed value, and the elasticity are
written to `reports/staffing_elasticity.json`. Change `ServiceConfig`
(`gross_margin_per_txn`, `capacity_per_head`) to your own economics and the
cost moves with it.
