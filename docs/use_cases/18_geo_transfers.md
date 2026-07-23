# Use case 18: Geographic transfer matching

Match far-commuting employees to closer stores when vacancies open, and buy
retention with no raise and no new hire — just a better assignment of people to
locations, solved optimally.

## The problem

Commute distance is a planted turnover driver in this repo: every kilometre
past five adds to an hourly employee's monthly exit hazard, up to a cap. So a
barista driving 16 km to one store while an identical store six kilometres from
their home is short-staffed is a retention loss the company is choosing not to
prevent. When vacancies open, offering those employees a transfer to a closer
store cuts their commute, cuts their exit hazard, and fills the opening — three
wins from one move. The catch is that it is a *capacitated assignment* problem:
each opening takes one person, each person moves once, and grabbing the biggest
individual saving first (greedy) leaves value on the table. It deserves the
Hungarian algorithm, not a spreadsheet sort.

## The mechanism (published ground truth)

Stores are placed on a plane, clustered by district the way real markets are,
and each employee's home is placed at their known commute distance from their
current store. Commute to any other store is then the straight-line distance
from home. A store's vacancies are its hourly target minus its current head
count. A transfer qualifies when the target store is in the employee's district,
has an opening, and is meaningfully closer.

The payoff is exact because the commute-to-hazard coefficient is published: the
change in a transferred employee's 12-month exit probability is computed
directly from the drop in their commute term, with the base hazard anchored to
the empirical rate so only the commute channel moves.

## Results

At one month on the simulated 139-store chain — 2,372 active hourly employees,
131 stores with 334 open hourly roles, and 420 employees with at least one
commute-reducing transfer available:

| Strategy | Transfers | Exits avoided (12mo) | Commute saved | Retention value |
|----------|-----------|----------------------|---------------|-----------------|
| **Optimal (Hungarian)** | 291 | **14.3** | 1,101 km | **$71,285/yr** |
| Greedy (best-first) | 262 | 13.4 | 1,066 km | $67,240/yr |
| No transfers | 0 | 0 | 0 | $0 |

Two things to take from it. First, the whole exercise is free retention:
~14 exits avoided a year and $71k saved by reassigning people the company was
already going to employ, into openings it was already going to fill. Second,
optimal matching beats greedy — 0.8 more exits avoided and 29 more transfers
placed from the *same* vacancies — because greedy spends a scarce opening on the
first employee who wants it, while the assignment solver keeps that opening for
the employee who has no other option. The gap is modest here because vacancies
are not very contested; it widens exactly when openings are scarce and many
employees compete for them, which is when it matters most.

## Run it

```bash
python examples/run_geo_transfers.py    # ~30 seconds
pytest tests/test_geo.py -q
```

The strategy comparison writes to `reports/geo_transfers.json` and the ranked
transfer plan (who moves from where to where, and the payoff) to
`reports/transfer_plan.csv`. `GeoGroundTruth` sets the store layout and the
minimum commute and reduction thresholds.

## Scope

Transfers are offers, not orders: the payoff is a planning estimate of who would
benefit most from a closer store, to prioritise voluntary transfer conversations
— not a mechanism to relocate people against their will. See the ethics note in
`adapting_to_real_data.md`.
