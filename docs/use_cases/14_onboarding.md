# Use case 14: First-90-day onboarding risk and the new-hire watchlist

Give the first 90 days their own view: a day-30 washout model, a 30/60/90
retention curve, and a new-hire watchlist — with training completion as the
signal every onboarding program tracks, and an honest account of what it
actually means.

## The problem

The largest hazard in this repo's ground truth is the new-hire washout, and
the 3-month turnover model (use case 1) already captures it. But a store
manager does not want a probability buried in the general risk model; they want
a short list of *this month's* new hires who are about to walk, and the reason
for each, in time to do something in the first 30 days. Half of hourly retail
hires leave within 90 days — the single highest-leverage retention window there
is, and the one the steady-state models are worst positioned to serve because
the signals that matter (did training finish, is the schedule chaotic in week
two) are onboarding-specific.

## The mechanism (published ground truth)

Following the call-out simulator (use case 6), onboarding is a self-contained
process layered on the real new-hire cohort, decoupled from the core
termination model so it can carry its own driver. Each hire's first-90-day
trajectory comes from a published `OnboardingGroundTruth`: a per-month washout
hazard over their day-30 signals (schedule volatility, hours gap, commute,
student status, store understaffing) plus a store-level onboarding-quality
term, and a 30-day training-completion flag.

The one subtlety worth stating plainly: **training completion is both a mild
protective effect and a marker of onboarding quality.** A shared store-level
quality term drives both finishing training and staying, so the raw
completer/non-completer gap overstates what mandating training would buy. This
is the same confounding lesson as turnover contagion (use case 9) and pay
elasticity (use case 10) — and because the ground truth is published, the
causal part and the marker part are both known.

## Results

Across ~10,500 hourly new hires:

**The washout model tracks the ceiling.** Scored on held-out hires, the day-30
model reaches an AUC of **0.682 against an oracle ceiling of 0.745 — 92% of the
best any model could do**. Onboarding is *more* predictable than steady-state
turnover (whose ceiling is ~0.67): the day-30 signals are strong, which is
exactly why a dedicated view pays off. The top-risk quintile washes out at
**74%** versus a 49% base rate.

| Milestone | Retention |
|-----------|-----------|
| Day 30 | 71% |
| Day 60 | 57% |
| Day 90 | 51% |

**The training gap is real but oversold.** Completers wash out at 43% versus
66% for non-completers — a 24-point gap. Most of an onboarding vendor's deck
stops there. But the ground truth plants only a modest *causal* training
effect; the rest of the gap is onboarding-quality confounding. Remove the
shared quality driver in the simulator and the gap collapses, which is the
proof. The honest read: finishing training is an excellent *early-warning
signal* and a weak *lever* — flag the non-completers, but don't expect a
mandate to close the whole gap.

## The watchlist

The operational output is a ranked day-30 list per store, each hire tagged with
plain-language reasons:

```
emp 9622 @ S124  risk 0.95  [training incomplete, volatile schedule, short on hours, understaffed store]
emp 6951 @ S047  risk 0.94  [training incomplete, volatile schedule, short on hours]
```

Every reason maps to an onboarding action — finish the training, stabilise the
schedule, close the hours gap, staff the store — rather than a score handed to
a manager with no next step.

## Run it

```bash
python examples/run_onboarding.py    # ~30 seconds
pytest tests/test_onboarding.py -q
```

The model metrics and milestone curve write to `reports/onboarding.json`, and
the ranked list to `reports/new_hire_watchlist.csv`. `OnboardingGroundTruth`
holds the washout and training coefficients; the watchlist size is the
`top_frac` argument.

## Scope

The watchlist is a retention tool: its actions are positive-sum (training,
schedule, hours, staffing), handed to the people who onboard new hires, not a
flight-risk score used to pre-emptively cut them. See the ethics note in
`adapting_to_real_data.md`.
