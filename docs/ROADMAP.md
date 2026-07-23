# Roadmap: use cases the industry runs that this repo doesn't cover yet

The thirteen implemented use cases cover the systems that plan, staff and
retain an hourly workforce. This page lists what large operators (Starbucks,
McDonald's, Chipotle, Taco Bell/Yum and their WFM vendors) run beyond them,
and what each would need before it belongs here. The bar for inclusion is
the same one every existing module met: a simulator with published ground
truth, an honest baseline, and out-of-time evaluation — no use case ships as
a demo on a static CSV.

## Shipped since first draft

**Staffing-to-sales elasticity** (use case 12). Quantifies lost sales per
understaffed hour by adding a service-loss term (queue abandonment above a
utilisation threshold, priced at contribution margin) on top of use case 5's
traffic, letting the $35/hour understaffing cost be *derived* (it lands at
$34.71) instead of assumed. See `elasticity.py` and
`docs/use_cases/12_staffing_elasticity.md`.

**Overtime and labor-cost anomaly detection** (use case 13). A punch-clock
layer on the schedule with planted anomalies (time padding, ghost shifts,
buddy punching), detected by robust per-grain residual monitors with measured
precision/recall against the planted log — recovering ~$1.5M/yr of leakage on
the simulated chain. See `punchclock.py` and
`docs/use_cases/13_labor_anomaly.md`.

## Near-term (the current simulator mostly supports them)

**First-90-day onboarding risk.** The washout is already the largest hazard
in the ground truth and the 3-month model captures it; a dedicated view
(new-hire watchlist, training-completion effects, 30/60/90 milestones)
needs training-program events added to the generator.

## Medium-term (new data surfaces required)

**Pay equity audit.** Residual pay-gap analysis after controlling for role,
tenure, market and performance. The simulator would need demographic
attributes assigned with *known* planted gaps so the audit's power and
false-positive rate can be validated — the same oracle trick as everything
else here, and the reason to do this carefully or not at all.

**Exit-interview and pulse-survey NLP.** Text is a different modality: a
synthetic exit-comment generator with themes tied to the true hazard
drivers, then topic extraction validated against those themes. Would pair
naturally with use case 4's driver ranking.

**Task-level labor standards.** Use case 5 converts transactions to heads
with a single service rate; real systems build up from task times (drive-
thru vs front counter vs mobile-order assembly). Needs an order-mix layer
in the traffic simulator.

**Geographic transfer matching.** Commute distance is a planted turnover
driver; matching employees to closer stores when vacancies open is a
bipartite assignment problem with a measurable retention payoff. Needs
store geo-coordinates in the generator.

## Explicitly out of scope

Individual surveillance products — keystroke monitoring, camera analytics,
"productivity scoring" of individuals — are out of scope regardless of
demand. See the ethics section in `adapting_to_real_data.md`; this repo
prices systems that fix schedules, pay and staffing, not systems that
watch people.

Suggestions and PRs are welcome; open an issue with the use case, the data
surface it needs, and the ground-truth mechanism that would let it be
validated.
