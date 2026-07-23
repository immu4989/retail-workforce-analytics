# Roadmap: use cases the industry runs that this repo doesn't cover yet

The nine implemented use cases cover the systems that plan, staff and retain
an hourly workforce. This page lists what large operators (Starbucks,
McDonald's, Chipotle, Taco Bell/Yum and their WFM vendors) run beyond them,
and what each would need before it belongs here. The bar for inclusion is
the same one every existing module met: a simulator with published ground
truth, an honest baseline, and out-of-time evaluation — no use case ships as
a demo on a static CSV.

## Near-term (the current simulator mostly supports them)

**Staffing-to-sales elasticity.** Quantify lost sales per understaffed hour
by joining use case 5's traffic to staffing levels. The generator already
couples understaffing to schedules and attrition; adding a service-loss
term (queue abandonment above a transactions-per-labor-hour threshold)
would let the $35/hour understaffing cost be *derived* instead of assumed.

**Overtime and labor-cost anomaly detection.** Punch-clock data =
scheduled hours + noise + injected anomalies (systematic OT inflation,
ghost shifts, buddy punching), then robust per-store residual monitoring
with a measurable precision/recall on the planted anomalies. Needs a
punch-data layer on top of the use case 5 scheduler. (The planted-bug
pattern this would use already ships for data-quality auditing — see
`realdata.py` and `make_messy_extract`.)

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
