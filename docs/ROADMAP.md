# Roadmap: use cases the industry runs that this repo doesn't cover yet

The seventeen implemented use cases cover the systems that plan, staff and
retain an hourly workforce. This page lists what large operators (Starbucks,
McDonald's, Chipotle, Taco Bell/Yum and their WFM vendors) run beyond them,
and what each would need before it belongs here. The bar for inclusion is
the same one every existing module met: a simulator with published ground
truth, an honest baseline, and out-of-time evaluation — no use case ships as
a demo on a static CSV.

## Shipped since first draft

Every near-term item from the original roadmap has since been built to the same
bar. What remains below needs new data surfaces.

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

**First-90-day onboarding risk** (use case 14). A dedicated new-hire view —
day-30 washout model against an oracle ceiling, 30/60/90 milestone retention,
and a ranked watchlist — with training completion added as a self-contained
onboarding process (like the call-out simulator) rather than by touching the
core generator. The training-completion gap is deliberately part causal, part
onboarding-quality confounding, mirroring use cases 9 and 10. See
`onboarding.py` and `docs/use_cases/14_onboarding.md`.

**Exit-interview and pulse-survey NLP** (use case 15). A synthetic
exit-comment generator whose hidden theme is tied to the true hazard driver
behind each leaver's exit, recovered unsupervised (TF-IDF + NMF, NMI 0.93) and
validated for *alignment* — the pay-theme cluster is written by the leavers the
hazard model underpaid. See `exitnlp.py` and `docs/use_cases/15_exit_nlp.md`.

**Task-level labor standards** (use case 16). An order-mix layer (front
counter / drive-thru / mobile / delivery) on the traffic simulator with
per-channel task times, showing the flat 18/hour rate under-provisions the
highest-mobile hours by ~9% even though it agrees on average; the task times
are recoverable within ~1%. See `tasks.py` and
`docs/use_cases/16_task_standards.md`.

**Pay-equity audit** (use case 17). Residual pay-gap analysis with a synthetic
group and a *known* planted gap, so the audit's recovery, the confounding fixed
by controls, and its power and false-positive rate are all measured. Done
carefully, with the group as a methodology-validation construct only. See
`payequity.py` and `docs/use_cases/17_pay_equity.md`.

## Still on the roadmap (new data surfaces required)

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
