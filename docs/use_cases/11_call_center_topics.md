# Use case 11: Employee call-center topic mining (voice of the frontline)

Mine the internal employee-support line — the calls baristas make about pay
problems, schedules, benefits and conflicts — to find out what is actually
going wrong, where, and what to fix first.

## The problem

Large chains run internal support centers for store employees (Starbucks'
Partner Contact Center is the best-known example). The transcripts are the
richest unprompted signal a company has about frontline experience: nobody
calls a support line to fill out a survey. Yet in most organisations the
calls are handled one at a time, tagged inconsistently by agents, and never
analysed as a corpus. The questions worth answering are aggregate ones:
What are the top call drivers? Which spike seasonally and which are drifting
up? Which districts are outliers? And do the calls carry operational signal
worth routing to payroll, scheduling policy, or a district manager?

## What the module does

```python
calls = simulate_calls(person_months)      # hidden true_topic per call
tm = CallTopicModel().fit(calls)           # TF-IDF + NMF, sklearn only
tm.evaluate(calls)                         # NMI / purity vs the hidden labels
operational_linkage(calls, person_months)  # do call rates track real conditions?
```

Every simulated call carries a hidden ground-truth topic, and — the part
that matters — call *propensities depend on simulation state*: employees
paid under market call about pay errors, stores with chaotic schedules call
about scheduling, November brings open-enrollment benefits calls, new hires
call about the app, recent manager changes drive conflict calls. So the
module's claims are checkable, exactly like everywhere else in this repo.

## Results

On 16,990 simulated calls, the unsupervised model recovers the planted
structure: **NMI 0.88, cluster purity 0.95**, with recovered top-terms that
name themselves (a cluster whose top terms are "pay, paid, premium, missing,
overtime, stub" needs no analyst). One honest wrinkle worth knowing about:
the rarest topic (workplace conflict, ~5% of calls) tends to get absorbed
into a larger cluster at k=6 — rare-but-important topics need a larger k or
a supervised pass in production, and sensitive categories should route to
humans regardless.

Seasonality falls out of the volume view — benefits calls spike at open
enrollment, app-tech calls track hiring waves:

![Call topics by month](../figures/callcenter_topics.png)

And the linkage check confirms calls are operational signal, not noise:
store-level scheduling-call rates correlate **+0.71** with true schedule
volatility across stores:

![Call linkage](../figures/callcenter_linkage.png)

The pay-error linkage is directionally right but weak (-0.19) — pay
dispersion is person-level, not store-level, so store aggregation washes it
out. That is itself the analytical lesson: route pay-call analysis by
*employee segment*, scheduling-call analysis by *store*.

## What it is worth

Three consumption modes, in ascending ambition:

* **Deflection.** Scheduling and app-tech questions — the two biggest
  drivers here — are exactly the calls self-service answers well. At a
  typical $6-10 fully-loaded cost per handled call, deflecting a third of a
  chain-scale line's volume is a **seven-figure annual saving** at
  200k-employee scale.
* **Root-cause routing.** A district whose pay-error calls jump 40% month
  over month has a payroll-configuration problem, not 40 coincidences. The
  topic-by-district trend table is a free early-warning system.
* **Retention linkage.** Use case 4 showed schedule chaos drives quits; this
  module shows the same stores generating scheduling calls months earlier.
  Calls are a leading indicator that costs nothing to collect.

Privacy note: transcripts are sensitive. Aggregate before reporting, keep
conflict/HR-case topics out of any automated routing that could identify a
caller, and treat topic mining as a diagnostic on processes, never on
individuals.

## Run it

```bash
python examples/run_comp_and_voice.py
pytest tests/test_comp_and_voice.py -k "topic or linkage or volume" -q
```
