# Use case 16: Task-level labor standards from order mix

Build the labor requirement up from per-order task times instead of a single
transactions-per-hour rate, and show what the flat rate misses when the order
mix shifts toward mobile.

## The problem

Use case 5 turns transactions into heads with one number: 18 transactions per
labor hour. It is a good average and a bad instant. A mobile order has to be
read, assembled, bagged and staged; a drive-thru order moves through a window;
a front-counter order is rung and handed over in one motion — different labor,
same "transaction." An hour that is 40% mobile needs more people than its count
implies, and an all-front-counter rush needs fewer. As mobile order share has
climbed across the industry, the flat rate has drifted from a good approximation
to a systematic one: it under-staffs exactly the hours that hurt most.

Real workforce-management systems build the requirement from task times per
order type. This module does that on top of use case 5's traffic, with the
task times as published ground truth.

## The mechanism (published ground truth)

Each store-hour's transactions are split across four channels — front counter,
drive-thru, mobile pickup, delivery — by a mix that varies with daypart and
drifts toward mobile over the horizon and by store. Each channel carries a
fully-loaded labor-seconds standard (`TaskGroundTruth.labor_seconds`), and
per-order times carry noise. The standards are scaled so the mix-weighted
average reproduces the flat 18/hour rate — so the flat rate is *right on
average* and any error it makes is purely about mix.

## Results

Across 139 stores and a year of hourly traffic (mean mobile share 27%):

**The two methods agree on average.** Flat-rate labor averages 2.38 hours per
store-hour, task-based 2.34 — a ratio of 0.98. By construction the flat rate is
not biased overall; the question is where it puts the labor.

**It mis-allocates by mix.** The staffing gap tracks mobile share (correlation
**0.42**), and at the extremes:

| Order mix | Flat-rate error |
|-----------|-----------------|
| Top mobile-share decile | **under-provisions by 9%** |
| Bottom mobile-share decile | over-provisions by 12% |

Same total labor, wrong hour-by-hour: the flat rate quietly moves people out of
the mobile-heavy hours (where orders take real assembly labor) and into the
quick front-counter hours. That is the gap a task-level standard closes, and it
widens every year mobile grows.

**The task times are recoverable.** Non-negative least squares of observed labor
seconds on channel order counts recovers the planted per-order standards within
**1.2%**:

| Channel | True | Recovered |
|---------|------|-----------|
| front counter | 135s | 136s |
| drive-thru | 157s | 159s |
| mobile pickup | 261s | 264s |
| delivery | 296s | 300s |

That is the oracle: the labor standard is not something you have to assume from
a vendor table — with order counts and clocked labor, it is estimable, and here
the estimate is checked against the truth that generated it.

## Run it

```bash
python examples/run_task_standards.py    # ~40 seconds
pytest tests/test_tasks.py -q
```

The staffing summary and the task-time recovery write to
`reports/task_standards.json`. `TaskGroundTruth` holds the per-channel labor
seconds and the daypart mix; change them to your own menu and the requirement
moves with it.
