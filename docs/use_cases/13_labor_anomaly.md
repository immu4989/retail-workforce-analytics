# Use case 13: Overtime and labor-cost anomaly detection

Put a punch-clock layer on top of the schedule, plant the three ways payroll
leaks — time padding, ghost shifts, buddy punching — and detect each with a
robust residual monitor whose precision and recall are measured against ground
truth before it ever touches real payroll.

## The problem

A large hourly workforce loses real money to labor-cost abuse: employees
padding punches past their scheduled hours, managers charging ghost shifts
with no one on the floor, and buddy punching — an absent employee clocked in
by a coworker. The hard part is not spotting one bad punch; it is telling
systematic abuse apart from the honest noise of a time clock, across every
employee and every store-month, without burying store managers in false
positives. A monitor that flags 5% of your workforce every week gets switched
off in a month.

This is a loss-prevention problem, and it is one the rest of this repo's
machinery handles well: plant the anomalies with a known log, then measure how
cleanly they can be recovered.

## The mechanism (published ground truth)

Honest punched hours are scheduled hours times symmetric, mean-zero noise (the
ordinary spread of clock-in/out). On top of that, `AnomalyGroundTruth` plants
three patterns at loss-prevention-plausible rates:

* **Time padding** — 6% of employees punch 8–25% over schedule, every month.
* **Ghost shifts** — 3% of store-months carry 6–18% excess aggregate labor.
* **Buddy punching** — a rare one-off spike (20–60%) on individual
  employee-months, kept off the padder population so the two don't blur.

## Three grains, three detectors

Each pattern lives at a different grain, so each gets its own monitor built on
a median/MAD robust z-score (outlier-resistant, so a handful of big anomalies
don't inflate the scale and mask the rest):

| Anomaly | Detector grain | Signal |
|---------|----------------|--------|
| Time padding | across employees | median monthly overage is a population outlier |
| Ghost shifts | across store-months | aggregate punched-over-scheduled is an outlier |
| Buddy punching | within each employee | a single month spikes against the employee's own baseline |

The buddy detector is the subtle one: scoring *within* each employee is exactly
what separates a buddy-punched week from a habitual padder, whose months all
look alike and so raise no internal outlier. And the padding detector uses the
employee's median month, not the mean, so a single buddy-punched month can't
make an otherwise-honest worker look like a padder.

## Results

Across 139 stores and five years of hourly punches, scored against the planted
labels:

| Detector | Precision | Recall | Average precision |
|----------|-----------|--------|-------------------|
| Time padding | 0.95 | 0.99 | 1.00 |
| Ghost shifts | 0.83 | 0.92 | 0.96 |
| Buddy punching | 0.83 | 0.84 | 0.90 |

Time padding is nearly perfectly separable because it is persistent — averaged
over months, a padder stands clear of the pack. Ghost and buddy detection are
harder and the precision reflects it: some clean store-months host a padder or
a buddy spike and tip over the aggregate threshold, which is a real
false-positive mode, not a bug. Every threshold is a knob (`z_threshold`), and
because the ground truth is known you can trade precision against recall on the
curve instead of guessing.

## What it recovers

Pricing the net excess hours at a $21 loaded wage:

| Leakage source | Annualised |
|----------------|-----------|
| Time padding | $0.78M |
| Ghost shifts | $0.31M |
| Buddy punching | $0.45M |
| **Total** | **$1.47M** |

That is on this simulated 139-store chain; the excess is measured net, so the
honest mean-zero punch noise cancels and what remains is the planted abuse. At
a national footprint the same leakage rate is an eight-figure line item, which
is why chains run exactly this kind of monitoring — and why getting its
false-positive rate right, provably, is the whole game.

## Run it

```bash
python examples/run_labor_anomaly.py    # ~40 seconds
pytest tests/test_punchclock.py -q
```

The detection metrics and dollar leakage are written to
`reports/labor_anomaly.json`. `AnomalyGroundTruth` sets the planted rates and
magnitudes; the detector thresholds are arguments to each `detect_*` function.

## Scope

This prices systems that recover money the company is already losing to abuse.
It is deliberately not individual productivity surveillance — see the ethics
note in `adapting_to_real_data.md`. The unit of action is a flagged
store-month or a review-worthy employee pattern handed to a loss-prevention
partner, not a real-time score on a worker's screen.
