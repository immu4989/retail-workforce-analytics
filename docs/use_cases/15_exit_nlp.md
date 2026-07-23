# Use case 15: Exit-interview and pulse-survey NLP

Recover *why* people leave from their own words — and prove the recovered
themes are the real reasons by tying each to the hazard driver that actually
pushed the leaver out.

## The problem

The exit interview is the company's last clear read on why someone quit, and it
arrives as free text nobody has time to code. So it gets skimmed, summarised
into a wordcloud, and ignored. The useful version answers two questions with
numbers: what share of exits are about pay versus schedule versus management,
and — the part every wordcloud skips — are those labels *right*? A topic model
that finds six tidy clusters has proven nothing if the cluster it calls "pay"
is not the one written by the people the company underpaid.

This use case pairs the topic-modelling machinery of use case 11 with the
driver model of use case 4, and adds the check that makes it trustworthy:
alignment against ground truth.

## The mechanism (published ground truth)

Every hourly leaver gets an exit comment whose hidden theme is drawn from their
*exit-state drivers*. Each theme maps to a driver and the risky direction of
it:

| Theme | True driver | Risk direction |
|-------|-------------|----------------|
| pay | pay ratio to market | underpaid |
| scheduling | schedule volatility | volatile |
| commute | commute distance | far |
| management | months since manager change | recent change |
| career | months since promotion | stagnant |
| workload | store staffing ratio | understaffed |

Each driver is standardised across leavers and enters its theme's score on the
risky side, a softmax picks the dominant theme, and the comment is built from
that theme's template bank — with an 18% chance of borrowing a sentence from a
second theme, because real leavers name more than one reason. Scoring on the
driver's *position* rather than its raw value matters: pay ratio barely varies
among leavers, but the ones it pushed out are still the bottom of that thin
distribution, and the theme follows them.

## Results

Across ~10,300 hourly exit comments:

**Recovery.** TF-IDF + NMF (scikit-learn only, no embeddings) recovers the six
planted themes at **NMI 0.93, purity 0.97** — lower than a perfect score
because the cross-theme sentences genuinely muddy the clusters, which is the
point. The top terms read cleanly:

```
pay         | pay, rate, hour, nearby, stores
scheduling  | week, schedule, shifts, unpredictable, hours, childcare
commute     | gas, time, commute, travel, long
management  | management, manager, new, work
career      | career, path, role, advancement, years
workload    | short, people, store, pressure, staffed
```

**Alignment.** This is the claim wordclouds cannot make. For every theme, the
leavers assigned to it are elevated on that theme's true driver versus everyone
else — all six aligned:

| Theme | Driver | Signed gap |
|-------|--------|-----------|
| management | months since manager change | recent by 17 months |
| career | months since promotion | stagnant by 25 months |
| commute | commute km | farther by 5.1 km |
| scheduling | schedule volatility | +2.1 hours of std-dev |
| pay | pay ratio | underpaid |
| workload | store staffing ratio | more understaffed |

The pay and workload gaps are small in raw units because leavers barely vary on
those drivers — but they are correctly signed, which is exactly what you would
hope to detect and could not verify without the ground truth.

**Regrettability.** Splitting themes by voluntary versus involuntary exit
separates the controllable, regrettable reasons (pay, schedule, management)
from the rest — the cut that decides where retention spend should go.

## Run it

```bash
python examples/run_exit_nlp.py    # ~30 seconds
pytest tests/test_exitnlp.py -q
```

Recovery metrics, the alignment table, and the regrettability split write to
`reports/exit_nlp.json`; the cluster top-terms to `reports/exit_theme_terms.csv`.
`ExitNLPGroundTruth` holds the per-theme weights.

## Scope

Exit text is aggregated to themes for routing to the responsible team (payroll,
scheduling policy, store leadership), never used to score an individual. The
same generator shape works for pulse-survey free text; see the ethics note in
`adapting_to_real_data.md`.
