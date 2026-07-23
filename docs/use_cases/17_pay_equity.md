# Use case 17: Pay-equity audit — and validating the audit itself

Run a residual pay-gap audit against a *known* planted gap, so the two things
that decide whether to trust it — does controlling for confounders fix the
estimate, and what are the audit's power and false-positive rate — can be
measured instead of hoped for.

> **This is the "do it carefully or not at all" use case.** The group label here
> is a synthetic, abstract construct (group A / group B) used only to validate
> audit *methodology*. Nothing in this repo infers or uses a real person's
> demographics, and the ethics note in
> [`adapting_to_real_data.md`](../adapting_to_real_data.md) applies in full: on
> real data, never use protected attributes or their proxies as model features,
> and treat a pay-equity audit as a governance exercise run with counsel, not a
> dashboard.

## The problem

A pay-equity audit regresses pay on a group indicator plus the legitimate
drivers — role, tenure, market, performance — and reads the group coefficient.
The regression is easy. Trusting it is not. Omit a control and a confounder (a
group that skews younger, shorter-tenure, lower-cost markets) shows up as bias
that is not there. Keep a real gap but run on a small or noisy sample and it
hides inside the standard error. On real data you cannot see either failure,
because you never know the true gap.

Here the gap is planted, so both are measurable.

## The mechanism (published ground truth)

Each hourly employee is assigned a synthetic group, optionally with the
group-B assignment tilted toward shorter tenure to create confounding, and a
*known* residual gap is applied — group B paid `(1 - gap)` times, with every
legitimate factor held equal. The audit is an OLS of log pay on the group
indicator plus tenure, performance, local market pay, and role fixed effects;
its power and false-positive rate are estimated by replaying the whole
assign-and-audit loop many times.

## Results

On ~12,850 hourly employees with a planted 4% residual gap:

**Recovery, and why controls matter.**

| Scenario | Unadjusted gap | Adjusted gap | Truth |
|----------|----------------|--------------|-------|
| Groups independent of tenure | 3.9% | **4.1%** | 4.0% |
| Group tied to shorter tenure | 7.2% (misleads) | **4.1%** | 4.0% |

With no confounding the raw difference already lands near the truth. The second
row is the point: when group membership correlates with tenure, the *unadjusted*
gap nearly doubles to 7.2% — a number an advocate would quote and a defender
would dispute — while the controlled audit still returns 4.1%. Same confounding
lesson as turnover contagion (use case 9) and pay elasticity (use case 10):
the naive number is an artifact, and here that is provable.

**Power and false-positive rate.** Replaying the audit 200 times:

- **Power 100%** to detect a 2% gap at this employer's scale — with ~13k
  employees the audit is not power-limited even for small gaps, so a *failure to
  find* a gap is informative, not just underpowered.
- **False-positive rate ~7%**, near the nominal 5% test level: run the audit on
  a workforce with no true gap and it rarely cries wolf.
- **Estimate bias ~0.0002** — the audit is essentially unbiased.

The operational takeaway is the honest inversion of the usual worry: at a large
employer the risk is not that the audit misses a real gap, it is that a
mis-specified one (wrong or missing controls) manufactures a gap that is not
there. That is exactly the failure this ground-truth harness lets you catch
before the audit is ever run on real payroll.

## Run it

```bash
python examples/run_pay_equity.py    # ~40 seconds
pytest tests/test_payequity.py -q
```

Recovery under both scenarios and the power/false-positive estimates write to
`reports/pay_equity.json`. `PayEquityGroundTruth` sets the group share, the
planted gap, and the confounding strength.
