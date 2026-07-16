# Use case 4: What drives turnover, and what fixing it is worth

Move from "who is at risk" to "which operational levers reduce the risk, and
what is each worth in dollars".

## The problem

A risk score changes nothing by itself. The decisions that reduce turnover
are operational: pay positioning against the local market, schedule
stability, staffing levels, promotion velocity, manager continuity. Leaders
will fund exactly one or two of these per year, so the analysis has to rank
them, price them, and be honest about which correlations are actionable
levers versus which are merely useful for targeting (commute distance
predicts attrition well, but no company can move anyone's apartment).

## What the toolkit does

Three views, each answering a different question:

**Which features does the model rely on?** Permutation importance on
out-of-time data, with every feature tagged *actionable* or *contextual*:

![Drivers](../figures/drivers.png)

**How does each feature move individual predictions?** SHAP values from the
booster, with per-employee reason codes for HR partners:

![SHAP beeswarm](../figures/shap_beeswarm.png)

```
p_6m = 0.90   1 month of tenure · 14 km commute · scheduled 14 hours/week under desired
p_6m = 0.83   1 month of tenure · paid 10% below local market · age 16-20
```

One implementation detail worth stealing: shap's TreeExplainer silently
returns non-additive attributions for scikit-learn boosters trained with
native categorical splits. This repo ordinal-encodes categoricals instead
(costing nothing measurable on ordered categories) and ships a test that
asserts SHAP additivity to the model margin plus ground-truth direction on
pay and tenure. If your SHAP charts have ever looked subtly wrong, check
this first.

**What is a policy change worth?** `InterventionSimulator` rescores the
current workforce under a hypothetical policy and prices the difference:

![Intervention value](../figures/intervention_value.png)

| Intervention (hourly population, 6 months) | Exits avoided | Value |
|--------------------------------------------|---------------|-------|
| Schedule people the hours they asked for | 23.0 | $120k |
| Cap schedule volatility at the median | 9.9 | $53k |
| Bring everyone to 95% of market pay | 7.2 | $38k |

## What the analysis found

On this simulated company the ranking is unambiguous: **under-scheduling is
the cheapest turnover lever on the board.** Closing the gap between desired
and scheduled hours is worth about $240k a year across 2,400 hourly
employees, triple the value of a pay floor, and unlike a pay floor its direct
cost can be negative (more scheduled hours from workers who wanted them
typically displaces overtime and new-hire training). Scaled to a
200,000-employee chain, the same per-employee effect is a **$20M per year**
lever. Tenure dominates raw predictive power, but tenure is not a lever;
scheduling is. That distinction, actionable versus contextual, is the entire
point of the analysis.

Because the data is synthetic, this finding is checkable: the generator's
ground truth really does penalise hours-gap and schedule volatility
(`ground_truth.json` ships with the data), and `ground_truth_comparison`
lines the recovered ranking up against the true coefficients. The pipeline
demonstrably recovers the drivers that were planted.

## The causal caveat, stated plainly

On synthetic data these estimates are causal because the features cause the
outcomes by construction. On real HR data they are correlations refracted
through a model. Treat the ranking as a portfolio of priced hypotheses:
pick the top lever, pilot it in 50 stores against matched controls, measure,
then scale. The simulator is also useful here as a rehearsal space: the
`n=1` pilot-design mistakes are free when the ground truth is known.

## Run it

```bash
python examples/run_pipeline.py     # writes drivers, SHAP, reason codes, intervention values
pytest tests/test_impact_and_explain.py -q
```
