# Use case 10: Compensation analytics — what pay changes actually buy

Estimate how pay increases and decreases move retention, then put four real
policies — a blanket raise, a targeted pay floor, a seasonal wage program,
and a merit freeze — through true counterfactual experiments and price them.

## The problem

Pay is the retention lever leadership asks about first and trusts least.
Every observational estimate is confounded (well-paid stores are well-run
stores), every vendor deck overstates it, and the finance question — "if we
spend $X on wages, what comes back?" — almost never gets an answer with an
uncertainty interval attached. Meanwhile pay *cuts* happen silently: a merit
freeze while market wages drift upward is a real-wage cut nobody has to
announce, and its cost shows up a year later as attrition nobody attributes.

## Three levels of rigour

**1. The regression a real team can run** (`pay_elasticity`) — monthly exit
hazard on pay position, naive and adjusted:

| Specification | Pay coefficient (log-odds) | Hazard change per +10% pay |
|---------------|---------------------------|-----------------------------|
| Naive (pay only) | -3.09 | -26.5% |
| Adjusted (standard controls) | -1.91 | -17.4% |
| **True (generator coefficient)** | **-2.50** | **-22.1%** |

The two estimates *bracket* the truth: the naive one absorbs everything
else bad stores do, the adjusted one over-controls and attenuates. On real
data you never see the third row; knowing the failure modes' direction is
the point of running this on a simulator first.

**2. Model what-ifs** — use case 4's intervention simulator with
`pay_shift(pct)`, instant but only as causal as the model.

**3. True experiments** (`run_wage_experiment`) — rerun the simulated world
with and without a `WageProgram` on the same seeds. Identical random draws
mean the arms are the same company, same people, same shocks, until the
program lands:

![Event study](../figures/wage_event_study.png)

## The four programs, priced

Paired runs on 3 seeds, evaluated over 24 post-program months, replacement
costs from use case 1's cost model, 90% bootstrap CIs:

| Program | Exit reduction | Exits avoided | Extra wage bill | Net (90% CI) |
|---------|----------------|---------------|-----------------|--------------|
| Blanket +5% market adjustment | 6.0% | 220 | $4.0M | **-$2.6M** [-3.1, -2.1] |
| Ongoing 95% pay floor (targeted) | 2.7% | 106 | $0.8M | **-$0.1M** [-0.5, +0.2] |
| Seasonal +3% every July | 8.0% | 295 | $4.9M | **-$3.0M** [-3.2, -2.9] |
| 12-month merit freeze | -0.8% (worse) | -27 | -$1.4M (saved) | **+$1.2M** [+1.0, +1.5] |

![Program ROI](../figures/wage_programs_roi.png)

Three findings worth internalising:

**Targeting beats blanket by ~2.5x per dollar.** The floor spends $7.4k per
exit avoided; the blanket raise spends $18k, because most of its budget
lands on people who were staying anyway. The floor is the only program whose
CI touches break-even on retention alone.

**Raises don't self-fund through retention at realistic elasticities.**
Even at the true -2.5 coefficient, a 5% raise's replacement savings recover
about a third of its cost. Anyone whose deck shows a wage increase paying
for itself purely via turnover is assuming an elasticity this simulator —
calibrated to the observational literature — cannot produce.

**The freeze's "+$1.2M" is the accounting trap, on purpose.** This ledger
counts replacement costs only. It is blind to what a chain actually buys
with experienced staff — throughput, service quality, shrink, the store's
capacity to train its own replacements — and to the compounding erosion of
repeated freezes. Real evidence (the good-jobs literature) locates most of
pay's return in productivity, not turnover line items. Present the freeze
row with that caveat or don't present it at all.

## Run it

```bash
python examples/run_comp_and_voice.py     # ~4 minutes: paired simulations
pytest tests/test_comp_and_voice.py -k "wage or freeze or floor or elasticity" -q
```
