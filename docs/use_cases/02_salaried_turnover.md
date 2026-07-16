# Use case 2: Turnover risk for store leadership

Predict, for every store manager and assistant store manager, the probability
of leaving within the next 6 and 12 months.

## The problem

Store managers churn far less than baristas (19% vs 63% annualised in the
simulation, in line with retail benchmarks), but each exit costs an order of
magnitude more and the damage compounds. Replacing a store manager runs about
50% of salary once recruiting, ramp time and the interim-coverage scramble
are counted: **$35,000 per SM and $27,500 per ASM** at this simulation's pay
levels. Worse, a manager exit destabilises the store: in both the real world
and this simulator's ground truth, hourly attrition jumps for the three
months after a store changes managers, so one $35k exit quietly triggers a
tail of $5k exits.

The salaried population also breaks the tools that work for hourly. There are
about 25x fewer manager-months to learn from, and the leavers are different:
stagnation (three-plus years without a move), being paid under market while
performing well (poaching risk), and district-level operational strain, not
schedule chaos.

## What the model does

`TurnoverModel("salaried")` is the same calibrated snapshot architecture with
two changes: shallower trees with lighter minimum-leaf constraints (variance
control for a small population) and training snapshots at monthly rather than
quarterly stride, because the salaried population is too small to thin.

```python
salaried = TurnoverModel("salaried", horizons=(6, 12)).fit(train_stride1, val)
```

## Results (out-of-time test window)

| Horizon | Base rate | ROC-AUC | Oracle ceiling | Signal captured | ECE | Lift @ top decile |
|---------|-----------|---------|----------------|-----------------|------|--------------------|
| 6 mo | 11% | 0.632 | 0.691 | 69% | 3.8% | 1.9x |
| 12 mo | 21% | 0.655 | 0.664 | 94% | 6.5% | 2.4x |

Two honest observations, both of which generalise to real deployments:

The ceiling itself is low. Even the true hazard only reaches AUC 0.66 to 0.69
here, because manager exits are individually rare events driven by small
persistent pressures plus a lot of chance. Anyone promising 0.8+ AUC on
manager attrition is describing a leaky backtest.

The model captures less of the available signal than the hourly one (69% at
6 months vs 97%+). That gap is the small-data penalty, and it is the honest
argument for pooling ASM and SM into one model with role as a feature, which
is what this implementation does.

## What it saves

The simulated company loses about 66 managers a year (30 SMs, 36 ASMs), a
**$2.0M annual burn**. A top-decile watchlist at 2.4x lift over the 12-month
horizon means a district manager reviews roughly 50 names to reach half the
year's regrettable exits before they resign. Two saved store managers pay for
a well-run stay-interview program for the entire leadership population; the
knock-on effect on hourly stability (use case 4 quantifies the
manager-change penalty) is on top of that.

## Run it

```bash
python examples/run_pipeline.py
pytest tests/test_models.py::test_salaried_population_is_disjoint -q
```
