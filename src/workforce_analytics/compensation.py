"""Use case 10: compensation analytics — what pay changes actually buy.

Pay is the lever leadership always asks about first and trusts least,
because every observational estimate of "pay drives retention" is entangled
with everything else a well-run store does. This module attacks the question
three ways, in increasing order of rigour:

1. **Observational elasticity** (:func:`pay_elasticity`) — the regression a
   people-analytics team can run on real data: exit hazard vs pay position,
   naive and confounder-adjusted, here checkable against the true
   coefficient the simulator used.

2. **Model what-ifs** — the use-case-4 intervention simulator rescoring the
   workforce under a pay policy (:func:`pay_shift` is the transform).

3. **True experiments** (:func:`run_wage_experiment`) — the one only a
   simulator allows: rerun the *world* with and without a
   :class:`~workforce_analytics.config.WageProgram` on the same seed, so
   every difference is causal, and repeat across seeds because a single
   paired run of a stochastic system proves nothing. The experiment reports
   exits avoided, the extra wage bill, and the net ROI with a bootstrap
   interval.

The same machinery prices pay *cuts* in disguise: a merit freeze
(``WageProgram(kind="freeze")``) lets market drift erode real pay position
and shows up as extra attrition with a dollar sign on it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import HOURLY_ROLES, SimulationConfig, WageProgram
from .cost_model import CostModel
from .generator import generate

HOURS_PER_MONTH = 4.33  # weeks/month multiplier on scheduled weekly hours


def pay_shift(pct: float):
    """Intervention-simulator transform: shift everyone's pay ratio by pct."""
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df["pay_ratio"] = df["pay_ratio"] * (1 + pct)
        return df
    return transform


# ---------------------------------------------------------------------------
# Observational elasticity
# ---------------------------------------------------------------------------

def pay_elasticity(person_months: pd.DataFrame,
                   roles: tuple[str, ...] = HOURLY_ROLES) -> pd.DataFrame:
    """Log-odds effect of pay position on monthly exit, naive vs adjusted.

    Returns one row per specification with the pay_ratio coefficient and an
    approximate hazard change per +10% of pay. On this data the true
    coefficient is ``GroundTruth.pay_ratio_hourly`` (-2.5): the naive row
    shows what skipping confounder control costs, the adjusted row shows how
    close a standard regression gets.
    """
    from sklearn.linear_model import LogisticRegression

    pm = person_months[person_months["role"].isin(roles)].copy()
    y = pm["terminated"].to_numpy()

    def fit(X: pd.DataFrame) -> float:
        model = LogisticRegression(penalty=None, max_iter=2000)
        model.fit(X, y)
        return float(model.coef_[0][list(X.columns).index("pay_ratio")])

    naive = fit(pm[["pay_ratio"]])

    controls = pm[["pay_ratio", "schedule_volatility", "hours_gap", "commute_km",
                   "performance_rating", "store_staffing_ratio"]].copy()
    controls["tenure_0_2"] = (pm["tenure_months"] < 3).astype(float)
    controls["tenure_3_11"] = pm["tenure_months"].between(3, 11).astype(float)
    controls["is_student"] = pm["is_student"].astype(float)
    controls["second_job"] = pm["second_job"].astype(float)
    controls["is_supervisor"] = (pm["role"] == "shift_supervisor").astype(float)
    month_of_year = pm["month"] % 12 + 1
    for m in (1, 8, 9):
        controls[f"month_{m}"] = (month_of_year == m).astype(float)
    adjusted = fit(controls)

    rows = []
    for name, beta in [("naive (pay only)", naive),
                       ("adjusted (standard controls)", adjusted)]:
        rows.append({
            "specification": name,
            "pay_ratio_log_odds": round(beta, 2),
            "hazard_change_per_+10pct_pay": f"{(np.exp(beta * 0.10) - 1) * 100:+.1f}%",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulator-level experiments
# ---------------------------------------------------------------------------

def _post_period_metrics(person_months: pd.DataFrame, start_month: int,
                         roles: tuple[str, ...]) -> dict:
    post = person_months[(person_months["month"] >= start_month)
                         & person_months["role"].isin(roles)]
    monthly_wage_bill = (post["pay_rate"] * post["scheduled_hours"]
                         * HOURS_PER_MONTH).sum()
    return {
        "person_months": len(post),
        "exits": int(post["terminated"].sum()),
        "monthly_exit_rate": float(post["terminated"].mean()),
        "wage_bill": float(monthly_wage_bill),
        "mean_pay_ratio": float(post["pay_ratio"].mean()),
    }


def run_wage_experiment(
    program: WageProgram,
    base_config: SimulationConfig | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    cost_model: CostModel | None = None,
    eval_start_month: int | None = None,
    baseline_cache: dict | None = None,
) -> dict:
    """Paired counterfactual: the same simulated world with/without a program.

    For each seed the baseline and program runs share every random draw, so
    per-seed differences are causal. Effects are averaged across seeds with a
    percentile bootstrap over the per-seed deltas. Costs: extra wage bill
    over the evaluation window vs replacement costs avoided.
    """
    base_config = base_config or SimulationConfig()
    cm = cost_model or CostModel()
    start = eval_start_month if eval_start_month is not None else \
        (program.start_month or 0)
    roles = tuple(program.roles)

    deltas = []
    for seed in seeds:
        if baseline_cache is not None and seed in baseline_cache:
            base = baseline_cache[seed]
        else:
            base = generate(replace(base_config, seed=seed, wage_programs=[]))
            if baseline_cache is not None:
                baseline_cache[seed] = base
        prog = generate(replace(base_config, seed=seed, wage_programs=[program]))
        mb = _post_period_metrics(base.person_months, start, roles)
        mp = _post_period_metrics(prog.person_months, start, roles)
        avg_cost = float(np.mean([cm.replacement_cost[r] for r in roles]))
        exits_avoided = mb["exits"] - mp["exits"]
        extra_wages = mp["wage_bill"] - mb["wage_bill"]
        deltas.append({
            "seed": seed,
            "exit_rate_base": mb["monthly_exit_rate"],
            "exit_rate_program": mp["monthly_exit_rate"],
            "exits_avoided": exits_avoided,
            "extra_wage_bill": extra_wages,
            "replacement_savings": exits_avoided * avg_cost,
            "net_value": exits_avoided * avg_cost - extra_wages,
            "pay_ratio_lift": mp["mean_pay_ratio"] - mb["mean_pay_ratio"],
        })
    df = pd.DataFrame(deltas)

    rng = np.random.default_rng(0)
    boot = [df["net_value"].sample(len(df), replace=True,
                                   random_state=int(rng.integers(1e9))).mean()
            for _ in range(2000)]
    rel_reduction = 1 - df["exit_rate_program"].mean() / df["exit_rate_base"].mean()
    return {
        "program": program.name,
        "n_seeds": len(seeds),
        "eval_start_month": start,
        "monthly_exit_rate_base": round(df["exit_rate_base"].mean(), 4),
        "monthly_exit_rate_program": round(df["exit_rate_program"].mean(), 4),
        "relative_exit_reduction_pct": round(rel_reduction * 100, 1),
        "exits_avoided_mean": round(df["exits_avoided"].mean(), 1),
        "extra_wage_bill_mean": round(df["extra_wage_bill"].mean(), 0),
        "replacement_savings_mean": round(df["replacement_savings"].mean(), 0),
        "net_value_mean": round(df["net_value"].mean(), 0),
        "net_value_ci90": [round(float(np.percentile(boot, 5)), 0),
                           round(float(np.percentile(boot, 95)), 0)],
        "pay_ratio_lift": round(df["pay_ratio_lift"].mean(), 4),
        "per_seed": df,
    }


def event_study(
    program: WageProgram,
    base_config: SimulationConfig | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    roles: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Monthly exit rates, baseline vs program, averaged over paired seeds.

    The figure this feeds is the honest way to present a wage change: rates
    are identical up to the program month (same seeds), then separate by
    exactly the causal effect.
    """
    base_config = base_config or SimulationConfig()
    roles = tuple(roles or program.roles)
    rows = []
    for seed in seeds:
        for label, programs in [("baseline", []), ("program", [program])]:
            res = generate(replace(base_config, seed=seed, wage_programs=programs))
            pm = res.person_months[res.person_months["role"].isin(roles)]
            monthly = pm.groupby("month")["terminated"].mean()
            for m, r in monthly.items():
                rows.append({"seed": seed, "arm": label, "month": m, "exit_rate": r})
    out = (pd.DataFrame(rows).groupby(["arm", "month"])["exit_rate"]
           .mean().reset_index())
    return out
