"""Dollar impact: turn attrition probabilities into money.

Every parameter here is explicit and adjustable, because the honest answer to
"what does turnover cost" is "it depends on your fully-loaded replacement
cost". The defaults are set from public benchmarks:

* Hourly replacement cost defaults to $5,000 per exit, in line with the
  Cornell Center for Hospitality Research estimate (≈$5,864 for hourly
  service roles, 2017 dollars) and SHRM's "6 to 9 months of pay" heuristic
  applied to retail hourly wages. It covers recruiting, onboarding, training
  hours, supervisor time, and the productivity ramp.
* Salaried replacement defaults to 50% of salary (SHRM's mid-range for
  managers): $27,500 for an assistant store manager at $55k, $35,000 for a
  store manager at $70k.

Three questions this module answers:

1. ``turnover_cost_summary`` — what attrition costs today (the baseline burn).
2. ``InterventionSimulator.run(..., cost_model=...)`` — what a policy change
   is worth (exits avoided x replacement cost).
3. ``targeting_roi`` — what a targeted retention program earns when pointed
   at the model's top risk decile instead of everyone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """Fully-loaded replacement cost per exit, by role, in dollars."""

    replacement_cost: dict[str, float] = field(default_factory=lambda: {
        "barista": 5000.0,
        "shift_supervisor": 7500.0,
        "assistant_store_manager": 27500.0,
        "store_manager": 35000.0,
    })

    def cost_of(self, roles: pd.Series) -> pd.Series:
        return roles.astype(str).map(self.replacement_cost)


def turnover_cost_summary(person_months: pd.DataFrame,
                          cost_model: CostModel | None = None) -> pd.DataFrame:
    """Annualised cost of attrition by role: the baseline burn rate.

    Uses realised exits in the data and expresses them per year of
    observation, so the output reads "this company burns $X per year on
    turnover in role R".
    """
    cm = cost_model or CostModel()
    n_years = (person_months["month"].max() + 1) / 12
    exits = person_months[person_months["terminated"] == 1]
    out = exits.groupby("role", observed=True).size().rename("total_exits").reset_index()
    out["exits_per_year"] = (out["total_exits"] / n_years).round(1)
    out["replacement_cost"] = out["role"].map(cm.replacement_cost)
    out["annual_cost"] = (out["exits_per_year"] * out["replacement_cost"]).round(0)
    avg_headcount = (person_months.groupby("month").size().mean())
    total = pd.DataFrame([{
        "role": "TOTAL", "total_exits": out["total_exits"].sum(),
        "exits_per_year": out["exits_per_year"].sum(),
        "replacement_cost": np.nan,
        "annual_cost": out["annual_cost"].sum(),
    }])
    result = pd.concat([out.sort_values("annual_cost", ascending=False), total],
                       ignore_index=True)
    result.attrs["avg_headcount"] = float(avg_headcount)
    return result


def targeting_roi(
    score_report: pd.DataFrame,
    labels: pd.Series,
    horizon: int,
    cost_model: CostModel | None = None,
    top_frac: float = 0.10,
    program_effectiveness: float = 0.20,
    program_cost_per_person: float = 300.0,
) -> dict:
    """ROI of a retention program aimed at the model's riskiest employees.

    Parameters
    ----------
    score_report : output of ``TurnoverModel.score_report`` (or ``predict``)
        for the population being targeted, one row per employee.
    labels : realised outcome for the same rows (1 = left within horizon);
        used to count how many actual leavers the program would have reached.
    program_effectiveness : assumed relative reduction in exit probability
        among treated employees. 0.20 is conservative for stay interviews +
        targeted scheduling/pay fixes; sensitivity-check it, never sell it
        as a point estimate.
    program_cost_per_person : cost of including one employee in the program.

    Returns a dict with reached leavers, exits avoided, gross savings,
    program cost, net savings, and the same program applied to everyone
    (the untargeted alternative) for comparison.
    """
    cm = cost_model or CostModel()
    col = f"p_{horizon}m"
    df = score_report.copy()
    df["label"] = np.asarray(labels, dtype=float)
    df["cost"] = cm.cost_of(df["role"])
    df = df.sort_values(col, ascending=False)

    n = len(df)
    n_top = int(np.ceil(n * top_frac))
    top = df.head(n_top)

    def program(rows: pd.DataFrame) -> dict:
        reached_leavers = rows["label"].sum()
        avoided = program_effectiveness * (rows["label"] * rows["cost"]).sum()
        spend = len(rows) * program_cost_per_person
        return {
            "treated": int(len(rows)),
            "reached_leavers": int(reached_leavers),
            "exits_avoided": round(program_effectiveness * reached_leavers, 1),
            "gross_savings": round(float(avoided), 0),
            "program_cost": float(spend),
            "net_savings": round(float(avoided - spend), 0),
        }

    targeted = program(top)
    untargeted = program(df)
    return {
        "horizon_months": horizon,
        "top_frac": top_frac,
        "program_effectiveness": program_effectiveness,
        "program_cost_per_person": program_cost_per_person,
        "targeted": targeted,
        "untargeted_everyone": untargeted,
        "roi_multiple_targeted": round(
            targeted["gross_savings"] / targeted["program_cost"], 2),
        "roi_multiple_untargeted": round(
            untargeted["gross_savings"] / untargeted["program_cost"], 2),
    }
