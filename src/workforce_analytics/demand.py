"""Use case 5: demand-driven labor forecasting and shift scheduling.

This is the highest-leverage workforce system in quick-service retail: predict
customer transactions per store per hour, convert them into required staff via
labor standards, and build next week's schedule against that curve subject to
labor-law and fair-workweek constraints. Chains report high-single-digit
labor-cost reductions from getting this right, because the alternative is
staffing every day like an average day.

Three pieces, mirroring the production architecture:

* :class:`TrafficSimulator` — hourly transactions per store with day-of-week
  structure, intraday peaks, annual seasonality, growth trend, planned promos
  and weather-like shocks. Ground truth is known, as everywhere in this repo.
* :class:`LaborDemandForecaster` — pooled gradient-boosted regressor on
  calendar and lag features, evaluated out-of-time against a seasonal-naive
  baseline (same hour last week). If a model cannot beat seasonal-naive it
  has no business scheduling anyone.
* :func:`build_week_schedule` — a greedy reference scheduler that covers the
  required-staff curve with 4-8 hour shifts under hard constraints: no
  clopening (a closer never opens the next morning), max 5 working days, and
  hours capped near each employee's desired hours. Production systems use
  MIP/CP solvers; this implementation is deliberately readable and shows
  where every constraint lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

OPEN_HOUR, CLOSE_HOUR = 7, 22          # store operating window
HOURS = list(range(OPEN_HOUR, CLOSE_HOUR))


@dataclass
class TrafficConfig:
    n_weeks: int = 104
    seed: int = 11
    base_txn_per_hour: dict[int, float] = field(default_factory=lambda: {
        1: 28.0, 2: 42.0, 3: 60.0})        # by store tier
    # Weekday shape: morning rush, lunch bump, afternoon lull, small dinner.
    trend_annual: float = 0.03
    promo_lift: float = 0.25
    promo_weeks_per_year: int = 6
    noise_sd: float = 0.10                  # multiplicative lognormal noise
    weather_shock_sd: float = 0.08          # day-level shared shock


DOW_MULT = np.array([1.00, 0.96, 0.97, 1.02, 1.12, 1.25, 1.10])   # Mon..Sun

_WEEKDAY_CURVE = np.array([0.9, 1.6, 1.5, 0.9, 0.7, 0.9, 1.3, 1.2, 0.8,
                           0.6, 0.55, 0.6, 0.7, 0.6, 0.45])
_WEEKEND_CURVE = np.array([0.5, 0.9, 1.2, 1.4, 1.4, 1.3, 1.3, 1.2, 1.0,
                           0.8, 0.7, 0.7, 0.7, 0.6, 0.5])


class TrafficSimulator:
    """Hourly store transactions with known seasonal/promo structure."""

    def __init__(self, stores: pd.DataFrame, config: TrafficConfig | None = None):
        self.stores = stores
        self.cfg = config or TrafficConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def run(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        frames = []
        n_hours = len(HOURS)
        promo_weeks = {
            s: set(rng.choice(cfg.n_weeks,
                              size=cfg.promo_weeks_per_year * cfg.n_weeks // 52,
                              replace=False))
            for s in self.stores["store_id"]}
        for _, st in self.stores.iterrows():
            base = cfg.base_txn_per_hour[int(st["tier"])] * float(st.get("cost_index", 1.0))
            store_level = base * rng.normal(1.0, 0.06)
            for week in range(cfg.n_weeks):
                season = 1 + 0.12 * np.sin(2 * np.pi * (week % 52) / 52 - 0.5)
                trend = (1 + cfg.trend_annual) ** (week / 52)
                promo = 1 + cfg.promo_lift * (week in promo_weeks[st["store_id"]])
                for dow in range(7):
                    curve = _WEEKEND_CURVE if dow >= 5 else _WEEKDAY_CURVE
                    day_shock = rng.normal(1.0, cfg.weather_shock_sd)
                    lam = (store_level * DOW_MULT[dow] * season * trend * promo
                           * day_shock * curve
                           * rng.lognormal(0, cfg.noise_sd, n_hours))
                    txn = rng.poisson(np.maximum(lam, 0.1))
                    frames.append(pd.DataFrame({
                        "store_id": st["store_id"], "week": week, "dow": dow,
                        "hour": HOURS, "transactions": txn,
                        "promo": int(week in promo_weeks[st["store_id"]]),
                    }))
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def _add_features(traffic: pd.DataFrame) -> pd.DataFrame:
    df = traffic.sort_values(["store_id", "week", "dow", "hour"]).copy()
    woy = df["week"] % 52
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52)
    key = ["store_id", "dow", "hour"]
    grp = df.groupby(key)["transactions"]
    df["lag_1w"] = grp.shift(1)
    df["lag_2w"] = grp.shift(2)
    df["trail_4w"] = grp.transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    return df


FORECAST_FEATURES = ["dow", "hour", "woy_sin", "woy_cos", "promo",
                     "lag_1w", "lag_2w", "trail_4w"]


class LaborDemandForecaster:
    """Pooled hourly-transaction forecaster with a seasonal-naive yardstick."""

    def __init__(self, **gbm_params):
        self.params = {"max_iter": 250, "learning_rate": 0.08,
                       "max_leaf_nodes": 63, "min_samples_leaf": 40,
                       "random_state": 0, **gbm_params}
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, traffic: pd.DataFrame, train_end_week: int) -> "LaborDemandForecaster":
        df = _add_features(traffic)
        tr = df[(df["week"] <= train_end_week) & df["lag_2w"].notna()]
        self.model = HistGradientBoostingRegressor(**self.params)
        self.model.fit(tr[FORECAST_FEATURES], tr["transactions"])
        return self

    def predict(self, traffic: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
        df = _add_features(traffic)
        out = df[df["week"].isin(weeks)].copy()
        out["forecast"] = np.maximum(self.model.predict(out[FORECAST_FEATURES]), 0)
        out["seasonal_naive"] = out["lag_1w"]
        return out

    @staticmethod
    def evaluate(preds: pd.DataFrame) -> pd.DataFrame:
        """WAPE and bias for the model vs the seasonal-naive baseline."""
        rows = []
        for name, col in [("model", "forecast"), ("seasonal_naive", "seasonal_naive")]:
            ok = preds[preds[col].notna()]
            err = (ok[col] - ok["transactions"])
            rows.append({
                "forecaster": name,
                "wape": float(err.abs().sum() / ok["transactions"].sum()),
                "bias_pct": float(err.sum() / ok["transactions"].sum() * 100),
                "n_store_hours": int(len(ok)),
            })
        return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------
# Labor standards and scheduling
# ---------------------------------------------------------------------------

def required_staff(txn_per_hour: np.ndarray, service_rate: float = 18.0,
                   min_staff: int = 2) -> np.ndarray:
    """Transactions/hour -> heads on floor, floored at a safe minimum."""
    return np.maximum(np.ceil(txn_per_hour / service_rate), min_staff).astype(int)


@dataclass
class ScheduleResult:
    shifts: pd.DataFrame          # employee_id, dow, start, end, hours
    coverage: pd.DataFrame        # dow, hour, required, scheduled
    summary: dict

    def violations(self) -> int:
        return self.summary["clopening_violations"] + self.summary["overwork_violations"]


def build_week_schedule(
    forecast: pd.DataFrame,
    roster: pd.DataFrame,
    service_rate: float = 18.0,
    min_rest_hours: int = 10,
    max_days: int = 5,
    min_shift: int = 4,
    max_shift: int = 8,
    previous_shifts: pd.DataFrame | None = None,
) -> ScheduleResult:
    """Cover one store-week's required-staff curve with legal shifts.

    Parameters
    ----------
    forecast : rows for ONE store and ONE week with columns dow, hour, forecast.
    roster : one row per employee with ``employee_id`` and ``desired_hours``.

    Greedy strategy: for each day, walk the required-staff curve and open a
    shift at every uncovered hour, extending it up to ``max_shift`` hours
    while it keeps covering demand. Each shift goes to the employee who
    worked the *same shift last week* when ``previous_shifts`` is given (the
    fair-workweek stickiness that makes schedules predictable for workers),
    otherwise to the legal employee with the most remaining desired hours.
    """
    prev_key: set = set()
    if previous_shifts is not None and len(previous_shifts):
        prev_key = {(r["employee_id"], r["dow"], r["start"], r["end"])
                    for _, r in previous_shifts.iterrows()}
    need = (forecast.pivot_table(index="dow", columns="hour", values="forecast")
            .reindex(columns=HOURS).fillna(0.0))
    req = {d: required_staff(need.loc[d].to_numpy(), service_rate) for d in need.index}

    state = roster[["employee_id", "desired_hours"]].copy().set_index("employee_id")
    state["hours_left"] = state["desired_hours"].astype(float)
    state["days_worked"] = 0
    state["last_end"] = {}
    last_end: dict = {e: {} for e in state.index}

    shifts, coverage_rows = [], []
    for dow in sorted(req):
        demand = req[dow].copy()
        scheduled = np.zeros(len(HOURS), dtype=int)
        # Open shifts until every hour's requirement is met (or roster runs dry).
        progress = True
        while (scheduled < demand).any() and progress:
            progress = False
            start_idx = int(np.argmax(scheduled < demand))
            # Extend while there is any uncovered demand in reach.
            end_idx = start_idx
            while (end_idx - start_idx) < max_shift and end_idx < len(HOURS) \
                    and (demand[start_idx:end_idx + 1] > scheduled[start_idx:end_idx + 1]).any():
                end_idx += 1
            end_idx = max(end_idx, min(start_idx + min_shift, len(HOURS)))
            start_h, end_h = HOURS[start_idx], HOURS[0] + end_idx

            shift_len = end_h - start_h
            candidates = state[(state["hours_left"] >= shift_len)
                               & (state["days_worked"] < max_days)]
            legal = []
            for emp in candidates.index:
                prev_end = last_end[emp].get(dow - 1)
                if prev_end is not None and (start_h + 24 - prev_end) < min_rest_hours:
                    continue    # clopening guard
                if dow in last_end[emp]:
                    continue    # one shift per day
                legal.append(emp)
            if not legal:
                break
            sticky = [e for e in legal if (e, dow, start_h, end_h) in prev_key]
            emp = sticky[0] if sticky else state.loc[legal, "hours_left"].idxmax()
            state.loc[emp, "hours_left"] -= shift_len
            state.loc[emp, "days_worked"] += 1
            last_end[emp][dow] = end_h
            scheduled[start_idx:end_idx] += 1
            shifts.append({"employee_id": emp, "dow": dow, "start": start_h,
                           "end": end_h, "hours": shift_len})
            progress = True

        for i, h in enumerate(HOURS):
            coverage_rows.append({"dow": dow, "hour": h,
                                  "required": int(demand[i]),
                                  "scheduled": int(scheduled[i])})

    shifts_df = pd.DataFrame(shifts)
    cov = pd.DataFrame(coverage_rows)
    under = int((cov["required"] - cov["scheduled"]).clip(lower=0).sum())
    over = int((cov["scheduled"] - cov["required"]).clip(lower=0).sum())

    # Hard-constraint audit (the scheduler should never produce these).
    clop = 0
    if len(shifts_df):
        for emp, g in shifts_df.groupby("employee_id"):
            g = g.sort_values("dow")
            for (_, a), (_, b) in zip(g.iterrows(), g.iloc[1:].iterrows()):
                if b["dow"] == a["dow"] + 1 and (b["start"] + 24 - a["end"]) < min_rest_hours:
                    clop += 1
    if len(shifts_df):
        worked = shifts_df.groupby("employee_id")["hours"].sum()
        allowed = roster.set_index("employee_id")["desired_hours"].reindex(worked.index)
        overwork = int((worked > allowed + 1e-9).sum())
    else:
        overwork = 0

    summary = {
        "required_person_hours": int(cov["required"].sum()),
        "scheduled_person_hours": int(cov["scheduled"].sum()),
        "understaffed_hours": under,
        "overstaffed_hours": over,
        "coverage_pct": float(round(100 * (1 - under / max(int(cov["required"].sum()), 1)), 1)),
        "clopening_violations": clop,
        "overwork_violations": overwork,
        "employees_used": int(shifts_df["employee_id"].nunique()) if len(shifts_df) else 0,
    }
    return ScheduleResult(shifts_df, cov, summary)


def schedule_stability(this_week: pd.DataFrame, last_week: pd.DataFrame) -> float:
    """Share of this week's shifts identical to last week (fair-workweek KPI).

    Predictive-scheduling laws (Seattle, NYC, Chicago, Oregon) financially
    penalise late schedule changes; a stable week-over-week schedule is both
    a compliance and a retention lever (schedule volatility is a turnover
    driver — use case 4).
    """
    key = ["employee_id", "dow", "start", "end"]
    merged = this_week[key].merge(last_week[key], on=key, how="inner")
    return round(len(merged) / max(len(this_week), 1), 3)
