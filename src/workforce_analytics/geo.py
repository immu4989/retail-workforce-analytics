"""Use case 18: geographic transfer matching.

Commute distance is a planted turnover driver: every kilometre past five adds
to an hourly employee's monthly exit hazard, capped at a big penalty. So when
vacancies open, moving a far-commuting employee to a closer store with an
opening buys retention for free — no raise, no new hire, just a better match of
people to locations. Deciding *who* moves *where* is a capacitated assignment
problem, and doing it well beats doing it greedily.

This module gives stores coordinates (clustered by district, the way real
markets are), places each employee's home at their known commute distance from
their current store, and then matches transfer-seekers to nearby vacancies to
maximise the retention payoff. Because the commute-to-hazard coefficient is
published ground truth, that payoff — expected exits avoided and dollars saved
— is exact, and the optimal assignment can be scored against the greedy and
do-nothing alternatives.

Pieces:

* :func:`assign_store_coordinates` / :func:`assign_homes` — the geography layer.
* :func:`transfer_options` — every beneficial (employee, closer vacant store)
  pair, with its commute reduction and hazard reduction.
* :func:`optimize_transfers` — capacitated max-payoff matching (Hungarian).
* :func:`compare_strategies` — optimal vs greedy vs no transfers, in exits
  avoided and dollars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HOURLY_ROLES, GroundTruth

VOLUNTARY_MONTHLY_HAZARD = 0.057   # empirical hourly base, used to anchor the payoff


@dataclass
class GeoGroundTruth:
    """Spatial layout controls (the commute-hazard coefficients live in
    :class:`~workforce_analytics.config.GroundTruth`)."""

    district_spacing_km: float = 34.0    # distance between district centres
    district_radius_km: float = 6.0      # store scatter within a district
    base_monthly_hazard: float = VOLUNTARY_MONTHLY_HAZARD
    min_commute_to_consider: float = 8.0  # only help employees commuting past this
    min_reduction_km: float = 0.5         # ignore trivially small commute savings
    within_district_only: bool = True     # transfers stay in the home district


def _commute_logodds(commute_km: np.ndarray, gt: GroundTruth) -> np.ndarray:
    """Published commute contribution to monthly exit log-odds (hourly)."""
    return np.minimum(
        gt.commute_per_km_hourly * np.maximum(commute_km - gt.commute_threshold_km_hourly, 0),
        gt.commute_cap_hourly)


def assign_store_coordinates(stores: pd.DataFrame, geo: GeoGroundTruth | None = None,
                             seed: int = 71) -> pd.DataFrame:
    """Place stores on a plane, clustered by district (km units)."""
    geo = geo or GeoGroundTruth()
    rng = np.random.default_rng(seed)
    districts = stores["district_id"].unique()
    side = int(np.ceil(np.sqrt(len(districts))))
    centres = {}
    for i, d in enumerate(districts):
        cx = (i % side) * geo.district_spacing_km
        cy = (i // side) * geo.district_spacing_km
        centres[d] = (cx + rng.normal(0, 2), cy + rng.normal(0, 2))
    out = stores.copy()
    ang = rng.uniform(0, 2 * np.pi, len(out))
    rad = rng.uniform(0, geo.district_radius_km, len(out))
    cxy = np.array([centres[d] for d in out["district_id"]])
    out["store_x"] = cxy[:, 0] + rad * np.cos(ang)
    out["store_y"] = cxy[:, 1] + rad * np.sin(ang)
    return out


def workforce_snapshot(person_months: pd.DataFrame, month: int) -> pd.DataFrame:
    """Active hourly employees at ``month`` with their commute and store."""
    pm = person_months
    snap = pm[(pm["month"] == month) & (pm["terminated"] == 0)
              & pm["role"].isin(HOURLY_ROLES)]
    return snap[["employee_id", "store_id", "district_id", "role",
                 "commute_km", "tenure_months"]].reset_index(drop=True)


def assign_homes(snapshot: pd.DataFrame, store_coords: pd.DataFrame,
                 seed: int = 72) -> pd.DataFrame:
    """Place each employee's home at their commute distance from their store."""
    rng = np.random.default_rng(seed)
    xy = store_coords.set_index("store_id")[["store_x", "store_y"]]
    df = snapshot.merge(xy, on="store_id", how="left")
    ang = rng.uniform(0, 2 * np.pi, len(df))
    df["home_x"] = df["store_x"] + df["commute_km"] * np.cos(ang)
    df["home_y"] = df["store_y"] + df["commute_km"] * np.sin(ang)
    return df


def store_vacancies(snapshot: pd.DataFrame, stores: pd.DataFrame) -> pd.Series:
    """Open hourly headcount per store: target minus current, floored at zero."""
    target = (stores.set_index("store_id")["target_baristas"]
              + stores.set_index("store_id")["target_shift_supervisors"])
    current = snapshot.groupby("store_id").size()
    vac = (target - current.reindex(target.index).fillna(0)).clip(lower=0)
    return vac.astype(int)


def _exit_prob(commute_km: np.ndarray, gt: GroundTruth, geo: GeoGroundTruth,
               mean_commute_lo: float, horizon: int) -> np.ndarray:
    """P(voluntary exit within ``horizon`` months) attributable to commute.

    The base hazard is anchored to the empirical rate at the average commute, so
    only the published commute channel moves the number; other drivers are held
    at their population average.
    """
    base_logit = np.log(geo.base_monthly_hazard / (1 - geo.base_monthly_hazard))
    lo = base_logit + _commute_logodds(commute_km, gt) - mean_commute_lo
    h = 1 / (1 + np.exp(-lo))
    return 1 - (1 - h) ** horizon


def transfer_options(homes: pd.DataFrame, store_coords: pd.DataFrame,
                     vacancies: pd.Series, gt: GroundTruth | None = None,
                     geo: GeoGroundTruth | None = None, horizon: int = 12) -> pd.DataFrame:
    """Every beneficial (employee, closer vacant store) pair with its payoff.

    A pair qualifies when the employee commutes past ``min_commute_to_consider``,
    the target store has a vacancy and (optionally) sits in their district, and
    the new commute is shorter than the current one.
    """
    gt = gt or GroundTruth()
    geo = geo or GeoGroundTruth()
    mean_commute_lo = float(_commute_logodds(homes["commute_km"].to_numpy(), gt).mean())

    vac_stores = store_coords[store_coords["store_id"].isin(
        vacancies[vacancies > 0].index)].copy()
    cand = homes[homes["commute_km"] >= geo.min_commute_to_consider].copy()

    rows = []
    for _, e in cand.iterrows():
        pool = vac_stores
        if geo.within_district_only:
            pool = pool[pool["district_id"] == e["district_id"]]
        pool = pool[pool["store_id"] != e["store_id"]]
        if pool.empty:
            continue
        new_commute = np.hypot(pool["store_x"] - e["home_x"], pool["store_y"] - e["home_y"])
        closer = new_commute <= e["commute_km"] - geo.min_reduction_km
        if not closer.any():
            continue
        p_old = _exit_prob(np.array([e["commute_km"]]), gt, geo, mean_commute_lo, horizon)[0]
        for sid, nc in zip(pool.loc[closer, "store_id"], new_commute[closer]):
            p_new = _exit_prob(np.array([nc]), gt, geo, mean_commute_lo, horizon)[0]
            if p_old - p_new <= 1e-9:
                continue    # reduction sits entirely in the capped-hazard region
            rows.append({
                "employee_id": e["employee_id"], "from_store": e["store_id"],
                "to_store": sid, "old_commute_km": round(float(e["commute_km"]), 2),
                "new_commute_km": round(float(nc), 2),
                "commute_reduction_km": round(float(e["commute_km"] - nc), 2),
                "exits_avoided": float(p_old - p_new),
            })
    return pd.DataFrame(rows)


def _assign(options: pd.DataFrame, vacancies: pd.Series) -> pd.DataFrame:
    """Capacitated max-payoff matching via the Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment

    if options.empty:
        return options
    emps = options["employee_id"].unique()
    e_idx = {e: i for i, e in enumerate(emps)}
    # Expand each vacant store into one column per open slot.
    slots = []
    for sid in options["to_store"].unique():
        slots += [(sid, k) for k in range(int(vacancies.get(sid, 0)))]
    s_idx = {sl: j for j, sl in enumerate(slots)}

    payoff = np.zeros((len(emps), len(slots)))
    best = (options.sort_values("exits_avoided", ascending=False)
            .drop_duplicates(["employee_id", "to_store"]))
    for _, r in best.iterrows():
        for k in range(int(vacancies.get(r["to_store"], 0))):
            payoff[e_idx[r["employee_id"]], s_idx[(r["to_store"], k)]] = r["exits_avoided"]

    ri, ci = linear_sum_assignment(-payoff)
    picks = []
    for i, j in zip(ri, ci):
        if payoff[i, j] <= 0:
            continue
        sid = slots[j][0]
        e = emps[i]
        row = best[(best["employee_id"] == e) & (best["to_store"] == sid)].iloc[0]
        picks.append(row)
    return pd.DataFrame(picks).reset_index(drop=True) if picks else options.iloc[0:0]


def optimize_transfers(options: pd.DataFrame, vacancies: pd.Series) -> pd.DataFrame:
    """Optimal capacitated assignment maximising total exits avoided."""
    return _assign(options, vacancies)


def greedy_transfers(options: pd.DataFrame, vacancies: pd.Series) -> pd.DataFrame:
    """Greedy baseline: employees in benefit order grab their best open slot."""
    remaining = {s: int(v) for s, v in vacancies.items()}
    taken_emp = set()
    picks = []
    for _, r in options.sort_values("exits_avoided", ascending=False).iterrows():
        if r["employee_id"] in taken_emp or remaining.get(r["to_store"], 0) <= 0:
            continue
        picks.append(r)
        taken_emp.add(r["employee_id"])
        remaining[r["to_store"]] -= 1
    return pd.DataFrame(picks).reset_index(drop=True) if picks else options.iloc[0:0]


def transfer_payoff(assignment: pd.DataFrame, replacement_cost: float = 5000.0) -> dict:
    """Total exits avoided, commute saved and dollars for an assignment."""
    if assignment.empty:
        return {"transfers": 0, "exits_avoided": 0.0, "commute_km_saved": 0.0,
                "dollars_saved": 0.0}
    exits = float(assignment["exits_avoided"].sum())
    return {
        "transfers": int(len(assignment)),
        "exits_avoided": round(exits, 2),
        "commute_km_saved": round(float(assignment["commute_reduction_km"].sum()), 1),
        "dollars_saved": round(exits * replacement_cost, 0),
    }


def compare_strategies(options: pd.DataFrame, vacancies: pd.Series,
                       replacement_cost: float = 5000.0) -> dict:
    """Optimal vs greedy vs no transfers, in exits avoided and dollars."""
    opt = transfer_payoff(optimize_transfers(options, vacancies), replacement_cost)
    greedy = transfer_payoff(greedy_transfers(options, vacancies), replacement_cost)
    return {
        "n_candidates": int(options["employee_id"].nunique()) if not options.empty else 0,
        "optimal": opt,
        "greedy": greedy,
        "no_transfers": {"transfers": 0, "exits_avoided": 0.0,
                         "commute_km_saved": 0.0, "dollars_saved": 0.0},
        "optimal_vs_greedy_extra_exits_avoided": round(
            opt["exits_avoided"] - greedy["exits_avoided"], 2),
    }
