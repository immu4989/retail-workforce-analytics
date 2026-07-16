"""Use case 7: hiring funnel analytics and requisition timing.

The headcount plan (use case 3) says *how many* to hire; this module says
*when to open the reqs* and *where the funnel leaks*. The reference points
are public: Chipotle's conversational-AI funnel work cut candidate
time-to-start from 12 days to 4 and nearly doubled application completion,
which is what made its hiring keep pace with 150%+ hourly turnover.

Pieces:

* :func:`simulate_funnel` — requisition-level simulation of the frontline
  hiring pipeline: applications -> screen -> interview (with the notorious
  interview no-show) -> offer -> accept -> day-one start (with ghosting).
  Stage probabilities and delays are the published ground truth; tight
  labour markets get fewer applicants and slower fills.
* :func:`funnel_report` — stage conversion and time-to-fill percentiles per
  district, the diagnostic view that locates the leak (screening? interview
  no-shows? offer declines?).
* :func:`req_timing` — joins time-to-fill onto the use-case-3 hiring plan to
  answer "post it when": need date minus p90 fill time, per district & role.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class FunnelGroundTruth:
    applicants_per_req_base: float = 6.0     # scaled by local unemployment
    p_screen_pass: float = 0.55
    p_interview_show: float = 0.62           # interview no-show is the big leak
    p_offer: float = 0.55
    p_accept: float = 0.80
    p_start: float = 0.85                    # 15% ghost after accepting
    # Base days from posting to start, by role, before market friction.
    base_days: dict = None
    reattempt_days: float = 14.0             # failed cycle -> repost delay

    def __post_init__(self):
        if self.base_days is None:
            self.base_days = {"barista": 16.0, "shift_supervisor": 24.0,
                              "assistant_store_manager": 40.0, "store_manager": 55.0}

    def as_dict(self) -> dict:
        return asdict(self)


def simulate_funnel(
    districts: pd.DataFrame,
    reqs_per_district_month: pd.DataFrame | None = None,
    n_months: int = 24,
    gt: FunnelGroundTruth | None = None,
    seed: int = 31,
) -> pd.DataFrame:
    """One row per requisition with stage counts and days-to-fill.

    ``reqs_per_district_month`` optionally fixes req volume (columns
    district_id, role, reqs_per_month); by default barista-heavy volumes are
    derived from district size proxies.
    """
    gt = gt or FunnelGroundTruth()
    rng = np.random.default_rng(seed)
    rows = []
    for _, d in districts.iterrows():
        # Tight market (low unemployment) -> fewer applicants per req.
        app_rate = gt.applicants_per_req_base * (d["unemployment_rate"] / 5.0)
        for month in range(n_months):
            if reqs_per_district_month is not None:
                vol = reqs_per_district_month[
                    reqs_per_district_month["district_id"] == d["district_id"]]
                role_volumes = list(zip(vol["role"], vol["reqs_per_month"]))
            else:
                role_volumes = [("barista", 8), ("shift_supervisor", 2),
                                ("assistant_store_manager", 0.5), ("store_manager", 0.4)]
            for role, lam in role_volumes:
                for _ in range(rng.poisson(lam)):
                    days, attempts, filled = 0.0, 0, False
                    apps = screens = shows = offers = accepts = starts = 0
                    while attempts < 4 and not filled:
                        attempts += 1
                        a = rng.poisson(max(app_rate, 1.0))
                        s = rng.binomial(a, gt.p_screen_pass)
                        sh = rng.binomial(s, gt.p_interview_show)
                        o = rng.binomial(sh, gt.p_offer)
                        # Accept rate softens slightly in tight labour markets.
                        p_acc = np.clip(gt.p_accept * (0.88 + 0.03 * d["unemployment_rate"]), 0, 0.95)
                        ac = rng.binomial(o, p_acc)
                        st = rng.binomial(ac, gt.p_start)
                        apps += a; screens += s; shows += sh
                        offers += o; accepts += ac; starts += st
                        cycle = rng.gamma(2.0, gt.base_days[role] / 2.0) \
                            * (1.5 - 0.08 * d["unemployment_rate"])
                        days += max(cycle, 3.0)
                        if st >= 1:
                            filled = True
                        else:
                            days += gt.reattempt_days
                    rows.append({
                        "district_id": d["district_id"], "role": role,
                        "month_opened": month, "applications": apps,
                        "screen_passes": screens, "interviews_attended": shows,
                        "offers": offers, "accepts": accepts, "starts": starts,
                        "filled": filled, "days_to_fill": round(days, 1) if filled else np.nan,
                        "attempts": attempts,
                    })
    return pd.DataFrame(rows)


def funnel_report(reqs: pd.DataFrame, by: str = "district_id") -> pd.DataFrame:
    """Stage conversion rates and fill-time percentiles per group."""
    g = reqs.groupby(by, observed=True)
    out = pd.DataFrame({
        "reqs": g.size(),
        "fill_rate": g["filled"].mean().round(3),
        "apps_per_req": (g["applications"].sum() / g.size()).round(1),
        "screen_pass_rate": (g["screen_passes"].sum() / g["applications"].sum()).round(3),
        "interview_show_rate": (g["interviews_attended"].sum()
                                / g["screen_passes"].sum()).round(3),
        "offer_rate": (g["offers"].sum() / g["interviews_attended"].sum()).round(3),
        "accept_rate": (g["accepts"].sum() / g["offers"].sum()).round(3),
        "start_rate": (g["starts"].sum() / g["accepts"].sum()).round(3),
        "ttf_p50_days": g["days_to_fill"].median().round(1),
        "ttf_p90_days": g["days_to_fill"].quantile(0.9).round(1),
    }).reset_index()
    return out


def req_timing(hiring_plan: pd.DataFrame, reqs: pd.DataFrame,
               horizon_days: int | None = None) -> pd.DataFrame:
    """When to post each district-role requisition batch.

    Joins p50/p90 fill times onto the use-case-3 hiring plan. ``lead_days``
    is the p90 (plan for the slow case; being early costs less than an
    unstaffed rush). Rows whose lead time exceeds the planning horizon are
    flagged ``post_now``.
    """
    ttf = (reqs.groupby(["district_id", "role"], observed=True)["days_to_fill"]
           .agg(ttf_p50_days="median", ttf_p90_days=lambda s: s.quantile(0.9))
           .round(1).reset_index())
    out = hiring_plan.merge(ttf, on=["district_id", "role"], how="left")
    horizon_days = horizon_days or int(out["horizon_months"].iloc[0] * 30)
    out["lead_days"] = out["ttf_p90_days"]
    # Spread hiring across the horizon; the last cohort must be posted at
    # least lead_days before the horizon closes.
    out["latest_post_day"] = (horizon_days - out["lead_days"]).clip(lower=0).round(0)
    out["post_now"] = out["lead_days"] >= horizon_days * 0.5
    return out
