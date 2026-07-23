"""Use case 13: overtime and labor-cost anomaly detection.

Payroll leakage in a large hourly workforce is a loss-prevention problem, not
a modelling one: time theft (padding punches past the schedule), ghost shifts
(labor charged with no one on the floor), and buddy punching (an absent
employee clocked in by a coworker) quietly inflate the wage bill. The hard
part is not spotting a single bad punch — it is separating systematic abuse
from the honest noise of a time clock, at the scale of every employee and
every store-month, without drowning managers in false positives.

This module builds a punch-clock layer on top of the schedule and detects
each pattern with a robust residual monitor. Because the anomalies are
*planted* with a per-record log (the oracle trick used throughout this repo),
precision and recall are measured against ground truth, not asserted, and the
detectors can be tuned against a known false-positive rate before they ever
touch real payroll.

Three anomalies, three grains, three detectors:

* **Time padding / OT inflation** — a persistent per-employee drift of punched
  above scheduled hours. Detected across employees: an employee whose *typical*
  overage is far from the population median (robust z on the mean residual).
* **Ghost shifts** — a store-month whose aggregate punched hours exceed what
  its schedules justify. Detected across store-months.
* **Buddy punching** — a one-off spike in a single employee-month, detected
  *within* each employee against their own baseline, which is exactly what
  separates a buddy-punched week from a habitual padder (whose months all look
  alike).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

_HOURLY_ROLES = ("barista", "shift_supervisor")
WEEKS_PER_MONTH = 4.33  # scheduled_hours is a weekly figure; person-months are monthly


@dataclass
class AnomalyGroundTruth:
    """Published ground truth for planted payroll anomalies.

    Shares and magnitudes are set to loss-prevention-plausible levels: a few
    percent of employees pad time, a few percent of store-months carry ghost
    labor, and buddy punching is a rare per-month event. ``punch_noise_sd`` is
    the honest clock-in/out variation every worker has, symmetric and mean
    zero, which is what the detectors must see through.
    """

    padder_share: float = 0.06
    padding_pct: tuple[float, float] = (0.08, 0.25)
    ghost_month_share: float = 0.03
    ghost_pct: tuple[float, float] = (0.06, 0.18)
    buddy_event_share: float = 0.015
    buddy_pct: tuple[float, float] = (0.20, 0.60)
    punch_noise_sd: float = 0.04


@dataclass
class PunchPanel:
    """Punched vs scheduled hours per hourly employee-month, with hidden labels.

    ``punches`` carries the observable columns plus the ground-truth flags
    (``is_padder``, ``is_ghost_month``, ``is_buddy``) a real payroll system
    would not have; ``injections`` is the planted log keyed by anomaly kind.
    """

    punches: pd.DataFrame
    injections: pd.DataFrame

    def planted(self, kind: str) -> pd.DataFrame:
        return self.injections[self.injections["kind"] == kind]


def robust_z(x: pd.Series) -> pd.Series:
    """Median/MAD z-score: outlier-resistant, so a few big anomalies don't
    inflate the scale and hide the rest."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad < 1e-9:
        return pd.Series(0.0, index=x.index)
    return (x - med) / (1.4826 * mad)


def simulate_punches(
    person_months: pd.DataFrame,
    gt: AnomalyGroundTruth | None = None,
    seed: int = 0,
) -> PunchPanel:
    """Generate a punch panel from the schedule, with planted anomalies.

    Honest punched hours are scheduled hours times symmetric mean-zero noise.
    Then a fixed share of employees pad every month, a fixed share of
    store-months carry ghost labor spread across the floor, and a rare set of
    single employee-months are buddy-punched. Padder and buddy populations are
    kept disjoint so each detector's recall is measured cleanly.
    """
    gt = gt or AnomalyGroundTruth()
    rng = np.random.default_rng(seed)
    d = person_months[person_months["role"].isin(_HOURLY_ROLES)][
        ["employee_id", "store_id", "district_id", "month", "role", "scheduled_hours"]
    ].sort_values(["employee_id", "month"]).reset_index(drop=True)

    n = len(d)
    sched = d["scheduled_hours"].to_numpy(dtype=float)
    punched = sched * (1 + rng.normal(0, gt.punch_noise_sd, n))

    # 1. Persistent per-employee time padding.
    emps = d["employee_id"].unique()
    padders = rng.choice(emps, int(gt.padder_share * len(emps)), replace=False)
    pad_by_emp = {e: rng.uniform(*gt.padding_pct) for e in padders}
    pad_factor = d["employee_id"].map(lambda e: pad_by_emp.get(e, 0.0)).to_numpy()
    punched *= 1 + pad_factor

    # 2. Ghost shifts: whole store-months with inflated aggregate labor.
    sm_id = d.groupby(["store_id", "month"]).ngroup()
    d["_store_month"] = sm_id
    sms = sm_id.unique()
    ghost_sms = set(rng.choice(sms, int(gt.ghost_month_share * len(sms)), replace=False))
    ghost_by_sm = {s: rng.uniform(*gt.ghost_pct) for s in ghost_sms}
    ghost_factor = d["_store_month"].map(lambda s: ghost_by_sm.get(s, 0.0)).to_numpy()
    punched *= 1 + ghost_factor

    # 3. Buddy punching: one-off spikes on eligible (non-padder) employee-months.
    eligible = np.where(~d["employee_id"].isin(padders).to_numpy())[0]
    buddy_idx = rng.choice(eligible, int(gt.buddy_event_share * n), replace=False)
    is_buddy = np.zeros(n, dtype=bool)
    is_buddy[buddy_idx] = True
    punched[buddy_idx] *= 1 + rng.uniform(*gt.buddy_pct, size=buddy_idx.size)

    d["punched_hours"] = np.round(punched, 1)
    d["residual_pct"] = (d["punched_hours"] - d["scheduled_hours"]) / d["scheduled_hours"]
    d["is_padder"] = d["employee_id"].isin(padders)
    d["is_ghost_month"] = d["_store_month"].isin(ghost_sms)
    d["is_buddy"] = is_buddy

    log = (
        [("time_padding", e, -1) for e in padders]
        + [("ghost_shift", -1, int(s)) for s in ghost_sms]
        + [("buddy_punch", int(d.loc[i, "employee_id"]), int(d.loc[i, "month"]))
           for i in buddy_idx]
    )
    injections = pd.DataFrame(log, columns=["kind", "employee_id", "key"])
    return PunchPanel(d, injections)


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------
def detect_time_padding(panel: PunchPanel, min_months: int = 6,
                        z_threshold: float = 3.5) -> pd.DataFrame:
    """Flag employees whose *typical* punch overage is a population outlier.

    Uses the median monthly residual, not the mean, so one buddy-punched month
    does not make an otherwise-honest employee look like a habitual padder —
    the median only moves when most of an employee's months run high.
    """
    d = panel.punches
    emp = d.groupby("employee_id").agg(
        median_residual_pct=("residual_pct", "median"),
        months=("residual_pct", "size"),
        is_padder=("is_padder", "max"),
    )
    emp = emp[emp["months"] >= min_months].copy()
    emp["robust_z"] = robust_z(emp["median_residual_pct"])
    emp["flagged"] = emp["robust_z"] > z_threshold
    return emp.sort_values("robust_z", ascending=False).reset_index()


def detect_ghost_shifts(panel: PunchPanel, min_employees: int = 4,
                       z_threshold: float = 3.5) -> pd.DataFrame:
    """Flag store-months whose aggregate hours exceed scheduled by an outlier margin."""
    d = panel.punches
    sm = d.groupby(["store_id", "month"]).agg(
        punched=("punched_hours", "sum"),
        scheduled=("scheduled_hours", "sum"),
        employees=("employee_id", "size"),
        is_ghost_month=("is_ghost_month", "max"),
    )
    sm = sm[sm["employees"] >= min_employees].copy()
    sm["excess_pct"] = sm["punched"] / sm["scheduled"] - 1
    sm["robust_z"] = robust_z(sm["excess_pct"])
    sm["flagged"] = sm["robust_z"] > z_threshold
    return sm.sort_values("robust_z", ascending=False).reset_index()


def detect_buddy_punching(panel: PunchPanel, min_months: int = 6,
                         z_threshold: float = 5.5) -> pd.DataFrame:
    """Flag single employee-months that spike against the employee's own baseline.

    The within-employee z is what separates buddy punching from habitual
    padding: a padder's months are uniformly high, so none stands out against
    the others, while a buddy-punched month is an isolated jump.
    """
    d = panel.punches.copy()
    counts = d.groupby("employee_id")["month"].transform("size")
    d = d[counts >= min_months].copy()
    d["within_emp_z"] = d.groupby("employee_id")["residual_pct"].transform(robust_z)
    d["flagged"] = d["within_emp_z"] > z_threshold
    cols = ["employee_id", "store_id", "month", "scheduled_hours", "punched_hours",
            "within_emp_z", "flagged", "is_buddy"]
    return d[cols].sort_values("within_emp_z", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Evaluation and dollars
# ----------------------------------------------------------------------
def evaluate_detection(flagged: pd.DataFrame, truth_col: str,
                       score_col: str = "robust_z") -> dict:
    """Precision/recall at the detector's threshold plus threshold-free AP."""
    truth = flagged[truth_col].astype(bool)
    flag = flagged["flagged"]
    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())
    return {
        "planted": int(truth.sum()),
        "flagged": int(flag.sum()),
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
        "average_precision": round(float(average_precision_score(truth, flagged[score_col])), 3),
    }


def labor_leakage(panel: PunchPanel, loaded_wage: float = 21.0) -> dict:
    """Dollar cost of the excess punched hours, annualised.

    Excess weekly hours per employee-month become monthly hours at
    ``WEEKS_PER_MONTH`` and are priced at the loaded wage. The excess is *net*
    (punched minus scheduled, not floored), so the symmetric mean-zero punch
    noise of honest employees cancels in aggregate and what remains is the
    planted anomaly inflation. The per-kind split uses the ground-truth flags;
    a padder working a ghost month lands in both buckets, so the kinds are
    attributions and need not sum exactly to the total.
    """
    d = panel.punches
    excess_weekly = d["punched_hours"] - d["scheduled_hours"]
    dollars = excess_weekly * WEEKS_PER_MONTH * loaded_wage
    n_years = (d["month"].max() + 1) / 12

    def annual(mask) -> float:
        return round(float(dollars[mask].sum() / n_years), 0)

    return {
        "annual_leakage_total": annual(slice(None)),
        "annual_leakage_time_padding": annual(d["is_padder"].to_numpy()
                                              & ~d["is_buddy"].to_numpy()),
        "annual_leakage_ghost_shifts": annual(d["is_ghost_month"].to_numpy()),
        "annual_leakage_buddy_punching": annual(d["is_buddy"].to_numpy()),
        "loaded_wage": loaded_wage,
    }
