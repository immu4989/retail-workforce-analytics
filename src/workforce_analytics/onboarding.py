"""Use case 14: first-90-day onboarding risk and the new-hire watchlist.

The first 90 days are where an hourly workforce bleeds: the largest hazard in
this repo's ground truth is the new-hire washout, and the 3-month turnover
model (use case 1) already sees it. What that model does not give an operator
is a *dedicated* new-hire view — a day-30 watchlist, 30/60/90 milestone
retention, and the one signal every onboarding program tracks: did the hire
finish training.

This module adds that view as a self-contained process, in the same style as
the call-out simulator (use case 6): a published-ground-truth washout hazard
over each new hire's first-month signals, decoupled from the core termination
model so it can carry its own driver — training completion — without touching
the generator or the numbers downstream of it.

The design keeps one honest subtlety. Training completion is both a mild
*protective* effect and a *marker* of onboarding quality: a shared store-level
onboarding-quality term drives both finishing training and staying. So the raw
"completers wash out far less" gap overstates what mandating training would
buy — the same confounding lesson as use cases 9 and 10, now for onboarding.
Because the ground truth is published, the causal part and the marker part are
both known.

Pieces:

* :func:`simulate_onboarding` — one row per new hire with day-30 features, a
  training-completion flag, the true washout probability, and the realised
  30/60/90 outcome.
* :class:`OnboardingModel` — predicts 90-day washout from day-30 signals,
  scored out-of-sample against the oracle ceiling (use case 1's trick).
* :func:`new_hire_watchlist` — the ranked day-30 action list per store.
* :func:`milestone_retention` — the 30/60/90 survival curve and the
  training-completion gap, with its causal caveat measured, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .config import HOURLY_ROLES

FEATURES = ["schedule_volatility", "hours_gap", "commute_km", "is_student",
            "store_staffing_ratio", "age_band", "training_completed_30d"]
_CATEGORICAL = ["age_band"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class OnboardingGroundTruth:
    """True effects for the first-90-day washout hazard and training completion.

    Washout is a per-month exit hazard over the first three months; the
    coefficients are log-odds. Training completion at 30 days is a marker of a
    shared store-level onboarding-quality term and only mildly protective on
    its own, so the observed completer/non-completer gap is larger than the
    causal effect.
    """

    # Monthly washout hazard (months 1, 2, 3 of tenure).
    base_logodds_month: tuple[float, float, float] = (-1.95, -2.35, -2.85)
    training_incomplete: float = 0.45        # causal bump if 30-day training unfinished
    volatility_per_hour: float = 0.16        # per hour of std-dev above 3
    hours_gap_per_hour: float = 0.11
    commute_per_km: float = 0.02             # per km beyond 8, capped
    commute_cap: float = 0.55
    student: float = 0.30
    understaffed_store: float = 0.40         # staffing ratio < 0.85 (chaotic floor)
    onboarding_quality_sd: float = 0.55      # store-level random effect (shared)
    onboarding_quality_protective: float = 0.9   # washout log-odds per unit quality

    # Training completion at 30 days (also driven by onboarding quality).
    train_base_logodds: float = 1.05         # ~64% base completion
    train_quality_effect: float = 1.1        # per unit onboarding quality
    train_volatility_effect: float = 0.12    # chaotic schedules hurt completion

    def as_dict(self) -> dict:
        return asdict(self)


def _new_hire_first_month(person_months: pd.DataFrame) -> pd.DataFrame:
    """First observed month of each employee genuinely hired in-window.

    Backfilled initial employees start with tenure > 1; true new hires enter
    the panel at tenure 0-1. One row per new hire, their day-0 signals.
    """
    hourly = person_months[person_months["role"].isin(HOURLY_ROLES)]
    first = hourly.sort_values("month").groupby("employee_id", as_index=False).first()
    return first[first["tenure_months"] <= 1].reset_index(drop=True)


def simulate_onboarding(person_months: pd.DataFrame,
                        gt: OnboardingGroundTruth | None = None,
                        seed: int = 29) -> pd.DataFrame:
    """Simulate the first-90-day trajectory for each new hire.

    Returns one row per new hire with day-30 features (including
    ``training_completed_30d``), the true per-hire washout probability, the
    realised ``washed_out_90d`` label and the ``exit_month`` (1/2/3 or 0 for
    still active at 90 days).
    """
    gt = gt or OnboardingGroundTruth()
    rng = np.random.default_rng(seed)
    hires = _new_hire_first_month(person_months).copy()
    n = len(hires)

    # Shared store-level onboarding quality (drives training + retention).
    stores = hires["store_id"].unique()
    q_by_store = {s: rng.normal(0, gt.onboarding_quality_sd) for s in stores}
    quality = hires["store_id"].map(q_by_store).to_numpy()

    vol = hires["schedule_volatility"].to_numpy(dtype=float)
    gap = hires["hours_gap"].to_numpy(dtype=float)
    commute = hires["commute_km"].to_numpy(dtype=float)
    student = hires["is_student"].to_numpy().astype(bool)
    understaffed = hires["store_staffing_ratio"].to_numpy(dtype=float) < 0.85

    # Training completion at 30 days.
    train_lo = (gt.train_base_logodds + gt.train_quality_effect * quality
                - gt.train_volatility_effect * np.maximum(vol - 3.0, 0))
    completed = rng.random(n) < _sigmoid(train_lo)

    # Per-month washout hazard, shared linear predictor plus a month baseline.
    shared = (gt.training_incomplete * (~completed)
              + gt.volatility_per_hour * np.maximum(vol - 3.0, 0)
              + gt.hours_gap_per_hour * gap
              + np.minimum(gt.commute_per_km * np.maximum(commute - 8, 0), gt.commute_cap)
              + gt.student * student
              + gt.understaffed_store * understaffed
              - gt.onboarding_quality_protective * quality)

    haz = np.zeros((n, 3))
    for m in range(3):
        haz[:, m] = _sigmoid(gt.base_logodds_month[m] + shared)
    true_prob = 1.0 - np.prod(1.0 - haz, axis=1)

    exit_month = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for m in range(3):
        leaves = alive & (rng.random(n) < haz[:, m])
        exit_month[leaves] = m + 1
        alive &= ~leaves

    hires["training_completed_30d"] = completed.astype(int)
    hires["onboarding_quality"] = np.round(quality, 3)
    hires["true_washout_prob"] = np.round(true_prob, 4)
    hires["exit_month"] = exit_month
    hires["washed_out_90d"] = (exit_month > 0).astype(int)
    keep = (["employee_id", "store_id", "district_id", "role", "age_band"]
            + ["schedule_volatility", "hours_gap", "commute_km", "is_student",
               "store_staffing_ratio", "training_completed_30d", "onboarding_quality",
               "true_washout_prob", "exit_month", "washed_out_90d"])
    return hires[keep]


class OnboardingModel:
    """Predict 90-day washout from day-30 signals (gradient-boosted classifier)."""

    def __init__(self, **gbm_params):
        self.features = list(FEATURES)
        self.params = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15,
                       "min_samples_leaf": 60, "random_state": 0, **gbm_params}
        self.model: HistGradientBoostingClassifier | None = None
        self._categories: dict[str, dict[str, float]] = {}

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in _CATEGORICAL:
            if not self._categories.get(c):
                cats = sorted(df[c].astype(str).unique())
                self._categories[c] = {v: float(i) for i, v in enumerate(cats)}
            X[c] = X[c].astype(str).map(self._categories[c]).fillna(-1.0)
        return X.astype(float)

    def fit(self, cohort: pd.DataFrame) -> "OnboardingModel":
        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(self._encode(cohort), cohort["washed_out_90d"])
        return self

    def predict(self, cohort: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._encode(cohort))[:, 1]

    def evaluate(self, cohort: pd.DataFrame) -> dict:
        """Out-of-sample AUC against the oracle ceiling (best attainable AUC)."""
        p = self.predict(cohort)
        y = cohort["washed_out_90d"].to_numpy()
        ceiling = roc_auc_score(y, cohort["true_washout_prob"].to_numpy())
        model_auc = roc_auc_score(y, p)
        n_top = max(int(len(cohort) * 0.20), 1)
        top = np.argsort(-p)[:n_top]
        return {
            "n_hires": int(len(cohort)),
            "washout_rate": round(float(y.mean()), 3),
            "model_auc": round(float(model_auc), 3),
            "ceiling_auc": round(float(ceiling), 3),
            "pct_of_ceiling": round(float(model_auc / ceiling * 100), 1),
            "top_quintile_washout_rate": round(float(y[top].mean()), 3),
        }


def new_hire_watchlist(model: OnboardingModel, cohort: pd.DataFrame,
                       top_frac: float = 0.20) -> pd.DataFrame:
    """Rank new hires by predicted washout risk with a plain-language reason."""
    df = cohort.copy()
    df["washout_risk"] = model.predict(df)

    def reason(r) -> str:
        flags = []
        if not r["training_completed_30d"]:
            flags.append("training incomplete")
        if r["schedule_volatility"] > 4.0:
            flags.append("volatile schedule")
        if r["hours_gap"] > 4.0:
            flags.append("short on hours")
        if r["commute_km"] > 20:
            flags.append("long commute")
        if r["store_staffing_ratio"] < 0.85:
            flags.append("understaffed store")
        return ", ".join(flags) or "elevated model risk"

    df["top_reasons"] = df.apply(reason, axis=1)
    n_top = max(int(len(df) * top_frac), 1)
    cols = ["employee_id", "store_id", "district_id", "washout_risk",
            "training_completed_30d", "top_reasons"]
    return (df.sort_values("washout_risk", ascending=False)
            .head(n_top)[cols].reset_index(drop=True))


def milestone_retention(cohort: pd.DataFrame) -> dict:
    """30/60/90 retention plus the training-completion gap and its causal caveat.

    The observed gap is how much less completers wash out; the causal effect is
    what the ground truth actually plants. Reporting both is the point — on real
    data you only ever see the first, and mistaking it for the second is how
    onboarding programs get oversold.
    """
    exit_m = cohort["exit_month"].to_numpy()
    surv = {f"day_{d}_retention": round(float(((exit_m == 0) | (exit_m > k)).mean()), 3)
            for k, d in [(1, 30), (2, 60), (3, 90)]}
    comp = cohort["training_completed_30d"] == 1
    observed_gap = (cohort.loc[~comp, "washed_out_90d"].mean()
                    - cohort.loc[comp, "washed_out_90d"].mean())
    return {
        **surv,
        "training_completion_rate": round(float(comp.mean()), 3),
        "washout_if_completed": round(float(cohort.loc[comp, "washed_out_90d"].mean()), 3),
        "washout_if_incomplete": round(float(cohort.loc[~comp, "washed_out_90d"].mean()), 3),
        "observed_completion_gap": round(float(observed_gap), 3),
        "note": "observed gap mixes a causal training effect with onboarding-quality "
                "confounding; treat it as an upper bound on what mandating training buys",
    }
