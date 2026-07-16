"""Use case 8: internal mobility — promotion readiness and bench strength.

Retail promotes from within far less than it could (industry internal
promotion rates run around 9%), yet internal promotion is the cheapest
leadership pipeline a chain has: an assistant store manager promoted from a
shift supervisor already knows the store, and use case 2 prices what an
external store-manager fill costs instead.

Two questions, two artifacts:

* **Who is ready?** :class:`PromotionModel` predicts each hourly employee's
  probability of being promoted within 6 months, trained on the promotion
  events the simulator actually generates (tenure- and performance-gated,
  vacancy-driven — the same shape real promotion processes have). Note what
  this predicts: *who the current process will promote*, which makes it an
  audit tool as much as a planning tool — if the model's top decile skews
  on anything other than performance and readiness, the process does too.

* **Is the bench deep enough?** :func:`bench_strength` compares each
  district's ready-now internal candidates against its expected leadership
  vacancies over the next year (expected exits from the salaried turnover
  model plus planned growth). A coverage ratio under 1 means external
  hires — slower and costlier — are already locked in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from .config import HOURLY_ROLES
from .snapshots import ALL_FEATURES, CATEGORICAL_FEATURES


def promotion_events(person_months: pd.DataFrame) -> pd.DataFrame:
    """One row per promotion: employee, month, from_role, to_role."""
    pm = person_months.sort_values(["employee_id", "month"])
    prev_role = pm.groupby("employee_id")["role"].shift(1)
    changed = pm[prev_role.notna() & (pm["role"] != prev_role)]
    return pd.DataFrame({
        "employee_id": changed["employee_id"].to_numpy(),
        "month": changed["month"].to_numpy(),
        "from_role": prev_role[changed.index].to_numpy(),
        "to_role": changed["role"].to_numpy(),
    })


def build_promotion_panel(snapshots: pd.DataFrame, person_months: pd.DataFrame,
                          horizon: int = 6) -> pd.DataFrame:
    """Snapshots for hourly employees + promoted-within-horizon label."""
    events = promotion_events(person_months)
    first_promo = events.groupby("employee_id")["month"].min()
    snaps = snapshots[snapshots["role"].isin(HOURLY_ROLES)].copy()
    promo_month = snaps["employee_id"].map(first_promo)
    label = ((promo_month > snaps["month"])
             & (promo_month <= snaps["month"] + horizon)).fillna(False)
    snaps[f"promoted_{horizon}m"] = label.astype(float)
    max_month = int(person_months["month"].max())
    snaps.loc[snaps["month"] + horizon > max_month, f"promoted_{horizon}m"] = np.nan
    return snaps


class PromotionModel:
    """Calibrated promotion-within-horizon classifier for hourly employees."""

    def __init__(self, horizon: int = 6, **gbm_params):
        self.horizon = horizon
        self.features = list(ALL_FEATURES)
        self.params = {"max_iter": 250, "learning_rate": 0.06,
                       "max_leaf_nodes": 31, "min_samples_leaf": 40,
                       "l2_regularization": 1.0, "random_state": 0, **gbm_params}
        self.model: CalibratedClassifierCV | None = None
        self._categories: dict[str, dict[str, float]] = {}

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in CATEGORICAL_FEATURES:
            if not self._categories.get(c):
                cats = sorted(df[c].astype(str).unique())
                self._categories[c] = {v: float(i) for i, v in enumerate(cats)}
            X[c] = X[c].astype(str).map(self._categories[c])
        return X.astype(float)

    def fit(self, train_panel: pd.DataFrame, val_panel: pd.DataFrame) -> "PromotionModel":
        col = f"promoted_{self.horizon}m"
        tr = train_panel[train_panel[col].notna()]
        va = val_panel[val_panel[col].notna()]
        clf = HistGradientBoostingClassifier(**self.params)
        clf.fit(self._encode(tr), tr[col])
        self.model = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        self.model.fit(self._encode(va), va[col])
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        rows = panel[panel["role"].isin(HOURLY_ROLES)]
        out = rows[["employee_id", "store_id", "district_id", "month", "role"]].copy()
        out[f"p_promo_{self.horizon}m"] = self.model.predict_proba(
            self._encode(rows))[:, 1]
        return out.reset_index(drop=True)

    def evaluate(self, panel: pd.DataFrame) -> dict:
        from sklearn.metrics import roc_auc_score, average_precision_score

        col = f"promoted_{self.horizon}m"
        rows = panel[panel[col].notna()].reset_index(drop=True)
        p = self.model.predict_proba(self._encode(rows))[:, 1]
        y = rows[col].to_numpy()
        n_top = max(int(len(rows) * 0.05), 1)
        top = np.argsort(-p)[:n_top]
        return {
            "n": int(len(rows)),
            "promotion_base_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 3),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "precision_top_5pct": round(float(y[top].mean()), 3),
        }


def bench_strength(
    promo_preds: pd.DataFrame,
    salaried_preds: pd.DataFrame,
    stores: pd.DataFrame,
    horizon: int = 12,
    ready_threshold: float | None = None,
) -> pd.DataFrame:
    """Ready-now internal candidates vs expected leadership vacancies.

    ``promo_preds`` — PromotionModel.predict on the current month (shift
    supervisors are the ASM/SM feeder pool). ``salaried_preds`` — salaried
    TurnoverModel.predict on the same month, whose summed ``p_{horizon}m``
    are expected leadership exits. Growth stores add one SM (+ ASM for
    larger formats) each.
    """
    col_promo = [c for c in promo_preds.columns if c.startswith("p_promo_")][0]
    feeders = promo_preds[promo_preds["role"] == "shift_supervisor"]
    if ready_threshold is None:
        # Top 30% of the feeder pool counts as ready-now; baristas are
        # excluded from the quantile — their promotion probabilities reflect
        # SS vacancies, not leadership readiness.
        ready_threshold = float(feeders[col_promo].quantile(0.70))
    ready = (feeders[feeders[col_promo] >= ready_threshold]
             .groupby("district_id", observed=True).size().rename("ready_now"))

    col_exit = f"p_{horizon}m"
    exits = (salaried_preds.groupby("district_id", observed=True)[col_exit]
             .sum().rename("expected_leadership_exits"))

    month = int(salaried_preds["month"].iloc[0])
    horizon_end = month + horizon
    growth = stores[(stores["open_month"] > month)
                    & (stores["open_month"] <= horizon_end)]
    growth_slots = (growth.assign(slots=1 + growth["has_asm"].astype(int))
                    .groupby("district_id")["slots"].sum().rename("growth_slots"))

    out = pd.concat([ready, exits, growth_slots], axis=1).fillna(0).reset_index()
    out["expected_vacancies"] = (out["expected_leadership_exits"]
                                 + out["growth_slots"]).round(1)
    out["coverage_ratio"] = (out["ready_now"]
                             / out["expected_vacancies"].replace(0, np.nan)).round(2)
    out["ready_now"] = out["ready_now"].astype(int)
    out["expected_leadership_exits"] = out["expected_leadership_exits"].round(1)
    return out.sort_values("coverage_ratio").reset_index(drop=True)
