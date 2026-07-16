"""Multi-horizon turnover probability models.

One calibrated gradient-boosted classifier per prediction horizon, per
population. Hourly retail roles (baristas, shift supervisors) and salaried
store leadership (assistant store managers, store managers) leave for
different reasons at different rates, so they are modelled separately:

>>> hourly = TurnoverModel(population="hourly", horizons=(3, 6, 12))
>>> hourly.fit(train, val)
>>> scores = hourly.predict(test)   # columns: p_3m, p_6m, p_12m

Design choices that matter in production:

* **Out-of-time calibration.** The classifier is fit on the train window and
  calibrated (isotonic) on a later validation window, so predicted
  probabilities can be summed into expected headcount losses.
* **Native categoricals and missing values.** ``HistGradientBoostingClassifier``
  consumes pandas ``category`` columns and ``NaN`` directly; no brittle
  one-hot/imputation pipeline to maintain.
* **Risk tiers, not just scores.** ``score_report`` buckets employees into
  deciles so partners can consume the output without touching probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from .config import HOURLY_ROLES, SALARIED_ROLES
from .snapshots import ALL_FEATURES

POPULATIONS = {
    "hourly": list(HOURLY_ROLES),
    "salaried": list(SALARIED_ROLES),
}
DEFAULT_HORIZONS = {
    "hourly": (3, 6, 12),
    "salaried": (6, 12),
}


class TurnoverModel:
    """Calibrated multi-horizon turnover classifier for one population."""

    def __init__(
        self,
        population: str = "hourly",
        horizons: tuple[int, ...] | None = None,
        features: list[str] | None = None,
        **gbm_params,
    ):
        if population not in POPULATIONS:
            raise ValueError(f"population must be one of {sorted(POPULATIONS)}")
        self.population = population
        self.roles = POPULATIONS[population]
        self.horizons = tuple(horizons or DEFAULT_HORIZONS[population])
        self.features = features or list(ALL_FEATURES)
        # The salaried population is an order of magnitude smaller, so its
        # defaults trade depth for variance control.
        defaults = {
            "hourly": {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31,
                       "min_samples_leaf": 40},
            "salaried": {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 15,
                         "min_samples_leaf": 15},
        }[population]
        self.gbm_params = {
            **defaults,
            "l2_regularization": 1.0,
            "categorical_features": "from_dtype",
            "random_state": 0,
            **gbm_params,
        }
        self.models_: dict[int, CalibratedClassifierCV] = {}

    # ------------------------------------------------------------------
    def _subset(self, snapshots: pd.DataFrame, horizon: int) -> pd.DataFrame:
        rows = snapshots[snapshots["role"].isin(self.roles)]
        return rows[rows[f"label_{horizon}m"].notna()]

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "TurnoverModel":
        """Fit one classifier per horizon on ``train``; calibrate on ``val``."""
        for h in self.horizons:
            tr, va = self._subset(train, h), self._subset(val, h)
            if len(tr) == 0 or len(va) == 0:
                raise ValueError(
                    f"No uncensored rows for horizon {h}m; widen the data window.")
            clf = HistGradientBoostingClassifier(**self.gbm_params)
            clf.fit(tr[self.features], tr[f"label_{h}m"])
            calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
            calibrated.fit(va[self.features], va[f"label_{h}m"])
            self.models_[h] = calibrated
        return self

    def predict(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Score any snapshot table; returns ids + one probability per horizon."""
        rows = snapshots[snapshots["role"].isin(self.roles)]
        out = rows[["employee_id", "store_id", "district_id", "month", "role"]].copy()
        for h in self.horizons:
            out[f"p_{h}m"] = self.models_[h].predict_proba(rows[self.features])[:, 1]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    def score_report(self, snapshots: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
        """Predictions plus a 1-10 risk decile (10 = highest risk)."""
        horizon = horizon or self.horizons[0]
        preds = self.predict(snapshots)
        col = f"p_{horizon}m"
        preds["risk_decile"] = (
            preds[col].rank(pct=True).mul(10).clip(upper=10).apply(np.ceil).astype(int))
        return preds.sort_values(col, ascending=False).reset_index(drop=True)
