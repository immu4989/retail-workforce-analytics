"""Deep discrete-time survival model: one network, every horizon.

Requires the optional ``torch`` dependency (``pip install torch``, or
``pip install -e ".[deep]"``).

The gradient-boosted models in :mod:`~workforce_analytics.turnover` train one
classifier per horizon. The survival view trains a single network on the
*monthly* hazard — "given this employee-month, what is the chance they leave
this month?" — and composes any horizon from it:

    P(gone within h months) = 1 - prod_{k=0..h-1} (1 - hazard(x at t+k))

Two things make this more than a re-implementation of the classifier:

* **Deterministic feature rollout.** When composing the product, tenure and
  calendar month are advanced month by month, so the prediction bakes in the
  employee crossing hazard cliffs (a student hitting August, a new hire
  maturing out of the washout) *within* the window. Static per-horizon
  classifiers can only see the snapshot.
* **A full survival curve per employee.** ``survival_curves`` returns
  S(1..h) for retention planning UIs, not just point probabilities.

Architecture: categorical embeddings + a small MLP on standardised numerics,
trained with BCE on the monthly event flag. On the simulated data it matches
the GBM at 3 months and comes within a point at longer horizons — worth
knowing before anyone reaches for a transformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .snapshots import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SurvivalNN needs the optional dependency: pip install torch "
            '(or pip install -e ".[deep]")') from e


class SurvivalNN:
    """Discrete-time hazard network over person-month snapshots."""

    def __init__(
        self,
        roles: list[str] | None = None,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 1024,
        random_state: int = 0,
    ):
        self.roles = roles
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.numeric = list(NUMERIC_FEATURES) + list(BINARY_FEATURES)
        self.categorical = list(CATEGORICAL_FEATURES)
        self._cats: dict[str, dict[str, int]] = {}
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self.net = None

    # ------------------------------------------------------------------
    def _fit_encoders(self, df: pd.DataFrame) -> None:
        for c in self.categorical:
            # Reserve index 0 for unseen categories.
            self._cats[c] = {v: i + 1 for i, v in
                             enumerate(sorted(df[c].astype(str).unique()))}
        X = df[self.numeric].to_numpy(dtype=np.float32)
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-6

    def _encode(self, df: pd.DataFrame):
        X_num = ((df[self.numeric].to_numpy(dtype=np.float32) - self._mu)
                 / self._sigma)
        X_cat = np.stack([
            df[c].astype(str).map(self._cats[c]).fillna(0).to_numpy(dtype=np.int64)
            for c in self.categorical], axis=1)
        return X_num, X_cat

    def _build_net(self):
        torch = _require_torch()
        import torch.nn as nn

        cat_sizes = [len(self._cats[c]) + 1 for c in self.categorical]
        emb_dims = [min(8, (s + 1) // 2) for s in cat_sizes]

        class HazardNet(nn.Module):
            def __init__(self, n_num, hidden, dropout):
                super().__init__()
                self.embeddings = nn.ModuleList(
                    [nn.Embedding(s, d) for s, d in zip(cat_sizes, emb_dims)])
                dims = [n_num + sum(emb_dims), *hidden]
                layers = []
                for a, b in zip(dims, dims[1:]):
                    layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
                layers.append(nn.Linear(dims[-1], 1))
                self.mlp = nn.Sequential(*layers)

            def forward(self, x_num, x_cat):
                embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
                return self.mlp(torch.cat([x_num, *embs], dim=1)).squeeze(-1)

        return HazardNet(len(self.numeric), self.hidden, self.dropout)

    # ------------------------------------------------------------------
    def fit(self, snapshots: pd.DataFrame, label_col: str = "label_1m") -> "SurvivalNN":
        """Train on monthly-event labels (build snapshots with horizon 1)."""
        torch = _require_torch()
        rows = snapshots
        if self.roles is not None:
            rows = rows[rows["role"].isin(self.roles)]
        rows = rows[rows[label_col].notna()]
        self._fit_encoders(rows)

        X_num, X_cat = self._encode(rows)
        y = rows[label_col].to_numpy(dtype=np.float32)

        torch.manual_seed(self.random_state)
        self.net = self._build_net()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # Plain BCE: monthly base rates (2-8%) are mild, and reweighting the
        # positives would destroy the hazard's probabilistic meaning, which
        # the survival product depends on.
        loss_fn = torch.nn.BCEWithLogitsLoss()

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_num), torch.from_numpy(X_cat), torch.from_numpy(y))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state))

        self.net.train()
        for _ in range(self.epochs):
            for xb_num, xb_cat, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.net(xb_num, xb_cat), yb)
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    # ------------------------------------------------------------------
    def _hazard(self, rows: pd.DataFrame) -> np.ndarray:
        torch = _require_torch()
        X_num, X_cat = self._encode(rows)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X_num), torch.from_numpy(X_cat))
            return torch.sigmoid(logits).numpy()

    def survival_curves(self, snapshots: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
        """S(t) for t = 1..horizon, one row per employee.

        Tenure, time-since fields and calendar month advance deterministically
        through the window; everything else is held at its snapshot value.
        """
        rows = snapshots
        if self.roles is not None:
            rows = rows[rows["role"].isin(self.roles)]
        rows = rows.reset_index(drop=True)

        surv = np.ones(len(rows), dtype=np.float64)
        out = {"employee_id": rows["employee_id"], "month": rows["month"]}
        future = rows.copy()
        for k in range(horizon):
            if k > 0:
                future = future.copy()
                for col in ("tenure_months", "months_since_mgr_change",
                            "months_since_promotion", "months_since_raise"):
                    future[col] = future[col] + 1
                future["month_of_year"] = (
                    future["month_of_year"].astype(int) % 12 + 1).astype(str)
            surv = surv * (1.0 - self._hazard(future))
            out[f"S_{k + 1}m"] = surv.copy()
        return pd.DataFrame(out)

    def predict(self, snapshots: pd.DataFrame,
                horizons: tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
        """Exit probabilities at the requested horizons (1 - S(h))."""
        curves = self.survival_curves(snapshots, horizon=max(horizons))
        rows = snapshots
        if self.roles is not None:
            rows = rows[rows["role"].isin(self.roles)]
        out = rows[["employee_id", "store_id", "district_id", "month", "role"]]\
            .reset_index(drop=True)
        for h in horizons:
            out[f"p_{h}m"] = 1.0 - curves[f"S_{h}m"]
        return out
