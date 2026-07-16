"""Evaluation for turnover models: discrimination, calibration and lift.

AUC alone does not tell an HR partner whether the model is useful. The three
questions that matter operationally are answered by three groups of metrics:

* *Can it rank leavers above stayers?* — ROC-AUC, PR-AUC.
* *Can we trust the probabilities?* — Brier score vs a base-rate baseline,
  expected calibration error, reliability-curve points. Calibration is what
  lets you sum probabilities into expected headcount losses.
* *Is acting on the top of the list worth it?* — lift and recall in the top
  risk decile, i.e. how many of the eventual leavers a retention programme
  covering 10% of employees would reach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> dict:
    """Lift and recall when intervening on the top k fraction by score."""
    n = len(y_true)
    n_top = max(int(np.ceil(n * k)), 1)
    order = np.argsort(-y_prob)
    top = np.asarray(y_true)[order][:n_top]
    base_rate = np.mean(y_true)
    return {
        "top_frac": k,
        "precision_at_k": float(np.mean(top)),
        "lift_at_k": float(np.mean(top) / base_rate) if base_rate > 0 else np.nan,
        "recall_at_k": float(np.sum(top) / np.sum(y_true)) if np.sum(y_true) > 0 else np.nan,
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted event rate per equal-count probability bin."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
        n=("y", "size"),
    ).reset_index(drop=True)
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    tbl = calibration_table(y_true, y_prob, n_bins)
    w = tbl["n"] / tbl["n"].sum()
    return float(np.sum(w * (tbl["mean_predicted"] - tbl["observed_rate"]).abs()))


def evaluate_horizon(y_true, y_prob, k: float = 0.10) -> dict:
    """All headline metrics for one horizon on one evaluation set."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    base = float(np.mean(y_true))
    metrics = {
        "n": int(len(y_true)),
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "brier_baseline": float(np.mean((base - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    metrics.update(lift_at_k(y_true, y_prob, k))
    return metrics


def evaluate_model(model, test: pd.DataFrame, k: float = 0.10) -> pd.DataFrame:
    """Evaluate a fitted :class:`TurnoverModel` on a snapshot table."""
    preds = model.predict(test)
    rows = test[test["role"].isin(model.roles)].reset_index(drop=True)
    results = []
    for h in model.horizons:
        mask = rows[f"label_{h}m"].notna().to_numpy()
        m = evaluate_horizon(rows.loc[mask, f"label_{h}m"],
                             preds.loc[mask, f"p_{h}m"], k=k)
        m["horizon_months"] = h
        m["population"] = model.population
        results.append(m)
    lead = ["population", "horizon_months", "n", "base_rate", "roc_auc", "pr_auc",
            "brier", "brier_baseline", "ece", "lift_at_k", "recall_at_k"]
    df = pd.DataFrame(results)
    return df[lead + [c for c in df.columns if c not in lead]]
