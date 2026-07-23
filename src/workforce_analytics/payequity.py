"""Use case 17: pay-equity audit, and validating the audit itself.

A pay-equity audit asks whether a group is paid less after controlling for the
legitimate drivers of pay — role, tenure, market, performance. On real data you
run the regression, read the group coefficient, and never know two things that
decide whether to act: does your audit have the *power* to detect a gap that is
really there, and what is its *false-positive rate* when there is no gap at all.
Omit a control and a confounder masquerades as bias; include a noisy one and a
real gap hides in the standard error.

This module makes both measurable. A **synthetic, abstract** group label (group
A / group B — deliberately not a real protected class) is assigned to employees,
optionally correlated with a legitimate factor to create confounding, and a
*known* residual pay gap is planted. The audit then has to recover it, and its
power and false-positive rate are estimated by replaying the audit over many
draws — the oracle trick applied to a method rather than a model.

This is the roadmap's "do it carefully or not at all" use case. The group is a
synthetic construct for validating audit methodology; nothing here infers or
uses a real person's demographics. See the ethics note in
``adapting_to_real_data.md``.

Claims it supports:

1. **Recovery** — with the right controls, the audit recovers the planted
   residual gap; its confidence interval covers the truth.
2. **Confounding** — when the group is correlated with tenure, the *unadjusted*
   gap is wrong and the *adjusted* gap is right, the same lesson as use cases 9
   and 10.
3. **Power and false-positive rate** — replaying the audit gives its detection
   power at the planted gap and its false-positive rate at zero gap (which
   should sit near the 5% test level).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HOURLY_ROLES


@dataclass
class PayEquityGroundTruth:
    """Planted group structure and residual pay gap.

    ``true_residual_gap`` is the fraction less that group B is paid with all
    legitimate factors held equal (0.04 = 4% lower). ``confounding_strength``
    tilts group-B assignment toward shorter-tenure employees, so the raw gap
    diverges from the residual gap unless tenure is controlled.
    """

    group_b_share: float = 0.45
    true_residual_gap: float = 0.04
    confounding_strength: float = 0.0    # 0 = groups independent of tenure

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


CONTROLS = ["tenure_months", "performance_rating", "log_market_pay"]


def build_employee_frame(person_months: pd.DataFrame,
                         stores: pd.DataFrame) -> pd.DataFrame:
    """One row per hourly employee from their most recent active month.

    Carries the legitimate pay controls (role, tenure, performance, local market
    via store cost index) and the observed pay rate.
    """
    pm = person_months[person_months["role"].isin(HOURLY_ROLES)]
    last = pm.sort_values("month").groupby("employee_id", as_index=False).last()
    cost = stores.set_index("store_id")["cost_index"]
    last = last[last["pay_rate"] > 0].copy()
    last["cost_index"] = last["store_id"].map(cost).fillna(1.0)
    # Market pay context: role base scaled by local cost of labor.
    role_base = last.groupby("role")["pay_rate"].transform("median")
    last["log_market_pay"] = np.log(role_base * last["cost_index"])
    return last[["employee_id", "store_id", "role", "tenure_months",
                 "performance_rating", "cost_index", "log_market_pay",
                 "pay_rate"]].reset_index(drop=True)


def assign_group_and_gap(employees: pd.DataFrame,
                         gt: PayEquityGroundTruth | None = None,
                         seed: int = 61) -> pd.DataFrame:
    """Assign the synthetic group and apply the planted residual pay gap."""
    gt = gt or PayEquityGroundTruth()
    rng = np.random.default_rng(seed)
    df = employees.copy()
    n = len(df)

    # Group-B propensity, optionally tilted toward shorter tenure (confounding).
    z_tenure = (df["tenure_months"] - df["tenure_months"].mean()) / df["tenure_months"].std()
    logit = (np.log(gt.group_b_share / (1 - gt.group_b_share))
             - gt.confounding_strength * z_tenure.to_numpy())
    p_b = 1 / (1 + np.exp(-logit))
    is_b = rng.random(n) < p_b
    df["group"] = np.where(is_b, "B", "A")

    # Planted residual gap: group B paid (1 - gap) times, all else equal.
    df["adjusted_pay"] = df["pay_rate"] * np.where(is_b, 1 - gt.true_residual_gap, 1.0)
    df["log_pay"] = np.log(df["adjusted_pay"])
    return df


def _ols_group_effect(df: pd.DataFrame, controls: list[str]) -> dict:
    """OLS of log_pay on group-B indicator + controls; return the group effect."""
    from scipy import stats

    y = df["log_pay"].to_numpy(dtype=float)
    g = (df["group"] == "B").to_numpy(dtype=float)
    parts = [np.ones(len(df)), g]
    for c in controls:
        x = df[c].to_numpy(dtype=float)
        parts.append((x - x.mean()) / x.std())
    # Role fixed effects (one-hot, drop first).
    roles = pd.get_dummies(df["role"], drop_first=True).to_numpy(dtype=float)
    X = np.column_stack(parts + ([roles] if roles.size else []))

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(df) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t = beta[1] / se[1]
    p = 2 * stats.t.sf(abs(t), dof)
    # log-pay coefficient -> approximate percent gap for group B.
    return {"gap_estimate": float(-beta[1]), "std_error": float(se[1]),
            "t_stat": float(t), "p_value": float(p),
            "ci_low": float(-(beta[1] + 1.96 * se[1])),
            "ci_high": float(-(beta[1] - 1.96 * se[1]))}


def audit_pay_gap(audited: pd.DataFrame, controls: list[str] | None = None) -> dict:
    """Run the audit: unadjusted gap, adjusted (controlled) gap, significance."""
    controls = controls or CONTROLS
    raw = float(np.log(audited.loc[audited["group"] == "A", "adjusted_pay"].mean())
                - np.log(audited.loc[audited["group"] == "B", "adjusted_pay"].mean()))
    adj = _ols_group_effect(audited, controls)
    return {
        "n_employees": int(len(audited)),
        "unadjusted_gap": round(raw, 4),
        "adjusted_gap": round(adj["gap_estimate"], 4),
        "adjusted_ci": [round(adj["ci_low"], 4), round(adj["ci_high"], 4)],
        "p_value": round(adj["p_value"], 4),
        "significant_at_05": adj["p_value"] < 0.05,
    }


def audit_power_fpr(employees: pd.DataFrame,
                    gt: PayEquityGroundTruth | None = None,
                    n_reps: int = 200, seed: int = 0) -> dict:
    """Estimate the audit's power (planted gap) and false-positive rate (no gap).

    Replays group assignment and the audit ``n_reps`` times. Power is the share
    of replicates that flag a significant gap when one is planted; the
    false-positive rate is the share that flag one when the true gap is zero.
    """
    gt = gt or PayEquityGroundTruth()
    rng = np.random.default_rng(seed)
    null_gt = PayEquityGroundTruth(gt.group_b_share, 0.0, gt.confounding_strength)

    power_hits, fpr_hits, gap_estimates = 0, 0, []
    for i in range(n_reps):
        s = int(rng.integers(1, 1_000_000))
        planted = audit_pay_gap(assign_group_and_gap(employees, gt, seed=s))
        power_hits += planted["significant_at_05"]
        gap_estimates.append(planted["adjusted_gap"])
        nulled = audit_pay_gap(assign_group_and_gap(employees, null_gt, seed=s + 1))
        fpr_hits += nulled["significant_at_05"]

    return {
        "n_reps": n_reps,
        "planted_gap": gt.true_residual_gap,
        "power": round(power_hits / n_reps, 3),
        "false_positive_rate": round(fpr_hits / n_reps, 3),
        "mean_gap_estimate": round(float(np.mean(gap_estimates)), 4),
        "gap_estimate_bias": round(float(np.mean(gap_estimates) - gt.true_residual_gap), 4),
    }
