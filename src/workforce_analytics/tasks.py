"""Use case 16: task-level labor standards from order mix.

Use case 5 converts transactions to heads with one number — 18 transactions per
labor hour. That is an *average*, and it is wrong the moment the order mix
moves. A mobile order that has to be assembled and staged is more labor than a
front-counter transaction rung up in one motion; an hour that is 40% mobile
needs more people than its transaction count implies, and an hour that is all
quick front-counter needs fewer. Real workforce-management systems build the
requirement up from task times per order type. This module does that, and
shows what the flat rate misses.

The mechanism is published ground truth: each order channel (front counter,
drive-thru, mobile pickup, delivery) carries a fully-loaded labor-seconds
standard, the mix shifts by daypart and drifts toward mobile over the horizon,
and per-order times carry noise. Two checkable claims come out:

1. **Recovery** — the per-channel task times can be recovered from observed
   labor and order counts by non-negative least squares, matching the planted
   standards within ~1%. This is the oracle: the labor standard is estimable,
   not assumed.
2. **The flat rate mis-staffs by mix** — calibrated to agree with task-based
   labor *on average*, the flat rate still under-provisions the highest-mobile
   hours by ~20% and over-provisions the lowest, and the error correlates with
   mobile share. Same total labor, wrong hour-by-hour allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNELS = ("front_counter", "drive_thru", "mobile_pickup", "delivery")
FLAT_SERVICE_RATE = 18.0


def _daypart(hour: int) -> int:
    return 0 if hour < 11 else 1 if hour < 14 else 2 if hour < 17 else 3


@dataclass
class TaskGroundTruth:
    """True labor standards and order-mix structure.

    ``labor_seconds`` is fully-loaded labor per order by channel (task time plus
    allowances), scaled so the mix-weighted average reproduces the flat 18/hour
    rate; the flat rate is therefore right on average and only wrong by mix.
    ``daypart_mix`` rows are channel shares for morning/lunch/afternoon/evening.
    """

    labor_seconds: dict = field(default_factory=lambda: {
        "front_counter": 135.0, "drive_thru": 157.0,
        "mobile_pickup": 261.0, "delivery": 296.0})
    daypart_mix: tuple = (
        (0.28, 0.42, 0.25, 0.05),   # morning
        (0.34, 0.46, 0.12, 0.08),   # lunch
        (0.40, 0.35, 0.15, 0.10),   # afternoon
        (0.28, 0.30, 0.20, 0.22),   # evening
    )
    mobile_growth: float = 0.15      # absolute share shifted to mobile by horizon end
    store_mobile_sd: float = 0.06    # per-store mobile-adoption spread
    time_noise_sd: float = 0.15      # per-order lognormal labor-time noise
    min_staff: int = 2

    def seconds_vector(self) -> np.ndarray:
        return np.array([self.labor_seconds[c] for c in CHANNELS])

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def simulate_order_mix(traffic: pd.DataFrame, gt: TaskGroundTruth | None = None,
                       seed: int = 53) -> pd.DataFrame:
    """Split each store-hour's transactions into channel order counts + labor.

    Returns the traffic rows plus one column per channel, the true
    ``labor_seconds`` for the hour (task times x orders x noise), and
    ``mobile_share``.
    """
    gt = gt or TaskGroundTruth()
    rng = np.random.default_rng(seed)
    t = traffic.copy().reset_index(drop=True)
    n = len(t)
    dp = t["hour"].map(_daypart).to_numpy()
    mix = np.array(gt.daypart_mix)
    max_week = max(int(t["week"].max()), 1)
    week_shift = gt.mobile_growth * (t["week"].to_numpy() / max_week)
    stores = t["store_id"].unique()
    store_shift = t["store_id"].map(
        {s: rng.normal(0, gt.store_mobile_sd) for s in stores}).to_numpy()

    txn = t["transactions"].to_numpy()
    counts = np.zeros((n, len(CHANNELS)), dtype=int)
    shift = week_shift + store_shift
    for i in range(n):
        p = mix[dp[i]].copy()
        p[0] = max(p[0] - shift[i], 0.01)          # mobile grows at front counter's expense
        p[2] = p[2] + max(shift[i], 0.0)
        p = np.clip(p, 0.01, None)
        p /= p.sum()
        counts[i] = rng.multinomial(int(txn[i]), p)

    sec = gt.seconds_vector()
    noise = rng.lognormal(0, gt.time_noise_sd, size=counts.shape)
    labor_seconds = (counts * sec * noise).sum(axis=1)

    for j, c in enumerate(CHANNELS):
        t[c] = counts[:, j]
    total = counts.sum(axis=1)
    t["labor_seconds"] = np.round(labor_seconds, 1)
    t["mobile_share"] = np.divide(counts[:, 2], total,
                                  out=np.zeros(n), where=total > 0)
    return t


def staffing_comparison(order_mix: pd.DataFrame,
                        service_rate: float = FLAT_SERVICE_RATE,
                        min_staff: int = 2) -> pd.DataFrame:
    """Per store-hour flat-rate vs task-based labor hours and integer heads."""
    df = order_mix.copy()
    df["flat_labor_hours"] = df["transactions"] / service_rate
    df["task_labor_hours"] = df["labor_seconds"] / 3600.0
    df["labor_hours_gap"] = df["task_labor_hours"] - df["flat_labor_hours"]
    df["flat_heads"] = np.maximum(np.ceil(df["flat_labor_hours"]), min_staff).astype(int)
    df["task_heads"] = np.maximum(np.ceil(df["task_labor_hours"]), min_staff).astype(int)
    df["head_gap"] = df["task_heads"] - df["flat_heads"]
    return df


def mix_staffing_summary(comparison: pd.DataFrame) -> dict:
    """How the flat rate mis-allocates labor as a function of order mix."""
    flat, task = comparison["flat_labor_hours"], comparison["task_labor_hours"]
    mob = comparison["mobile_share"]
    gap = comparison["labor_hours_gap"]
    hi = mob >= mob.quantile(0.90)
    lo = mob <= mob.quantile(0.10)
    return {
        "n_store_hours": int(len(comparison)),
        "avg_flat_labor_hours": round(float(flat.mean()), 3),
        "avg_task_labor_hours": round(float(task.mean()), 3),
        "avg_ratio_task_to_flat": round(float(task.mean() / flat.mean()), 3),
        "corr_mobile_share_vs_gap": round(float(np.corrcoef(mob, gap)[0, 1]), 3),
        "top_mobile_decile_flat_under_provision_pct": round(
            float(task[hi].sum() / flat[hi].sum() - 1) * 100, 1),
        "bottom_mobile_decile_flat_over_provision_pct": round(
            float(task[lo].sum() / flat[lo].sum() - 1) * 100, 1),
        "share_hours_flat_understaffs": round(float((comparison["head_gap"] > 0).mean()), 3),
    }


def recover_task_seconds(order_mix: pd.DataFrame,
                         gt: TaskGroundTruth | None = None) -> pd.DataFrame:
    """Recover per-channel labor standards from labor + order counts (oracle).

    Non-negative least squares of observed labor seconds on channel order
    counts; the coefficients are the estimated per-order task times, compared to
    the planted ground truth.
    """
    from scipy.optimize import nnls

    gt = gt or TaskGroundTruth()
    X = order_mix[list(CHANNELS)].to_numpy(dtype=float)
    y = order_mix["labor_seconds"].to_numpy(dtype=float)
    coef, _ = nnls(X, y)
    true = gt.seconds_vector()
    return pd.DataFrame({
        "channel": CHANNELS,
        "true_seconds": true,
        "recovered_seconds": np.round(coef, 1),
        "pct_error": np.round((coef - true) / true * 100, 2),
    })
