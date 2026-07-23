"""Use case 12: staffing-to-sales elasticity, and deriving the cost of an
understaffed hour.

Use cases 5 and 6 price a person-hour of understaffing at a flat $35 — lost
transactions plus service decay — and every scheduling and reserve-staffing
dollar figure rests on that one assumed number. This module replaces the
assumption with a mechanism.

The idea is a service-loss curve. Each worker on the floor can process up to
``capacity_per_head`` transactions an hour; the labor standard that sets
required staff (18/hour in :func:`~workforce_analytics.demand.required_staff`)
is deliberately below that ceiling, so a store that meets the standard has
headroom and loses no sales. Push utilisation past a threshold — by pulling a
head off the floor — and a queue forms: some arriving customers balk or
renege, and those transactions, each worth ``gross_margin_per_txn`` in
contribution, are gone. Because the mechanism is published ground truth (like
the hazard model everywhere else here), the $35 becomes *derived*: it is the
average contribution margin recovered by putting the first missing head back
on the floor, and it can be checked, not asserted.

Two things fall out of the same curve:

* :func:`derive_understaffing_cost` — the dollar cost of a person-hour short,
  built up from margin-per-transaction and the abandonment curve instead of
  assumed. With the defaults it lands at ~$35, validating use case 5.
* :func:`staffing_sales_elasticity` — the percent change in sales per percent
  change in labor hours at the operating point, the number a retail finance
  team asks for when a workforce team requests labor budget. It is small near
  adequate staffing and rises steeply once a store is short-handed, which is
  the whole reason targeted staffing beats staffing every hour alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .demand import required_staff


@dataclass
class ServiceConfig:
    """Published ground truth for the service-loss mechanism.

    Defaults describe a quick-service store: a worker can push
    ``capacity_per_head`` transactions an hour under pressure, comfortably
    above the 18/hour planning standard, so meeting the standard loses no
    sales. Abandonment starts once utilisation passes ``abandon_threshold``
    and rises linearly (a readable stand-in for the convex delay/abandon
    curve of an M/M/c queue) up to ``max_abandon``. ``gross_margin_per_txn``
    is contribution margin, not ticket size — roughly a $9 ticket at ~45%.
    """

    capacity_per_head: float = 22.0
    abandon_threshold: float = 0.85
    abandon_slope: float = 0.9
    max_abandon: float = 0.6
    gross_margin_per_txn: float = 4.0

    def abandonment(self, demand: np.ndarray, staff: np.ndarray) -> np.ndarray:
        """Share of arriving demand lost to the queue at this staffing."""
        staff = np.maximum(np.asarray(staff, dtype=float), 1.0)
        utilisation = np.asarray(demand, dtype=float) / (staff * self.capacity_per_head)
        excess = self.abandon_slope * (utilisation - self.abandon_threshold)
        return np.clip(excess, 0.0, self.max_abandon)


def service_outcome(demand, staff, cfg: ServiceConfig | None = None) -> dict:
    """Served transactions, lost transactions and lost margin, elementwise.

    This is the ground-truth mechanism: given arriving demand and the heads
    actually on the floor, how many transactions are served and how much
    contribution margin walks out the door.
    """
    cfg = cfg or ServiceConfig()
    demand = np.asarray(demand, dtype=float)
    lost_share = cfg.abandonment(demand, staff)
    lost = demand * lost_share
    return {
        "served": demand - lost,
        "lost": lost,
        "lost_margin": lost * cfg.gross_margin_per_txn,
    }


def service_loss_curve(
    traffic: pd.DataFrame,
    cfg: ServiceConfig | None = None,
    max_short: int = 3,
    service_rate: float = 18.0,
) -> pd.DataFrame:
    """Lost margin per store-hour as staffing is cut below the requirement.

    For every store-hour, ``required_staff`` sets the head count that meets
    the labor standard. Row ``short = k`` reports the mean lost margin per
    store-hour when the floor runs ``k`` heads below that, and ``marginal``
    is the extra margin lost by cutting the k-th head — the quantity that,
    at ``short`` 0->1, is the honest analogue of the assumed $35.
    """
    cfg = cfg or ServiceConfig()
    demand = traffic["transactions"].to_numpy(dtype=float)
    required = required_staff(demand, service_rate)

    rows = []
    prev = None
    for k in range(max_short + 1):
        staff = np.maximum(required - k, 1)
        lost_margin = service_outcome(demand, staff, cfg)["lost_margin"]
        mean_loss = float(lost_margin.mean())
        # Marginal cost of the k-th head short, averaged only over the
        # store-hours where cutting that head actually loses a sale (a store
        # that is dead at 2pm loses nothing for being one head short).
        if prev is None:
            marginal = 0.0
        else:
            delta = lost_margin - prev
            hurt = delta > 1e-9
            marginal = float(delta[hurt].mean()) if hurt.any() else 0.0
        rows.append({
            "short": k,
            "mean_lost_margin_per_hour": round(mean_loss, 3),
            "marginal_cost_per_head_short": round(marginal, 2),
            "share_hours_affected": round(float((lost_margin > 1e-9).mean()), 4),
        })
        prev = lost_margin
    return pd.DataFrame(rows)


def derive_understaffing_cost(
    traffic: pd.DataFrame,
    cfg: ServiceConfig | None = None,
    service_rate: float = 18.0,
    assumed: float = 35.0,
) -> dict:
    """Derive the $/understaffed-person-hour the scheduler assumes.

    Returns the marginal cost of the first head short (the like-for-like
    replacement for the flat ``assumed`` value the schedule cost model uses),
    the average marginal cost across the first three heads short (which is
    higher, because service loss is convex), and the loss rate at required
    staffing (should be ~0: meeting the standard leaves headroom).
    """
    cfg = cfg or ServiceConfig()
    curve = service_loss_curve(traffic, cfg, max_short=3, service_rate=service_rate)
    first = float(curve.loc[curve["short"] == 1, "marginal_cost_per_head_short"].iloc[0])
    convex = curve.loc[curve["short"] >= 1, "marginal_cost_per_head_short"]
    return {
        "derived_cost_first_head_short": round(first, 2),
        "avg_marginal_cost_3_heads_short": round(float(convex.mean()), 2),
        "loss_rate_at_required_staffing": float(
            curve.loc[curve["short"] == 0, "share_hours_affected"].iloc[0]),
        "assumed_cost": assumed,
        "ratio_derived_to_assumed": round(first / assumed, 3),
        "gross_margin_per_txn": cfg.gross_margin_per_txn,
        "capacity_per_head": cfg.capacity_per_head,
    }


def staffing_sales_elasticity(
    traffic: pd.DataFrame,
    cfg: ServiceConfig | None = None,
    service_rate: float = 18.0,
    reference_short: int = 1,
) -> dict:
    """Percent change in sales per percent change in labor, at an operating point.

    Elasticity is evaluated where it matters — a store already ``reference_short``
    heads below the requirement — by a symmetric finite difference in staffing
    around that point. Sales are served transactions (margin cancels in a
    ratio). The number is deliberately reported at a short-staffed point
    because that is where labor has marginal sales value; at or above the
    standard the store is on the flat of the curve and elasticity is ~0.
    """
    cfg = cfg or ServiceConfig()
    demand = traffic["transactions"].to_numpy(dtype=float)
    required = required_staff(demand, service_rate)
    base = np.maximum(required - reference_short, 1)

    # Only hours where the operating point carries queue loss inform the
    # elasticity; elsewhere labor has no marginal sales effect by construction.
    active = service_outcome(demand, base, cfg)["lost"] > 1e-9
    d, l = demand[active], base[active]
    up = service_outcome(d, l + 1, cfg)["served"]
    down = service_outcome(d, np.maximum(l - 1, 1), cfg)["served"]
    served = service_outcome(d, l, cfg)["served"]

    d_sales = (up - down) / 2.0
    elasticity = np.average(d_sales / served * (l / 1.0), weights=served)
    return {
        "reference_short_heads": reference_short,
        "sales_elasticity_wrt_labor": round(float(elasticity), 3),
        "share_store_hours_on_margin": round(float(active.mean()), 4),
        "note": "elasticity of served transactions w.r.t. heads on floor, "
                "evaluated at a short-staffed operating point",
    }
