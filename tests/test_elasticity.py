"""Use case 12: the service-loss mechanism that derives the understaffing cost.

The ground truth is the ServiceConfig curve; the tests check the properties
that make the derived $/understaffed-hour trustworthy — no loss when the
labor standard is met, convex loss as staffing is cut, and a first-head cost
that lands on the $35 the scheduler assumes."""

import numpy as np
import pytest

from workforce_analytics import (
    ServiceConfig,
    derive_understaffing_cost,
    service_loss_curve,
    service_outcome,
    staffing_sales_elasticity,
)
from workforce_analytics.demand import TrafficConfig, TrafficSimulator


@pytest.fixture(scope="module")
def traffic(sim):
    stores = sim.stores[sim.stores["open_month"] == 0].reset_index(drop=True)
    return TrafficSimulator(stores, TrafficConfig(n_weeks=26, seed=11)).run()


@pytest.fixture(scope="module")
def curve(traffic):
    return service_loss_curve(traffic)


def test_service_outcome_conserves_transactions():
    demand = np.array([10.0, 50.0, 120.0])
    out = service_outcome(demand, staff=np.array([3, 3, 3]))
    np.testing.assert_allclose(out["served"] + out["lost"], demand)
    assert (out["lost"] >= 0).all()
    assert (out["served"] >= 0).all()


def test_no_loss_when_labor_standard_is_met(curve):
    """Meeting required_staff leaves capacity headroom, so no sales are lost."""
    at_required = curve.loc[curve["short"] == 0].iloc[0]
    assert at_required["mean_lost_margin_per_hour"] == 0.0
    assert at_required["share_hours_affected"] == 0.0


def test_loss_grows_and_is_convex_in_understaffing(curve):
    loss = curve.set_index("short")["mean_lost_margin_per_hour"]
    assert loss[0] < loss[1] < loss[2]           # monotone
    assert loss[2] > 2 * loss[1]                 # convex: 2nd head hurts more


def test_derived_cost_validates_the_assumed_35(traffic):
    d = derive_understaffing_cost(traffic)
    assert d["loss_rate_at_required_staffing"] == 0.0
    # The mechanism should reproduce the scheduler's assumed $35 to within 15%.
    assert 0.85 <= d["ratio_derived_to_assumed"] <= 1.15
    # Service loss is convex, so averaging over deeper shortfalls costs more.
    assert d["avg_marginal_cost_3_heads_short"] > d["derived_cost_first_head_short"]


def test_higher_margin_raises_derived_cost(traffic):
    lo = derive_understaffing_cost(traffic, ServiceConfig(gross_margin_per_txn=3.0))
    hi = derive_understaffing_cost(traffic, ServiceConfig(gross_margin_per_txn=5.0))
    assert hi["derived_cost_first_head_short"] > lo["derived_cost_first_head_short"]


def test_more_capacity_lowers_derived_cost(traffic):
    tight = derive_understaffing_cost(traffic, ServiceConfig(capacity_per_head=20.0))
    roomy = derive_understaffing_cost(traffic, ServiceConfig(capacity_per_head=26.0))
    assert roomy["derived_cost_first_head_short"] < tight["derived_cost_first_head_short"]


def test_elasticity_is_positive_and_bounded(traffic):
    e = staffing_sales_elasticity(traffic, reference_short=1)
    assert 0.0 < e["sales_elasticity_wrt_labor"] < 1.0


def test_elasticity_rises_as_stores_get_shorter_staffed(traffic):
    e1 = staffing_sales_elasticity(traffic, reference_short=1)
    e2 = staffing_sales_elasticity(traffic, reference_short=2)
    assert e2["sales_elasticity_wrt_labor"] > e1["sales_elasticity_wrt_labor"]


def test_curve_is_reproducible(traffic):
    a = service_loss_curve(traffic)
    b = service_loss_curve(traffic)
    assert a.equals(b)
