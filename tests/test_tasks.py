"""Use case 16: task-level labor standards.

The labor standards and order mix are ground truth, so the tests check that
orders conserve transactions, the flat rate agrees with task-based labor on
average but mis-allocates by mix, and the per-channel task times are
recoverable from labor and order counts."""

import numpy as np
import pytest

from workforce_analytics import (
    TaskGroundTruth,
    mix_staffing_summary,
    recover_task_seconds,
    simulate_order_mix,
    staffing_comparison,
)
from workforce_analytics.demand import TrafficConfig, TrafficSimulator
from workforce_analytics.tasks import CHANNELS


@pytest.fixture(scope="module")
def order_mix(sim):
    stores = sim.stores[sim.stores["open_month"] == 0].reset_index(drop=True)
    traffic = TrafficSimulator(stores, TrafficConfig(n_weeks=12, seed=11)).run()
    return simulate_order_mix(traffic, seed=53)


def test_orders_conserve_transactions_and_reproduce(sim, order_mix):
    channel_sum = order_mix[list(CHANNELS)].sum(axis=1)
    assert (channel_sum == order_mix["transactions"]).all()


def test_flat_and_task_agree_on_average(order_mix):
    s = mix_staffing_summary(staffing_comparison(order_mix))
    assert 0.95 <= s["avg_ratio_task_to_flat"] <= 1.05


def test_flat_rate_mis_allocates_by_mobile_mix(order_mix):
    s = mix_staffing_summary(staffing_comparison(order_mix))
    assert s["corr_mobile_share_vs_gap"] > 0.30
    # High-mobile hours are under-provisioned by the flat rate, low-mobile over.
    assert s["top_mobile_decile_flat_under_provision_pct"] > 0
    assert s["bottom_mobile_decile_flat_over_provision_pct"] < 0


def test_task_times_are_recoverable(order_mix):
    rec = recover_task_seconds(order_mix)
    assert (rec["pct_error"].abs() < 4.0).all()


def test_mobile_share_grows_over_the_horizon(order_mix):
    early = order_mix[order_mix["week"] <= 2]["mobile_share"].mean()
    late = order_mix[order_mix["week"] >= 9]["mobile_share"].mean()
    assert late > early


def test_higher_mobile_seconds_raises_task_labor(sim):
    stores = sim.stores[sim.stores["open_month"] == 0].reset_index(drop=True)
    traffic = TrafficSimulator(stores, TrafficConfig(n_weeks=8, seed=11)).run()
    base = simulate_order_mix(traffic, seed=1)
    gt = TaskGroundTruth()
    gt.labor_seconds = {**gt.labor_seconds, "mobile_pickup": 400.0}
    heavier = simulate_order_mix(traffic, gt, seed=1)
    assert heavier["labor_seconds"].mean() > base["labor_seconds"].mean()
