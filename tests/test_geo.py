"""Use case 18: geographic transfer matching.

The commute-to-hazard coefficient is ground truth, so the payoff of a transfer
is exact. The tests check the geography is clustered by district, every transfer
strictly reduces commute and respects vacancy capacity, and the Hungarian
optimum is never beaten by the greedy baseline."""

import numpy as np
import pytest

from workforce_analytics import (
    assign_homes,
    assign_store_coordinates,
    compare_strategies,
    greedy_transfers,
    optimize_transfers,
    store_vacancies,
    transfer_options,
    transfer_payoff,
    workforce_snapshot,
)

MONTH = 40


@pytest.fixture(scope="module")
def geo_setup(sim):
    coords = assign_store_coordinates(sim.stores, seed=71)
    snap = workforce_snapshot(sim.person_months, month=MONTH)
    homes = assign_homes(snap, coords, seed=72)
    vac = store_vacancies(snap, sim.stores)
    options = transfer_options(homes, coords, vac)
    return coords, snap, homes, vac, options


def test_stores_cluster_by_district(geo_setup):
    coords = geo_setup[0]
    # Mean within-district pairwise distance is well below across-district.
    def spread(df):
        p = df[["store_x", "store_y"]].to_numpy()
        return np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1)).mean()
    within = coords.groupby("district_id").apply(spread, include_groups=False).mean()
    allpts = spread(coords)
    assert within < allpts


def test_every_transfer_reduces_commute(geo_setup):
    options = geo_setup[4]
    assert (options["commute_reduction_km"] > 0).all()
    assert (options["exits_avoided"] > 0).all()


def test_assignment_respects_capacity_and_uniqueness(geo_setup):
    _, _, _, vac, options = geo_setup
    assign = optimize_transfers(options, vac)
    assert assign["employee_id"].is_unique
    per_store = assign.groupby("to_store").size()
    assert (per_store <= vac.reindex(per_store.index)).all()
    assert (assign["commute_reduction_km"] > 0).all()


def test_optimal_beats_or_matches_greedy(geo_setup):
    _, _, _, vac, options = geo_setup
    opt = transfer_payoff(optimize_transfers(options, vac))
    greedy = transfer_payoff(greedy_transfers(options, vac))
    assert opt["exits_avoided"] >= greedy["exits_avoided"] - 1e-9


def test_transfers_beat_doing_nothing(geo_setup):
    _, _, _, vac, options = geo_setup
    cmp = compare_strategies(options, vac)
    assert cmp["optimal"]["exits_avoided"] > 0
    assert cmp["optimal"]["dollars_saved"] > 0
    assert cmp["optimal_vs_greedy_extra_exits_avoided"] >= -1e-9


def test_homes_sit_at_commute_distance(geo_setup):
    coords, _, homes, _, _ = geo_setup
    xy = coords.set_index("store_id")[["store_x", "store_y"]]
    s = homes.merge(xy, on="store_id", suffixes=("", "_s"))
    dist = np.hypot(s["home_x"] - s["store_x"], s["home_y"] - s["store_y"])
    np.testing.assert_allclose(dist.to_numpy(), s["commute_km"].to_numpy(), atol=1e-6)
