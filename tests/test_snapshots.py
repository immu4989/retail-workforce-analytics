import numpy as np

from workforce_analytics.snapshots import ALL_FEATURES


def test_labels_match_actual_terminations(sim, snapshots):
    """Spot-check: label_6m == an exit recorded in (t, t+6] for that employee."""
    pm = sim.person_months
    term_month = pm[pm["terminated"] == 1].set_index("employee_id")["month"]
    sample = snapshots[snapshots["label_6m"].notna()].sample(500, random_state=0)
    tm = sample["employee_id"].map(term_month)
    expected = ((tm > sample["month"]) & (tm <= sample["month"] + 6)).fillna(False)
    assert (sample["label_6m"].astype(bool) == expected).all()


def test_censored_windows_are_nan(sim, snapshots):
    max_month = sim.person_months["month"].max()
    late = snapshots[snapshots["month"] + 12 > max_month]
    assert late["label_12m"].isna().all()
    ok = snapshots[snapshots["month"] + 12 <= max_month]
    assert ok["label_12m"].notna().all()


def test_snapshot_rows_are_active_employees(sim, snapshots):
    """No one appears in a snapshot for the month they left."""
    pm = sim.person_months
    termed = pm[pm["terminated"] == 1][["employee_id", "month"]]
    merged = snapshots.merge(termed, on=["employee_id", "month"], how="inner")
    assert len(merged) == 0


def test_no_future_information_in_features(snapshots):
    """Feature columns must be point-in-time: only ids, features, labels."""
    label_cols = [c for c in snapshots.columns if c.startswith("label_")]
    id_cols = ["employee_id", "store_id", "district_id", "month"]
    assert set(snapshots.columns) == set(id_cols + ALL_FEATURES + label_cols)


def test_time_split_is_disjoint_and_ordered(snapshots, splits):
    train, val, test = splits
    assert train["month"].max() <= 28
    assert 28 < val["month"].min() and val["month"].max() <= 34
    assert test["month"].min() > 34


def test_rolling_volatility_uses_only_past(sim):
    """schedule_volatility_3m at month t must not depend on months > t."""
    from workforce_analytics import build_snapshots

    pm = sim.person_months
    cutoff = 20
    full = build_snapshots(pm, horizons=(3,), snapshot_months=[cutoff])
    truncated = build_snapshots(pm[pm["month"] <= cutoff], horizons=(3,),
                                snapshot_months=[cutoff])
    a = full.sort_values("employee_id")["schedule_volatility_3m"].to_numpy()
    b = truncated.sort_values("employee_id")["schedule_volatility_3m"].to_numpy()
    np.testing.assert_allclose(a, b)
