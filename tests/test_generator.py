import numpy as np

from workforce_analytics.config import ALL_ROLES, HOURLY_ROLES


def annualized(monthly_rate: float) -> float:
    return 1 - (1 - monthly_rate) ** 12


def test_turnover_rates_are_realistic(sim):
    """Annualized turnover should sit in published retail ranges per role."""
    pm = sim.person_months
    expected = {
        "barista": (0.45, 0.85),
        "shift_supervisor": (0.20, 0.50),
        "assistant_store_manager": (0.15, 0.40),
        "store_manager": (0.08, 0.30),
    }
    for role, (lo, hi) in expected.items():
        rate = annualized(pm.loc[pm["role"] == role, "terminated"].mean())
        assert lo < rate < hi, f"{role} annualized turnover {rate:.1%} outside [{lo:.0%}, {hi:.0%}]"


def test_each_employee_terminates_at_most_once(sim):
    terms = sim.person_months.groupby("employee_id")["terminated"].sum()
    assert terms.max() <= 1


def test_no_person_months_after_termination(sim):
    pm = sim.person_months
    term_month = pm[pm["terminated"] == 1].set_index("employee_id")["month"]
    last_month = pm.groupby("employee_id")["month"].max()
    joined = term_month.to_frame("term").join(last_month.rename("last"))
    assert (joined["last"] == joined["term"]).all()


def test_workforce_stays_staffed(sim):
    """Hiring should roughly keep up with attrition."""
    ratio = sim.person_months.groupby("month")["store_staffing_ratio"].mean()
    assert ratio.iloc[12:].min() > 0.75


def test_new_hire_hazard_exceeds_veteran_hazard(sim):
    """The tenure washout in the ground truth should appear in the data."""
    pm = sim.person_months
    hourly = pm[pm["role"].isin(HOURLY_ROLES)]
    new = hourly[hourly["tenure_months"] < 3]["terminated"].mean()
    vet = hourly[hourly["tenure_months"] >= 24]["terminated"].mean()
    assert new > 2 * vet


def test_pay_and_features_in_sane_ranges(sim):
    pm = sim.person_months
    assert pm["pay_ratio"].between(0.6, 1.6).all()
    assert pm["commute_km"].between(0, 80).all()
    assert (pm["hours_gap"] >= 0).all()
    assert set(pm["role"].unique()) == set(ALL_ROLES)
    assert pm["performance_rating"].between(1, 5).all()


def test_reproducible_with_seed(sim, cfg):
    from workforce_analytics import generate

    again = generate(cfg)
    assert len(again.person_months) == len(sim.person_months)
    np.testing.assert_allclose(
        again.person_months["pay_ratio"].to_numpy(),
        sim.person_months["pay_ratio"].to_numpy())
