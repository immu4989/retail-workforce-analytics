"""Use case 14: first-90-day onboarding risk.

Checks the washout process is realistic and reproducible, the model tracks the
oracle ceiling, and the training-completion gap behaves like a confounded
marker — the observed gap shrinks once the shared onboarding-quality driver is
removed, which is what proves it is not all causal."""

import numpy as np
import pytest

from workforce_analytics import (
    OnboardingGroundTruth,
    OnboardingModel,
    milestone_retention,
    new_hire_watchlist,
    simulate_onboarding,
)
from workforce_analytics.config import HOURLY_ROLES


@pytest.fixture(scope="module")
def cohort(sim):
    return simulate_onboarding(sim.person_months, seed=29)


@pytest.fixture(scope="module")
def split(cohort):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(cohort))
    cut = int(0.7 * len(cohort))
    return cohort.iloc[idx[:cut]], cohort.iloc[idx[cut:]]


def test_cohort_is_hourly_new_hires_and_reproducible(sim):
    a = simulate_onboarding(sim.person_months, seed=29)
    b = simulate_onboarding(sim.person_months, seed=29)
    assert a.equals(b)
    assert set(a["role"].unique()) <= set(HOURLY_ROLES)


def test_washout_rate_is_realistic(cohort):
    assert 0.35 <= cohort["washed_out_90d"].mean() <= 0.60


def test_model_tracks_the_oracle_ceiling(split):
    train, test = split
    m = OnboardingModel().fit(train)
    e = m.evaluate(test)
    assert e["ceiling_auc"] > e["model_auc"]      # can't beat the truth
    assert e["model_auc"] > 0.60
    assert e["pct_of_ceiling"] >= 80.0


def test_milestone_retention_is_monotone(cohort):
    m = milestone_retention(cohort)
    assert m["day_30_retention"] >= m["day_60_retention"] >= m["day_90_retention"]


def test_completers_wash_out_less(cohort):
    m = milestone_retention(cohort)
    assert m["washout_if_completed"] < m["washout_if_incomplete"]
    assert m["observed_completion_gap"] > 0.10


def test_training_gap_is_partly_confounding(sim):
    """Removing the shared onboarding-quality driver (so training no longer
    marks quality) shrinks the completer gap: the observed gap was not all
    causal."""
    confounded = milestone_retention(simulate_onboarding(sim.person_months, seed=5))
    deconfounded = milestone_retention(simulate_onboarding(
        sim.person_months,
        OnboardingGroundTruth(train_quality_effect=0.0, onboarding_quality_sd=0.0),
        seed=5))
    assert deconfounded["observed_completion_gap"] < confounded["observed_completion_gap"]


def test_watchlist_concentrates_risk(split, cohort):
    train, test = split
    m = OnboardingModel().fit(train)
    wl = new_hire_watchlist(m, test, top_frac=0.20)
    assert len(wl) == max(int(len(test) * 0.20), 1)
    assert (wl["washout_risk"].diff().dropna() <= 1e-9).all()   # sorted desc
    assert wl["top_reasons"].str.len().gt(0).all()
