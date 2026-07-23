"""Use case 15: exit-interview NLP.

The themes are planted from each leaver's true exit-state drivers, so the tests
check three things against that ground truth: the topic model recovers the
themes, each recovered theme is elevated on its true driver (alignment), and
the whole thing is reproducible."""

import pytest

from workforce_analytics import (
    ExitNLPGroundTruth,
    ExitThemeModel,
    simulate_exit_comments,
    theme_by_regrettability,
    theme_driver_alignment,
)
from workforce_analytics.exitnlp import THEMES


@pytest.fixture(scope="module")
def comments(sim):
    return simulate_exit_comments(sim.person_months, seed=47)


def test_comments_cover_all_themes_and_reproduce(sim, comments):
    again = simulate_exit_comments(sim.person_months, seed=47)
    assert comments.equals(again)
    assert set(comments["true_theme"].unique()) == set(THEMES)


def test_topic_model_recovers_themes(comments):
    e = ExitThemeModel().fit(comments).evaluate(comments)
    assert e["nmi"] >= 0.80
    assert e["purity"] >= 0.85
    # Six clusters map onto six distinct themes.
    assert len(set(e["cluster_to_theme"].values())) == len(THEMES)


def test_every_theme_aligns_with_its_true_driver(comments):
    align = theme_driver_alignment(comments)
    assert align["aligned"].all(), align[~align["aligned"]]


def test_regrettability_split_sums_to_one(comments):
    tab = theme_by_regrettability(comments)
    for _, grp in tab.groupby("termination_type"):
        assert abs(grp["share"].sum() - 1.0) < 1e-6


def test_stronger_weight_sharpens_a_theme(sim):
    """Raising a theme's weight should pull more leavers into it."""
    base = simulate_exit_comments(sim.person_months, seed=5)
    boosted = simulate_exit_comments(
        sim.person_months, ExitNLPGroundTruth(pay_weight=5.0), seed=5)
    assert (boosted["true_theme"] == "pay").sum() > (base["true_theme"] == "pay").sum()
