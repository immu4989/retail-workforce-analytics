import numpy as np
import pandas as pd
import pytest

from workforce_analytics import (
    CallTopicModel,
    SimulationConfig,
    WageProgram,
    generate,
    operational_linkage,
    pay_elasticity,
    run_wage_experiment,
    simulate_calls,
)
from workforce_analytics.config import HOURLY_ROLES

SMALL = SimulationConfig(n_districts=4, n_months=36, seed=9)


# ------------------------------------------------------------- compensation

def test_empty_programs_do_not_change_the_world(sim, cfg):
    from dataclasses import replace
    again = generate(replace(cfg, wage_programs=[]))
    np.testing.assert_allclose(again.person_months["pay_ratio"].to_numpy(),
                               sim.person_months["pay_ratio"].to_numpy())


def test_paired_runs_identical_before_program():
    from dataclasses import replace
    prog = WageProgram("bump", pct=0.05, start_month=20)
    base = generate(replace(SMALL, wage_programs=[]))
    treat = generate(replace(SMALL, wage_programs=[prog]))
    pre_b = base.person_months[base.person_months["month"] < 20]
    pre_t = treat.person_months[treat.person_months["month"] < 20]
    assert len(pre_b) == len(pre_t)
    np.testing.assert_allclose(pre_b["pay_ratio"].to_numpy(),
                               pre_t["pay_ratio"].to_numpy())
    # After the program, treated hourly pay position is visibly higher.
    post_t = treat.person_months.query("month >= 20 and role in @HOURLY_ROLES")
    post_b = base.person_months.query("month >= 20 and role in @HOURLY_ROLES")
    assert post_t["pay_ratio"].mean() > post_b["pay_ratio"].mean() + 0.02


def test_floor_program_only_lifts_below_market():
    from dataclasses import replace
    prog = WageProgram("floor", kind="floor", floor_ratio=0.97, start_month=12)
    treat = generate(replace(SMALL, wage_programs=[prog]))
    post = treat.person_months.query("month >= 13 and role in @HOURLY_ROLES")
    # With the floor maintained monthly, almost nobody sits below it.
    assert (post["pay_ratio"] < 0.955).mean() < 0.02


def test_freeze_erodes_pay_position():
    from dataclasses import replace
    prog = WageProgram("freeze", kind="freeze", start_month=12, end_month=35)
    base = generate(replace(SMALL, wage_programs=[]))
    treat = generate(replace(SMALL, wage_programs=[prog]))
    post_b = base.person_months.query("month >= 24 and role in @HOURLY_ROLES")
    post_t = treat.person_months.query("month >= 24 and role in @HOURLY_ROLES")
    assert post_t["pay_ratio"].mean() < post_b["pay_ratio"].mean() - 0.005
    assert post_t["terminated"].mean() > post_b["terminated"].mean() - 0.002


def test_wage_experiment_accounting():
    prog = WageProgram("bump", pct=0.05, start_month=18)
    exp = run_wage_experiment(prog, base_config=SMALL, seeds=(0, 1),
                              eval_start_month=18)
    assert exp["pay_ratio_lift"] > 0.01
    assert exp["extra_wage_bill_mean"] > 0
    assert exp["monthly_exit_rate_program"] < exp["monthly_exit_rate_base"]
    lo, hi = exp["net_value_ci90"]
    assert lo <= exp["net_value_mean"] <= hi


def test_pay_elasticity_brackets_are_sane(sim):
    table = pay_elasticity(sim.person_months)
    coefs = table.set_index("specification")["pay_ratio_log_odds"]
    # Both specifications must find that higher pay reduces exits, with the
    # true effect (-2.5) inside a generous bracket around them.
    assert (coefs < -0.5).all()
    assert coefs.min() > -6.0


# ------------------------------------------------------------- call center

@pytest.fixture(scope="module")
def calls(sim):
    return simulate_calls(sim.person_months, seed=3)


def test_call_volume_is_realistic(sim, calls):
    pm_hourly = sim.person_months[sim.person_months["role"].isin(HOURLY_ROLES)]
    rate = len(calls) / len(pm_hourly)
    assert 0.03 < rate < 0.30           # calls per employee-month
    assert calls["true_topic"].nunique() == 6
    assert calls["transcript"].str.len().median() > 60
    # Paraphrase noise: transcripts are (near) unique strings.
    assert calls["transcript"].nunique() > 0.95 * len(calls)


def test_topic_model_recovers_planted_topics(calls):
    tm = CallTopicModel().fit(calls)
    ev = tm.evaluate(calls)
    assert ev["nmi"] > 0.6
    assert ev["purity"] > 0.75
    # The recovered clusters must name most of the true topics.
    assert len(set(ev["cluster_to_topic"].values())) >= 4


def test_operational_linkage_signs(sim, calls):
    linkage = operational_linkage(calls, sim.person_months)
    sched = linkage[linkage["topic"] == "scheduling"].iloc[0]
    pay = linkage[linkage["topic"] == "pay_error"].iloc[0]
    assert sched["correlation_across_stores"] > 0.3
    assert pay["correlation_across_stores"] < 0
