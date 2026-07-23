"""Use case 17: pay-equity audit, validated against a planted gap.

Because the residual gap is ground truth, the tests check that the audit
recovers it, that controlling for a confounder is what makes the estimate
right, and that the audit's power and false-positive rate behave."""

import pytest

from workforce_analytics import (
    PayEquityGroundTruth,
    assign_group_and_gap,
    audit_pay_gap,
    audit_power_fpr,
    build_employee_frame,
)


@pytest.fixture(scope="module")
def employees(sim):
    return build_employee_frame(sim.person_months, sim.stores)


def test_employee_frame_is_one_row_per_employee(employees):
    assert employees["employee_id"].is_unique
    assert (employees["pay_rate"] > 0).all()


def test_audit_recovers_the_planted_gap(employees):
    audited = assign_group_and_gap(
        employees, PayEquityGroundTruth(true_residual_gap=0.04), seed=61)
    result = audit_pay_gap(audited)
    assert abs(result["adjusted_gap"] - 0.04) < 0.005     # recovers within half a point
    assert result["significant_at_05"]
    lo, hi = result["adjusted_ci"]
    assert lo < result["adjusted_gap"] < hi               # CI brackets the estimate


def test_controls_fix_confounding(employees):
    """Group correlated with tenure inflates the unadjusted gap; controlling
    for tenure brings the estimate back to the truth."""
    audited = assign_group_and_gap(
        employees, PayEquityGroundTruth(true_residual_gap=0.04, confounding_strength=1.0),
        seed=61)
    result = audit_pay_gap(audited)
    assert result["unadjusted_gap"] > result["adjusted_gap"] + 0.01
    assert abs(result["adjusted_gap"] - 0.04) < 0.01


def test_no_planted_gap_is_not_flagged(employees):
    audited = assign_group_and_gap(
        employees, PayEquityGroundTruth(true_residual_gap=0.0), seed=7)
    result = audit_pay_gap(audited)
    assert result["adjusted_ci"][0] <= 0.0 <= result["adjusted_ci"][1]


def test_power_and_false_positive_rate(employees):
    pf = audit_power_fpr(
        employees, PayEquityGroundTruth(true_residual_gap=0.03), n_reps=40, seed=0)
    assert pf["power"] >= 0.8                       # well-powered at this sample size
    assert pf["false_positive_rate"] <= 0.15        # near the nominal 5% level
    assert abs(pf["gap_estimate_bias"]) < 0.005     # unbiased estimator


def test_assignment_is_reproducible(employees):
    a = assign_group_and_gap(employees, seed=61)
    b = assign_group_and_gap(employees, seed=61)
    assert (a["group"] == b["group"]).all()
