"""The real-data audit is validated the way everything else here is: bugs are
*planted* with a per-employee log, then recall and precision are measured
against that ground truth instead of asserted."""

import pytest
from sklearn.metrics import roc_auc_score

from workforce_analytics import build_snapshots
from workforce_analytics.realdata import (
    _truncate_at_termination,
    audit_split,
    make_messy_extract,
    validate_person_months,
)


@pytest.fixture(scope="module")
def messy(sim):
    return make_messy_extract(sim.person_months, seed=3)


@pytest.fixture(scope="module")
def messy_report(messy):
    return validate_person_months(messy.person_months)


def _finding(report, code):
    match = [f for f in report.findings if f.code == code]
    assert match, f"expected finding {code!r}, got {[f.code for f in report.findings]}"
    return match[0]


def test_clean_simulator_output_passes(sim):
    report = validate_person_months(sim.person_months)
    assert report.ok
    assert report.findings == []


def test_missing_columns_is_an_error(sim):
    report = validate_person_months(sim.person_months.drop(columns=["pay_ratio"]))
    assert not report.ok
    assert _finding(report, "missing_columns").severity == "error"
    with pytest.raises(ValueError):
        report.raise_if_errors()


def test_trailing_payroll_rows_caught_exactly(messy, messy_report):
    """The one planted contract violation: perfect precision and recall."""
    found = set(_finding(messy_report, "post_termination_rows").employee_ids)
    assert found == set(messy.planted("trailing_payroll_rows"))


def _detection_rates(messy, messy_report, bug, code, min_rows):
    """Recall on detectable planted ids (long-enough spells) and precision.

    Spell length counts active months only: the linters truncate at the
    termination row, so planted trailing rows don't make a spell "longer"."""
    rows = _truncate_at_termination(messy.person_months).groupby("employee_id").size()
    planted = set(messy.planted(bug))
    detectable = {i for i in planted if rows[i] >= min_rows}
    found = set(_finding(messy_report, code).employee_ids)
    recall = len(found & detectable) / len(detectable)
    precision = len(found & planted) / len(found)
    return recall, precision


def test_termination_pay_spike_detection(messy, messy_report):
    # Single-month leavers have no prior month to compare, hence min_rows=2.
    recall, precision = _detection_rates(
        messy, messy_report, "termination_pay_spike", "termination_pay_spike",
        min_rows=2)
    assert recall >= 0.99
    assert precision >= 0.99


def test_notice_period_hours_collapse_detection(messy, messy_report):
    # Spells under 4 months have no trailing baseline, hence min_rows=4.
    recall, precision = _detection_rates(
        messy, messy_report, "notice_period_hours_collapse",
        "notice_period_hours_collapse", min_rows=4)
    assert recall >= 0.99
    assert precision >= 0.99


def test_backfilled_ratings_flagged(messy_report):
    assert _finding(messy_report, "backfilled_ratings").severity == "warning"


def test_audit_split_accepts_time_split(splits):
    train, _, test = splits
    assert audit_split(train, test) == []


def test_audit_split_flags_random_split(snapshots):
    shuffled = snapshots.sample(frac=1.0, random_state=0)
    half = len(shuffled) // 2
    findings = audit_split(shuffled.iloc[:half], shuffled.iloc[half:])
    assert [f.code for f in findings] == ["overlapping_split_months"]


def test_notice_leakage_inflates_single_feature_auc(sim, messy):
    """The planted in-notice bug produces exactly the fake signal the doc
    warns about: scheduled_hours alone gets a large AUC lift on messy data."""
    def hours_auc(pm):
        snaps = build_snapshots(pm, horizons=(3,))
        h = snaps[snaps["role"].isin(["barista", "shift_supervisor"])
                  & snaps["label_3m"].notna()]
        return roc_auc_score(h["label_3m"], -h["scheduled_hours"])

    clean, leaky = hours_auc(sim.person_months), hours_auc(messy.person_months)
    assert leaky > clean + 0.03


def test_pay_spike_inflates_exit_pay_stats(sim, messy):
    """Severance stamped on termination rows corrupts exit-pay analytics."""
    def mean_final_pay(pm):
        term = pm[pm["terminated"] == 1]
        return term.groupby("employee_id")["pay_rate"].last().mean()

    clean = mean_final_pay(sim.person_months)
    assert mean_final_pay(messy.person_months) > clean * 1.05
