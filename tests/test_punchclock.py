"""Use case 13: payroll-anomaly detection, validated against planted truth.

Each anomaly type is injected with a per-record log, so the detectors' recall,
precision and average precision are measured, not asserted — and a clean panel
with no anomalies must stay quiet."""

import pytest

from workforce_analytics import (
    AnomalyGroundTruth,
    detect_buddy_punching,
    detect_ghost_shifts,
    detect_time_padding,
    evaluate_detection,
    labor_leakage,
    simulate_punches,
)


@pytest.fixture(scope="module")
def panel(sim):
    return simulate_punches(sim.person_months, seed=0)


def test_panel_covers_only_hourly_and_is_reproducible(sim):
    a = simulate_punches(sim.person_months, seed=0)
    b = simulate_punches(sim.person_months, seed=0)
    assert set(a.punches["role"].unique()) <= {"barista", "shift_supervisor"}
    assert a.punches.equals(b.punches)


def test_time_padding_detected(panel):
    m = evaluate_detection(detect_time_padding(panel), "is_padder")
    assert m["recall"] >= 0.95
    assert m["precision"] >= 0.85
    assert m["average_precision"] >= 0.95


def test_ghost_shifts_detected(panel):
    m = evaluate_detection(detect_ghost_shifts(panel), "is_ghost_month")
    assert m["recall"] >= 0.85
    assert m["average_precision"] >= 0.90


def test_buddy_punching_detected(panel):
    m = evaluate_detection(detect_buddy_punching(panel), "is_buddy", "within_emp_z")
    assert m["recall"] >= 0.75
    assert m["average_precision"] >= 0.85


def test_padders_do_not_dominate_buddy_flags(panel):
    """A habitual padder's months look alike, so within-employee detection
    should not mistake them for one-off buddy punches."""
    flagged = detect_buddy_punching(panel)
    hits = flagged[flagged["flagged"]].merge(
        panel.punches[["employee_id", "month", "is_padder"]], on=["employee_id", "month"])
    assert hits["is_padder"].mean() < 0.15


def test_clean_panel_stays_quiet(sim):
    """No planted anomalies -> the detectors should flag almost nothing."""
    clean = simulate_punches(sim.person_months, AnomalyGroundTruth(
        padder_share=0.0, ghost_month_share=0.0, buddy_event_share=0.0), seed=1)
    pad = detect_time_padding(clean)
    ghost = detect_ghost_shifts(clean)
    assert pad["flagged"].mean() < 0.01
    assert ghost["flagged"].mean() < 0.01


def test_leakage_is_positive_and_split_by_kind(panel):
    leak = labor_leakage(panel)
    assert leak["annual_leakage_total"] > 0
    assert leak["annual_leakage_time_padding"] > 0
    assert leak["annual_leakage_ghost_shifts"] > 0
    assert leak["annual_leakage_buddy_punching"] > 0


def test_more_padding_raises_leakage(sim):
    light = labor_leakage(simulate_punches(sim.person_months, AnomalyGroundTruth(
        padding_pct=(0.05, 0.08)), seed=2))
    heavy = labor_leakage(simulate_punches(sim.person_months, AnomalyGroundTruth(
        padding_pct=(0.20, 0.35)), seed=2))
    assert heavy["annual_leakage_time_padding"] > light["annual_leakage_time_padding"]
