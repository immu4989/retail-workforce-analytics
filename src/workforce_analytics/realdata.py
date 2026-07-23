"""Audit a real HRIS extract before modelling on it.

``docs/adapting_to_real_data.md`` documents the person-month contract and the
data mistakes that sink turnover projects: backfilled features, final-month
artifacts on termination rows, random splits. This module turns that page
into code:

* :func:`validate_person_months` checks the contract (schema, one spell per
  employee id, contiguous months, no rows after termination) and runs
  leakage linters for the failure modes that produce a fake AUC.
* :func:`audit_split` catches the random-split mistake after the fact.
* :func:`make_messy_extract` takes clean simulator output and *injects* the
  documented bugs with a per-employee log, so every linter's recall and
  false-positive rate is measured against planted ground truth — the same
  oracle trick the rest of the repo uses for model claims.

Linter thresholds are calibrated on clean simulator output (seeds 7 and 11):
~19% of long-tenure employees legitimately keep one rating, ~0.1% of leavers
show a >10% final-month pay jump, and 0% show a final-month hours collapse.
The flag thresholds below sit well above those baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .snapshots import BINARY_FEATURES, ID_COLUMNS, NUMERIC_FEATURES

# The person-month contract: everything build_snapshots consumes.
# schedule_volatility_3m and month_of_year are derived downstream.
REQUIRED_COLUMNS = (
    ID_COLUMNS
    + [c for c in NUMERIC_FEATURES if c != "schedule_volatility_3m"]
    + ["schedule_volatility", "role", "age_band"]
    + BINARY_FEATURES
    + ["terminated", "termination_type"]
)

# Population-level flag thresholds (clean baselines in parentheses).
CONSTANT_RATING_FLAG = 0.35      # share of >=24-month employees (clean ~0.19)
PAY_SPIKE_FLAG = 0.02            # share of leavers with >10% final jump (clean ~0.001)
HOURS_COLLAPSE_FLAG = 0.02      # share of hourly leavers <50% of trailing median (clean 0.0)
PAY_SPIKE_JUMP = 0.10            # merit raises cap well below this
HOURS_COLLAPSE_RATIO = 0.5

_HOURLY_ROLES = ("barista", "shift_supervisor")


@dataclass
class Finding:
    """One audit result: a contract violation (error) or leakage signal (warning)."""

    code: str
    severity: str  # "error" | "warning"
    message: str
    n_affected: int = 0
    employee_ids: list = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        return f"[{self.severity.upper():7s}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    findings: list

    @property
    def errors(self) -> list:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        """No contract violations (leakage warnings may still be present)."""
        return not self.errors

    def summary(self) -> str:
        if not self.findings:
            return "person-month table passes the contract; no leakage signals."
        return "\n".join(str(f) for f in self.findings)

    def raise_if_errors(self) -> "ValidationReport":
        if self.errors:
            raise ValueError(
                "person-month contract violations:\n"
                + "\n".join(str(f) for f in self.errors)
            )
        return self


def _affected_ids(ids) -> list:
    """Full affected-id list (repr-suppressed on the dataclass)."""
    return list(ids)


# ----------------------------------------------------------------------
# Contract checks (errors)
# ----------------------------------------------------------------------
def _check_schema(pm: pd.DataFrame) -> list:
    missing = [c for c in REQUIRED_COLUMNS if c not in pm.columns]
    if missing:
        return [Finding(
            "missing_columns", "error",
            f"required columns absent: {missing} — see the contract table in "
            "docs/adapting_to_real_data.md", n_affected=len(missing))]
    return []


def _check_nulls(pm: pd.DataFrame) -> list:
    check = [c for c in REQUIRED_COLUMNS if c != "termination_type"]
    nulls = pm[check].isna().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        return [Finding(
            "null_values", "error",
            f"nulls in required columns: {dict(bad)}", n_affected=int(bad.sum()))]
    return []


def _check_duplicates(pm: pd.DataFrame) -> list:
    dup = pm.duplicated(["employee_id", "month"])
    if dup.any():
        ids = pm.loc[dup, "employee_id"].unique()
        return [Finding(
            "duplicate_person_months", "error",
            f"{int(dup.sum())} duplicated (employee_id, month) rows across "
            f"{len(ids)} employees — one row per employee per active month",
            n_affected=len(ids), employee_ids=_affected_ids(ids))]
    return []


def _check_contiguity(pm: pd.DataFrame) -> list:
    gaps = pm.groupby("employee_id")["month"].agg(lambda s: (np.diff(s) != 1).sum())
    bad = gaps[gaps > 0]
    if len(bad):
        return [Finding(
            "gap_in_months", "error",
            f"{len(bad)} employees have non-contiguous months — usually a rehire "
            "reusing the old employee_id (give each employment spell a new id) "
            "or months missing from the extract",
            n_affected=len(bad), employee_ids=_affected_ids(bad.index.to_numpy()))]
    return []


def _check_termination_rows(pm: pd.DataFrame) -> list:
    findings = []
    term = pm[pm["terminated"] == 1]

    counts = term.groupby("employee_id").size()
    multi = counts[counts > 1]
    if len(multi):
        findings.append(Finding(
            "multiple_termination_rows", "error",
            f"{len(multi)} employees terminate more than once — split rehires "
            "into separate employee_ids",
            n_affected=len(multi), employee_ids=_affected_ids(multi.index.to_numpy())))

    term_month = term.groupby("employee_id")["month"].min()
    last_month = pm.groupby("employee_id")["month"].max()
    trailing = last_month.loc[term_month.index] > term_month
    if trailing.any():
        ids = trailing[trailing].index.to_numpy()
        findings.append(Finding(
            "post_termination_rows", "error",
            f"{len(ids)} employees have rows after their termination month — "
            "trailing payroll rows (final pay-out, benefits) must be dropped, "
            "or tenure and exposure are overstated",
            n_affected=len(ids), employee_ids=_affected_ids(ids)))

    bad_type = ~term["termination_type"].isin(["voluntary", "involuntary"])
    if bad_type.any():
        ids = term.loc[bad_type, "employee_id"].unique()
        findings.append(Finding(
            "missing_termination_type", "error",
            f"{len(ids)} terminated employees lack a voluntary/involuntary "
            "termination_type (needed for the regrettable-attrition view)",
            n_affected=len(ids), employee_ids=_affected_ids(ids)))
    return findings


def _check_tenure_step(pm: pd.DataFrame) -> list:
    g = pm.groupby("employee_id")
    step = (g["tenure_months"].diff() - g["month"].diff()).abs()
    bad_rows = pm[step > 0]
    if len(bad_rows):
        ids = bad_rows["employee_id"].unique()
        return [Finding(
            "tenure_out_of_step", "error",
            f"{len(ids)} employees have tenure_months not advancing with month — "
            "tenure is being read from a snapshot column instead of computed "
            "from hire date as of each month",
            n_affected=len(ids), employee_ids=_affected_ids(ids))]
    return []


# ----------------------------------------------------------------------
# Leakage linters (warnings)
# ----------------------------------------------------------------------
def _truncate_at_termination(pm: pd.DataFrame) -> pd.DataFrame:
    """Drop rows after each employee's termination month, so the linters
    measure the final *active* month even when trailing payroll rows (a
    contract error reported separately) are present."""
    term_month = pm[pm["terminated"] == 1].groupby("employee_id")["month"].min()
    tm = pm["employee_id"].map(term_month)
    return pm[tm.isna() | (pm["month"] <= tm)]


def _lint_backfilled_ratings(pm: pd.DataFrame) -> list:
    g = pm.groupby("employee_id")
    span = g["month"].agg(lambda s: s.max() - s.min())
    long_ids = span[span >= 24].index
    if len(long_ids) < 50:
        return []
    nunique = g["performance_rating"].nunique().loc[long_ids]
    share = float((nunique == 1).mean())
    if share > CONSTANT_RATING_FLAG:
        ids = nunique[nunique == 1].index.to_numpy()
        return [Finding(
            "backfilled_ratings", "warning",
            f"{share:.0%} of employees with 24+ months of history show a single "
            f"performance_rating for their entire spell (clean baseline ~19%, "
            f"flag at {CONSTANT_RATING_FLAG:.0%}) — the HRIS has likely stamped "
            "the latest rating onto history; rebuild the column from "
            "effective-dated review records",
            n_affected=len(ids), employee_ids=_affected_ids(ids))]
    return []


def _lint_termination_pay_spike(pm: pd.DataFrame) -> list:
    pm = _truncate_at_termination(pm)
    last2 = pm.groupby("employee_id").tail(2)
    jump = last2.groupby("employee_id")["pay_rate"].agg(
        lambda s: s.iloc[-1] / s.iloc[0] - 1 if len(s) == 2 else np.nan)
    term_ids = pm.loc[pm["terminated"] == 1, "employee_id"].unique()
    jump = jump.reindex(term_ids).dropna()
    if len(jump) < 50:
        return []
    spiked = jump[jump > PAY_SPIKE_JUMP]
    share = len(spiked) / len(jump)
    if share > PAY_SPIKE_FLAG:
        ids = spiked.index.to_numpy()
        return [Finding(
            "termination_pay_spike", "warning",
            f"{share:.1%} of leavers show a >{PAY_SPIKE_JUMP:.0%} pay jump in their "
            f"final month (clean baseline ~0.1%, flag at {PAY_SPIKE_FLAG:.0%}) — "
            "final adjustments (accrued-PTO payout, severance) stamped on the "
            "termination row; they corrupt exit-pay and cost analytics",
            n_affected=len(ids), employee_ids=_affected_ids(ids))]
    return []


def _lint_notice_period_hours(pm: pd.DataFrame) -> list:
    hourly = _truncate_at_termination(pm)
    hourly = hourly[hourly["role"].isin(_HOURLY_ROLES)]
    term_ids = hourly.loc[hourly["terminated"] == 1, "employee_id"].unique()
    spells = hourly[hourly["employee_id"].isin(term_ids)]

    def final_ratio(s):
        if len(s) < 4:
            return np.nan
        return s.iloc[-1] / np.median(s.iloc[:-1])

    ratio = spells.groupby("employee_id")["scheduled_hours"].agg(final_ratio).dropna()
    if len(ratio) < 50:
        return []
    collapsed = ratio[ratio < HOURS_COLLAPSE_RATIO]
    share = len(collapsed) / len(ratio)
    if share > HOURS_COLLAPSE_FLAG:
        ids = collapsed.index.to_numpy()
        return [Finding(
            "notice_period_hours_collapse", "warning",
            f"{share:.1%} of hourly leavers show final-month scheduled hours below "
            f"{HOURS_COLLAPSE_RATIO:.0%} of their trailing median (clean baseline 0%, "
            f"flag at {HOURS_COLLAPSE_FLAG:.0%}) — employees in their notice period "
            "taken off the schedule; snapshots near these months learn to predict "
            "paperwork, not attrition risk. Exclude in-notice employees or drop "
            "their final months",
            n_affected=len(ids), employee_ids=_affected_ids(ids))]
    return []


def validate_person_months(person_months: pd.DataFrame) -> ValidationReport:
    """Audit a person-month table against the contract and leakage linters.

    Errors are hard contract violations that break :func:`build_snapshots`
    or silently corrupt labels. Warnings are population-level leakage
    signals: the table is well-formed, but a model trained on it will post
    an AUC it cannot reproduce in production.
    """
    schema = _check_schema(person_months)
    if schema:
        return ValidationReport(schema)

    pm = person_months.sort_values(["employee_id", "month"])
    findings = (
        _check_nulls(pm)
        + _check_duplicates(pm)
        + _check_contiguity(pm)
        + _check_termination_rows(pm)
        + _check_tenure_step(pm)
        + _lint_backfilled_ratings(pm)
        + _lint_termination_pay_spike(pm)
        + _lint_notice_period_hours(pm)
    )
    return ValidationReport(findings)


def audit_split(train: pd.DataFrame, test: pd.DataFrame) -> list:
    """Catch the random-split mistake on already-built snapshot tables.

    An out-of-time split has every training snapshot month strictly before
    every test month. Interleaved months mean rows were split at random, and
    the same employee's adjacent months sit on both sides — the model
    memorises identity and the test AUC is fiction.
    """
    if train["month"].max() >= test["month"].min():
        overlap_ids = np.intersect1d(train["employee_id"].unique(),
                                     test["employee_id"].unique())
        return [Finding(
            "overlapping_split_months", "error",
            f"train months reach {int(train['month'].max())} but test starts at "
            f"{int(test['month'].min())} — this is a random split, not "
            f"out-of-time ({len(overlap_ids)} employees appear on both sides); "
            "use time_split",
            n_affected=len(overlap_ids), employee_ids=_affected_ids(overlap_ids))]
    return []


# ----------------------------------------------------------------------
# Messy-extract generator (planted ground truth for the linters)
# ----------------------------------------------------------------------
@dataclass
class MessyExtract:
    """Clean simulator output with documented HRIS bugs injected.

    ``injections`` has one row per (employee_id, bug) actually planted, so
    linter recall and precision are measurable, not asserted.
    """

    person_months: pd.DataFrame
    injections: pd.DataFrame

    def planted(self, bug: str) -> np.ndarray:
        return self.injections.loc[self.injections["bug"] == bug,
                                   "employee_id"].to_numpy()


def make_messy_extract(
    person_months: pd.DataFrame,
    seed: int = 0,
    backfill_share: float = 0.6,
    pay_spike_share: float = 0.35,
    hours_collapse_share: float = 0.35,
    trailing_rows_share: float = 0.08,
) -> MessyExtract:
    """Inject the mistakes from ``adapting_to_real_data.md`` into clean data.

    * ``backfilled_rating`` — the latest performance rating stamped onto the
      employee's entire history (HRIS overwrite on a share of all employees).
    * ``termination_pay_spike`` — final-month pay multiplied by 1.15-1.6x
      (accrued-PTO payout / severance on the termination row).
    * ``notice_period_hours_collapse`` — scheduled hours near zero in the
      final two months (in-notice employees taken off the schedule). This is
      the injection that leaks into snapshots: the month before exit is a
      valid snapshot row whose label the collapsed hours give away.
    * ``trailing_payroll_rows`` — 1-2 extra rows after the termination month
      (trailing payroll), a hard contract violation.
    """
    rng = np.random.default_rng(seed)
    pm = person_months.sort_values(["employee_id", "month"]).reset_index(drop=True)
    log = []

    all_ids = pm["employee_id"].unique()
    term_ids = pm.loc[pm["terminated"] == 1, "employee_id"].unique()
    hourly_term_ids = pm.loc[(pm["terminated"] == 1)
                             & pm["role"].isin(_HOURLY_ROLES), "employee_id"].unique()

    def pick(ids, share):
        n = int(round(share * len(ids)))
        return rng.choice(ids, size=n, replace=False)

    # 1. Backfilled ratings: stamp the final rating on every month.
    backfilled = pick(all_ids, backfill_share)
    final_rating = pm.groupby("employee_id")["performance_rating"].last()
    mask = pm["employee_id"].isin(backfilled)
    pm.loc[mask, "performance_rating"] = (
        pm.loc[mask, "employee_id"].map(final_rating).to_numpy())
    log += [(i, "backfilled_rating") for i in backfilled]

    last_row = pm.groupby("employee_id").tail(1).index

    # 2. Termination pay spike on the final row.
    spiked = pick(term_ids, pay_spike_share)
    rows = last_row[pm.loc[last_row, "employee_id"].isin(spiked).to_numpy()]
    pm.loc[rows, "pay_rate"] *= rng.uniform(1.15, 1.6, len(rows))
    log += [(i, "termination_pay_spike") for i in spiked]

    # 3. Notice-period hours collapse on the final two months.
    collapsed = pick(hourly_term_ids, hours_collapse_share)
    last2 = pm.groupby("employee_id").tail(2).index
    rows = last2[pm.loc[last2, "employee_id"].isin(collapsed).to_numpy()]
    pm.loc[rows, "scheduled_hours"] = np.round(rng.uniform(0, 6, len(rows)), 1)
    log += [(i, "notice_period_hours_collapse") for i in collapsed]

    # 4. Trailing payroll rows after termination (kept disjoint from the
    # pay-spike employees so each linter's recall is measured cleanly).
    trailing = pick(np.setdiff1d(term_ids, spiked), trailing_rows_share)
    rows = last_row[pm.loc[last_row, "employee_id"].isin(trailing).to_numpy()]
    extra = pm.loc[rows].copy()
    extra["month"] += 1
    extra["tenure_months"] += 1
    extra["terminated"] = 0
    extra["termination_type"] = ""
    extra["scheduled_hours"] = 0.0
    log += [(i, "trailing_payroll_rows") for i in trailing]

    messy = (pd.concat([pm, extra], ignore_index=True)
             .sort_values(["employee_id", "month"]).reset_index(drop=True))
    injections = pd.DataFrame(log, columns=["employee_id", "bug"])
    return MessyExtract(messy, injections)
