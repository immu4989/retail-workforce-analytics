"""Audit an HRIS extract before modelling: the linters from
``docs/adapting_to_real_data.md`` run against planted bugs.

Clean simulator output passes the audit. Then the documented HRIS mistakes
(backfilled ratings, severance stamped on termination rows, in-notice
employees left in the data, trailing payroll rows) are injected with a
per-employee log, and the audit has to find them — plus a measurement of the
damage each bug does if it slips through.

Run from the repo root (takes ~30 seconds):

    python examples/run_real_data_audit.py

Outputs land in ``reports/real_data_audit.json``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from sklearn.metrics import roc_auc_score

from workforce_analytics import (
    SimulationConfig,
    audit_split,
    build_snapshots,
    generate,
    make_messy_extract,
    time_split,
    validate_person_months,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def hours_auc(person_months) -> float:
    """Single-feature AUC of scheduled_hours for 3-month exit, hourly roles."""
    snaps = build_snapshots(person_months, horizons=(3,))
    h = snaps[snaps["role"].isin(["barista", "shift_supervisor"])
              & snaps["label_3m"].notna()]
    return float(roc_auc_score(h["label_3m"], -h["scheduled_hours"]))


def mean_final_pay(person_months) -> float:
    term = person_months[person_months["terminated"] == 1]
    return float(term.groupby("employee_id")["pay_rate"].last().mean())


def main() -> None:
    sim = generate(SimulationConfig(n_districts=6, n_months=48, seed=11))
    pm = sim.person_months

    print("== 1. Clean simulator output ==")
    clean_report = validate_person_months(pm)
    print(clean_report.summary())

    print("\n== 2. Same data with the documented HRIS bugs planted ==")
    messy = make_messy_extract(pm, seed=3)
    counts = messy.injections["bug"].value_counts()
    for bug, n in counts.items():
        print(f"   planted {bug}: {n} employees")

    print("\n== 3. What the audit finds ==")
    messy_report = validate_person_months(messy.person_months)
    print(messy_report.summary())

    print("\n== 4. The damage if the bugs slip through ==")
    auc_clean, auc_messy = hours_auc(pm), hours_auc(messy.person_months)
    print(f"   scheduled_hours alone, AUC for 3-month exit: "
          f"{auc_clean:.3f} clean -> {auc_messy:.3f} with in-notice leakage")
    pay_clean, pay_messy = mean_final_pay(pm), mean_final_pay(messy.person_months)
    print(f"   mean final-month pay of leavers: ${pay_clean:.2f} clean -> "
          f"${pay_messy:.2f} with termination-row adjustments "
          f"(+{pay_messy / pay_clean - 1:.0%})")

    print("\n== 5. The random-split mistake ==")
    snaps = build_snapshots(pm, horizons=(3,))
    train, _, test = time_split(snaps, train_end=28, val_end=34)
    print(f"   time_split: {audit_split(train, test) or 'clean'}")
    shuffled = snaps.sample(frac=1.0, random_state=0)
    half = len(shuffled) // 2
    for f in audit_split(shuffled.iloc[:half], shuffled.iloc[half:]):
        print(f"   random split: {f}")

    REPORTS.mkdir(exist_ok=True)
    out = {
        "clean_findings": [f.code for f in clean_report.findings],
        "planted": {bug: int(n) for bug, n in counts.items()},
        "messy_findings": {f.code: f.n_affected for f in messy_report.findings},
        "hours_auc_clean": round(auc_clean, 3),
        "hours_auc_messy": round(auc_messy, 3),
        "mean_final_pay_clean": round(pay_clean, 2),
        "mean_final_pay_messy": round(pay_messy, 2),
    }
    with open(REPORTS / "real_data_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {REPORTS / 'real_data_audit.json'}")


if __name__ == "__main__":
    main()
