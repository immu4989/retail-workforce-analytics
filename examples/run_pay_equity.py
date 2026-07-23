"""Pay-equity audit, and validating the audit itself (use case 17).

Assign a synthetic, abstract group label, plant a known residual pay gap, and
run the audit: recover the gap, show that controlling for a confounder is what
makes the estimate right, and measure the audit's power and false-positive rate
by replaying it. The group is a synthetic construct for validating the METHOD;
nothing here uses a real person's demographics.

Run from the repo root (takes ~40 seconds):

    python examples/run_pay_equity.py

Outputs land in ``reports/pay_equity.json``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from workforce_analytics import (
    PayEquityGroundTruth,
    SimulationConfig,
    assign_group_and_gap,
    audit_pay_gap,
    audit_power_fpr,
    build_employee_frame,
    generate,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    print("1/3 building the employee frame (synthetic groups, planted gap)...")
    result = generate(SimulationConfig())
    employees = build_employee_frame(result.person_months, result.stores)
    print(f"      {len(employees):,} hourly employees with role/tenure/market/performance controls")

    print("2/3 the audit recovers a planted 4% gap — and controls fix confounding...")
    clean = audit_pay_gap(assign_group_and_gap(
        employees, PayEquityGroundTruth(true_residual_gap=0.04), seed=61))
    confounded = audit_pay_gap(assign_group_and_gap(
        employees, PayEquityGroundTruth(true_residual_gap=0.04, confounding_strength=1.0),
        seed=61))
    print(f"      no confounding : unadjusted {clean['unadjusted_gap']:.1%}  "
          f"adjusted {clean['adjusted_gap']:.1%}  (planted 4.0%)")
    print(f"      group tied to tenure : unadjusted {confounded['unadjusted_gap']:.1%} "
          f"(misleads)  adjusted {confounded['adjusted_gap']:.1%}  (planted 4.0%)")

    print("3/3 the audit's power and false-positive rate, by replay...")
    power = audit_power_fpr(
        employees, PayEquityGroundTruth(true_residual_gap=0.02), n_reps=200, seed=0)
    print(f"      at a 2% planted gap: power {power['power']:.0%} to detect it, "
          f"false-positive rate {power['false_positive_rate']:.0%} at zero gap, "
          f"estimate bias {power['gap_estimate_bias']:+.4f}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "pay_equity.json").write_text(json.dumps({
        "recovery_no_confounding": clean,
        "recovery_confounded": confounded,
        "power_fpr": power,
    }, indent=2))
    print(f"\nwrote {REPORTS / 'pay_equity.json'}")


if __name__ == "__main__":
    main()
