"""Payroll-anomaly detection on a punch-clock layer (use case 13).

Simulate scheduled-vs-punched hours with three planted anomalies — time
padding, ghost shifts, buddy punching — then run a robust residual monitor for
each and score it against the planted ground truth. The point is a
loss-prevention tool whose precision and recall are known before it touches
real payroll, plus the dollar leakage it recovers.

Run from the repo root (takes ~40 seconds):

    python examples/run_labor_anomaly.py

Outputs land in ``reports/labor_anomaly.json``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from workforce_analytics import (
    SimulationConfig,
    detect_buddy_punching,
    detect_ghost_shifts,
    detect_time_padding,
    evaluate_detection,
    generate,
    labor_leakage,
    simulate_punches,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    print("1/3 simulating workforce + punch clock...")
    pm = generate(SimulationConfig()).person_months
    panel = simulate_punches(pm, seed=0)
    counts = panel.injections["kind"].value_counts()
    print(f"      {len(panel.punches):,} hourly employee-months | planted "
          f"{counts.get('time_padding', 0)} padders, "
          f"{counts.get('ghost_shift', 0)} ghost store-months, "
          f"{counts.get('buddy_punch', 0)} buddy punches")

    print("2/3 running the three robust residual monitors...")
    detections = {
        "time_padding": evaluate_detection(detect_time_padding(panel), "is_padder"),
        "ghost_shifts": evaluate_detection(detect_ghost_shifts(panel), "is_ghost_month"),
        "buddy_punching": evaluate_detection(
            detect_buddy_punching(panel), "is_buddy", "within_emp_z"),
    }
    for name, m in detections.items():
        print(f"      {name:15s} precision {m['precision']:.2f}  recall "
              f"{m['recall']:.2f}  AP {m['average_precision']:.2f}  "
              f"({m['flagged']} flagged / {m['planted']} planted)")

    print("3/3 pricing the leakage...")
    leak = labor_leakage(panel)
    print(f"      annual payroll leakage ${leak['annual_leakage_total']:,.0f} "
          f"(padding ${leak['annual_leakage_time_padding']:,.0f}, "
          f"ghost ${leak['annual_leakage_ghost_shifts']:,.0f}, "
          f"buddy ${leak['annual_leakage_buddy_punching']:,.0f})")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "labor_anomaly.json").write_text(
        json.dumps({"detection": detections, "leakage": leak}, indent=2))
    print(f"\nwrote {REPORTS / 'labor_anomaly.json'}")


if __name__ == "__main__":
    main()
