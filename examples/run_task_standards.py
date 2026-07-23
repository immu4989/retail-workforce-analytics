"""Task-level labor standards from order mix (use case 16).

Split store traffic into order channels (front counter, drive-thru, mobile
pickup, delivery), build the labor requirement up from per-channel task times,
and compare it to the flat 18-transactions-per-hour rate use case 5 assumes.
The flat rate agrees on average but under-provisions the highest-mobile hours;
the task times themselves are recovered from labor and order data.

Run from the repo root (takes ~40 seconds):

    python examples/run_task_standards.py

Outputs land in ``reports/task_standards.json``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from workforce_analytics import (
    SimulationConfig,
    TrafficConfig,
    TrafficSimulator,
    generate,
    mix_staffing_summary,
    recover_task_seconds,
    simulate_order_mix,
    staffing_comparison,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    print("1/3 simulating store traffic + order mix...")
    result = generate(SimulationConfig())
    stores = result.stores[result.stores["open_month"] == 0].reset_index(drop=True)
    traffic = TrafficSimulator(stores, TrafficConfig(n_weeks=52)).run()
    order_mix = simulate_order_mix(traffic, seed=53)
    print(f"      {len(order_mix):,} store-hours split across "
          f"{len(('front_counter', 'drive_thru', 'mobile_pickup', 'delivery'))} channels; "
          f"mean mobile share {order_mix['mobile_share'].mean():.0%}")

    print("2/3 flat 18/hr rate vs task-time buildup...")
    comparison = staffing_comparison(order_mix)
    summary = mix_staffing_summary(comparison)
    print(f"      average labor hours: flat {summary['avg_flat_labor_hours']} vs "
          f"task {summary['avg_task_labor_hours']} (ratio "
          f"{summary['avg_ratio_task_to_flat']}) — they agree on average")
    print(f"      but the flat rate UNDER-provisions the top mobile-share decile by "
          f"{summary['top_mobile_decile_flat_under_provision_pct']}% and "
          f"OVER-provisions the bottom by "
          f"{abs(summary['bottom_mobile_decile_flat_over_provision_pct'])}%")
    print(f"      staffing error tracks mobile share (corr "
          f"{summary['corr_mobile_share_vs_gap']})")

    print("3/3 recovering the per-channel task times (oracle)...")
    recovery = recover_task_seconds(order_mix)
    for _, r in recovery.iterrows():
        print(f"      {r['channel']:14s} true {r['true_seconds']:.0f}s  "
              f"recovered {r['recovered_seconds']:.0f}s  ({r['pct_error']:+.1f}%)")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "task_standards.json").write_text(json.dumps({
        "staffing_summary": summary,
        "task_time_recovery": recovery.to_dict(orient="records"),
    }, indent=2))
    print(f"\nwrote {REPORTS / 'task_standards.json'}")


if __name__ == "__main__":
    main()
