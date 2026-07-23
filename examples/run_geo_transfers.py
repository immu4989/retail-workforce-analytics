"""Geographic transfer matching (use case 18).

Give stores coordinates, place each employee's home at their commute distance,
then match far-commuting employees to closer stores with vacancies to buy
retention for free. The optimal (Hungarian) assignment is scored against the
greedy and do-nothing baselines, with the payoff — exits avoided and dollars —
exact because the commute-to-hazard coefficient is published ground truth.

Run from the repo root (takes ~30 seconds):

    python examples/run_geo_transfers.py

Outputs land in ``reports/geo_transfers.json`` and ``reports/transfer_plan.csv``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from workforce_analytics import (
    SimulationConfig,
    assign_homes,
    assign_store_coordinates,
    compare_strategies,
    generate,
    optimize_transfers,
    store_vacancies,
    transfer_options,
    workforce_snapshot,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SCORE_MONTH = 48


def main() -> None:
    print("1/3 laying out stores + homes and finding vacancies...")
    result = generate(SimulationConfig())
    coords = assign_store_coordinates(result.stores, seed=71)
    snapshot = workforce_snapshot(result.person_months, month=SCORE_MONTH)
    homes = assign_homes(snapshot, coords, seed=72)
    vacancies = store_vacancies(snapshot, result.stores)
    print(f"      {len(snapshot):,} active hourly employees; "
          f"{int((vacancies > 0).sum())} stores with {int(vacancies.sum())} open hourly roles")

    print("2/3 finding beneficial transfers (closer store, same district, has a vacancy)...")
    options = transfer_options(homes, coords, vacancies)
    print(f"      {options['employee_id'].nunique()} employees have at least one "
          f"commute-reducing transfer available")

    print("3/3 optimal matching vs greedy vs doing nothing...")
    comparison = compare_strategies(options, vacancies)
    for name in ("optimal", "greedy", "no_transfers"):
        s = comparison[name]
        print(f"      {name:13s} {s['transfers']:4d} transfers | "
              f"{s['exits_avoided']:5.1f} exits avoided | "
              f"{s['commute_km_saved']:7.0f} km saved | ${s['dollars_saved']:,.0f}/yr")
    print(f"      optimal buys {comparison['optimal_vs_greedy_extra_exits_avoided']} "
          f"more exits avoided than greedy, same vacancies")

    plan = optimize_transfers(options, vacancies)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "geo_transfers.json").write_text(json.dumps(comparison, indent=2))
    plan.to_csv(REPORTS / "transfer_plan.csv", index=False)
    print(f"\nwrote {REPORTS / 'geo_transfers.json'} and transfer_plan.csv")


if __name__ == "__main__":
    main()
