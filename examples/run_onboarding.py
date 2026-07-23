"""First-90-day onboarding risk and the new-hire watchlist (use case 14).

Simulate each new hire's first-90-day trajectory — training completion at 30
days and the 30/60/90 washout outcome — from a published hazard, then train a
day-30 washout model scored against the oracle ceiling, print the 30/60/90
retention curve, and rank the highest-risk new hires into a watchlist.

Run from the repo root (takes ~30 seconds):

    python examples/run_onboarding.py

Outputs land in ``reports/onboarding.json`` and ``reports/new_hire_watchlist.csv``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from workforce_analytics import (
    OnboardingModel,
    SimulationConfig,
    generate,
    milestone_retention,
    new_hire_watchlist,
    simulate_onboarding,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    print("1/3 simulating new-hire onboarding trajectories...")
    pm = generate(SimulationConfig()).person_months
    cohort = simulate_onboarding(pm, seed=29)
    print(f"      {len(cohort):,} new hires | "
          f"{cohort['washed_out_90d'].mean():.0%} wash out within 90 days | "
          f"{cohort['training_completed_30d'].mean():.0%} finish 30-day training")

    # Hold out 30% of hires to score the model honestly.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(cohort))
    cut = int(0.7 * len(cohort))
    train, test = cohort.iloc[idx[:cut]], cohort.iloc[idx[cut:]]

    print("2/3 day-30 washout model vs the oracle ceiling...")
    model = OnboardingModel().fit(train)
    metrics = model.evaluate(test)
    print(f"      AUC {metrics['model_auc']} vs ceiling {metrics['ceiling_auc']} "
          f"({metrics['pct_of_ceiling']}% of best possible); top quintile washes "
          f"out {metrics['top_quintile_washout_rate']:.0%}")

    milestones = milestone_retention(cohort)
    print(f"      retention 30/60/90: {milestones['day_30_retention']:.0%} / "
          f"{milestones['day_60_retention']:.0%} / {milestones['day_90_retention']:.0%}")
    print(f"      training gap: completers wash out {milestones['washout_if_completed']:.0%} "
          f"vs {milestones['washout_if_incomplete']:.0%} — but part of that "
          f"{milestones['observed_completion_gap']:.0%} gap is onboarding-quality "
          f"confounding, not a training effect")

    print("3/3 building the new-hire watchlist...")
    watchlist = new_hire_watchlist(model, test, top_frac=0.20)
    print(f"      {len(watchlist)} hires flagged; top 5:")
    for _, r in watchlist.head(5).iterrows():
        print(f"        emp {r['employee_id']} @ {r['store_id']}  "
              f"risk {r['washout_risk']:.2f}  [{r['top_reasons']}]")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "onboarding.json").write_text(
        json.dumps({"evaluation": metrics, "milestones": milestones}, indent=2))
    watchlist.to_csv(REPORTS / "new_hire_watchlist.csv", index=False)
    print(f"\nwrote {REPORTS / 'onboarding.json'} and new_hire_watchlist.csv")


if __name__ == "__main__":
    main()
