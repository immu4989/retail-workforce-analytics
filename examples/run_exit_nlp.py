"""Exit-interview NLP: recover why people leave, from their own words (use case 15).

Generate synthetic exit comments whose hidden theme is tied to the true hazard
driver behind each leaver's exit, recover the themes unsupervised (TF-IDF +
NMF), and check that each recovered theme lines up with its driver — pay-theme
leavers are underpaid, schedule-theme leavers have volatile schedules.

Run from the repo root (takes ~30 seconds):

    python examples/run_exit_nlp.py

Outputs land in ``reports/exit_nlp.json`` and ``reports/exit_theme_terms.csv``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from workforce_analytics import (
    ExitThemeModel,
    SimulationConfig,
    generate,
    simulate_exit_comments,
    theme_by_regrettability,
    theme_driver_alignment,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    print("1/3 simulating exit comments tied to true exit drivers...")
    pm = generate(SimulationConfig()).person_months
    comments = simulate_exit_comments(pm, seed=47)
    print(f"      {len(comments):,} hourly exit comments across {len(comments['true_theme'].unique())} themes")

    print("2/3 recovering themes unsupervised (TF-IDF + NMF)...")
    model = ExitThemeModel().fit(comments)
    metrics = model.evaluate(comments)
    print(f"      NMI {metrics['nmi']}  purity {metrics['purity']} "
          f"vs the planted themes")
    cluster_to_theme = metrics["cluster_to_theme"]
    for _, r in model.top_terms(6).iterrows():
        theme = cluster_to_theme[r["cluster"]]
        print(f"        cluster {r['cluster']} -> {theme:11s} | {r['top_terms']}")

    print("3/3 checking themes line up with true drivers...")
    align = theme_driver_alignment(comments)
    for _, r in align.iterrows():
        mark = "ok" if r["aligned"] else "MISALIGNED"
        print(f"      {r['theme']:11s} <- {r['driver']:24s} gap {r['aligned_gap']:+.3f}  [{mark}]")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "exit_nlp.json").write_text(json.dumps({
        "recovery": metrics,
        "alignment": align.to_dict(orient="records"),
        "regrettability": theme_by_regrettability(comments).to_dict(orient="records"),
    }, indent=2, default=str))
    model.top_terms(10).to_csv(REPORTS / "exit_theme_terms.csv", index=False)
    print(f"\nwrote {REPORTS / 'exit_nlp.json'} and exit_theme_terms.csv")


if __name__ == "__main__":
    main()
