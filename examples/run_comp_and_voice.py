"""Compensation experiments and call-center topic mining (use cases 10-11).

Run from the repo root. The wage experiments rerun the full simulator in
paired arms across seeds, so this one takes a few minutes:

    python examples/run_comp_and_voice.py

Outputs land in ``reports/`` and ``docs/figures/``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from workforce_analytics import (
    CallTopicModel,
    SimulationConfig,
    WageProgram,
    event_study,
    generate,
    operational_linkage,
    pay_elasticity,
    run_wage_experiment,
    simulate_calls,
    topic_trends,
)
from workforce_analytics.callcenter import TOPICS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"

SEEDS = (0, 1, 2)
PROGRAM_MONTH = 36

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, GREEN, RED = "#2a78d6", "#008300", "#e34948"


def style_axis(ax, title: str = ""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print("1/5 observational pay elasticity (recovery check)...")
    base = generate(SimulationConfig())
    elasticity = pay_elasticity(base.person_months)
    elasticity.to_csv(REPORTS / "pay_elasticity.csv", index=False)
    print(elasticity.to_string(index=False), "\n      (true coefficient: -2.5)")

    # ------------------------------------------------------------------
    print("2/5 paired wage-program experiments (this is the slow part)...")
    programs = [
        WageProgram("blanket +5% market adjustment", pct=0.05,
                    start_month=PROGRAM_MONTH),
        WageProgram("ongoing 95% pay floor (targeted)", kind="floor",
                    floor_ratio=0.95, start_month=PROGRAM_MONTH),
        WageProgram("seasonal +3% every July", pct=0.03,
                    every_month_of_year=7, start_month=30),
        WageProgram("12-month merit freeze", kind="freeze",
                    start_month=PROGRAM_MONTH, end_month=PROGRAM_MONTH + 11),
    ]
    cache: dict = {}
    experiments = []
    for prog in programs:
        exp = run_wage_experiment(prog, seeds=SEEDS,
                                  eval_start_month=PROGRAM_MONTH,
                                  baseline_cache=cache)
        exp.pop("per_seed")
        experiments.append(exp)
        print(f"      {exp['program']}: exits avoided {exp['exits_avoided_mean']}, "
              f"net ${exp['net_value_mean']:,.0f} CI90 {exp['net_value_ci90']}")
    (REPORTS / "wage_experiments.json").write_text(json.dumps(experiments, indent=2))

    print("3/5 event study for the blanket adjustment...")
    study = event_study(programs[0], seeds=SEEDS)
    study.to_csv(REPORTS / "wage_event_study.csv", index=False)

    # ------------------------------------------------------------------
    print("4/5 call-center simulation + topic model...")
    calls = simulate_calls(base.person_months)
    tm = CallTopicModel().fit(calls)
    tm_eval = tm.evaluate(calls)
    (REPORTS / "callcenter_topics.json").write_text(json.dumps(tm_eval, indent=2))
    tm.top_terms().to_csv(REPORTS / "callcenter_top_terms.csv", index=False)
    linkage = operational_linkage(calls, base.person_months)
    linkage.to_csv(REPORTS / "callcenter_linkage.csv", index=False)
    print(f"      {len(calls):,} calls | NMI {tm_eval['nmi']} | purity {tm_eval['purity']}")
    print(linkage.to_string(index=False))

    # ------------------------------------------------------------------
    print("5/5 figures...")

    # Fig A: event study — identical worlds until the program lands.
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
    fig.set_facecolor(SURFACE)
    for arm, color in [("baseline", MUTED), ("program", BLUE)]:
        g = study[study["arm"] == arm]
        ax.plot(g["month"], g["exit_rate"] * 100, color=color, linewidth=2,
                label=arm if arm == "baseline" else "+5% adjustment")
    ax.axvline(PROGRAM_MONTH, color=RED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(PROGRAM_MONTH + 0.7, ax.get_ylim()[1] * 0.97, "program starts",
            color=INK2, fontsize=9, va="top")
    style_axis(ax, "Same simulated world, with and without a +5% raise (3 paired seeds)")
    ax.set_xlabel("Simulation month")
    ax.set_ylabel("Hourly monthly exit rate (%)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "wage_event_study.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig B: program net value with CI whiskers.
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=150)
    fig.set_facecolor(SURFACE)
    names = [e["program"].replace(" (targeted)", "\n(targeted)") for e in experiments]
    vals = np.array([e["net_value_mean"] for e in experiments]) / 1e6
    lows = np.array([e["net_value_ci90"][0] for e in experiments]) / 1e6
    highs = np.array([e["net_value_ci90"][1] for e in experiments]) / 1e6
    colors = [BLUE if v >= 0 else RED for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.5, color=colors, zorder=3)
    ax.errorbar(vals, y, xerr=[vals - lows, highs - vals], fmt="none",
                ecolor=INK, elinewidth=1.4, capsize=3, zorder=4)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, names, fontsize=9)
    style_axis(ax, "Net value of wage programs, replacement costs only (90% CI)")
    ax.set_xlabel("Million dollars over the 24-month window")
    fig.tight_layout()
    fig.savefig(FIGURES / "wage_programs_roi.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig C: topic mix by calendar month — small multiples, one panel per topic.
    trends = topic_trends(calls)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), dpi=150, sharex=True)
    fig.set_facecolor(SURFACE)
    for ax, topic in zip(axes.flat, TOPICS):
        g = trends[trends["true_topic"] == topic]
        ax.plot(g["month_of_year"], g["calls_per_month"], color=BLUE, linewidth=2)
        style_axis(ax)
        ax.set_title(topic.replace("_", " "), color=INK2, fontsize=10, loc="left")
        ax.set_xticks([1, 4, 7, 10])
    fig.suptitle("Employee support calls per month, by topic and calendar month",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.supxlabel("Calendar month", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "callcenter_topics.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig D: operational linkage — scheduling calls track true volatility.
    pm_h = base.person_months[base.person_months["role"].isin(
        ["barista", "shift_supervisor"])]
    store_state = pm_h.groupby("store_id", observed=True).agg(
        n=("employee_id", "size"), vol=("schedule_volatility", "mean"))
    sched_rate = (calls[calls["true_topic"] == "scheduling"]
                  .groupby("store_id", observed=True).size()
                  .reindex(store_state.index).fillna(0) / store_state["n"] * 1000)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.scatter(store_state["vol"], sched_rate, s=36, color=BLUE,
               edgecolors=SURFACE, linewidths=1.5, zorder=3)
    style_axis(ax, "Stores that call about scheduling have chaotic schedules")
    ax.set_xlabel("Store mean schedule volatility (hours std-dev)")
    ax.set_ylabel("Scheduling calls per 1,000 employee-months")
    fig.tight_layout()
    fig.savefig(FIGURES / "callcenter_linkage.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)

    print(f"done. reports -> {REPORTS}  figures -> {FIGURES}")


if __name__ == "__main__":
    main()
