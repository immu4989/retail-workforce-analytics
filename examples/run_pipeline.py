"""End-to-end pipeline: simulate a workforce, train every model, produce the
hiring plan, driver analysis and figures used in the README.

Run from the repo root (takes ~1 minute):

    python examples/run_pipeline.py

Outputs land in ``reports/`` (CSV + JSON) and ``docs/figures/`` (PNG).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from workforce_analytics import (
    InterventionSimulator,
    SimulationConfig,
    TurnoverModel,
    build_hiring_plan,
    build_snapshots,
    close_hours_gap,
    driver_importance,
    evaluate_with_ceiling,
    generate,
    raise_pay_floor,
    stabilize_schedules,
    time_split,
    validate_expected_attrition,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"

TRAIN_END, VAL_END, SCORE_MONTH = 36, 44, 48

# Chart tokens (light mode) — see docs/figures for the rendered output.
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, GREEN = "#2a78d6", "#008300"


def style_axis(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
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
    print("1/6 simulating workforce (12 districts, 60 months)...")
    result = generate(SimulationConfig())
    pm = result.person_months
    print(f"      {len(result.employees):,} employees, {len(pm):,} person-months")

    print("2/6 building point-in-time snapshots...")
    snaps = build_snapshots(pm, horizons=(3, 6, 12))
    train_h, val, test = time_split(snaps, TRAIN_END, VAL_END, train_stride=3)
    train_s, _, _ = time_split(snaps, TRAIN_END, VAL_END, train_stride=1)

    print("3/6 training turnover models...")
    hourly = TurnoverModel("hourly", horizons=(3, 6, 12)).fit(train_h, val)
    salaried = TurnoverModel("salaried", horizons=(6, 12)).fit(train_s, val)

    metrics_h = evaluate_with_ceiling(hourly, test)
    metrics_s = evaluate_with_ceiling(salaried, test)
    metrics = pd.concat([metrics_h, metrics_s], ignore_index=True)
    metrics.round(4).to_csv(REPORTS / "model_metrics.csv", index=False)
    print(metrics[["population", "horizon_months", "roc_auc", "ceiling_auc",
                   "signal_captured", "ece", "lift_at_k"]].round(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("4/6 headcount plan + validation...")
    month_snap = snaps[snaps["month"] == SCORE_MONTH]
    preds = pd.concat([hourly.predict(month_snap), salaried.predict(month_snap)],
                      ignore_index=True)
    preds6 = preds[preds["p_6m"].notna()]
    plan = build_hiring_plan(preds6, result.stores, horizon=6)
    plan.to_csv(REPORTS / "hiring_plan_6m.csv", index=False)
    hc_check = validate_expected_attrition(preds6, pm, horizon=6)
    hc_check.to_csv(REPORTS / "headcount_validation_6m.csv", index=False)

    # ------------------------------------------------------------------
    print("5/6 driver analysis + interventions...")
    importance = driver_importance(hourly, test, horizon=6, n_repeats=5)
    importance.round(5).to_csv(REPORTS / "drivers_hourly_6m.csv", index=False)

    simulator = InterventionSimulator(hourly, month_snap)
    interventions = pd.concat([
        simulator.run(raise_pay_floor(0.95), "pay floor at 95% of market", horizon=6),
        simulator.run(stabilize_schedules(), "cap schedule volatility at median", horizon=6),
        simulator.run(close_hours_gap(), "schedule people their desired hours", horizon=6),
    ], ignore_index=True)
    interventions.to_csv(REPORTS / "interventions_6m.csv", index=False)
    print(interventions[interventions["district_id"] == "ALL"]
          [["intervention", "expected_exits_before", "exits_avoided", "reduction_pct"]]
          .to_string(index=False))

    summary = {
        "n_employees": int(len(result.employees)),
        "n_person_months": int(len(pm)),
        "annualized_turnover": {
            role: round(float(1 - (1 - pm.loc[pm["role"] == role, "terminated"].mean()) ** 12), 3)
            for role in pm["role"].unique()
        },
        "headcount_total_predicted_exits_6m": float(hc_check["predicted_exits"].sum()),
        "headcount_total_actual_exits_6m": int(hc_check["actual_exits"].sum()),
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------
    print("6/6 rendering figures...")

    # Fig 1: tenure hazard curve, hourly roles.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.set_facecolor(SURFACE)
    for role, color in [("barista", BLUE), ("shift_supervisor", GREEN)]:
        r = pm[(pm["role"] == role) & (pm["tenure_months"] <= 36)]
        curve = r.groupby("tenure_months")["terminated"].mean()
        label = role.replace("_", " ")
        ax.plot(curve.index, curve.to_numpy() * 100, color=color, linewidth=2,
                solid_capstyle="round", label=label)
    style_axis(ax, "Monthly exit rate by tenure — the new-hire washout")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Exit rate per month (%)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "tenure_hazard.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)

    # Fig 2: model AUC vs oracle ceiling per horizon.
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.set_facecolor(SURFACE)
    rows = metrics.reset_index(drop=True)
    labels = [f"{p}\n{h}m" for p, h in zip(rows["population"], rows["horizon_months"])]
    x = np.arange(len(rows))
    ax.bar(x, rows["roc_auc"] - 0.5, bottom=0.5, width=0.55, color=BLUE, zorder=3)
    ax.scatter(x, rows["ceiling_auc"], marker="_", s=520, color=INK,
               linewidth=2, zorder=4)
    for xi, auc in enumerate(rows["roc_auc"]):
        ax.text(xi, auc - 0.008, f"{auc:.2f}", ha="center", va="top",
                color="#ffffff", fontsize=8.5)
    style_axis(ax, "Model ROC-AUC (bars) vs the best any model could do")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 0.72)
    ax.set_ylabel("ROC-AUC")
    gap_idx = int(np.argmax(rows["ceiling_auc"] - rows["roc_auc"]))
    ax.annotate("oracle ceiling (true hazard)",
                xy=(gap_idx, rows["ceiling_auc"].iloc[gap_idx]),
                xytext=(gap_idx, rows["ceiling_auc"].iloc[gap_idx] + 0.016),
                ha="center", color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=1))
    fig.tight_layout()
    fig.savefig(FIGURES / "auc_vs_ceiling.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)

    # Fig 3: driver importance, colored by lever type.
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.set_facecolor(SURFACE)
    colors = [BLUE if lv == "actionable" else MUTED for lv in top["lever"]]
    ax.barh(top["description"], top["auc_drop_mean"], color=colors, height=0.55,
            zorder=3)
    style_axis(ax, "What drives hourly turnover (permutation importance, 6-month model)")
    ax.set_xlabel("AUC drop when feature is shuffled")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["actionable lever", "context (targeting only)"],
              frameon=False, labelcolor=INK2, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "drivers.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: headcount validation — predicted vs actual exits per district.
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=150)
    fig.set_facecolor(SURFACE)
    lim = max(hc_check["predicted_exits"].max(), hc_check["actual_exits"].max()) * 1.15
    ax.plot([0, lim], [0, lim], color=BASELINE, linewidth=1)
    ax.scatter(hc_check["actual_exits"], hc_check["predicted_exits"], s=64,
               color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
    style_axis(ax, "Expected vs realised exits by district (6 months)")
    ax.set_xlabel("Actual exits")
    ax.set_ylabel("Predicted exits (sum of probabilities)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(FIGURES / "headcount_validation.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)

    print(f"done. reports -> {REPORTS}  figures -> {FIGURES}")


if __name__ == "__main__":
    main()
