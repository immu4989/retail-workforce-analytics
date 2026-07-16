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
    CostModel,
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
    reason_codes,
    shap_importance,
    shap_matrix,
    stabilize_schedules,
    targeting_roi,
    time_split,
    turnover_cost_summary,
    validate_expected_attrition,
)
from workforce_analytics.config import HOURLY_ROLES
from workforce_analytics.evaluation import calibration_table, evaluate_horizon

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
    print("1/10 simulating workforce (12 districts, 60 months)...")
    result = generate(SimulationConfig())
    pm = result.person_months
    print(f"      {len(result.employees):,} employees, {len(pm):,} person-months")

    print("2/10 building point-in-time snapshots...")
    snaps = build_snapshots(pm, horizons=(3, 6, 12))
    train_h, val, test = time_split(snaps, TRAIN_END, VAL_END, train_stride=3)
    train_s, _, _ = time_split(snaps, TRAIN_END, VAL_END, train_stride=1)

    print("3/10 training turnover models...")
    hourly = TurnoverModel("hourly", horizons=(3, 6, 12)).fit(train_h, val)
    salaried = TurnoverModel("salaried", horizons=(6, 12)).fit(train_s, val)

    metrics_h = evaluate_with_ceiling(hourly, test)
    metrics_s = evaluate_with_ceiling(salaried, test)
    metrics = pd.concat([metrics_h, metrics_s], ignore_index=True)
    metrics.round(4).to_csv(REPORTS / "model_metrics.csv", index=False)
    print(metrics[["population", "horizon_months", "roc_auc", "ceiling_auc",
                   "signal_captured", "ece", "lift_at_k"]].round(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("4/10 headcount plan + validation...")
    month_snap = snaps[snaps["month"] == SCORE_MONTH]
    preds = pd.concat([hourly.predict(month_snap), salaried.predict(month_snap)],
                      ignore_index=True)
    preds6 = preds[preds["p_6m"].notna()]
    plan = build_hiring_plan(preds6, result.stores, horizon=6)
    plan.to_csv(REPORTS / "hiring_plan_6m.csv", index=False)
    hc_check = validate_expected_attrition(preds6, pm, horizon=6)
    hc_check.to_csv(REPORTS / "headcount_validation_6m.csv", index=False)

    # ------------------------------------------------------------------
    print("5/10 driver analysis + interventions...")
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
    print("6/10 rendering core figures...")

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

    # ------------------------------------------------------------------
    print("7/10 cost model: baseline burn, intervention value, targeting ROI...")
    cm = CostModel()
    burn = turnover_cost_summary(pm, cm)
    burn.to_csv(REPORTS / "turnover_cost_baseline.csv", index=False)
    print(burn.to_string(index=False))

    interventions_usd = pd.concat([
        simulator.run(raise_pay_floor(0.95), "pay floor at 95% of market",
                      horizon=6, cost_model=cm),
        simulator.run(stabilize_schedules(), "cap schedule volatility at median",
                      horizon=6, cost_model=cm),
        simulator.run(close_hours_gap(), "schedule people their desired hours",
                      horizon=6, cost_model=cm),
    ], ignore_index=True)
    interventions_usd.to_csv(REPORTS / "interventions_6m.csv", index=False)

    test6 = test[test["role"].isin(HOURLY_ROLES) & test["label_6m"].notna()]
    test6 = test6.reset_index(drop=True)
    roi = targeting_roi(hourly.predict(test6), test6["label_6m"], horizon=6,
                        cost_model=cm)
    (REPORTS / "targeting_roi_6m.json").write_text(json.dumps(roi, indent=2))
    print(f"      targeted retention ROI {roi['roi_multiple_targeted']}x vs "
          f"{roi['roi_multiple_untargeted']}x untargeted")

    # ------------------------------------------------------------------
    print("8/10 SHAP explanations + reason codes...")
    shap_imp = shap_importance(hourly, test, horizon=6, max_rows=4000)
    shap_imp.round(4).to_csv(REPORTS / "shap_importance_hourly_6m.csv", index=False)
    reasons = reason_codes(hourly, month_snap, horizon=6, top_n=3, max_rows=4000)
    reasons.head(200).to_csv(REPORTS / "reason_codes_sample.csv", index=False)
    print(reasons.head(3)[["role", "p_6m", "reason_1", "reason_2"]].to_string(index=False))

    # ------------------------------------------------------------------
    print("9/10 deep survival model (optional, needs torch)...")
    nn_metrics = None
    try:
        from workforce_analytics import SurvivalNN

        snaps1 = build_snapshots(pm, horizons=(1,))
        tr1, va1, _ = time_split(snaps1, TRAIN_END, VAL_END, train_stride=1)
        nn_model = SurvivalNN(roles=list(HOURLY_ROLES), epochs=8).fit(
            pd.concat([tr1, va1]))
        rows_h = test[test["role"].isin(HOURLY_ROLES)].reset_index(drop=True)
        nn_preds = nn_model.predict(rows_h, horizons=(3, 6, 12))
        nn_rows = []
        for h in (3, 6, 12):
            mask = rows_h[f"label_{h}m"].notna().to_numpy()
            m = evaluate_horizon(rows_h.loc[mask, f"label_{h}m"],
                                 nn_preds.loc[mask, f"p_{h}m"])
            gbm = metrics_h[metrics_h["horizon_months"] == h].iloc[0]
            nn_rows.append({"horizon_months": h, "nn_auc": round(m["roc_auc"], 3),
                            "gbm_auc": round(gbm["roc_auc"], 3),
                            "nn_ece": round(m["ece"], 3),
                            "gbm_ece": round(gbm["ece"], 3),
                            "nn_lift_at_k": round(m["lift_at_k"], 2),
                            "gbm_lift_at_k": round(gbm["lift_at_k"], 2)})
        nn_metrics = pd.DataFrame(nn_rows)
        nn_metrics.to_csv(REPORTS / "nn_vs_gbm_hourly.csv", index=False)
        print(nn_metrics.to_string(index=False))
    except ImportError:
        print("      torch not installed; skipping (pip install -e '.[deep]')")

    # ------------------------------------------------------------------
    print("10/10 rendering business figures...")

    # Fig 5: SHAP beeswarm, top 8 features, hand-rolled in the house style.
    from matplotlib.colors import LinearSegmentedColormap
    sv, sv_rows = shap_matrix(hourly, test, horizon=6, max_rows=1500)
    order = sv.abs().mean().sort_values(ascending=False).head(8).index.tolist()
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    fig.set_facecolor(SURFACE)
    for i, feat in enumerate(reversed(order)):
        vals = sv[feat].to_numpy()
        raw = pd.to_numeric(sv_rows[feat], errors="coerce").to_numpy(dtype=float)
        pct = pd.Series(raw).rank(pct=True).fillna(0.5).to_numpy()
        ax.scatter(vals, np.full_like(vals, i) + rng.uniform(-0.28, 0.28, len(vals)),
                   c=cmap(pct), s=9, alpha=0.75, linewidths=0, zorder=3)
    style_axis(ax, "SHAP values, 6-month hourly model (red = high feature value)")
    ax.set_yticks(range(len(order)),
                  [f.replace("_", " ") for f in reversed(order)])
    ax.tick_params(axis="y", labelsize=9.5)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=2)
    ax.set_xlabel("Impact on predicted log-odds of leaving")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_beeswarm.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig 6: what the 6-month barista hiring plan is made of, by district.
    b = plan[plan["role"] == "barista"].sort_values("hires_needed")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.set_facecolor(SURFACE)
    left = np.zeros(len(b))
    parts = [("expected_attrition", BLUE, "expected attrition"),
             ("current_gap", GREEN, "open positions today"),
             ("growth_positions", "#e87ba4", "new-store growth")]
    parts = [p for p in parts if b[p[0]].sum() > 0]
    for col, color, label in parts:
        ax.barh(b["district_id"], b[col], left=left, color=color, height=0.6,
                label=label, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        left += b[col].to_numpy(dtype=float)
    for y, total in enumerate(b["hires_needed"]):
        ax.text(left[y] + 0.8, y, str(int(total)), va="center", color=INK2,
                fontsize=9)
    style_axis(ax, "Baristas to hire in the next 6 months, by district")
    ax.set_xlabel("Positions")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "hiring_plan.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig 7: dollar value of schedule/pay interventions (hourly, 6 months).
    tot = interventions_usd[interventions_usd["district_id"] == "ALL"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    fig.set_facecolor(SURFACE)
    names = tot["intervention"].map({
        "pay floor at 95% of market": "pay floor at\n95% of market",
        "cap schedule volatility at median": "cap schedule\nvolatility at median",
        "schedule people their desired hours": "schedule people\ntheir desired hours",
    }).fillna(tot["intervention"])
    ax.barh(names, tot["dollars_saved"] / 1000, color=BLUE, height=0.5, zorder=3)
    for y, v in enumerate(tot["dollars_saved"] / 1000):
        ax.text(v + 3, y, f"${v:,.0f}k", va="center", color=INK2, fontsize=9.5)
    n_hourly = int(tot["n_employees"].iloc[0])
    style_axis(ax, f"Replacement costs avoided per 6 months ({n_hourly:,} hourly employees)")
    ax.set_xlabel("Thousand dollars")
    ax.set_xlim(0, tot["dollars_saved"].max() / 1000 * 1.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "intervention_value.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)

    # Fig 8: calibration, hourly 6m.
    rows_h6 = test[test["role"].isin(HOURLY_ROLES) & test["label_6m"].notna()]
    p6 = hourly.predict(rows_h6)["p_6m"]
    cal = calibration_table(rows_h6["label_6m"].to_numpy(), p6.to_numpy(), n_bins=10)
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=150)
    fig.set_facecolor(SURFACE)
    lim = max(cal["mean_predicted"].max(), cal["observed_rate"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=BASELINE, linewidth=1)
    ax.plot(cal["mean_predicted"], cal["observed_rate"], color=BLUE, linewidth=2,
            marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    style_axis(ax, "Predicted vs observed 6-month exit rate (deciles)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed exit rate")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig 9: survival curves for three risk archetypes (needs torch).
    if nn_metrics is not None:
        month_h = month_snap[month_snap["role"].isin(HOURLY_ROLES)]
        curves = nn_model.survival_curves(month_h, horizon=12)
        p12 = 1 - curves["S_12m"]
        idx = {q: (p12 - p12.quantile(q)).abs().idxmin() for q in (0.1, 0.5, 0.9)}
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        fig.set_facecolor(SURFACE)
        months = np.arange(0, 13)
        for (q, i), color, label in zip(
                idx.items(), [GREEN, BLUE, "#e34948"],
                ["low risk (p10)", "typical (median)", "high risk (p90)"]):
            s = [1.0] + [curves.loc[i, f"S_{k}m"] for k in range(1, 13)]
            ax.plot(months, s, color=color, linewidth=2, solid_capstyle="round")
            ax.text(12.15, s[-1], label, color=INK2, fontsize=9, va="center")
        style_axis(ax, "12-month retention curves from the deep survival model")
        ax.set_xlabel("Months from today")
        ax.set_ylabel("Probability still employed")
        ax.set_xlim(0, 15.5)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(FIGURES / "survival_curves.png", facecolor=SURFACE,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"done. reports -> {REPORTS}  figures -> {FIGURES}")


if __name__ == "__main__":
    main()
