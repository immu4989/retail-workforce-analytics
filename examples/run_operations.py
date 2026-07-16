"""Operations pipeline: demand & scheduling, call-outs, hiring funnel,
internal mobility, and turnover contagion (use cases 5-9).

Run from the repo root after (or independently of) run_pipeline.py:

    python examples/run_operations.py

Outputs land in ``reports/`` and ``docs/figures/``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from workforce_analytics import (
    CalloutModel,
    LaborDemandForecaster,
    PromotionModel,
    SimulationConfig,
    TrafficConfig,
    TrafficSimulator,
    TurnoverModel,
    bench_strength,
    build_callout_panel,
    build_promotion_panel,
    build_snapshots,
    build_week_schedule,
    contagion_analysis,
    funnel_report,
    generate,
    req_timing,
    required_staff,
    reserve_staffing_plan,
    schedule_stability,
    simulate_absences,
    simulate_funnel,
    time_split,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"

TRAIN_END, VAL_END, SCORE_MONTH = 36, 44, 48
LOADED_WAGE = 21.0        # $/hour, fully loaded
UNDERSTAFFED_COST = 35.0  # $/person-hour short: lost transactions + service decay

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, GREEN, RED = "#2a78d6", "#008300", "#e34948"


def style_axis(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
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

    print("1/6 simulating workforce + hourly store traffic...")
    result = generate(SimulationConfig())
    pm = result.person_months
    stores = result.stores[result.stores["open_month"] == 0].reset_index(drop=True)
    traffic = TrafficSimulator(stores, TrafficConfig(n_weeks=104)).run()
    print(f"      {len(traffic):,} store-hours of traffic across {len(stores)} stores")

    # ------------------------------------------------------------------
    print("2/6 demand forecast + schedule build...")
    fc = LaborDemandForecaster().fit(traffic, train_end_week=90)
    test_weeks = list(range(92, 104))
    preds = fc.predict(traffic, weeks=test_weeks)
    fc_metrics = fc.evaluate(preds)
    fc_metrics.to_csv(REPORTS / "demand_forecast_metrics.csv", index=False)
    print(fc_metrics.to_string(index=False))

    # Schedule two consecutive weeks for one store, with fair-workweek stickiness.
    sid = stores["store_id"].iloc[0]
    roster_n = int(stores["target_baristas"].iloc[0]
                   + stores["target_shift_supervisors"].iloc[0]) + 8
    rng = np.random.default_rng(3)
    roster = pd.DataFrame({"employee_id": range(roster_n),
                           "desired_hours": rng.choice([20, 24, 28, 32, 38], roster_n)})
    wk_a = build_week_schedule(preds[(preds.store_id == sid) & (preds.week == 100)], roster)
    wk_b = build_week_schedule(preds[(preds.store_id == sid) & (preds.week == 101)], roster,
                               previous_shifts=wk_a.shifts)
    stability = schedule_stability(wk_b.shifts, wk_a.shifts)

    # Dollar comparison: forecast-driven vs "every day is an average day"
    # staffing, scored symmetrically against realised demand. The scheduler's
    # own overhead (shift granularity) applies equally to either plan, so it
    # stays out of this comparison and is reported separately above.
    over_f = under_f = over_flat = under_flat = 0
    for store in stores["store_id"]:
        sp = preds[preds["store_id"] == store]
        recent = traffic[(traffic["store_id"] == store)
                         & traffic["week"].between(79, 90)]
        flat = required_staff(recent.groupby("hour")["transactions"].mean()
                              .sort_index().to_numpy())
        for wk in test_weeks:
            g = sp[sp["week"] == wk]
            if len(g) == 0 or g["forecast"].isna().any():
                continue
            for _, gg in g.groupby("dow"):
                gg = gg.sort_values("hour")
                act_req = required_staff(gg["transactions"].to_numpy())
                plan_fc = required_staff(gg["forecast"].to_numpy())
                over_f += int(np.maximum(plan_fc - act_req, 0).sum())
                under_f += int(np.maximum(act_req - plan_fc, 0).sum())
                over_flat += int(np.maximum(flat - act_req, 0).sum())
                under_flat += int(np.maximum(act_req - flat, 0).sum())
    n_store_weeks = len(stores) * len(test_weeks)
    cost_fc = (over_f * LOADED_WAGE + under_f * UNDERSTAFFED_COST) / n_store_weeks
    cost_flat = (over_flat * LOADED_WAGE + under_flat * UNDERSTAFFED_COST) / n_store_weeks
    sched_summary = {
        "example_store_week": wk_b.summary,
        "schedule_stability_with_stickiness": stability,
        "staffing_cost_per_store_week_forecast": round(cost_fc, 0),
        "staffing_cost_per_store_week_flat": round(cost_flat, 0),
        "savings_per_store_week": round(cost_flat - cost_fc, 0),
        "chain_annual_savings": round((cost_flat - cost_fc) * len(stores) * 52, 0),
        "assumptions": {"loaded_wage": LOADED_WAGE,
                        "understaffed_cost_per_hour": UNDERSTAFFED_COST},
    }
    (REPORTS / "schedule_summary.json").write_text(json.dumps(sched_summary, indent=2))
    print(f"      stability {stability} | savings/store-week "
          f"${sched_summary['savings_per_store_week']:,.0f} | chain/yr "
          f"${sched_summary['chain_annual_savings']:,.0f}")

    # ------------------------------------------------------------------
    print("3/6 call-out prediction + reserve staffing...")
    absences = simulate_absences(pm)
    snaps = build_snapshots(pm, horizons=(6, 12))
    panel = build_callout_panel(snaps, absences)
    tr, va, te = time_split(panel, TRAIN_END, VAL_END, train_stride=2)
    callout = CalloutModel().fit(pd.concat([tr, va]))
    co_metrics = callout.evaluate(te)
    (REPORTS / "callout_metrics.json").write_text(json.dumps(co_metrics, indent=2))
    print(f"      {co_metrics}")
    reserve = reserve_staffing_plan(callout.predict(panel[panel["month"] == SCORE_MONTH]))
    reserve.to_csv(REPORTS / "reserve_staffing.csv", index=False)

    # ------------------------------------------------------------------
    print("4/6 hiring funnel + requisition timing...")
    reqs = simulate_funnel(result.districts, n_months=24)
    fr = funnel_report(reqs)
    fr.to_csv(REPORTS / "funnel_report.csv", index=False)
    plan = pd.read_csv(REPORTS / "hiring_plan_6m.csv") \
        if (REPORTS / "hiring_plan_6m.csv").exists() else None
    if plan is not None:
        timing = req_timing(plan, reqs)
        timing.to_csv(REPORTS / "req_timing.csv", index=False)
        print(timing[timing["post_now"]].groupby("role").size()
              .rename("district-role batches to post now").to_string())

    # ------------------------------------------------------------------
    print("5/6 promotion readiness + bench strength...")
    promo_panel = build_promotion_panel(snaps, pm, horizon=6)
    ptr, pva, pte = time_split(promo_panel, TRAIN_END, VAL_END, train_stride=2)
    promo = PromotionModel(horizon=6).fit(ptr, pva)
    promo_metrics = promo.evaluate(pte)
    (REPORTS / "promotion_metrics.json").write_text(json.dumps(promo_metrics, indent=2))
    print(f"      {promo_metrics}")

    tr_s, va_s, _ = time_split(snaps, TRAIN_END, VAL_END, train_stride=1)
    salaried = TurnoverModel("salaried", horizons=(12,)).fit(tr_s, va_s)
    bench = bench_strength(
        promo.predict(promo_panel[promo_panel["month"] == SCORE_MONTH]),
        salaried.predict(snaps[snaps["month"] == SCORE_MONTH]),
        result.stores, horizon=12)
    bench.to_csv(REPORTS / "bench_strength.csv", index=False)

    # ------------------------------------------------------------------
    print("6/6 contagion analysis + figures...")
    cont = contagion_analysis(pm)
    cont["raw"].merge(cont["adjusted"], on="peer_exit_rate_3m_bucket",
                      suffixes=("_raw", "_adj"), how="left") \
        .to_csv(REPORTS / "contagion.csv", index=False)

    # Fig A: one store-week, actual vs forecast with required staff.
    g = preds[(preds["store_id"] == sid) & (preds["week"] == 100)].sort_values(["dow", "hour"])
    xs = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.plot(xs, g["transactions"], color=MUTED, linewidth=1.2, label="actual")
    ax.plot(xs, g["forecast"], color=BLUE, linewidth=2, label="forecast")
    for d in range(1, 7):
        ax.axvline(d * 15 - 0.5, color=GRID, linewidth=0.8)
    ax.set_xticks([7.5 + 15 * d for d in range(7)],
                  ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    style_axis(ax, "Hourly transactions, one store-week: forecast vs actual")
    ax.set_ylabel("Transactions/hour")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "demand_forecast.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig B: call-out seasonality, actual vs model expectation by calendar month.
    te_pred = callout.predict(te)
    te_h = te[te["role"].isin(["barista", "shift_supervisor"])].reset_index(drop=True)
    # The target is next month's call-outs, so index by the OUTCOME month.
    outcome_month = te_h["month_of_year"].astype(int).to_numpy() % 12 + 1
    seas = pd.DataFrame({
        "month_of_year": outcome_month,
        "actual": te_h["callouts_next_month"].to_numpy(),
        "predicted": te_pred["expected_callouts"].to_numpy(),
    }).groupby("month_of_year").mean().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.plot(seas.index, seas["actual"], color=MUTED, linewidth=1.5, marker="o",
            markersize=4, label="actual")
    ax.plot(seas.index, seas["predicted"], color=BLUE, linewidth=2, marker="o",
            markersize=4, markeredgecolor=SURFACE, label="model")
    style_axis(ax, "Call-outs per employee-month by calendar month (test window)")
    ax.set_xlabel("Calendar month")
    ax.set_xticks(range(1, 13))
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "callout_seasonality.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)

    # Fig C: the hiring funnel, chain-wide.
    stages = ["applications", "screen_passes", "interviews_attended", "offers",
              "accepts", "starts"]
    totals = [int(reqs[s].sum()) for s in stages]
    labels = ["applied", "passed screen", "attended interview", "offered",
              "accepted", "started day 1"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.barh(labels[::-1], totals[::-1], color=BLUE, height=0.55, zorder=3)
    for y, v in enumerate(totals[::-1]):
        ax.text(v + max(totals) * 0.01, y, f"{v:,}", va="center", color=INK2, fontsize=9)
    style_axis(ax, "Frontline hiring funnel, 24 simulated months")
    ax.set_xlim(0, max(totals) * 1.12)
    fig.tight_layout()
    fig.savefig(FIGURES / "hiring_funnel.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Fig D: turnover contagion, raw vs adjusted relative risk.
    raw, adj = cont["raw"], cont["adjusted"]
    order = raw["peer_exit_rate_3m_bucket"].tolist()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.bar(x - 0.18, raw["relative_risk"], width=0.32, color=BLUE,
           label="raw (looks like contagion)", zorder=3)
    adj_map = adj.set_index("peer_exit_rate_3m_bucket")["adjusted_relative_risk"]
    ax.bar(x + 0.18, [adj_map.get(b, np.nan) for b in order], width=0.32,
           color=GREEN, label="adjusted for store conditions", zorder=3)
    ax.axhline(1.0, color=BASELINE, linewidth=1)
    style_axis(ax, "Exit risk vs share of teammates who just left")
    ax.set_xticks(x, order)
    ax.set_xlabel("Peer exits in prior 3 months / team size")
    ax.set_ylabel("Relative risk vs <5% bucket")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "contagion.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    print(f"done. reports -> {REPORTS}  figures -> {FIGURES}")


if __name__ == "__main__":
    main()
