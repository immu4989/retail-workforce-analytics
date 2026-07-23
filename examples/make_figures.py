"""Generate the figures for use cases 12-18 (the systems added after the
original eleven), in the repo's shared matplotlib style.

Run from the repo root (takes ~2 minutes):

    python examples/make_figures.py

Writes seven PNGs into ``docs/figures/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from workforce_analytics import (
    PayEquityGroundTruth,
    SimulationConfig,
    TrafficConfig,
    TrafficSimulator,
    assign_group_and_gap,
    assign_homes,
    assign_store_coordinates,
    audit_pay_gap,
    build_employee_frame,
    compare_strategies,
    derive_understaffing_cost,
    detect_buddy_punching,
    detect_ghost_shifts,
    detect_time_padding,
    evaluate_detection,
    generate,
    milestone_retention,
    optimize_transfers,
    recover_task_seconds,
    service_loss_curve,
    simulate_exit_comments,
    simulate_onboarding,
    simulate_order_mix,
    simulate_punches,
    store_vacancies,
    theme_driver_alignment,
    transfer_options,
    workforce_snapshot,
)
from workforce_analytics.exitnlp import THEME_DRIVERS
from workforce_analytics.tasks import CHANNELS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, GREEN, RED = "#2a78d6", "#008300", "#e34948"
AMBER, PURPLE, TEAL = "#c8862b", "#6a5acd", "#2a9d8f"
CATS = [BLUE, GREEN, AMBER, PURPLE, TEAL, RED]


def style_axis(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _save(fig, name: str):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print(f"      wrote docs/figures/{name}")


def fig_elasticity(plt, sim, traffic):
    curve = service_loss_curve(traffic)
    derived = derive_understaffing_cost(traffic)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = curve["short"][1:]
    y = curve["marginal_cost_per_head_short"][1:]
    ax.bar(x, y, color=BLUE, width=0.6, zorder=3)
    ax.axhline(35.0, color=RED, ls="--", lw=1.5, zorder=2,
               label="assumed $35 / understaffed hour")
    ax.scatter([1], [derived["derived_cost_first_head_short"]], color=INK, zorder=5)
    ax.annotate(f"derived ${derived['derived_cost_first_head_short']:.2f}",
                (1, derived["derived_cost_first_head_short"]),
                textcoords="offset points", xytext=(12, 6), color=INK, fontsize=9)
    style_axis(ax, "Use case 12 — the cost of an understaffed hour, derived")
    ax.set_xlabel("heads short of the labor standard")
    ax.set_ylabel("marginal $ lost per head-hour")
    ax.set_xticks([1, 2, 3])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, "service_loss_curve.png")


def fig_anomaly(plt, sim):
    panel = simulate_punches(sim.person_months, seed=0)
    dets = {
        "time\npadding": evaluate_detection(detect_time_padding(panel), "is_padder"),
        "ghost\nshifts": evaluate_detection(detect_ghost_shifts(panel), "is_ghost_month"),
        "buddy\npunching": evaluate_detection(
            detect_buddy_punching(panel), "is_buddy", "within_emp_z"),
    }
    labels = list(dets)
    x = np.arange(len(labels))
    prec = [dets[k]["precision"] for k in labels]
    rec = [dets[k]["recall"] for k in labels]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - 0.2, prec, 0.38, color=BLUE, label="precision", zorder=3)
    ax.bar(x + 0.2, rec, 0.38, color=GREEN, label="recall", zorder=3)
    for i, k in enumerate(labels):
        ax.annotate(f"AP {dets[k]['average_precision']:.2f}", (i, 1.02),
                    ha="center", color=MUTED, fontsize=8)
    style_axis(ax, "Use case 13 — payroll-anomaly detection vs planted truth")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, "anomaly_detection.png")


def fig_onboarding(plt, sim):
    cohort = simulate_onboarding(sim.person_months, seed=29)
    comp = cohort[cohort["training_completed_30d"] == 1]
    inc = cohort[cohort["training_completed_30d"] == 0]

    def survival(df):
        e = df["exit_month"].to_numpy()
        return [1.0] + [float(((e == 0) | (e > k)).mean()) for k in (1, 2, 3)]

    days = [0, 30, 60, 90]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(days, survival(comp), "-o", color=GREEN, lw=2, label="finished 30-day training")
    ax.plot(days, survival(inc), "-o", color=RED, lw=2, label="did not finish training")
    ax.plot(days, survival(cohort), "--", color=MUTED, lw=1.4, label="all new hires")
    style_axis(ax, "Use case 14 — first-90-day retention, by training completion")
    ax.set_xlabel("days since hire")
    ax.set_ylabel("share still employed")
    ax.set_xticks(days)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, "onboarding_milestones.png")


def fig_exit_alignment(plt, sim):
    comments = simulate_exit_comments(sim.person_months, seed=47)
    themes, on, off = [], [], []
    for theme, (driver, sign) in THEME_DRIVERS.items():
        s = comments[driver]
        pct = s.rank(pct=True)
        m_on = pct[comments["true_theme"] == theme].mean()
        m_off = pct[comments["true_theme"] != theme].mean()
        if sign == "-":
            m_on, m_off = 1 - m_on, 1 - m_off
        themes.append(theme)
        on.append(m_on)
        off.append(m_off)
    x = np.arange(len(themes))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - 0.2, on, 0.38, color=BLUE, label="leavers on this theme", zorder=3)
    ax.bar(x + 0.2, off, 0.38, color=BASELINE, label="everyone else", zorder=3)
    style_axis(ax, "Use case 15 — each exit theme sits high on its true driver")
    ax.set_xticks(x)
    ax.set_xticklabels(themes, fontsize=8)
    ax.set_ylabel("driver percentile (risk direction)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, "exit_theme_alignment.png")


def fig_task_recovery(plt, sim, traffic):
    order_mix = simulate_order_mix(traffic, seed=53)
    rec = recover_task_seconds(order_mix)
    x = np.arange(len(CHANNELS))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - 0.2, rec["true_seconds"], 0.38, color=BLUE, label="true task time", zorder=3)
    ax.bar(x + 0.2, rec["recovered_seconds"], 0.38, color=GREEN,
           label="recovered by NNLS", zorder=3)
    style_axis(ax, "Use case 16 — per-channel labor standards, recovered")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CHANNELS], fontsize=8)
    ax.set_ylabel("labor seconds per order")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, "task_time_recovery.png")


def fig_pay_equity(plt, sim):
    emp = build_employee_frame(sim.person_months, sim.stores)
    clean = audit_pay_gap(assign_group_and_gap(
        emp, PayEquityGroundTruth(true_residual_gap=0.04), seed=61))
    conf = audit_pay_gap(assign_group_and_gap(
        emp, PayEquityGroundTruth(true_residual_gap=0.04, confounding_strength=1.0), seed=61))
    labels = ["no confounding", "group tied to tenure"]
    unadj = [clean["unadjusted_gap"] * 100, conf["unadjusted_gap"] * 100]
    adj = [clean["adjusted_gap"] * 100, conf["adjusted_gap"] * 100]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - 0.2, unadj, 0.38, color=RED, label="unadjusted gap", zorder=3)
    ax.bar(x + 0.2, adj, 0.38, color=GREEN, label="adjusted (with controls)", zorder=3)
    ax.axhline(4.0, color=INK, ls="--", lw=1.4, label="true planted gap (4%)")
    style_axis(ax, "Use case 17 — controls recover the true gap; the raw number misleads")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("estimated pay gap (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, "pay_equity_confounding.png")


def fig_geo_map(plt, sim):
    coords = assign_store_coordinates(sim.stores, seed=71)
    snap = workforce_snapshot(sim.person_months, month=48)
    homes = assign_homes(snap, coords, seed=72)
    vac = store_vacancies(snap, sim.stores)
    options = transfer_options(homes, coords, vac)
    plan = optimize_transfers(options, vac)
    xy = coords.set_index("store_id")[["store_x", "store_y"]]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(coords["store_x"], coords["store_y"], s=14, color=BASELINE,
               zorder=2, label="stores")
    home_xy = homes.set_index("employee_id")[["home_x", "home_y"]]
    for _, r in plan.iterrows():
        h = home_xy.loc[r["employee_id"]]
        t = xy.loc[r["to_store"]]
        ax.annotate("", xy=(t["store_x"], t["store_y"]), xytext=(h["home_x"], h["home_y"]),
                    arrowprops=dict(arrowstyle="->", color=BLUE, alpha=0.5, lw=0.8), zorder=3)
    style_axis(ax, f"Use case 18 — {len(plan)} optimal transfers to closer stores")
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, "geo_transfer_map.png")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    print("simulating base workforce + traffic...")
    sim = generate(SimulationConfig())
    stores = sim.stores[sim.stores["open_month"] == 0].reset_index(drop=True)
    traffic = TrafficSimulator(stores, TrafficConfig(n_weeks=52)).run()

    print("rendering figures for use cases 12-18...")
    fig_elasticity(plt, sim, traffic)
    fig_anomaly(plt, sim)
    fig_onboarding(plt, sim)
    fig_exit_alignment(plt, sim)
    fig_task_recovery(plt, sim, traffic)
    fig_pay_equity(plt, sim)
    fig_geo_map(plt, sim)
    print("done.")


if __name__ == "__main__":
    main()
