"""Synthetic retail-workforce simulator with a known data-generating process.

Simulates a multi-district retail company month by month: stores, hourly
employees (baristas, shift supervisors) and salaried employees (assistant
store managers, store managers). Each month every active employee faces a
termination hazard produced by a discrete-time logistic model whose
coefficients live in :class:`~workforce_analytics.config.GroundTruth`.

Because the true drivers of attrition are known, anything the downstream
models claim about drivers can be validated against the generating process —
something that is impossible with real HR data or with the usual static
attrition demo datasets.

Outputs
-------
``SimulationResult`` holds four tidy tables:

* ``districts`` — one row per district (labour-market context)
* ``stores`` — one row per store (tier, staffing targets, open month)
* ``employees`` — one row per employee (static attributes + final outcome)
* ``person_months`` — one row per employee per active month, with the
  time-varying state *as it was that month* and a ``terminated`` flag for
  the month the exit happened. This is the raw material for point-in-time
  snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    AGE_BANDS,
    HOURLY_ROLES,
    SALARIED_ROLES,
    GroundTruth,
    SimulationConfig,
)

UNDERSTAFFED_THRESHOLD = 0.85


@dataclass
class SimulationResult:
    districts: pd.DataFrame
    stores: pd.DataFrame
    employees: pd.DataFrame
    person_months: pd.DataFrame
    config: SimulationConfig

    def save(self, out_dir) -> None:
        from pathlib import Path
        import json

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.districts.to_csv(out / "districts.csv", index=False)
        self.stores.to_csv(out / "stores.csv", index=False)
        self.employees.to_csv(out / "employees.csv", index=False)
        self.person_months.to_csv(out / "person_months.csv", index=False)
        with open(out / "ground_truth.json", "w") as f:
            json.dump(self.config.ground_truth.as_dict(), f, indent=2)


def _tenure_bucket(tenure: np.ndarray) -> np.ndarray:
    """Map tenure in months to hazard buckets."""
    buckets = np.full(tenure.shape, "24+", dtype=object)
    buckets[tenure < 24] = "12-23"
    buckets[tenure < 12] = "6-11"
    buckets[tenure < 6] = "3-5"
    buckets[tenure < 3] = "0-2"
    return buckets


class WorkforceSimulator:
    """Month-by-month simulation of hiring, promotion and attrition."""

    def __init__(self, config: SimulationConfig | None = None):
        self.cfg = config or SimulationConfig()
        self.gt: GroundTruth = self.cfg.ground_truth
        self.rng = np.random.default_rng(self.cfg.seed)
        self._next_employee_id = 0
        self._market_mult = 1.0

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------
    def _build_districts(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        n = cfg.n_districts
        growth = np.zeros(n, dtype=bool)
        growth[rng.choice(n, size=min(cfg.n_growth_districts, n), replace=False)] = True
        return pd.DataFrame({
            "district_id": [f"D{i:02d}" for i in range(n)],
            "cost_index": np.clip(rng.normal(1.0, 0.08, n), 0.8, 1.25),
            "labor_tightness": rng.uniform(0.45, 0.80, n),  # monthly vacancy fill prob
            "unemployment_rate": np.round(rng.uniform(3.0, 7.0, n), 1),
            "is_growth": growth,
        })

    def _build_stores(self, districts: pd.DataFrame) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        rows = []
        sid = 0
        for _, d in districts.iterrows():
            n_stores = rng.integers(cfg.stores_per_district_min, cfg.stores_per_district_max + 1)
            open_months = [0] * n_stores
            if d["is_growth"] and cfg.n_months > 24:
                extra = cfg.new_stores_per_growth_district
                open_months += sorted(rng.integers(12, cfg.n_months - 9, extra).tolist())
                n_stores += extra
            for open_month in open_months:
                tier = int(rng.choice([1, 2, 3], p=cfg.tier_probs))
                rows.append({
                    "store_id": f"S{sid:03d}",
                    "district_id": d["district_id"],
                    "tier": tier,
                    "open_month": int(open_month),
                    "volatility_base": float(rng.uniform(1.5, 4.5)),  # hrs std-dev of weekly schedule
                    "target_baristas": cfg.target_baristas[tier],
                    "target_shift_supervisors": cfg.target_shift_supervisors[tier],
                    "has_asm": tier >= 2,
                })
                sid += 1
        return pd.DataFrame(rows)

    def _sample_age_band(self, role: str, size: int) -> np.ndarray:
        probs = {
            "barista": [0.30, 0.30, 0.20, 0.13, 0.07],
            "shift_supervisor": [0.05, 0.30, 0.35, 0.22, 0.08],
            "assistant_store_manager": [0.0, 0.15, 0.45, 0.32, 0.08],
            "store_manager": [0.0, 0.05, 0.40, 0.42, 0.13],
        }[role]
        return self.rng.choice(np.array(AGE_BANDS, dtype=object), size=size, p=probs)

    def _new_employees(self, role: str, store_rows: pd.DataFrame, month: int,
                       initial: bool = False) -> pd.DataFrame:
        """Create hires for the given role, one per row of ``store_rows``."""
        cfg, rng = self.cfg, self.rng
        n = len(store_rows)
        ids = np.arange(self._next_employee_id, self._next_employee_id + n)
        self._next_employee_id += n

        age_band = self._sample_age_band(role, n)
        is_student = np.zeros(n, dtype=bool)
        if role == "barista":
            is_student = (
                ((age_band == "16-20") & (rng.random(n) < 0.55))
                | ((age_band == "21-25") & (rng.random(n) < 0.30))
            )
        second_job = (rng.random(n) < 0.18) & np.isin(np.array([role] * n), HOURLY_ROLES)

        if initial:
            # Backfill tenure so the starting population is not all new hires.
            mean_tenure = {"barista": 14, "shift_supervisor": 30,
                           "assistant_store_manager": 36, "store_manager": 55}[role]
            hire_month = month - rng.geometric(1.0 / mean_tenure, n)
        else:
            hire_month = np.full(n, month)

        market = (cfg.market_pay[role]
                  * store_rows["cost_index"].to_numpy(dtype=float) * self._market_mult)
        # Hourly pay bands are tight; salaried offers disperse more widely.
        pay_sd = 0.08 if role in SALARIED_ROLES else 0.05
        pay_rate = market * np.clip(rng.normal(1.0, pay_sd, n), 0.80, 1.25)

        commute_scale = 0.75 if role in SALARIED_ROLES else 0.55
        commute = np.clip(rng.lognormal(1.7, commute_scale, n), 0.5, 60.0)

        desired = np.where(is_student, rng.uniform(16, 22, n), rng.uniform(28, 38, n))
        if role == "shift_supervisor":
            desired = rng.uniform(34, 40, n)

        perf_latent = rng.normal(0, 1, n)
        return pd.DataFrame({
            "employee_id": ids,
            "store_id": store_rows["store_id"].to_numpy(),
            "district_id": store_rows["district_id"].to_numpy(),
            "role": role,
            "age_band": age_band,
            "is_student": is_student,
            "second_job": second_job,
            "commute_km": np.round(commute, 1),
            "desired_hours": np.round(desired, 0),
            "perf_latent": perf_latent,
            "performance_rating": np.clip(np.round(3 + 0.9 * perf_latent
                                                   + rng.normal(0, 0.5, n)), 1, 5),
            "pay_rate": pay_rate,
            "hire_month": hire_month,
            "last_raise_month": hire_month,
            "last_promo_month": hire_month,
            "active": True,
            "term_month": -1,
            "term_type": "",
        })

    # ------------------------------------------------------------------
    # Hazard model (the ground truth)
    # ------------------------------------------------------------------
    def _monthly_log_odds(self, emp: pd.DataFrame, month: int,
                          staffing_ratio: pd.Series,
                          district_understaffed_share: pd.Series) -> np.ndarray:
        gt = self.gt
        month_of_year = month % 12 + 1
        role = emp["role"].to_numpy()
        hourly = np.isin(role, HOURLY_ROLES)
        tenure = (month - emp["hire_month"]).to_numpy()

        lo = np.array([gt.base_log_odds[r] for r in role])
        buckets = _tenure_bucket(tenure)
        lo += np.where(
            hourly,
            np.array([gt.tenure_log_odds[b] for b in buckets]),
            np.array([gt.tenure_log_odds_salaried[b] for b in buckets]),
        )

        pay_centered = emp["pay_ratio"].to_numpy() - 1.0
        lo += np.where(hourly, gt.pay_ratio_hourly, gt.pay_ratio_salaried) * pay_centered

        high_perf = emp["performance_rating"].to_numpy() >= 4
        low_perf = emp["performance_rating"].to_numpy() <= 2
        lo += (~hourly) * high_perf * (pay_centered < -0.03) * gt.underpaid_high_performer_salaried
        lo += low_perf * np.where(hourly, gt.low_performance_hourly, gt.low_performance_salaried)

        vol = emp["schedule_volatility"].to_numpy()
        lo += hourly * gt.schedule_volatility_per_hour * np.maximum(vol - 3.0, 0)
        lo += hourly * gt.hours_gap_per_hour * emp["hours_gap"].to_numpy()

        commute = emp["commute_km"].to_numpy()
        lo += hourly * np.minimum(
            gt.commute_per_km_hourly * np.maximum(commute - gt.commute_threshold_km_hourly, 0),
            gt.commute_cap_hourly)
        lo += (~hourly) * np.minimum(
            gt.commute_per_km_salaried * np.maximum(commute - gt.commute_threshold_km_salaried, 0),
            gt.commute_cap_salaried)

        mgr_recent = emp["months_since_mgr_change"].to_numpy() <= 3
        # Store managers are not destabilised by their own arrival.
        not_sm = role != "store_manager"
        lo += mgr_recent * not_sm * np.where(
            hourly, gt.manager_change_recent_hourly, gt.manager_change_recent_salaried)

        if month_of_year in (8, 9):
            lo += emp["is_student"].to_numpy() * gt.student_back_to_school
        if month_of_year == 1:
            lo += hourly * gt.post_holiday_january_hourly

        understaffed = staffing_ratio.to_numpy() < UNDERSTAFFED_THRESHOLD
        lo += hourly * understaffed * gt.understaffed_store_hourly
        lo += (~hourly) * gt.district_understaffing_salaried * district_understaffed_share.to_numpy()

        promo_recent = (month - emp["last_promo_month"].to_numpy() <= 6) & \
                       (emp["last_promo_month"].to_numpy() > emp["hire_month"].to_numpy())
        lo += hourly * promo_recent * gt.recent_promotion_hourly

        since_promo = month - np.maximum(emp["last_promo_month"].to_numpy(),
                                         emp["hire_month"].to_numpy())
        lo += (~hourly) * (since_promo > 36) * gt.stagnation_salaried

        age = emp["age_band"].to_numpy()
        lo += (age == "16-20") * gt.age_16_20
        lo += (age == "50+") * gt.age_50_plus
        lo += emp["second_job"].to_numpy() * gt.second_job
        return lo

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        cfg, rng = self.cfg, self.rng
        districts = self._build_districts()
        stores = self._build_stores(districts)
        stores = stores.merge(districts[["district_id", "cost_index", "labor_tightness",
                                         "unemployment_rate"]], on="district_id")
        store_mgr_change = dict.fromkeys(stores["store_id"], -24)  # month of last SM change

        # Staff every store open at month 0 to target.
        frames = []
        open_now = stores[stores["open_month"] == 0]
        for _, s in open_now.iterrows():
            srow = s.to_frame().T
            frames.append(self._new_employees(
                "barista", srow.loc[srow.index.repeat(s["target_baristas"])], 0, initial=True))
            frames.append(self._new_employees(
                "shift_supervisor",
                srow.loc[srow.index.repeat(s["target_shift_supervisors"])], 0, initial=True))
            if s["has_asm"]:
                frames.append(self._new_employees("assistant_store_manager", srow, 0, initial=True))
            frames.append(self._new_employees("store_manager", srow, 0, initial=True))
        emp = pd.concat(frames, ignore_index=True)

        records = []
        for month in range(cfg.n_months):
            self._market_mult = (1 + cfg.market_drift_monthly) ** month
            active = emp["active"]

            # --- pay dynamics -------------------------------------------------
            anniversary = active & (month > emp["hire_month"]) & \
                ((month - emp["hire_month"]) % 12 == 0)
            if anniversary.any():
                raises = 1 + np.clip(rng.normal(cfg.merit_raise_mean, cfg.merit_raise_sd,
                                                int(anniversary.sum())), 0.005, 0.08)
                emp.loc[anniversary, "pay_rate"] = emp.loc[anniversary, "pay_rate"] * raises
                emp.loc[anniversary, "last_raise_month"] = month

            market_by_store = stores.set_index("store_id")["cost_index"] * self._market_mult
            market_role = emp["role"].map(cfg.market_pay)
            emp["pay_ratio"] = emp["pay_rate"] / (
                market_role * emp["store_id"].map(market_by_store))

            # Refresh observed performance rating annually.
            refresh = active & ((month - emp["hire_month"]) % 12 == 6)
            if refresh.any():
                emp.loc[refresh, "performance_rating"] = np.clip(
                    np.round(3 + 0.9 * emp.loc[refresh, "perf_latent"]
                             + rng.normal(0, 0.5, int(refresh.sum()))), 1, 5)

            # --- staffing & scheduling ---------------------------------------
            open_stores = stores[stores["open_month"] <= month]
            hourly_active = emp[active & emp["role"].isin(HOURLY_ROLES)]
            counts = hourly_active.groupby("store_id").size()
            targets = (open_stores.set_index("store_id")["target_baristas"]
                       + open_stores.set_index("store_id")["target_shift_supervisors"])
            staffing = (counts.reindex(targets.index).fillna(0) / targets).clip(0, 1.3)
            emp_staffing = emp["store_id"].map(staffing).fillna(1.0)

            understaffed_stores = staffing < UNDERSTAFFED_THRESHOLD
            dist_of_store = open_stores.set_index("store_id")["district_id"]
            share = understaffed_stores.groupby(dist_of_store).mean()
            emp_dist_share = emp["district_id"].map(share).fillna(0.0)

            vol_base = emp["store_id"].map(stores.set_index("store_id")["volatility_base"])
            vol = vol_base * rng.lognormal(0, 0.35, len(emp))
            vol = np.where(emp_staffing < UNDERSTAFFED_THRESHOLD, vol * 1.35, vol)
            vol = np.where(emp["role"].isin(SALARIED_ROLES), 0.5, vol)
            emp["schedule_volatility"] = np.round(vol, 2)

            base_sched = np.select(
                [emp["role"] == "barista", emp["role"] == "shift_supervisor"],
                [26.0, 34.0], default=42.0)
            sched = base_sched + np.where(emp_staffing < UNDERSTAFFED_THRESHOLD, 4.0, 0.0) \
                + rng.normal(0, 2.5, len(emp))
            sched = np.minimum(sched, np.maximum(emp["desired_hours"], 20) + 4)
            emp["scheduled_hours"] = np.round(np.clip(sched, 8, 60), 1)
            gap = np.where(emp["role"].isin(HOURLY_ROLES),
                           np.maximum(emp["desired_hours"] - emp["scheduled_hours"], 0), 0.0)
            emp["hours_gap"] = np.round(gap, 1)

            emp["months_since_mgr_change"] = (
                month - emp["store_id"].map(store_mgr_change)).clip(upper=48)

            # --- termination draw --------------------------------------------
            act_idx = emp.index[active]
            lo = self._monthly_log_odds(emp.loc[act_idx], month,
                                        emp_staffing.loc[act_idx],
                                        emp_dist_share.loc[act_idx])
            p = 1 / (1 + np.exp(-lo))
            leaves = rng.random(len(act_idx)) < p
            leaver_idx = act_idx[leaves]

            low_perf = emp.loc[leaver_idx, "performance_rating"] <= 2
            inv_share = np.where(emp.loc[leaver_idx, "role"].isin(HOURLY_ROLES),
                                 self.gt.involuntary_share_low_perf_hourly,
                                 self.gt.involuntary_share_low_perf_salaried)
            involuntary = (low_perf.to_numpy() & (rng.random(len(leaver_idx)) < inv_share)) | \
                          (rng.random(len(leaver_idx)) < 0.03)

            # --- record person-months (state as of this month) ----------------
            snap = emp.loc[act_idx, [
                "employee_id", "store_id", "district_id", "role", "age_band",
                "is_student", "second_job", "commute_km", "pay_rate", "pay_ratio",
                "schedule_volatility", "scheduled_hours", "hours_gap",
                "months_since_mgr_change", "performance_rating",
            ]].copy()
            snap.insert(0, "month", month)
            snap["tenure_months"] = (month - emp.loc[act_idx, "hire_month"]).to_numpy()
            snap["months_since_promotion"] = (
                month - np.maximum(emp.loc[act_idx, "last_promo_month"],
                                   emp.loc[act_idx, "hire_month"])).to_numpy()
            snap["months_since_raise"] = (
                month - emp.loc[act_idx, "last_raise_month"]).to_numpy()
            snap["store_staffing_ratio"] = np.round(emp_staffing.loc[act_idx], 3).to_numpy()
            snap["district_understaffed_share"] = np.round(
                emp_dist_share.loc[act_idx], 3).to_numpy()
            snap["district_unemployment"] = emp.loc[act_idx, "district_id"].map(
                districts.set_index("district_id")["unemployment_rate"]).to_numpy()
            snap["terminated"] = leaves.astype(int)
            snap["termination_type"] = ""
            snap.loc[snap.index[leaves], "termination_type"] = np.where(
                involuntary, "involuntary", "voluntary")
            records.append(snap)

            emp.loc[leaver_idx, "active"] = False
            emp.loc[leaver_idx, "term_month"] = month
            emp.loc[leaver_idx, "term_type"] = np.where(involuntary, "involuntary", "voluntary")

            sm_left = emp.loc[leaver_idx][emp.loc[leaver_idx, "role"] == "store_manager"]
            for sid in sm_left["store_id"]:
                store_mgr_change[sid] = month

            # --- hiring & promotion ------------------------------------------
            if month < cfg.n_months - 1:
                emp = self._fill_vacancies(emp, stores, month, store_mgr_change)

        person_months = pd.concat(records, ignore_index=True)
        employees = emp[[
            "employee_id", "store_id", "district_id", "role", "age_band", "is_student",
            "second_job", "commute_km", "hire_month", "term_month", "term_type", "active",
        ]].copy()
        return SimulationResult(districts, stores, employees, person_months, cfg)

    # ------------------------------------------------------------------
    def _fill_vacancies(self, emp: pd.DataFrame, stores: pd.DataFrame, month: int,
                        store_mgr_change: dict) -> pd.DataFrame:
        """Fill open roles via promotion or external hire, with market friction."""
        cfg, rng = self.cfg, self.rng
        next_month = month + 1
        open_stores = stores[stores["open_month"] <= next_month]
        active = emp[emp["active"]]
        new_frames = []

        for _, s in open_stores.iterrows():
            at_store = active[active["store_id"] == s["store_id"]]
            fill_p = min(s["labor_tightness"] * (1.6 if s["open_month"] >= next_month - 1 else 1.0), 0.95)
            srow = s.to_frame().T

            # Store manager: prefer promoting the ASM.
            if (at_store["role"] == "store_manager").sum() == 0:
                asms = at_store[(at_store["role"] == "assistant_store_manager")
                                & (month - at_store["hire_month"] >= 12)]
                if len(asms) > 0 and rng.random() < 0.6:
                    promotee = asms.sort_values("perf_latent").index[-1]
                    self._promote(emp, promotee, "store_manager", s, month)
                    store_mgr_change[s["store_id"]] = month
                elif rng.random() < fill_p:
                    new_frames.append(self._new_employees("store_manager", srow, next_month))
                    store_mgr_change[s["store_id"]] = next_month
                active = emp[emp["active"]]
                at_store = active[active["store_id"] == s["store_id"]]

            # Assistant store manager: prefer promoting a shift supervisor.
            if s["has_asm"] and (at_store["role"] == "assistant_store_manager").sum() == 0:
                sss = at_store[(at_store["role"] == "shift_supervisor")
                               & (month - at_store["hire_month"] >= 12)]
                if len(sss) > 0 and rng.random() < 0.5:
                    promotee = sss.sort_values("perf_latent").index[-1]
                    self._promote(emp, promotee, "assistant_store_manager", s, month)
                elif rng.random() < fill_p:
                    new_frames.append(self._new_employees("assistant_store_manager", srow, next_month))
                active = emp[emp["active"]]
                at_store = active[active["store_id"] == s["store_id"]]

            # Shift supervisors: promote baristas or hire.
            n_ss = (at_store["role"] == "shift_supervisor").sum()
            for _ in range(int(s["target_shift_supervisors"] - n_ss)):
                baristas = at_store[(at_store["role"] == "barista")
                                    & (month - at_store["hire_month"] >= 9)]
                if len(baristas) > 0 and rng.random() < 0.5:
                    promotee = baristas.sort_values("perf_latent").index[-1]
                    self._promote(emp, promotee, "shift_supervisor", s, month)
                    active = emp[emp["active"]]
                    at_store = active[active["store_id"] == s["store_id"]]
                elif rng.random() < fill_p:
                    new_frames.append(self._new_employees("shift_supervisor", srow, next_month))

            # Baristas: external hires only.
            n_b = (at_store["role"] == "barista").sum()
            vacancies = int(s["target_baristas"] - n_b)
            if vacancies > 0:
                filled = int(rng.binomial(vacancies, fill_p))
                if filled > 0:
                    new_frames.append(self._new_employees(
                        "barista", srow.loc[srow.index.repeat(filled)], next_month))

        if new_frames:
            emp = pd.concat([emp] + new_frames, ignore_index=True)
        return emp

    def _promote(self, emp: pd.DataFrame, idx, new_role: str, store: pd.Series,
                 month: int) -> None:
        market = self.cfg.market_pay[new_role] * store["cost_index"] * self._market_mult
        emp.loc[idx, "role"] = new_role
        emp.loc[idx, "pay_rate"] = market * float(np.clip(self.rng.normal(1.02, 0.04), 0.9, 1.15))
        emp.loc[idx, "last_promo_month"] = month
        emp.loc[idx, "last_raise_month"] = month
        if new_role not in HOURLY_ROLES:
            emp.loc[idx, "second_job"] = False


def generate(config: SimulationConfig | None = None) -> SimulationResult:
    """Run a full simulation and return the result tables."""
    return WorkforceSimulator(config).run()
