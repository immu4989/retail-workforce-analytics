"""Use case 11: employee call-center topic modelling (voice of the frontline).

Large chains run internal support lines for their store employees (Starbucks'
Partner Contact Center is the canonical example): pay problems, scheduling
questions, benefits, leave, app trouble, conflicts. The transcripts are one
of the few places the frontline tells the company what is actually wrong, in
its own words — and almost nobody mines them.

The repo's usual discipline applies. Every simulated call carries a hidden
``true_topic``, generated from template banks with heavy paraphrase noise,
and the topic drivers are planted in the *simulation state*: underpaid
employees call about pay errors, chaotic-schedule stores call about
scheduling, November brings open-enrollment benefits calls. That gives three
checkable claims instead of a wordcloud:

1. The unsupervised topic model (TF-IDF + NMF, scikit-learn only) recovers
   the planted topics — measured with normalised mutual information and
   cluster purity, not eyeballed.
2. Topic volumes carry operational signal: store-level scheduling-call rates
   correlate with true schedule volatility, pay-call rates with pay position.
3. The trend view catches what changed: seasonal spikes and district
   hot-spots that route straight to the responsible team (payroll ops,
   scheduling policy, benefits comms).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .config import HOURLY_ROLES

TOPICS = ("pay_error", "scheduling", "benefits", "leave", "app_tech", "conflict")


@dataclass
class CallCenterGroundTruth:
    """Monthly per-employee call propensities and their state-dependent boosts.

    Log-rate model per topic; calls per employee-month drawn Poisson. The
    boosts are the planted 'operational linkage' the analytics must recover.
    """

    base_log_rate: dict = field(default_factory=lambda: {
        "pay_error": -4.0, "scheduling": -3.8, "benefits": -4.4,
        "leave": -4.8, "app_tech": -4.3, "conflict": -5.2,
    })
    pay_error_underpaid: float = 1.0          # pay_ratio < 0.95
    pay_error_no_recent_raise: float = 0.5    # months_since_raise > 12
    scheduling_volatility_per_hour: float = 0.25   # per hour of std-dev above 3
    scheduling_hours_gap: float = 0.4         # hours_gap > 4
    benefits_open_enrollment: float = 1.6     # November
    leave_tenured: float = 0.6                # tenure >= 12 months
    app_tech_new_hire: float = 1.2            # tenure < 3 months
    conflict_mgr_change: float = 0.9          # manager changed <= 3 months ago

    def as_dict(self) -> dict:
        return asdict(self)


# Template banks: verbose, overlapping vocabulary with topic-specific cores,
# so the topic model has to work for its supper but can win.
_TEMPLATES = {
    "pay_error": [
        "my paycheck is short this week and the direct deposit amount does not match my hours",
        "missing overtime pay on my last pay stub, the premium hours were not paid out",
        "my raise was approved last month but my hourly rate on the paystub never changed",
        "paycheck deduction looks wrong and payroll took out too much tax this period",
        "i did not get paid for the holiday shift and the pay statement shows zero premium",
        "my wages for last week are missing two shifts that i definitely worked and clocked",
    ],
    "scheduling": [
        "my schedule keeps changing every week and i cannot plan childcare around these shifts",
        "i was posted for a closing shift then an opening shift with no rest in between",
        "my posted hours were cut this week without notice and i need more shifts",
        "the new schedule came out two days before the week starts which is too late",
        "i asked for more hours months ago but the schedule still gives me short weeks",
        "swapped a shift with a coworker and the schedule still shows me on the old shift",
    ],
    "benefits": [
        "question about health insurance enrollment and which medical plan covers my family",
        "how do i sign up for the stock plan and the tuition reimbursement benefit",
        "my dental coverage claim was denied and i want to check my plan eligibility",
        "need help with the 401k retirement match and changing my contribution percent",
        "open enrollment deadline question, want to compare the medical plan options",
        "does my part time status qualify me for the health benefits and sick pay accrual",
    ],
    "leave": [
        "i need to request parental leave and want to know how much is paid",
        "how do i file for a medical leave of absence, my doctor says six weeks",
        "questions about family leave paperwork and whether my job is protected while out",
        "returning from leave next month and need to confirm my reinstatement date",
        "bereavement leave policy question after a death in my family this week",
        "jury duty next month, how do i report the absence so it is excused",
    ],
    "app_tech": [
        "the scheduling app will not let me log in and keeps saying invalid password",
        "the timeclock did not record my punch this morning and the app crashed twice",
        "cannot access my pay stubs in the portal, the page just spins and errors",
        "new hire here, my employee id does not work in the app or the learning portal",
        "the shift swap feature in the app is broken and shows an error every time",
        "trying to update my direct deposit in the portal but it will not save",
    ],
    "conflict": [
        "i want to report a conflict with my new store manager about how breaks are handled",
        "a coworker keeps making hostile comments and my complaints to the manager went nowhere",
        "need to speak to someone confidentially about how the new manager treats the team",
        "unfair treatment on shift assignments since the management change, want to escalate",
        "requesting a transfer because of an ongoing dispute with a supervisor at my store",
        "how do i file a formal complaint about behavior at my store, it is getting worse",
    ],
}
_FILLER = [
    "calling from my store", "this has happened twice now", "please help",
    "i already asked my manager", "nobody could answer this at the store",
    "i work most weekday shifts", "this is really stressing me out",
    "i love the job otherwise", "can someone call me back", "thanks for the help",
]


def simulate_calls(person_months: pd.DataFrame,
                   gt: CallCenterGroundTruth | None = None,
                   seed: int = 41) -> pd.DataFrame:
    """One row per call: employee, month, hidden true topic, transcript."""
    gt = gt or CallCenterGroundTruth()
    rng = np.random.default_rng(seed)
    pm = person_months[person_months["role"].isin(HOURLY_ROLES)]
    month_of_year = (pm["month"] % 12 + 1).to_numpy()

    boosts = {
        "pay_error": (pm["pay_ratio"].to_numpy() < 0.95) * gt.pay_error_underpaid
        + (pm["months_since_raise"].to_numpy() > 12) * gt.pay_error_no_recent_raise,
        "scheduling": gt.scheduling_volatility_per_hour
        * np.maximum(pm["schedule_volatility"].to_numpy() - 3, 0)
        + (pm["hours_gap"].to_numpy() > 4) * gt.scheduling_hours_gap,
        "benefits": (month_of_year == 11) * gt.benefits_open_enrollment,
        "leave": (pm["tenure_months"].to_numpy() >= 12) * gt.leave_tenured,
        "app_tech": (pm["tenure_months"].to_numpy() < 3) * gt.app_tech_new_hire,
        "conflict": (pm["months_since_mgr_change"].to_numpy() <= 3) * gt.conflict_mgr_change,
    }

    rows = []
    for topic in TOPICS:
        n_calls = rng.poisson(np.exp(gt.base_log_rate[topic] + boosts[topic]))
        idx = np.repeat(np.arange(len(pm)), n_calls)
        bank = _TEMPLATES[topic]
        for i in idx:
            # Callers ramble: each transcript blends two of the topic's
            # concerns plus filler, so no single template fingerprints it.
            a, b = rng.choice(len(bank), size=2, replace=False)
            extras = rng.choice(_FILLER, size=rng.integers(1, 3), replace=False)
            words = (bank[a] + " " + bank[b]).split()
            keep = rng.random(len(words)) > 0.10
            text = " ".join([w for w, k in zip(words, keep) if k]
                            + list(extras))
            r = pm.iloc[i]
            rows.append({
                "month": int(r["month"]), "employee_id": r["employee_id"],
                "store_id": r["store_id"], "district_id": r["district_id"],
                "true_topic": topic, "transcript": text,
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


class CallTopicModel:
    """TF-IDF + NMF topic model with ground-truth-checkable evaluation."""

    def __init__(self, n_topics: int = len(TOPICS), random_state: int = 0):
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=5,
                                          max_features=3000)
        self.nmf = NMF(n_components=n_topics, random_state=random_state,
                       init="nndsvda", max_iter=400)

    def fit(self, calls: pd.DataFrame) -> "CallTopicModel":
        X = self.vectorizer.fit_transform(calls["transcript"])
        self.doc_topic_ = self.nmf.fit_transform(X)
        return self

    def top_terms(self, n: int = 8) -> pd.DataFrame:
        terms = np.array(self.vectorizer.get_feature_names_out())
        rows = []
        for k, comp in enumerate(self.nmf.components_):
            rows.append({"cluster": k,
                         "top_terms": ", ".join(terms[np.argsort(-comp)[:n]])})
        return pd.DataFrame(rows)

    def assign(self, calls: pd.DataFrame) -> pd.DataFrame:
        out = calls.copy()
        out["cluster"] = self.doc_topic_.argmax(axis=1)
        return out

    def evaluate(self, calls: pd.DataFrame) -> dict:
        """NMI + purity of unsupervised clusters against the hidden labels."""
        from sklearn.metrics import normalized_mutual_info_score

        assigned = self.assign(calls)
        nmi = normalized_mutual_info_score(assigned["true_topic"], assigned["cluster"])
        purity = (assigned.groupby("cluster")["true_topic"]
                  .apply(lambda s: s.value_counts().iloc[0]).sum() / len(assigned))
        mapping = (assigned.groupby("cluster")["true_topic"]
                   .agg(lambda s: s.mode().iloc[0]).to_dict())
        return {"n_calls": int(len(assigned)), "nmi": round(float(nmi), 3),
                "purity": round(float(purity), 3),
                "cluster_to_topic": mapping}


def topic_trends(assigned: pd.DataFrame, label_col: str = "true_topic") -> pd.DataFrame:
    """Calls per 1,000 hourly employees by calendar month and topic."""
    df = assigned.copy()
    df["month_of_year"] = df["month"] % 12 + 1
    counts = (df.groupby(["month_of_year", label_col], observed=True).size()
              .rename("calls").reset_index())
    months_per_moy = df.groupby(df["month"] % 12 + 1)["month"].nunique()
    counts["calls_per_month"] = counts["calls"] / counts["month_of_year"].map(months_per_moy)
    return counts


def operational_linkage(assigned: pd.DataFrame, person_months: pd.DataFrame,
                        label_col: str = "true_topic") -> pd.DataFrame:
    """Store-level correlation between topic call rates and their true drivers.

    The check that call volumes are *signal*: scheduling-call rates should
    track store schedule volatility, pay-call rates should track (inversely)
    pay position. Returns per-check correlation across stores.
    """
    pm = person_months[person_months["role"].isin(HOURLY_ROLES)]
    store_state = pm.groupby("store_id", observed=True).agg(
        person_months=("employee_id", "size"),
        volatility=("schedule_volatility", "mean"),
        pay_ratio=("pay_ratio", "mean"))

    checks = []
    for topic, driver, expected in [("scheduling", "volatility", "+"),
                                    ("pay_error", "pay_ratio", "-")]:
        rate = (assigned[assigned[label_col] == topic]
                .groupby("store_id", observed=True).size()
                .reindex(store_state.index).fillna(0)
                / store_state["person_months"] * 1000)
        corr = float(np.corrcoef(rate, store_state[driver])[0, 1])
        checks.append({"topic": topic, "true_driver": driver,
                       "expected_sign": expected,
                       "correlation_across_stores": round(corr, 3)})
    return pd.DataFrame(checks)
