"""Use case 15: exit-interview and pulse-survey NLP.

When someone quits, the exit interview is the company's last and clearest read
on why — and it arrives as free text nobody has time to code by hand. This
module generates synthetic exit comments whose hidden theme is tied to the
*actual hazard driver* that pushed each leaver out (from their state at exit),
then recovers those themes unsupervised and checks the recovery three ways.

It pairs the topic-modelling machinery of use case 11 with the driver model of
use case 4. The extra, harder claim here is alignment: it is not enough for the
topic model to find six clean clusters; the cluster it labels "pay" must be the
one written by the leavers the hazard model actually underpaid. Because both the
themes and the drivers are ground truth, that alignment is measurable.

Three checkable claims:

1. **Recovery** — TF-IDF + NMF (scikit-learn only) recovers the planted themes,
   scored by normalised mutual information and purity, not eyeballed.
2. **Alignment** — leavers assigned to each theme are elevated on that theme's
   true driver (pay-theme leavers are underpaid, schedule-theme leavers have
   volatile schedules), measured as a mean gap versus everyone else.
3. **Regrettability** — themes split by whether the exit was voluntary, so the
   controllable reasons (pay, schedule, management) can be told apart from the
   rest.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .config import HOURLY_ROLES

THEMES = ("pay", "scheduling", "commute", "management", "career", "workload")

# Each theme's driver and the direction that raises its exit-comment odds.
THEME_DRIVERS = {
    "pay": ("pay_ratio", "-"),
    "scheduling": ("schedule_volatility", "+"),
    "commute": ("commute_km", "+"),
    "management": ("months_since_mgr_change", "-"),
    "career": ("months_since_promotion", "+"),
    "workload": ("store_staffing_ratio", "-"),
}


@dataclass
class ExitNLPGroundTruth:
    """Weights turning a leaver's exit-state into exit-theme probabilities.

    Each theme's driver is standardised across the leaver population and, on the
    side that raises exit risk (underpaid, volatile, long commute, recent
    manager change, stagnant, understaffed), enters the theme's score. Scoring
    on the driver's *position* rather than its raw value means even a
    low-variance driver like pay ratio still separates the leavers it pushed
    out. A softmax then picks each leaver's dominant theme; ``baseline`` keeps
    every theme reachable and ``temperature`` sets how sharply the driver wins.
    """

    baseline: float = 0.5
    temperature: float = 1.0
    pay_weight: float = 2.0
    scheduling_weight: float = 2.0
    commute_weight: float = 2.0
    management_weight: float = 2.0
    career_weight: float = 2.0
    workload_weight: float = 2.0

    def as_dict(self) -> dict:
        return asdict(self)


_TEMPLATES = {
    "pay": [
        "the pay just was not enough to cover my bills and other stores nearby pay more per hour",
        "i asked for a raise for months and my wage never moved so i found a better paying job",
        "my hourly rate fell behind the market and the annual increase did not keep up with rent",
        "left for a role that pays two dollars more an hour for the same work, it was about money",
        "could not survive on these wages, the paycheck did not stretch to the end of the month",
        "compensation was the whole reason, my pay rate stagnated while everything got expensive",
    ],
    "scheduling": [
        "my schedule changed every single week and i could never plan childcare or a second job",
        "the shifts were posted too late and got cut without notice so my hours were unpredictable",
        "clopening shifts wore me down, closing then opening with no rest between them",
        "i could not get a stable set of hours and the constant schedule changes made life impossible",
        "the unpredictable roster meant i never knew my income week to week and had to leave",
        "asked for consistent shifts for months, the volatile schedule never improved so i quit",
    ],
    "commute": [
        "the drive to this store was too long and the commute ate my time and gas money",
        "i moved further away and the hour each way on the road was no longer worth it",
        "the location was far from home and the daily travel became impossible to sustain",
        "found a job five minutes from my house, the long commute here was the deciding factor",
        "spending two hours a day driving to and from the store finally burned me out",
        "the distance to work was the problem, the gas and travel time cost too much",
    ],
    "management": [
        "the new store manager changed everything and i did not feel supported on the floor",
        "there was constant conflict with management and my concerns were never taken seriously",
        "after the management change the culture soured and favoritism on shifts got worse",
        "my supervisor was difficult to work with and leadership did nothing when i raised it",
        "the way the new manager treated the team pushed a lot of good people out including me",
        "poor management and no support from the boss made coming to work miserable",
    ],
    "career": [
        "there was no path to promotion here, i was passed over and stopped seeing a future",
        "i stayed years without advancing so i left for a role with actual growth opportunity",
        "no career development or chance to move up, my skills were going nowhere in this job",
        "wanted to grow into a lead role but promotions never came so i moved on to advance",
        "felt stuck at the same level for too long with no path forward or new responsibility",
        "the lack of advancement and learning meant i had to leave to build my career",
    ],
    "workload": [
        "we were chronically understaffed and the workload on a short-handed floor was brutal",
        "burned out from covering shifts because the store never had enough people scheduled",
        "the pace was exhausting with too few staff, every shift was a scramble to keep up",
        "constantly short-staffed so the pressure and stress of doing three jobs got to me",
        "the understaffing meant no breaks and impossible workloads, i could not keep going",
        "too much work for too few people, the store was always short and it wore me out",
    ],
}
_FILLER = [
    "overall i did like my coworkers", "it was a hard decision to leave", "thanks for the chance",
    "i learned a lot here", "wish things had been different", "nothing personal against the team",
    "just had to do what was best for me", "the job had its good days", "no hard feelings",
]


def _directional_z(x: np.ndarray, sign: str) -> np.ndarray:
    """Standardise, orient so the risk side is positive, and keep only that side."""
    sd = x.std()
    z = (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)
    z = z if sign == "+" else -z
    return np.maximum(z, 0.0)


def _theme_scores(pm: pd.DataFrame, gt: ExitNLPGroundTruth) -> np.ndarray:
    """One score column per theme from the leaver's standardised exit-state."""
    weight = {"pay": gt.pay_weight, "scheduling": gt.scheduling_weight,
              "commute": gt.commute_weight, "management": gt.management_weight,
              "career": gt.career_weight, "workload": gt.workload_weight}
    cols = []
    for theme in THEMES:
        driver, sign = THEME_DRIVERS[theme]
        z = _directional_z(pm[driver].to_numpy(dtype=float), sign)
        cols.append(gt.baseline + weight[theme] * z)
    return np.column_stack(cols)


def simulate_exit_comments(person_months: pd.DataFrame,
                           gt: ExitNLPGroundTruth | None = None,
                           voluntary_only: bool = False,
                           seed: int = 47) -> pd.DataFrame:
    """One row per hourly leaver: exit-state drivers, hidden theme, comment."""
    gt = gt or ExitNLPGroundTruth()
    rng = np.random.default_rng(seed)
    pm = person_months[(person_months["terminated"] == 1)
                       & person_months["role"].isin(HOURLY_ROLES)].copy()
    if voluntary_only:
        pm = pm[pm["termination_type"] == "voluntary"]
    pm = pm.reset_index(drop=True)

    scores = _theme_scores(pm, gt) / gt.temperature
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    picks = np.array([rng.choice(len(THEMES), p=probs[i]) for i in range(len(pm))])
    themes = np.array(THEMES)[picks]

    comments = []
    for i, theme in enumerate(themes):
        bank = _TEMPLATES[theme]
        a, b = rng.choice(len(bank), size=2, replace=False)
        sentences = [bank[a], bank[b]]
        # Real leavers often name a secondary reason: 18% of comments borrow a
        # sentence from another theme, which is what keeps recovery honest.
        if rng.random() < 0.18:
            other = rng.choice([t for t in THEMES if t != theme])
            sentences.append(_TEMPLATES[other][rng.integers(len(_TEMPLATES[other]))])
        rng.shuffle(sentences)
        extra = rng.choice(_FILLER, size=rng.integers(0, 2), replace=False)
        words = " ".join(sentences).split()
        keep = rng.random(len(words)) > 0.08
        comments.append(" ".join([w for w, k in zip(words, keep) if k] + list(extra)))

    out = pm[["employee_id", "store_id", "district_id", "month", "termination_type",
              "pay_ratio", "schedule_volatility", "commute_km",
              "months_since_mgr_change", "months_since_promotion",
              "store_staffing_ratio"]].copy()
    out["true_theme"] = themes
    out["comment"] = comments
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


class ExitThemeModel:
    """TF-IDF + NMF theme model for exit comments (scikit-learn only)."""

    def __init__(self, n_themes: int = len(THEMES), random_state: int = 0):
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_features=3000)
        self.nmf = NMF(n_components=n_themes, random_state=random_state,
                       init="nndsvda", max_iter=400)

    def fit(self, comments: pd.DataFrame) -> "ExitThemeModel":
        X = self.vectorizer.fit_transform(comments["comment"])
        self.doc_theme_ = self.nmf.fit_transform(X)
        return self

    def assign(self, comments: pd.DataFrame) -> pd.DataFrame:
        out = comments.copy()
        out["cluster"] = self.doc_theme_.argmax(axis=1)
        return out

    def top_terms(self, n: int = 8) -> pd.DataFrame:
        terms = np.array(self.vectorizer.get_feature_names_out())
        return pd.DataFrame([
            {"cluster": k, "top_terms": ", ".join(terms[np.argsort(-comp)[:n]])}
            for k, comp in enumerate(self.nmf.components_)])

    def evaluate(self, comments: pd.DataFrame) -> dict:
        """NMI + purity of unsupervised clusters against the hidden themes."""
        from sklearn.metrics import normalized_mutual_info_score

        a = self.assign(comments)
        nmi = normalized_mutual_info_score(a["true_theme"], a["cluster"])
        purity = (a.groupby("cluster")["true_theme"]
                  .apply(lambda s: s.value_counts().iloc[0]).sum() / len(a))
        mapping = (a.groupby("cluster")["true_theme"]
                   .agg(lambda s: s.mode().iloc[0]).to_dict())
        return {"n_comments": int(len(a)), "nmi": round(float(nmi), 3),
                "purity": round(float(purity), 3), "cluster_to_theme": mapping}


def theme_driver_alignment(comments: pd.DataFrame,
                           theme_col: str = "true_theme") -> pd.DataFrame:
    """Do leavers on each theme show its true driver, vs everyone else?

    For each theme, the mean of its driver among that theme's leavers minus the
    mean among the rest, signed so a correct alignment is always positive.
    """
    rows = []
    for theme, (driver, sign) in THEME_DRIVERS.items():
        on = comments[comments[theme_col] == theme][driver]
        off = comments[comments[theme_col] != theme][driver]
        gap = float(on.mean() - off.mean())
        directed = gap if sign == "+" else -gap
        rows.append({
            "theme": theme, "driver": driver, "expected_sign": sign,
            "driver_mean_on_theme": round(float(on.mean()), 3),
            "driver_mean_off_theme": round(float(off.mean()), 3),
            "aligned_gap": round(directed, 3),
            "aligned": directed > 0,
        })
    return pd.DataFrame(rows)


def theme_by_regrettability(comments: pd.DataFrame) -> pd.DataFrame:
    """Theme mix split by voluntary vs involuntary exit, share within each."""
    tab = (comments.groupby(["termination_type", "true_theme"], observed=True).size()
           .rename("n").reset_index())
    tab["share"] = tab["n"] / tab.groupby("termination_type")["n"].transform("sum")
    return tab.round(3)
