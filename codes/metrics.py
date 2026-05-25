"""
metrics.py — DMC Coverage Map Metrics

Reads `data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv`
and computes the following metrics:

1) DMC Prevalence
   - % of trials with a DMC overall
   - Breakdown by phase, sponsor type, study type, therapeutic area

2) Oversight Gaps
   - Rule-based DMC expected/indicated flag based on FDA/EMA guidance features
   - DMC reporting gaps among trials where independent monitoring is expected
   - Proportion of Phase III or IV interventional industry trials without a DMC
   - Highlight high-risk therapeutic areas (oncology, cardiovascular, infectious disease)

3) DMC & Trial Outcomes
   - Compare termination/withdrawal/suspension rates between trials with vs without a DMC
   - Convert whyStopped free text into structured termination categories
   - Keep a review queue for cases that need LLM or manual adjudication

Outputs CSVs to the `data/metrics folder and prints concise summaries.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from .dmc_expected import DMC_EXPECTED_REASONS, REASON_LABELS, DMCExpectedClassifier, OpenAIDMCExpectedLLMClient
    from .termination_classifier import TerminationReasonClassifier, build_default_classifier
except ImportError:  # pragma: no cover - supports `python codes/metrics.py`
    from dmc_expected import DMC_EXPECTED_REASONS, REASON_LABELS, DMCExpectedClassifier, OpenAIDMCExpectedLLMClient
    from termination_classifier import TerminationReasonClassifier, build_default_classifier


@dataclass
class Paths:
    """Container for input/output paths."""
    input_csv: str = "data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv"
    out_dir: str = "data/metrics"
    structured_trials_csv: str = "data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended_structured.csv"
    prevalence_overall_csv: str = "data/metrics/metric_prevalence_overall.csv"
    prevalence_phase_csv: str = "data/metrics/metric_prevalence_by_phase.csv"
    prevalence_sponsor_csv: str = "data/metrics/metric_prevalence_by_sponsor.csv"
    prevalence_studytype_csv: str = "data/metrics/metric_prevalence_by_studytype.csv"
    prevalence_ta_csv: str = "data/metrics/metric_prevalence_by_therapeutic_area.csv"
    dmc_expected_vs_actual_csv: str = "data/metrics/metric_dmc_expected_vs_actual.csv"
    dmc_expected_by_reason_csv: str = "data/metrics/metric_dmc_expected_by_reason.csv"
    dmc_expected_by_phase_csv: str = "data/metrics/metric_dmc_expected_gap_by_phase.csv"
    dmc_expected_by_sponsor_csv: str = "data/metrics/metric_dmc_expected_gap_by_sponsor.csv"
    dmc_expected_by_ta_csv: str = "data/metrics/metric_dmc_expected_gap_by_therapeutic_area.csv"
    oversight_gaps_csv: str = "data/metrics/metric_oversight_gaps_phase34_industry_interventional.csv"
    outcomes_by_dmc_csv: str = "data/metrics/metric_outcomes_by_dmc.csv"
    whystopped_by_dmc_csv: str = "data/metrics/metric_whystopped_by_dmc.csv"
    termination_category_by_dmc_csv: str = "data/metrics/metric_termination_category_by_dmc.csv"
    termination_review_queue_csv: str = "data/metrics/termination_reason_review_queue.csv"
    proxy_validation_csv: str = "data/metrics/metric_proxy_validation.csv"
    coverage_index_phase_csv: str = "data/metrics/metric_coverage_index_by_phase.csv"
    coverage_index_sponsor_csv: str = "data/metrics/metric_coverage_index_by_sponsor.csv"
    coverage_index_ta_csv: str = "data/metrics/metric_coverage_index_by_ta.csv"
    logistic_regression_csv: str = "data/metrics/metric_logistic_regression.csv"


class DMCMetrics:
    """
    Compute DMC coverage and oversight metrics from a prepared CTG dataset.

    Expected columns in input CSV:
        - nct_id (unique)
        - has_dmc (boolean or string)
        - phase (str; e.g., "Phase 3" or "Phase 3/4")
        - study_type (str; 'Interventional'/'Observational')
        - sponsor_class (str; 'INDUSTRY','NIH','OTHER','NETWORK','ACADEMIC' etc.)
        - overall_status (str; 'COMPLETED','TERMINATED','WITHDRAWN','SUSPENDED')
        - why_stopped (str or NaN)
        - start_date (str ISO)
        - start_year (int)
        - primary_completion_date (str)
        - conditions (list-like or stringified list)
        - keywords (list-like or stringified list)
        - intervention_types (list-like or stringified list)
        - primary_outcomes / secondary_outcomes (list-like or stringified list)
        - therapeutic_area (str)

    Notes:
        - has_dmc is coerced to boolean dtype ('boolean') with NA as <NA>.
        - sponsor_class normalized to upper case; common buckets derived.
        - phase is parsed to detect presence of Phase 3 or Phase 4.
    """

    # Keywords for whyStopped classification
    SAFETY_KEYS = [
        "safety", "adverse event", "adverse events", "toxicity",
        "death", "mortality", "side effect", "risk", "harm", "serious adverse"
    ]
    FUTILITY_KEYS = [
        "futility", "lack of efficacy", "no efficacy", "ineffective",
        "conditional power", "no benefit", "insufficient efficacy", "interim analysis futility"
    ]

    # TA labels to highlight as high-risk areas
    HIGH_RISK_TA = {"Oncology", "Cardiovascular", "Infectious Disease"}

    TERMINATION_CONTEXT_COLUMNS: Tuple[str, ...] = (
        "brief_title",
        "official_title",
        "conditions",
        "intervention_names",
        "primary_outcomes",
        "secondary_outcomes",
    )

    def __init__(
        self,
        paths: Optional[Paths] = None,
        dmc_expected_classifier: Optional[DMCExpectedClassifier] = None,
        termination_classifier: Optional[TerminationReasonClassifier] = None,
    ) -> None:
        self.paths = paths or Paths()
        self.dmc_expected_classifier = dmc_expected_classifier or DMCExpectedClassifier()
        self.termination_classifier = termination_classifier or TerminationReasonClassifier()
        self.df: pd.DataFrame = pd.DataFrame()


    def load(self) -> None:
        """Load and normalize the input dataset.

        If a cached structured CSV exists that is at least as new as the raw
        input CSV, it is loaded directly and classification is skipped.
        Delete the structured CSV to force re-classification (e.g. after rule
        changes or enabling/disabling the LLM).
        """
        if not os.path.exists(self.paths.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {self.paths.input_csv}")

        # ── Cache hit: skip classification entirely ───────────────────────────
        struct_path = self.paths.structured_trials_csv
        if (
            os.path.exists(struct_path)
            and os.path.getmtime(struct_path) >= os.path.getmtime(self.paths.input_csv)
        ):
            df = pd.read_csv(struct_path, low_memory=False)
            if "has_dmc" in df.columns:
                df["has_dmc"] = (
                    df["has_dmc"]
                    .map(lambda x: np.nan if pd.isna(x) else str(x).strip().lower())
                    .map({"true": True, "false": False})
                    .astype("boolean")
                )
            self.df = df
            return

        df = pd.read_csv(self.paths.input_csv, low_memory=False)

        # Ensure uniqueness by NCT ID
        if "nct_id" not in df.columns:
            raise ValueError("Column 'nct_id' is required.")
        df = df.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)

        # Coerce has_dmc
        if "has_dmc" not in df.columns:
            raise ValueError("Column 'has_dmc' is required.")
        df["has_dmc"] = (
            df["has_dmc"]
            .map(lambda x: np.nan if pd.isna(x) else str(x).strip().lower())
            .map({"true": True, "false": False})
            .astype("boolean")
        )

        # Normalize sponsor class
        if "sponsor_class" not in df.columns and "lead_sponsor_class" in df.columns:
            df["sponsor_class"] = df["lead_sponsor_class"]
        if "sponsor_class" not in df.columns:
            raise ValueError("Column 'sponsor_class' is required.")
        df["sponsor_class"] = df["sponsor_class"].astype(str).str.upper().str.strip()
        df["sponsor_bucket"] = df["sponsor_class"].replace({
            "INDUSTRY": "Industry",
            "NIH": "NIH",
            "OTHER": "Other",
            "ACADEMIC": "Academic",
            "NETWORK": "Academic",  # common mapping
        })
        df.loc[~df["sponsor_bucket"].isin(["Industry", "NIH", "Academic", "Other"]), "sponsor_bucket"] = "Other"

        # Study type
        if "study_type" not in df.columns:
            raise ValueError("Column 'study_type' is required.")
        df["study_type"] = df["study_type"].astype(str).str.title().str.strip()

        # Therapeutic area
        if "therapeutic_area" not in df.columns:
            df["therapeutic_area"] = "Other/Unclassified"

        # Phase flags
        df["is_phase3"] = df["phase"].fillna("").str.contains(r"\bPhase 3\b", case=False, regex=True)
        df["is_phase4"] = df["phase"].fillna("").str.contains(r"\bPhase 4\b", case=False, regex=True)
        df["is_phase34"] = df["is_phase3"] | df["is_phase4"]

        # Outcomes grouping for “trial outcomes”
        if "overall_status" not in df.columns:
            raise ValueError("Column 'overall_status' is required.")
        df["overall_status"] = df["overall_status"].astype(str).str.upper().str.strip()

        df["is_terminated_like"] = df["overall_status"].isin(["TERMINATED", "WITHDRAWN", "SUSPENDED"])
        df["is_completed"] = df["overall_status"].eq("COMPLETED")

        # why_stopped normalized text. Keep original casing for the classifier and
        # a lower-case view for legacy keyword metrics.
        if "why_stopped" not in df.columns:
            df["why_stopped"] = ""
        df["why_stopped_text"] = df["why_stopped"].fillna("").astype(str).str.strip()
        df.loc[df["why_stopped_text"].str.lower().isin(["nan", "none", "null"]), "why_stopped_text"] = ""
        df["why_stopped_text_lc"] = df["why_stopped_text"].str.lower()

        df = self._add_dmc_expected_classification(df)
        df = self._add_termination_classification(df)

        # Persist
        self.df = df

        # Save structured CSV immediately so subsequent runs load from cache.
        os.makedirs(self.paths.out_dir, exist_ok=True)
        self.df.to_csv(struct_path, index=False)

    def _add_dmc_expected_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add auditable DMC-expected labels using vectorised batch classification."""
        expected_df = self.dmc_expected_classifier.classify_dataframe(df)
        return pd.concat([df, expected_df], axis=1)

    def _add_termination_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised termination classification with LLM fallback for ambiguous rows."""
        class_df = self.termination_classifier.classify_dataframe(
            df["why_stopped_text"], df=df
        )
        return pd.concat([df, class_df], axis=1)

    def compute_and_save_all(self) -> None:
        """Compute all metrics and save CSVs."""
        os.makedirs(self.paths.out_dir, exist_ok=True)

        self.df.to_csv(self.paths.structured_trials_csv, index=False)

        prev_overall = self._metric_prevalence_overall()
        prev_overall.to_csv(self.paths.prevalence_overall_csv, index=False)

        prev_phase = self._metric_prevalence_by("phase", normalize=False)
        prev_phase.to_csv(self.paths.prevalence_phase_csv, index=False)

        prev_sponsor = self._metric_prevalence_by("sponsor_bucket", normalize=False)
        prev_sponsor.to_csv(self.paths.prevalence_sponsor_csv, index=False)

        prev_studytype = self._metric_prevalence_by("study_type", normalize=False)
        prev_studytype.to_csv(self.paths.prevalence_studytype_csv, index=False)

        prev_ta = self._metric_prevalence_by("therapeutic_area", normalize=False)
        prev_ta.to_csv(self.paths.prevalence_ta_csv, index=False)

        dmc_expected_vs_actual = self._metric_dmc_expected_vs_actual()
        dmc_expected_vs_actual.to_csv(self.paths.dmc_expected_vs_actual_csv, index=False)

        dmc_expected_by_reason = self._metric_dmc_expected_by_reason()
        dmc_expected_by_reason.to_csv(self.paths.dmc_expected_by_reason_csv, index=False)

        dmc_expected_by_phase = self._metric_dmc_expected_gap_by("phase")
        dmc_expected_by_phase.to_csv(self.paths.dmc_expected_by_phase_csv, index=False)

        dmc_expected_by_sponsor = self._metric_dmc_expected_gap_by("sponsor_bucket")
        dmc_expected_by_sponsor.to_csv(self.paths.dmc_expected_by_sponsor_csv, index=False)

        dmc_expected_by_ta = self._metric_dmc_expected_gap_by("therapeutic_area")
        dmc_expected_by_ta.to_csv(self.paths.dmc_expected_by_ta_csv, index=False)

        gaps = self._metric_oversight_gaps_phase34_industry_interventional()
        gaps.to_csv(self.paths.oversight_gaps_csv, index=False)

        outcomes = self._metric_outcomes_by_dmc()
        outcomes.to_csv(self.paths.outcomes_by_dmc_csv, index=False)

        whystop_keywords = self._metric_whystopped_keywords_by_dmc()
        whystop_keywords.to_csv(self.paths.whystopped_by_dmc_csv, index=False)

        termination_categories = self._metric_termination_categories_by_dmc()
        termination_categories.to_csv(self.paths.termination_category_by_dmc_csv, index=False)

        review_queue = self._termination_review_queue()
        review_queue.to_csv(self.paths.termination_review_queue_csv, index=False)

        proxy_validation = self._metric_proxy_validation()
        proxy_validation.to_csv(self.paths.proxy_validation_csv, index=False)

        self._metric_dmc_coverage_index()

        logistic_results = self._metric_logistic_regression()
        logistic_results.to_csv(self.paths.logistic_regression_csv, index=False)

        # Load coverage-index-by-TA for summary printing
        coverage_index_ta = self._compute_coverage_index("therapeutic_area")

        # Print concise summaries
        self._print_summaries(
            prev_overall,
            dmc_expected_vs_actual,
            gaps,
            outcomes,
            termination_categories,
            proxy_validation,
            coverage_index_ta,
            logistic_results,
        )

    # Metrics

    def _metric_prevalence_overall(self) -> pd.DataFrame:
        """
        Overall prevalence of DMC presence.
        Returns columns: total_trials, with_dmc, without_dmc, with_dmc_pct
        """
        df = self.df.copy()
        total = len(df)
        with_dmc = int(df["has_dmc"].fillna(False).sum())
        without_dmc = int((~df["has_dmc"].fillna(False)).sum())
        with_dmc_pct = (with_dmc / total * 100.0) if total else 0.0

        out = pd.DataFrame([{
            "total_trials": total,
            "with_dmc": with_dmc,
            "without_dmc": without_dmc,
            "with_dmc_pct": round(with_dmc_pct, 2),
        }])
        return out

    def _metric_prevalence_by(self, by: str, normalize: bool = False) -> pd.DataFrame:
        """
        Prevalence of DMC by a category column.

        Parameters
        ----------
        by : str
            Column to group by (e.g., 'phase','sponsor_bucket','study_type','therapeutic_area').
        normalize : bool
            If True, include percentages per group.

        Returns
        -------
        DataFrame with columns: [by, total, with_dmc, without_dmc, with_dmc_pct]
        """
        if by not in self.df.columns:
            raise ValueError(f"Column '{by}' not in dataframe.")

        g = self.df.groupby(by, dropna=False, observed=True)
        total = g.size().rename("total")
        with_dmc = g["has_dmc"].apply(lambda s: s.fillna(False).sum()).rename("with_dmc")
        without_dmc = total - with_dmc
        res = pd.concat([total, with_dmc, without_dmc], axis=1).reset_index()
        if normalize:
            res["with_dmc_pct"] = np.where(
                res["total"] > 0, (res["with_dmc"] / res["total"]) * 100.0, 0.0
            )
        else:
            res["with_dmc_pct"] = np.where(
                res["total"] > 0, (res["with_dmc"] / res["total"]) * 100.0, 0.0
            )
        res["with_dmc_pct"] = res["with_dmc_pct"].round(2)
        return res.sort_values(by=["with_dmc_pct", "total"], ascending=[False, False]).reset_index(drop=True)

    def _metric_dmc_expected_vs_actual(self) -> pd.DataFrame:
        """
        Compare rule-based DMC expectation against reported DMC presence.

        Returns one row for trials matching the DMC-expected rule and one row
        for trials that do not. The policy-relevant count is the expected row's
        `without_dmc` value.
        """
        df = self.df.copy()
        df["dmc_expected_value"] = df["dmc_expected"].fillna(False).astype(bool)

        grp = (
            df.groupby("dmc_expected_value", dropna=False)
            .agg(
                total_trials=("nct_id", "count"),
                with_dmc=("has_dmc", lambda s: int(s.fillna(False).sum())),
            )
            .reset_index()
        )
        grp["without_dmc"] = grp["total_trials"] - grp["with_dmc"]
        grp["with_dmc_pct"] = np.where(
            grp["total_trials"] > 0, grp["with_dmc"] / grp["total_trials"] * 100.0, 0.0
        ).round(2)
        grp["without_dmc_pct"] = np.where(
            grp["total_trials"] > 0, grp["without_dmc"] / grp["total_trials"] * 100.0, 0.0
        ).round(2)
        grp["dmc_expected_label"] = grp["dmc_expected_value"].map({
            True: "DMC indicated",
            False: "DMC not indicated",
        })
        grp["_sort"] = grp["dmc_expected_value"].map({True: 0, False: 1})
        out_cols = [
            "dmc_expected_label",
            "total_trials",
            "with_dmc",
            "without_dmc",
            "with_dmc_pct",
            "without_dmc_pct",
        ]
        return grp.sort_values("_sort").reset_index(drop=True)[out_cols]

    def _metric_dmc_expected_by_reason(self) -> pd.DataFrame:
        """
        Count which DMC-expected criteria were triggered and the DMC gap within
        each criterion. Trials can appear in more than one reason row.
        """
        df = self.df.copy()
        expected = df["dmc_expected"].fillna(False).astype(bool)
        total_expected = int(expected.sum())
        reason_to_column = {
            "late_phase": "dmc_expected_late_phase",
            "serious_or_life_threatening_condition": "dmc_expected_serious_condition",
            "vulnerable_age_population": "dmc_expected_vulnerable_population",
            "large_enrollment": "dmc_expected_large_enrollment",
            "regulated_intervention": "dmc_expected_regulated_intervention",
            "long_duration": "dmc_expected_long_duration",
            "serious_endpoint": "dmc_expected_serious_endpoint",
        }

        rows: List[Dict[str, Any]] = []
        for reason in DMC_EXPECTED_REASONS:
            flag_col = reason_to_column[reason]
            mask = expected & df[flag_col].fillna(False).astype(bool)
            sub = df.loc[mask]
            total = len(sub)
            with_dmc = int(sub["has_dmc"].fillna(False).sum())
            without_dmc = total - with_dmc
            rows.append({
                "dmc_expected_reason": reason,
                "dmc_expected_reason_label": REASON_LABELS[reason],
                "total_expected_with_reason": total,
                "share_of_expected_pct": round((total / total_expected * 100.0) if total_expected else 0.0, 2),
                "with_dmc": with_dmc,
                "without_dmc": without_dmc,
                "with_dmc_pct": round((with_dmc / total * 100.0) if total else 0.0, 2),
                "without_dmc_pct": round((without_dmc / total * 100.0) if total else 0.0, 2),
            })

        return (
            pd.DataFrame(rows)
            .sort_values(["without_dmc", "total_expected_with_reason"], ascending=[False, False])
            .reset_index(drop=True)
        )

    def _metric_dmc_expected_gap_by(self, by: str) -> pd.DataFrame:
        """
        DMC reporting gap among only DMC-expected trials, grouped by a category.
        """
        if by not in self.df.columns:
            raise ValueError(f"Column '{by}' not in dataframe.")

        df = self.df.loc[self.df["dmc_expected"].fillna(False).astype(bool)].copy()
        if df.empty:
            return pd.DataFrame(
                columns=[
                    by,
                    "total_expected",
                    "expected_with_dmc",
                    "expected_without_dmc",
                    "expected_with_dmc_pct",
                    "expected_without_dmc_pct",
                ]
            )

        df[by] = df[by].fillna("Missing").replace("", "Missing")
        grp = (
            df.groupby(by, dropna=False, observed=True)
            .agg(
                total_expected=("nct_id", "count"),
                expected_with_dmc=("has_dmc", lambda s: int(s.fillna(False).sum())),
            )
            .reset_index()
        )
        grp["expected_without_dmc"] = grp["total_expected"] - grp["expected_with_dmc"]
        grp["expected_with_dmc_pct"] = np.where(
            grp["total_expected"] > 0,
            grp["expected_with_dmc"] / grp["total_expected"] * 100.0,
            0.0,
        ).round(2)
        grp["expected_without_dmc_pct"] = np.where(
            grp["total_expected"] > 0,
            grp["expected_without_dmc"] / grp["total_expected"] * 100.0,
            0.0,
        ).round(2)
        return (
            grp.sort_values(["expected_without_dmc_pct", "total_expected"], ascending=[False, False])
            .reset_index(drop=True)
        )

    def _metric_oversight_gaps_phase34_industry_interventional(self) -> pd.DataFrame:
        """
        Compute proportion of Phase III or IV Interventional Industry trials without a DMC
        and highlight key therapeutic areas.
        Returns columns:
            subgroup, total, without_dmc, without_dmc_pct
        """
        df = self.df.copy()
        mask = (
            df["is_phase34"]
            & df["study_type"].eq("Interventional")
            & df["sponsor_bucket"].eq("Industry")
        )
        sub = df.loc[mask].copy()

        rows: List[Dict] = []

        # Overall Phase III/IV Interventional Industry
        total = len(sub)
        without = int((~sub["has_dmc"].fillna(False)).sum())
        rows.append({
            "subgroup": "All Phase III/IV • Interventional • Industry",
            "total": total,
            "without_dmc": without,
            "without_dmc_pct": round((without / total * 100.0) if total else 0.0, 2),
        })

        # High-risk therapeutic areas
        ta_counts = []
        for ta in sorted(self.HIGH_RISK_TA):
            sub_ta = sub.loc[sub["therapeutic_area"].eq(ta)]
            t = len(sub_ta)
            w = int((~sub_ta["has_dmc"].fillna(False)).sum())
            ta_counts.append((ta, t, w))

        for ta, t, w in ta_counts:
            rows.append({
                "subgroup": f"{ta} • Phase III/IV • Interventional • Industry",
                "total": t,
                "without_dmc": w,
                "without_dmc_pct": round((w / t * 100.0) if t else 0.0, 2),
            })

        return pd.DataFrame(rows)

    def _metric_outcomes_by_dmc(self) -> pd.DataFrame:
        """
        Compare termination-like rates between trials with vs without a DMC.
        'termination-like' includes TERMINATED/WITHDRAWN/SUSPENDED vs COMPLETED.

        Returns columns:
            has_dmc_label, total, completed, terminated, withdrawn, suspended,
            terminated_like, terminated_like_pct
        """
        df = self.df.copy()
        dummies = pd.get_dummies(df["overall_status"], prefix="status")
        df = pd.concat([df, dummies], axis=1)
        for col in ["status_COMPLETED", "status_TERMINATED", "status_WITHDRAWN", "status_SUSPENDED"]:
            if col not in df.columns:
                df[col] = 0

        grp = (
            df.groupby(df["has_dmc"].fillna(False))
            .agg(
                total=("nct_id", "count"),
                completed=("status_COMPLETED", "sum"),
                terminated=("status_TERMINATED", "sum"),
                withdrawn=("status_WITHDRAWN", "sum"),
                suspended=("status_SUSPENDED", "sum"),
                terminated_like=("is_terminated_like", "sum"),
            )
            .reset_index()
        )

        grp["has_dmc_label"] = grp["has_dmc"].map({True: "With DMC", False: "No DMC"})
        grp["terminated_like_pct"] = np.where(
            grp["total"] > 0, grp["terminated_like"] / grp["total"] * 100.0, 0.0
        ).round(2)
        for status_col in ["completed", "terminated", "withdrawn", "suspended"]:
            grp[f"{status_col}_pct"] = np.where(
                grp["total"] > 0, grp[status_col] / grp["total"] * 100.0, 0.0
            ).round(2)
        out_cols = [
            "has_dmc_label",
            "total",
            "completed",
            "terminated",
            "withdrawn",
            "suspended",
            "terminated_like",
            "completed_pct",
            "terminated_pct",
            "withdrawn_pct",
            "suspended_pct",
            "terminated_like_pct",
        ]
        return grp[out_cols].sort_values("terminated_like_pct", ascending=False).reset_index(drop=True)

    def _metric_whystopped_keywords_by_dmc(self) -> pd.DataFrame:
        """
        Analyze whyStopped text: prevalence of safety/futility keywords by DMC status.

        Returns columns:
            has_dmc_label, n_with_text, safety_hits, futility_hits,
            safety_rate_pct, futility_rate_pct
        """
        df = self.df.copy()
        df["has_text"] = df["why_stopped_text"].str.len().fillna(0) > 0
        df_text = df.loc[df["has_text"]].copy()

        def contains_any(text: str, keys: Iterable[str]) -> bool:
            if not isinstance(text, str) or not text:
                return False
            t = text.lower()
            return any(k in t for k in keys)

        df_text["safety_hit"] = df_text["why_stopped_text"].apply(lambda t: contains_any(t, self.SAFETY_KEYS))
        df_text["futility_hit"] = df_text["why_stopped_text"].apply(lambda t: contains_any(t, self.FUTILITY_KEYS))

        grp = df_text.groupby(df_text["has_dmc"].fillna(False)).agg(
            n_with_text=("nct_id", "count"),
            safety_hits=("safety_hit", "sum"),
            futility_hits=("futility_hit", "sum"),
        ).reset_index()

        grp["has_dmc_label"] = grp["has_dmc"].map({True: "With DMC", False: "No DMC"})
        grp["safety_rate_pct"] = np.where(grp["n_with_text"] > 0, grp["safety_hits"] / grp["n_with_text"] * 100.0, 0.0)
        grp["futility_rate_pct"] = np.where(grp["n_with_text"] > 0, grp["futility_hits"] / grp["n_with_text"] * 100.0, 0.0)
        grp["safety_rate_pct"] = grp["safety_rate_pct"].round(2)
        grp["futility_rate_pct"] = grp["futility_rate_pct"].round(2)

        out_cols = ["has_dmc_label", "n_with_text", "safety_hits", "futility_hits", "safety_rate_pct", "futility_rate_pct"]
        return grp[out_cols].sort_values("safety_rate_pct", ascending=False).reset_index(drop=True)

    def _metric_termination_categories_by_dmc(self) -> pd.DataFrame:
        """
        Analyze structured termination categories by DMC status.

        Only stopped trials with non-empty whyStopped text are included in the
        denominator. This keeps completed trials from overwhelming the category
        distribution with missing reasons.
        """
        df = self.df.copy()
        df["has_text"] = df["why_stopped_text"].str.len().fillna(0) > 0
        df_text = df.loc[df["is_terminated_like"] & df["has_text"]].copy()

        if df_text.empty:
            return pd.DataFrame(
                columns=[
                    "has_dmc_label",
                    "termination_category",
                    "n_trials",
                    "category_rate_pct",
                    "avg_confidence",
                    "needs_llm_count",
                    "llm_classified_count",
                    "rule_classified_count",
                ]
            )

        totals = df_text.groupby(df_text["has_dmc"].fillna(False)).size().rename("group_total")
        grp = (
            df_text.groupby([df_text["has_dmc"].fillna(False), "termination_category"], observed=True)
            .agg(
                n_trials=("nct_id", "count"),
                avg_confidence=("termination_confidence", "mean"),
                needs_llm_count=("termination_needs_llm", "sum"),
                llm_classified_count=(
                    "termination_classification_method",
                    lambda s: int((s == "llm").sum()),
                ),
                rule_classified_count=(
                    "termination_classification_method",
                    lambda s: int((s == "rule").sum()),
                ),
            )
            .reset_index()
            .rename(columns={"has_dmc": "has_dmc_value"})
        )
        grp = grp.merge(totals.reset_index().rename(columns={"has_dmc": "has_dmc_value"}), on="has_dmc_value")
        grp["has_dmc_label"] = grp["has_dmc_value"].map({True: "With DMC", False: "No DMC"})
        grp["category_rate_pct"] = np.where(
            grp["group_total"] > 0, grp["n_trials"] / grp["group_total"] * 100.0, 0.0
        ).round(2)
        grp["avg_confidence"] = grp["avg_confidence"].round(3)

        category_order = {
            "safety": 0,
            "efficacy": 1,
            "futility": 2,
            "recruitment_failure": 3,
            "administrative": 4,
            "business_decision": 5,
            "mixed_reasons": 6,
            "uncertain": 7,
            "unclear": 8,
        }
        grp["_category_order"] = grp["termination_category"].map(category_order).fillna(99)
        out_cols = [
            "has_dmc_label",
            "termination_category",
            "n_trials",
            "category_rate_pct",
            "avg_confidence",
            "needs_llm_count",
            "llm_classified_count",
            "rule_classified_count",
        ]
        return (
            grp.sort_values(["has_dmc_label", "_category_order", "n_trials"], ascending=[True, True, False])
            .reset_index(drop=True)[out_cols]
        )

    # ── New publication-quality metrics ─────────────────────────────────────

    def _metric_proxy_validation(self) -> pd.DataFrame:
        """
        Validate the DMC-indicated proxy by comparing risk characteristics
        between DMC-indicated and not-indicated interventional trials.

        Returns one row per characteristic with values for each group.
        """
        df = self.df.loc[self.df["is_interventional_trial"].fillna(False).astype(bool)].copy()
        ind = df.loc[df["dmc_expected"].fillna(False).astype(bool)]
        not_ind = df.loc[~df["dmc_expected"].fillna(False).astype(bool)]

        enroll = pd.to_numeric(df["enrollment"], errors="coerce")
        enroll_ind = pd.to_numeric(ind["enrollment"], errors="coerce")
        enroll_not = pd.to_numeric(not_ind["enrollment"], errors="coerce")

        dur_ind = pd.to_numeric(ind["trial_duration_months"], errors="coerce").dropna()
        dur_not = pd.to_numeric(not_ind["trial_duration_months"], errors="coerce").dropna()

        def pct(series: pd.Series, col: str) -> float:
            vals = series[col].fillna(False).astype(bool)
            return round(vals.mean() * 100.0, 2) if len(vals) else 0.0

        characteristics = [
            ("pct_late_phase",              pct(ind, "dmc_expected_late_phase"),              pct(not_ind, "dmc_expected_late_phase")),
            ("pct_serious_condition",       pct(ind, "dmc_expected_serious_condition"),       pct(not_ind, "dmc_expected_serious_condition")),
            ("pct_serious_endpoint",        pct(ind, "dmc_expected_serious_endpoint"),        pct(not_ind, "dmc_expected_serious_endpoint")),
            ("pct_vulnerable_population",   pct(ind, "dmc_expected_vulnerable_population"),   pct(not_ind, "dmc_expected_vulnerable_population")),
            ("pct_large_enrollment",        pct(ind, "dmc_expected_large_enrollment"),        pct(not_ind, "dmc_expected_large_enrollment")),
            ("pct_long_duration",           pct(ind, "dmc_expected_long_duration"),           pct(not_ind, "dmc_expected_long_duration")),
            ("pct_regulated_intervention",  pct(ind, "dmc_expected_regulated_intervention"),  pct(not_ind, "dmc_expected_regulated_intervention")),
            ("median_enrollment",           round(enroll_ind.median(), 1),                    round(enroll_not.median(), 1)),
            ("mean_duration_months",        round(dur_ind.mean(), 1),                         round(dur_not.mean(), 1)),
            ("n_trials",                    len(ind),                                         len(not_ind)),
        ]
        rows = []
        for char, ind_val, not_val in characteristics:
            if isinstance(ind_val, float) and isinstance(not_val, float) and not_val != 0:
                diff = round(ind_val - not_val, 2)
            else:
                diff = None
            rows.append({
                "characteristic": char,
                "dmc_indicated": ind_val,
                "not_indicated": not_val,
                "difference": diff,
            })
        return pd.DataFrame(rows)

    def _compute_coverage_index(self, by: str) -> pd.DataFrame:
        """
        Compute DMC coverage index per subgroup among DMC-indicated trials.

        coverage_index = (observed_with_dmc_pct) / reference_rate
        where reference_rate is the overall % with DMC among all DMC-indicated.
        """
        df_ind = self.df.loc[self.df["dmc_expected"].fillna(False).astype(bool)]
        total_indicated = len(df_ind)
        total_with_dmc = int(df_ind["has_dmc"].fillna(False).sum())
        reference_rate = (total_with_dmc / total_indicated * 100.0) if total_indicated else 0.0

        gap = self._metric_dmc_expected_gap_by(by)
        gap["reference_rate"] = round(reference_rate, 2)
        gap["coverage_index"] = np.where(
            reference_rate > 0,
            (gap["expected_with_dmc_pct"] / reference_rate).round(3),
            np.nan,
        )
        return gap

    def _metric_dmc_coverage_index(self) -> None:
        """Compute and save DMC Coverage Index by phase, sponsor, and therapeutic area."""
        for dim, path in (
            ("phase",             self.paths.coverage_index_phase_csv),
            ("sponsor_bucket",    self.paths.coverage_index_sponsor_csv),
            ("therapeutic_area",  self.paths.coverage_index_ta_csv),
        ):
            tbl = self._compute_coverage_index(dim)
            tbl.to_csv(path, index=False)

    def _metric_logistic_regression(self) -> pd.DataFrame:
        """
        Primary, sensitivity, component-based, and optional mixed-effects logistic
        models. All restricted to interventional trials.

        Rationale: DMC-indicated status (dmc_expected) is a composite of trial-level
        risk features. Including both the composite and its components in one model
        would introduce structural overlap and over-adjustment. The primary model is
        therefore parsimonious, adjusting only for sponsor type and calendar year.
        Sensitivity analyses re-run the same parsimonious model under broader and
        stricter DMC-indicated definitions. The component-based model replaces the
        composite with its seven underlying risk features for mechanistic insight.

        Returns a DataFrame with columns:
            model, predictor, coef, odds_ratio, ci_lower_95, ci_upper_95, p_value, n

        model tags
        ----------
        primary            has_dmc ~ dmc_expected + sponsor + year
        sensitivity_broad  same formula, broad DMC-indicated definition
        sensitivity_strict same formula, strict DMC-indicated definition
        component_based    composite replaced by 7 individual risk features
        mixed_effects_gee  primary formula + TA clustering via GEE
        """
        _EMPTY = pd.DataFrame(
            columns=["model", "predictor", "coef", "odds_ratio", "ci_lower_95", "ci_upper_95", "p_value", "n"]
        )
        if not _HAS_STATSMODELS:
            return _EMPTY

        df = self.df.loc[self.df["is_interventional_trial"].fillna(False).astype(bool)].copy()

        df["has_dmc_bin"] = df["has_dmc"].fillna(False).astype(int)
        df["dmc_expected_bin"] = df["dmc_expected"].fillna(False).astype(int)
        df["dmc_expected_broad_bin"] = df["dmc_expected_broad"].fillna(False).astype(int)
        if "dmc_expected_strict" in df.columns:
            df["dmc_expected_strict_bin"] = df["dmc_expected_strict"].fillna(False).astype(int)

        median_year = pd.to_numeric(df["start_year"], errors="coerce").median()
        df["start_year_centered"] = pd.to_numeric(df["start_year"], errors="coerce") - median_year
        df["sponsor_bucket"] = df["sponsor_bucket"].fillna("Other").astype(str)

        # Individual risk-feature flags as binary integers for the component-based model
        _COMPONENT_COLS: Dict[str, str] = {
            "comp_late_phase":             "dmc_expected_late_phase",
            "comp_large_enrollment":       "dmc_expected_large_enrollment",
            "comp_regulated_intervention": "dmc_expected_regulated_intervention",
            "comp_long_duration":          "dmc_expected_long_duration",
            "comp_serious_condition":      "dmc_expected_serious_condition",
            "comp_vulnerable_population":  "dmc_expected_vulnerable_population",
            "comp_serious_endpoint":       "dmc_expected_serious_endpoint",
        }
        for bin_col, src_col in _COMPONENT_COLS.items():
            if src_col in df.columns:
                df[bin_col] = df[src_col].fillna(False).astype(int)

        # Sponsor: Industry is the reference category throughout
        adj = "C(sponsor_bucket, Treatment('Industry')) + start_year_centered"
        all_rows: List[Dict[str, Any]] = []

        def _extract(model_name: str, result: Any) -> None:
            ci = result.conf_int()
            n = int(result.nobs)
            for pred in result.params.index:
                coef = float(result.params[pred])
                all_rows.append({
                    "model": model_name,
                    "predictor": pred,
                    "coef": round(coef, 4),
                    "odds_ratio": round(math.exp(coef), 4),
                    "ci_lower_95": round(math.exp(float(ci.loc[pred, 0])), 4),
                    "ci_upper_95": round(math.exp(float(ci.loc[pred, 1])), 4),
                    "p_value": round(float(result.pvalues[pred]), 4),
                    "n": n,
                })

        def _fit(model_name: str, formula: str, extra_na_cols: List[str]) -> None:
            cols = ["has_dmc_bin", "start_year_centered"] + [
                c for c in extra_na_cols if c in df.columns
            ]
            sub = df.dropna(subset=cols)
            sub = sub.loc[sub["sponsor_bucket"].notna()]
            if len(sub) < 10:
                return
            try:
                result = smf.logit(formula, data=sub).fit(disp=0, maxiter=200)
                _extract(model_name, result)
            except Exception:
                pass

        # 1. Primary model — parsimonious: composite + sponsor + year only
        _fit("primary", f"has_dmc_bin ~ dmc_expected_bin + {adj}", ["dmc_expected_bin"])

        # 2. Sensitivity: broad definition (any single guidance feature)
        _fit("sensitivity_broad", f"has_dmc_bin ~ dmc_expected_broad_bin + {adj}", ["dmc_expected_broad_bin"])

        # 3. Sensitivity: strict definition (tier1 AND tier2/3 required)
        if "dmc_expected_strict_bin" in df.columns:
            _fit("sensitivity_strict", f"has_dmc_bin ~ dmc_expected_strict_bin + {adj}", ["dmc_expected_strict_bin"])

        # 4. Component-based model — 7 individual risk features replace the composite
        comp_cols = [c for c in _COMPONENT_COLS if c in df.columns]
        if comp_cols:
            _fit("component_based", f"has_dmc_bin ~ {' + '.join(comp_cols)} + {adj}", comp_cols)

        # 5. Mixed-effects — GEE with therapeutic area as the clustering unit
        all_rows.extend(self._fit_mixed_effects_logistic(df, adj))

        return pd.DataFrame(all_rows) if all_rows else _EMPTY

    def _fit_mixed_effects_logistic(
        self,
        df: pd.DataFrame,
        adj_formula: str,
    ) -> List[Dict[str, Any]]:
        """
        GEE logistic regression grouped by therapeutic area (exchangeable correlation).

        This is a population-average approximation of a mixed-effects logistic model.
        It accounts for within-domain clustering in the outcome without requiring a
        classical GLMM implementation. Returns an empty list if fitting fails.
        """
        if not _HAS_STATSMODELS:
            return []

        required = ["has_dmc_bin", "dmc_expected_bin", "start_year_centered",
                    "sponsor_bucket", "therapeutic_area"]
        sub = df.dropna(subset=[c for c in required if c in df.columns])
        if len(sub) < 20 or sub["therapeutic_area"].nunique() < 2:
            return []

        formula = f"has_dmc_bin ~ dmc_expected_bin + {adj_formula}"
        try:
            from statsmodels.genmod.generalized_estimating_equations import GEE
            from statsmodels.genmod.families import Binomial
            from statsmodels.genmod.cov_struct import Exchangeable

            result = GEE.from_formula(
                formula,
                groups="therapeutic_area",
                data=sub,
                family=Binomial(),
                cov_struct=Exchangeable(),
            ).fit(disp=False)

            ci = result.conf_int()
            rows: List[Dict[str, Any]] = []
            for pred in result.params.index:
                coef = float(result.params[pred])
                rows.append({
                    "model": "mixed_effects_gee",
                    "predictor": pred,
                    "coef": round(coef, 4),
                    "odds_ratio": round(math.exp(coef), 4),
                    "ci_lower_95": round(math.exp(float(ci.loc[pred, 0])), 4),
                    "ci_upper_95": round(math.exp(float(ci.loc[pred, 1])), 4),
                    "p_value": round(float(result.pvalues[pred]), 4),
                    "n": int(result.nobs),
                })
            return rows
        except Exception:
            return []

    def _termination_review_queue(self) -> pd.DataFrame:
        """
        Rows that deserve LLM or manual review.

        This gives you a controlled worklist instead of sending every
        whyStopped value to an LLM.
        """
        df = self.df.copy()
        df["has_text"] = df["why_stopped_text"].str.len().fillna(0) > 0
        mask = (
            df["is_terminated_like"]
            & df["has_text"]
            & (
                df["termination_needs_llm"]
                | df["termination_category"].isin(["uncertain", "mixed_reasons", "unclear"])
            )
        )
        cols = [
            "nct_id",
            "has_dmc",
            "overall_status",
            "why_stopped",
            "termination_category",
            "termination_confidence",
            "termination_matched_terms",
            "termination_secondary_categories",
            "termination_rationale",
            "brief_title",
            "lead_sponsor_name",
            "sponsor_class",
            "phase",
            "study_type",
            "therapeutic_area",
        ]
        existing_cols = [col for col in cols if col in df.columns]
        return df.loc[mask, existing_cols].sort_values(
            ["termination_category", "termination_confidence", "nct_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    # Utilities

    @staticmethod
    def _has_value(value: Any) -> bool:
        """True when a dataframe cell has usable context."""
        if value is None:
            return False
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        try:
            return not bool(pd.isna(value))
        except (TypeError, ValueError):
            return True

    def _print_summaries(
        self,
        prev_overall: pd.DataFrame,
        dmc_expected_vs_actual: pd.DataFrame,
        gaps: pd.DataFrame,
        outcomes: pd.DataFrame,
        termination_categories: pd.DataFrame,
        proxy_validation: pd.DataFrame,
        coverage_index_ta: pd.DataFrame,
        logistic_results: pd.DataFrame,
    ) -> None:
        """Print a concise human-readable summary to the console."""
        print("\n=== DMC Prevalence (Overall) ===")
        print(prev_overall.to_string(index=False))

        print("\n=== DMC Indicated vs Reported DMC ===")
        print(dmc_expected_vs_actual.to_string(index=False))

        print("\n=== Oversight Gaps: Phase III/IV • Interventional • Industry ===")
        print(gaps.to_string(index=False))

        print("\n=== Outcomes by DMC (Termination-like %) ===")
        print(outcomes.to_string(index=False))

        print("\n=== Structured Termination Categories by DMC ===")
        print(termination_categories.to_string(index=False))

        print("\n=== Proxy Validation: DMC-Indicated vs Not-Indicated (Interventional) ===")
        print(proxy_validation.to_string(index=False))

        print("\n=== DMC Coverage Index by Therapeutic Area (worst-covered first) ===")
        ta_sorted = coverage_index_ta.sort_values("coverage_index").reset_index(drop=True)
        print(ta_sorted[["therapeutic_area", "total_expected", "expected_with_dmc_pct", "coverage_index"]].to_string(index=False))

        if not logistic_results.empty and "model" in logistic_results.columns:
            for model_name in logistic_results["model"].unique():
                block = logistic_results.loc[logistic_results["model"] == model_name].drop(columns=["model"])
                print(f"\n=== Logistic Regression [{model_name}] — DMC Presence (Interventional Trials) ===")
                print(block.to_string(index=False))


# CLI entry

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    paths = Paths()
    enable_llm = os.getenv("ENABLE_LLM", "").strip().lower() in {"1", "true", "yes"}
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    dmc_llm_client = OpenAIDMCExpectedLLMClient(model=llm_model) if enable_llm else None
    dmc_classifier = DMCExpectedClassifier(llm_client=dmc_llm_client)

    termination_classifier = build_default_classifier(enable_llm=enable_llm, model=llm_model)

    m = DMCMetrics(paths, dmc_expected_classifier=dmc_classifier, termination_classifier=termination_classifier)
    m.load()
    m.compute_and_save_all()
