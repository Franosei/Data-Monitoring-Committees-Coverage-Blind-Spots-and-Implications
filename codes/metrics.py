"""
metrics.py — DMC Coverage Map Metrics

Reads `data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv`
and computes the following metrics:

1) DMC Prevalence
   - % of trials with a DMC overall
   - Breakdown by phase, sponsor type, study type, therapeutic area

2) Oversight Gaps
   - Proportion of Phase III or IV interventional industry trials without a DMC
   - Highlight high-risk therapeutic areas (oncology, cardiovascular, infectious disease)

3) DMC & Trial Outcomes
   - Compare termination/withdrawal/suspension rates between trials with vs without a DMC
   - Why-stopped analysis: safety/futility keywords prevalence by DMC status

Outputs CSVs to the `data/metrics folder and prints concise summaries.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Paths:
    """Container for input/output paths."""
    input_csv: str = "data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv"
    out_dir: str = "data/metrics"
    prevalence_overall_csv: str = "data/metrics/metric_prevalence_overall.csv"
    prevalence_phase_csv: str = "data/metrics/metric_prevalence_by_phase.csv"
    prevalence_sponsor_csv: str = "data/metrics/metric_prevalence_by_sponsor.csv"
    prevalence_studytype_csv: str = "data/metrics/metric_prevalence_by_studytype.csv"
    prevalence_ta_csv: str = "data/metrics/metric_prevalence_by_therapeutic_area.csv"
    oversight_gaps_csv: str = "data/metrics/metric_oversight_gaps_phase34_industry_interventional.csv"
    outcomes_by_dmc_csv: str = "data/metrics/metric_outcomes_by_dmc.csv"
    whystopped_by_dmc_csv: str = "data/metrics/metric_whystopped_by_dmc.csv"


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
        - therapeutic_area (str)

    Notes:
        - has_dmc is coerced to boolean dtype ('boolean') with NA as <NA>.
        - sponsor_class normalized to upper case; common buckets derived.
        - phase is parsed to detect presence of Phase 3 or Phase 4.
    """

    # Keywords for whyStopped classification
    SAFETY_KEYS = [
        "safety", "ae", "adverse event", "adverse events", "toxicity",
        "death", "mortality", "side effect", "risk", "harm", "serious adverse"
    ]
    FUTILITY_KEYS = [
        "futility", "lack of efficacy", "no efficacy", "ineffective",
        "conditional power", "no benefit", "insufficient efficacy", "interim analysis futility"
    ]

    # TA labels to highlight as high-risk areas
    HIGH_RISK_TA = {"Oncology", "Cardiovascular", "Infectious Disease"}

    def __init__(self, paths: Optional[Paths] = None) -> None:
        self.paths = paths or Paths()
        self.df: pd.DataFrame = pd.DataFrame()


    def load(self) -> None:
        """Load and normalize the input dataset."""
        if not os.path.exists(self.paths.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {self.paths.input_csv}")

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

        # why_stopped normalized text
        df["why_stopped_text"] = df.get("why_stopped", np.nan)
        df["why_stopped_text"] = df["why_stopped_text"].astype(str).fillna("").str.lower()

        # Persist
        self.df = df

    def compute_and_save_all(self) -> None:
        """Compute all metrics and save CSVs."""
        os.makedirs(self.paths.out_dir, exist_ok=True)

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

        gaps = self._metric_oversight_gaps_phase34_industry_interventional()
        gaps.to_csv(self.paths.oversight_gaps_csv, index=False)

        outcomes = self._metric_outcomes_by_dmc()
        outcomes.to_csv(self.paths.outcomes_by_dmc_csv, index=False)

        whystop = self._metric_whystopped_keywords_by_dmc()
        whystop.to_csv(self.paths.whystopped_by_dmc_csv, index=False)

        # Print concise summaries
        self._print_summaries(prev_overall, gaps, outcomes, whystop)

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
            has_dmc_label, total, terminated_like, completed, terminated_like_pct
        """
        df = self.df.copy()
        grp = df.groupby(df["has_dmc"].fillna(False)).agg(
            total=("nct_id", "count"),
            terminated_like=("is_terminated_like", "sum"),
            completed=("is_completed", "sum"),
        ).reset_index()

        grp["has_dmc_label"] = grp["has_dmc"].map({True: "With DMC", False: "No DMC"})
        grp["terminated_like_pct"] = np.where(
            grp["total"] > 0, grp["terminated_like"] / grp["total"] * 100.0, 0.0
        ).round(2)
        out_cols = ["has_dmc_label", "total", "terminated_like", "completed", "terminated_like_pct"]
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

    # Utilities

    def _print_summaries(
        self,
        prev_overall: pd.DataFrame,
        gaps: pd.DataFrame,
        outcomes: pd.DataFrame,
        whystop: pd.DataFrame,
    ) -> None:
        """Print a concise human-readable summary to the console."""
        print("\n=== DMC Prevalence (Overall) ===")
        print(prev_overall.to_string(index=False))

        print("\n=== Oversight Gaps: Phase III/IV • Interventional • Industry ===")
        print(gaps.to_string(index=False))

        print("\n=== Outcomes by DMC (Termination-like %) ===")
        print(outcomes.to_string(index=False))

        print("\n=== whyStopped Keyword Signals (Safety/Futility) by DMC ===")
        print(whystop.to_string(index=False))


# CLI entry

if __name__ == "__main__":
    paths = Paths()
    m = DMCMetrics(paths)
    m.load()
    m.compute_and_save_all()
