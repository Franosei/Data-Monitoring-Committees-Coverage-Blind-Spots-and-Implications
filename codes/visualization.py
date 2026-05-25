"""
visualization.py — Publication-ready DMC visuals.

Reads:
    data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv
    data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended_structured.csv
    data/metrics/*.csv

Generates high-resolution figures to ./visualization/:
  1) Heatmap — DMC prevalence (%) by Sponsor × Therapeutic Area
  2) Annual DMC prevalence by Sponsor
  3) Annual DMC prevalence by Phase
  4) Risk-feature count vs DMC coverage curve
  5) Oversight gap heatmap among DMC-indicated trials
  6) Trial population funnel
  7) Termination category rates by DMC status
  8) True DMC Coverage Index by Therapeutic Area

Notes:
- Matplotlib only.
- Coverage Index = therapeutic-area DMC coverage / overall DMC coverage.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


class DMCVisualizer:
    """Create publication-ready figures for the DMC ClinicalTrials.gov analysis."""

    DEFAULT_LINE_COLORS = {
        "Industry": "#2a9d8f",
        "Academic": "#e76f51",
        "NIH": "#577590",
        "Other": "#8d99ae",
    }

    DEFAULT_PHASE_COLORS = {
        "Early Phase 1": "#606c38",
        "Phase 1": "#264653",
        "Phase 1/2": "#2a9d8f",
        "Phase 2": "#e76f51",
        "Phase 2/3": "#fb8500",
        "Phase 3": "#577590",
        "Phase 4": "#8d99ae",
    }

    COLOR_NEUTRAL = "#1f2937"
    COLOR_GRID = "#bfbfbf"
    FIG_WIDE = (12, 7)

    FLAG_COLUMNS: List[str] = [
        "dmc_expected_late_phase",
        "dmc_expected_serious_condition",
        "dmc_expected_vulnerable_population",
        "dmc_expected_large_enrollment",
        "dmc_expected_regulated_intervention",
        "dmc_expected_long_duration",
        "dmc_expected_serious_endpoint",
    ]

    def __init__(
        self,
        input_csv: str = "data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv",
        structured_csv: str = "data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended_structured.csv",
        metrics_dir: str = "data/metrics",
        out_dir: str = "visualization",
        dpi: int = 300,
        top_ta: int = 12,
        colors: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.input_csv = input_csv
        self.structured_csv = structured_csv
        self.metrics_dir = metrics_dir
        self.out_dir = out_dir
        self.dpi = int(dpi)
        self.top_ta = int(top_ta)

        self.LINE_COLORS = self.DEFAULT_LINE_COLORS.copy()
        self.PHASE_COLORS = self.DEFAULT_PHASE_COLORS.copy()

        if colors:
            self.LINE_COLORS.update(colors.get("sponsor", {}))
            self.PHASE_COLORS.update(colors.get("phase", {}))

        self.df: pd.DataFrame = pd.DataFrame()
        self.sdf: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _pct(numerator: int, denominator: int) -> float:
        return (numerator / denominator * 100.0) if denominator else 0.0

    def _ensure_outdir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _to_boolean_series(series: pd.Series) -> pd.Series:
        return (
            series.map(lambda x: np.nan if pd.isna(x) else str(x).strip().lower())
            .map({"true": True, "false": False})
            .astype("boolean")
        )

    @staticmethod
    def _normalise_sponsor_bucket(df: pd.DataFrame) -> pd.DataFrame:
        df["sponsor_class"] = df["sponsor_class"].astype(str).str.upper().str.strip()
        df["sponsor_bucket"] = df["sponsor_class"].replace(
            {
                "INDUSTRY": "Industry",
                "ACADEMIC": "Academic",
                "NIH": "NIH",
                "NETWORK": "Academic",
                "OTHER": "Other",
            }
        )
        df.loc[
            ~df["sponsor_bucket"].isin(["Industry", "Academic", "NIH", "Other"]),
            "sponsor_bucket",
        ] = "Other"
        return df

    def load(self) -> None:
        """Load raw analysis CSV for prevalence and trend figures."""
        df = pd.read_csv(self.input_csv, low_memory=False)
        df = df.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)

        df["has_dmc"] = self._to_boolean_series(df["has_dmc"])
        df = self._normalise_sponsor_bucket(df)

        df["phase"] = df["phase"].fillna("").astype(str).str.strip()
        df["phase_label"] = df["phase"].replace({"": "N/A", "NaN": "N/A"})
        df["phase_is_na"] = df["phase_label"].eq("N/A")

        if "start_year" not in df.columns:
            df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year

        if "therapeutic_area" not in df.columns:
            df["therapeutic_area"] = "Other/Unclassified"

        self.df = df

    def load_structured(self) -> None:
        """Load structured CSV with DMC-indicated flags and termination labels."""
        sdf = pd.read_csv(self.structured_csv, low_memory=False)
        sdf = sdf.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)

        sdf["has_dmc"] = self._to_boolean_series(sdf["has_dmc"])

        bool_cols = ["dmc_expected", "is_interventional_trial", "dmc_expected_broad"] + self.FLAG_COLUMNS
        for col in bool_cols:
            if col in sdf.columns:
                sdf[col] = sdf[col].map(
                    lambda x: True
                    if str(x).strip().lower() == "true"
                    else False
                    if str(x).strip().lower() == "false"
                    else np.nan
                )

        if "sponsor_bucket" not in sdf.columns:
            sdf = self._normalise_sponsor_bucket(sdf)

        if "therapeutic_area" not in sdf.columns:
            sdf["therapeutic_area"] = "Other/Unclassified"

        self.sdf = sdf

    def plot_heatmap_sponsor_ta(self, filename: str = "fig_dmc_heatmap_sponsor_ta.png") -> None:
        """DMC prevalence (%) by sponsor and therapeutic area."""
        self._ensure_outdir()
        df = self.df.copy()

        ta_top = df["therapeutic_area"].value_counts().head(self.top_ta).index.tolist()
        sub = df[df["therapeutic_area"].isin(ta_top)].copy()

        grp = (
            sub.groupby(["sponsor_bucket", "therapeutic_area"], observed=True)
            .agg(
                total=("nct_id", "count"),
                with_dmc=("has_dmc", lambda s: int(s.fillna(False).sum())),
            )
            .reset_index()
        )
        grp["with_dmc_pct"] = grp.apply(lambda r: self._pct(r["with_dmc"], r["total"]), axis=1)

        sponsors = sorted(grp["sponsor_bucket"].unique().tolist())
        tas = sorted(ta_top)

        mat = np.full((len(sponsors), len(tas)), np.nan)
        for i, sponsor in enumerate(sponsors):
            for j, ta in enumerate(tas):
                row = grp[(grp["sponsor_bucket"] == sponsor) & (grp["therapeutic_area"] == ta)]
                if not row.empty:
                    mat[i, j] = row["with_dmc_pct"].values[0]

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=np.nanmin(mat), vmax=np.nanmax(mat))

        ax.set_yticks(range(len(sponsors)))
        ax.set_yticklabels(sponsors)
        ax.set_xticks(range(len(tas)))
        ax.set_xticklabels(tas, rotation=45, ha="right")
        ax.set_title(
            "DMC Prevalence (%) by Sponsor × Therapeutic Area",
            color=self.COLOR_NEUTRAL,
            pad=12,
            fontsize=14,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("With DMC (%)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_trend_by_sponsor(self, filename: str = "fig_trend_by_sponsor_year.png") -> None:
        """Annual DMC prevalence by sponsor type."""
        self._ensure_outdir()
        df = self.df.copy()

        grp = (
            df.groupby(["start_year", "sponsor_bucket"], dropna=True, observed=True)
            .agg(
                total=("nct_id", "count"),
                with_dmc=("has_dmc", lambda s: int(s.fillna(False).sum())),
            )
            .reset_index()
            .sort_values(["sponsor_bucket", "start_year"])
        )
        grp["with_dmc_pct"] = grp.apply(lambda r: self._pct(r["with_dmc"], r["total"]), axis=1)

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)

        for sponsor, sub in grp.groupby("sponsor_bucket"):
            ax.plot(
                sub["start_year"],
                sub["with_dmc_pct"],
                marker="o",
                linewidth=2,
                label=sponsor,
                color=self.LINE_COLORS.get(sponsor, "#333333"),
            )

        ax.set_title("Annual DMC Prevalence by Sponsor Type", color=self.COLOR_NEUTRAL, pad=12, fontsize=14)
        ax.set_xlabel("Start Year", color=self.COLOR_NEUTRAL)
        ax.set_ylabel("With DMC (%)", color=self.COLOR_NEUTRAL)
        ax.grid(True, color=self.COLOR_GRID, alpha=0.5)
        ax.legend(frameon=False, ncol=2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_trend_by_phase(self, filename: str = "fig_trend_by_phase_year.png") -> None:
        """Annual DMC prevalence by phase, excluding N/A."""
        self._ensure_outdir()
        df = self.df[~self.df["phase_is_na"]].copy()

        grp = (
            df.groupby(["start_year", "phase_label"], dropna=True, observed=True)
            .agg(
                total=("nct_id", "count"),
                with_dmc=("has_dmc", lambda s: int(s.fillna(False).sum())),
            )
            .reset_index()
            .sort_values(["phase_label", "start_year"])
        )
        grp["with_dmc_pct"] = grp.apply(lambda r: self._pct(r["with_dmc"], r["total"]), axis=1)

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)

        for phase, sub in grp.groupby("phase_label"):
            ax.plot(
                sub["start_year"],
                sub["with_dmc_pct"],
                marker="o",
                linewidth=2,
                label=phase,
                color=self.PHASE_COLORS.get(phase, "#333333"),
            )

        ax.set_title("Annual DMC Prevalence by Phase (Excluding N/A)", color=self.COLOR_NEUTRAL, pad=12, fontsize=14)
        ax.set_xlabel("Start Year", color=self.COLOR_NEUTRAL)
        ax.set_ylabel("With DMC (%)", color=self.COLOR_NEUTRAL)
        ax.grid(True, color=self.COLOR_GRID, alpha=0.5)
        ax.legend(frameon=False, ncol=2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_risk_coverage_curve(self, filename: str = "fig_risk_coverage_curve.png") -> None:
        """DMC coverage by number of active DMC guidance features."""
        self._ensure_outdir()

        if self.sdf.empty:
            self.load_structured()

        df = self.sdf.loc[self.sdf["is_interventional_trial"].fillna(False).astype(bool)].copy()
        flag_cols = [col for col in self.FLAG_COLUMNS if col in df.columns]

        df["feature_count"] = df[flag_cols].fillna(False).astype(int).sum(axis=1)

        xs, ys, lo_cis, hi_cis, ns = [], [], [], [], []

        for feature_count in sorted(df["feature_count"].unique()):
            sub = df[df["feature_count"] == feature_count]
            n = len(sub)
            k = int(sub["has_dmc"].fillna(False).sum())
            pct = self._pct(k, n)

            lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")

            xs.append(feature_count)
            ys.append(pct)
            lo_cis.append(lo * 100)
            hi_cis.append(hi * 100)
            ns.append(n)

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)

        ax.plot(xs, ys, marker="o", linewidth=2.5, color="#577590", zorder=3)
        ax.fill_between(xs, lo_cis, hi_cis, alpha=0.18, color="#577590")

        for x, y, n in zip(xs, ys, ns):
            ax.annotate(
                f"N={n:,}",
                xy=(x, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=self.COLOR_NEUTRAL,
            )

        ax.set_xlabel("Number of DMC Guidance Features Active (0–7)", color=self.COLOR_NEUTRAL)
        ax.set_ylabel("Interventional Trials with DMC (%)", color=self.COLOR_NEUTRAL)
        ax.set_title(
            "DMC Coverage Rate by Risk Feature Count\n(95% Wilson CI)",
            color=self.COLOR_NEUTRAL,
            pad=12,
            fontsize=14,
        )
        ax.set_xticks(range(len(self.FLAG_COLUMNS) + 1))
        ax.set_ylim(0, 100)
        ax.grid(True, color=self.COLOR_GRID, alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_oversight_gap_heatmap(self, filename: str = "fig_oversight_gap_heatmap.png") -> None:
        """% of DMC-indicated trials without DMC by sponsor and therapeutic area."""
        self._ensure_outdir()

        if self.sdf.empty:
            self.load_structured()

        df = self.sdf.loc[self.sdf["dmc_expected"].fillna(False).astype(bool)].copy()

        ta_top = df["therapeutic_area"].value_counts().head(10).index.tolist()
        sub = df[df["therapeutic_area"].isin(ta_top)].copy()

        grp = (
            sub.groupby(["sponsor_bucket", "therapeutic_area"], observed=True)
            .agg(
                total=("nct_id", "count"),
                without_dmc=("has_dmc", lambda s: int((~s.fillna(False)).sum())),
            )
            .reset_index()
        )
        grp["gap_pct"] = grp.apply(lambda r: self._pct(r["without_dmc"], r["total"]), axis=1)

        sponsors = sorted(grp["sponsor_bucket"].unique().tolist())
        tas = sorted(ta_top)

        mat = np.full((len(sponsors), len(tas)), np.nan)

        for i, sponsor in enumerate(sponsors):
            for j, ta in enumerate(tas):
                row = grp[(grp["sponsor_bucket"] == sponsor) & (grp["therapeutic_area"] == ta)]
                if not row.empty:
                    mat[i, j] = row["gap_pct"].values[0]

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

        for i in range(len(sponsors)):
            for j in range(len(tas)):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if mat[i, j] > 60 else self.COLOR_NEUTRAL,
                    )

        ax.set_yticks(range(len(sponsors)))
        ax.set_yticklabels(sponsors)
        ax.set_xticks(range(len(tas)))
        ax.set_xticklabels(tas, rotation=45, ha="right")

        ax.set_title(
            "Oversight Gap (% Without DMC) Among DMC-Indicated Trials\nby Sponsor × Therapeutic Area",
            color=self.COLOR_NEUTRAL,
            pad=12,
            fontsize=13,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Without DMC (%)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_funnel(self, filename: str = "fig_population_funnel.png") -> None:
        """Population funnel from all trials to DMC-indicated trials without DMC."""
        self._ensure_outdir()

        if self.sdf.empty:
            self.load_structured()

        sdf = self.sdf.copy()

        n_all = len(sdf)
        n_interventional = int(sdf["is_interventional_trial"].fillna(False).sum())
        n_indicated = int(sdf["dmc_expected"].fillna(False).sum())
        n_gap = int(
            sdf.loc[sdf["dmc_expected"].fillna(False), "has_dmc"]
            .fillna(False)
            .eq(False)
            .sum()
        )

        labels = [
            "All Trials",
            "Interventional",
            "DMC-Indicated\n(Risk Proxy)",
            "DMC-Indicated\nWithout DMC",
        ]
        counts = [n_all, n_interventional, n_indicated, n_gap]
        colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        max_n = counts[0]
        y_positions = [3, 2, 1, 0]
        bar_height = 0.55

        for i, (y, n, label, color) in enumerate(zip(y_positions, counts, labels, colors)):
            width = n / max_n
            left = (1 - width) / 2
            pct_of_prior = f"({n / counts[i - 1] * 100:.1f}% of prior)" if i > 0 else ""

            ax.barh(y, width, left=left, height=bar_height, color=color, zorder=2)
            ax.text(
                0.5,
                y,
                f"{label}\nN={n:,} {pct_of_prior}",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
                zorder=3,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 3.7)
        ax.axis("off")
        ax.set_title(
            "Trial Population Funnel: From All Trials to the Oversight Gap",
            color=self.COLOR_NEUTRAL,
            pad=14,
            fontsize=14,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_termination_by_dmc(self, filename: str = "fig_termination_by_dmc.png") -> None:
        """Termination category rates by DMC status."""
        self._ensure_outdir()

        csv_path = os.path.join(self.metrics_dir, "metric_termination_category_by_dmc.csv")
        if not os.path.exists(csv_path):
            return

        tdf = pd.read_csv(csv_path)

        target_categories = [
            "safety",
            "efficacy",
            "futility",
            "recruitment_failure",
            "administrative",
            "business_decision",
        ]

        tdf = tdf[tdf["termination_category"].isin(target_categories)].copy()

        pivot = (
            tdf.pivot_table(
                index="termination_category",
                columns="has_dmc_label",
                values="category_rate_pct",
                aggfunc="first",
            )
            .reindex(target_categories)
            .fillna(0)
        )

        with_dmc = pivot["With DMC"].values if "With DMC" in pivot.columns else np.zeros(len(target_categories))
        no_dmc = pivot["No DMC"].values if "No DMC" in pivot.columns else np.zeros(len(target_categories))

        x_labels = [cat.replace("_", " ").title() for cat in target_categories]
        x = np.arange(len(target_categories))
        width = 0.38

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)

        bars_1 = ax.bar(x - width / 2, with_dmc, width, label="With DMC", color="#577590")
        bars_2 = ax.bar(x + width / 2, no_dmc, width, label="No DMC", color="#e76f51")

        for bar in list(bars_1) + list(bars_2):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.3,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=self.COLOR_NEUTRAL,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_ylabel("% of Stopped Trials in Category", color=self.COLOR_NEUTRAL)
        ax.set_title(
            "Termination Category Rates by DMC Status\n(Stopped Trials with Documented Reason)",
            color=self.COLOR_NEUTRAL,
            pad=12,
            fontsize=14,
        )
        ax.legend(frameon=False)
        ax.grid(axis="y", color=self.COLOR_GRID, alpha=0.5)
        ax.set_ylim(0, max(with_dmc.max(), no_dmc.max()) * 1.18)

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def plot_coverage_index(self, filename: str = "fig_coverage_index_by_ta.png") -> None:
        """
        True DMC Coverage Index by therapeutic area.

        Coverage rate = % with DMC among DMC-indicated trials.
        Coverage index = therapeutic-area coverage rate / overall coverage rate.

        Interpretation:
            1.00  = same as overall DMC coverage among DMC-indicated trials
            <1.00 = below-average coverage
            >1.00 = above-average coverage
        """
        self._ensure_outdir()

        csv_path = os.path.join(self.metrics_dir, "metric_dmc_expected_gap_by_therapeutic_area.csv")
        if not os.path.exists(csv_path):
            return

        cdf = pd.read_csv(csv_path)
        cdf = cdf[cdf["total_expected"] >= 20].copy()

        if cdf.empty:
            return

        if "expected_with_dmc" not in cdf.columns:
            cdf["expected_with_dmc"] = (
                cdf["total_expected"] * cdf["expected_with_dmc_pct"] / 100.0
            )

        total_with_dmc = cdf["expected_with_dmc"].sum()
        total_expected = cdf["total_expected"].sum()

        reference_rate = (total_with_dmc / total_expected * 100.0) if total_expected else 0.0
        if reference_rate == 0:
            return

        cdf["coverage_index"] = cdf["expected_with_dmc_pct"] / reference_rate
        cdf = cdf.sort_values("coverage_index", ascending=True).reset_index(drop=True)

        def bar_color(index_value: float) -> str:
            if index_value < 0.75:
                return "#c1121f"
            if index_value < 1.00:
                return "#fb8500"
            return "#2a9d8f"

        colors = [bar_color(v) for v in cdf["coverage_index"]]

        fig, ax = plt.subplots(
            figsize=(10, max(5, len(cdf) * 0.42 + 1.5)),
            dpi=self.dpi,
        )

        y = np.arange(len(cdf))

        ax.barh(
            y,
            cdf["coverage_index"],
            color=colors,
            height=0.65,
            zorder=2,
        )

        ax.axvline(
            1.0,
            color=self.COLOR_NEUTRAL,
            linestyle="--",
            linewidth=1.5,
            label="Overall reference: 1.00",
            zorder=3,
        )

        for i, (_, row) in enumerate(cdf.iterrows()):
            ax.text(
                row["coverage_index"] + 0.02,
                i,
                f"{row['coverage_index']:.2f} ({row['expected_with_dmc_pct']:.1f}%)",
                va="center",
                fontsize=8,
                color=self.COLOR_NEUTRAL,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(cdf["therapeutic_area"], fontsize=9)

        ax.set_xlabel(
            "Coverage Index = Therapeutic Area DMC Coverage / Overall DMC Coverage",
            color=self.COLOR_NEUTRAL,
        )

        ax.set_title(
            "DMC Coverage Index by Therapeutic Area\n"
            f"(DMC-Indicated Trials Only; Overall Coverage = {reference_rate:.1f}%)",
            color=self.COLOR_NEUTRAL,
            pad=12,
            fontsize=13,
        )

        legend_handles = [
            mpatches.Patch(color="#c1121f", label="< 0.75 (Critical gap)"),
            mpatches.Patch(color="#fb8500", label="0.75–1.00 (Below average)"),
            mpatches.Patch(color="#2a9d8f", label="≥ 1.00 (At/above average)"),
            plt.Line2D(
                [0],
                [0],
                color=self.COLOR_NEUTRAL,
                linestyle="--",
                label="Overall reference: 1.00",
            ),
        ]

        ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc="lower right")

        max_x = max(1.6, cdf["coverage_index"].max() + 0.25)
        ax.set_xlim(0, max_x)
        ax.grid(axis="x", color=self.COLOR_GRID, alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
        plt.close(fig)

    def run_all(self) -> None:
        """Generate all publication figures."""
        if self.df.empty:
            self.load()

        if self.sdf.empty:
            self.load_structured()

        self.plot_heatmap_sponsor_ta()
        self.plot_trend_by_sponsor()
        self.plot_trend_by_phase()
        self.plot_risk_coverage_curve()
        self.plot_oversight_gap_heatmap()
        self.plot_funnel()
        self.plot_termination_by_dmc()
        self.plot_coverage_index()

        print(f"Saved 8 figures in: {self.out_dir}")


if __name__ == "__main__":
    visualizer = DMCVisualizer(
        input_csv="data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv",
        structured_csv="data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended_structured.csv",
        metrics_dir="data/metrics",
        out_dir="visualization",
        dpi=300,
        top_ta=12,
    )
    visualizer.run_all()