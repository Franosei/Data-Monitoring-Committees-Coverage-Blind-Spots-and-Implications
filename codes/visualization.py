"""
visualization.py — Focused DMC visuals implemented as a class (ONLY 3 PLOTS)

Reads:
    data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv

Generates high-resolution figures to ./visualization/:
  1) Heatmap — DMC prevalence (%) by Sponsor × Therapeutic Area (top-N TAs)
       -> visualization/fig_dmc_heatmap_sponsor_ta.png
  2) Annual DMC prevalence by Sponsor (multi-line)
       -> visualization/fig_trend_by_sponsor_year.png
  3) Annual DMC prevalence by Phase (multi-line) — EXCLUDES N/A
       -> visualization/fig_trend_by_phase_year.png

Notes:
- Matplotlib only (no seaborn). Colors chosen for publication.
"""

from __future__ import annotations
import os
from typing import Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DMCVisualizer:
    """
    Class that loads the prepared CTG dataset and produces three publication-ready plots.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV (one row per NCT ID) with columns:
        nct_id, has_dmc, sponsor_class, study_type, phase, start_date/start_year,
        therapeutic_area, overall_status, etc.
    out_dir : str
        Output directory to save figures.
    dpi : int
        Resolution for saved figures.
    top_ta : int
        Number of therapeutic areas to include in the heatmap (by trial count).
    colors : Optional[Dict[str, str]]
        Optional override palette for sponsors and phases.

    Methods
    -------
    load()
        Load and normalize the dataset (coerce types, create helper columns).
    plot_heatmap_sponsor_ta()
        Save heatmap of DMC prevalence by Sponsor × Therapeutic Area.
    plot_trend_by_sponsor()
        Save multi-line trend of annual DMC prevalence by sponsor.
    plot_trend_by_phase()
        Save multi-line trend of annual DMC prevalence by phase (excluding N/A).
    run_all()
        Convenience method to generate all three figures.
    """

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

    def __init__(
        self,
        input_csv: str = "data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv",
        out_dir: str = "visualization",
        dpi: int = 300,
        top_ta: int = 12,
        colors: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.input_csv = input_csv
        self.out_dir = out_dir
        self.dpi = int(dpi)
        self.top_ta = int(top_ta)

        self.LINE_COLORS = self.DEFAULT_LINE_COLORS.copy()
        self.PHASE_COLORS = self.DEFAULT_PHASE_COLORS.copy()
        if colors:
            self.LINE_COLORS.update(colors.get("sponsor", {}))
            self.PHASE_COLORS.update(colors.get("phase", {}))

        self.df: pd.DataFrame = pd.DataFrame()

    # Utils

    @staticmethod
    def _pct(a: int, b: int) -> float:
        """Return percentage a/b*100 with safe zero handling."""
        return (a / b * 100.0) if b else 0.0

    def _ensure_outdir(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.out_dir, exist_ok=True)

    # Data

    def load(self) -> None:
        """Load and normalize the dataset for plotting."""
        df = pd.read_csv(self.input_csv, low_memory=False).drop_duplicates(subset=["nct_id"]).reset_index(drop=True)

        # has_dmc -> boolean
        df["has_dmc"] = (
            df["has_dmc"]
            .map(lambda x: np.nan if pd.isna(x) else str(x).strip().lower())
            .map({"true": True, "false": False})
            .astype("boolean")
        )

        # sponsor bucket
        df["sponsor_class"] = df["sponsor_class"].astype(str).str.upper().str.strip()
        df["sponsor_bucket"] = df["sponsor_class"].replace({
            "INDUSTRY": "Industry",
            "ACADEMIC": "Academic",
            "NIH": "NIH",
            "NETWORK": "Academic",
            "OTHER": "Other",
        })
        df.loc[~df["sponsor_bucket"].isin(["Industry", "Academic", "NIH", "Other"]), "sponsor_bucket"] = "Other"

        # phase labels and N/A flag
        df["phase"] = df["phase"].fillna("").astype(str).str.strip()
        df["phase_label"] = df["phase"].replace({"": "N/A", "NaN": "N/A"})
        df["phase_is_na"] = df["phase_label"].eq("N/A")

        # year
        if "start_year" not in df.columns:
            df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year

        # therapeutic area
        if "therapeutic_area" not in df.columns:
            df["therapeutic_area"] = "Other/Unclassified"

        self.df = df

    # Plots

    def plot_heatmap_sponsor_ta(self, filename: str = "fig_dmc_heatmap_sponsor_ta.png") -> None:
        """Heatmap: DMC prevalence (%) by Sponsor × Therapeutic Area (top-N by count)."""
        self._ensure_outdir()
        out_path = os.path.join(self.out_dir, filename)
        df = self.df.copy()

        ta_top = df["therapeutic_area"].value_counts().head(self.top_ta).index.tolist()
        sub = df[df["therapeutic_area"].isin(ta_top)].copy()

        grp = (
            sub.groupby(["sponsor_bucket", "therapeutic_area"], observed=True)
               .agg(total=("nct_id", "count"),
                    with_dmc=("has_dmc", lambda s: s.fillna(False).sum()))
               .reset_index()
        )
        grp["with_dmc_pct"] = grp.apply(lambda r: self._pct(r["with_dmc"], r["total"]), axis=1)

        sponsors = sorted(grp["sponsor_bucket"].unique().tolist())
        tas = sorted(ta_top)
        mat = np.full((len(sponsors), len(tas)), np.nan)
        for i, sp in enumerate(sponsors):
            for j, ta in enumerate(tas):
                row = grp[(grp["sponsor_bucket"] == sp) & (grp["therapeutic_area"] == ta)]
                if not row.empty:
                    mat[i, j] = row["with_dmc_pct"].values[0]

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_yticks(range(len(sponsors)))
        ax.set_yticklabels(sponsors)
        ax.set_xticks(range(len(tas)))
        ax.set_xticklabels(tas, rotation=45, ha="right")
        ax.set_title("DMC Prevalence (%) by Sponsor × Therapeutic Area",
                     color=self.COLOR_NEUTRAL, pad=12, fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("With DMC (%)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi)
        plt.close(fig)

    def plot_trend_by_sponsor(self, filename: str = "fig_trend_by_sponsor_year.png") -> None:
        """Multi-line trend: annual DMC prevalence by sponsor bucket."""
        self._ensure_outdir()
        out_path = os.path.join(self.out_dir, filename)
        df = self.df.copy()

        g = (
            df.groupby(["start_year", "sponsor_bucket"], dropna=True, observed=True)
              .agg(total=("nct_id", "count"),
                   with_dmc=("has_dmc", lambda s: s.fillna(False).sum()))
              .reset_index()
              .sort_values(["sponsor_bucket", "start_year"])
        )
        g["with_dmc_pct"] = g.apply(lambda r: self._pct(r["with_dmc"], r["total"]), axis=1)

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)
        for sp, sub in g.groupby("sponsor_bucket"):
            color = self.LINE_COLORS.get(sp, "#333333")
            ax.plot(sub["start_year"], sub["with_dmc_pct"],
                    marker="o", linewidth=2, label=sp, color=color)

        ax.set_title("Annual DMC Prevalence by Sponsor Type",
                     color=self.COLOR_NEUTRAL, pad=12, fontsize=14)
        ax.set_xlabel("Start Year", color=self.COLOR_NEUTRAL)
        ax.set_ylabel("With DMC (%)", color=self.COLOR_NEUTRAL)
        ax.grid(True, color=self.COLOR_GRID, alpha=0.5)
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi)
        plt.close(fig)

    def plot_trend_by_phase(self, filename: str = "fig_trend_by_phase_year.png") -> None:
        """Multi-line trend: annual DMC prevalence by phase, excluding N/A."""
        self._ensure_outdir()
        out_path = os.path.join(self.out_dir, filename)
        df = self.df[~self.df["phase_is_na"]].copy()

        g = (
            df.groupby(["start_year", "phase_label"], dropna=True, observed=True)
              .agg(total=("nct_id", "count"),
                   with_dmc=("has_dmc", lambda s: s.fillna(False).sum()))
              .reset_index()
              .sort_values(["phase_label", "start_year"])
        )
        g["with_dmc_pct"] = g.apply(lambda r: self._pct(r["with_dmc"], r["total"]), axis=1)

        phases = g["phase_label"].dropna().unique().tolist()
        palette = {ph: self.PHASE_COLORS.get(ph, "#333333") for ph in phases}

        fig, ax = plt.subplots(figsize=self.FIG_WIDE, dpi=self.dpi)
        for ph, sub in g.groupby("phase_label"):
            ax.plot(sub["start_year"], sub["with_dmc_pct"],
                    marker="o", linewidth=2, label=ph, color=palette.get(ph, "#333333"))

        ax.set_title("Annual DMC Prevalence by Phase (Excluding N/A)",
                     color=self.COLOR_NEUTRAL, pad=12, fontsize=14)
        ax.set_xlabel("Start Year", color=self.COLOR_NEUTRAL)
        ax.set_ylabel("With DMC (%)", color=self.COLOR_NEUTRAL)
        ax.grid(True, color=self.COLOR_GRID, alpha=0.5)
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi)
        plt.close(fig)

    # Orchestrator

    def run_all(self) -> None:
        """Generate all three figures."""
        if self.df.empty:
            self.load()
        self.plot_heatmap_sponsor_ta()
        self.plot_trend_by_sponsor()
        self.plot_trend_by_phase()
        print(f"Saved figures in: {self.out_dir}")


# CLI

if __name__ == "__main__":
    viz = DMCVisualizer(
        input_csv="data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv",
        out_dir="visualization",
        dpi=300,
        top_ta=12,
    )
    viz.load()
    viz.run_all()
