"""Data Drift Detector with temporal animation.

Compare reference (training) and production data distributions,
visualize drift in embedding space, and compute statistical tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import param

import holoviews as hv
import hvplot.pandas  # noqa: F401 — needed for .hvplot.line() in _temporal_animation
import panel as pn

hv.extension("bokeh")


class DriftDetector(pn.viewable.Viewer):
    """Detect and visualize distribution drift between two datasets.

    Features
    --------
    - KDE comparison per feature
    - Statistical tests: KS-test, PSI per feature
    - Temporal animation if timestamps are available
    """

    state = param.ClassSelector(class_=object, default=None, doc="Optional DeepLensState")
    reference_df = param.DataFrame(doc="Reference (training) data")
    production_df = param.DataFrame(doc="Production data")
    feature_columns = param.List(default=[], doc="Feature columns to compare")
    timestamp_col = param.String(default="", doc="Timestamp column for temporal animation")

    def __init__(self, **params):
        super().__init__(**params)
        if not self.feature_columns and self.reference_df is not None:
            self.feature_columns = list(
                self.reference_df.select_dtypes(include=[np.number]).columns
            )
            if self.timestamp_col and self.timestamp_col in self.feature_columns:
                self.feature_columns.remove(self.timestamp_col)
        self._drift_scores_df: pd.DataFrame | None = None

    def _get_drift_scores(self) -> pd.DataFrame:
        """Cached drift scores computation."""
        if self._drift_scores_df is not None:
            return self._drift_scores_df
        self._drift_scores_df = self._compute_drift_scores()
        return self._drift_scores_df

    def _compute_drift_scores(self) -> pd.DataFrame:
        """Compute per-feature drift scores using KS-test and PSI."""
        from scipy.stats import ks_2samp

        rows = []
        for col in (self.feature_columns or []):
            if col not in self.reference_df.columns or col not in self.production_df.columns:
                continue

            ref = self.reference_df[col].dropna().values
            prod = self.production_df[col].dropna().values

            if len(ref) == 0 or len(prod) == 0:
                continue

            ks_stat, ks_pval = ks_2samp(ref, prod)
            psi = self._compute_psi(ref, prod)

            rows.append({
                "Feature": col,
                "KS Statistic": round(ks_stat, 4),
                "KS p-value": round(ks_pval, 4),
                "PSI": round(psi, 4),
                "Drift": "Yes" if ks_pval < 0.05 else "No",
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_psi(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index between two distributions."""
        bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
        prod_counts = np.histogram(production, bins=bin_edges)[0].astype(float)

        ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
        prod_pct = np.clip(prod_counts / prod_counts.sum(), 1e-6, None)

        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return max(0.0, float(psi))  # PSI is non-negative by definition

    def _drift_scores_plot(self) -> hv.Element:
        """Bar chart of drift scores per feature."""
        scores_df = self._get_drift_scores()
        if scores_df.empty:
            return hv.Text(0, 0, "No features to compare")

        colors = ["#e74c3c" if d == "Yes" else "#2ecc71" for d in scores_df["Drift"]]
        scores_df = scores_df.copy()
        scores_df["color"] = colors

        bars = hv.Bars(
            scores_df, kdims=["Feature"], vdims=["KS Statistic", "color"]
        ).opts(
            color="color",
            invert_axes=True,
            width=500,
            height=max(250, len(scores_df) * 25),
            tools=["hover"],
            title="Drift Scores (KS Statistic)",
            xlabel="KS Statistic",
        )

        # VLine for threshold on value axis (bars are inverted, so VLine = KS axis)
        threshold = hv.VLine(0.05).opts(color="orange", line_dash="dashed", line_width=2)
        return bars * threshold

    def _kde_comparison(self, feature: str) -> hv.Element:
        """KDE overlay comparing reference vs production for a single feature."""
        ref_vals = self.reference_df[feature].dropna()
        prod_vals = self.production_df[feature].dropna()

        ref_dist = hv.Distribution(ref_vals, label="Reference").opts(color="#3498db", alpha=0.6)
        prod_dist = hv.Distribution(prod_vals, label="Production").opts(color="#e74c3c", alpha=0.6)

        return (ref_dist * prod_dist).opts(
            width=500,
            height=300,
            title=f"Distribution: {feature}",
            legend_position="top_right",
        )

    def _temporal_animation(self) -> pn.pane.Markdown | pn.pane.HoloViews:
        """Drift over time windows for top features."""
        if not self.timestamp_col or self.timestamp_col not in self.production_df.columns:
            return pn.pane.Markdown("*No timestamp column provided for temporal analysis.*")

        from scipy.stats import ks_2samp

        prod = self.production_df.sort_values(self.timestamp_col)

        n_windows = min(20, len(prod) // 10)
        if n_windows < 2:
            return pn.pane.Markdown("*Not enough data for temporal analysis.*")

        window_indices = np.array_split(range(len(prod)), n_windows)
        top_features = (self.feature_columns or [])[:5]

        drift_over_time = []
        for i, indices in enumerate(window_indices):
            window_data = prod.iloc[indices]
            window_scores = []
            for col in top_features:
                ref = self.reference_df[col].dropna().values
                win = window_data[col].dropna().values
                if len(win) > 1:
                    ks_stat, _ = ks_2samp(ref, win)
                    window_scores.append(ks_stat)
            if window_scores:
                drift_over_time.append({
                    "Window": i,
                    "Mean KS": np.mean(window_scores),
                    "Max KS": np.max(window_scores),
                })

        if not drift_over_time:
            return pn.pane.Markdown("*Could not compute temporal drift.*")

        drift_df = pd.DataFrame(drift_over_time)
        plot = drift_df.hvplot.line(
            x="Window",
            y=["Mean KS", "Max KS"],
            title="Drift Over Time",
            width=600,
            height=300,
        )

        threshold = hv.HLine(0.1).opts(color="red", line_dash="dashed")
        return pn.pane.HoloViews(plot * threshold)

    def _scores_table(self) -> pn.widgets.Tabulator:
        """Full drift scores as a table."""
        scores_df = self._get_drift_scores()
        return pn.widgets.Tabulator(
            scores_df, width=600, layout="fit_columns", show_index=False
        )

    def __panel__(self):
        if self.reference_df is None or self.production_df is None:
            return pn.pane.Markdown(
                "### Drift Detector\n*Provide reference and production DataFrames.*"
            )

        features = self.feature_columns or []
        if not features:
            return pn.pane.Markdown("### Drift Detector\n*No numeric features found.*")

        feature_select = pn.widgets.Select(
            name="Feature", options=features, value=features[0]
        )

        @pn.depends(feature_select)
        def selected_kde(feature):
            return pn.pane.HoloViews(self._kde_comparison(feature))

        return pn.Column(
            pn.Row(
                pn.pane.HoloViews(self._drift_scores_plot()),
                pn.Column("### Drift Summary", self._scores_table()),
            ),
            pn.layout.Divider(),
            pn.Row(
                pn.Column(feature_select, selected_kde),
                self._temporal_animation(),
            ),
            sizing_mode="stretch_width",
        )
