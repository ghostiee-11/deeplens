"""Smart Dataset Profiler for DeepLens.

Provides a ``DatasetProfiler`` Panel Viewer that renders a comprehensive
overview of any tabular dataset: missing-value heatmaps, correlation matrices,
class balance charts, feature distribution grids, IQR-based outlier summaries,
and a data-type table.

It can be driven from a ``DeepLensState`` instance or used standalone by
passing ``df``, ``feature_columns``, and ``label_column`` directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import param
import panel as pn
import holoviews as hv

hv.extension("bokeh")

# ---------------------------------------------------------------------------
# Lazy import guard so the module is importable without a display server
# ---------------------------------------------------------------------------

_EMPTY_TEXT_OPTS = dict(text_font_size="11pt", text_align="center")


def _placeholder(msg: str) -> hv.Text:
    return hv.Text(0.5, 0.5, msg).opts(**_EMPTY_TEXT_OPTS)


# ---------------------------------------------------------------------------
# DatasetProfiler
# ---------------------------------------------------------------------------


class DatasetProfiler(pn.viewable.Viewer):
    """Interactive dataset profiling dashboard.

    Parameters
    ----------
    state:
        A ``DeepLensState`` instance.  When provided, ``df``,
        ``feature_columns``, and ``label_column`` are read reactively from it.
    df:
        A pandas ``DataFrame`` (standalone mode).
    feature_columns:
        List of numeric column names to profile.
    label_column:
        Name of the target/label column (optional).

    Features
    --------
    - Overview card — sample count, feature count, missing %, duplicates, memory
    - Missing-values heatmap
    - Correlation matrix heatmap with colorbar
    - Class-balance bar chart (when label column exists)
    - Feature distribution grid (small histograms)
    - Outlier summary table (IQR method)
    - Data-types summary table
    """

    # ── Public params ────────────────────────────────────────────────────
    state = param.ClassSelector(
        class_=object, default=None, doc="DeepLensState instance (optional)"
    )
    df = param.DataFrame(default=None, doc="Dataset (standalone mode)")
    feature_columns = param.List(default=[], doc="Numeric feature column names")
    label_column = param.String(default="", doc="Target / label column name")

    # ── Internal ─────────────────────────────────────────────────────────
    _n_hist_cols = param.Integer(default=3, precedence=-1)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_df(self) -> pd.DataFrame | None:
        """Return the active DataFrame from state or standalone param."""
        if self.state is not None and getattr(self.state, "df", None) is not None:
            return self.state.df
        return self.df

    def _resolve_features(self) -> list[str]:
        if self.state is not None and getattr(self.state, "feature_columns", []):
            return list(self.state.feature_columns)
        return list(self.feature_columns)

    def _resolve_label(self) -> str:
        if self.state is not None:
            return getattr(self.state, "label_column", "") or ""
        return self.label_column or ""

    def _watch_state(self) -> list[str]:
        """Return param dependency strings for state-driven updates."""
        return [
            "state.df",
            "state.feature_columns",
            "state.label_column",
            "df",
            "feature_columns",
            "label_column",
        ]

    # ── 1. Overview card ─────────────────────────────────────────────────

    @param.depends(
        "state.df", "state.feature_columns", "state.label_column",
        "df", "feature_columns", "label_column",
    )
    def _overview_card(self) -> pn.Row:
        """Sample count, feature count, missing %, duplicates, memory usage."""
        df = self._resolve_df()

        if df is None or df.empty:
            return pn.pane.Markdown("*Load a dataset to see the overview.*")

        n_samples = len(df)
        n_features = len(self._resolve_features())
        total_cells = df.size
        missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0.0
        n_duplicates = int(df.duplicated().sum())
        memory_kb = df.memory_usage(deep=True).sum() / 1024

        def _num(name: str, value, fmt: str = "{value}") -> pn.indicators.Number:
            return pn.indicators.Number(
                name=name,
                value=value,
                format=fmt,
                font_size="22pt",
                title_size="10pt",
            )

        return pn.Row(
            _num("Samples", n_samples),
            _num("Features", n_features),
            _num("Missing %", round(missing_pct, 2), fmt="{value:.2f}%"),
            _num("Duplicates", n_duplicates),
            _num("Memory (KB)", round(memory_kb, 1), fmt="{value:.1f} KB"),
            sizing_mode="stretch_width",
        )

    # ── 2. Missing-values heatmap ────────────────────────────────────────

    @param.depends(
        "state.df", "state.feature_columns",
        "df", "feature_columns",
    )
    def _missing_heatmap(self) -> hv.Element:
        """HeatMap showing missing value proportion per column."""
        df = self._resolve_df()
        if df is None or df.empty:
            return _placeholder("No data")

        cols = list(df.columns)
        n = len(df)

        records = []
        for col in cols:
            pct = df[col].isnull().mean() * 100
            records.append(("Missing %", col, round(pct, 2)))

        heatmap = hv.HeatMap(
            records,
            kdims=["Metric", "Column"],
            vdims=["Missing %"],
        )

        return heatmap.opts(
            title="Missing Values (%)",
            colorbar=True,
            cmap="Reds",
            xrotation=0,
            yrotation=0,
            width=max(300, min(80 * len(cols), 900)),
            height=150,
            tools=["hover"],
            toolbar="above",
            clim=(0, 100),
        )

    # ── 3. Correlation matrix ────────────────────────────────────────────

    @param.depends(
        "state.df", "state.feature_columns",
        "df", "feature_columns",
    )
    def _correlation_matrix(self) -> hv.Element:
        """HeatMap of Pearson correlations between numeric feature columns."""
        df = self._resolve_df()
        features = self._resolve_features()

        if df is None or df.empty or len(features) < 2:
            return _placeholder("Need >= 2 numeric features for correlation")

        num_df = df[features].select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return _placeholder("Not enough numeric columns")

        corr = num_df.corr()

        records = []
        for row_col in corr.columns:
            for col_col in corr.columns:
                val = corr.loc[row_col, col_col]
                records.append((col_col, row_col, round(float(val), 3)))

        side = max(350, min(70 * len(corr.columns), 700))

        heatmap = hv.HeatMap(
            records,
            kdims=["Column X", "Column Y"],
            vdims=["Correlation"],
        )

        return heatmap.opts(
            title="Feature Correlation Matrix",
            colorbar=True,
            cmap="RdBu_r",
            clim=(-1, 1),
            xrotation=45,
            width=side,
            height=side,
            tools=["hover"],
            toolbar="above",
        )

    # ── 4. Class balance ─────────────────────────────────────────────────

    @param.depends(
        "state.df", "state.label_column",
        "df", "label_column",
    )
    def _class_balance(self) -> hv.Element | pn.pane.Markdown:
        """Bar chart of target-class distribution."""
        df = self._resolve_df()
        label = self._resolve_label()

        if df is None or df.empty:
            return pn.pane.Markdown("*No data loaded.*")

        if not label or label not in df.columns:
            return pn.pane.Markdown("*No label column specified.*")

        counts = df[label].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        data = [(str(row["Class"]), int(row["Count"])) for _, row in counts.iterrows()]

        bars = hv.Bars(data, kdims=["Class"], vdims=["Count"])

        return bars.opts(
            title=f"Class Balance — {label}",
            width=450,
            height=300,
            xrotation=45,
            color="Count",
            cmap="Category10",
            tools=["hover"],
            toolbar="above",
        )

    # ── 5. Feature distributions (histogram grid) ────────────────────────

    @param.depends(
        "state.df", "state.feature_columns",
        "df", "feature_columns",
    )
    def _feature_distributions(self) -> pn.GridBox | pn.pane.Markdown:
        """Small sparkline-style histograms for each numeric feature."""
        df = self._resolve_df()
        features = self._resolve_features()

        if df is None or df.empty or not features:
            return pn.pane.Markdown("*No numeric features to plot.*")

        plots = []
        for col in features:
            series = df[col].dropna()
            if series.empty or not pd.api.types.is_numeric_dtype(series):
                continue

            counts, edges = np.histogram(series, bins=20)
            centers = (edges[:-1] + edges[1:]) / 2
            hist = hv.Histogram((edges, counts))

            plot = hist.opts(
                title=col,
                width=220,
                height=140,
                toolbar=None,
                tools=["hover"],
                color="#4C72B0",
                alpha=0.8,
                xlabel="",
                ylabel="Count",
            )
            plots.append(pn.pane.HoloViews(plot, linked_axes=False))

        if not plots:
            return pn.pane.Markdown("*No plottable numeric features.*")

        ncols = self._n_hist_cols
        return pn.GridBox(*plots, ncols=ncols, sizing_mode="stretch_width")

    # ── 6. Outlier summary ───────────────────────────────────────────────

    @param.depends(
        "state.df", "state.feature_columns",
        "df", "feature_columns",
    )
    def _outlier_summary(self) -> pn.widgets.DataFrame | pn.pane.Markdown:
        """IQR-based outlier detection table: column, count, percentage."""
        df = self._resolve_df()
        features = self._resolve_features()

        if df is None or df.empty or not features:
            return pn.pane.Markdown("*No numeric features available.*")

        rows = []
        for col in features:
            series = df[col].dropna()
            if series.empty or not pd.api.types.is_numeric_dtype(series):
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
            pct = round(n_outliers / len(series) * 100, 2) if len(series) > 0 else 0.0
            rows.append(
                {
                    "Column": col,
                    "Q1": round(float(q1), 4),
                    "Q3": round(float(q3), 4),
                    "IQR": round(float(iqr), 4),
                    "Lower fence": round(float(lower), 4),
                    "Upper fence": round(float(upper), 4),
                    "Outlier count": n_outliers,
                    "Outlier %": pct,
                }
            )

        if not rows:
            return pn.pane.Markdown("*No numeric columns to analyse.*")

        out_df = pd.DataFrame(rows)

        return pn.widgets.DataFrame(
            out_df,
            name="Outlier Summary (IQR)",
            show_index=False,
            sizing_mode="stretch_width",
            height=min(400, 40 + 35 * len(rows)),
        )

    # ── 7. Data-types summary ────────────────────────────────────────────

    @param.depends("state.df", "df")
    def _dtype_summary(self) -> pn.widgets.DataFrame | pn.pane.Markdown:
        """Table: column name, dtype, unique count, missing count."""
        df = self._resolve_df()

        if df is None or df.empty:
            return pn.pane.Markdown("*No data loaded.*")

        rows = []
        for col in df.columns:
            series = df[col]
            rows.append(
                {
                    "Column": col,
                    "Dtype": str(series.dtype),
                    "Unique values": int(series.nunique(dropna=False)),
                    "Missing count": int(series.isnull().sum()),
                    "Missing %": round(series.isnull().mean() * 100, 2),
                }
            )

        dtype_df = pd.DataFrame(rows)

        return pn.widgets.DataFrame(
            dtype_df,
            name="Data Types",
            show_index=False,
            sizing_mode="stretch_width",
            height=min(500, 40 + 35 * len(rows)),
        )

    # ── Panel layout ─────────────────────────────────────────────────────

    def __panel__(self) -> pn.Column:
        overview_section = pn.Column(
            "### Overview",
            self._overview_card(),
            sizing_mode="stretch_width",
        )

        dtype_section = pn.Column(
            "### Data Types Summary",
            self._dtype_summary(),
            sizing_mode="stretch_width",
        )

        missing_section = pn.Column(
            "### Missing Values Heatmap",
            pn.pane.HoloViews(self._missing_heatmap(), linked_axes=False),
            sizing_mode="stretch_width",
        )

        corr_section = pn.Column(
            "### Correlation Matrix",
            pn.pane.HoloViews(self._correlation_matrix(), linked_axes=False),
            sizing_mode="stretch_width",
        )

        balance_section = pn.Column(
            "### Class Balance",
            self._class_balance(),
            sizing_mode="stretch_width",
        )

        dist_section = pn.Column(
            "### Feature Distributions",
            self._feature_distributions(),
            sizing_mode="stretch_width",
        )

        outlier_section = pn.Column(
            "### Outlier Summary (IQR)",
            self._outlier_summary(),
            sizing_mode="stretch_width",
        )

        return pn.Column(
            "## Dataset Profiler",
            overview_section,
            pn.layout.Divider(),
            dtype_section,
            pn.layout.Divider(),
            missing_section,
            pn.layout.Divider(),
            corr_section,
            pn.layout.Divider(),
            balance_section,
            pn.layout.Divider(),
            dist_section,
            pn.layout.Divider(),
            outlier_section,
            sizing_mode="stretch_width",
        )
