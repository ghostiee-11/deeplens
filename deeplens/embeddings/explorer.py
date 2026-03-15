"""Embedding Explorer — the hero visualization module.

Interactive scatter plot of 2-D embeddings rendered with Datashader,
featuring lasso selection, similarity search, auto-clustering, and
a linked details panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import param

import holoviews as hv
from holoviews.operation.datashader import rasterize, dynspread
import panel as pn

hv.extension("bokeh")

COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


class EmbeddingExplorer(pn.viewable.Viewer):
    """Interactive embedding space explorer.

    Features
    --------
    - Datashader rendering for 100K+ points via ``rasterize()``
    - Lasso / box selection → detail panel + cluster stats
    - Color by label, prediction, confidence, auto-cluster
    - Similarity search: tap a point → highlight k-nearest neighbors
    - Auto-clustering via KMeans/DBSCAN
    - Live DR parameter tuning via linked Panel widgets
    """

    state = param.ClassSelector(class_=object, doc="DeepLensState instance")

    color_by = param.Selector(
        objects=["label", "prediction", "confidence", "cluster", "error"],
        default="label",
    )
    point_size = param.Integer(default=5, bounds=(1, 20))
    k_neighbors = param.Integer(default=10, bounds=(1, 50), doc="k for similarity search")
    use_datashader = param.Boolean(default=True, doc="Use Datashader for large datasets")
    datashader_threshold = param.Integer(default=5000, doc="Use Datashader above this sample count")

    def __init__(self, **params):
        super().__init__(**params)
        self._selection_stream = hv.streams.Selection1D()
        self._tap_stream = hv.streams.Tap()

        # Widgets
        self._color_widget = pn.widgets.Select.from_param(
            self.param.color_by, name="Color by attribute",
        )
        self._size_widget = pn.widgets.IntSlider.from_param(
            self.param.point_size, name="Point size (px)",
        )
        self._k_widget = pn.widgets.IntSlider.from_param(
            self.param.k_neighbors, name="Nearest neighbors (k)",
        )

        # DR controls (if state has a reducer)
        self._dr_widgets = self._build_dr_widgets()

    def _build_dr_widgets(self) -> list:
        """Build DR parameter widgets from the state."""
        widgets = []
        if self.state is not None:
            widgets.append(
                pn.widgets.Select(
                    name="Reduction",
                    options=["pca", "tsne", "umap"],
                    value=getattr(self.state, "reduction_method", "pca"),
                )
            )
        return widgets

    def _get_plot_df(self) -> pd.DataFrame:
        """Build the DataFrame used for plotting."""
        if self.state is None or not self.state.has_embeddings:
            return pd.DataFrame({"x": [], "y": []})

        df = pd.DataFrame({
            "x": self.state.embeddings_2d[:, 0],
            "y": self.state.embeddings_2d[:, 1],
        })

        # Add color columns
        if self.state.labels is not None:
            df["label"] = self.state.labels
        if self.state.predictions is not None:
            df["prediction"] = self.state.predictions
        if self.state.probabilities is not None:
            df["confidence"] = np.max(self.state.probabilities, axis=1)
        if self.state.cluster_labels is not None:
            df["cluster"] = self.state.cluster_labels
        if self.state.predictions is not None and self.state.labels is not None:
            df["error"] = (self.state.predictions != self.state.labels).astype(int)

        # Auto-cluster if not yet done
        if "cluster" not in df.columns and len(df) > 0:
            from sklearn.cluster import KMeans

            n_clusters = min(self.state.n_clusters, len(df))
            if n_clusters >= 2:
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df["cluster"] = km.fit_predict(self.state.embeddings_2d).astype(str)
                self.state.cluster_labels = np.asarray(df["cluster"])

        return df

    @param.depends("color_by", "point_size")
    def _embedding_plot(self):
        """Create the main embedding scatter plot."""
        df = self._get_plot_df()
        if df.empty:
            return hv.Text(0, 0, "No embeddings loaded").opts(text_font_size="14pt")

        color_col = self.color_by if self.color_by in df.columns else None

        # Treat low-cardinality numeric columns as categorical (e.g. digit labels 0-9)
        _CATEGORICAL_COLS = {"label", "prediction", "cluster", "error"}
        if color_col and color_col in _CATEGORICAL_COLS and not pd.api.types.is_string_dtype(df[color_col]):
            n_unique = df[color_col].nunique()
            if n_unique <= 20:
                df[color_col] = df[color_col].astype(str)

        vdims = [c for c in ["label", "prediction", "confidence", "cluster", "error"] if c in df.columns]

        points = hv.Points(df, kdims=["x", "y"], vdims=vdims)

        n_samples = len(df)
        use_ds = self.use_datashader and n_samples > self.datashader_threshold

        _plot_opts = dict(
            responsive=True,
            min_width=400,
            height=500,
            toolbar="above",
            xlabel="Embedding dimension 1",
            ylabel="Embedding dimension 2",
        )

        if use_ds:
            if color_col and pd.api.types.is_string_dtype(df[color_col]):
                # Categorical: use rasterize with category aggregation
                plot = rasterize(points, aggregator="count").opts(
                    cnorm="log",
                    colorbar=True,
                    tools=["lasso_select", "box_select", "tap", "wheel_zoom", "pan", "reset"],
                    active_tools=["wheel_zoom"],
                    title=f"Embedding Space ({n_samples:,} points)",
                    **_plot_opts,
                )
                plot = dynspread(plot, threshold=0.5, max_px=3)
            else:
                plot = rasterize(points).opts(
                    cnorm="log",
                    colorbar=True,
                    tools=["lasso_select", "box_select", "tap", "wheel_zoom", "pan", "reset"],
                    active_tools=["wheel_zoom"],
                    title=f"Embedding Space ({n_samples:,} points)",
                    **_plot_opts,
                )
                plot = dynspread(plot, threshold=0.5, max_px=3)
        else:
            # Direct rendering for smaller datasets
            opts = dict(
                size=self.point_size,
                tools=["lasso_select", "box_select", "tap", "hover", "wheel_zoom", "pan", "reset"],
                active_tools=["wheel_zoom"],
                alpha=0.7,
                title=f"Embedding Space ({n_samples:,} points)",
                **_plot_opts,
            )
            if color_col and color_col in df.columns:
                if pd.api.types.is_string_dtype(df[color_col]):
                    opts["color"] = color_col
                    opts["cmap"] = "Category10"
                    opts["legend_position"] = "right"
                else:
                    opts["color"] = color_col
                    opts["colorbar"] = True
                    opts["cmap"] = "Viridis"
            plot = points.opts(**opts)

        # Attach selection stream
        self._selection_stream.source = points
        self._tap_stream.source = points

        return plot

    @param.depends("_selection_stream.index")
    def _selection_details(self):
        """Panel showing details of selected points."""
        indices = self._selection_stream.index
        if not indices:
            return pn.pane.Markdown(
                "### Selection Details\n\n"
                "*Lasso-select or box-select points to see details.*\n\n"
                "**Tip:** Click a single point for similarity search.",
                width=300,
            )

        # Update state
        self.state.selected_indices = list(indices)

        df = self.state.df
        if df is None:
            return pn.pane.Markdown("No data loaded")

        sel_df = df.iloc[indices].copy()
        n_sel = len(sel_df)

        parts = [f"### Selected: {n_sel:,} points\n"]

        # Class distribution
        if self.state.label_column and self.state.label_column in sel_df.columns:
            dist = sel_df[self.state.label_column].value_counts()
            parts.append("**Class distribution:**")
            for cls, count in dist.items():
                pct = count / n_sel * 100
                parts.append(f"- {cls}: {count} ({pct:.1f}%)")
            parts.append("")

        # Accuracy on selection (if model exists)
        if self.state.predictions is not None and self.state.labels is not None:
            sel_preds = self.state.predictions[indices]
            sel_labels = self.state.labels[indices]
            acc = np.mean(sel_preds == sel_labels)
            parts.append(f"**Selection accuracy:** {acc:.1%}\n")

        # Feature statistics
        if self.state.feature_columns:
            cols_present = [c for c in self.state.feature_columns if c in sel_df.columns]
            if cols_present:
                stats = sel_df[cols_present].describe().round(3)
                parts.append("**Feature statistics:**\n```")
                parts.append(stats.to_string())
                parts.append("```")

        md = pn.pane.Markdown("\n".join(parts), width=300)

        # Sample table
        display_cols = [c for c in sel_df.columns if c not in ("target",)][:8]
        table = pn.widgets.Tabulator(
            sel_df[display_cols].head(50),
            width=300,
            height=250,
            layout="fit_columns",
            show_index=True,
        )

        return pn.Column(md, table, width=370)

    @param.depends("_tap_stream.x", "_tap_stream.y")
    def _similarity_panel(self):
        """Show k-nearest neighbors when a point is tapped."""
        x, y = self._tap_stream.x, self._tap_stream.y
        if x is None or y is None or not self.state.has_embeddings:
            return pn.pane.Markdown("", width=300)

        # Find closest point to tap
        emb_2d = self.state.embeddings_2d
        dists_2d = np.sqrt((emb_2d[:, 0] - x) ** 2 + (emb_2d[:, 1] - y) ** 2)
        tapped_idx = int(np.argmin(dists_2d))

        # Find k-NN in the ORIGINAL embedding space
        if self.state.embeddings_raw is not None:
            from sklearn.metrics import pairwise_distances

            query = self.state.embeddings_raw[tapped_idx : tapped_idx + 1]
            dists = pairwise_distances(query, self.state.embeddings_raw, metric="cosine").ravel()
        else:
            dists = dists_2d

        k = min(self.k_neighbors, len(dists) - 1)
        nn_indices = np.argsort(dists)[1 : k + 1]  # exclude self
        nn_dists = dists[nn_indices]

        parts = [f"### Similarity Search (k={k})\n"]
        parts.append(f"**Tapped point:** index {tapped_idx}\n")

        if self.state.labels is not None:
            parts.append(f"**Label:** {self.state.labels[tapped_idx]}\n")

        parts.append("**Nearest neighbors:**\n")
        for rank, (idx, d) in enumerate(zip(nn_indices, nn_dists), 1):
            label = self.state.labels[idx] if self.state.labels is not None else "?"
            parts.append(f"{rank}. idx={idx}, dist={d:.4f}, label={label}")

        return pn.pane.Markdown("\n".join(parts), width=300)

    def _cluster_stats_panel(self):
        """Show statistics for auto-detected clusters."""
        df = self._get_plot_df()
        if "cluster" not in df.columns or df.empty:
            return pn.pane.Markdown("*No clusters computed*")

        cluster_counts = df["cluster"].value_counts().sort_index()
        parts = ["### Cluster Summary\n"]
        for cluster_id, count in cluster_counts.items():
            pct = count / len(df) * 100
            parts.append(f"- **Cluster {cluster_id}:** {count} ({pct:.1f}%)")

            # Accuracy per cluster if model exists
            if self.state.predictions is not None and self.state.labels is not None:
                mask = df["cluster"] == cluster_id
                cluster_preds = self.state.predictions[np.asarray(mask)]
                cluster_labels = self.state.labels[np.asarray(mask)]
                acc = np.mean(cluster_preds == cluster_labels)
                parts[-1] += f" — acc: {acc:.1%}"

        return pn.pane.Markdown("\n".join(parts), width=300)

    def _quality_indicators(self):
        """Show DR quality metrics as Panel indicators."""
        if self.state is None or not self.state.has_embeddings or self.state.embeddings_raw is None:
            return pn.pane.Markdown("")

        from deeplens.embeddings.reduce import DimensionalityReducer

        metrics = DimensionalityReducer.quality_metrics(
            self.state.embeddings_raw, self.state.embeddings_2d
        )

        return pn.Row(
            pn.indicators.Number(
                name="Trustworthiness",
                value=round(metrics["trustworthiness"], 3),
                format="{value:.3f}",
                colors=[(0.8, "red"), (0.9, "orange"), (1.0, "green")],
                font_size="18pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Stress",
                value=round(metrics["stress"], 3),
                format="{value:.3f}",
                colors=[(0.1, "green"), (0.3, "orange"), (1.0, "red")],
                font_size="18pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Samples",
                value=metrics["n_samples"],
                format="{value:,}",
                font_size="18pt",
                title_size="10pt",
            ),
        )

    def __panel__(self):
        controls = pn.Column(
            "## Controls",
            self._color_widget,
            self._size_widget,
            self._k_widget,
            *self._dr_widgets,
            pn.layout.Divider(),
            self._quality_indicators,
            pn.layout.Divider(),
            self._cluster_stats_panel,
            width=220,
            min_width=180,
            sizing_mode="fixed",
            scroll=True,
        )

        details = pn.Column(
            self._selection_details,
            pn.layout.Divider(),
            self._similarity_panel,
            width=250,
            min_width=200,
            sizing_mode="fixed",
            scroll=True,
        )

        main_plot = pn.pane.HoloViews(
            self._embedding_plot,
            sizing_mode="stretch_both",
            min_width=350,
            min_height=400,
        )

        return pn.Row(
            controls,
            main_plot,
            details,
            sizing_mode="stretch_both",
        )
