"""Error Analysis module for DeepLens.

Provides interactive diagnostics focused on where and why a model fails:

- Misclassification scatter — 2-D embedding with errors highlighted in red
- Confusion pairs — bar chart of the most-confused class pairs
- Feature distributions — KDE overlays for correct vs. incorrect predictions
- Hardest samples — table ranked by confidence margin (lowest = hardest)
- Error rate by cluster — per-cluster error rate bar chart

The ``ErrorAnalyzer`` class follows the same ``pn.viewable.Viewer`` +
``param.depends`` pattern used by ``ModelInspector`` and ``ModelArena``.
It accepts either a populated ``DeepLensState`` *or* raw arrays for
standalone use.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import param

import holoviews as hv
import panel as pn

hv.extension("bokeh")

# ── colour palette ──────────────────────────────────────────────────────────
_CORRECT_COLOR = "#2ecc71"   # green
_ERROR_COLOR = "#e74c3c"     # red
_KDE_CORRECT = "#3498db"     # blue
_KDE_ERROR = "#e74c3c"       # red


class ErrorAnalyzer(pn.viewable.Viewer):
    """Interactive error-analysis dashboard.

    Features
    --------
    - **Misclassification scatter** — 2-D embedding coloured by
      correct (green) / incorrect (red).
    - **Confusion pairs** — ``hv.Bars`` of the *N* most-confused class pairs.
    - **Feature distributions** — KDE overlays comparing correctly vs.
      incorrectly classified samples for the top (or chosen) features.
    - **Hardest samples** — ``pn.widgets.Tabulator`` sorted by ascending
      confidence margin (``max_prob − second_max_prob``).
    - **Error rate by cluster** — bar chart of per-cluster error rates when
      cluster labels are available.

    Usage — with DeepLensState
    --------------------------
    >>> ea = ErrorAnalyzer(state=state)
    >>> ea.servable()

    Usage — standalone
    ------------------
    >>> ea = ErrorAnalyzer(
    ...     model=clf, X=X, y=y, embeddings_2d=emb_2d,
    ...     feature_names=feature_names,
    ... )
    >>> ea.servable()
    """

    # ── Public params ────────────────────────────────────────────────────
    state = param.ClassSelector(class_=object, default=None, doc="DeepLensState instance")

    # Standalone mode — populated when state is None
    model = param.Parameter(default=None, doc="Trained sklearn-compatible model")
    X = param.Array(default=None, doc="Feature matrix (n_samples, n_features)")
    y = param.Array(default=None, doc="True labels array")
    embeddings_2d = param.Array(default=None, doc="2-D reduced embeddings (n_samples, 2)")
    feature_names = param.List(default=[], doc="Feature column names")

    # Layout control
    top_n_pairs = param.Integer(default=10, bounds=(1, 50), doc="Number of confusion pairs to show")
    top_n_features = param.Integer(default=3, bounds=(1, 20), doc="Number of features for KDE")

    def __init__(self, **params):
        super().__init__(**params)
        # Pre-compute everything that doesn't depend on reactive params
        self._probabilities: np.ndarray | None = None
        self._refresh()

    # ── Private helpers ──────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Recompute derived arrays from whichever data source is active."""
        labels, predictions = self._labels(), self._predictions()
        if labels is None or predictions is None:
            return

        model = self._model()
        X = self._X()
        if model is not None and X is not None and hasattr(model, "predict_proba"):
            try:
                self._probabilities = model.predict_proba(X)
            except Exception:
                self._probabilities = None
        elif self.state is not None and self.state.probabilities is not None:
            self._probabilities = self.state.probabilities

    # ── Data accessors (normalise state vs. standalone) ──────────────────

    def _labels(self) -> np.ndarray | None:
        if self.state is not None and self.state.labels is not None:
            return np.asarray(self.state.labels)
        if self.y is not None:
            return np.asarray(self.y)
        return None

    def _predictions(self) -> np.ndarray | None:
        if self.state is not None and self.state.predictions is not None:
            return np.asarray(self.state.predictions)
        if self.model is not None and self.X is not None and self.y is not None:
            try:
                return np.asarray(self.model.predict(self.X))
            except Exception:
                return None
        return None

    def _model(self):
        if self.state is not None and self.state.trained_model is not None:
            return self.state.trained_model
        return self.model

    def _X(self) -> np.ndarray | None:
        if self.state is not None and self.state.df is not None and self.state.feature_columns:
            return self.state.df[self.state.feature_columns].values
        if self.X is not None:
            return np.asarray(self.X)
        return None

    def _embeddings_2d(self) -> np.ndarray | None:
        if self.state is not None and self.state.embeddings_2d is not None:
            return self.state.embeddings_2d
        return self.embeddings_2d

    def _feature_names(self) -> list[str]:
        if self.state is not None and self.state.feature_columns:
            return list(self.state.feature_columns)
        if self.feature_names:
            return list(self.feature_names)
        X = self._X()
        if X is not None:
            return [f"feature_{i}" for i in range(X.shape[1])]
        return []

    def _cluster_labels(self) -> np.ndarray | None:
        if self.state is not None and self.state.cluster_labels is not None:
            return np.asarray(self.state.cluster_labels)
        return None

    def _has_data(self) -> bool:
        return self._labels() is not None and self._predictions() is not None

    # ── Plot 1: Misclassification scatter ────────────────────────────────

    @param.depends("state.labels", "state.predictions", "state.embeddings_2d")
    def misclassification_scatter(self) -> hv.Element:
        """2-D scatter with misclassified points highlighted in red."""
        if not self._has_data():
            return hv.Text(0, 0, "No predictions available").opts(
                text_font_size="12pt",
            )

        emb = self._embeddings_2d()
        if emb is None or emb.ndim != 2 or emb.shape[1] != 2:
            return hv.Text(0, 0, "2-D embeddings required for scatter").opts(
                text_font_size="12pt",
            )

        labels = self._labels()
        preds = self._predictions()

        correct = (labels == preds).astype(str)
        # Map True/False → human-readable category for legend
        status = np.where(labels == preds, "Correct", "Error")

        df = pd.DataFrame({
            "x": emb[:, 0],
            "y": emb[:, 1],
            "status": status,
            "true": labels.astype(str),
            "predicted": preds.astype(str),
        })

        return hv.Points(
            df, kdims=["x", "y"], vdims=["status", "true", "predicted"],
        ).opts(
            color="status",
            cmap={"Correct": _CORRECT_COLOR, "Error": _ERROR_COLOR},
            size=5,
            alpha=0.75,
            width=600,
            height=450,
            tools=["hover", "lasso_select", "box_select"],
            legend_position="top_right",
            title="Misclassification Scatter",
            toolbar="above",
        )

    # ── Plot 2: Confusion pairs ──────────────────────────────────────────

    @param.depends("state.labels", "state.predictions", "top_n_pairs")
    def confusion_pairs(self) -> hv.Element:
        """Bar chart of the most-confused class pairs (true→predicted)."""
        if not self._has_data():
            return hv.Text(0, 0, "No predictions available").opts(
                text_font_size="12pt",
            )

        labels = self._labels()
        preds = self._predictions()

        # Only misclassified samples
        mask = labels != preds
        if not np.any(mask):
            return hv.Text(0, 0, "No misclassifications — perfect model!").opts(
                text_font_size="12pt",
            )

        true_err = labels[mask].astype(str)
        pred_err = preds[mask].astype(str)

        counter: Counter = Counter(zip(true_err, pred_err))
        top = counter.most_common(self.top_n_pairs)

        pairs = [f"{t}↔{p}" for (t, p), _ in top]
        counts = [cnt for _, cnt in top]

        return hv.Bars(
            list(zip(pairs, counts)),
            kdims=["Pair"],
            vdims=["Count"],
        ).opts(
            width=600,
            height=350,
            xrotation=45,
            color=_ERROR_COLOR,
            alpha=0.85,
            tools=["hover"],
            title=f"Top-{self.top_n_pairs} Confusion Pairs",
            toolbar="above",
        )

    # ── Plot 3: Feature distributions (KDE) ─────────────────────────────

    @param.depends("state.labels", "state.predictions", "top_n_features")
    def feature_distributions(self) -> hv.Element:
        """Overlaid KDE plots: correct (blue) vs. incorrect (red) predictions."""
        if not self._has_data():
            return hv.Text(0, 0, "No predictions available").opts(
                text_font_size="12pt",
            )

        X = self._X()
        if X is None:
            return hv.Text(0, 0, "Feature matrix (X) required for KDE").opts(
                text_font_size="12pt",
            )

        labels = self._labels()
        preds = self._predictions()
        feat_names = self._feature_names()

        correct_mask = labels == preds
        n_features = min(self.top_n_features, X.shape[1])

        overlays = []
        for i in range(n_features):
            name = feat_names[i] if i < len(feat_names) else f"feature_{i}"
            col_correct = X[correct_mask, i]
            col_error = X[~correct_mask, i]

            if len(col_correct) < 2 or len(col_error) < 2:
                continue

            kde_correct = hv.Distribution(col_correct, label="Correct").opts(
                color=_KDE_CORRECT, alpha=0.55,
            )
            kde_error = hv.Distribution(col_error, label="Error").opts(
                color=_KDE_ERROR, alpha=0.55,
            )
            overlay = (kde_correct * kde_error).opts(
                width=400,
                height=280,
                title=f"Feature: {name}",
                legend_position="top_right",
                toolbar="above",
            )
            overlays.append(overlay)

        if not overlays:
            return hv.Text(0, 0, "Insufficient data for KDE").opts(
                text_font_size="12pt",
            )

        # Stack overlays in a NdLayout for nice multi-panel display
        layout = hv.Layout(overlays).cols(min(3, len(overlays)))
        return layout

    # ── Plot 4: Hardest samples table ────────────────────────────────────

    @param.depends("state.labels", "state.predictions", "state.probabilities")
    def hardest_samples(self) -> pn.widgets.Tabulator:
        """Table of samples sorted by ascending confidence margin (hardest first)."""
        if not self._has_data():
            return pn.pane.Markdown("*No predictions available.*")

        self._refresh()
        labels = self._labels()
        preds = self._predictions()
        probs = self._probabilities

        records = []
        n = len(labels)

        if probs is not None and probs.ndim == 2 and probs.shape[0] == n:
            # Sort probabilities per row to get margin = top1 - top2
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            top1 = sorted_probs[:, 0]
            top2 = sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else np.zeros(n)
            margins = top1 - top2
            confidence = top1
        else:
            # No probabilities — use a uniform placeholder so table still renders
            margins = np.ones(n)
            confidence = np.ones(n)

        for idx in range(n):
            records.append({
                "index": int(idx),
                "true_label": str(labels[idx]),
                "predicted_label": str(preds[idx]),
                "confidence": float(round(confidence[idx], 4)),
                "margin": float(round(margins[idx], 4)),
                "correct": bool(labels[idx] == preds[idx]),
            })

        df = pd.DataFrame(records).sort_values("margin").reset_index(drop=True)

        return pn.widgets.Tabulator(
            df,
            width=700,
            height=320,
            layout="fit_columns",
            pagination="local",
            page_size=15,
            titles={
                "index": "Index",
                "true_label": "True Label",
                "predicted_label": "Predicted",
                "confidence": "Confidence",
                "margin": "Margin",
                "correct": "Correct",
            },
            formatters={
                "correct": {"type": "tickCross"},
                "confidence": {"type": "progress", "min": 0, "max": 1, "color": "blue"},
            },
        )

    # ── Plot 5: Error rate by cluster ────────────────────────────────────

    @param.depends(
        "state.labels", "state.predictions", "state.cluster_labels",
    )
    def error_rate_by_cluster(self) -> hv.Element:
        """Bar chart of per-cluster error rate (requires cluster labels)."""
        if not self._has_data():
            return hv.Text(0, 0, "No predictions available").opts(
                text_font_size="12pt",
            )

        cluster_labels = self._cluster_labels()
        if cluster_labels is None:
            return hv.Text(0, 0, "No cluster labels — run clustering first").opts(
                text_font_size="12pt",
            )

        labels = self._labels()
        preds = self._predictions()

        unique_clusters = sorted(set(cluster_labels))
        cluster_ids = []
        error_rates = []
        error_counts = []
        total_counts = []

        for cid in unique_clusters:
            mask = cluster_labels == cid
            n_total = int(np.sum(mask))
            n_errors = int(np.sum(labels[mask] != preds[mask]))
            rate = n_errors / n_total if n_total > 0 else 0.0
            cluster_ids.append(f"Cluster {cid}")
            error_rates.append(round(rate, 4))
            error_counts.append(n_errors)
            total_counts.append(n_total)

        df = pd.DataFrame({
            "cluster": cluster_ids,
            "error_rate": error_rates,
            "errors": error_counts,
            "total": total_counts,
        })

        return hv.Bars(
            df, kdims=["cluster"], vdims=["error_rate", "errors", "total"],
        ).opts(
            width=600,
            height=350,
            xrotation=45,
            color="error_rate",
            cmap="RdYlGn_r",
            clim=(0, 1),
            colorbar=True,
            tools=["hover"],
            ylim=(0, 1.05),
            title="Error Rate by Cluster",
            toolbar="above",
        )

    # ── Panel layout ─────────────────────────────────────────────────────

    def __panel__(self) -> pn.Column:
        if not self._has_data():
            return pn.Column(
                "## Error Analysis",
                pn.pane.Markdown(
                    "*No predictions found.  Train a model (or pass `model`, `X`, `y`) "
                    "to enable error analysis.*"
                ),
                sizing_mode="stretch_width",
            )

        scatter_pane = pn.pane.HoloViews(self.misclassification_scatter, linked_axes=False)
        pairs_pane = pn.pane.HoloViews(self.confusion_pairs, linked_axes=False)
        kde_pane = pn.pane.HoloViews(self.feature_distributions, linked_axes=False)
        cluster_pane = pn.pane.HoloViews(self.error_rate_by_cluster, linked_axes=False)

        controls = pn.WidgetBox(
            pn.widgets.IntSlider.from_param(self.param.top_n_pairs, name="# Confusion Pairs"),
            pn.widgets.IntSlider.from_param(self.param.top_n_features, name="# KDE Features"),
            width=220,
        )

        top_row = pn.Row(scatter_pane, pn.Column(pairs_pane, controls))

        return pn.Column(
            "## Error Analysis",
            pn.layout.Divider(),
            top_row,
            pn.layout.Divider(),
            "### Feature Distributions (Correct vs. Error)",
            kde_pane,
            pn.layout.Divider(),
            "### Hardest Samples (lowest confidence margin)",
            self.hardest_samples,
            pn.layout.Divider(),
            "### Error Rate by Cluster",
            cluster_pane,
            sizing_mode="stretch_width",
        )
