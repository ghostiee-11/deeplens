"""Multi-Model Comparison Arena.

Compare 2+ models side-by-side with agreement/disagreement zone
visualization in the embedding space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import param

import holoviews as hv
import panel as pn

hv.extension("bokeh")

_ZONE_COLORS = {
    "both_correct": "#2ecc71",
    "both_wrong": "#e74c3c",
    "only_a_correct": "#3498db",
    "only_b_correct": "#f39c12",
}


class ModelArena(pn.viewable.Viewer):
    """Compare two models' predictions with agreement zone visualization.

    Features
    --------
    - Overlay predictions on the same embedding space
    - Agreement zones: both correct, both wrong, A-only correct, B-only correct
    - Metrics comparison table
    - Complementarity score (potential ensemble gain)
    """

    state = param.ClassSelector(class_=object, default=None, doc="Optional DeepLensState")
    model_a = param.Parameter(doc="First trained model")
    model_b = param.Parameter(doc="Second trained model")
    X = param.Array(doc="Feature matrix (n_samples, n_features)")
    y = param.Array(doc="True labels")
    feature_names = param.List(default=None, doc="Feature column names")
    embeddings_2d = param.Array(doc="2-D embeddings for visualization")

    def __init__(self, **params):
        super().__init__(**params)
        self._preds_a = None
        self._preds_b = None
        self._zones = None
        self._compute_predictions()

    def _compute_predictions(self):
        """Compute predictions and zones for both models."""
        if self.model_a is None or self.model_b is None or self.X is None or self.y is None:
            return

        self._preds_a = self.model_a.predict(self.X)
        self._preds_b = self.model_b.predict(self.X)

        a_correct = self._preds_a == self.y
        b_correct = self._preds_b == self.y

        zones = np.full(len(self.y), "both_wrong", dtype=object)
        zones[a_correct & b_correct] = "both_correct"
        zones[a_correct & ~b_correct] = "only_a_correct"
        zones[~a_correct & b_correct] = "only_b_correct"
        self._zones = zones

    def _get_embeddings_2d(self) -> np.ndarray:
        """Return 2-D embeddings, computing PCA if not provided."""
        if self.embeddings_2d is not None:
            return self.embeddings_2d
        from deeplens.embeddings.reduce import DimensionalityReducer

        self.embeddings_2d = DimensionalityReducer(method="pca").reduce(self.X)
        return self.embeddings_2d

    @param.depends("model_a", "model_b", "X", "y")
    def _agreement_plot(self) -> hv.Element:
        """Scatter plot colored by agreement zones."""
        if self._zones is None:
            return hv.Text(0, 0, "No predictions computed")
        emb = self._get_embeddings_2d()

        df = pd.DataFrame({
            "x": emb[:, 0],
            "y": emb[:, 1],
            "zone": self._zones,
            "pred_a": self._preds_a,
            "pred_b": self._preds_b,
            "true": self.y,
        })

        return hv.Points(
            df, kdims=["x", "y"], vdims=["zone", "pred_a", "pred_b", "true"]
        ).opts(
            color="zone",
            cmap=_ZONE_COLORS,
            size=5,
            alpha=0.7,
            width=650,
            height=450,
            tools=["hover", "lasso_select", "box_select"],
            legend_position="right",
            title="Model Agreement Zones",
        )

    @param.depends("model_a", "model_b", "X", "y")
    def _metrics_table(self):
        """Side-by-side metrics comparison."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        if self._preds_a is None or self._preds_b is None:
            return pn.pane.Markdown("*No predictions*")

        avg = "weighted" if len(np.unique(self.y)) > 2 else "binary"
        metrics = {}
        for name, preds in [("Model A", self._preds_a), ("Model B", self._preds_b)]:
            metrics[name] = {
                "Accuracy": accuracy_score(self.y, preds),
                "F1": f1_score(self.y, preds, average=avg, zero_division=0),
                "Precision": precision_score(self.y, preds, average=avg, zero_division=0),
                "Recall": recall_score(self.y, preds, average=avg, zero_division=0),
            }

        df = pd.DataFrame(metrics).round(4)
        return pn.widgets.Tabulator(df, width=400, layout="fit_columns")

    @param.depends("model_a", "model_b", "X", "y")
    def _zone_summary(self):
        """Summary of agreement/disagreement zones."""
        if self._zones is None:
            return pn.pane.Markdown("*No data*")

        total = len(self._zones)
        parts = ["### Agreement Zones\n"]

        labels = {
            "both_correct": "Both correct",
            "both_wrong": "Both wrong",
            "only_a_correct": "Only Model A correct",
            "only_b_correct": "Only Model B correct",
        }

        for zone_key, label in labels.items():
            count = int(np.sum(self._zones == zone_key))
            pct = count / total * 100
            parts.append(f"- **{label}:** {count} ({pct:.1f}%)")

        # Complementarity score
        a_correct = self._preds_a == self.y
        b_correct = self._preds_b == self.y
        union_correct = a_correct | b_correct
        ensemble_acc = float(np.mean(union_correct))
        best_single = max(float(np.mean(a_correct)), float(np.mean(b_correct)))
        gain = ensemble_acc - best_single

        parts.append(f"\n### Complementarity")
        parts.append(f"**Potential ensemble accuracy:** {ensemble_acc:.1%}")
        parts.append(f"**Gain over best single model:** +{gain:.1%}")

        return pn.pane.Markdown("\n".join(parts), sizing_mode="stretch_width", max_width=400)

    def __panel__(self):
        if self._zones is None:
            return pn.pane.Markdown("### Model Arena\n*Provide two trained models to compare.*")

        return pn.Column(
            pn.Row(
                pn.pane.HoloViews(self._agreement_plot),
                pn.Column(self._zone_summary, self._metrics_table),
            ),
            sizing_mode="stretch_width",
        )
