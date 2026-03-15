"""Model Inspector — confusion matrix, metrics, ROC curves, decision boundary.

Provides interactive model diagnostics as HoloViews elements with
cross-filter integration via ``DeepLensState``.  Clicking a confusion
matrix cell updates ``state.selected_indices`` to the corresponding
(mis)classified samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import param

import holoviews as hv
import panel as pn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)

hv.extension("bokeh")


class ModelInspector(pn.viewable.Viewer):
    """Interactive model performance inspector.

    Features
    --------
    - Confusion matrix (``hv.HeatMap``) with Tap stream for cross-filtering
    - Metrics dashboard (accuracy, F1, precision, recall) as ``pn.indicators.Number``
    - Per-class F1 bar chart (``hv.Bars``)
    - ROC curves with AUC annotation (one-vs-rest for multiclass)
    - Decision boundary overlay for 2-D embeddings (``hv.Image`` + ``hv.Points``)
    """

    state = param.ClassSelector(class_=object, doc="DeepLensState instance")

    # ── Internal bookkeeping ─────────────────────────────────────────────
    _tap_stream = param.ClassSelector(
        class_=hv.streams.Tap, default=None, precedence=-1,
    )

    def __init__(self, **params):
        super().__init__(**params)
        self._tap_stream = hv.streams.Tap()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _has_predictions(self) -> bool:
        """Return True when both labels and predictions are available."""
        if self.state is None:
            return False
        return (
            self.state.labels is not None
            and self.state.predictions is not None
            and len(self.state.labels) > 0
            and len(self.state.predictions) > 0
        )

    def _get_class_names(self) -> list[str]:
        """Ordered class names, falling back to unique sorted labels."""
        if self.state.class_names:
            return list(self.state.class_names)
        unique = sorted(set(self.state.labels) | set(self.state.predictions))
        return [str(c) for c in unique]

    def _is_binary(self) -> bool:
        return len(self._get_class_names()) == 2

    # ── Confusion matrix ────────────────────────────────────────────────

    @param.depends("state.labels", "state.predictions")
    def _confusion_matrix(self) -> hv.HeatMap:
        """Confusion matrix as ``hv.HeatMap`` with a Tap stream."""
        if not self._has_predictions():
            return hv.Text(0, 0, "No predictions available").opts(
                text_font_size="12pt",
            )

        class_names = self._get_class_names()
        cm = confusion_matrix(
            self.state.labels, self.state.predictions, labels=class_names,
        )

        records = []
        for i, true_cls in enumerate(class_names):
            for j, pred_cls in enumerate(class_names):
                records.append((pred_cls, true_cls, int(cm[i, j])))

        heatmap = hv.HeatMap(records, kdims=["Predicted", "True"], vdims=["Count"])

        # Attach tap stream for cross-filtering
        self._tap_stream.source = heatmap

        return heatmap.opts(
            tools=["hover", "tap"],
            colorbar=True,
            cmap="Blues",
            width=450,
            height=400,
            xrotation=45,
            title="Confusion Matrix",
            invert_yaxis=True,
            toolbar="above",
        )

    @param.depends("_tap_stream.x", "_tap_stream.y")
    def _on_cm_tap(self) -> pn.pane.Markdown:
        """React to a tap on the confusion matrix cell."""
        pred_cls = self._tap_stream.x
        true_cls = self._tap_stream.y

        if pred_cls is None or true_cls is None or not self._has_predictions():
            return pn.pane.Markdown(
                "*Click a confusion matrix cell to select those samples.*",
                width=300,
            )

        labels = np.asarray(self.state.labels)
        predictions = np.asarray(self.state.predictions)

        # Tap stream returns strings from HeatMap axis labels;
        # cast labels/predictions to str for consistent comparison
        labels_str = labels.astype(str)
        preds_str = predictions.astype(str)
        mask = (labels_str == str(true_cls)) & (preds_str == str(pred_cls))
        indices = list(np.where(mask)[0])
        self.state.selected_indices = indices

        n = len(indices)
        is_correct = str(true_cls) == str(pred_cls)
        kind = "correct" if is_correct else "misclassified"
        return pn.pane.Markdown(
            f"**Selected {n} {kind} samples**\n\n"
            f"True: **{true_cls}** | Predicted: **{pred_cls}**",
            width=300,
        )

    # ── Metrics dashboard ────────────────────────────────────────────────

    @param.depends("state.labels", "state.predictions")
    def _metrics_dashboard(self) -> pn.Row:
        """Accuracy, F1, precision, recall as indicator widgets."""
        if not self._has_predictions():
            return pn.pane.Markdown("*Train a model to see metrics.*")

        labels = self.state.labels
        predictions = self.state.predictions
        # Use "weighted" for both binary and multiclass to avoid pos_label
        # issues when class labels are strings rather than 0/1 integers.
        average = "weighted"

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average=average, zero_division=0)
        prec = precision_score(labels, predictions, average=average, zero_division=0)
        rec = recall_score(labels, predictions, average=average, zero_division=0)

        def _indicator(name: str, value: float) -> pn.indicators.Number:
            return pn.indicators.Number(
                name=name,
                value=round(value, 4),
                format="{value:.4f}",
                colors=[(0.6, "red"), (0.8, "orange"), (1.0, "green")],
                font_size="18pt",
                title_size="10pt",
            )

        return pn.Row(
            _indicator("Accuracy", acc),
            _indicator("F1", f1),
            _indicator("Precision", prec),
            _indicator("Recall", rec),
        )

    # ── Per-class F1 bars ────────────────────────────────────────────────

    @param.depends("state.labels", "state.predictions")
    def _per_class_f1(self) -> hv.Bars:
        """Per-class F1 score as ``hv.Bars``."""
        if not self._has_predictions():
            return hv.Text(0, 0, "No predictions available").opts(
                text_font_size="12pt",
            )

        class_names = self._get_class_names()
        f1_scores = f1_score(
            self.state.labels,
            self.state.predictions,
            labels=class_names,
            average=None,
            zero_division=0,
        )

        data = [(str(cls), float(score)) for cls, score in zip(class_names, f1_scores)]

        return hv.Bars(data, kdims=["Class"], vdims=["F1"]).opts(
            width=450,
            height=300,
            xrotation=45,
            color="F1",
            cmap="RdYlGn",
            clim=(0, 1),
            colorbar=True,
            title="Per-Class F1 Score",
            tools=["hover"],
            ylim=(0, 1.05),
            toolbar="above",
        )

    # ── ROC curves ───────────────────────────────────────────────────────

    @param.depends("state.labels", "state.predictions", "state.probabilities")
    def _roc_curves(self) -> hv.Overlay:
        """ROC curves with AUC annotation (one-vs-rest for multiclass)."""
        if not self._has_predictions() or self.state.probabilities is None:
            return hv.Text(0, 0, "No probabilities available").opts(
                text_font_size="12pt",
            )

        class_names = self._get_class_names()
        labels = self.state.labels
        probs = self.state.probabilities

        # Ensure labels are encoded as integers for binarisation
        label_to_idx = {c: i for i, c in enumerate(class_names)}
        y_true_idx = np.array([label_to_idx.get(l, -1) for l in labels])

        curves = []

        for i, cls in enumerate(class_names):
            binary_true = (y_true_idx == i).astype(int)

            # Handle case where probability matrix columns don't align
            if i >= probs.shape[1]:
                continue

            scores = probs[:, i]

            # Need both classes present for ROC
            if len(set(binary_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(binary_true, scores)
            roc_auc = auc(fpr, tpr)

            curve = hv.Curve(
                (fpr, tpr),
                kdims=["False Positive Rate"],
                vdims=["True Positive Rate"],
                label=f"{cls} (AUC={roc_auc:.3f})",
            )
            curves.append(curve)

        if not curves:
            return hv.Text(0, 0, "Cannot compute ROC").opts(text_font_size="12pt")

        # Diagonal reference line
        diag = hv.Curve(
            ([0, 1], [0, 1]),
            kdims=["False Positive Rate"],
            vdims=["True Positive Rate"],
            label="Random",
        ).opts(line_dash="dashed", color="gray", alpha=0.6)

        overlay = hv.Overlay([diag] + curves).opts(
            width=500,
            height=400,
            title="ROC Curves (One-vs-Rest)",
            legend_position="bottom_right",
            toolbar="above",
        )

        return overlay

    # ── Decision boundary ────────────────────────────────────────────────

    @param.depends(
        "state.trained_model",
        "state.embeddings_2d",
        "state.labels",
        "state.predictions",
    )
    def _decision_boundary(self) -> hv.Overlay:
        """Decision boundary for 2-D embeddings via mesh grid prediction."""
        if (
            self.state is None
            or not self.state.has_model
            or not self.state.has_embeddings
        ):
            return hv.Text(0, 0, "Need model + 2D embeddings").opts(
                text_font_size="12pt",
            )

        emb_2d = self.state.embeddings_2d
        if emb_2d.ndim != 2 or emb_2d.shape[1] != 2:
            return hv.Text(0, 0, "Decision boundary requires 2-D embeddings").opts(
                text_font_size="12pt",
            )

        model = self.state.trained_model

        # Check if model can predict on 2-D input
        # We need the model to have been trained on 2-D features
        try:
            model.predict(emb_2d[:1])
        except Exception:
            return hv.Text(
                0, 0, "Model not compatible with 2-D embeddings",
            ).opts(text_font_size="12pt")

        # Build mesh grid
        x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
        y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
        pad_x = (x_max - x_min) * 0.1
        pad_y = (y_max - y_min) * 0.1
        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y

        resolution = 200
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Predict on mesh
        try:
            if hasattr(model, "predict_proba"):
                zz = model.predict_proba(grid_points)
                # Use max probability as confidence surface
                zz = np.max(zz, axis=1).reshape(xx.shape)
            else:
                preds = model.predict(grid_points)
                # Encode predictions as integers for image rendering
                class_names = self._get_class_names()
                label_map = {c: i for i, c in enumerate(class_names)}
                zz = np.array([label_map.get(p, 0) for p in preds]).reshape(xx.shape)
        except Exception:
            return hv.Text(0, 0, "Error computing decision boundary").opts(
                text_font_size="12pt",
            )

        bounds = (x_min, y_min, x_max, y_max)
        img = hv.Image(zz, bounds=bounds, kdims=["x", "y"]).opts(
            cmap="RdYlBu",
            alpha=0.4,
            colorbar=True,
        )

        # Overlay training points
        df = pd.DataFrame({"x": emb_2d[:, 0], "y": emb_2d[:, 1]})
        if self.state.labels is not None:
            df["label"] = [str(l) for l in self.state.labels]
            vdims = ["label"]
        else:
            vdims = []

        points = hv.Points(df, kdims=["x", "y"], vdims=vdims)
        point_opts = dict(
            size=4,
            alpha=0.7,
            tools=["hover"],
        )
        if "label" in df.columns:
            point_opts["color"] = "label"
            point_opts["cmap"] = "Category10"

        points = points.opts(**point_opts)

        return (img * points).opts(
            width=500,
            height=400,
            title="Decision Boundary",
            toolbar="above",
        )

    # ── Panel layout ─────────────────────────────────────────────────────

    def __panel__(self) -> pn.Column:
        metrics_row = pn.panel(self._metrics_dashboard)

        cm_section = pn.Row(
            pn.pane.HoloViews(self._confusion_matrix, linked_axes=False),
            pn.Column(
                self._on_cm_tap,
                width=320,
            ),
        )

        bars_and_roc = pn.Row(
            pn.pane.HoloViews(self._per_class_f1, linked_axes=False),
            pn.pane.HoloViews(self._roc_curves, linked_axes=False),
        )

        boundary = pn.pane.HoloViews(self._decision_boundary, linked_axes=False)

        return pn.Column(
            "## Model Inspector",
            metrics_row,
            pn.layout.Divider(),
            cm_section,
            pn.layout.Divider(),
            bars_and_roc,
            pn.layout.Divider(),
            boundary,
            sizing_mode="stretch_width",
        )
