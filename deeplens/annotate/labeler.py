"""Active Learning Annotator — lasso-select, label, and export.

Provides an interactive annotation workflow built on top of model
uncertainty (entropy of prediction probabilities).  Users can lasso-
select uncertain regions in the embedding space, assign labels via a
text input, review a full annotation history with undo support, and
export the results as CSV or JSON for retraining.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import param

import holoviews as hv
import panel as pn

hv.extension("bokeh")

# ── Entropy helpers ──────────────────────────────────────────────────

def _entropy(probs: np.ndarray) -> np.ndarray:
    """Per-sample entropy from a (N, C) probability matrix.

    Uses base-2 logarithm so the result is in bits.  Zero probabilities
    are handled gracefully (0 * log(0) → 0).
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log2(probs), axis=1)


def _max_entropy(n_classes: int) -> float:
    """Maximum possible entropy for *n_classes* (uniform distribution)."""
    if n_classes <= 1:
        return 1.0
    return np.log2(n_classes)


# ── Main component ───────────────────────────────────────────────────

class ActiveLearningAnnotator(pn.viewable.Viewer):
    """Interactive annotation tool driven by model uncertainty.

    Parameters
    ----------
    state : object
        A ``DeepLensState`` instance carrying embeddings, probabilities,
        labels, and the shared ``annotations`` dict.  Typed as
        ``object`` to avoid circular imports.
    """

    state = param.ClassSelector(class_=object, doc="DeepLensState instance")

    # Annotation controls
    label_text = param.String(default="", doc="Label to assign to selected points")
    queue_size = param.Integer(default=20, bounds=(5, 200), doc="Number of points in suggestion queue")

    def __init__(self, **params):
        super().__init__(**params)

        # Internal bookkeeping
        self._annotation_log: list[dict[str, Any]] = []
        self._selection_stream = hv.streams.Selection1D()

        # ── Widgets ──────────────────────────────────────────────────
        self._label_input = pn.widgets.TextInput(
            name="Label", placeholder="Enter label for selected points…", width=250,
        )
        self._assign_btn = pn.widgets.Button(
            name="Assign Label", button_type="primary", width=120,
        )
        self._assign_btn.on_click(self._on_assign)

        self._suggest_btn = pn.widgets.Button(
            name="Suggest Next Batch", button_type="success", width=150,
        )
        self._suggest_btn.on_click(self._on_suggest)

        self._undo_btn = pn.widgets.Button(
            name="Undo Last", button_type="warning", width=100,
        )
        self._undo_btn.on_click(self._on_undo)

        self._export_csv_btn = pn.widgets.Button(
            name="Export CSV", button_type="default", width=100,
        )
        self._export_csv_btn.on_click(self._on_export_csv)

        self._export_json_btn = pn.widgets.Button(
            name="Export JSON", button_type="default", width=100,
        )
        self._export_json_btn.on_click(self._on_export_json)

        self._queue_slider = pn.widgets.IntSlider.from_param(
            self.param.queue_size, name="Queue size",
        )

        # Status / download panes
        self._status = pn.pane.Markdown("", width=350)
        self._download_pane = pn.Column(width=350)

        # History table (initially empty)
        self._history_table = pn.widgets.Tabulator(
            self._history_df(),
            layout="fit_columns",
            width=380,
            height=250,
            show_index=False,
            disabled=True,
        )

    # ── Entropy / uncertainty ────────────────────────────────────────

    def _compute_entropy(self) -> np.ndarray | None:
        """Return per-sample entropy or ``None`` if no probabilities."""
        if self.state is None:
            return None
        probs = getattr(self.state, "probabilities", None)
        if probs is None or len(probs) == 0:
            return None
        return _entropy(probs)

    # ── Plot ─────────────────────────────────────────────────────────

    @param.depends("state.embeddings_2d", "state.probabilities", "state.annotations")
    def _uncertainty_plot(self):
        """Scatter plot colored by prediction entropy (uncertainty)."""
        if self.state is None or not getattr(self.state, "has_embeddings", False):
            return hv.Text(0, 0, "No embeddings loaded").opts(text_font_size="14pt")

        emb = self.state.embeddings_2d
        entropy = self._compute_entropy()

        df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1]})

        if entropy is not None:
            df["uncertainty"] = entropy
        else:
            # Fallback: uniform uncertainty when no model is loaded
            df["uncertainty"] = 0.0

        # Mark already-annotated points
        annotations = getattr(self.state, "annotations", {}) or {}
        df["annotated"] = [
            annotations.get(i, "") for i in range(len(df))
        ]

        vdims = ["uncertainty", "annotated"]
        points = hv.Points(df, kdims=["x", "y"], vdims=vdims)

        self._selection_stream.source = points

        n_samples = len(df)
        opts = dict(
            width=700,
            height=500,
            color="uncertainty",
            cmap="RdYlGn_r",  # red = high entropy, green = low
            colorbar=True,
            size=5,
            alpha=0.7,
            tools=["lasso_select", "box_select", "hover", "wheel_zoom", "pan", "reset"],
            active_tools=["wheel_zoom"],
            title=f"Uncertainty Heatmap ({n_samples:,} points)",
            clabel="Entropy (bits)",
        )
        plot = points.opts(**opts)

        # Overlay annotated points as larger markers so they stand out
        annotated_mask = df["annotated"] != ""
        if annotated_mask.any():
            ann_df = df[annotated_mask].copy()
            ann_points = hv.Points(
                ann_df, kdims=["x", "y"], vdims=vdims,
            ).opts(
                size=10,
                marker="square",
                color="teal",
                alpha=0.9,
                line_color="white",
                line_width=1,
            )
            plot = plot * ann_points

        return plot

    # ── Suggestion queue ─────────────────────────────────────────────

    @param.depends("queue_size", "state.probabilities", "state.annotations")
    def _suggestion_queue(self):
        """Markdown listing the highest-uncertainty unlabeled points."""
        entropy = self._compute_entropy()
        if entropy is None:
            return pn.pane.Markdown(
                "### Suggestion Queue\n\n"
                "*Load a model with prediction probabilities to enable "
                "uncertainty-based suggestions.*",
                width=350,
            )

        annotations = getattr(self.state, "annotations", {}) or {}
        unlabeled_mask = np.array([
            i not in annotations for i in range(len(entropy))
        ])
        if not unlabeled_mask.any():
            return pn.pane.Markdown(
                "### Suggestion Queue\n\nAll points have been labeled!",
                width=350,
            )

        # Sort unlabeled by descending entropy
        indices = np.where(unlabeled_mask)[0]
        sorted_idx = indices[np.argsort(entropy[indices])[::-1]]
        top = sorted_idx[: self.queue_size]

        n_classes = self.state.probabilities.shape[1] if self.state.probabilities.ndim == 2 else 2
        max_ent = _max_entropy(n_classes)

        parts = [f"### Suggestion Queue (top {len(top)})\n"]
        for rank, idx in enumerate(top, 1):
            ent_val = entropy[idx]
            pct = ent_val / max_ent * 100 if max_ent > 0 else 0
            label_str = ""
            if self.state.labels is not None:
                label_str = f" | true={self.state.labels[idx]}"
            parts.append(f"{rank}. **idx {idx}** — entropy {ent_val:.3f} ({pct:.0f}%){label_str}")

        return pn.pane.Markdown("\n".join(parts), width=350)

    # ── Progress indicator ───────────────────────────────────────────

    @param.depends("state.annotations", "state.probabilities")
    def _progress_indicator(self):
        """Show annotation progress and estimated accuracy gain."""
        annotations = getattr(self.state, "annotations", {}) or {}
        n_labeled = len(annotations)
        n_total = self.state.n_samples if self.state is not None else 0

        if n_total == 0:
            return pn.pane.Markdown("*No data loaded.*", width=350)

        # Estimate accuracy gain: simple heuristic based on fraction of
        # uncertain points labeled.  A real system would retrain; here we
        # give a rough motivational signal.
        entropy = self._compute_entropy()
        gain_pct = 0.0
        uncertain_total = 0
        if entropy is not None:
            n_classes = (
                self.state.probabilities.shape[1]
                if self.state.probabilities is not None and self.state.probabilities.ndim == 2
                else 2
            )
            threshold = _max_entropy(n_classes) * 0.5
            uncertain_mask = entropy > threshold
            uncertain_total = int(uncertain_mask.sum())
            if uncertain_total > 0:
                labeled_uncertain = sum(
                    1 for i in annotations if i < len(entropy) and uncertain_mask[i]
                )
                # Rough estimate: labeling all uncertain points would give
                # ~5% accuracy gain (a strong simplifying assumption).
                gain_pct = (labeled_uncertain / uncertain_total) * 5.0

        indicators = pn.Row(
            pn.indicators.Number(
                name="Labeled",
                value=n_labeled,
                format="{value}",
                font_size="18pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Total",
                value=n_total,
                format="{value:,}",
                font_size="18pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Uncertain",
                value=uncertain_total,
                format="{value:,}",
                colors=[(0, "green"), (1, "orange")],
                font_size="18pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Est. Gain",
                value=round(gain_pct, 2),
                format="+{value:.2f}%",
                colors=[(0, "gray"), (1, "green")],
                font_size="18pt",
                title_size="10pt",
            ),
        )

        progress = pn.indicators.Progress(
            name="Annotation Progress",
            value=int(n_labeled / n_total * 100) if n_total else 0,
            max=100,
            bar_color="info",
            width=350,
        )

        return pn.Column(
            "### Progress",
            indicators,
            progress,
            width=370,
        )

    # ── History table ────────────────────────────────────────────────

    def _history_df(self) -> pd.DataFrame:
        """Build a DataFrame from the annotation log."""
        if not self._annotation_log:
            return pd.DataFrame(columns=["#", "Indices", "Label", "Count", "Time"])
        rows = []
        for i, entry in enumerate(self._annotation_log, 1):
            idx_str = str(entry["indices"][:5])
            if len(entry["indices"]) > 5:
                idx_str = idx_str[:-1] + ", …]"
            rows.append({
                "#": i,
                "Indices": idx_str,
                "Label": entry["label"],
                "Count": entry["count"],
                "Time": entry["time"],
            })
        return pd.DataFrame(rows)

    def _refresh_history(self):
        """Update the history Tabulator widget."""
        self._history_table.value = self._history_df()

    # ── Button callbacks ─────────────────────────────────────────────

    def _on_assign(self, event=None):
        """Assign the label to all currently selected points."""
        label = self._label_input.value.strip()
        if not label:
            self._status.object = "**Error:** Please enter a label."
            return

        indices = self._selection_stream.index
        if not indices:
            self._status.object = "**Error:** No points selected. Use lasso/box select first."
            return

        if self.state is None:
            self._status.object = "**Error:** No state loaded."
            return

        # Write to state.annotations
        annotations = dict(getattr(self.state, "annotations", {}) or {})
        for idx in indices:
            annotations[idx] = label
        self.state.annotations = annotations

        # Record in log for undo
        self._annotation_log.append({
            "indices": list(indices),
            "label": label,
            "count": len(indices),
            "time": time.strftime("%H:%M:%S"),
        })

        self._status.object = f"Assigned **'{label}'** to **{len(indices)}** points."
        self._refresh_history()

        # Trigger reactivity by touching the param
        self.param.trigger("state")

    def _on_suggest(self, event=None):
        """Select the top-N uncertain points by updating the selection stream."""
        entropy = self._compute_entropy()
        if entropy is None:
            self._status.object = "**Error:** No probabilities available for uncertainty ranking."
            return

        annotations = getattr(self.state, "annotations", {}) or {}
        unlabeled_mask = np.array([
            i not in annotations for i in range(len(entropy))
        ])
        if not unlabeled_mask.any():
            self._status.object = "All points have been labeled!"
            return

        indices = np.where(unlabeled_mask)[0]
        sorted_idx = indices[np.argsort(entropy[indices])[::-1]]
        top = sorted_idx[: self.queue_size]

        self._selection_stream.event(index=list(top))
        self.state.selected_indices = list(top)
        self._status.object = (
            f"Suggested **{len(top)}** highest-uncertainty points. "
            "Enter a label and click **Assign Label**."
        )

    def _on_undo(self, event=None):
        """Undo the last annotation batch."""
        if not self._annotation_log:
            self._status.object = "Nothing to undo."
            return

        last = self._annotation_log.pop()
        annotations = dict(getattr(self.state, "annotations", {}) or {})
        for idx in last["indices"]:
            annotations.pop(idx, None)
        self.state.annotations = annotations

        self._status.object = (
            f"Undone: removed **'{last['label']}'** from **{last['count']}** points."
        )
        self._refresh_history()
        self.param.trigger("state")

    # ── Export ────────────────────────────────────────────────────────

    def _annotations_df(self) -> pd.DataFrame:
        """Build an export-ready DataFrame of all annotations."""
        annotations = getattr(self.state, "annotations", {}) or {}
        if not annotations:
            return pd.DataFrame(columns=["index", "label"])

        records = [{"index": k, "label": v} for k, v in sorted(annotations.items())]
        df = pd.DataFrame(records)

        # Attach embedding coordinates if available
        if self.state is not None and getattr(self.state, "has_embeddings", False):
            emb = self.state.embeddings_2d
            xs, ys = [], []
            for idx in df["index"]:
                if idx < len(emb):
                    xs.append(emb[idx, 0])
                    ys.append(emb[idx, 1])
                else:
                    xs.append(np.nan)
                    ys.append(np.nan)
            df["emb_x"] = xs
            df["emb_y"] = ys

        # Attach original label if available
        if self.state is not None and getattr(self.state, "labels", None) is not None:
            labels = self.state.labels
            df["original_label"] = [
                labels[i] if i < len(labels) else None for i in df["index"]
            ]

        return df

    def _on_export_csv(self, event=None):
        """Provide a CSV download."""
        df = self._annotations_df()
        if df.empty:
            self._status.object = "**Error:** No annotations to export."
            return

        sio = io.StringIO()
        df.to_csv(sio, index=False)
        self._download_pane.clear()
        self._download_pane.append(
            pn.widgets.FileDownload(
                sio,
                filename="annotations.csv",
                button_type="success",
                label="Download CSV",
            )
        )
        self._status.object = f"CSV ready — **{len(df)}** annotations."

    def _on_export_json(self, event=None):
        """Provide a JSON download."""
        df = self._annotations_df()
        if df.empty:
            self._status.object = "**Error:** No annotations to export."
            return

        payload = df.to_dict(orient="records")
        sio = io.StringIO(json.dumps(payload, indent=2, default=str))
        self._download_pane.clear()
        self._download_pane.append(
            pn.widgets.FileDownload(
                sio,
                filename="annotations.json",
                button_type="success",
                label="Download JSON",
            )
        )
        self._status.object = f"JSON ready — **{len(df)}** annotations."

    # ── Layout ───────────────────────────────────────────────────────

    def __panel__(self):
        controls = pn.Column(
            "## Active Learning Annotator",
            pn.layout.Divider(),
            self._label_input,
            pn.Row(self._assign_btn, self._undo_btn),
            pn.Row(self._suggest_btn),
            self._queue_slider,
            pn.layout.Divider(),
            self._status,
            pn.layout.Divider(),
            "### Export",
            pn.Row(self._export_csv_btn, self._export_json_btn),
            self._download_pane,
            width=250,
            min_width=200,
        )

        if self.state is None:
            return pn.Row(
                controls,
                pn.pane.Markdown(
                    "*No state loaded. Provide a DeepLensState instance to enable annotation.*",
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
            )

        right_panel = pn.Column(
            self._progress_indicator,
            pn.layout.Divider(),
            self._suggestion_queue,
            pn.layout.Divider(),
            "### Annotation History",
            self._history_table,
            sizing_mode="stretch_width",
            min_width=250,
            max_width=400,
        )

        main_plot = pn.pane.HoloViews(
            self._uncertainty_plot,
            sizing_mode="stretch_both",
        )

        return pn.Row(
            controls,
            main_plot,
            right_panel,
            sizing_mode="stretch_both",
        )
