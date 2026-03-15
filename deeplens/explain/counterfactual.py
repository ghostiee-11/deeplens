"""Counterfactual Explorer — interactive what-if analysis.

Modify feature values with sliders and watch the prediction change
in real-time. Find the minimum change needed to flip a prediction.
"""

from __future__ import annotations

import numpy as np
import param

import holoviews as hv
import panel as pn

hv.extension("bokeh")


class CounterfactualExplorer(pn.viewable.Viewer):
    """Interactive counterfactual explanation explorer.

    Features
    --------
    - Pick a point → see its current prediction
    - Modify features with sliders → watch prediction probability update live
    - Auto-find minimum change to flip the prediction (binary search)
    - Feasibility constraints: highlights "unrealistic" changes
    - Path visualization: original → counterfactual trajectory in embedding space
    """

    state = param.ClassSelector(class_=object, doc="DeepLensState instance")
    selected_index = param.Integer(default=-1, doc="Index of the point being explored")

    def __init__(self, **params):
        super().__init__(**params)
        self._feature_sliders: dict[str, pn.widgets.FloatSlider] = {}
        self._original_values: dict[str, float] = {}

    def _build_sliders(self, idx: int) -> pn.Column:
        """Build feature sliders for the selected point."""
        if not self.state.has_model or self.state.df is None:
            return pn.pane.Markdown("*Train a model first.*")

        df = self.state.df
        features = self.state.feature_columns
        self._feature_sliders.clear()
        self._original_values.clear()

        sliders = []
        for feat in features:
            val = float(df[feat].iloc[idx])
            self._original_values[feat] = val

            col_min = float(df[feat].min())
            col_max = float(df[feat].max())
            if col_max == col_min:
                # Constant feature — create a small range around the value
                col_min = val - 1.0
                col_max = val + 1.0
            margin = (col_max - col_min) * 0.1
            col_min -= margin
            col_max += margin

            slider = pn.widgets.FloatSlider(
                name=feat,
                start=col_min,
                end=col_max,
                value=val,
                step=max((col_max - col_min) / 200, 1e-6),
            )
            self._feature_sliders[feat] = slider
            sliders.append(slider)

        # Watch all sliders
        for s in sliders:
            s.param.watch(lambda event: self._on_slider_change(), "value")

        reset_btn = pn.widgets.Button(
            name="Reset to Original",
            button_type="warning",
            sizing_mode="stretch_width",
            stylesheets=[":host .bk-btn { min-height: 36px; font-weight: 600; }"],
        )
        reset_btn.on_click(self._reset_sliders)

        find_btn = pn.widgets.Button(
            name="Find Minimum Flip",
            button_type="success",
            sizing_mode="stretch_width",
            stylesheets=[":host .bk-btn { min-height: 36px; font-weight: 600; }"],
        )
        find_btn.on_click(self._find_minimum_flip)

        return pn.Column(
            "### Modify Features",
            *sliders,
            pn.Row(reset_btn, find_btn, sizing_mode="stretch_width", margin=(10, 0)),
        )

    def _get_current_values(self) -> np.ndarray:
        """Get current slider values as a feature array."""
        features = self.state.feature_columns
        return np.array([self._feature_sliders[f].value for f in features]).reshape(1, -1)

    def _on_slider_change(self):
        """Called when any slider changes — update prediction display."""
        # Trigger re-render via param change
        self.param.trigger("selected_index")

    def _reset_sliders(self, event=None):
        """Reset all sliders to original values."""
        for feat, slider in self._feature_sliders.items():
            slider.value = self._original_values[feat]

    def _find_minimum_flip(self, event=None):
        """Find the minimum feature change that flips the prediction."""
        if not self.state.has_model:
            return

        model = self.state.trained_model
        features = self.state.feature_columns
        original = np.array([self._original_values[f] for f in features]).reshape(1, -1)
        original_pred = model.predict(original)[0]

        # Try each feature independently — binary search for flip point
        best_change = None
        best_magnitude = float("inf")

        for i, feat in enumerate(features):
            slider = self._feature_sliders[feat]
            low, high = slider.start, slider.end
            orig_val = self._original_values[feat]

            # Search toward max
            flip_val = self._binary_search_flip(
                model, original, i, orig_val, high, original_pred
            )
            if flip_val is not None:
                mag = abs(flip_val - orig_val)
                if mag < best_magnitude:
                    best_magnitude = mag
                    best_change = (feat, flip_val)

            # Search toward min
            flip_val = self._binary_search_flip(
                model, original, i, orig_val, low, original_pred
            )
            if flip_val is not None:
                mag = abs(flip_val - orig_val)
                if mag < best_magnitude:
                    best_magnitude = mag
                    best_change = (feat, flip_val)

        if best_change:
            feat, val = best_change
            self._feature_sliders[feat].value = val

    @staticmethod
    def _binary_search_flip(model, X, feature_idx, start_val, end_val, original_pred, n_steps=50):
        """Search for the closest value that flips the prediction.

        Uses a grid sweep (not binary search) to handle non-monotonic
        decision boundaries correctly.
        """
        X_test = X.copy()
        values = np.linspace(start_val, end_val, n_steps)
        best_val = None
        best_dist = float("inf")

        for val in values:
            X_test[0, feature_idx] = val
            pred = model.predict(X_test)[0]
            if pred != original_pred:
                dist = abs(val - start_val)
                if dist < best_dist:
                    best_dist = dist
                    best_val = float(val)

        return best_val

    @param.depends("state.selected_indices")
    def _prediction_panel(self):
        """Show current vs modified prediction."""
        if not self.state.selected_indices or not self.state.has_model:
            return pn.pane.Markdown(
                "### Counterfactual Explorer\n"
                "*Select a point in the embedding space, then modify features to see how the prediction changes.*"
            )

        idx = self.state.selected_indices[0]
        if idx != self.selected_index:
            self.selected_index = idx

        model = self.state.trained_model
        features = self.state.feature_columns

        # Original prediction
        original = self.state.df[features].values[idx : idx + 1]
        orig_pred = model.predict(original)[0]
        orig_label = self.state.labels[idx] if self.state.labels is not None else "?"

        parts = [f"### Point #{idx}", f"**True label:** {orig_label}", f"**Original prediction:** {orig_pred}"]

        # Modified prediction (if sliders exist)
        if self._feature_sliders:
            modified = self._get_current_values()
            mod_pred = model.predict(modified)[0]

            if hasattr(model, "predict_proba"):
                orig_proba = model.predict_proba(original)[0]
                mod_proba = model.predict_proba(modified)[0]

                parts.append(f"\n**Modified prediction:** {mod_pred}")
                parts.append(f"\n**Prediction flipped:** {'YES' if mod_pred != orig_pred else 'no'}")

                # Probability comparison
                parts.append("\n**Class probabilities:**\n")
                classes = model.classes_ if hasattr(model, "classes_") else range(len(orig_proba))
                for cls, op, mp in zip(classes, orig_proba, mod_proba):
                    delta = mp - op
                    arrow = "+" if delta > 0 else ""
                    parts.append(f"- {cls}: {op:.3f} → {mp:.3f} ({arrow}{delta:.3f})")

                # Feature changes
                parts.append("\n**Feature changes:**\n")
                for feat in features:
                    if feat in self._original_values and feat in self._feature_sliders:
                        orig = self._original_values[feat]
                        curr = self._feature_sliders[feat].value
                        if abs(curr - orig) > 1e-6:
                            parts.append(f"- **{feat}:** {orig:.4f} → {curr:.4f}")

        return pn.pane.Markdown("\n".join(parts), width=400)

    @param.depends("state.selected_indices")
    def _path_visualization(self):
        """Show the counterfactual path in embedding space."""
        if not self.state.selected_indices or not self.state.has_embeddings:
            return pn.pane.Markdown("")

        idx = self.state.selected_indices[0]
        orig_point = self.state.embeddings_2d[idx]

        # If we have modified values, compute the modified point's embedding position
        if not self._feature_sliders or self.state.embeddings_raw is None:
            return pn.pane.Markdown("")

        original_marker = hv.Points(
            [(orig_point[0], orig_point[1])], label="Original"
        ).opts(size=15, color="blue", marker="circle")

        return pn.pane.HoloViews(original_marker, width=300, height=300)

    @param.depends("state.selected_indices")
    def _main_view(self):
        if not self.state.has_model:
            return pn.pane.Markdown(
                "### Counterfactual Explorer\n\n"
                "*Train a model first using the sidebar, then select a point.*"
            )
        if not self.state.selected_indices:
            return pn.pane.Markdown(
                "### Counterfactual Explorer\n\n"
                "Select a point in the **Explore** tab to start.\n\n"
                "Use lasso or click a point, then come back here to modify "
                "features with sliders and explore counterfactuals.\n\n"
                "*What minimal change would flip this prediction?*"
            )

        idx = self.state.selected_indices[0]
        sliders = self._build_sliders(idx)

        return pn.Row(
            sliders,
            self._prediction_panel,
            sizing_mode="stretch_width",
        )

    def __panel__(self):
        return pn.Column(self._main_view, sizing_mode="stretch_both")
