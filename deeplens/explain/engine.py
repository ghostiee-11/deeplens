"""Explainability engine — connects SHAP to the embedding explorer via cross-filtering."""

from __future__ import annotations

import numpy as np
import param

import holoviews as hv
import panel as pn

from deeplens.explain import shap_plots

hv.extension("bokeh")


class ExplainabilityEngine(pn.viewable.Viewer):
    """Links SHAP explanations to the embedding space.

    - Tap a point in embedding space → see its SHAP waterfall
    - Lasso-select a cluster → see aggregate SHAP summary
    - Select features in importance chart → highlight in embedding space
    """

    state = param.ClassSelector(class_=object, doc="DeepLensState instance")
    max_display = param.Integer(default=15, bounds=(5, 50))

    def __init__(self, **params):
        super().__init__(**params)
        self._explainer = None
        self._shap_cache: dict[int, object] = {}

    def _get_explainer(self):
        """Lazy-init SHAP explainer."""
        if self._explainer is not None:
            return self._explainer

        if not self.state.has_model or not self.state.feature_columns:
            return None

        import shap

        model = self.state.trained_model
        X = self.state.df[self.state.feature_columns].values

        # Use TreeExplainer for tree-based models, else KernelExplainer
        try:
            self._explainer = shap.TreeExplainer(model)
        except Exception:
            background = shap.sample(X, min(100, len(X)))
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            self._explainer = shap.KernelExplainer(predict_fn, background)

        return self._explainer

    def _compute_shap_for_index(self, idx: int) -> dict:
        """Compute SHAP values for a single sample (cached)."""
        if idx in self._shap_cache:
            return self._shap_cache[idx]

        explainer = self._get_explainer()
        if explainer is None:
            return {}

        X = self.state.df[self.state.feature_columns].values[idx : idx + 1]
        sv = explainer.shap_values(X)

        result = {
            "shap_values": sv[0] if isinstance(sv, list) else sv.squeeze(),
            "base_value": (
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else float(explainer.expected_value)
            ),
        }
        self._shap_cache[idx] = result
        return result

    def _compute_shap_for_selection(self) -> dict | None:
        """Compute SHAP values for the current selection."""
        if not self.state.selected_indices or not self.state.has_model:
            return None

        explainer = self._get_explainer()
        if explainer is None:
            return None

        indices = self.state.selected_indices
        X = self.state.df[self.state.feature_columns].values[indices]
        sv = explainer.shap_values(X)

        if isinstance(sv, list):
            sv = sv[0]  # First class for multiclass

        return {
            "shap_values": sv,
            "base_value": (
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else float(explainer.expected_value)
            ),
        }

    @param.depends("state.selected_indices")
    def _waterfall_panel(self):
        """Show SHAP waterfall for the first selected point."""
        if not self.state.selected_indices or not self.state.has_model:
            return pn.pane.Markdown(
                "### SHAP Waterfall\n*Select a point to see its explanation.*"
            )

        idx = self.state.selected_indices[0]
        result = self._compute_shap_for_index(idx)
        if not result:
            return pn.pane.Markdown("*Train a model first to see SHAP explanations.*")

        sv = result["shap_values"]
        base = result["base_value"]

        plot = shap_plots.waterfall(
            sv, self.state.feature_columns, base_value=base, max_display=self.max_display
        )
        label = self.state.labels[idx] if self.state.labels is not None else "?"
        pred = self.state.predictions[idx] if self.state.predictions is not None else "?"

        header = pn.pane.Markdown(
            f"### Point #{idx}\n**Label:** {label} | **Predicted:** {pred}"
        )
        return pn.Column(header, pn.pane.HoloViews(plot))

    @param.depends("state.selected_indices")
    def _importance_panel(self):
        """Show feature importance for the current selection."""
        if not self.state.selected_indices or not self.state.has_model:
            return pn.pane.Markdown(
                "### Feature Importance\n*Select points to see aggregate importance.*"
            )

        result = self._compute_shap_for_selection()
        if result is None:
            return pn.pane.Markdown("*Could not compute SHAP values.*")

        sv = result["shap_values"]
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        plot = shap_plots.importance(sv, self.state.feature_columns, max_display=self.max_display)
        header = pn.pane.Markdown(
            f"### Feature Importance ({len(self.state.selected_indices)} selected)"
        )
        return pn.Column(header, pn.pane.HoloViews(plot))

    @param.depends("state.selected_indices")
    def _beeswarm_panel(self):
        """Show beeswarm for the current selection."""
        if not self.state.selected_indices or not self.state.has_model:
            return pn.pane.Markdown(
                "### SHAP Distribution\n*Select multiple points to see distribution.*"
            )

        if len(self.state.selected_indices) < 5:
            return pn.pane.Markdown(
                "### SHAP Distribution\n*Select at least 5 points for the beeswarm plot.*"
            )

        result = self._compute_shap_for_selection()
        if result is None:
            return pn.pane.Markdown("*Could not compute SHAP values.*")

        sv = result["shap_values"]
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        indices = self.state.selected_indices
        fv = self.state.df[self.state.feature_columns].values[indices]

        plot = shap_plots.beeswarm(sv, fv, self.state.feature_columns, max_display=self.max_display)
        return pn.Column(
            pn.pane.Markdown(f"### SHAP Distribution ({len(indices)} points)"),
            pn.pane.HoloViews(plot),
        )

    def __panel__(self):
        return pn.Tabs(
            ("Waterfall", self._waterfall_panel),
            ("Importance", self._importance_panel),
            ("Distribution", self._beeswarm_panel),
            dynamic=True,
        )
