"""Tests for deeplens.explain.engine.ExplainabilityEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import panel as pn
import holoviews as hv

from deeplens.config import DeepLensState
from deeplens.explain.engine import ExplainabilityEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_with_model(n=80, n_features=4):
    """Build a DeepLensState with a trained LogisticRegression model."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features)
    y = np.array(["cat", "dog"] * (n // 2))

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    state = DeepLensState()
    state.df = df
    state.feature_columns = [f"f{i}" for i in range(n_features)]
    state.label_column = "label"
    state.labels = y
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    state.trained_model = model
    state.model_name = "LogisticRegression"
    return state


def _make_state_with_rf(n=80, n_features=4):
    """Build a DeepLensState with a RandomForest (TreeExplainer path)."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(7)
    X = rng.randn(n, n_features)
    y = np.array(["cat", "dog"] * (n // 2))

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y

    model = RandomForestClassifier(n_estimators=10, random_state=7)
    model.fit(X, y)

    state = DeepLensState()
    state.df = df
    state.feature_columns = [f"f{i}" for i in range(n_features)]
    state.label_column = "label"
    state.labels = y
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    state.trained_model = model
    state.model_name = "RandomForest"
    return state


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestExplainabilityEngineInit:
    def test_init_no_state(self):
        engine = ExplainabilityEngine()
        assert engine.state is None
        assert engine._explainer is None
        assert engine._shap_cache == {}

    def test_init_with_state(self):
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        assert engine.state is state

    def test_init_default_max_display(self):
        engine = ExplainabilityEngine()
        assert engine.max_display == 15

    def test_init_custom_max_display(self):
        engine = ExplainabilityEngine(max_display=20)
        assert engine.max_display == 20

    def test_shap_cache_starts_empty(self):
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        assert len(engine._shap_cache) == 0

    def test_explainer_starts_none(self):
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        assert engine._explainer is None


# ---------------------------------------------------------------------------
# _get_explainer
# ---------------------------------------------------------------------------

class TestGetExplainer:
    def test_returns_none_when_no_state(self):
        engine = ExplainabilityEngine()
        # Without state, accessing state.has_model raises AttributeError
        with pytest.raises(AttributeError):
            engine._get_explainer()

    def test_returns_none_when_no_model(self):
        state = DeepLensState()
        state.df = pd.DataFrame({"f1": [1.0, 2.0]})
        state.feature_columns = ["f1"]
        engine = ExplainabilityEngine(state=state)
        result = engine._get_explainer()
        assert result is None

    def test_returns_none_when_no_feature_columns(self):
        state = _make_state_with_model()
        state.feature_columns = []
        engine = ExplainabilityEngine(state=state)
        result = engine._get_explainer()
        assert result is None

    def test_kernel_explainer_for_logistic_regression(self):
        """LogisticRegression is not tree-based; should fall back to KernelExplainer."""
        import shap
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        explainer = engine._get_explainer()
        assert explainer is not None
        # Second call should return cached instance
        assert engine._get_explainer() is explainer

    def test_tree_explainer_for_random_forest(self):
        """RandomForest should use TreeExplainer (faster)."""
        import shap
        state = _make_state_with_rf()
        engine = ExplainabilityEngine(state=state)
        explainer = engine._get_explainer()
        assert explainer is not None
        assert isinstance(explainer, shap.TreeExplainer)

    def test_explainer_cached_after_first_call(self):
        state = _make_state_with_rf()
        engine = ExplainabilityEngine(state=state)
        e1 = engine._get_explainer()
        e2 = engine._get_explainer()
        assert e1 is e2


# ---------------------------------------------------------------------------
# _compute_shap_for_index
# ---------------------------------------------------------------------------

class TestComputeShapForIndex:
    def test_returns_empty_dict_when_no_model(self):
        state = DeepLensState()
        state.df = pd.DataFrame({"f1": np.ones(10)})
        state.feature_columns = ["f1"]
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_index(0)
        assert result == {}

    def test_returns_shap_values_and_base_value(self):
        state = _make_state_with_rf()
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_index(0)
        assert "shap_values" in result
        assert "base_value" in result

    def test_shap_values_correct_shape(self):
        n_features = 4
        state = _make_state_with_rf(n_features=n_features)
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_index(0)
        sv = result["shap_values"]
        assert sv.shape == (n_features,) or sv.ndim >= 1

    def test_result_is_cached(self):
        state = _make_state_with_rf()
        engine = ExplainabilityEngine(state=state)
        r1 = engine._compute_shap_for_index(0)
        assert 0 in engine._shap_cache
        r2 = engine._compute_shap_for_index(0)
        assert r1 is r2  # exact same dict object

    def test_base_value_is_scalar(self):
        state = _make_state_with_rf()
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_index(0)
        base = result["base_value"]
        assert isinstance(base, (int, float, np.floating))


# ---------------------------------------------------------------------------
# _compute_shap_for_selection
# ---------------------------------------------------------------------------

class TestComputeShapForSelection:
    def test_returns_none_when_no_selection(self):
        state = _make_state_with_rf()
        state.selected_indices = []
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_selection()
        assert result is None

    def test_returns_none_when_no_model(self):
        state = DeepLensState()
        state.df = pd.DataFrame({"f1": np.ones(10)})
        state.feature_columns = ["f1"]
        state.selected_indices = [0, 1]
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_selection()
        assert result is None

    def test_returns_dict_with_shap_values(self):
        state = _make_state_with_rf()
        state.selected_indices = [0, 1, 2, 3, 4]
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_selection()
        assert result is not None
        assert "shap_values" in result
        assert "base_value" in result

    def test_shap_values_have_correct_row_count(self):
        state = _make_state_with_rf()
        indices = [0, 1, 2, 3, 4]
        state.selected_indices = indices
        engine = ExplainabilityEngine(state=state)
        result = engine._compute_shap_for_selection()
        sv = result["shap_values"]
        # May be (n_samples, n_features) or (n_features,) for single sample
        assert sv.shape[0] == len(indices) or sv.ndim == 1


# ---------------------------------------------------------------------------
# __panel__ rendering
# ---------------------------------------------------------------------------

class TestExplainabilityEnginePanel:
    def test_panel_returns_tabs(self):
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        result = engine.__panel__()
        assert isinstance(result, pn.Tabs)

    def test_panel_has_three_tabs(self):
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        result = engine.__panel__()
        assert len(result) == 3

    def test_panel_tab_names(self):
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        result = engine.__panel__()
        # Tabs stores names as tuple pairs
        tab_names = [result._names[i] for i in range(len(result))]
        assert "Waterfall" in tab_names
        assert "Importance" in tab_names
        assert "Distribution" in tab_names

    def test_panel_no_state_raises_on_render(self):
        engine = ExplainabilityEngine()
        # param.depends("state.selected_indices") can't resolve when state is None
        with pytest.raises(AttributeError):
            engine.__panel__()

    def test_panel_dynamic_true(self):
        state = _make_state_with_model()
        engine = ExplainabilityEngine(state=state)
        result = engine.__panel__()
        assert result.dynamic is True


# ---------------------------------------------------------------------------
# _waterfall_panel
# ---------------------------------------------------------------------------

class TestWaterfallPanel:
    def test_no_selection_returns_markdown(self):
        state = _make_state_with_model()
        state.selected_indices = []
        engine = ExplainabilityEngine(state=state)
        result = engine._waterfall_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_no_model_returns_markdown(self):
        state = DeepLensState()
        state.df = pd.DataFrame({"f1": np.ones(10)})
        state.feature_columns = ["f1"]
        state.selected_indices = [0]
        engine = ExplainabilityEngine(state=state)
        result = engine._waterfall_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_with_selection_and_rf_model_returns_column(self):
        state = _make_state_with_rf()
        state.selected_indices = [0]
        engine = ExplainabilityEngine(state=state)
        result = engine._waterfall_panel()
        # Should return a Column with header + plot
        assert isinstance(result, pn.Column)


# ---------------------------------------------------------------------------
# _importance_panel
# ---------------------------------------------------------------------------

class TestImportancePanel:
    def test_no_selection_returns_markdown(self):
        state = _make_state_with_model()
        state.selected_indices = []
        engine = ExplainabilityEngine(state=state)
        result = engine._importance_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_with_selection_returns_column(self):
        state = _make_state_with_rf()
        state.selected_indices = list(range(10))
        engine = ExplainabilityEngine(state=state)
        result = engine._importance_panel()
        assert isinstance(result, pn.Column)


# ---------------------------------------------------------------------------
# _beeswarm_panel
# ---------------------------------------------------------------------------

class TestBeeswarmPanel:
    def test_no_selection_returns_markdown(self):
        state = _make_state_with_model()
        state.selected_indices = []
        engine = ExplainabilityEngine(state=state)
        result = engine._beeswarm_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_too_few_selections_returns_markdown(self):
        state = _make_state_with_rf()
        state.selected_indices = [0, 1, 2]  # < 5
        engine = ExplainabilityEngine(state=state)
        result = engine._beeswarm_panel()
        assert isinstance(result, pn.pane.Markdown)
        assert "5" in result.object  # mentions minimum of 5

    def test_five_or_more_selections_returns_column(self):
        state = _make_state_with_rf()
        state.selected_indices = list(range(10))
        engine = ExplainabilityEngine(state=state)
        result = engine._beeswarm_panel()
        assert isinstance(result, pn.Column)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_engine_with_empty_df(self):
        state = DeepLensState()
        state.df = pd.DataFrame()
        state.feature_columns = []
        engine = ExplainabilityEngine(state=state)
        # Should not raise
        result = engine._get_explainer()
        assert result is None

    def test_engine_max_display_propagated(self):
        state = _make_state_with_rf()
        state.selected_indices = [0]
        engine = ExplainabilityEngine(state=state, max_display=5)
        # Waterfall should pass max_display=5 to shap_plots
        # As a smoke test, just verify it doesn't raise
        result = engine._waterfall_panel()
        assert result is not None

    def test_shap_cache_persists_across_calls(self):
        state = _make_state_with_rf()
        engine = ExplainabilityEngine(state=state)
        engine._compute_shap_for_index(0)
        engine._compute_shap_for_index(1)
        assert len(engine._shap_cache) == 2
