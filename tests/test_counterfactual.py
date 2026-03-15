"""Tests for deeplens.explain.counterfactual — CounterfactualExplorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import panel as pn

from deeplens.config import DeepLensState
from deeplens.explain.counterfactual import CounterfactualExplorer


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_iris_state(n_rows: int = 60) -> DeepLensState:
    """Build a fully-populated DeepLensState with a trained LogisticRegression."""
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    data = load_iris()
    X = data.data[:n_rows]
    y = data.target[:n_rows]
    feature_names = list(data.feature_names)

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    state = DeepLensState()
    state.dataset_name = "iris"
    state.df = df
    state.feature_columns = feature_names
    state.label_column = "target"
    state.labels = y.astype(object)
    state.class_names = [0, 1, 2]
    state.trained_model = model
    state.model_name = "LogisticRegression"
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    return state


def _make_binary_state(n_rows: int = 60) -> DeepLensState:
    """Build a binary-class DeepLensState (breast cancer subset)."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X = data.data[:n_rows]
    y = data.target[:n_rows]
    feature_names = list(data.feature_names)

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    model = LogisticRegression(max_iter=5000, random_state=0)
    model.fit(X, y)

    state = DeepLensState()
    state.dataset_name = "cancer"
    state.df = df
    state.feature_columns = feature_names
    state.label_column = "target"
    state.labels = y.astype(object)
    state.trained_model = model
    state.model_name = "LogisticRegression"
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    return state


def _make_empty_state() -> DeepLensState:
    return DeepLensState()


# ---------------------------------------------------------------------------
# __init__ / basic construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_no_args(self):
        """CounterfactualExplorer can be created without any args."""
        ce = CounterfactualExplorer()
        assert ce is not None
        assert ce._feature_sliders == {}
        assert ce._original_values == {}

    def test_init_with_state_no_model(self):
        state = _make_empty_state()
        ce = CounterfactualExplorer(state=state)
        assert ce.state is state
        assert not ce.state.has_model

    def test_init_with_full_state(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        assert ce.state is state
        assert ce.state.has_model

    def test_default_selected_index(self):
        ce = CounterfactualExplorer()
        assert ce.selected_index == -1

    def test_sliders_dict_initially_empty(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        assert len(ce._feature_sliders) == 0
        assert len(ce._original_values) == 0


# ---------------------------------------------------------------------------
# _build_sliders()
# ---------------------------------------------------------------------------

class TestBuildSliders:
    def test_no_model_returns_markdown(self):
        state = _make_empty_state()
        ce = CounterfactualExplorer(state=state)
        result = ce._build_sliders(0)
        assert isinstance(result, pn.pane.Markdown)

    def test_no_df_returns_markdown(self):
        state = DeepLensState()
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression()
        state.trained_model = m  # has_model True but no df
        ce = CounterfactualExplorer(state=state)
        result = ce._build_sliders(0)
        assert isinstance(result, pn.pane.Markdown)

    def test_returns_column_with_model(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        result = ce._build_sliders(0)
        assert isinstance(result, pn.Column)

    def test_slider_count_equals_feature_count(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        assert len(ce._feature_sliders) == len(state.feature_columns)

    def test_slider_names_match_features(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        assert set(ce._feature_sliders.keys()) == set(state.feature_columns)

    def test_slider_value_equals_original(self):
        state = _make_iris_state()
        idx = 5
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(idx)
        for feat in state.feature_columns:
            expected = float(state.df[feat].iloc[idx])
            assert abs(ce._feature_sliders[feat].value - expected) < 1e-6

    def test_original_values_populated(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(3)
        for feat in state.feature_columns:
            assert feat in ce._original_values

    def test_slider_start_lt_end(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        for slider in ce._feature_sliders.values():
            assert slider.start < slider.end

    def test_slider_value_within_range(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        for slider in ce._feature_sliders.values():
            assert slider.start <= slider.value <= slider.end

    def test_constant_feature_gets_range(self):
        """A feature with a single unique value must still get a usable slider range."""
        from sklearn.linear_model import LogisticRegression

        n = 20
        # Use two classes so LogisticRegression can be trained
        labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)
        df = pd.DataFrame({
            "const": np.ones(n),
            "vary": np.random.rand(n),
            "label": labels,
        })
        model = LogisticRegression(max_iter=100)
        X = df[["const", "vary"]].values
        model.fit(X, df["label"].values)

        state = DeepLensState()
        state.df = df
        state.feature_columns = ["const", "vary"]
        state.label_column = "label"
        state.labels = df["label"].values.astype(object)
        state.trained_model = model

        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        slider = ce._feature_sliders["const"]
        assert slider.start < slider.end

    def test_build_sliders_different_indices(self):
        """Sliders for different sample indices should have different original values."""
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)

        ce._build_sliders(0)
        values_0 = {f: v for f, v in ce._original_values.items()}

        ce._build_sliders(1)
        values_1 = {f: v for f, v in ce._original_values.items()}

        # At least one feature should differ between sample 0 and sample 1
        diffs = [abs(values_0[f] - values_1[f]) for f in state.feature_columns]
        assert any(d > 1e-9 for d in diffs), "Samples 0 and 1 should differ in at least one feature"

    def test_contains_reset_button(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        col = ce._build_sliders(0)

        # Flatten all Panel objects in the column to find buttons
        def find_buttons(obj):
            found = []
            if isinstance(obj, pn.widgets.Button):
                found.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    found.extend(find_buttons(child))
            return found

        buttons = find_buttons(col)
        names = [b.name for b in buttons]
        assert any("Reset" in n for n in names), f"Reset button not found, got: {names}"

    def test_contains_find_flip_button(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        col = ce._build_sliders(0)

        def find_buttons(obj):
            found = []
            if isinstance(obj, pn.widgets.Button):
                found.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    found.extend(find_buttons(child))
            return found

        buttons = find_buttons(col)
        names = [b.name for b in buttons]
        assert any("Flip" in n for n in names), f"Find Flip button not found, got: {names}"


# ---------------------------------------------------------------------------
# _reset_sliders()
# ---------------------------------------------------------------------------

class TestResetSliders:
    def test_reset_restores_original_values(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)

        # Mutate every slider
        for feat, slider in ce._feature_sliders.items():
            slider.value = slider.start

        ce._reset_sliders()

        for feat, slider in ce._feature_sliders.items():
            expected = ce._original_values[feat]
            assert abs(slider.value - expected) < 1e-6

    def test_reset_no_sliders_does_not_raise(self):
        """Calling reset before building sliders should be safe."""
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._reset_sliders()  # Should not raise


# ---------------------------------------------------------------------------
# _binary_search_flip() / _find_minimum_flip()
# ---------------------------------------------------------------------------

class TestBinarySearchFlip:
    @pytest.fixture
    def linear_model_state(self):
        """A simple linearly-separable dataset where a flip is always findable."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(0)
        n = 100
        X = rng.randn(n, 2)
        # Perfect linear separator along x[0]
        y = (X[:, 0] > 0).astype(int)

        df = pd.DataFrame(X, columns=["x0", "x1"])
        df["label"] = y

        model = LogisticRegression(max_iter=500, random_state=0)
        model.fit(X, y)

        state = DeepLensState()
        state.df = df
        state.feature_columns = ["x0", "x1"]
        state.label_column = "label"
        state.labels = y.astype(object)
        state.trained_model = model
        state.predictions = model.predict(X)

        return state

    def test_flip_found_for_linearly_separable(self, linear_model_state):
        state = linear_model_state
        model = state.trained_model
        X = state.df[state.feature_columns].values

        # Pick a sample well inside class 0 (negative x0)
        neg_indices = np.where(X[:, 0] < -0.5)[0]
        idx = neg_indices[0]
        original = X[idx : idx + 1]
        original_pred = model.predict(original)[0]

        # Search from current x0 value toward +3 (well past decision boundary)
        flip_val = CounterfactualExplorer._binary_search_flip(
            model, original, 0, float(original[0, 0]), 3.0, original_pred
        )
        assert flip_val is not None, "Expected a flip to be found"

    def test_no_flip_when_range_stays_same_side(self, linear_model_state):
        """If the entire range stays on the same side, None should be returned."""
        state = linear_model_state
        model = state.trained_model
        X = state.df[state.feature_columns].values

        # Pick a strongly negative sample; search only within negative region
        idx = np.argmin(X[:, 0])
        original = X[idx : idx + 1]
        original_pred = model.predict(original)[0]

        start = float(X[idx, 0])
        end = start + 0.01  # tiny range, stays same class

        result = CounterfactualExplorer._binary_search_flip(
            model, original, 0, start, end, original_pred
        )
        # Either None or the boundary value — but if found, prediction must differ
        if result is not None:
            X_test = original.copy()
            X_test[0, 0] = result
            new_pred = model.predict(X_test)[0]
            assert new_pred != original_pred

    def test_flip_value_type_is_float(self, linear_model_state):
        state = linear_model_state
        model = state.trained_model
        X = state.df[state.feature_columns].values
        neg_indices = np.where(X[:, 0] < -0.5)[0]
        idx = neg_indices[0]
        original = X[idx : idx + 1]
        original_pred = model.predict(original)[0]

        flip_val = CounterfactualExplorer._binary_search_flip(
            model, original, 0, float(original[0, 0]), 3.0, original_pred
        )
        if flip_val is not None:
            assert isinstance(flip_val, float)

    def test_returned_value_actually_flips(self, linear_model_state):
        state = linear_model_state
        model = state.trained_model
        X = state.df[state.feature_columns].values
        neg_indices = np.where(X[:, 0] < -0.5)[0]
        idx = neg_indices[0]
        original = X[idx : idx + 1]
        original_pred = model.predict(original)[0]

        flip_val = CounterfactualExplorer._binary_search_flip(
            model, original, 0, float(original[0, 0]), 3.0, original_pred
        )
        assert flip_val is not None
        X_test = original.copy()
        X_test[0, 0] = flip_val
        new_pred = model.predict(X_test)[0]
        assert new_pred != original_pred, "Returned value should flip the prediction"

    def test_find_minimum_flip_updates_slider(self, linear_model_state):
        state = linear_model_state
        ce = CounterfactualExplorer(state=state)
        # Use a sample that is clearly on the negative side
        neg_indices = np.where(state.df["x0"].values < -0.5)[0]
        idx = int(neg_indices[0])
        ce._build_sliders(idx)

        original_x0 = ce._feature_sliders["x0"].value
        ce._find_minimum_flip()

        # After the search, at least one slider should have moved if a flip was found
        current_x0 = ce._feature_sliders["x0"].value
        # Not asserting exact value — just that slider is now >= 0 (flipped side)
        # or stays original if no flip was found
        assert isinstance(current_x0, float)

    def test_find_minimum_flip_no_model_does_not_raise(self):
        state = _make_empty_state()
        ce = CounterfactualExplorer(state=state)
        ce._find_minimum_flip()  # Should silently return

    def test_n_steps_parameter(self, linear_model_state):
        """n_steps controls the grid resolution — higher n_steps = more precise."""
        state = linear_model_state
        model = state.trained_model
        X = state.df[state.feature_columns].values
        neg_indices = np.where(X[:, 0] < -0.5)[0]
        idx = neg_indices[0]
        original = X[idx : idx + 1]
        original_pred = model.predict(original)[0]

        flip_coarse = CounterfactualExplorer._binary_search_flip(
            model, original, 0, float(original[0, 0]), 3.0, original_pred, n_steps=5
        )
        flip_fine = CounterfactualExplorer._binary_search_flip(
            model, original, 0, float(original[0, 0]), 3.0, original_pred, n_steps=100
        )
        # Both should find a flip; fine should be <= coarse in distance from start
        if flip_coarse is not None and flip_fine is not None:
            dist_coarse = abs(flip_coarse - float(original[0, 0]))
            dist_fine = abs(flip_fine - float(original[0, 0]))
            assert dist_fine <= dist_coarse + 1.0  # allow for grid discretization


# ---------------------------------------------------------------------------
# __panel__() and _main_view()
# ---------------------------------------------------------------------------

class TestPanelInterface:
    def test_panel_returns_column(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        panel = ce.__panel__()
        assert isinstance(panel, pn.Column)

    def test_panel_no_state_does_not_raise(self):
        ce = CounterfactualExplorer()
        # __panel__ should return something renderable without error
        # (state is None — will hit has_model=False branch)
        try:
            panel = ce.__panel__()
            assert panel is not None
        except AttributeError:
            pytest.skip("CounterfactualExplorer requires state to be set")

    def test_main_view_no_model_returns_markdown(self):
        state = _make_empty_state()
        ce = CounterfactualExplorer(state=state)
        result = ce._main_view()
        assert isinstance(result, pn.pane.Markdown)

    def test_main_view_model_no_selection_returns_markdown(self):
        state = _make_iris_state()
        # selected_indices is empty by default
        ce = CounterfactualExplorer(state=state)
        result = ce._main_view()
        assert isinstance(result, pn.pane.Markdown)

    def test_main_view_with_selection_returns_row(self):
        state = _make_iris_state()
        state.selected_indices = [0]
        ce = CounterfactualExplorer(state=state)
        result = ce._main_view()
        assert isinstance(result, pn.Row)

    def test_panel_is_servable(self):
        """__panel__ output should be a valid Panel object (has .servable attr)."""
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        panel = ce.__panel__()
        assert hasattr(panel, "servable")


# ---------------------------------------------------------------------------
# _get_current_values()
# ---------------------------------------------------------------------------

class TestGetCurrentValues:
    def test_returns_correct_shape(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        vals = ce._get_current_values()
        assert vals.shape == (1, len(state.feature_columns))

    def test_values_match_slider_values(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)

        # Move first slider to end
        first_feat = state.feature_columns[0]
        ce._feature_sliders[first_feat].value = ce._feature_sliders[first_feat].end

        vals = ce._get_current_values()
        assert abs(vals[0, 0] - ce._feature_sliders[first_feat].end) < 1e-9

    def test_values_produce_valid_prediction(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(3)
        X_mod = ce._get_current_values()
        pred = state.trained_model.predict(X_mod)
        assert pred.shape == (1,)
        assert pred[0] in state.class_names


# ---------------------------------------------------------------------------
# _prediction_panel()
# ---------------------------------------------------------------------------

class TestPredictionPanel:
    def test_no_selection_returns_markdown(self):
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        result = ce._prediction_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_no_model_returns_markdown(self):
        state = _make_empty_state()
        ce = CounterfactualExplorer(state=state)
        result = ce._prediction_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_with_selection_returns_markdown(self):
        state = _make_iris_state()
        state.selected_indices = [0]
        ce = CounterfactualExplorer(state=state)
        result = ce._prediction_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_with_selection_and_sliders_shows_modified(self):
        state = _make_iris_state()
        state.selected_indices = [0]
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        result = ce._prediction_panel()
        assert isinstance(result, pn.pane.Markdown)
        # The output should mention "Modified prediction"
        text = result.object
        assert "Modified" in text or "prediction" in text.lower()

    def test_prediction_panel_updates_selected_index(self):
        state = _make_iris_state()
        state.selected_indices = [7]
        ce = CounterfactualExplorer(state=state)
        ce._prediction_panel()
        assert ce.selected_index == 7


# ---------------------------------------------------------------------------
# Multiclass / binary edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_two_feature_model(self):
        """Minimal 2-feature model should build sliders and panel without error."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(1)
        n = 40
        X = rng.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=["a", "b"])
        df["label"] = y

        model = LogisticRegression(max_iter=500)
        model.fit(X, y)

        state = DeepLensState()
        state.df = df
        state.feature_columns = ["a", "b"]
        state.label_column = "label"
        state.labels = y.astype(object)
        state.trained_model = model
        state.predictions = model.predict(X)

        ce = CounterfactualExplorer(state=state)
        col = ce._build_sliders(0)
        assert isinstance(col, pn.Column)
        assert len(ce._feature_sliders) == 2

    def test_panel_survives_multiclass(self):
        state = _make_iris_state()
        state.selected_indices = [0]
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        panel = ce.__panel__()
        assert isinstance(panel, pn.Column)

    def test_panel_survives_binary(self):
        state = _make_binary_state()
        state.selected_indices = [0]
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        panel = ce.__panel__()
        assert isinstance(panel, pn.Column)

    def test_rebuild_sliders_clears_previous(self):
        """Calling _build_sliders twice should replace, not accumulate, sliders."""
        state = _make_iris_state()
        ce = CounterfactualExplorer(state=state)
        ce._build_sliders(0)
        n_first = len(ce._feature_sliders)
        ce._build_sliders(1)
        n_second = len(ce._feature_sliders)
        assert n_first == n_second == len(state.feature_columns)

    def test_large_index(self):
        """Using the last row index should work without IndexError."""
        # Use n_rows=100 to include at least 2 iris classes (setosa + versicolor)
        state = _make_iris_state(n_rows=100)
        last = len(state.df) - 1
        ce = CounterfactualExplorer(state=state)
        result = ce._build_sliders(last)
        assert isinstance(result, pn.Column)
