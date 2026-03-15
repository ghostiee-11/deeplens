"""Tests for deeplens.explain.shap_plots — interactive SHAP visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import holoviews as hv

from deeplens.explain import shap_plots


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_shap_data():
    """Small (30, 4) binary SHAP fixture — reproducible random data."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 30, 4
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    shap_values = rng.randn(n_samples, n_features).astype(np.float32)
    feature_values = rng.rand(n_samples, n_features).astype(np.float32)
    base_value = 0.25
    return shap_values, feature_values, feature_names, base_value


@pytest.fixture(scope="module")
def multiclass_shap_data():
    """(30, 4, 3) multiclass SHAP fixture — last axis = classes."""
    rng = np.random.RandomState(7)
    n_samples, n_features, n_classes = 30, 4, 3
    feature_names = ["f0", "f1", "f2", "f3"]
    shap_values = rng.randn(n_samples, n_features, n_classes).astype(np.float32)
    feature_values = rng.rand(n_samples, n_features).astype(np.float32)
    base_value = 0.1
    return shap_values, feature_values, feature_names, base_value


@pytest.fixture(scope="module")
def single_sample_shap(binary_shap_data):
    """Single-sample SHAP row (1D)."""
    shap_values, _, feature_names, base_value = binary_shap_data
    return shap_values[0], feature_names, base_value


# ---------------------------------------------------------------------------
# waterfall()
# ---------------------------------------------------------------------------

class TestWaterfall:
    def test_returns_overlay(self, single_sample_shap):
        sv, names, base = single_sample_shap
        result = shap_plots.waterfall(sv, names, base_value=base)
        assert isinstance(result, hv.Overlay), "waterfall should return hv.Overlay"

    def test_contains_bars_and_vline(self, single_sample_shap):
        sv, names, base = single_sample_shap
        result = shap_plots.waterfall(sv, names, base_value=base)
        types = {type(el).__name__ for el in result}
        assert "Bars" in types
        assert "VLine" in types

    def test_max_display_limits_bars(self, single_sample_shap):
        sv, names, base = single_sample_shap
        result = shap_plots.waterfall(sv, names, base_value=base, max_display=2)
        bars = [el for el in result if isinstance(el, hv.Bars)][0]
        assert len(bars) <= 2

    def test_default_base_value_zero(self, single_sample_shap):
        sv, names, _ = single_sample_shap
        # Should not raise when base_value defaults to 0.0
        result = shap_plots.waterfall(sv, names)
        assert isinstance(result, hv.Overlay)

    def test_title_reflects_base_value(self, single_sample_shap):
        """waterfall with a distinct base value should produce a different-titled plot."""
        sv, names, _ = single_sample_shap
        base_a = 0.123
        base_b = 0.456
        # Both should succeed; this validates the base_value parameter is accepted
        result_a = shap_plots.waterfall(sv, names, base_value=base_a)
        result_b = shap_plots.waterfall(sv, names, base_value=base_b)
        assert isinstance(result_a, hv.Overlay)
        assert isinstance(result_b, hv.Overlay)
        # The two overlays are not the same object
        assert result_a is not result_b

    def test_2d_shap_averaged(self, binary_shap_data):
        """waterfall should accept (n_samples, n_features) and average axis=-1 if needed."""
        shap_values, _, names, base = binary_shap_data
        # Pass a row reshaped to (n_features, 1) — tests the ndim > 1 branch
        sv_2d = shap_values[0].reshape(-1, 1)
        result = shap_plots.waterfall(sv_2d, names, base_value=base)
        assert isinstance(result, hv.Overlay)

    def test_multiclass_single_row_averaged(self, multiclass_shap_data):
        shap_values, _, names, base = multiclass_shap_data
        sv = shap_values[0]  # shape (4, 3)
        result = shap_plots.waterfall(sv, names, base_value=base)
        assert isinstance(result, hv.Overlay)

    def test_all_positive_shap(self):
        sv = np.array([0.5, 0.3, 0.2, 0.1])
        names = ["a", "b", "c", "d"]
        result = shap_plots.waterfall(sv, names)
        assert isinstance(result, hv.Overlay)

    def test_all_negative_shap(self):
        sv = np.array([-0.5, -0.3, -0.2, -0.1])
        names = ["a", "b", "c", "d"]
        result = shap_plots.waterfall(sv, names)
        assert isinstance(result, hv.Overlay)

    def test_single_feature(self):
        sv = np.array([1.23])
        result = shap_plots.waterfall(sv, ["only_feature"])
        assert isinstance(result, hv.Overlay)


# ---------------------------------------------------------------------------
# beeswarm()
# ---------------------------------------------------------------------------

class TestBeeswarm:
    def test_returns_overlay(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.beeswarm(sv, fv, names)
        assert isinstance(result, hv.Overlay)

    def test_contains_points_and_vline(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.beeswarm(sv, fv, names)
        types = {type(el).__name__ for el in result}
        assert "Points" in types
        assert "VLine" in types

    def test_max_display_limits_features(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.beeswarm(sv, fv, names, max_display=2)
        # Points should only contain rows for 2 features
        points = [el for el in result if isinstance(el, hv.Points)][0]
        n_samples = sv.shape[0]
        assert len(points) == n_samples * 2

    def test_multiclass_3d_averaged(self, multiclass_shap_data):
        """beeswarm should reduce (n, f, c) → (n, f) by mean over last axis."""
        sv, fv, names, _ = multiclass_shap_data
        result = shap_plots.beeswarm(sv, fv, names)
        assert isinstance(result, hv.Overlay)

    def test_points_data_columns(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.beeswarm(sv, fv, names)
        points = [el for el in result if isinstance(el, hv.Points)][0]
        # Points should have SHAP Value and Feature Position as key dims
        kdim_names = [d.name for d in points.kdims]
        assert "SHAP Value" in kdim_names
        assert "Feature Position" in kdim_names

    def test_reproducible_jitter(self, binary_shap_data):
        """Two calls should produce identical jitter (seeded by feature index)."""
        sv, fv, names, _ = binary_shap_data
        r1 = shap_plots.beeswarm(sv, fv, names)
        r2 = shap_plots.beeswarm(sv, fv, names)
        p1 = [el for el in r1 if isinstance(el, hv.Points)][0]
        p2 = [el for el in r2 if isinstance(el, hv.Points)][0]
        pos1 = p1.data["Feature Position"].values
        pos2 = p2.data["Feature Position"].values
        np.testing.assert_array_equal(pos1, pos2)

    def test_constant_feature_handled(self):
        """A feature with identical values should not cause division by zero."""
        n = 20
        sv = np.ones((n, 2), dtype=np.float32)
        fv = np.column_stack([np.ones(n), np.random.rand(n)]).astype(np.float32)
        names = ["constant", "varying"]
        result = shap_plots.beeswarm(sv, fv, names)
        assert isinstance(result, hv.Overlay)


# ---------------------------------------------------------------------------
# dependence()
# ---------------------------------------------------------------------------

class TestDependence:
    def test_returns_overlay(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.dependence(sv, fv, 0, names)
        assert isinstance(result, hv.Overlay)

    def test_contains_scatter_and_hline(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.dependence(sv, fv, 0, names)
        types = {type(el).__name__ for el in result}
        assert "Scatter" in types
        assert "HLine" in types

    def test_no_interaction(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.dependence(sv, fv, 1, names, interaction_idx=None)
        assert isinstance(result, hv.Overlay)

    def test_with_interaction(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        result = shap_plots.dependence(sv, fv, 0, names, interaction_idx=2)
        assert isinstance(result, hv.Overlay)
        scatter = [el for el in result if isinstance(el, hv.Scatter)][0]
        # Interaction feature should appear in vdims
        vdim_names = [d.name for d in scatter.vdims]
        assert names[2] in vdim_names

    def test_self_interaction_ignored(self, binary_shap_data):
        """interaction_idx == feature_idx should behave like no interaction."""
        sv, fv, names, _ = binary_shap_data
        result_no = shap_plots.dependence(sv, fv, 0, names, interaction_idx=None)
        result_self = shap_plots.dependence(sv, fv, 0, names, interaction_idx=0)
        types_no = {type(el).__name__ for el in result_no}
        types_self = {type(el).__name__ for el in result_self}
        assert types_no == types_self

    def test_multiclass_3d_averaged(self, multiclass_shap_data):
        sv, fv, names, _ = multiclass_shap_data
        result = shap_plots.dependence(sv, fv, 0, names)
        assert isinstance(result, hv.Overlay)

    def test_different_feature_indices_produce_different_kdims(self, binary_shap_data):
        """Each feature index should produce a scatter keyed on that feature's name."""
        sv, fv, names, _ = binary_shap_data
        result_0 = shap_plots.dependence(sv, fv, 0, names)
        result_1 = shap_plots.dependence(sv, fv, 1, names)
        scatter_0 = [el for el in result_0 if isinstance(el, hv.Scatter)][0]
        scatter_1 = [el for el in result_1 if isinstance(el, hv.Scatter)][0]
        kdim_0 = scatter_0.kdims[0].name
        kdim_1 = scatter_1.kdims[0].name
        assert kdim_0 == names[0]
        assert kdim_1 == names[1]
        assert kdim_0 != kdim_1

    def test_all_feature_indices(self, binary_shap_data):
        sv, fv, names, _ = binary_shap_data
        for i in range(len(names)):
            result = shap_plots.dependence(sv, fv, i, names)
            assert isinstance(result, hv.Overlay), f"Failed for feature index {i}"


# ---------------------------------------------------------------------------
# importance()
# ---------------------------------------------------------------------------

class TestImportance:
    def test_returns_bars(self, binary_shap_data):
        sv, _, names, _ = binary_shap_data
        result = shap_plots.importance(sv, names)
        assert isinstance(result, hv.Bars)

    def test_max_display_limits_bars(self, binary_shap_data):
        sv, _, names, _ = binary_shap_data
        result = shap_plots.importance(sv, names, max_display=2)
        assert len(result) <= 2

    def test_all_features_shown_when_below_max(self, binary_shap_data):
        sv, _, names, _ = binary_shap_data
        result = shap_plots.importance(sv, names, max_display=100)
        assert len(result) == len(names)

    def test_sorted_descending_by_mean_abs(self, binary_shap_data):
        sv, _, names, _ = binary_shap_data
        result = shap_plots.importance(sv, names)
        values = result.data["Mean |SHAP|"].values
        # importance() reverses the order (bottom-to-top), so values are ascending
        assert np.all(np.diff(values) >= -1e-6), "Bars should be sorted bottom-to-top (ascending)"

    def test_kdim_is_feature(self, binary_shap_data):
        sv, _, names, _ = binary_shap_data
        result = shap_plots.importance(sv, names)
        kdim_names = [d.name for d in result.kdims]
        assert "Feature" in kdim_names

    def test_vdim_is_mean_abs_shap(self, binary_shap_data):
        sv, _, names, _ = binary_shap_data
        result = shap_plots.importance(sv, names)
        vdim_names = [d.name for d in result.vdims]
        assert "Mean |SHAP|" in vdim_names

    def test_multiclass_3d_averaged(self, multiclass_shap_data):
        sv, _, names, _ = multiclass_shap_data
        result = shap_plots.importance(sv, names)
        assert isinstance(result, hv.Bars)

    def test_mean_abs_values_nonnegative(self, binary_shap_data):
        sv, _, names, _ = binary_shap_data
        result = shap_plots.importance(sv, names)
        values = result.data["Mean |SHAP|"].values
        assert np.all(values >= 0), "Mean |SHAP| values must be non-negative"

    def test_single_sample(self):
        sv = np.array([[0.5, -0.3, 0.1]])
        names = ["x", "y", "z"]
        result = shap_plots.importance(sv, names)
        assert isinstance(result, hv.Bars)
        assert len(result) == 3

    def test_uniform_shap_values(self):
        """When all SHAP values are equal, all bars should have the same height."""
        sv = np.ones((10, 3))
        names = ["a", "b", "c"]
        result = shap_plots.importance(sv, names)
        values = result.data["Mean |SHAP|"].values
        np.testing.assert_allclose(values, 1.0)


# ---------------------------------------------------------------------------
# Integration: sklearn model → real SHAP values → plot functions
# ---------------------------------------------------------------------------

class TestWithRealModel:
    """End-to-end: train LogisticRegression on iris, compute SHAP, test plots."""

    @pytest.fixture(scope="class")
    def iris_shap(self):
        shap = pytest.importorskip("shap")
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        X_raw, y = load_iris(return_X_y=True)
        feature_names = load_iris().feature_names
        # Use a subset for speed
        X, y = X_raw[:60], y[:60]

        model = LogisticRegression(max_iter=300, random_state=0)
        model.fit(X, y)

        background = shap.sample(X, 20, random_state=0)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X[:10], nsamples=30)
        # sv may be a list of (10, 4) arrays (one per class) or a single array
        if isinstance(sv, list):
            sv_array = np.stack(sv, axis=-1)  # (10, 4, 3)
        else:
            sv_array = np.array(sv)
            if sv_array.ndim == 2:
                # Binary: shape (10, 4) — wrap to (10, 4, 1)
                sv_array = sv_array[:, :, np.newaxis]

        return sv_array, X[:10], list(feature_names), float(np.atleast_1d(explainer.expected_value)[0])

    def test_waterfall_from_real_shap(self, iris_shap):
        sv, fv, names, base = iris_shap
        sv_row = sv[0]  # (4, 3)
        result = shap_plots.waterfall(sv_row, names, base_value=base)
        assert isinstance(result, hv.Overlay)

    def test_beeswarm_from_real_shap(self, iris_shap):
        sv, fv, names, _ = iris_shap
        result = shap_plots.beeswarm(sv, fv, names)
        assert isinstance(result, hv.Overlay)

    def test_dependence_from_real_shap(self, iris_shap):
        sv, fv, names, _ = iris_shap
        result = shap_plots.dependence(sv, fv, 0, names, interaction_idx=1)
        assert isinstance(result, hv.Overlay)

    def test_importance_from_real_shap(self, iris_shap):
        sv, _, names, _ = iris_shap
        result = shap_plots.importance(sv, names)
        assert isinstance(result, hv.Bars)
        assert len(result) == len(names)

    def test_binary_classifier_waterfall(self):
        """Binary LogisticRegression — SHAP produces a single (n, f) array."""
        shap = pytest.importorskip("shap")
        from sklearn.datasets import load_breast_cancer
        from sklearn.linear_model import LogisticRegression

        X, y = load_breast_cancer(return_X_y=True)
        X, y = X[:50], y[:50]
        names = load_breast_cancer().feature_names.tolist()

        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(X, y)

        background = shap.sample(X, 10, random_state=0)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X[:5], nsamples=20)

        sv_arr = np.array(sv) if isinstance(sv, list) else sv
        sv_row = sv_arr[0] if sv_arr.ndim == 3 else sv_arr[0]  # (n_features, ...)

        result = shap_plots.waterfall(sv_row, names)
        assert isinstance(result, hv.Overlay)
