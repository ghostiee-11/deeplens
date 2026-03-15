"""Tests for deeplens.models.error_analysis.ErrorAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv
import panel as pn

from deeplens.config import DeepLensState
from deeplens.models.error_analysis import ErrorAnalyzer


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def binary_state():
    """A fully-populated DeepLensState with binary labels and a trained LR model."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(42)
    n = 80
    X = rng.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)

    # Simple 2-D PCA-like embeddings (just use first 2 features)
    emb_2d = X[:, :2]

    state = DeepLensState()
    state.dataset_name = "binary_test"
    state.df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    state.feature_columns = ["f0", "f1", "f2", "f3"]
    state.label_column = "label"
    state.labels = y
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    state.trained_model = model
    state.model_name = "LogisticRegression"
    state.embeddings_2d = emb_2d
    return state


@pytest.fixture
def multiclass_state():
    """Three-class state for multiclass tests."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(7)
    n = 90
    X = rng.randn(n, 5)
    y = np.array(["cat", "dog", "fish"] * 30)

    model = LogisticRegression(max_iter=300, random_state=7)
    model.fit(X, y)

    state = DeepLensState()
    state.dataset_name = "multiclass_test"
    state.df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    state.feature_columns = [f"f{i}" for i in range(5)]
    state.label_column = "label"
    state.labels = y
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    state.trained_model = model
    state.model_name = "LogisticRegression"
    state.embeddings_2d = X[:, :2]
    state.class_names = ["cat", "dog", "fish"]
    return state


@pytest.fixture
def state_with_clusters(binary_state):
    """Binary state augmented with cluster labels."""
    binary_state.cluster_labels = np.array([0, 1] * 40)
    return binary_state


@pytest.fixture
def standalone_analyzer():
    """ErrorAnalyzer initialised without state (standalone mode)."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(99)
    n = 60
    X = rng.randn(n, 3)
    y = (X[:, 0] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=99)
    model.fit(X, y)

    emb_2d = X[:, :2]

    return ErrorAnalyzer(
        model=model,
        X=X,
        y=y,
        embeddings_2d=emb_2d,
        feature_names=["alpha", "beta", "gamma"],
    )


# ── Test 1: Initialisation ───────────────────────────────────────────────────

class TestInit:
    def test_init_with_state(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        assert ea.state is binary_state

    def test_init_standalone(self, standalone_analyzer):
        assert standalone_analyzer.state is None
        assert standalone_analyzer.model is not None
        assert standalone_analyzer.X is not None

    def test_init_empty_returns_viewer(self):
        ea = ErrorAnalyzer()
        assert isinstance(ea, pn.viewable.Viewer)

    def test_has_data_with_state(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        assert ea._has_data() is True

    def test_has_data_empty(self):
        ea = ErrorAnalyzer()
        assert ea._has_data() is False


# ── Test 2: Misclassification scatter ────────────────────────────────────────

class TestMisclassificationScatter:
    def test_returns_hv_points_with_state(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.misclassification_scatter()
        assert isinstance(result, hv.Points)

    def test_returns_hv_points_standalone(self, standalone_analyzer):
        result = standalone_analyzer.misclassification_scatter()
        assert isinstance(result, hv.Points)

    def test_scatter_no_data_returns_text(self):
        ea = ErrorAnalyzer()
        result = ea.misclassification_scatter()
        assert isinstance(result, hv.Text)

    def test_scatter_missing_embeddings_returns_text(self, binary_state):
        binary_state.embeddings_2d = None
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.misclassification_scatter()
        assert isinstance(result, hv.Text)

    def test_scatter_contains_status_column(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.misclassification_scatter()
        assert "status" in result.vdims or any(
            d.name == "status" for d in result.vdims
        )

    def test_scatter_correct_point_count(self, binary_state):
        """Number of points equals number of samples."""
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.misclassification_scatter()
        # HoloViews Points data should have same length as input
        data = result.data
        if isinstance(data, pd.DataFrame):
            assert len(data) == len(binary_state.labels)


# ── Test 3: Confusion pairs ──────────────────────────────────────────────────

class TestConfusionPairs:
    def test_returns_bars_with_errors(self, multiclass_state):
        ea = ErrorAnalyzer(state=multiclass_state)
        result = ea.confusion_pairs()
        # Should render a Bars element (there will almost certainly be errors)
        assert isinstance(result, (hv.Bars, hv.Text))

    def test_perfect_model_returns_text(self):
        state = DeepLensState()
        y = np.array([0, 1, 0, 1])
        state.labels = y
        state.predictions = y.copy()   # perfect predictions
        ea = ErrorAnalyzer(state=state)
        result = ea.confusion_pairs()
        assert isinstance(result, hv.Text)

    def test_no_data_returns_text(self):
        ea = ErrorAnalyzer()
        result = ea.confusion_pairs()
        assert isinstance(result, hv.Text)

    def test_top_n_respected(self, multiclass_state):
        ea = ErrorAnalyzer(state=multiclass_state, top_n_pairs=2)
        result = ea.confusion_pairs()
        if isinstance(result, hv.Bars):
            data = result.data
            if isinstance(data, pd.DataFrame):
                assert len(data) <= 2
            elif isinstance(data, list):
                assert len(data) <= 2

    def test_pair_format_arrow(self, multiclass_state):
        """Pair labels should use the '↔' separator."""
        ea = ErrorAnalyzer(state=multiclass_state)
        result = ea.confusion_pairs()
        if isinstance(result, hv.Bars):
            data = result.data
            if isinstance(data, pd.DataFrame):
                for val in data.iloc[:, 0]:
                    assert "↔" in str(val)


# ── Test 4: Feature distributions ───────────────────────────────────────────

class TestFeatureDistributions:
    def test_returns_layout_or_element(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.feature_distributions()
        assert isinstance(result, (hv.Layout, hv.Overlay, hv.Element))

    def test_no_data_returns_text(self):
        ea = ErrorAnalyzer()
        result = ea.feature_distributions()
        assert isinstance(result, hv.Text)

    def test_no_X_returns_text(self):
        state = DeepLensState()
        state.labels = np.array([0, 1, 0, 1])
        state.predictions = np.array([0, 0, 0, 1])
        # No df / feature_columns → _X() returns None
        ea = ErrorAnalyzer(state=state)
        result = ea.feature_distributions()
        assert isinstance(result, hv.Text)

    def test_standalone_kde_uses_feature_names(self, standalone_analyzer):
        result = standalone_analyzer.feature_distributions()
        # Should produce a Layout or Overlay without raising
        assert isinstance(result, (hv.Layout, hv.Overlay, hv.Element))

    def test_top_n_features_limits_panels(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state, top_n_features=2)
        result = ea.feature_distributions()
        if isinstance(result, hv.Layout):
            assert len(result) <= 2


# ── Test 5: Hardest samples ──────────────────────────────────────────────────

class TestHardestSamples:
    def test_returns_tabulator_with_probs(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.hardest_samples()
        assert isinstance(result, pn.widgets.Tabulator)

    def test_tabulator_has_expected_columns(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.hardest_samples()
        df = result.value
        required = {"index", "true_label", "predicted_label", "confidence", "margin"}
        assert required.issubset(set(df.columns))

    def test_sorted_ascending_margin(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.hardest_samples()
        df = result.value
        margins = df["margin"].values
        assert np.all(margins[:-1] <= margins[1:]), "Table not sorted by ascending margin"

    def test_returns_tabulator_without_probs(self):
        state = DeepLensState()
        state.labels = np.array([0, 1, 0, 1, 0])
        state.predictions = np.array([0, 1, 1, 1, 0])
        ea = ErrorAnalyzer(state=state)
        result = ea.hardest_samples()
        assert isinstance(result, pn.widgets.Tabulator)

    def test_no_data_returns_markdown(self):
        ea = ErrorAnalyzer()
        result = ea.hardest_samples()
        assert isinstance(result, pn.pane.Markdown)

    def test_row_count_matches_samples(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.hardest_samples()
        assert len(result.value) == len(binary_state.labels)


# ── Test 6: Error rate by cluster ────────────────────────────────────────────

class TestErrorRateByCluster:
    def test_returns_bars_with_clusters(self, state_with_clusters):
        ea = ErrorAnalyzer(state=state_with_clusters)
        result = ea.error_rate_by_cluster()
        assert isinstance(result, hv.Bars)

    def test_no_clusters_returns_text(self, binary_state):
        # binary_state has no cluster_labels
        ea = ErrorAnalyzer(state=binary_state)
        result = ea.error_rate_by_cluster()
        assert isinstance(result, hv.Text)

    def test_no_data_returns_text(self):
        ea = ErrorAnalyzer()
        result = ea.error_rate_by_cluster()
        assert isinstance(result, hv.Text)

    def test_cluster_bar_count_matches_n_clusters(self, state_with_clusters):
        ea = ErrorAnalyzer(state=state_with_clusters)
        result = ea.error_rate_by_cluster()
        data = result.data
        n_unique = len(np.unique(state_with_clusters.cluster_labels))
        if isinstance(data, pd.DataFrame):
            assert len(data) == n_unique

    def test_error_rates_between_0_and_1(self, state_with_clusters):
        ea = ErrorAnalyzer(state=state_with_clusters)
        result = ea.error_rate_by_cluster()
        data = result.data
        if isinstance(data, pd.DataFrame) and "error_rate" in data.columns:
            assert data["error_rate"].between(0, 1).all()


# ── Test 7: __panel__ layout ─────────────────────────────────────────────────

class TestPanelLayout:
    def test_panel_returns_column_with_data(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        layout = ea.__panel__()
        assert isinstance(layout, pn.Column)

    def test_panel_returns_column_without_data(self):
        ea = ErrorAnalyzer()
        layout = ea.__panel__()
        assert isinstance(layout, pn.Column)

    def test_panel_contains_dividers(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        layout = ea.__panel__()
        # Check at least one Divider is present anywhere in the tree
        has_divider = any(
            isinstance(obj, pn.layout.Divider)
            for obj in layout.objects
        )
        assert has_divider


# ── Test 8: Data accessor methods ────────────────────────────────────────────

class TestDataAccessors:
    def test_labels_from_state(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        labels = ea._labels()
        np.testing.assert_array_equal(labels, binary_state.labels)

    def test_labels_from_standalone(self, standalone_analyzer):
        labels = standalone_analyzer._labels()
        assert labels is not None
        assert len(labels) == 60

    def test_predictions_from_state(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        preds = ea._predictions()
        np.testing.assert_array_equal(preds, binary_state.predictions)

    def test_feature_names_from_state(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        names = ea._feature_names()
        assert names == ["f0", "f1", "f2", "f3"]

    def test_feature_names_from_standalone(self, standalone_analyzer):
        names = standalone_analyzer._feature_names()
        assert names == ["alpha", "beta", "gamma"]

    def test_feature_names_fallback(self):
        """When no names are provided, fall back to feature_N strings."""
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.RandomState(1)
        X = rng.randn(30, 2)
        y = (X[:, 0] > 0).astype(int)
        model = DecisionTreeClassifier(random_state=1)
        model.fit(X, y)

        ea = ErrorAnalyzer(model=model, X=X, y=y)
        names = ea._feature_names()
        assert names == ["feature_0", "feature_1"]

    def test_cluster_labels_from_state(self, state_with_clusters):
        ea = ErrorAnalyzer(state=state_with_clusters)
        cl = ea._cluster_labels()
        assert cl is not None
        assert len(cl) == len(state_with_clusters.labels)

    def test_cluster_labels_none_when_absent(self, binary_state):
        ea = ErrorAnalyzer(state=binary_state)
        assert ea._cluster_labels() is None
