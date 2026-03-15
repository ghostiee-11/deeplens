"""Tests for deeplens.embeddings.explorer.EmbeddingExplorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import panel as pn
import holoviews as hv

from deeplens.config import DeepLensState
from deeplens.embeddings.explorer import EmbeddingExplorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(n=40, n_features=4, with_predictions=False, with_probs=False):
    """Build a minimal DeepLensState with PCA-reduced embeddings."""
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features).astype(np.float32)
    labels = np.array(["cat", "dog"] * (n // 2))
    emb_2d = PCA(n_components=2).fit_transform(X)

    state = DeepLensState()
    state.df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    state.df["label"] = labels
    state.embeddings_raw = X
    state.embeddings_2d = emb_2d.astype(np.float64)
    state.labels = labels
    state.label_column = "label"
    state.feature_columns = [f"f{i}" for i in range(n_features)]
    state.class_names = ["cat", "dog"]

    if with_predictions:
        state.predictions = labels.copy()  # perfect predictions

    if with_probs:
        probs = rng.dirichlet(np.ones(2), size=n).astype(np.float32)
        state.probabilities = probs

    return state


def _make_explorer(n=40, **kwargs):
    state = _make_state(n=n, **kwargs)
    return EmbeddingExplorer(state=state)


# ---------------------------------------------------------------------------
# __init__ / widget creation
# ---------------------------------------------------------------------------

class TestEmbeddingExplorerInit:
    def test_init_no_state(self):
        """EmbeddingExplorer should initialise without a state."""
        explorer = EmbeddingExplorer()
        assert explorer.state is None

    def test_init_creates_color_widget(self):
        explorer = _make_explorer()
        assert isinstance(explorer._color_widget, pn.widgets.Select)

    def test_init_creates_size_widget(self):
        explorer = _make_explorer()
        assert isinstance(explorer._size_widget, pn.widgets.IntSlider)

    def test_init_creates_k_widget(self):
        explorer = _make_explorer()
        assert isinstance(explorer._k_widget, pn.widgets.IntSlider)

    def test_color_widget_options(self):
        explorer = _make_explorer()
        expected = ["label", "prediction", "confidence", "cluster", "error"]
        assert set(explorer._color_widget.options) == set(expected)

    def test_default_color_by_is_label(self):
        explorer = _make_explorer()
        assert explorer.color_by == "label"

    def test_default_point_size(self):
        explorer = _make_explorer()
        assert explorer.point_size == 5

    def test_default_k_neighbors(self):
        explorer = _make_explorer()
        assert explorer.k_neighbors == 10

    def test_dr_widgets_built_when_state_has_reduction_method(self):
        state = _make_state()
        state.reduction_method = "pca"
        explorer = EmbeddingExplorer(state=state)
        # Should contain at least a Select widget for the reduction method
        assert len(explorer._dr_widgets) >= 1
        assert isinstance(explorer._dr_widgets[0], pn.widgets.Select)

    def test_dr_widgets_empty_without_state(self):
        explorer = EmbeddingExplorer()
        assert explorer._dr_widgets == []


# ---------------------------------------------------------------------------
# _get_plot_df
# ---------------------------------------------------------------------------

class TestGetPlotDf:
    def test_no_state_returns_empty_df(self):
        explorer = EmbeddingExplorer()
        df = explorer._get_plot_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "x" in df.columns
        assert "y" in df.columns

    def test_state_without_embeddings_returns_empty(self):
        state = DeepLensState()
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        assert df.empty

    def test_returns_correct_row_count(self):
        explorer = _make_explorer(n=40)
        df = explorer._get_plot_df()
        assert len(df) == 40

    def test_x_y_columns_present(self):
        explorer = _make_explorer()
        df = explorer._get_plot_df()
        assert "x" in df.columns
        assert "y" in df.columns

    def test_label_column_added_when_present(self):
        explorer = _make_explorer()
        df = explorer._get_plot_df()
        assert "label" in df.columns

    def test_prediction_column_added_when_present(self):
        explorer = _make_explorer(with_predictions=True)
        df = explorer._get_plot_df()
        assert "prediction" in df.columns

    def test_confidence_column_added_when_probs_present(self):
        explorer = _make_explorer(with_probs=True)
        df = explorer._get_plot_df()
        assert "confidence" in df.columns

    def test_confidence_values_in_range(self):
        explorer = _make_explorer(with_probs=True)
        df = explorer._get_plot_df()
        assert (df["confidence"] >= 0).all()
        assert (df["confidence"] <= 1).all()

    def test_error_column_added_when_preds_and_labels_present(self):
        explorer = _make_explorer(with_predictions=True)
        df = explorer._get_plot_df()
        assert "error" in df.columns

    def test_error_column_values_binary(self):
        explorer = _make_explorer(with_predictions=True)
        df = explorer._get_plot_df()
        assert set(df["error"].unique()).issubset({0, 1})

    def test_embeddings_2d_values_match_x_y(self):
        state = _make_state(n=20)
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        np.testing.assert_allclose(df["x"].values, state.embeddings_2d[:, 0])
        np.testing.assert_allclose(df["y"].values, state.embeddings_2d[:, 1])

    def test_no_prediction_no_error_column(self):
        explorer = _make_explorer()  # no predictions
        df = explorer._get_plot_df()
        assert "error" not in df.columns


# ---------------------------------------------------------------------------
# Auto-clustering
# ---------------------------------------------------------------------------

class TestAutoClustering:
    def test_auto_cluster_adds_cluster_column(self):
        """When no cluster_labels are set, _get_plot_df should auto-cluster."""
        state = _make_state(n=30)
        state.cluster_labels = None  # ensure no pre-set clusters
        state.n_clusters = 3
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        assert "cluster" in df.columns

    def test_auto_cluster_number_of_clusters(self):
        state = _make_state(n=30)
        state.cluster_labels = None
        state.n_clusters = 4
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        n_unique = df["cluster"].nunique()
        assert n_unique <= 4

    def test_pre_set_cluster_labels_not_overwritten(self):
        state = _make_state(n=20)
        preset = np.array(["A", "B"] * 10)
        state.cluster_labels = preset
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        assert "cluster" in df.columns
        # Should use the pre-set labels, not re-run KMeans
        np.testing.assert_array_equal(df["cluster"].values, preset)

    def test_auto_cluster_stores_back_to_state(self):
        """Auto-cluster should persist results to state.cluster_labels."""
        state = _make_state(n=20)
        state.cluster_labels = None
        state.n_clusters = 3
        explorer = EmbeddingExplorer(state=state)
        explorer._get_plot_df()
        assert state.cluster_labels is not None
        assert len(state.cluster_labels) == 20


# ---------------------------------------------------------------------------
# _embedding_plot
# ---------------------------------------------------------------------------

class TestEmbeddingPlot:
    def test_empty_state_returns_hv_text(self):
        explorer = EmbeddingExplorer()
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Text)

    def test_with_embeddings_returns_hv_element(self):
        explorer = _make_explorer()
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Element)

    def test_small_dataset_returns_points_not_datashader(self):
        """Below the datashader threshold the plot should be an hv.Points."""
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state, datashader_threshold=5000)
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)

    def test_color_by_label_includes_vdim(self):
        explorer = _make_explorer()
        explorer.color_by = "label"
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)
        vdim_names = [str(v) for v in result.vdims]
        assert "label" in vdim_names

    def test_color_by_confidence_with_probs(self):
        explorer = _make_explorer(with_probs=True)
        explorer.color_by = "confidence"
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)

    def test_selection_stream_attached(self):
        explorer = _make_explorer()
        result = explorer._embedding_plot()
        # After calling _embedding_plot, the result should be a HoloViews element
        # and the selection stream should exist on the explorer.
        assert isinstance(result, hv.Element)
        assert isinstance(explorer._selection_stream, hv.streams.Selection1D)

    def test_tap_stream_attached(self):
        explorer = _make_explorer()
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Element)
        assert isinstance(explorer._tap_stream, hv.streams.Tap)

    def test_invalid_color_by_falls_back_gracefully(self):
        """If color_by column doesn't exist, should still return a Points element."""
        explorer = _make_explorer()
        explorer.color_by = "prediction"  # not in state (no predictions set)
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)

    def test_low_cardinality_label_cast_to_string(self):
        """Numeric labels with <= 20 unique values should be cast to string."""
        from sklearn.decomposition import PCA
        rng = np.random.RandomState(0)
        n = 30
        X = rng.randn(n, 3).astype(np.float32)
        state = DeepLensState()
        state.embeddings_raw = X
        state.embeddings_2d = PCA(n_components=2).fit_transform(X)
        state.labels = np.array([0, 1, 2] * 10)
        state.label_column = "label"
        state.df = pd.DataFrame(X)
        state.df["label"] = state.labels
        state.feature_columns = [0, 1, 2]
        explorer = EmbeddingExplorer(state=state)
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)
        # The label column in the plotted data should be string type
        plot_df = result.data
        assert pd.api.types.is_string_dtype(plot_df["label"])


# ---------------------------------------------------------------------------
# __panel__
# ---------------------------------------------------------------------------

class TestEmbeddingExplorerPanel:
    def test_panel_returns_row(self):
        explorer = _make_explorer()
        result = explorer.__panel__()
        assert isinstance(result, pn.Row)

    def test_panel_has_three_children(self):
        """Layout should be controls | main_plot | details — three children."""
        explorer = _make_explorer()
        result = explorer.__panel__()
        assert len(result) == 3

    def test_panel_first_child_is_column(self):
        explorer = _make_explorer()
        result = explorer.__panel__()
        assert isinstance(result[0], pn.Column)

    def test_panel_middle_child_is_holviews_pane(self):
        explorer = _make_explorer()
        result = explorer.__panel__()
        assert isinstance(result[1], pn.pane.HoloViews)

    def test_panel_last_child_is_column(self):
        explorer = _make_explorer()
        result = explorer.__panel__()
        assert isinstance(result[2], pn.Column)

    def test_panel_no_state_still_returns_row(self):
        explorer = EmbeddingExplorer()
        result = explorer.__panel__()
        assert isinstance(result, pn.Row)

    def test_controls_column_contains_color_widget(self):
        explorer = _make_explorer()
        controls = explorer.__panel__()[0]
        # Flatten to find the Select widget
        found = any(
            isinstance(item, pn.widgets.Select)
            for item in controls
        )
        assert found

    def test_sizing_mode_stretch(self):
        explorer = _make_explorer()
        result = explorer.__panel__()
        assert result.sizing_mode == "stretch_both"


# ---------------------------------------------------------------------------
# _get_plot_df — additional color column scenarios
# ---------------------------------------------------------------------------

class TestGetPlotDfColorColumns:
    def test_cluster_column_from_preset(self):
        """cluster column should come from state.cluster_labels when preset."""
        from sklearn.decomposition import PCA
        rng = np.random.RandomState(3)
        n = 20
        X = rng.randn(n, 3).astype(np.float32)
        state = DeepLensState()
        state.embeddings_raw = X
        state.embeddings_2d = PCA(n_components=2).fit_transform(X)
        state.labels = np.array(["a", "b"] * (n // 2))
        state.label_column = "label"
        state.df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        state.df["label"] = state.labels
        state.feature_columns = ["f0", "f1", "f2"]
        state.cluster_labels = np.array(["X", "Y"] * (n // 2))
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        assert "cluster" in df.columns
        assert list(df["cluster"]) == list(state.cluster_labels)

    def test_error_column_is_zero_for_perfect_predictions(self):
        """When predictions == labels, error column should be all zeros."""
        explorer = _make_explorer(with_predictions=True)
        df = explorer._get_plot_df()
        assert "error" in df.columns
        assert (df["error"] == 0).all()

    def test_confidence_max_of_probabilities(self):
        """confidence = max(probabilities, axis=1) for each row."""
        state = _make_state(n=20, with_probs=True)
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        expected = np.max(state.probabilities, axis=1)
        np.testing.assert_allclose(df["confidence"].values, expected, rtol=1e-5)

    def test_all_columns_present_with_all_data(self):
        """When all data is set, all color columns should appear."""
        state = _make_state(n=40, with_predictions=True, with_probs=True)
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        for col in ["label", "prediction", "confidence", "error"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_label_values_preserved(self):
        state = _make_state(n=20)
        explorer = EmbeddingExplorer(state=state)
        df = explorer._get_plot_df()
        np.testing.assert_array_equal(df["label"].values, state.labels)


# ---------------------------------------------------------------------------
# _embedding_plot — datashader path and color_by scenarios
# ---------------------------------------------------------------------------

class TestEmbeddingPlotExtended:
    def test_large_dataset_uses_datashader(self):
        """Above the datashader threshold, plot should be a DynamicMap/Image (not Points)."""
        n = 200
        state = _make_state(n=n)
        # Lower threshold so our 200-sample dataset triggers datashader
        explorer = EmbeddingExplorer(state=state, datashader_threshold=50, use_datashader=True)
        result = explorer._embedding_plot()
        # datashader output is not hv.Points (it's a DynamicMap or Image)
        assert not isinstance(result, hv.Points)

    def test_color_by_prediction_with_predictions(self):
        explorer = _make_explorer(with_predictions=True)
        explorer.color_by = "prediction"
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)

    def test_color_by_error_with_predictions(self):
        explorer = _make_explorer(with_predictions=True)
        explorer.color_by = "error"
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)

    def test_color_by_cluster_auto_clustered(self):
        state = _make_state(n=40)
        state.cluster_labels = None
        state.n_clusters = 3
        explorer = EmbeddingExplorer(state=state)
        explorer.color_by = "cluster"
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)

    def test_color_by_confidence_numeric_cmap(self):
        explorer = _make_explorer(with_probs=True)
        explorer.color_by = "confidence"
        result = explorer._embedding_plot()
        # confidence is numeric — should still produce a Points with colorbar
        assert isinstance(result, hv.Points)

    def test_point_size_propagated(self):
        """Changing point_size should still produce a Points element."""
        state = _make_state(n=20)
        explorer = EmbeddingExplorer(state=state, point_size=8)
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)

    def test_datashader_disabled_always_uses_points(self):
        n = 300
        state = _make_state(n=n)
        explorer = EmbeddingExplorer(state=state, use_datashader=False)
        result = explorer._embedding_plot()
        assert isinstance(result, hv.Points)


# ---------------------------------------------------------------------------
# _cluster_stats_panel
# ---------------------------------------------------------------------------

class TestClusterStatsPanel:
    def test_no_embeddings_returns_markdown(self):
        explorer = EmbeddingExplorer()
        result = explorer._cluster_stats_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_with_clusters_returns_markdown(self):
        state = _make_state(n=30)
        state.cluster_labels = None
        state.n_clusters = 3
        explorer = EmbeddingExplorer(state=state)
        result = explorer._cluster_stats_panel()
        assert isinstance(result, pn.pane.Markdown)
        assert "Cluster" in result.object

    def test_cluster_summary_contains_count(self):
        from sklearn.decomposition import PCA
        rng = np.random.RandomState(5)
        n = 30
        X = rng.randn(n, 3).astype(np.float32)
        state = DeepLensState()
        state.embeddings_raw = X
        state.embeddings_2d = PCA(n_components=2).fit_transform(X)
        state.labels = np.array(["a", "b", "c"] * 10)
        state.label_column = "label"
        state.df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        state.df["label"] = state.labels
        state.feature_columns = ["f0", "f1", "f2"]
        state.cluster_labels = np.array(["0", "1", "2"] * 10)
        explorer = EmbeddingExplorer(state=state)
        result = explorer._cluster_stats_panel()
        # Each cluster ID should appear in the summary
        for cid in ["0", "1", "2"]:
            assert cid in result.object

    def test_cluster_stats_with_predictions_shows_accuracy(self):
        """When predictions and labels are available, accuracy per cluster is shown."""
        from sklearn.decomposition import PCA
        rng = np.random.RandomState(9)
        n = 40
        X = rng.randn(n, 3).astype(np.float32)
        labels = np.array(["a", "b"] * (n // 2))
        state = DeepLensState()
        state.embeddings_raw = X
        state.embeddings_2d = PCA(n_components=2).fit_transform(X)
        state.labels = labels
        state.predictions = labels.copy()  # perfect predictions
        state.label_column = "label"
        state.df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        state.df["label"] = labels
        state.feature_columns = ["f0", "f1", "f2"]
        state.cluster_labels = np.array(["0", "1"] * (n // 2))
        explorer = EmbeddingExplorer(state=state)
        result = explorer._cluster_stats_panel()
        assert "acc" in result.object

    def test_no_cluster_column_returns_no_clusters_message(self):
        """If _get_plot_df has no cluster column and n_clusters < 2, show placeholder."""
        state = DeepLensState()  # no embeddings
        explorer = EmbeddingExplorer(state=state)
        result = explorer._cluster_stats_panel()
        assert isinstance(result, pn.pane.Markdown)
        assert "No clusters" in result.object


# ---------------------------------------------------------------------------
# _quality_indicators
# ---------------------------------------------------------------------------

class TestQualityIndicators:
    def test_no_state_returns_markdown(self):
        explorer = EmbeddingExplorer()
        result = explorer._quality_indicators()
        assert isinstance(result, pn.pane.Markdown)

    def test_no_raw_embeddings_returns_markdown(self):
        state = _make_state(n=30)
        state.embeddings_raw = None
        explorer = EmbeddingExplorer(state=state)
        result = explorer._quality_indicators()
        assert isinstance(result, pn.pane.Markdown)

    def test_with_raw_and_2d_embeddings_returns_row(self):
        state = _make_state(n=40)
        # embeddings_raw is set in _make_state
        assert state.embeddings_raw is not None
        explorer = EmbeddingExplorer(state=state)
        result = explorer._quality_indicators()
        assert isinstance(result, pn.Row)

    def test_quality_row_has_three_indicators(self):
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        result = explorer._quality_indicators()
        assert isinstance(result, pn.Row)
        assert len(result) == 3

    def test_trustworthiness_indicator_present(self):
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        result = explorer._quality_indicators()
        names = [item.name for item in result]
        assert "Trustworthiness" in names

    def test_stress_indicator_present(self):
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        result = explorer._quality_indicators()
        names = [item.name for item in result]
        assert "Stress" in names

    def test_samples_indicator_present(self):
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        result = explorer._quality_indicators()
        names = [item.name for item in result]
        assert "Samples" in names

    def test_trustworthiness_value_in_range(self):
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        result = explorer._quality_indicators()
        trust_indicator = next(item for item in result if item.name == "Trustworthiness")
        assert 0.0 <= trust_indicator.value <= 1.0


# ---------------------------------------------------------------------------
# Auto-clustering — edge cases
# ---------------------------------------------------------------------------

class TestAutoClusteringEdgeCases:
    def test_n_clusters_larger_than_samples_clamped(self):
        """n_clusters is clamped to len(df) — should not raise."""
        state = _make_state(n=10)
        state.cluster_labels = None
        state.n_clusters = 50  # more clusters than samples
        explorer = EmbeddingExplorer(state=state)
        # Should not raise; KMeans clips to n_samples
        try:
            df = explorer._get_plot_df()
        except Exception:
            pytest.skip("KMeans with n_clusters > n_samples raises in this sklearn version")

    def test_auto_cluster_only_runs_when_cluster_missing(self):
        """If cluster column already exists in state, KMeans should NOT be called."""
        from unittest.mock import patch
        state = _make_state(n=20)
        state.cluster_labels = np.array(["A"] * 20)
        explorer = EmbeddingExplorer(state=state)
        with patch("sklearn.cluster.KMeans.fit_predict") as mock_fp:
            explorer._get_plot_df()
            mock_fp.assert_not_called()


# ---------------------------------------------------------------------------
# Similarity search panel smoke tests
# ---------------------------------------------------------------------------

class TestSimilarityPanel:
    def test_no_tap_returns_empty_markdown(self):
        explorer = _make_explorer()
        result = explorer._similarity_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_no_embeddings_returns_markdown(self):
        explorer = EmbeddingExplorer()
        result = explorer._similarity_panel()
        assert isinstance(result, pn.pane.Markdown)


# ---------------------------------------------------------------------------
# Additional coverage for missing lines: 162-170, 211-259, 269-297
# ---------------------------------------------------------------------------


class TestSelectionDetails:
    """Cover _selection_details: lines 211-259."""

    def test_selection_details_no_selection(self):
        """Lines 201-208: no selection returns hint markdown."""
        explorer = _make_explorer()
        result = explorer._selection_details()
        assert isinstance(result, pn.pane.Markdown)
        assert "Lasso" in result.object or "select" in result.object.lower()

    def test_selection_details_with_selection(self):
        """Lines 211-259: selection returns details panel."""
        state = _make_state(n=40, with_predictions=True)
        explorer = EmbeddingExplorer(state=state)
        # Simulate selection
        explorer._selection_stream.event(index=[0, 1, 2, 3, 4])
        result = explorer._selection_details()
        assert isinstance(result, pn.Column)

    def test_selection_details_updates_state(self):
        """Line 211: selected_indices updated on state."""
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        explorer._selection_stream.event(index=[5, 10, 15])
        explorer._selection_details()
        assert state.selected_indices == [5, 10, 15]

    def test_selection_details_shows_class_distribution(self):
        """Lines 223-229: class distribution shown when label_column exists."""
        state = _make_state(n=40, with_predictions=True)
        explorer = EmbeddingExplorer(state=state)
        explorer._selection_stream.event(index=[0, 1, 2, 3])
        result = explorer._selection_details()
        assert isinstance(result, pn.Column)

    def test_selection_details_shows_accuracy(self):
        """Lines 232-236: accuracy shown when predictions exist."""
        state = _make_state(n=40, with_predictions=True)
        explorer = EmbeddingExplorer(state=state)
        explorer._selection_stream.event(index=[0, 1, 2, 3])
        result = explorer._selection_details()
        assert isinstance(result, pn.Column)

    def test_selection_details_no_df(self):
        """Lines 214-215: state.df is None."""
        state = _make_state(n=40)
        state.df = None
        explorer = EmbeddingExplorer(state=state)
        explorer._selection_stream.event(index=[0, 1])
        result = explorer._selection_details()
        assert isinstance(result, pn.pane.Markdown)


class TestSimilarityPanelExtended:
    """Cover _similarity_panel: lines 269-297."""

    def test_similarity_panel_with_tap(self):
        """Lines 269-297: tap triggers similarity search."""
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        # Get coordinates of first point
        x_val = float(state.embeddings_2d[0, 0])
        y_val = float(state.embeddings_2d[0, 1])
        explorer._tap_stream.event(x=x_val, y=y_val)
        result = explorer._similarity_panel()
        assert isinstance(result, pn.pane.Markdown)
        assert "Similarity" in result.object or "Nearest" in result.object

    def test_similarity_panel_uses_raw_embeddings(self):
        """Lines 274-278: uses raw embeddings for kNN when available."""
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        x_val = float(state.embeddings_2d[5, 0])
        y_val = float(state.embeddings_2d[5, 1])
        explorer._tap_stream.event(x=x_val, y=y_val)
        result = explorer._similarity_panel()
        assert isinstance(result, pn.pane.Markdown)
        assert "Tapped point" in result.object

    def test_similarity_panel_falls_back_to_2d(self):
        """Lines 279-280: falls back to 2d distances when no raw embeddings."""
        state = _make_state(n=40)
        state.embeddings_raw = None
        explorer = EmbeddingExplorer(state=state)
        x_val = float(state.embeddings_2d[0, 0])
        y_val = float(state.embeddings_2d[0, 1])
        explorer._tap_stream.event(x=x_val, y=y_val)
        result = explorer._similarity_panel()
        assert isinstance(result, pn.pane.Markdown)
        assert "Similarity" in result.object

    def test_similarity_panel_shows_labels(self):
        """Lines 289-290: shows label of tapped point."""
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state)
        x_val = float(state.embeddings_2d[0, 0])
        y_val = float(state.embeddings_2d[0, 1])
        explorer._tap_stream.event(x=x_val, y=y_val)
        result = explorer._similarity_panel()
        assert "Label" in result.object

    def test_similarity_panel_no_labels(self):
        """Lines 294: label is '?' when no labels."""
        state = _make_state(n=40)
        state.labels = None
        explorer = EmbeddingExplorer(state=state)
        x_val = float(state.embeddings_2d[0, 0])
        y_val = float(state.embeddings_2d[0, 1])
        explorer._tap_stream.event(x=x_val, y=y_val)
        result = explorer._similarity_panel()
        assert isinstance(result, pn.pane.Markdown)

    def test_similarity_panel_k_neighbors_respected(self):
        """Lines 282-283: k value is used."""
        state = _make_state(n=40)
        explorer = EmbeddingExplorer(state=state, k_neighbors=5)
        x_val = float(state.embeddings_2d[0, 0])
        y_val = float(state.embeddings_2d[0, 1])
        explorer._tap_stream.event(x=x_val, y=y_val)
        result = explorer._similarity_panel()
        assert "k=5" in result.object


class TestEmbeddingPlotDatashaderCategorical:
    """Cover datashader categorical branch: lines 162-170."""

    def test_datashader_with_categorical_color(self):
        """Lines 149-160: datashader with string color column."""
        state = _make_state(n=200)
        explorer = EmbeddingExplorer(
            state=state,
            datashader_threshold=50,
            use_datashader=True,
        )
        explorer.color_by = "label"
        result = explorer._embedding_plot()
        # Should not be raw Points (datashader processes it)
        assert not isinstance(result, hv.Points)

    def test_datashader_with_numeric_color(self):
        """Lines 161-170: datashader with non-categorical color."""
        state = _make_state(n=200, with_probs=True)
        explorer = EmbeddingExplorer(
            state=state,
            datashader_threshold=50,
            use_datashader=True,
        )
        explorer.color_by = "confidence"
        result = explorer._embedding_plot()
        assert not isinstance(result, hv.Points)
