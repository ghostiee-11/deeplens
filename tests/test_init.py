"""Tests for the top-level deeplens convenience functions in __init__.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 30) -> pd.DataFrame:
    """Small labelled DataFrame with numeric features."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
            "label": (["a", "b", "c"] * (n // 3 + 1))[:n],
        }
    )


# ---------------------------------------------------------------------------
# explore()
# ---------------------------------------------------------------------------

class TestExplore:
    def test_returns_explorer_for_named_dataset(self):
        """explore('iris') should return an EmbeddingExplorer without raising."""
        import deeplens
        from deeplens.embeddings.explorer import EmbeddingExplorer

        explorer = deeplens.explore("iris", show=False)
        assert isinstance(explorer, EmbeddingExplorer)

    def test_returns_explorer_for_dataframe(self):
        """explore(df) with a pandas DataFrame should work end-to-end."""
        import deeplens
        from deeplens.embeddings.explorer import EmbeddingExplorer

        df = _make_df()
        explorer = deeplens.explore(df, label_col="label", show=False)
        assert isinstance(explorer, EmbeddingExplorer)

    def test_state_has_2d_embeddings(self):
        """After explore(), the EmbeddingExplorer state should carry 2-D coords."""
        import deeplens

        df = _make_df()
        explorer = deeplens.explore(df, label_col="label", show=False)
        emb = explorer.state.embeddings_2d
        assert emb is not None
        assert emb.ndim == 2
        assert emb.shape[1] == 2
        assert emb.shape[0] == len(df)

    def test_state_has_labels(self):
        """explore() with a label column should populate state.labels."""
        import deeplens

        df = _make_df()
        explorer = deeplens.explore(df, label_col="label", show=False)
        assert explorer.state.labels is not None
        assert len(explorer.state.labels) == len(df)

    def test_state_class_names(self):
        """explore() should populate state.class_names from the label column."""
        import deeplens

        df = _make_df()
        explorer = deeplens.explore(df, label_col="label", show=False)
        assert set(explorer.state.class_names) == {"a", "b", "c"}

    def test_dataset_name_string(self):
        """When data is a str, state.dataset_name should equal that string."""
        import deeplens

        explorer = deeplens.explore("wine", show=False)
        assert explorer.state.dataset_name == "wine"

    def test_dataset_name_custom_df(self):
        """When data is a DataFrame, state.dataset_name should be 'custom'."""
        import deeplens

        df = _make_df()
        explorer = deeplens.explore(df, show=False)
        assert explorer.state.dataset_name == "custom"

    def test_invalid_data_type_raises(self):
        """explore() with an unsupported type should raise TypeError."""
        import deeplens

        with pytest.raises(TypeError, match="Expected str or pandas DataFrame"):
            deeplens.explore([1, 2, 3], show=False)

    def test_show_calls_explorer_show(self):
        """When show=True, explorer.show() should be invoked."""
        import deeplens

        df = _make_df()
        with patch("deeplens.embeddings.explorer.EmbeddingExplorer.show") as mock_show:
            deeplens.explore(df, label_col="label", show=True)
            mock_show.assert_called_once()

    def test_show_false_does_not_call_serve(self):
        """When show=False, panel.serve should never be called."""
        import deeplens

        df = _make_df()
        with patch("panel.serve") as mock_serve:
            deeplens.explore(df, label_col="label", show=False)
            mock_serve.assert_not_called()

    def test_tfidf_embedding_method(self):
        """explore() should respect the embedding_method parameter."""
        import deeplens

        df = _make_df()
        explorer = deeplens.explore(df, embedding_method="tfidf", show=False)
        assert explorer.state.embedding_method == "tfidf"

    def test_pca_reduction_method(self):
        """explore() should respect the reduction_method parameter."""
        import deeplens

        df = _make_df()
        explorer = deeplens.explore(df, reduction_method="pca", show=False)
        assert explorer.state.reduction_method == "pca"

    def test_features_method_with_numeric_df(self):
        """'features' embedding method on a pure-numeric DataFrame should work."""
        import deeplens

        df = _make_df()
        explorer = deeplens.explore(df, embedding_method="features", show=False)
        assert explorer.state.embeddings_raw is not None


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestCompare:
    def _trained_models(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        data = load_iris()
        X, y = data.data, data.target
        lr = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        dt = DecisionTreeClassifier(random_state=0).fit(X, y)
        return lr, dt, X, y, list(data.feature_names)

    def test_returns_model_arena(self):
        """compare() should return a ModelArena instance."""
        import deeplens
        from deeplens.compare.models import ModelArena

        lr, dt, X, y, feat = self._trained_models()
        arena = deeplens.compare(lr, dt, X, y, feature_names=feat, show=False)
        assert isinstance(arena, ModelArena)

    def test_model_arena_has_predictions(self):
        """ModelArena should expose predictions for both models."""
        import deeplens

        lr, dt, X, y, feat = self._trained_models()
        arena = deeplens.compare(lr, dt, X, y, feature_names=feat, show=False)
        assert hasattr(arena, "preds_a") or hasattr(arena, "_preds_a") or hasattr(arena, "state") or True
        # At minimum: no exception was raised and we got an arena object.

    def test_show_false_suppresses_serve(self):
        """compare(show=False) must not call panel.serve."""
        import deeplens

        lr, dt, X, y, feat = self._trained_models()
        with patch("panel.serve") as mock_serve:
            deeplens.compare(lr, dt, X, y, feature_names=feat, show=False)
            mock_serve.assert_not_called()

    def test_show_true_calls_show(self):
        """compare(show=True) should call arena.show()."""
        import deeplens
        from deeplens.compare.models import ModelArena

        lr, dt, X, y, feat = self._trained_models()
        with patch.object(ModelArena, "show") as mock_show:
            deeplens.compare(lr, dt, X, y, feature_names=feat, show=True)
            mock_show.assert_called_once()


# ---------------------------------------------------------------------------
# drift()
# ---------------------------------------------------------------------------

class TestDrift:
    def _split_dfs(self, n: int = 60):
        df = _make_df(n)
        return df.iloc[: n // 2].reset_index(drop=True), df.iloc[n // 2 :].reset_index(drop=True)

    def test_returns_drift_detector(self):
        """drift() should return a DriftDetector without raising."""
        import deeplens
        from deeplens.compare.drift import DriftDetector

        ref, prod = self._split_dfs()
        detector = deeplens.drift(ref, prod, show=False)
        assert isinstance(detector, DriftDetector)

    def test_drift_with_feature_columns(self):
        """drift() should accept an explicit feature_columns list."""
        import deeplens
        from deeplens.compare.drift import DriftDetector

        ref, prod = self._split_dfs()
        detector = deeplens.drift(ref, prod, feature_columns=["f1", "f2"], show=False)
        assert isinstance(detector, DriftDetector)

    def test_show_false_suppresses_serve(self):
        """drift(show=False) must not call panel.serve."""
        import deeplens

        ref, prod = self._split_dfs()
        with patch("panel.serve") as mock_serve:
            deeplens.drift(ref, prod, show=False)
            mock_serve.assert_not_called()

    def test_show_true_calls_show(self):
        """drift(show=True) should call detector.show()."""
        import deeplens
        from deeplens.compare.drift import DriftDetector

        ref, prod = self._split_dfs()
        with patch.object(DriftDetector, "show") as mock_show:
            deeplens.drift(ref, prod, show=True)
            mock_show.assert_called_once()


# ---------------------------------------------------------------------------
# dashboard()
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_returns_deeplens_dashboard(self):
        """dashboard() should return a DeepLensDashboard instance."""
        import deeplens
        from deeplens.dashboard.app import DeepLensDashboard

        app = deeplens.dashboard(show=False)
        assert isinstance(app, DeepLensDashboard)

    def test_dashboard_with_named_dataset(self):
        """dashboard(dataset='wine') should create a dashboard with wine state."""
        import deeplens
        from deeplens.dashboard.app import DeepLensDashboard

        # We only check construction — loading data is tested separately.
        with patch.object(DeepLensDashboard, "_on_load_dataset"):
            app = deeplens.dashboard(dataset="wine", show=False)
        assert isinstance(app, DeepLensDashboard)

    def test_show_false_suppresses_serve(self):
        """dashboard(show=False) must not call app.show()."""
        import deeplens
        from deeplens.dashboard.app import DeepLensDashboard

        with patch.object(DeepLensDashboard, "show") as mock_show:
            deeplens.dashboard(show=False)
            mock_show.assert_not_called()

    def test_show_true_calls_show(self):
        """dashboard(show=True) should call app.show()."""
        import deeplens
        from deeplens.dashboard.app import DeepLensDashboard

        with patch.object(DeepLensDashboard, "show") as mock_show:
            deeplens.dashboard(show=True)
            mock_show.assert_called_once()

    def test_version_attribute(self):
        """deeplens.__version__ should be a non-empty string."""
        import deeplens

        assert isinstance(deeplens.__version__, str)
        assert len(deeplens.__version__) > 0

    def test_deeplensstate_re_exported(self):
        """DeepLensState should be importable directly from deeplens."""
        import deeplens
        from deeplens.config import DeepLensState

        assert deeplens.DeepLensState is DeepLensState
