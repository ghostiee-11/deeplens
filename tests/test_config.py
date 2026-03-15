"""Tests for deeplens.config.DeepLensState."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeplens.config import DeepLensState


class TestStateDefaults:
    def test_state_defaults(self):
        state = DeepLensState()
        assert state.dataset_name == ""
        assert state.df is None
        assert state.feature_columns == []
        assert state.label_column == ""
        assert state.class_names == []
        assert state.trained_model is None
        assert state.selected_indices == []
        assert state.theme == "dark"
        assert state.llm_provider == "none"

    def test_state_default_reduction_method(self):
        state = DeepLensState()
        assert state.reduction_method == "pca"
        assert state.embedding_method == "tfidf"


class TestStateNSamples:
    def test_n_samples_with_data(self, iris_state):
        assert iris_state.n_samples == 150

    def test_n_samples_empty(self):
        state = DeepLensState()
        assert state.n_samples == 0

    def test_n_features(self, iris_state):
        assert iris_state.n_features == len(iris_state.feature_columns)
        assert iris_state.n_features > 0


class TestStateHasModel:
    def test_has_model_false(self, iris_state):
        assert iris_state.has_model is False

    def test_has_model_true(self, iris_with_model):
        assert iris_with_model.has_model is True


class TestStateHasEmbeddings:
    def test_has_embeddings_false(self, iris_state):
        assert iris_state.has_embeddings is False

    def test_has_embeddings_true(self, iris_with_embeddings):
        assert iris_with_embeddings.has_embeddings is True


class TestStateSummary:
    def test_summary_basic(self, iris_state):
        s = iris_state.summary()
        assert "iris" in s
        assert "150" in s

    def test_summary_with_model(self, iris_with_model):
        s = iris_with_model.summary()
        assert "LogisticRegression" in s
        assert "Accuracy" in s

    def test_summary_truncation(self, iris_state):
        s = iris_state.summary(max_tokens=5)
        # With max_tokens=5 (20 chars), the summary should be truncated
        assert "truncated" in s


class TestSelectedDF:
    def test_selected_df_none_when_empty(self, iris_state):
        assert iris_state.selected_df is None

    def test_selected_df_with_indices(self, iris_state):
        iris_state.selected_indices = [0, 1, 2]
        sel = iris_state.selected_df
        assert sel is not None
        assert len(sel) == 3

    def test_selected_df_none_when_no_df(self):
        state = DeepLensState()
        state.selected_indices = [0, 1]
        assert state.selected_df is None


class TestSelectedDFExtended:
    """Additional selected_df coverage."""

    def test_selected_df_returns_correct_rows(self, iris_state):
        iris_state.selected_indices = [10, 20, 30]
        sel = iris_state.selected_df
        assert sel is not None
        assert len(sel) == 3
        # Verify the actual rows match iloc positions
        expected = iris_state.df.iloc[[10, 20, 30]]
        pd.testing.assert_frame_equal(sel, expected)

    def test_selected_df_single_index(self, iris_state):
        iris_state.selected_indices = [5]
        sel = iris_state.selected_df
        assert sel is not None
        assert len(sel) == 1

    def test_selected_df_all_indices(self, iris_state):
        iris_state.selected_indices = list(range(150))
        sel = iris_state.selected_df
        assert sel is not None
        assert len(sel) == 150


class TestSummaryExtended:
    """Extended summary() coverage for various state combinations."""

    def test_summary_empty_state(self):
        state = DeepLensState()
        s = state.summary()
        assert "0 samples" in s
        assert "0 features" in s

    def test_summary_includes_class_names(self, iris_state):
        s = iris_state.summary()
        assert "Classes" in s
        assert "setosa" in s

    def test_summary_without_model(self, iris_state):
        s = iris_state.summary()
        assert "Model" not in s
        assert "Accuracy" not in s

    def test_summary_with_model_accuracy(self, iris_with_model):
        s = iris_with_model.summary()
        assert "Accuracy" in s
        # Accuracy should be a valid float between 0 and 1
        import re
        match = re.search(r"Accuracy:\s*([\d.]+)", s)
        assert match is not None
        acc = float(match.group(1))
        assert 0.0 <= acc <= 1.0

    def test_summary_with_selection(self, iris_state):
        iris_state.selected_indices = [0, 1, 2, 3, 4]
        iris_state.feature_columns = ["sepal length (cm)", "sepal width (cm)"]
        s = iris_state.summary()
        assert "Selected: 5 points" in s

    def test_summary_with_selection_stats(self, iris_state):
        iris_state.selected_indices = [0, 1, 2]
        iris_state.feature_columns = ["sepal length (cm)", "sepal width (cm)"]
        s = iris_state.summary()
        assert "Selection stats" in s

    def test_summary_selection_without_features(self, iris_state):
        iris_state.selected_indices = [0, 1]
        iris_state.feature_columns = []
        s = iris_state.summary()
        assert "Selected: 2 points" in s
        assert "Selection stats" not in s

    def test_summary_with_shap_values(self, iris_with_model):
        # Create a mock SHAP explanation with .values attribute
        class MockShapExplanation:
            def __init__(self, values):
                self.values = values

        state = iris_with_model
        n = len(state.df)
        n_features = len(state.feature_columns)
        # Simulate 2D SHAP values (N, features)
        state.shap_values = MockShapExplanation(
            np.random.randn(n, n_features).astype(np.float32)
        )
        state.selected_indices = [0, 1, 2, 3, 4]
        s = state.summary()
        assert "SHAP" in s
        assert "selected" in s

    def test_summary_with_shap_3d_values(self, iris_with_model):
        # SHAP multiclass: (N, features, classes) — vals.ndim > 1 after mean
        class MockShapExplanation:
            def __init__(self, values):
                self.values = values

        state = iris_with_model
        n = len(state.df)
        n_features = len(state.feature_columns)
        n_classes = 3
        state.shap_values = MockShapExplanation(
            np.abs(np.random.randn(n, n_features, n_classes)).astype(np.float32)
        )
        state.selected_indices = [0, 1, 2]
        s = state.summary()
        assert "SHAP" in s


class TestToSnapshot:
    """Tests for to_snapshot() and snapshot_json()."""

    def test_to_snapshot_keys(self, iris_state):
        snap = iris_state.to_snapshot()
        required_keys = [
            "dataset_name", "n_samples", "n_features", "feature_columns",
            "label_column", "class_names", "model_name", "embedding_method",
            "reduction_method", "n_clusters", "annotations", "selected_indices",
        ]
        for key in required_keys:
            assert key in snap, f"Missing key: {key}"

    def test_to_snapshot_values(self, iris_state):
        snap = iris_state.to_snapshot()
        assert snap["dataset_name"] == "iris"
        assert snap["n_samples"] == 150
        assert snap["n_features"] == len(iris_state.feature_columns)
        assert snap["class_names"] == iris_state.class_names

    def test_to_snapshot_no_model_no_accuracy(self, iris_state):
        snap = iris_state.to_snapshot()
        assert "accuracy" not in snap

    def test_to_snapshot_with_model_has_accuracy(self, iris_with_model):
        snap = iris_with_model.to_snapshot()
        assert "accuracy" in snap
        assert 0.0 <= snap["accuracy"] <= 1.0

    def test_to_snapshot_model_history(self, iris_with_model):
        from sklearn.linear_model import LogisticRegression
        model2 = LogisticRegression(max_iter=100).fit(
            iris_with_model.df[iris_with_model.feature_columns],
            iris_with_model.labels,
        )
        iris_with_model.model_history = [
            ("LogisticRegression", iris_with_model.trained_model),
            ("LogisticRegression_v2", model2),
        ]
        snap = iris_with_model.to_snapshot()
        assert "models_trained" in snap
        assert "LogisticRegression" in snap["models_trained"]
        assert "LogisticRegression_v2" in snap["models_trained"]

    def test_snapshot_json_is_valid_json(self, iris_state):
        import json
        json_str = iris_state.snapshot_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["dataset_name"] == "iris"

    def test_snapshot_json_pretty_printed(self, iris_state):
        json_str = iris_state.snapshot_json()
        # json.dumps with indent=2 will have newlines
        assert "\n" in json_str

    def test_to_snapshot_annotations(self, iris_state):
        iris_state.annotations = {0: "interesting", 5: "outlier"}
        snap = iris_state.to_snapshot()
        assert snap["annotations"] == {0: "interesting", 5: "outlier"}

    def test_to_snapshot_selected_indices(self, iris_state):
        iris_state.selected_indices = [1, 2, 3]
        snap = iris_state.to_snapshot()
        assert snap["selected_indices"] == [1, 2, 3]

    def test_to_snapshot_empty_state(self):
        state = DeepLensState()
        snap = state.to_snapshot()
        assert snap["n_samples"] == 0
        assert snap["n_features"] == 0
        assert snap["feature_columns"] == []
