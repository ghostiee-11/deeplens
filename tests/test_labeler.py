"""Tests for deeplens.annotate.labeler — ActiveLearningAnnotator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import panel as pn
import pytest

from deeplens.annotate.labeler import ActiveLearningAnnotator, _entropy, _max_entropy
from deeplens.config import DeepLensState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state():
    """Minimal DeepLensState with embeddings and probabilities."""
    rng = np.random.RandomState(7)
    n = 40
    state = DeepLensState()
    state.dataset_name = "test"
    state.df = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n), "label": ["a", "b"] * 20})
    state.feature_columns = ["f1", "f2"]
    state.label_column = "label"
    state.labels = np.array(["a", "b"] * 20)
    state.class_names = ["a", "b"]

    # 2-D embeddings
    state.embeddings_2d = rng.randn(n, 2)

    # Fake probabilities (n, 2)
    raw = rng.dirichlet([1, 1], size=n)
    state.probabilities = raw
    return state


@pytest.fixture
def annotator(base_state):
    """ActiveLearningAnnotator with base_state."""
    return ActiveLearningAnnotator(state=base_state)


@pytest.fixture
def no_state_annotator():
    """ActiveLearningAnnotator with no state (state=None)."""
    return ActiveLearningAnnotator()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_two_classes(self):
        probs = np.array([[0.5, 0.5]])
        result = _entropy(probs)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-6)

    def test_certain_prediction_zero_entropy(self):
        probs = np.array([[1.0, 0.0]])
        result = _entropy(probs)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)

    def test_three_class_uniform(self):
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        result = _entropy(probs)
        expected = np.log2(3)
        np.testing.assert_allclose(result[0], expected, atol=1e-5)

    def test_vectorized_shape(self):
        rng = np.random.RandomState(0)
        probs = rng.dirichlet([1, 1, 1], size=10)
        result = _entropy(probs)
        assert result.shape == (10,)

    def test_zero_probs_handled_gracefully(self):
        probs = np.array([[0.0, 1.0]])
        result = _entropy(probs)
        assert np.isfinite(result[0])


class TestMaxEntropy:
    def test_two_classes(self):
        assert _max_entropy(2) == pytest.approx(1.0, abs=1e-9)

    def test_four_classes(self):
        assert _max_entropy(4) == pytest.approx(2.0, abs=1e-9)

    def test_single_class(self):
        # Edge case: 0 or 1 classes should return 1.0
        assert _max_entropy(1) == 1.0

    def test_one_hundred_classes(self):
        assert _max_entropy(100) == pytest.approx(np.log2(100), abs=1e-9)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestAnnotatorInit:
    def test_init_without_state(self, no_state_annotator):
        """Annotator should construct fine with state=None."""
        ann = no_state_annotator
        assert ann.state is None
        assert ann._annotation_log == []

    def test_init_with_state(self, annotator):
        """Annotator should store the provided state."""
        assert annotator.state is not None
        assert annotator._annotation_log == []

    def test_widgets_created(self, annotator):
        """Key widgets must be created during __init__."""
        assert isinstance(annotator._label_input, pn.widgets.TextInput)
        assert isinstance(annotator._assign_btn, pn.widgets.Button)
        assert isinstance(annotator._suggest_btn, pn.widgets.Button)
        assert isinstance(annotator._undo_btn, pn.widgets.Button)
        assert isinstance(annotator._export_csv_btn, pn.widgets.Button)
        assert isinstance(annotator._export_json_btn, pn.widgets.Button)

    def test_queue_size_default(self, annotator):
        assert annotator.queue_size == 20

    def test_annotation_log_starts_empty(self, annotator):
        assert annotator._annotation_log == []

    def test_history_table_starts_empty(self, annotator):
        assert len(annotator._history_table.value) == 0


# ---------------------------------------------------------------------------
# _compute_entropy
# ---------------------------------------------------------------------------

class TestComputeEntropy:
    def test_returns_array_when_probs_set(self, annotator):
        result = annotator._compute_entropy()
        assert result is not None
        assert result.shape == (40,)
        assert np.all(result >= 0)

    def test_returns_none_with_no_state(self, no_state_annotator):
        result = no_state_annotator._compute_entropy()
        assert result is None

    def test_returns_none_when_probs_are_none(self, base_state):
        base_state.probabilities = None
        ann = ActiveLearningAnnotator(state=base_state)
        assert ann._compute_entropy() is None


# ---------------------------------------------------------------------------
# Suggestion queue
# ---------------------------------------------------------------------------

class TestOnSuggest:
    def test_suggest_sets_selection(self, annotator):
        """_on_suggest should update the selection stream."""
        annotator._on_suggest()
        # After suggest, status should reference the number of suggested points
        assert "Suggested" in annotator._status.object

    def test_suggest_no_probs_sets_error_status(self, base_state):
        base_state.probabilities = None
        ann = ActiveLearningAnnotator(state=base_state)
        ann._on_suggest()
        assert "Error" in ann._status.object

    def test_suggest_all_labeled(self, annotator):
        """If all points are already annotated, _on_suggest should report it."""
        n = len(annotator.state.df)
        annotator.state.annotations = {i: "a" for i in range(n)}
        annotator._on_suggest()
        assert "labeled" in annotator._status.object.lower()

    def test_suggest_respects_queue_size(self, annotator):
        """The number of suggested indices should not exceed queue_size."""
        annotator.queue_size = 5
        annotator._on_suggest()
        # After suggest, selection stream holds at most queue_size items
        suggested = annotator._selection_stream.index
        assert len(suggested) <= 5


# ---------------------------------------------------------------------------
# _on_assign / _on_undo
# ---------------------------------------------------------------------------

class TestOnAssign:
    def test_assign_with_no_label_sets_error(self, annotator):
        annotator._label_input.value = ""
        annotator._on_assign()
        assert "Error" in annotator._status.object

    def test_assign_with_no_selection_sets_error(self, annotator):
        annotator._label_input.value = "setosa"
        # selection stream has no indices by default
        annotator._on_assign()
        assert "Error" in annotator._status.object

    def test_assign_adds_to_annotations(self, annotator):
        annotator._label_input.value = "setosa"
        # Manually inject selected indices into the selection stream
        annotator._selection_stream.event(index=[0, 1, 2])
        annotator._on_assign()
        annotations = annotator.state.annotations
        assert annotations.get(0) == "setosa"
        assert annotations.get(1) == "setosa"
        assert annotations.get(2) == "setosa"

    def test_assign_records_in_log(self, annotator):
        annotator._label_input.value = "versicolor"
        annotator._selection_stream.event(index=[3, 4])
        annotator._on_assign()
        assert len(annotator._annotation_log) == 1
        assert annotator._annotation_log[0]["label"] == "versicolor"
        assert annotator._annotation_log[0]["count"] == 2

    def test_assign_success_status(self, annotator):
        annotator._label_input.value = "virginica"
        annotator._selection_stream.event(index=[5])
        annotator._on_assign()
        assert "virginica" in annotator._status.object
        assert "1" in annotator._status.object

    def test_undo_removes_annotations(self, annotator):
        annotator._label_input.value = "setosa"
        annotator._selection_stream.event(index=[10, 11])
        annotator._on_assign()

        # Sanity: they are now annotated
        assert annotator.state.annotations.get(10) == "setosa"

        annotator._on_undo()
        assert 10 not in annotator.state.annotations
        assert 11 not in annotator.state.annotations

    def test_undo_with_empty_log(self, annotator):
        annotator._on_undo()
        assert "Nothing to undo" in annotator._status.object

    def test_undo_updates_history_table(self, annotator):
        annotator._label_input.value = "a"
        annotator._selection_stream.event(index=[0])
        annotator._on_assign()
        assert len(annotator._history_table.value) == 1

        annotator._on_undo()
        assert len(annotator._history_table.value) == 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestAnnotationExport:
    def _annotate_some(self, annotator, indices, label="classX"):
        annotator._label_input.value = label
        annotator._selection_stream.event(index=indices)
        annotator._on_assign()

    def test_export_csv_no_annotations_gives_error(self, annotator):
        annotator._on_export_csv()
        assert "Error" in annotator._status.object

    def test_export_json_no_annotations_gives_error(self, annotator):
        annotator._on_export_json()
        assert "Error" in annotator._status.object

    def test_export_csv_after_annotation(self, annotator):
        self._annotate_some(annotator, [0, 1, 2], label="setosa")
        annotator._on_export_csv()
        assert "CSV ready" in annotator._status.object
        assert len(annotator._download_pane) == 1

    def test_export_json_after_annotation(self, annotator):
        self._annotate_some(annotator, [5, 6], label="versicolor")
        annotator._on_export_json()
        assert "JSON ready" in annotator._status.object
        assert len(annotator._download_pane) == 1

    def test_annotations_df_has_correct_columns(self, annotator):
        self._annotate_some(annotator, [0, 1], label="a")
        df = annotator._annotations_df()
        assert "index" in df.columns
        assert "label" in df.columns

    def test_annotations_df_includes_embedding_coords(self, annotator):
        self._annotate_some(annotator, [0, 1], label="a")
        df = annotator._annotations_df()
        assert "emb_x" in df.columns
        assert "emb_y" in df.columns

    def test_annotations_df_includes_original_label(self, annotator):
        self._annotate_some(annotator, [0], label="new_label")
        df = annotator._annotations_df()
        assert "original_label" in df.columns

    def test_annotations_df_empty_when_no_annotations(self, annotator):
        df = annotator._annotations_df()
        assert df.empty
        assert list(df.columns) == ["index", "label"]

    def test_multiple_export_clears_previous_download(self, annotator):
        self._annotate_some(annotator, [0], label="a")
        annotator._on_export_csv()
        first_len = len(annotator._download_pane)

        # Export again — pane should be cleared and one new item added
        annotator._on_export_csv()
        assert len(annotator._download_pane) == first_len  # still 1


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------

class TestAnnotatorPanel:
    def test_panel_method_returns_row(self, annotator):
        layout = annotator.__panel__()
        assert isinstance(layout, pn.Row)

    def test_panel_has_three_children(self, annotator):
        layout = annotator.__panel__()
        assert len(layout) == 3

    def test_panel_without_state_does_not_raise(self, no_state_annotator):
        layout = no_state_annotator.__panel__()
        assert layout is not None

    def test_panel_returns_panel_object(self, annotator):
        layout = annotator.__panel__()
        assert isinstance(layout, pn.viewable.Viewable)
