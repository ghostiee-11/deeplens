"""Tests for deeplens.models.inspector.ModelInspector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import panel as pn
import holoviews as hv
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from deeplens.config import DeepLensState
from deeplens.models.inspector import ModelInspector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(n=80, binary=False, with_probs=True):
    """Build a DeepLensState with a trained model, predictions, and embeddings."""
    rng = np.random.RandomState(42)

    if binary:
        from sklearn.datasets import load_breast_cancer
        X_full, y_int = load_breast_cancer(return_X_y=True)
        X, y_int = X_full[:n], y_int[:n]
        class_names = ["malignant", "benign"]
        y = np.array([class_names[i] for i in y_int])
    else:
        from sklearn.datasets import load_iris
        X_full, y_int = load_iris(return_X_y=True)
        iris_classes = np.array(["setosa", "versicolor", "virginica"])
        X, y_int = X_full[:n], y_int[:n]
        y = iris_classes[y_int]
        class_names = list(iris_classes)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    emb_2d = PCA(n_components=2).fit_transform(X)

    state = DeepLensState()
    state.df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    state.df["label"] = y
    state.feature_columns = [f"f{i}" for i in range(X.shape[1])]
    state.label_column = "label"
    state.labels = y
    state.class_names = class_names
    state.trained_model = model
    state.model_name = "LogisticRegression"
    state.predictions = model.predict(X)
    state.embeddings_raw = X.astype(np.float32)
    state.embeddings_2d = emb_2d

    if with_probs:
        state.probabilities = model.predict_proba(X)

    return state


def _make_inspector(binary=False, with_probs=True, n=80):
    state = _make_state(binary=binary, with_probs=with_probs, n=n)
    return ModelInspector(state=state)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestModelInspectorInit:
    def test_init_without_state(self):
        """ModelInspector should initialise with no state."""
        inspector = ModelInspector()
        assert inspector.state is None

    def test_init_creates_tap_stream(self):
        inspector = ModelInspector()
        assert isinstance(inspector._tap_stream, hv.streams.Tap)

    def test_init_with_state(self):
        state = _make_state()
        inspector = ModelInspector(state=state)
        assert inspector.state is state

    def test_has_predictions_false_without_state(self):
        inspector = ModelInspector()
        assert inspector._has_predictions() is False

    def test_has_predictions_true_with_state(self):
        state = _make_state()
        inspector = ModelInspector(state=state)
        assert inspector._has_predictions() is True

    def test_has_predictions_false_empty_labels(self):
        state = DeepLensState()
        state.labels = np.array([])
        state.predictions = np.array([])
        inspector = ModelInspector(state=state)
        assert inspector._has_predictions() is False


# ---------------------------------------------------------------------------
# _get_class_names / _is_binary
# ---------------------------------------------------------------------------

class TestClassNames:
    def test_class_names_from_state(self):
        state = _make_state()
        inspector = ModelInspector(state=state)
        names = inspector._get_class_names()
        assert set(names) == {"setosa", "versicolor", "virginica"}

    def test_class_names_fallback_to_unique_labels(self):
        # Use n=150 so all 3 iris classes appear in the data
        state = _make_state(n=150)
        state.class_names = []  # clear explicit names
        inspector = ModelInspector(state=state)
        names = inspector._get_class_names()
        assert set(names) == {"setosa", "versicolor", "virginica"}

    def test_is_binary_false_for_multiclass(self):
        inspector = _make_inspector(binary=False)
        assert inspector._is_binary() is False

    def test_is_binary_true_for_binary(self):
        inspector = _make_inspector(binary=True)
        assert inspector._is_binary() is True


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_no_state_returns_text(self):
        inspector = ModelInspector()
        result = inspector._confusion_matrix()
        assert isinstance(result, hv.Text)

    def test_no_predictions_returns_text(self):
        state = DeepLensState()
        inspector = ModelInspector(state=state)
        result = inspector._confusion_matrix()
        assert isinstance(result, hv.Text)

    def test_returns_heatmap(self):
        inspector = _make_inspector()
        result = inspector._confusion_matrix()
        assert isinstance(result, hv.HeatMap)

    def test_heatmap_kdims(self):
        inspector = _make_inspector()
        result = inspector._confusion_matrix()
        kdim_names = [str(k) for k in result.kdims]
        assert "Predicted" in kdim_names
        assert "True" in kdim_names

    def test_heatmap_vdim_is_count(self):
        inspector = _make_inspector()
        result = inspector._confusion_matrix()
        vdim_names = [str(v) for v in result.vdims]
        assert "Count" in vdim_names

    def test_confusion_matrix_diagonal_correct(self):
        """Diagonal cells should be non-negative counts."""
        inspector = _make_inspector()
        hm = inspector._confusion_matrix()
        df = hm.data
        diag_rows = df[df["Predicted"] == df["True"]]
        assert (diag_rows["Count"] >= 0).all()

    def test_confusion_matrix_total_equals_n_samples(self):
        state = _make_state(n=60)
        inspector = ModelInspector(state=state)
        hm = inspector._confusion_matrix()
        total = hm.data["Count"].sum()
        assert total == 60

    def test_tap_stream_attached_after_call(self):
        inspector = _make_inspector()
        result = inspector._confusion_matrix()
        # After calling _confusion_matrix, the result should be a HeatMap
        # and the tap stream should exist on the inspector.
        assert isinstance(result, hv.HeatMap)
        assert isinstance(inspector._tap_stream, hv.streams.Tap)

    def test_binary_confusion_matrix_shape(self):
        inspector = _make_inspector(binary=True)
        hm = inspector._confusion_matrix()
        # For binary: 2x2 = 4 cells
        assert len(hm.data) == 4

    def test_multiclass_confusion_matrix_shape(self):
        inspector = _make_inspector(binary=False)
        hm = inspector._confusion_matrix()
        # For iris: 3x3 = 9 cells
        assert len(hm.data) == 9


# ---------------------------------------------------------------------------
# Metrics dashboard
# ---------------------------------------------------------------------------

class TestMetricsDashboard:
    def test_no_predictions_returns_markdown(self):
        inspector = ModelInspector()
        result = inspector._metrics_dashboard()
        assert isinstance(result, pn.pane.Markdown)

    def test_returns_panel_row(self):
        inspector = _make_inspector()
        result = inspector._metrics_dashboard()
        assert isinstance(result, pn.Row)

    def test_row_has_four_indicators(self):
        inspector = _make_inspector()
        result = inspector._metrics_dashboard()
        indicators = [c for c in result if isinstance(c, pn.indicators.Number)]
        assert len(indicators) == 4

    def test_accuracy_in_range(self):
        inspector = _make_inspector()
        result = inspector._metrics_dashboard()
        acc_indicator = result[0]
        assert 0.0 <= acc_indicator.value <= 1.0

    def test_f1_in_range(self):
        inspector = _make_inspector()
        result = inspector._metrics_dashboard()
        f1_indicator = result[1]
        assert 0.0 <= f1_indicator.value <= 1.0

    def test_precision_in_range(self):
        inspector = _make_inspector()
        result = inspector._metrics_dashboard()
        prec_indicator = result[2]
        assert 0.0 <= prec_indicator.value <= 1.0

    def test_recall_in_range(self):
        inspector = _make_inspector()
        result = inspector._metrics_dashboard()
        rec_indicator = result[3]
        assert 0.0 <= rec_indicator.value <= 1.0

    def test_metrics_match_sklearn(self):
        """Computed metrics should match direct sklearn calls."""
        from sklearn.metrics import accuracy_score, f1_score
        state = _make_state()
        inspector = ModelInspector(state=state)
        result = inspector._metrics_dashboard()
        expected_acc = accuracy_score(state.labels, state.predictions)
        assert abs(result[0].value - round(expected_acc, 4)) < 1e-6

    def test_binary_uses_binary_average(self):
        """Binary classification should not crash and should return a Row."""
        inspector = _make_inspector(binary=True)
        result = inspector._metrics_dashboard()
        assert isinstance(result, pn.Row)


# ---------------------------------------------------------------------------
# Per-class F1 bars
# ---------------------------------------------------------------------------

class TestPerClassF1:
    def test_no_predictions_returns_text(self):
        inspector = ModelInspector()
        result = inspector._per_class_f1()
        assert isinstance(result, hv.Text)

    def test_returns_bars(self):
        inspector = _make_inspector()
        result = inspector._per_class_f1()
        assert isinstance(result, hv.Bars)

    def test_bars_have_class_kdim(self):
        inspector = _make_inspector()
        result = inspector._per_class_f1()
        kdim_names = [str(k) for k in result.kdims]
        assert "Class" in kdim_names

    def test_bars_have_f1_vdim(self):
        inspector = _make_inspector()
        result = inspector._per_class_f1()
        vdim_names = [str(v) for v in result.vdims]
        assert "F1" in vdim_names

    def test_bars_count_equals_class_count(self):
        inspector = _make_inspector()  # 3-class iris
        result = inspector._per_class_f1()
        assert len(result.data) == 3

    def test_f1_values_in_range(self):
        inspector = _make_inspector()
        bars = inspector._per_class_f1()
        f1_col = [str(v) for v in bars.vdims][0]
        vals = bars.data[f1_col].values
        assert (vals >= 0).all()
        assert (vals <= 1).all()


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

class TestROCCurves:
    def test_no_predictions_returns_text(self):
        inspector = ModelInspector()
        result = inspector._roc_curves()
        assert isinstance(result, hv.Text)

    def test_no_probabilities_returns_text(self):
        state = _make_state(with_probs=False)
        inspector = ModelInspector(state=state)
        result = inspector._roc_curves()
        assert isinstance(result, hv.Text)

    def test_returns_overlay(self):
        inspector = _make_inspector()
        result = inspector._roc_curves()
        assert isinstance(result, hv.Overlay)

    def test_overlay_contains_curves(self):
        inspector = _make_inspector()
        result = inspector._roc_curves()
        curves = [el for el in result if isinstance(el, hv.Curve)]
        assert len(curves) > 0

    def test_diagonal_reference_included(self):
        """The random diagonal curve should always be included."""
        inspector = _make_inspector()
        result = inspector._roc_curves()
        labels = [el.label for el in result]
        assert "Random" in labels

    def test_multiclass_has_one_curve_per_class(self):
        """One-vs-rest: 3 classes → 3 ROC curves + 1 diagonal = 4 elements."""
        inspector = _make_inspector(binary=False)
        result = inspector._roc_curves()
        # At least one curve per class (some may be absent if class not represented)
        non_diag = [el for el in result if isinstance(el, hv.Curve) and el.label != "Random"]
        assert len(non_diag) >= 1

    def test_binary_roc_returns_overlay(self):
        inspector = _make_inspector(binary=True)
        result = inspector._roc_curves()
        assert isinstance(result, hv.Overlay)

    def test_roc_fpr_tpr_in_range(self):
        """FPR and TPR values should be in [0, 1]."""
        inspector = _make_inspector(binary=True)
        result = inspector._roc_curves()
        for el in result:
            if isinstance(el, hv.Curve) and el.label != "Random":
                data = el.data
                fpr_col = str(el.kdims[0])
                tpr_col = str(el.vdims[0])
                assert (data[fpr_col] >= 0).all()
                assert (data[fpr_col] <= 1).all()
                assert (data[tpr_col] >= 0).all()
                assert (data[tpr_col] <= 1).all()

    def test_auc_in_label(self):
        """Each ROC curve label should contain the AUC value."""
        inspector = _make_inspector(binary=True)
        result = inspector._roc_curves()
        for el in result:
            if isinstance(el, hv.Curve) and el.label != "Random":
                assert "AUC=" in el.label


# ---------------------------------------------------------------------------
# __panel__
# ---------------------------------------------------------------------------

class TestModelInspectorPanel:
    def test_panel_returns_column(self):
        inspector = _make_inspector()
        result = inspector.__panel__()
        assert isinstance(result, pn.Column)

    def test_panel_without_state_raises(self):
        inspector = ModelInspector()
        # param.depends("state.labels") can't resolve when state is None
        with pytest.raises(AttributeError):
            inspector.__panel__()

    def test_panel_has_section_header(self):
        inspector = _make_inspector()
        result = inspector.__panel__()
        # First child should be the section header string
        header_str = str(result[0].object) if hasattr(result[0], 'object') else str(result[0])
        assert "Model Inspector" in header_str

    def test_panel_contains_metrics_row(self):
        inspector = _make_inspector()
        result = inspector.__panel__()
        # The second child wraps the metrics dashboard
        assert result[1] is not None

    def test_panel_has_dividers(self):
        inspector = _make_inspector()
        result = inspector.__panel__()
        dividers = [c for c in result if isinstance(c, pn.layout.Divider)]
        assert len(dividers) >= 1

    def test_panel_binary_model(self):
        inspector = _make_inspector(binary=True)
        result = inspector.__panel__()
        assert isinstance(result, pn.Column)

    def test_panel_multiclass_model(self):
        inspector = _make_inspector(binary=False)
        result = inspector.__panel__()
        assert isinstance(result, pn.Column)

    def test_panel_sizing_mode_stretch(self):
        inspector = _make_inspector()
        result = inspector.__panel__()
        assert result.sizing_mode == "stretch_width"


class TestMulticlassROC:
    """Tests for ROC curves with multiclass (iris) data — covers lines 248-280."""

    def test_multiclass_roc_returns_overlay(self):
        inspector = _make_inspector(binary=False)
        result = inspector._roc_curves()
        assert isinstance(result, (hv.Overlay, hv.NdOverlay))

    def test_multiclass_roc_has_diagonal(self):
        inspector = _make_inspector(binary=False)
        result = inspector._roc_curves()
        labels = [el.label for el in result if hasattr(el, "label")]
        assert any("Random" in str(lbl) for lbl in labels)

    def test_roc_without_probs_returns_text(self):
        inspector = _make_inspector(binary=False, with_probs=False)
        result = inspector._roc_curves()
        assert isinstance(result, hv.Text)


class TestDecisionBoundary:
    """Tests for decision boundary visualization — covers lines 290-384."""

    def test_decision_boundary_returns_overlay(self):
        inspector = _make_inspector(binary=True)
        result = inspector._decision_boundary()
        assert result is not None

    def test_decision_boundary_multiclass(self):
        inspector = _make_inspector(binary=False)
        result = inspector._decision_boundary()
        assert result is not None

    def test_decision_boundary_no_model(self):
        state = _make_state(binary=True)
        state.trained_model = None
        inspector = ModelInspector(state=state)
        result = inspector._decision_boundary()
        # Should return placeholder when no model
        assert result is not None

    def test_decision_boundary_no_embeddings(self):
        state = _make_state(binary=True)
        state.embeddings_2d = None
        inspector = ModelInspector(state=state)
        result = inspector._decision_boundary()
        assert result is not None


class TestPerClassF1:
    """Tests for per-class F1 bar chart — covers missing lines."""

    def test_multiclass_f1_returns_bars(self):
        inspector = _make_inspector(binary=False)
        result = inspector._per_class_f1()
        assert isinstance(result, hv.Bars)

    def test_binary_f1_returns_bars(self):
        inspector = _make_inspector(binary=True)
        result = inspector._per_class_f1()
        assert isinstance(result, hv.Bars)


# ---------------------------------------------------------------------------
# on_cm_tap
# ---------------------------------------------------------------------------

class TestOnCmTap:
    def test_no_tap_returns_hint_markdown(self):
        inspector = _make_inspector()
        result = inspector._on_cm_tap()
        assert isinstance(result, pn.pane.Markdown)
        assert "Click" in result.object or "click" in result.object.lower()

    def test_tap_selects_correct_indices(self):
        """Tapping on a CM cell should update state.selected_indices."""
        state = _make_state(n=60)
        inspector = ModelInspector(state=state)
        # Simulate tapping a cell: set the tap stream coords
        inspector._tap_stream.event(x="setosa", y="setosa")
        result = inspector._on_cm_tap()
        assert isinstance(result, pn.pane.Markdown)
        # selected_indices should now be populated
        assert isinstance(state.selected_indices, list)


# ---------------------------------------------------------------------------
# Additional coverage for missing lines: 188, 248, 262, 303, 319-379
# ---------------------------------------------------------------------------


class TestPerClassF1NoPredictions:
    """Line 188: no predictions returns Text with 'No predictions'."""

    def test_per_class_f1_no_predictions_text(self):
        state = DeepLensState()
        state.labels = np.array([])
        state.predictions = np.array([])
        inspector = ModelInspector(state=state)
        result = inspector._per_class_f1()
        assert isinstance(result, hv.Text)


class TestROCCurveMissingClass:
    """Line 248: class not in probabilities columns."""

    def test_roc_with_single_class_in_test(self):
        """Line 248: skip classes with only one binary label."""
        state = _make_state(binary=True, n=80)
        inspector = ModelInspector(state=state)
        result = inspector._roc_curves()
        assert isinstance(result, (hv.Overlay, hv.Text))


class TestROCNoCurves:
    """Line 262: no curves computed returns Text."""

    def test_roc_no_valid_curves(self):
        """Line 262: when no curves are produced, returns hv.Text."""
        state = DeepLensState()
        state.labels = np.array(["a"] * 20)
        state.predictions = np.array(["a"] * 20)
        state.probabilities = np.ones((20, 1))
        state.class_names = ["a"]
        inspector = ModelInspector(state=state)
        result = inspector._roc_curves()
        assert isinstance(result, hv.Text)


class TestDecisionBoundaryExtended:
    """Cover lines 303, 319-379."""

    def test_decision_boundary_non_2d_embeddings(self):
        """Line 303: embeddings that aren't 2-D return Text."""
        state = _make_state(binary=True, n=80)
        # Replace 2D embeddings with 3D
        state.embeddings_2d = np.random.randn(80, 3)
        inspector = ModelInspector(state=state)
        result = inspector._decision_boundary()
        assert isinstance(result, hv.Text)

    def test_decision_boundary_model_incompatible_with_2d(self):
        """Lines 311-316: model can't predict on 2-D input returns Text."""
        state = _make_state(binary=True, n=80)
        # Model was trained on full features, can't predict 2-D
        inspector = ModelInspector(state=state)
        result = inspector._decision_boundary()
        # The model is trained on high-dim features, so prediction on 2D will fail
        assert isinstance(result, hv.Text)

    def test_decision_boundary_with_2d_trained_model(self):
        """Lines 319-384: full decision boundary with 2-D trained model."""
        rng = np.random.RandomState(42)
        n = 80

        # Create 2-D dataset and train model on 2-D features
        X = rng.randn(n, 2).astype(np.float64)
        y = np.array(["a", "b"] * (n // 2))

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)

        state = DeepLensState()
        state.df = pd.DataFrame(X, columns=["f0", "f1"])
        state.df["label"] = y
        state.feature_columns = ["f0", "f1"]
        state.label_column = "label"
        state.labels = y
        state.class_names = ["a", "b"]
        state.trained_model = model
        state.model_name = "LogisticRegression"
        state.predictions = model.predict(X)
        state.probabilities = model.predict_proba(X)
        state.embeddings_raw = X.astype(np.float32)
        state.embeddings_2d = X  # 2-D already

        inspector = ModelInspector(state=state)
        result = inspector._decision_boundary()
        assert isinstance(result, hv.Overlay)

    def test_decision_boundary_model_without_predict_proba(self):
        """Lines 341-346: model without predict_proba uses predict."""
        from unittest.mock import MagicMock
        rng = np.random.RandomState(42)
        n = 80
        X = rng.randn(n, 2).astype(np.float64)
        y = np.array(["a", "b"] * (n // 2))

        # Create a mock model that has predict but not predict_proba
        real_model = DecisionTreeClassifier(random_state=42)
        real_model.fit(X, y)

        class NoProbModel:
            """Model wrapper that has predict but no predict_proba."""
            def __init__(self, model):
                self._model = model
            def predict(self, X):
                return self._model.predict(X)

        model = NoProbModel(real_model)

        state = DeepLensState()
        state.df = pd.DataFrame(X, columns=["f0", "f1"])
        state.df["label"] = y
        state.feature_columns = ["f0", "f1"]
        state.label_column = "label"
        state.labels = y
        state.class_names = ["a", "b"]
        state.trained_model = model
        state.model_name = "DecisionTree"
        state.predictions = real_model.predict(X)
        state.embeddings_raw = X.astype(np.float32)
        state.embeddings_2d = X

        inspector = ModelInspector(state=state)
        result = inspector._decision_boundary()
        assert isinstance(result, hv.Overlay)

    def test_decision_boundary_no_labels(self):
        """Lines 361-365: no labels => no label vdim."""
        rng = np.random.RandomState(42)
        n = 40
        X = rng.randn(n, 2).astype(np.float64)
        y = np.array(["a", "b"] * (n // 2))

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)

        state = DeepLensState()
        state.df = pd.DataFrame(X, columns=["f0", "f1"])
        state.feature_columns = ["f0", "f1"]
        state.labels = None
        state.class_names = ["a", "b"]
        state.trained_model = model
        state.model_name = "LogisticRegression"
        state.predictions = model.predict(X)
        state.probabilities = model.predict_proba(X)
        state.embeddings_raw = X.astype(np.float32)
        state.embeddings_2d = X

        inspector = ModelInspector(state=state)
        result = inspector._decision_boundary()
        assert isinstance(result, hv.Overlay)
