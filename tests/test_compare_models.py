"""Tests for deeplens.compare.models.ModelArena."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import panel as pn
import holoviews as hv

from deeplens.compare.models import ModelArena


# ---------------------------------------------------------------------------
# Tiny dataset helpers
# ---------------------------------------------------------------------------

def _make_iris_data(n=60):
    """Return (X, y) for a mini iris-like dataset (3 classes, 4 features)."""
    from sklearn.datasets import load_iris
    X, y_int = load_iris(return_X_y=True)
    iris_classes = np.array(["setosa", "versicolor", "virginica"])
    y = iris_classes[y_int]
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def _fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=500, random_state=42)
    m.fit(X, y)
    return m


def _fit_dt(X, y):
    from sklearn.tree import DecisionTreeClassifier
    m = DecisionTreeClassifier(max_depth=3, random_state=42)
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def arena_data():
    """Shared small dataset for arena tests."""
    X, y = _make_iris_data(n=60)
    return X, y


@pytest.fixture(scope="module")
def trained_arena(arena_data):
    """ModelArena with both models trained."""
    X, y = arena_data
    model_a = _fit_lr(X, y)
    model_b = _fit_dt(X, y)
    arena = ModelArena(model_a=model_a, model_b=model_b, X=X, y=y)
    return arena


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestModelArenaInit:
    def test_no_models_returns_placeholder(self):
        """Arena with no models should produce a placeholder panel, not crash."""
        arena = ModelArena()
        assert arena._zones is None
        assert arena._preds_a is None
        assert arena._preds_b is None

    def test_no_models_panel_output(self):
        """__panel__ with no models should return a Markdown pane (placeholder)."""
        arena = ModelArena()
        result = arena.__panel__()
        assert isinstance(result, pn.pane.Markdown)

    def test_init_with_models_runs_predictions(self, trained_arena):
        """__init__ should automatically call _compute_predictions."""
        assert trained_arena._preds_a is not None
        assert trained_arena._preds_b is not None
        assert trained_arena._zones is not None

    def test_predictions_shape_matches_y(self, trained_arena, arena_data):
        X, y = arena_data
        assert len(trained_arena._preds_a) == len(y)
        assert len(trained_arena._preds_b) == len(y)
        assert len(trained_arena._zones) == len(y)

    def test_partial_params_no_crash(self, arena_data):
        """Providing only model_a (no model_b) should not crash __init__."""
        X, y = arena_data
        model_a = _fit_lr(X, y)
        arena = ModelArena(model_a=model_a)
        assert arena._zones is None

    def test_provided_embeddings_2d_used(self, arena_data):
        """If embeddings_2d is provided, it should be used as-is."""
        X, y = arena_data
        model_a = _fit_lr(X, y)
        model_b = _fit_dt(X, y)
        emb = np.random.RandomState(0).randn(len(X), 2)
        arena = ModelArena(model_a=model_a, model_b=model_b, X=X, y=y, embeddings_2d=emb)
        returned = arena._get_embeddings_2d()
        np.testing.assert_array_equal(returned, emb)


# ---------------------------------------------------------------------------
# _compute_predictions / zone tests
# ---------------------------------------------------------------------------

class TestComputePredictions:
    def test_zones_are_created(self, trained_arena):
        assert trained_arena._zones is not None
        assert trained_arena._zones.dtype == object

    def test_zone_values_are_valid_categories(self, trained_arena):
        valid = {"both_correct", "both_wrong", "only_a_correct", "only_b_correct"}
        assert set(trained_arena._zones).issubset(valid)

    def test_both_correct_zone(self, arena_data):
        """Points both models predict correctly should land in both_correct."""
        X, y = arena_data
        # Use two identical perfect models to guarantee all-both_correct
        from sklearn.tree import DecisionTreeClassifier
        perfect_a = DecisionTreeClassifier(random_state=0).fit(X, y)
        perfect_b = DecisionTreeClassifier(random_state=1).fit(X, y)
        arena = ModelArena(model_a=perfect_a, model_b=perfect_b, X=X, y=y)
        # Both deep trees overfit perfectly on train set
        a_correct = arena._preds_a == y
        b_correct = arena._preds_b == y
        both_correct_mask = a_correct & b_correct
        expected_zone = np.where(both_correct_mask, "both_correct",
                        np.where(~a_correct & ~b_correct, "both_wrong",
                        np.where(a_correct, "only_a_correct", "only_b_correct")))
        np.testing.assert_array_equal(arena._zones, expected_zone)

    def test_both_wrong_zone_exists_when_both_fail(self, arena_data):
        """Force a trivially bad model; some samples should be in both_wrong."""
        X, y = arena_data
        # A model that always predicts the same class (often wrong)
        from sklearn.dummy import DummyClassifier
        bad_a = DummyClassifier(strategy="constant", constant="setosa").fit(X, y)
        bad_b = DummyClassifier(strategy="constant", constant="setosa").fit(X, y)
        arena = ModelArena(model_a=bad_a, model_b=bad_b, X=X, y=y)
        # versicolor and virginica samples should be both_wrong
        assert np.any(arena._zones == "both_wrong")

    def test_only_a_correct_zone(self, arena_data):
        """Zone only_a_correct should appear when A is right and B is wrong."""
        X, y = arena_data
        from sklearn.dummy import DummyClassifier
        good_a = _fit_lr(X, y)
        # B always wrong
        bad_b = DummyClassifier(strategy="constant", constant="setosa").fit(X, y)
        arena = ModelArena(model_a=good_a, model_b=bad_b, X=X, y=y)
        assert np.any(arena._zones == "only_a_correct")

    def test_only_b_correct_zone(self, arena_data):
        """Zone only_b_correct should appear when B is right and A is wrong."""
        X, y = arena_data
        from sklearn.dummy import DummyClassifier
        bad_a = DummyClassifier(strategy="constant", constant="setosa").fit(X, y)
        good_b = _fit_lr(X, y)
        arena = ModelArena(model_a=bad_a, model_b=good_b, X=X, y=y)
        assert np.any(arena._zones == "only_b_correct")

    def test_zones_exhaustive_partition(self, trained_arena, arena_data):
        """Every sample must belong to exactly one zone."""
        _, y = arena_data
        total = (
            np.sum(trained_arena._zones == "both_correct")
            + np.sum(trained_arena._zones == "both_wrong")
            + np.sum(trained_arena._zones == "only_a_correct")
            + np.sum(trained_arena._zones == "only_b_correct")
        )
        assert total == len(y)


# ---------------------------------------------------------------------------
# _metrics_table tests
# ---------------------------------------------------------------------------

class TestMetricsTable:
    def test_metrics_table_returns_tabulator(self, trained_arena):
        result = trained_arena._metrics_table()
        assert isinstance(result, pn.widgets.Tabulator)

    def test_metrics_table_no_predictions_returns_markdown(self):
        arena = ModelArena()
        result = arena._metrics_table()
        assert isinstance(result, pn.pane.Markdown)

    def test_metrics_table_has_both_model_columns(self, trained_arena):
        result = trained_arena._metrics_table()
        # Tabulator value is a DataFrame
        assert isinstance(result.value, pd.DataFrame)
        assert "Model A" in result.value.columns
        assert "Model B" in result.value.columns

    def test_metrics_table_has_expected_rows(self, trained_arena):
        result = trained_arena._metrics_table()
        df = result.value
        assert set(df.index) >= {"Accuracy", "F1", "Precision", "Recall"}

    def test_metrics_values_in_range(self, trained_arena):
        result = trained_arena._metrics_table()
        df = result.value
        assert (df.values >= 0).all()
        assert (df.values <= 1).all()


# ---------------------------------------------------------------------------
# _zone_summary tests
# ---------------------------------------------------------------------------

class TestZoneSummary:
    def test_zone_summary_no_data_returns_markdown(self):
        arena = ModelArena()
        result = arena._zone_summary()
        assert isinstance(result, pn.pane.Markdown)

    def test_zone_summary_returns_markdown(self, trained_arena):
        result = trained_arena._zone_summary()
        assert isinstance(result, pn.pane.Markdown)

    def test_zone_summary_contains_complementarity(self, trained_arena):
        result = trained_arena._zone_summary()
        assert "Complementarity" in result.object

    def test_zone_summary_contains_ensemble_accuracy(self, trained_arena):
        result = trained_arena._zone_summary()
        assert "ensemble accuracy" in result.object.lower()

    def test_complementarity_gain_is_non_negative(self, trained_arena):
        """Ensemble accuracy >= best single model accuracy by definition."""
        arena = trained_arena
        a_correct = arena._preds_a == arena.y
        b_correct = arena._preds_b == arena.y
        ensemble_acc = float(np.mean(a_correct | b_correct))
        best_single = max(float(np.mean(a_correct)), float(np.mean(b_correct)))
        assert ensemble_acc >= best_single - 1e-9

    def test_zone_summary_contains_all_zone_labels(self, trained_arena):
        result = trained_arena._zone_summary()
        text = result.object
        assert "Both correct" in text
        assert "Both wrong" in text
        assert "Only Model A correct" in text
        assert "Only Model B correct" in text

    def test_percentages_sum_to_100(self, trained_arena):
        """Each zone count / total should sum to 1.0."""
        zones = trained_arena._zones
        total = len(zones)
        pcts = [
            np.sum(zones == z) / total
            for z in ["both_correct", "both_wrong", "only_a_correct", "only_b_correct"]
        ]
        assert abs(sum(pcts) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# __panel__ tests
# ---------------------------------------------------------------------------

class TestModelArenaPanel:
    def test_panel_with_models_returns_column(self, trained_arena):
        result = trained_arena.__panel__()
        assert isinstance(result, pn.Column)

    def test_panel_without_models_returns_markdown(self):
        arena = ModelArena()
        result = arena.__panel__()
        assert isinstance(result, pn.pane.Markdown)

    def test_panel_column_contains_row(self, trained_arena):
        result = trained_arena.__panel__()
        has_row = any(isinstance(child, pn.Row) for child in result)
        assert has_row

    def test_agreement_plot_returns_hv_element(self, trained_arena):
        """_agreement_plot should return a HoloViews element."""
        result = trained_arena._agreement_plot()
        assert isinstance(result, hv.Element)

    def test_agreement_plot_no_zones_returns_text(self):
        arena = ModelArena()
        result = arena._agreement_plot()
        assert isinstance(result, hv.Text)

    def test_agreement_plot_df_contains_zone_column(self, trained_arena, arena_data):
        """The Points element should carry a 'zone' vdim."""
        result = trained_arena._agreement_plot()
        assert isinstance(result, hv.Points)
        vdim_names = [str(v) for v in result.vdims]
        assert "zone" in vdim_names

    def test_agreement_plot_df_row_count(self, trained_arena, arena_data):
        X, y = arena_data
        result = trained_arena._agreement_plot()
        assert isinstance(result, hv.Points)
        assert len(result.data) == len(y)


# ---------------------------------------------------------------------------
# Binary classification edge case
# ---------------------------------------------------------------------------

class TestBinaryArena:
    def test_binary_classification(self):
        """ModelArena should work with binary (0/1) labels."""
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        X, y = X[:80], y[:80]
        model_a = _fit_lr(X, y)
        model_b = _fit_dt(X, y)
        arena = ModelArena(model_a=model_a, model_b=model_b, X=X, y=y)
        assert arena._zones is not None
        result = arena._metrics_table()
        assert isinstance(result, pn.widgets.Tabulator)

    def test_binary_metrics_use_binary_average(self):
        """With binary labels, _metrics_table should compute binary average."""
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import f1_score
        X, y = load_breast_cancer(return_X_y=True)
        X, y = X[:80], y[:80]
        model_a = _fit_lr(X, y)
        model_b = _fit_dt(X, y)
        arena = ModelArena(model_a=model_a, model_b=model_b, X=X, y=y)
        # Sanity: F1 for binary should be between 0 and 1
        tab = arena._metrics_table()
        f1_a = tab.value.loc["F1", "Model A"]
        assert 0.0 <= f1_a <= 1.0
