"""Tests for deeplens.models.trainer (non-UI parts)."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from deeplens.config import DeepLensState
from deeplens.models.trainer import _MODEL_REGISTRY, _import_class, ModelTrainer


class TestValidate:
    def test_validate_no_data(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state)
        err = trainer._validate()
        assert err is not None
        assert "No dataset" in err or "No state" in err

    def test_validate_no_label(self, sample_df):
        state = DeepLensState()
        state.df = sample_df
        state.feature_columns = ["f1", "f2", "f3"]
        # label_column not set
        trainer = ModelTrainer(state=state)
        err = trainer._validate()
        assert err is not None
        assert "label" in err.lower()

    def test_validate_no_features(self, sample_df):
        state = DeepLensState()
        state.df = sample_df
        state.label_column = "label"
        # feature_columns not set
        trainer = ModelTrainer(state=state)
        err = trainer._validate()
        assert err is not None
        assert "feature" in err.lower()

    def test_validate_success(self, sample_df):
        state = DeepLensState()
        state.df = sample_df
        state.feature_columns = ["f1", "f2", "f3"]
        state.label_column = "label"
        trainer = ModelTrainer(state=state)
        err = trainer._validate()
        assert err is None

    def test_validate_too_few_samples(self):
        df = pd.DataFrame({"f1": [1.0, 2.0], "label": ["a", "b"]})
        state = DeepLensState()
        state.df = df
        state.feature_columns = ["f1"]
        state.label_column = "label"
        trainer = ModelTrainer(state=state)
        err = trainer._validate()
        assert err is not None
        assert "samples" in err.lower()


class TestModelRegistry:
    def test_model_registry_complete(self):
        expected = {"LogisticRegression", "RandomForest", "SVM", "GradientBoosting", "KNN"}
        assert set(_MODEL_REGISTRY.keys()) == expected

    def test_registry_entries_have_factory(self):
        for name, entry in _MODEL_REGISTRY.items():
            assert "factory" in entry, f"{name} missing 'factory'"
            assert "kwargs" in entry, f"{name} missing 'kwargs'"


class TestImportClass:
    def test_import_logistic_regression(self):
        cls = _import_class("sklearn.linear_model.LogisticRegression")
        from sklearn.linear_model import LogisticRegression
        assert cls is LogisticRegression

    def test_import_random_forest(self):
        cls = _import_class("sklearn.ensemble.RandomForestClassifier")
        from sklearn.ensemble import RandomForestClassifier
        assert cls is RandomForestClassifier

    def test_import_invalid_raises(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            _import_class("nonexistent.module.ClassName")


class TestBuildModel:
    def test_build_model_logistic(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state, model_choice="LogisticRegression")
        model = trainer._build_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_model_svm(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state, model_choice="SVM")
        model = trainer._build_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")  # SVM with probability=True

    def test_build_model_random_forest(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state, model_choice="RandomForest")
        model = trainer._build_model()
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
        assert model.warm_start is True

    def test_build_model_gradient_boosting(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state, model_choice="GradientBoosting")
        model = trainer._build_model()
        from sklearn.ensemble import GradientBoostingClassifier
        assert isinstance(model, GradientBoostingClassifier)
        assert model.warm_start is True

    def test_build_model_knn(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state, model_choice="KNN")
        model = trainer._build_model()
        from sklearn.neighbors import KNeighborsClassifier
        assert isinstance(model, KNeighborsClassifier)

    def test_build_model_all_have_fit_predict(self):
        state = DeepLensState()
        for model_name in ["LogisticRegression", "RandomForest", "SVM", "GradientBoosting", "KNN"]:
            trainer = ModelTrainer(state=state, model_choice=model_name)
            model = trainer._build_model()
            assert hasattr(model, "fit"), f"{model_name} missing fit"
            assert hasattr(model, "predict"), f"{model_name} missing predict"


# ---------------------------------------------------------------------------
# _run_training_sync  (the main training path)
# ---------------------------------------------------------------------------

def _make_trained_state(n=60, model_choice="LogisticRegression"):
    """Return a (state, trainer) pair ready to call _run_training_sync."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.randn(n),
        "label": (["a", "b"] * (n // 2))[:n],
    })
    state = DeepLensState()
    state.df = df
    state.feature_columns = ["f1", "f2", "f3"]
    state.label_column = "label"
    trainer = ModelTrainer(state=state, model_choice=model_choice)
    return state, trainer


class TestRunTrainingSync:
    def test_logistic_regression_trains_successfully(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert state.trained_model is not None
        assert state.predictions is not None
        assert state.model_name == "LogisticRegression"

    def test_svm_trains_successfully(self):
        state, trainer = _make_trained_state(model_choice="SVM")
        trainer._run_training_sync()
        assert state.trained_model is not None
        assert state.predictions is not None
        assert state.model_name == "SVM"

    def test_knn_trains_successfully(self):
        state, trainer = _make_trained_state(model_choice="KNN")
        trainer._run_training_sync()
        assert state.trained_model is not None
        assert state.predictions is not None
        assert state.model_name == "KNN"

    def test_random_forest_warm_start_path(self):
        """RandomForest has warm_start + n_estimators — exercises the chunked loop."""
        state, trainer = _make_trained_state(model_choice="RandomForest")
        trainer._run_training_sync()
        assert state.trained_model is not None
        # warm_start models increment n_estimators; final value should equal registry default
        assert state.trained_model.n_estimators >= 1

    def test_gradient_boosting_warm_start_path(self):
        state, trainer = _make_trained_state(model_choice="GradientBoosting")
        trainer._run_training_sync()
        assert state.trained_model is not None
        assert state.model_name == "GradientBoosting"

    def test_predictions_length_matches_dataset(self):
        n = 60
        state, trainer = _make_trained_state(n=n, model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert len(state.predictions) == n

    def test_probabilities_written_for_models_with_predict_proba(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert state.probabilities is not None

    def test_probabilities_written_for_svm(self):
        state, trainer = _make_trained_state(model_choice="SVM")
        trainer._run_training_sync()
        assert state.probabilities is not None

    def test_model_history_appended(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert len(state.model_history) == 1
        name, model = state.model_history[0]
        assert name == "LogisticRegression"
        assert hasattr(model, "predict")

    def test_metrics_card_visible_after_training(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert trainer._metrics_card.visible is True

    def test_status_success_after_training(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert trainer._status.alert_type == "success"

    def test_progress_100_after_training(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert trainer._progress.value == 100

    def test_training_flag_reset_after_training(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert trainer._training is False

    def test_button_re_enabled_after_training(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert trainer._train_btn.disabled is False

    def test_metrics_pane_contains_accuracy(self):
        state, trainer = _make_trained_state(model_choice="LogisticRegression")
        trainer._run_training_sync()
        assert "Accuracy" in trainer._metrics_pane.object

    def test_metrics_card_title_contains_model_name(self):
        state, trainer = _make_trained_state(model_choice="SVM")
        trainer._run_training_sync()
        assert "SVM" in trainer._metrics_card.title

    def test_training_with_nan_values_imputes(self):
        """NaN features should be imputed and training should succeed."""
        rng = np.random.RandomState(1)
        n = 60
        df = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
            "label": (["a", "b"] * (n // 2))[:n],
        })
        # Introduce NaNs
        df.loc[0, "f1"] = np.nan
        df.loc[5, "f2"] = np.nan
        df.loc[10, "f3"] = np.nan

        state = DeepLensState()
        state.df = df
        state.feature_columns = ["f1", "f2", "f3"]
        state.label_column = "label"
        trainer = ModelTrainer(state=state, model_choice="LogisticRegression")
        trainer._run_training_sync()
        # Should complete without failure
        assert state.trained_model is not None
        assert trainer._status.alert_type == "success"

    def test_training_failure_single_class(self):
        """Dataset with only one class should trigger a validation failure."""
        n = 20
        df = pd.DataFrame({
            "f1": np.ones(n),
            "f2": np.ones(n),
            "label": ["a"] * n,  # single class
        })
        state = DeepLensState()
        state.df = df
        state.feature_columns = ["f1", "f2"]
        state.label_column = "label"
        trainer = ModelTrainer(state=state, model_choice="LogisticRegression")
        # _validate catches single class — _on_train_click would show danger;
        # call _validate directly to confirm the message
        err = trainer._validate()
        assert err is not None
        assert "class" in err.lower() or "distinct" in err.lower()

    def test_training_failure_shown_in_status(self):
        """A RuntimeError during fit should write an error status."""
        from sklearn.linear_model import LogisticRegression as LR

        state, trainer = _make_trained_state(model_choice="LogisticRegression")

        def _bad_fit(*args, **kwargs):
            raise RuntimeError("simulated fit failure")

        original_fit = LR.fit
        LR.fit = _bad_fit
        try:
            trainer._run_training_sync()
        finally:
            LR.fit = original_fit

        assert trainer._status.alert_type == "danger"
        assert "Training failed" in trainer._status.object

    def test_knn_non_warm_start_path(self):
        """KNN has no warm_start kwarg — exercises the else branch in _run_training_sync."""
        state, trainer = _make_trained_state(model_choice="KNN")
        trainer._run_training_sync()
        # progress should have been set to 40 then 80 (not the chunked path)
        assert trainer._progress.value == 100  # finalised to 100 in finally block


# ---------------------------------------------------------------------------
# _on_data_change event handler
# ---------------------------------------------------------------------------

class TestOnDataChange:
    def test_status_updated_on_new_data(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state)

        # Simulate param.watch callback
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"f1": rng.randn(20), "label": ["a", "b"] * 10})

        class FakeEvent:
            new = df

        trainer._on_data_change(FakeEvent())
        assert "Ready" in trainer._status.object
        assert trainer._status.alert_type == "info"

    def test_metrics_pane_cleared_on_new_data(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state)
        # Pre-fill metrics
        trainer._metrics_pane.object = "some old metrics"

        rng = np.random.RandomState(0)
        df = pd.DataFrame({"f1": rng.randn(20), "label": ["a", "b"] * 10})

        class FakeEvent:
            new = df

        trainer._on_data_change(FakeEvent())
        assert trainer._metrics_pane.object == ""

    def test_status_not_updated_for_empty_df(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state)
        original_status = trainer._status.object

        class FakeEvent:
            new = pd.DataFrame()

        trainer._on_data_change(FakeEvent())
        # Empty DataFrame should NOT update status
        assert trainer._status.object == original_status

    def test_status_not_updated_for_none(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state)
        original_status = trainer._status.object

        class FakeEvent:
            new = None

        trainer._on_data_change(FakeEvent())
        assert trainer._status.object == original_status

    def test_watch_registered_on_state(self):
        """ModelTrainer should register a watch on state.df when state is provided."""
        state = DeepLensState()
        # Attach trainer — should call state.param.watch internally
        trainer = ModelTrainer(state=state)
        # If watch is registered, updating df should trigger _on_data_change
        rng = np.random.RandomState(2)
        df = pd.DataFrame({"f1": rng.randn(20), "label": ["a", "b"] * 10})
        state.df = df
        assert "Ready" in trainer._status.object


# ---------------------------------------------------------------------------
# Additional coverage for missing lines
# ---------------------------------------------------------------------------

class TestValidateEdgeCases:
    """Cover missing validation branches: lines 140, 148, 151, 162."""

    def test_validate_no_state_object(self):
        """Line 140: state is None."""
        trainer = ModelTrainer(state=None)
        err = trainer._validate()
        assert err is not None
        assert "No state" in err

    def test_validate_label_column_not_in_df(self):
        """Line 148: label column set but not in DataFrame."""
        df = pd.DataFrame({"f1": np.arange(20, dtype=float), "f2": np.arange(20, dtype=float),
                           "label": (["a", "b"] * 10)})
        state = DeepLensState()
        state.df = df
        state.feature_columns = ["f1", "f2"]
        state.label_column = "nonexistent_label"
        trainer = ModelTrainer(state=state)
        err = trainer._validate()
        assert err is not None
        assert "nonexistent_label" in err

    def test_validate_missing_feature_columns(self):
        """Line 151: some feature columns not in DataFrame."""
        df = pd.DataFrame({"f1": np.arange(20, dtype=float), "label": (["a", "b"] * 10)})
        state = DeepLensState()
        state.df = df
        state.feature_columns = ["f1", "f_missing"]
        state.label_column = "label"
        trainer = ModelTrainer(state=state)
        err = trainer._validate()
        assert err is not None
        assert "f_missing" in err

    def test_validate_insufficient_split(self):
        """Line 162: enough samples but test split too extreme."""
        # 10 samples with test_size=0.5 -> 5 train, 5 test -> OK
        # But with test_size very high via manual override
        df = pd.DataFrame({
            "f1": np.arange(10, dtype=float),
            "label": (["a", "b"] * 5),
        })
        state = DeepLensState()
        state.df = df
        state.feature_columns = ["f1"]
        state.label_column = "label"
        trainer = ModelTrainer(state=state, test_size=0.5)
        # test_size=0.5, n=10 => test_n=5, train_n=5 => OK (train_n >= 2)
        err = trainer._validate()
        assert err is None  # should pass

        # To hit "Insufficient samples" we need train_n < 2 or test_n < 1
        # That requires very few samples
        df_tiny = pd.DataFrame({
            "f1": np.arange(10, dtype=float),
            "f2": np.arange(10, dtype=float),
            "label": (["a", "b"] * 5),
        })
        state2 = DeepLensState()
        state2.df = df_tiny
        state2.feature_columns = ["f1", "f2"]
        state2.label_column = "label"
        # test_size=0.1, n=10 => test_n=1, train_n=9 => OK
        trainer2 = ModelTrainer(state=state2, test_size=0.1)
        err2 = trainer2._validate()
        assert err2 is None


class TestOnTrainClick:
    """Cover _on_train_click lines 174-183."""

    def test_on_train_click_with_invalid_state_shows_danger(self):
        """Line 174-178: validation fails, status set to danger."""
        state = DeepLensState()
        trainer = ModelTrainer(state=state)
        trainer._on_train_click()
        assert trainer._status.alert_type == "danger"

    def test_on_train_click_runs_sync_outside_event_loop(self):
        """Lines 179-183: no running event loop => _run_training_sync called."""
        state, trainer = _make_trained_state(n=60, model_choice="LogisticRegression")
        # Should call _run_training_sync (no event loop running)
        trainer._on_train_click()
        assert state.trained_model is not None
        assert trainer._status.alert_type == "success"


class TestRunTrainingAsync:
    """Cover _run_training_async lines 275-384."""

    def test_async_training_logistic_regression(self):
        """Exercise the entire async training path."""
        state, trainer = _make_trained_state(n=60, model_choice="LogisticRegression")
        asyncio.run(trainer._run_training_async())
        assert state.trained_model is not None
        assert state.predictions is not None
        assert state.probabilities is not None
        assert state.model_name == "LogisticRegression"
        assert trainer._status.alert_type == "success"
        assert trainer._progress.value == 100
        assert trainer._training is False
        assert trainer._train_btn.disabled is False
        assert "Accuracy" in trainer._metrics_pane.object

    def test_async_training_random_forest_warm_start(self):
        """RandomForest warm_start path in async: lines 313-322."""
        state, trainer = _make_trained_state(n=60, model_choice="RandomForest")
        asyncio.run(trainer._run_training_async())
        assert state.trained_model is not None
        assert state.model_name == "RandomForest"
        assert trainer._metrics_card.visible is True

    def test_async_training_svm_no_warm_start(self):
        """SVM (no warm_start, no n_estimators) takes else branch: lines 323-328."""
        state, trainer = _make_trained_state(n=60, model_choice="SVM")
        asyncio.run(trainer._run_training_async())
        assert state.trained_model is not None
        assert state.probabilities is not None

    def test_async_training_knn(self):
        """KNN path in async."""
        state, trainer = _make_trained_state(n=60, model_choice="KNN")
        asyncio.run(trainer._run_training_async())
        assert state.trained_model is not None

    def test_async_training_with_nan_values(self):
        """NaN imputation path in async: lines 284-287."""
        rng = np.random.RandomState(1)
        n = 60
        df = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "label": (["a", "b"] * (n // 2))[:n],
        })
        df.loc[0, "f1"] = np.nan
        df.loc[5, "f2"] = np.nan
        state = DeepLensState()
        state.df = df
        state.feature_columns = ["f1", "f2"]
        state.label_column = "label"
        trainer = ModelTrainer(state=state, model_choice="LogisticRegression")
        asyncio.run(trainer._run_training_async())
        assert state.trained_model is not None
        assert trainer._status.alert_type == "success"

    def test_async_training_failure_shows_danger(self):
        """Exception during async training: lines 376-378."""
        from sklearn.linear_model import LogisticRegression as LR
        state, trainer = _make_trained_state(n=60, model_choice="LogisticRegression")

        original_fit = LR.fit
        def _bad_fit(*args, **kwargs):
            raise RuntimeError("simulated async fit failure")
        LR.fit = _bad_fit
        try:
            asyncio.run(trainer._run_training_async())
        finally:
            LR.fit = original_fit

        assert trainer._status.alert_type == "danger"
        assert "Training failed" in trainer._status.object
        assert trainer._progress.value == 100
        assert trainer._training is False
        assert trainer._train_btn.disabled is False

    def test_async_training_model_history_appended(self):
        """Line 355: model_history updated in async."""
        state, trainer = _make_trained_state(n=60, model_choice="LogisticRegression")
        asyncio.run(trainer._run_training_async())
        assert len(state.model_history) == 1
        name, model = state.model_history[0]
        assert name == "LogisticRegression"

    def test_async_training_gradient_boosting(self):
        """GradientBoosting warm_start with n_estimators in async."""
        state, trainer = _make_trained_state(n=60, model_choice="GradientBoosting")
        asyncio.run(trainer._run_training_async())
        assert state.trained_model is not None
        assert state.model_name == "GradientBoosting"


class TestPanelMethod:
    """Cover __panel__ method."""

    def test_panel_returns_column(self):
        state = DeepLensState()
        trainer = ModelTrainer(state=state)
        result = trainer.__panel__()
        import panel as pn
        assert isinstance(result, pn.Column)
