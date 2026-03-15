"""Model training with streaming progress and metric display.

Trains sklearn classifiers on the current dataset in ``DeepLensState``,
updating ``state.trained_model``, ``state.predictions`` and
``state.probabilities`` when training completes.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import param
import panel as pn
import numpy as np

_executor = ThreadPoolExecutor(max_workers=2)


# ── Model catalogue ─────────────────────────────────────────────────────
_MODEL_REGISTRY: dict[str, dict] = {
    "LogisticRegression": {
        "factory": "sklearn.linear_model.LogisticRegression",
        "kwargs": {"max_iter": 500, "warm_start": True},
    },
    "RandomForest": {
        "factory": "sklearn.ensemble.RandomForestClassifier",
        "kwargs": {"n_estimators": 100, "warm_start": True, "n_jobs": -1},
    },
    "SVM": {
        "factory": "sklearn.svm.SVC",
        "kwargs": {"probability": True, "kernel": "rbf"},
    },
    "GradientBoosting": {
        "factory": "sklearn.ensemble.GradientBoostingClassifier",
        "kwargs": {"n_estimators": 100, "warm_start": True},
    },
    "KNN": {
        "factory": "sklearn.neighbors.KNeighborsClassifier",
        "kwargs": {"n_neighbors": 5, "n_jobs": -1},
    },
}

MIN_SAMPLES = 10


def _import_class(dotted_path: str):
    """Lazily import a class from a dotted module path."""
    parts = dotted_path.rsplit(".", 1)
    module_path, class_name = parts[0], parts[1]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class ModelTrainer(pn.viewable.Viewer):
    """Panel component for training sklearn models with streaming progress.

    Parameters
    ----------
    state : DeepLensState
        Shared application state.  After training the component writes
        ``trained_model``, ``predictions``, ``probabilities`` and
        ``model_name`` back to *state*.
    """

    state = param.ClassSelector(class_=object, doc="DeepLensState instance")

    model_choice = param.Selector(
        objects=list(_MODEL_REGISTRY),
        default="LogisticRegression",
        doc="Classifier to train",
    )
    test_size = param.Number(
        default=0.2,
        bounds=(0.1, 0.5),
        step=0.05,
        doc="Fraction of data held out for evaluation",
    )

    _training = param.Boolean(default=False, doc="Whether training is in progress")

    def __init__(self, **params):
        super().__init__(**params)

        # ── widgets ──────────────────────────────────────────────────────
        self._model_select = pn.widgets.Select.from_param(
            self.param.model_choice, name="Model",
        )
        self._split_slider = pn.widgets.FloatSlider.from_param(
            self.param.test_size, name="Test split", format="0.00",
        )
        self._train_btn = pn.widgets.Button(
            name="Train", button_type="primary", icon="player-play",
        )
        self._train_btn.on_click(self._on_train_click)

        self._progress = pn.indicators.Progress(
            name="Training", active=False, value=0, bar_color="info",
            visible=False, sizing_mode="stretch_width",
        )

        self._status = pn.pane.Alert(
            "Load a dataset to begin.", alert_type="info",
            sizing_mode="stretch_width",
        )

        self._metrics_pane = pn.pane.Markdown(
            "", sizing_mode="stretch_width",
        )
        self._metrics_card = pn.Card(
            self._metrics_pane,
            title="Training Results",
            collapsible=True,
            collapsed=False,
            visible=False,
            sizing_mode="stretch_width",
        )

        # Update status when dataset is loaded
        if self.state is not None:
            self.state.param.watch(self._on_data_change, ["df"])
            if self.state.df is not None and len(self.state.df) > 0:
                self._status.object = "Ready to train. Select a model and click Train."
                self._status.alert_type = "info"

    def _on_data_change(self, event):
        if event.new is not None and len(event.new) > 0:
            self._status.object = "Ready to train. Select a model and click Train."
            self._status.alert_type = "info"
            # Clear stale results from previous dataset
            self._metrics_pane.object = ""

    # ── helpers ──────────────────────────────────────────────────────────

    def _validate(self) -> str | None:
        """Return an error message if the state is not ready for training."""
        s = self.state
        if s is None:
            return "No state object attached."
        if s.df is None or len(s.df) == 0:
            return "No dataset loaded."
        if not s.feature_columns:
            return "No feature columns defined in state."
        if not s.label_column:
            return "No label column defined in state."
        if s.label_column not in s.df.columns:
            return f"Label column '{s.label_column}' not found in DataFrame."
        missing = [c for c in s.feature_columns if c not in s.df.columns]
        if missing:
            return f"Feature columns missing from DataFrame: {missing}"
        n = len(s.df)
        if n < MIN_SAMPLES:
            return f"Need at least {MIN_SAMPLES} samples (got {n})."
        n_classes = s.df[s.label_column].nunique()
        if n_classes < 2:
            return "Need at least 2 distinct classes for classification."
        # Ensure enough samples for the requested split
        test_n = int(n * self.test_size)
        train_n = n - test_n
        if train_n < 2 or test_n < 1:
            return "Insufficient samples for the requested train/test split."
        return None

    def _build_model(self):
        """Instantiate the selected sklearn model (lazy import)."""
        entry = _MODEL_REGISTRY[self.model_choice]
        cls = _import_class(entry["factory"])
        return cls(**entry["kwargs"])

    # ── training logic ───────────────────────────────────────────────────

    def _on_train_click(self, event=None):
        err = self._validate()
        if err:
            self._status.object = err
            self._status.alert_type = "danger"
            return
        try:
            asyncio.get_running_loop()
            async def _do_train():
                await self._run_training_async()
            pn.state.execute(_do_train)
        except RuntimeError:
            self._run_training_sync()

    def _run_training_sync(self):
        """Synchronous training — used in tests and CLI."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        s = self.state
        X = s.df[s.feature_columns].values.astype(np.float64)
        y = np.asarray(s.df[s.label_column])

        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            X = SimpleImputer(strategy="mean").fit_transform(X)

        self._training = True
        self._train_btn.disabled = True
        self._progress.visible = True
        self._progress.active = True
        self._progress.value = 10
        self._status.object = f"Training **{self.model_choice}**..."
        self._status.alert_type = "warning"
        self._metrics_pane.object = ""
        self._metrics_card.visible = False

        try:
            self._progress.value = 20
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y,
            )

            model = self._build_model()
            entry = _MODEL_REGISTRY[self.model_choice]
            if entry["kwargs"].get("warm_start") and hasattr(model, "n_estimators"):
                total = model.n_estimators
                step = max(1, total // 5)
                for i in range(step, total + 1, step):
                    model.n_estimators = i
                    model.fit(X_train, y_train)
                    pct = 20 + int(60 * (i / total))
                    self._progress.value = min(pct, 80)
            else:
                self._progress.value = 40
                model.fit(X_train, y_train)
                self._progress.value = 80

            preds_all = model.predict(X)
            probs_all = model.predict_proba(X) if hasattr(model, "predict_proba") else None

            preds_test = model.predict(X_test)
            acc = accuracy_score(y_test, preds_test)
            f1 = f1_score(
                y_test, preds_test,
                average="weighted",
                zero_division=0,
            )
            self._progress.value = 95

            s.trained_model = model
            s.model_name = self.model_choice
            s.model_history = s.model_history + [(self.model_choice, model)]
            s.predictions = preds_all
            if probs_all is not None:
                s.probabilities = probs_all

            n_train, n_test = len(X_train), len(X_test)
            self._metrics_pane.object = (
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Accuracy | {acc:.4f} |\n"
                f"| F1 (weighted) | {f1:.4f} |\n"
                f"| Train samples | {n_train} |\n"
                f"| Test samples | {n_test} |\n"
            )
            self._metrics_card.title = f"Results -{self.model_choice}"
            self._metrics_card.collapsed = False
            self._metrics_card.visible = True
            self._status.object = "Training complete."
            self._status.alert_type = "success"

        except Exception as exc:
            self._status.object = f"Training failed: {exc}"
            self._status.alert_type = "danger"

        finally:
            self._progress.value = 100
            self._progress.active = False
            self._training = False
            self._train_btn.disabled = False

    async def _run_training_async(self):
        """Train the model asynchronously — heavy compute runs in thread pool."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        loop = asyncio.get_event_loop()
        s = self.state
        X = s.df[s.feature_columns].values.astype(np.float64)
        y = np.asarray(s.df[s.label_column])

        # Handle NaN values in features
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)

        # ── progress: start ──────────────────────────────────────────────
        self._training = True
        self._train_btn.disabled = True
        self._progress.visible = True
        self._progress.active = True
        self._progress.value = 10
        self._status.object = f"Training **{self.model_choice}**..."
        self._status.alert_type = "warning"
        self._metrics_pane.object = ""
        self._metrics_card.visible = False

        try:
            # ── split ────────────────────────────────────────────────────
            self._progress.value = 20
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=42,
                stratify=y,
            )

            # ── fit (offloaded to thread pool) ───────────────────────────
            model = self._build_model()
            entry = _MODEL_REGISTRY[self.model_choice]
            if entry["kwargs"].get("warm_start") and hasattr(model, "n_estimators"):
                total = model.n_estimators
                step = max(1, total // 5)
                for i in range(step, total + 1, step):
                    model.n_estimators = i
                    await loop.run_in_executor(
                        _executor, model.fit, X_train, y_train
                    )
                    pct = 20 + int(60 * (i / total))
                    self._progress.value = min(pct, 80)
            else:
                self._progress.value = 40
                await loop.run_in_executor(
                    _executor, model.fit, X_train, y_train
                )
                self._progress.value = 80

            # ── predict on full dataset (offloaded) ──────────────────────
            preds_all = await loop.run_in_executor(
                _executor, model.predict, X
            )
            probs_all = None
            if hasattr(model, "predict_proba"):
                probs_all = await loop.run_in_executor(
                    _executor, model.predict_proba, X
                )

            # ── test-set metrics ─────────────────────────────────────────
            preds_test = await loop.run_in_executor(
                _executor, model.predict, X_test
            )
            acc = accuracy_score(y_test, preds_test)
            f1 = f1_score(
                y_test, preds_test,
                average="weighted",
                zero_division=0,
            )
            self._progress.value = 95

            # ── write back to state ──────────────────────────────────────
            s.trained_model = model
            s.model_name = self.model_choice
            s.model_history = s.model_history + [(self.model_choice, model)]
            s.predictions = preds_all
            if probs_all is not None:
                s.probabilities = probs_all

            # ── display metrics ──────────────────────────────────────────
            n_train, n_test = len(X_train), len(X_test)
            self._metrics_pane.object = (
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Accuracy | {acc:.4f} |\n"
                f"| F1 (weighted) | {f1:.4f} |\n"
                f"| Train samples | {n_train} |\n"
                f"| Test samples | {n_test} |\n"
            )
            self._metrics_card.title = f"Results -{self.model_choice}"
            self._metrics_card.collapsed = False
            self._metrics_card.visible = True
            self._status.object = "Training complete."
            self._status.alert_type = "success"

        except Exception as exc:
            self._status.object = f"Training failed: {exc}"
            self._status.alert_type = "danger"

        finally:
            self._progress.value = 100
            self._progress.active = False
            self._training = False
            self._train_btn.disabled = False

    # ── panel rendering ──────────────────────────────────────────────────

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("## Model Trainer"),
            self._status,
            self._model_select,
            self._split_slider,
            self._train_btn,
            self._progress,
            self._metrics_card,
            sizing_mode="stretch_width",
        )
