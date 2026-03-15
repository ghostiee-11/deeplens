"""Main DeepLens dashboard - FastListTemplate composing all modules.

Provides a tab-based layout with lazy-loaded modules, sidebar controls
for dataset selection and model training, and cross-filter wiring via
a shared ``DeepLensState`` instance.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import param
import panel as pn

from deeplens.config import DeepLensState
from deeplens.data.loaders import load_sklearn, infer_columns
from deeplens.embeddings.compute import EmbeddingComputer
from deeplens.embeddings.reduce import DimensionalityReducer

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)

pn.extension("tabulator", sizing_mode="stretch_width")

# ── Dataset catalogue ────────────────────────────────────────────────────
_SKLEARN_DATASETS = {
    "iris": "Iris (150 samples, 4 features)",
    "wine": "Wine (178 samples, 13 features)",
    "digits": "Digits (1797 samples, 64 features)",
    "breast_cancer": "Breast Cancer (569 samples, 30 features)",
    "20newsgroups": "20 Newsgroups (text, ~18K samples)",
}


def _safe_import(module_path: str, class_name: str) -> type | None:
    """Import a class, returning ``None`` if a dependency is missing."""
    import importlib

    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not import %s.%s: %s", module_path, class_name, exc)
        return None


# ── Custom CSS ────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
/* ── Hide Bokeh watermark ────────────────────────────────────── */
.bk-logo { display: none !important; }

/* ── Scrollable tabs on narrow screens ───────────────────────── */
.bk-headers {
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    scrollbar-width: thin;
}
.bk-tab {
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    transition: color 0.2s, border-color 0.2s;
    padding: 8px 16px !important;
    font-size: clamp(0.75rem, 1.2vw, 0.95rem) !important;
}
.bk-tab:hover {
    opacity: 0.85;
}

/* ── Responsive sidebar ──────────────────────────────────────── */
@media (max-width: 768px) {
    #sidebar {
        width: 100% !important;
        max-width: 100% !important;
        position: relative !important;
    }
    /* Stack main content below sidebar on mobile */
    .pn-main-area {
        padding: 8px !important;
    }
    .bk-tab {
        padding: 6px 10px !important;
        font-size: 0.8rem !important;
    }
    .pn-indicator-value {
        font-size: 14pt !important;
    }
    /* Let plots shrink gracefully */
    .bk-Canvas, .bk-Figure {
        max-width: 100% !important;
        overflow-x: auto !important;
    }
}
@media (min-width: 769px) and (max-width: 1024px) {
    #sidebar {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    .bk-tab {
        padding: 7px 12px !important;
    }
}
@media (min-width: 1025px) and (max-width: 1400px) {
    #sidebar {
        width: 320px !important;
    }
}

/* ── Fluid main content ──────────────────────────────────────── */
.pn-main-area {
    min-width: 0 !important;
    overflow-x: hidden;
}
.bk-Column, .bk-Row {
    max-width: 100% !important;
}

/* ── Responsive plots / HoloViews ─────────────────────────────── */
.bk-Canvas {
    max-width: 100%;
}
.pn-holoviews {
    width: 100% !important;
    max-width: 100% !important;
}

/* ── Accordion spacing ───────────────────────────────────────── */
.accordion .card {
    margin-bottom: 2px !important;
}
.accordion .card-header {
    padding: 8px 12px !important;
}

/* ── Loading overlay ─────────────────────────────────────────── */
.pn-loading::after {
    background-color: rgba(0, 0, 0, 0.3) !important;
}

/* ── Indicator polish ────────────────────────────────────────── */
.pn-indicator-value {
    font-weight: 600;
}

/* ── Responsive tables ───────────────────────────────────────── */
table {
    max-width: 100% !important;
    overflow-x: auto;
    display: block;
    font-size: clamp(0.75rem, 1vw, 0.9rem);
}

/* ── Float panel responsive ──────────────────────────────────── */
@media (max-width: 768px) {
    .jsPanel {
        width: 95vw !important;
        max-width: 95vw !important;
        left: 2.5vw !important;
    }
}

/* ── Screen reader only ─────────────────────────────────────── */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* ── Skip to content link ──────────────────────────────────── */
.skip-to-content {
    position: absolute;
    top: -40px;
    left: 0;
    background: #1f77b4;
    color: #fff;
    padding: 8px 16px;
    z-index: 10000;
    font-size: 14px;
    text-decoration: none;
    border-radius: 0 0 4px 0;
    transition: top 0.2s;
}
.skip-to-content:focus {
    top: 0;
}

/* ── Alert box polish ────────────────────────────────────────── */
.alert {
    border-radius: 6px !important;
    font-size: 0.9em;
}

/* ── Scrollbar styling ───────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.35);
}

/* ── Smooth transitions ──────────────────────────────────────── */
.bk-Column, .bk-Row, #sidebar {
    transition: width 0.2s ease, max-width 0.2s ease;
}

/* ── Markdown content responsive ─────────────────────────────── */
.bk .markdown p, .bk .markdown li {
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.bk .markdown img {
    max-width: 100%;
    height: auto;
}
"""


# ── Dashboard ────────────────────────────────────────────────────────────

class DeepLensDashboard(pn.viewable.Viewer):
    """Top-level DeepLens dashboard.

    Composes all analysis modules into a single ``FastListTemplate``
    application with cross-filtering driven by a shared ``DeepLensState``.
    """

    state = param.ClassSelector(class_=object, doc="Shared DeepLensState")

    def __init__(self, **params):
        if "state" not in params or params["state"] is None:
            params["state"] = DeepLensState()
        super().__init__(**params)

        # Lazy module cache - populated on first tab activation
        self._tab_cache: dict[str, pn.viewable.Viewer | pn.Column] = {}

        # LLM provider (created once, shared)
        self._llm = self._create_llm()

        # Embedding / reduction helpers
        self._embedder = EmbeddingComputer()
        self._reducer = DimensionalityReducer()

        # ── Sidebar widgets ──────────────────────────────────────────
        self._dataset_select = pn.widgets.Select(
            name="Dataset",
            options=_SKLEARN_DATASETS,
            value=_SKLEARN_DATASETS.get("iris"),
            width=260,
        )
        self._load_btn = pn.widgets.Button(
            name="Load Dataset", button_type="primary", icon="database",
            width=260,
        )
        self._load_btn.on_click(self._on_load_dataset)

        # File upload
        self._file_input = pn.widgets.FileInput(
            accept=".csv,.tsv,.json,.jsonl,.parquet,.xlsx,.xls",
            multiple=False,
            sizing_mode="stretch_width",
        )
        self._upload_btn = pn.widgets.Button(
            name="Upload File", button_type="success", icon="upload",
            width=260,
        )
        self._upload_btn.on_click(self._on_upload_file)

        # Remote URL fetch
        self._url_input = pn.widgets.TextInput(
            name="Remote URL",
            placeholder="https://example.com/data.csv",
            sizing_mode="stretch_width",
        )
        self._fetch_btn = pn.widgets.Button(
            name="Fetch URL", button_type="warning", icon="cloud-download",
            width=260,
        )
        self._fetch_btn.on_click(self._on_fetch_url)

        # Pre-trained model upload
        self._model_file_input = pn.widgets.FileInput(
            accept=".pkl,.joblib,.pickle",
            multiple=False,
            sizing_mode="stretch_width",
        )
        self._model_upload_btn = pn.widgets.Button(
            name="Load Model", button_type="primary", icon="brain",
            width=260,
        )
        self._model_upload_btn.on_click(self._on_upload_model)

        self._status = pn.pane.Alert(
            "Select a dataset and click Load.", alert_type="info",
            sizing_mode="stretch_width",
            css_classes=["status-alert"],
            styles={"role": "status"},
        )

        # Trainer widget (always available in sidebar)
        self._trainer = self._create_trainer()

        # NL filter widget
        self._nl_filter = self._create_nl_filter()

        # LLM analyst chat (launched via FloatPanel)
        self._analyst = self._create_analyst()
        self._chat_float = None
        self._open_chat_btn = pn.widgets.Button(
            name="Open AI Analyst Chat",
            button_type="success",
            icon="message-circle",
            sizing_mode="stretch_width",
        )
        self._open_chat_btn.on_click(self._on_open_chat)

        # Download snapshot button
        self._download_btn = pn.widgets.Button(
            name="Download Snapshot",
            button_type="light",
            icon="download",
            sizing_mode="stretch_width",
        )
        self._download_btn.on_click(self._on_download_snapshot)
        self._download_file = pn.widgets.FileDownload(
            callback=self._generate_snapshot,
            filename="deeplens_snapshot.json",
            button_type="light",
            icon="download",
            label="Download Snapshot",
            sizing_mode="stretch_width",
        )

        # Export as notebook button
        self._notebook_export = pn.widgets.FileDownload(
            callback=self._generate_notebook,
            filename="deeplens_analysis.ipynb",
            button_type="light",
            icon="notebook",
            label="Export as Notebook",
            sizing_mode="stretch_width",
        )

        # Guided tour button (HTML so Shepherd.js can attach a listener)
        self._tour_btn = pn.pane.HTML(
            '<button id="start-tour-btn" style="'
            "width:100%;padding:8px 16px;border:1px solid rgba(255,255,255,0.25);"
            "border-radius:4px;background:transparent;color:#ccc;cursor:pointer;"
            "font-size:0.95em;display:flex;align-items:center;justify-content:center;"
            'gap:6px;transition:background 0.2s,color 0.2s;"'
            ' onmouseover="this.style.background=\'rgba(255,255,255,0.1)\';this.style.color=\'#fff\';"'
            ' onmouseout="this.style.background=\'transparent\';this.style.color=\'#ccc\';"'
            '>'
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
            'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
            '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'
            "</svg>"
            "Take a Tour"
            "</button>",
            sizing_mode="stretch_width",
        )

        # ── Dataset summary indicators ────────────────────────────────
        self._summary_samples = pn.indicators.Number(
            name="Samples", value=0, format="{value:,}",
            font_size="18pt", title_size="10pt",
        )
        self._summary_features = pn.indicators.Number(
            name="Features", value=0, format="{value:,}",
            font_size="18pt", title_size="10pt",
        )
        self._summary_classes = pn.indicators.Number(
            name="Classes", value=0, format="{value:,}",
            font_size="18pt", title_size="10pt",
        )
        self._summary_row = pn.Row(
            self._summary_samples, self._summary_features, self._summary_classes,
            visible=False, sizing_mode="stretch_width",
        )

        # ── Main tabs (lazy-loaded) ──────────────────────────────────
        _welcome = pn.pane.Markdown(
            "## Welcome to DeepLens\n\n"
            "**AI Model Interpretability Explorer** - understand your models "
            "through interactive visualizations.\n\n"
            "### Quick Start\n"
            "1. **Load a dataset** from the sidebar (Iris is pre-selected)\n"
            "2. **Explore** the embedding space with interactive scatter plots\n"
            "3. **Train a model** to unlock Explain, Inspect, Compare & Annotate tabs\n\n"
            "### What each tab does\n"
            "| Tab | Description |\n"
            "|-----|-------------|\n"
            "| **Explore** | Interactive 2-D embedding scatter with lasso selection & similarity search |\n"
            "| **Profile** | Dataset profiling - missing values, correlations, class balance, outliers |\n"
            "| **Explain** | SHAP waterfall & counterfactual explorer for individual predictions |\n"
            "| **Inspect** | Confusion matrix, ROC curves, metrics & error analysis deep dive |\n"
            "| **Compare** | Side-by-side model comparison with agreement zones |\n"
            "| **Drift** | Feature drift detection (KS test, PSI) |\n"
            "| **Quality** | DR quality assessment (Shepard diagrams, trustworthiness) |\n"
            "| **Annotate** | Active learning annotation with uncertainty sampling |\n\n"
            "**Tip:** Press `?` for keyboard shortcuts · `Alt+1-8` to switch tabs\n",
            sizing_mode="stretch_both",
        )
        self._tabs = pn.Tabs(
            ("Explore", _welcome),
            ("Profile", pn.pane.Markdown("*Load a dataset to see profiling overview.*")),
            ("Explain", pn.pane.Markdown("*Load a dataset and train a model to see SHAP explanations & counterfactuals.*")),
            ("Inspect", pn.pane.Markdown("*Train a model to see confusion matrix, ROC curves & metrics.*")),
            ("Compare", pn.pane.Markdown("*Train 2+ models to compare them side-by-side with agreement zones.*")),
            ("Drift", pn.pane.Markdown("*Load a dataset to detect feature drift via KS test & PSI.*")),
            ("Quality", pn.pane.Markdown("*Load a dataset to assess dimensionality reduction quality.*")),
            ("Annotate", pn.pane.Markdown("*Train a model to use the active learning annotation tool.*")),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        self._tabs.param.watch(self._on_tab_change, "active")

        # Reactivity: rebuild active tab when key state params change
        self.state.param.watch(self._on_embeddings_ready, ["embeddings_2d"])
        self.state.param.watch(self._on_model_ready, ["trained_model"])

    # ── LLM factory ──────────────────────────────────────────────────

    @staticmethod
    def _create_llm():
        """Create an LLM provider, falling back to a no-op stub."""
        try:
            from deeplens.analyst.llm import create_llm
            return create_llm("none")
        except Exception:
            return None

    # ── Export / Snapshot ───────────────────────────────────────────

    def _on_download_snapshot(self, event=None):
        """Trigger snapshot download (fallback for button)."""
        pass  # FileDownload widget handles it via callback

    def _generate_snapshot(self):
        """Generate the snapshot JSON for FileDownload."""
        import io
        content = self.state.snapshot_json()
        return io.StringIO(content)

    def _generate_notebook(self):
        """Generate a Jupyter notebook reproducing the current analysis."""
        import io
        try:
            from deeplens.export.notebook import NotebookExporter
            exporter = NotebookExporter(self.state)
            return io.StringIO(exporter.to_json())
        except ImportError:
            return io.StringIO('{"error": "NotebookExporter not available"}')

    # ── Sidebar component factories ──────────────────────────────────

    def _create_trainer(self):
        """Create the ModelTrainer sidebar widget."""
        cls = _safe_import("deeplens.models.trainer", "ModelTrainer")
        if cls is None:
            return pn.pane.Markdown("*ModelTrainer unavailable.*")
        return cls(state=self.state)

    def _create_nl_filter(self):
        """Create the NLFilter sidebar widget."""
        cls = _safe_import("deeplens.analyst.nl_filter", "NLFilter")
        if cls is None:
            return pn.pane.Markdown("*NLFilter unavailable (missing deps).*")
        kwargs: dict[str, Any] = {"state": self.state}
        if self._llm is not None:
            kwargs["llm"] = self._llm
        return cls(**kwargs)

    def _create_analyst(self):
        """Create the DeepLensAnalyst sidebar chat widget."""
        cls = _safe_import("deeplens.analyst.chat", "DeepLensAnalyst")
        if cls is None:
            return pn.pane.Markdown("*LLM Analyst unavailable (missing deps).*")
        kwargs: dict[str, Any] = {"state": self.state}
        if self._llm is not None:
            kwargs["llm"] = self._llm
        return cls(**kwargs)

    def _on_open_chat(self, event=None):
        """Open the AI Analyst in a floating panel."""
        if self._chat_float is None:
            analyst_panel = self._analyst.__panel__() if hasattr(self._analyst, "__panel__") else self._analyst
            self._chat_float = pn.layout.FloatPanel(
                analyst_panel,
                name="DeepLens AI Analyst",
                contained=False,
                position="center",
                width=600,
                height=500,
            )
        self._chat_float.visible = True

    # ── Dataset loading ──────────────────────────────────────────────

    def _dataset_key(self) -> str:
        """Resolve the selected dataset key from the Select widget value."""
        value = self._dataset_select.value
        # value is the descriptive string; find the matching key
        for key, desc in _SKLEARN_DATASETS.items():
            if desc == value:
                return key
        # Fallback: try using value directly
        return value

    def _on_load_dataset(self, event=None):
        """Load the selected sklearn dataset and populate state."""
        key = self._dataset_key()
        self._status.object = f"Loading **{key}**..."
        self._status.alert_type = "info"

        try:
            df = load_sklearn(key)
        except Exception as exc:
            self._status.object = f"Failed to load dataset: {exc}"
            self._status.alert_type = "danger"
            return

        self._ingest_dataframe(df, name=key)

    def _on_upload_file(self, event=None):
        """Handle file upload from the FileInput widget."""
        if self._file_input.value is None:
            self._status.object = "No file selected. Choose a file first."
            self._status.alert_type = "warning"
            return

        filename = self._file_input.filename or "uploaded"
        self._status.object = f"Reading **{filename}**..."
        self._status.alert_type = "info"

        try:
            df = self._read_file_bytes(self._file_input.value, filename)
        except Exception as exc:
            self._status.object = f"Failed to read file: {exc}"
            self._status.alert_type = "danger"
            return

        self._ingest_dataframe(df, name=filename)

    def _on_fetch_url(self, event=None):
        """Fetch a dataset from a remote URL."""
        url = (self._url_input.value or "").strip()
        if not url:
            self._status.object = "Enter a URL first."
            self._status.alert_type = "warning"
            return

        self._status.object = f"Fetching **{url}**..."
        self._status.alert_type = "info"

        try:
            df = self._read_url(url)
        except Exception as exc:
            self._status.object = f"Failed to fetch URL: {exc}"
            self._status.alert_type = "danger"
            return

        # Use last path segment as name
        name = url.rstrip("/").split("/")[-1].split("?")[0] or "remote"
        self._ingest_dataframe(df, name=name)

    def _on_upload_model(self, event=None):
        """Load a pre-trained sklearn model from an uploaded .pkl/.joblib file."""
        if self._model_file_input.value is None:
            self._status.object = "No model file selected. Choose a `.pkl` or `.joblib` file."
            self._status.alert_type = "warning"
            return

        if self.state.df is None:
            self._status.object = "Load a dataset first, then upload your model."
            self._status.alert_type = "warning"
            return

        filename = self._model_file_input.filename or "model"
        self._status.object = f"Loading model **{filename}**..."
        self._status.alert_type = "info"

        try:
            import io
            import pickle
            data = self._model_file_input.value

            # Try joblib first (more common for sklearn), fall back to pickle
            try:
                import joblib
                model = joblib.load(io.BytesIO(data))
            except Exception:
                model = pickle.loads(data)

            # Validate model has predict method
            if not hasattr(model, "predict"):
                self._status.object = "Uploaded object does not have a `predict` method. Expected an sklearn-compatible model."
                self._status.alert_type = "danger"
                return

            # Run predictions on current dataset
            feature_cols = self.state.feature_columns
            if not feature_cols:
                self._status.object = "No feature columns in current dataset. Load a dataset with numeric features first."
                self._status.alert_type = "danger"
                return

            X = self.state.df[feature_cols].values.astype(np.float64)

            # Handle NaN
            if np.isnan(X).any():
                from sklearn.impute import SimpleImputer
                X = SimpleImputer(strategy="mean").fit_transform(X)

            preds = model.predict(X)
            self.state.predictions = preds

            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                self.state.probabilities = probs

            self.state.trained_model = model
            self.state.model_name = filename.rsplit(".", 1)[0]

            # Compute accuracy if labels available
            acc_msg = ""
            if self.state.labels is not None:
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(self.state.labels, preds)
                acc_msg = f" Accuracy: **{acc:.4f}**."

            self._status.object = (
                f"Model **{filename}** loaded. "
                f"Predicted {len(preds)} samples.{acc_msg}"
            )
            self._status.alert_type = "success"

            # Clear tab cache so model-dependent tabs rebuild
            for key in ("Explain", "Inspect", "Compare", "Annotate"):
                self._tab_cache.pop(key, None)
            self._refresh_active_tab()

        except Exception as exc:
            self._status.object = f"Failed to load model: {exc}"
            self._status.alert_type = "danger"

    @staticmethod
    def _read_file_bytes(data: bytes, filename: str) -> pd.DataFrame:
        """Parse uploaded file bytes into a DataFrame."""
        import io

        lower = filename.lower()
        if lower.endswith((".csv", ".tsv")):
            sep = "\t" if lower.endswith(".tsv") else ","
            return pd.read_csv(io.BytesIO(data), sep=sep)
        elif lower.endswith(".json"):
            return pd.read_json(io.BytesIO(data))
        elif lower.endswith(".jsonl"):
            return pd.read_json(io.BytesIO(data), lines=True)
        elif lower.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
        elif lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(data))
        else:
            # Try CSV as fallback
            return pd.read_csv(io.BytesIO(data))

    @staticmethod
    def _read_url(url: str) -> pd.DataFrame:
        """Fetch and parse a remote dataset from URL."""
        lower = url.lower().split("?")[0]
        if lower.endswith(".parquet"):
            return pd.read_parquet(url)
        elif lower.endswith(".json"):
            return pd.read_json(url)
        elif lower.endswith(".jsonl"):
            return pd.read_json(url, lines=True)
        elif lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(url)
        elif lower.endswith(".tsv"):
            return pd.read_csv(url, sep="\t")
        else:
            # Default to CSV
            return pd.read_csv(url)

    def _ingest_dataframe(self, df: pd.DataFrame, name: str = "custom"):
        """Common pipeline: infer columns, compute embeddings, populate state.

        When a running Panel server event loop is available, delegates heavy
        computation to a background thread via ``run_in_executor``.  Falls
        back to synchronous execution (e.g. during tests or CLI).
        """
        try:
            asyncio.get_running_loop()
            # We're inside an event loop - wrap in async def for pn.state.execute
            async def _do_ingest():
                await self._ingest_dataframe_async(df, name)
            pn.state.execute(_do_ingest)
        except RuntimeError:
            # No running event loop - run synchronously
            self._ingest_dataframe_sync(df, name)

    def _ingest_dataframe_sync(self, df: pd.DataFrame, name: str = "custom"):
        """Synchronous ingestion - used in tests and CLI."""
        self._ingest_prepare(df, name)

        cols = infer_columns(df)
        text_col = cols.get("text", "")
        feature_cols = cols.get("features", [])

        self._status.object = "Computing embeddings..."
        try:
            if text_col:
                self._embedder.method = "tfidf"
                raw = self._embedder.compute(df, text_col=text_col)
            elif feature_cols:
                self._embedder.method = "features"
                raw = self._embedder.compute(df[feature_cols])
            else:
                self._status.object = (
                    f"Loaded **{name}** ({len(df)} rows, {len(df.columns)} cols). "
                    f"No numeric features found for embedding."
                )
                self._status.alert_type = "warning"
                self.state.loading = False
                self._tabs.loading = False
                self._tab_cache.clear()
                self._refresh_active_tab()
                return

            self.state.embeddings_raw = raw
            self.state.embedding_method = self._embedder.method

            self._status.object = "Reducing dimensions (PCA)..."
            reduced = self._reducer.reduce(raw)
            self.state.embeddings_2d = reduced
            self.state.reduction_method = self._reducer.method

        except Exception as exc:
            self._status.object = f"Embedding error: {exc}"
            self._status.alert_type = "danger"
            self.state.loading = False
            self._tabs.loading = False
            return

        self._ingest_finalize(df, name)

    async def _ingest_dataframe_async(self, df: pd.DataFrame, name: str = "custom"):
        """Async version - offloads heavy compute to thread pool."""
        loop = asyncio.get_running_loop()
        self._ingest_prepare(df, name)

        cols = infer_columns(df)
        text_col = cols.get("text", "")
        feature_cols = cols.get("features", [])

        self._status.object = "Computing embeddings..."
        try:
            if text_col:
                self._embedder.method = "tfidf"
                raw = await loop.run_in_executor(
                    _executor, lambda: self._embedder.compute(df, text_col=text_col)
                )
            elif feature_cols:
                self._embedder.method = "features"
                raw = await loop.run_in_executor(
                    _executor, self._embedder.compute, df[feature_cols]
                )
            else:
                self._status.object = (
                    f"Loaded **{name}** ({len(df)} rows, {len(df.columns)} cols). "
                    f"No numeric features found for embedding."
                )
                self._status.alert_type = "warning"
                self.state.loading = False
                self._tabs.loading = False
                self._tab_cache.clear()
                self._refresh_active_tab()
                return

            self.state.embeddings_raw = raw
            self.state.embedding_method = self._embedder.method

            self._status.object = "Reducing dimensions (PCA)..."
            reduced = await loop.run_in_executor(
                _executor, self._reducer.reduce, raw
            )
            self.state.embeddings_2d = reduced
            self.state.reduction_method = self._reducer.method

        except Exception as exc:
            self._status.object = f"Embedding error: {exc}"
            self._status.alert_type = "danger"
            self.state.loading = False
            self._tabs.loading = False
            return

        self._ingest_finalize(df, name)

    def _ingest_prepare(self, df: pd.DataFrame, name: str):
        """Reset state and populate metadata - shared by sync/async paths."""
        self.state.loading = True
        self._tabs.loading = True
        self._tab_cache.clear()

        self.state.embeddings_raw = None
        self.state.embeddings_2d = None
        self.state.labels = None
        self.state.predictions = None
        self.state.probabilities = None
        self.state.trained_model = None
        self.state.shap_values = None
        self.state.shap_expected = None
        self.state.cluster_labels = None
        self.state.selected_indices = []
        self.state.annotations = {}
        self.state.reference_df = None
        self.state.production_df = None

        cols = infer_columns(df)
        label_col = cols.get("label", "")
        text_col = cols.get("text", "")
        feature_cols = cols.get("features", [])

        self.state.dataset_name = name
        self.state.df = df
        self.state.label_column = label_col
        self.state.text_column = text_col
        self.state.feature_columns = list(feature_cols) if feature_cols else []

        if label_col and label_col in df.columns:
            unique_labels = df[label_col].unique().tolist()
            self.state.class_names = [str(c) for c in sorted(unique_labels)]
            self.state.labels = np.asarray(df[label_col])

    def _ingest_finalize(self, df: pd.DataFrame, name: str):
        """Update status, indicators, and refresh tabs - shared by sync/async."""
        self.state.loading = False
        self._tabs.loading = False

        cols = infer_columns(df)
        label_col = cols.get("label", "")
        feature_cols = cols.get("features", [])

        n = len(df)
        nf = len(feature_cols)
        n_classes = df[label_col].nunique() if label_col else 0
        self._status.object = (
            f"Loaded **{name}**: {n} samples, {nf} features. "
            f"Embeddings ready ({self._embedder.method} + {self._reducer.method})."
        )
        self._status.alert_type = "success"

        self._summary_samples.value = n
        self._summary_features.value = nf
        self._summary_classes.value = n_classes
        self._summary_row.visible = True

        self._tab_cache.clear()
        self._refresh_active_tab()

    # ── Tab lazy-loading ─────────────────────────────────────────────

    def _on_tab_change(self, event):
        """Called when the user switches tabs.

        Model-dependent tabs (Explain, Inspect, Compare, Annotate) are
        always rebuilt fresh when selected to avoid stale placeholder
        content that was cached before the model was trained.
        """
        idx = self._tabs.active
        tab_names = ["Explore", "Profile", "Explain", "Inspect", "Compare", "Drift", "Quality", "Annotate"]
        if idx < len(tab_names):
            name = tab_names[idx]
            model_tabs = {"Explain", "Inspect", "Compare", "Annotate"}
            if name in model_tabs:
                self._tab_cache.pop(name, None)
        self._refresh_active_tab()

    def _on_embeddings_ready(self, event):
        """Embeddings changed - invalidate all embedding-dependent tabs."""
        for key in ("Explore", "Profile", "Quality", "Drift"):
            self._tab_cache.pop(key, None)
        self._refresh_active_tab()

    def _on_model_ready(self, event):
        """Model trained - refresh model-dependent tabs."""
        for key in ("Explain", "Inspect", "Compare", "Annotate"):
            self._tab_cache.pop(key, None)
        self._refresh_active_tab()

    def _refresh_active_tab(self):
        """Build (or retrieve from cache) the content for the active tab."""
        idx = self._tabs.active
        tab_names = ["Explore", "Profile", "Explain", "Inspect", "Compare", "Drift", "Quality", "Annotate"]
        if idx >= len(tab_names):
            return
        name = tab_names[idx]

        # Don't cache during loading - state is incomplete
        if self.state.loading:
            self._tabs[idx] = (name, pn.pane.Markdown("*Loading dataset...*"))
            return

        if name not in self._tab_cache:
            content = self._build_tab(name)
            if content is not None:
                self._tab_cache[name] = content

        cached = self._tab_cache.get(name)
        if cached is not None:
            self._tabs[idx] = (name, cached)

    def _build_tab(self, name: str) -> pn.viewable.Viewer | pn.Column | None:
        """Lazy-build a tab's content. Returns None if prerequisites unmet."""

        if name == "Explore":
            return self._build_explore()
        elif name == "Profile":
            return self._build_profile()
        elif name == "Explain":
            return self._build_explain()
        elif name == "Inspect":
            return self._build_inspect()
        elif name == "Compare":
            return self._build_compare()
        elif name == "Drift":
            return self._build_drift()
        elif name == "Quality":
            return self._build_quality()
        elif name == "Annotate":
            return self._build_annotate()
        return None

    # ── Tab builders ─────────────────────────────────────────────────

    def _build_profile(self) -> pn.viewable.Viewer | pn.pane.Markdown:
        if self.state.df is None:
            return pn.pane.Markdown("*Load a dataset to see profiling overview.*")
        cls = _safe_import("deeplens.data.profiler", "DatasetProfiler")
        if cls is None:
            return pn.pane.Markdown("*DatasetProfiler unavailable.*")
        try:
            return cls(
                df=self.state.df,
                feature_columns=self.state.feature_columns,
                label_column=self.state.label_column,
            )
        except Exception as exc:
            return pn.pane.Markdown(f"*Profiler error: {exc}*")

    def _build_explore(self) -> pn.viewable.Viewer | pn.pane.Markdown:
        if not self.state.has_embeddings:
            return pn.pane.Markdown("*Load a dataset to see the embedding explorer.*")
        cls = _safe_import("deeplens.embeddings.explorer", "EmbeddingExplorer")
        if cls is None:
            return pn.pane.Markdown("*EmbeddingExplorer unavailable.*")
        return cls(state=self.state)

    def _build_explain(self) -> pn.Column | pn.pane.Markdown:
        if not self.state.has_model:
            return pn.pane.Markdown(
                "### Explain Tab\n\n"
                "SHAP waterfall plots & counterfactual exploration for individual predictions.\n\n"
                "*Train a model in the sidebar to unlock this tab.*"
            )

        parts: list[Any] = []

        # Explainability engine
        engine_cls = _safe_import("deeplens.explain.engine", "ExplainabilityEngine")
        if engine_cls is not None:
            try:
                parts.append(engine_cls(state=self.state))
            except Exception as exc:
                logger.warning("ExplainabilityEngine init failed: %s", exc)
                parts.append(pn.pane.Markdown(f"*SHAP unavailable: {exc}*"))
        else:
            parts.append(pn.pane.Markdown("*SHAP engine unavailable (install shap).*"))

        # Counterfactual explorer
        cf_cls = _safe_import("deeplens.explain.counterfactual", "CounterfactualExplorer")
        if cf_cls is not None:
            try:
                parts.append(cf_cls(state=self.state))
            except Exception as exc:
                logger.warning("CounterfactualExplorer init failed: %s", exc)
                parts.append(pn.pane.Markdown(f"*Counterfactual unavailable: {exc}*"))

        if len(parts) == 2:
            return pn.Row(*parts, sizing_mode="stretch_both")
        return pn.Column(*parts, sizing_mode="stretch_both")

    def _build_inspect(self) -> pn.viewable.Viewer | pn.pane.Markdown:
        if not self.state.has_model:
            return pn.pane.Markdown(
                "### Inspect Tab\n\n"
                "Confusion matrix, ROC curves & classification metrics.\n\n"
                "*Train a model in the sidebar to unlock this tab.*"
            )
        parts: list[Any] = []

        cls = _safe_import("deeplens.models.inspector", "ModelInspector")
        if cls is not None:
            try:
                parts.append(cls(state=self.state))
            except Exception as exc:
                parts.append(pn.pane.Markdown(f"*ModelInspector error: {exc}*"))

        # Error Analysis deep dive
        ea_cls = _safe_import("deeplens.models.error_analysis", "ErrorAnalyzer")
        if ea_cls is not None:
            try:
                parts.append(ea_cls(state=self.state))
            except Exception as exc:
                logger.warning("ErrorAnalyzer init failed: %s", exc)

        if not parts:
            return pn.pane.Markdown("*ModelInspector unavailable.*")
        return pn.Column(*parts, sizing_mode="stretch_both")

    def _build_compare(self) -> pn.viewable.Viewer | pn.pane.Markdown:
        history = self.state.model_history
        if len(history) < 2:
            n_trained = len(history)
            return pn.pane.Markdown(
                f"### Model Arena\n\n"
                f"*Train at least **2 different models** to compare them.*\n\n"
                f"Models trained so far: **{n_trained}**\n\n"
                f"Go to the **Train Model** section in the sidebar, "
                f"select a different model type, and train again."
            )
        cls = _safe_import("deeplens.compare.models", "ModelArena")
        if cls is None:
            return pn.pane.Markdown("*ModelArena unavailable.*")
        # Use last two distinct models from history
        _, model_a = history[-2]
        _, model_b = history[-1]
        try:
            return cls(
                state=self.state,
                model_a=model_a,
                model_b=model_b,
                X=self.state.df[self.state.feature_columns].values if self.state.feature_columns else None,
                y=self.state.labels,
                embeddings_2d=self.state.embeddings_2d,
                feature_names=self.state.feature_columns,
            )
        except Exception as exc:
            return pn.pane.Markdown(f"*ModelArena error: {exc}*")

    def _build_drift(self) -> pn.viewable.Viewer | pn.pane.Markdown:
        if self.state.df is None:
            return pn.pane.Markdown("*Load a dataset first.*")
        cls = _safe_import("deeplens.compare.drift", "DriftDetector")
        if cls is None:
            return pn.pane.Markdown("*DriftDetector unavailable.*")

        # Split dataset in half as a demo reference/production split
        df = self.state.df
        mid = len(df) // 2
        ref_df = self.state.reference_df if self.state.reference_df is not None else df.iloc[:mid]
        prod_df = self.state.production_df if self.state.production_df is not None else df.iloc[mid:]
        try:
            return cls(
                state=self.state,
                reference_df=ref_df,
                production_df=prod_df,
                feature_columns=self.state.feature_columns or None,
            )
        except Exception as exc:
            return pn.pane.Markdown(f"*DriftDetector error: {exc}*")

    def _build_quality(self) -> pn.viewable.Viewer | pn.pane.Markdown:
        if not self.state.has_embeddings:
            return pn.pane.Markdown("*Load a dataset to assess DR quality.*")
        cls = _safe_import("deeplens.quality.dr_quality", "DRQualityDashboard")
        if cls is None:
            return pn.pane.Markdown("*DRQualityDashboard unavailable.*")
        try:
            return cls(
                state=self.state,
                embeddings_raw=self.state.embeddings_raw,
            )
        except Exception as exc:
            return pn.pane.Markdown(f"*DRQuality error: {exc}*")

    def _build_annotate(self) -> pn.viewable.Viewer | pn.pane.Markdown:
        if not self.state.has_model:
            return pn.pane.Markdown(
                "### Annotate Tab\n\n"
                "Active learning annotation with uncertainty sampling.\n\n"
                "*Train a model in the sidebar to unlock this tab.*"
            )
        cls = _safe_import("deeplens.annotate.labeler", "ActiveLearningAnnotator")
        if cls is None:
            return pn.pane.Markdown("*ActiveLearningAnnotator unavailable.*")
        try:
            return cls(state=self.state)
        except Exception as exc:
            return pn.pane.Markdown(f"*Annotator error: {exc}*")

    # ── Layout ───────────────────────────────────────────────────────

    def _build_sidebar(self) -> pn.Column:
        """Assemble sidebar components with collapsible accordion sections."""
        header = pn.pane.Markdown(
            "## DeepLens",
            styles={"text-align": "center", "margin-bottom": "0px"},
        )

        accordion = pn.Accordion(
            (
                "Dataset",
                pn.Column(
                    self._dataset_select,
                    self._load_btn,
                    self._status,
                    self._summary_row,
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "Upload / Fetch",
                pn.Column(
                    pn.pane.Markdown(
                        "**Upload a file**",
                        styles={"font-size": "0.85em"},
                    ),
                    self._file_input,
                    self._upload_btn,
                    pn.layout.Divider(),
                    pn.pane.Markdown(
                        "**Fetch from URL**",
                        styles={"font-size": "0.85em"},
                    ),
                    self._url_input,
                    self._fetch_btn,
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "Train Model",
                pn.Column(self._trainer, sizing_mode="stretch_width"),
            ),
            (
                "Upload Pre-trained Model",
                pn.Column(
                    pn.pane.Markdown(
                        "*Upload a `.pkl` or `.joblib` sklearn model*",
                        styles={"font-size": "0.85em", "color": "#888"},
                    ),
                    self._model_file_input,
                    self._model_upload_btn,
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "NL Filter",
                pn.Column(self._nl_filter, sizing_mode="stretch_width"),
            ),
            (
                "AI Analyst",
                pn.Column(
                    self._open_chat_btn,
                    pn.pane.Markdown(
                        "*Opens in a floating window for better experience.*",
                        styles={"font-size": "0.8em", "color": "#888"},
                    ),
                    sizing_mode="stretch_width",
                ),
            ),
            active=[0],
            toggle=True,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            header,
            accordion,
            pn.layout.Divider(),
            self._download_file,
            self._notebook_export,
            self._tour_btn,
            sizing_mode="stretch_width",
            scroll=True,
        )

    def __panel__(self):
        """Render as a ``FastListTemplate``."""
        # Combined: keyboard shortcuts JS + skip link (hidden) + main anchor
        _kb_js = pn.pane.HTML(
            """<a href="#main-content" class="sr-only" style="position:absolute;left:-9999px;">Skip to main content</a>
            <div id="main-content" tabindex="-1" style="width:0;height:0;overflow:hidden;"></div>
            <script>
            (function() {
                // Build help modal once
                var modal = document.createElement('div');
                modal.id = 'kb-help-modal';
                modal.style.cssText = 'display:none;position:fixed;inset:0;z-index:99999;'
                    + 'background:rgba(0,0,0,0.6);align-items:center;justify-content:center;';
                modal.innerHTML = '<div style="background:#1e1e2e;color:#cdd6f4;border-radius:12px;'
                    + 'padding:28px 36px;max-width:480px;width:90%;box-shadow:0 8px 32px rgba(0,0,0,0.5);'
                    + 'font-family:system-ui,sans-serif;">'
                    + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
                    + '<h2 style="margin:0;font-size:1.3em;">Keyboard Shortcuts</h2>'
                    + '<button id="kb-close" style="background:none;border:none;color:#888;font-size:1.5em;cursor:pointer;">&times;</button></div>'
                    + '<table style="width:100%;border-collapse:collapse;">'
                    + '<tr><td style="padding:6px 12px;color:#89b4fa;font-family:monospace;">Alt + 1-8</td>'
                    + '<td style="padding:6px 12px;">Switch between tabs</td></tr>'
                    + '<tr><td style="padding:6px 12px;color:#89b4fa;font-family:monospace;">?</td>'
                    + '<td style="padding:6px 12px;">Show this help</td></tr>'
                    + '<tr><td style="padding:6px 12px;color:#89b4fa;font-family:monospace;">Esc</td>'
                    + '<td style="padding:6px 12px;">Close modal / deselect</td></tr>'
                    + '</table>'
                    + '<p style="margin:16px 0 0;font-size:0.85em;color:#888;">Explore (1) · Profile (2) · Explain (3) · Inspect (4) · Compare (5) · Drift (6) · Quality (7) · Annotate (8)</p>'
                    + '</div>';
                document.body.appendChild(modal);

                function showModal() { modal.style.display = 'flex'; }
                function hideModal() { modal.style.display = 'none'; }

                modal.addEventListener('click', function(e) {
                    if (e.target === modal || e.target.id === 'kb-close') hideModal();
                });

                document.addEventListener('keydown', function(e) {
                    if (e.altKey && e.key >= '1' && e.key <= '8') {
                        e.preventDefault();
                        var idx = parseInt(e.key) - 1;
                        var tabs = document.querySelectorAll('.bk-tab');
                        if (tabs[idx]) tabs[idx].click();
                    }
                    if (e.key === '?' && !e.ctrlKey && !e.altKey && !e.metaKey
                        && document.activeElement.tagName !== 'INPUT'
                        && document.activeElement.tagName !== 'TEXTAREA') {
                        e.preventDefault();
                        showModal();
                    }
                    if (e.key === 'Escape') hideModal();
                });
            })();
            </script>""",
            width=0, height=0, margin=0, sizing_mode="fixed",
        )

        template = pn.template.FastListTemplate(
            title="DeepLens - AI Model Interpretability Explorer",
            theme="dark",
            sidebar=[self._build_sidebar()],
            main=[_kb_js, self._tabs],
            accent_base_color="#1f77b4",
            header_background="#1f77b4",
            sidebar_width=350,
            raw_css=[_CUSTOM_CSS],
        )
        return template

    # ── Convenience launchers ────────────────────────────────────────

    @classmethod
    def create(cls, dataset: str = "iris", llm_provider: str = "none") -> "DeepLensDashboard":
        """Create a dashboard with initial dataset loaded.

        Parameters
        ----------
        dataset : str
            Name of a sklearn dataset to load immediately.
        llm_provider : str
            LLM provider to use (``'gemini'``, ``'groq'``, ``'ollama'``, or ``'none'``).
        """
        state = DeepLensState()
        dashboard = cls(state=state)

        # Override LLM if requested
        if llm_provider != "none":
            try:
                from deeplens.analyst.llm import create_llm
                dashboard._llm = create_llm(llm_provider)
            except Exception as exc:
                logger.warning("Could not create LLM provider '%s': %s", llm_provider, exc)

        # Trigger dataset load
        if dataset in _SKLEARN_DATASETS:
            # Set the select widget to the right value
            dashboard._dataset_select.value = _SKLEARN_DATASETS[dataset]
            dashboard._on_load_dataset()

        return dashboard


def launch(
    dataset: str = "iris",
    llm_provider: str = "none",
    show: bool = True,
    port: int = 0,
):
    """Create and serve the DeepLens dashboard.

    Parameters
    ----------
    dataset : str
        Sklearn dataset name to load on startup.
    llm_provider : str
        LLM provider (``'gemini'``, ``'groq'``, ``'ollama'``, ``'none'``).
    show : bool
        Whether to open a browser tab automatically.
    port : int
        Port to serve on. ``0`` lets Panel pick a free port.

    Returns
    -------
    DeepLensDashboard
        The dashboard instance (already serving if ``show=True``).
    """
    dashboard = DeepLensDashboard.create(dataset=dataset, llm_provider=llm_provider)
    template = dashboard.__panel__()
    if show:
        kwargs: dict[str, Any] = {}
        if port:
            kwargs["port"] = port
        template.show(**kwargs)
    else:
        template.servable()
    return dashboard
