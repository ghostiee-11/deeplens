"""Shared state for the DeepLens dashboard.

A single ``DeepLensState`` instance is created by the dashboard and passed
to every module. Modules watch the params they care about via
``param.depends`` so that the entire UI stays in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import param


class DeepLensState(param.Parameterized):
    """Central reactive state shared across all DeepLens modules."""

    # ── Data ────────────────────────────────────────────────────────────
    dataset_name = param.String(default="", doc="Name of the loaded dataset")
    df = param.DataFrame(doc="Raw dataset as a DataFrame")
    feature_columns = param.List(default=[], doc="Numeric feature column names")
    label_column = param.String(default="", doc="Target / label column name")
    text_column = param.String(default="", doc="Text column name (if any)")
    image_column = param.String(default="", doc="Image path column name (if any)")

    # ── Embeddings ──────────────────────────────────────────────────────
    embeddings_raw = param.Array(doc="Raw high-dimensional embeddings (N, D)")
    embeddings_2d = param.Array(doc="2-D reduced embeddings (N, 2)")
    embedding_method = param.String(default="tfidf", doc="Method used for embeddings")
    reduction_method = param.String(default="pca", doc="DR method used")

    # ── Labels & predictions ────────────────────────────────────────────
    labels = param.Array(doc="Ground-truth labels array")
    predictions = param.Array(doc="Model prediction array")
    probabilities = param.Array(doc="Model prediction probabilities (N, C)")
    class_names = param.List(default=[], doc="Ordered class name strings")

    # ── Model ───────────────────────────────────────────────────────────
    trained_model = param.Parameter(doc="Trained sklearn-compatible model")
    model_name = param.String(default="", doc="Name of the current model")

    # ── SHAP ────────────────────────────────────────────────────────────
    shap_values = param.Parameter(doc="SHAP Explanation object or array")
    shap_expected = param.Parameter(doc="SHAP expected (base) value")

    # ── Selection (cross-filter bridge) ─────────────────────────────────
    selected_indices = param.List(default=[], doc="Indices currently selected across views")

    # ── Clustering ──────────────────────────────────────────────────────
    cluster_labels = param.Array(doc="Auto-cluster labels for embeddings")
    n_clusters = param.Integer(default=5, bounds=(2, 50), doc="Number of clusters")

    # ── Annotation ──────────────────────────────────────────────────────
    annotations = param.Dict(default={}, doc="Manual annotations {index: label}")

    # ── Drift ───────────────────────────────────────────────────────────
    reference_df = param.DataFrame(doc="Reference (training) data for drift detection")
    production_df = param.DataFrame(doc="Production data for drift detection")

    # ── LLM ─────────────────────────────────────────────────────────────
    llm_provider = param.Selector(
        objects=["gemini", "groq", "ollama", "none"],
        default="none",
        doc="Active LLM provider",
    )

    # ── Model history (for Compare tab) ────────────────────────────────
    model_history = param.List(default=[], doc="List of (name, model) tuples for trained models")

    # ── UI ──────────────────────────────────────────────────────────────
    loading = param.Boolean(default=False, doc="Whether data ingestion is in progress")
    theme = param.Selector(
        objects=["dark", "default"],
        default="dark",
        doc="Dashboard color theme",
    )

    # ── Helpers ─────────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        if self.df is not None:
            return len(self.df)
        return 0

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def has_model(self) -> bool:
        return self.trained_model is not None

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings_2d is not None and len(self.embeddings_2d) > 0

    @property
    def has_shap(self) -> bool:
        return self.shap_values is not None

    @property
    def selected_df(self) -> pd.DataFrame | None:
        """Return the subset of ``df`` corresponding to the current selection."""
        if not self.selected_indices or self.df is None:
            return None
        return self.df.iloc[self.selected_indices]

    def summary(self, max_tokens: int = 1500) -> str:
        """Compact text summary of current state for LLM context injection."""
        parts: list[str] = []
        parts.append(f"Dataset: {self.dataset_name} ({self.n_samples} samples, {self.n_features} features)")

        if self.class_names:
            parts.append(f"Classes: {self.class_names}")

        if self.has_model:
            parts.append(f"Model: {self.model_name}")
            if self.predictions is not None and self.labels is not None:
                acc = np.mean(self.predictions == self.labels)
                parts.append(f"Accuracy: {acc:.3f}")

        if self.selected_indices:
            sel = self.selected_df
            parts.append(f"Selected: {len(self.selected_indices)} points")
            if sel is not None and len(self.feature_columns) > 0:
                desc = sel[self.feature_columns].describe().round(3).to_string()
                parts.append(f"Selection stats:\n{desc}")

        if self.has_shap and self.selected_indices:
            sv = self.shap_values
            if hasattr(sv, "values"):
                vals = np.abs(sv.values[self.selected_indices]).mean(axis=0)
                if vals.ndim > 1:
                    vals = vals.mean(axis=-1)
                top_k = min(5, len(vals))
                top_idx = np.argsort(vals)[-top_k:][::-1]
                names = self.feature_columns or [f"f{i}" for i in range(len(vals))]
                top = [(names[i], float(vals[i])) for i in top_idx]
                parts.append(f"Top SHAP features (selected): {top}")

        text = "\n".join(parts)
        # Rough token estimate (1 token ≈ 4 chars)
        if len(text) > max_tokens * 4:
            text = text[: max_tokens * 4] + "\n…(truncated)"
        return text

    def to_snapshot(self) -> dict:
        """Serialize current analysis state for export/reproducibility."""
        import json
        snap = {
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
            "class_names": self.class_names,
            "model_name": self.model_name,
            "embedding_method": self.embedding_method,
            "reduction_method": self.reduction_method,
            "n_clusters": self.n_clusters,
            "annotations": self.annotations,
            "selected_indices": self.selected_indices,
        }
        if self.predictions is not None and self.labels is not None:
            snap["accuracy"] = float(np.mean(self.predictions == self.labels))
        if self.model_history:
            snap["models_trained"] = [name for name, _ in self.model_history]
        return snap

    def snapshot_json(self) -> str:
        """Return snapshot as a JSON string."""
        import json
        return json.dumps(self.to_snapshot(), indent=2, default=str)
