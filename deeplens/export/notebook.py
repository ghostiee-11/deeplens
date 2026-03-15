"""Jupyter notebook exporter for DeepLens analysis sessions.

Generates a self-contained ``.ipynb`` file that reproduces the current
DeepLens analysis — loading the same dataset, training the same model,
computing embeddings, and (optionally) running SHAP — using only standard
scientific-Python libraries so the notebook runs without DeepLens installed.

Usage::

    from deeplens.export import NotebookExporter

    exporter = NotebookExporter(state)
    exporter.save("analysis.ipynb")          # writes file
    nb_dict = exporter.generate()            # returns nbformat-compatible dict
    json_str = exporter.to_json()            # returns JSON string
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from deeplens.config import DeepLensState


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SKLEARN_DATASETS = {
    "iris": "load_iris",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
    "digits": "load_digits",
    "diabetes": "load_diabetes",
}

# nbformat v4 kernel / language metadata
_NOTEBOOK_METADATA: dict[str, Any] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
    },
}


def _md_cell(source: str) -> dict[str, Any]:
    """Return an nbformat v4 markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip(),
    }


def _code_cell(source: str) -> dict[str, Any]:
    """Return an nbformat v4 code cell dict."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip(),
    }


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class NotebookExporter:
    """Export the current DeepLens analysis state as a Jupyter notebook.

    Parameters
    ----------
    state:
        A populated :class:`~deeplens.config.DeepLensState` instance.

    Examples
    --------
    >>> exporter = NotebookExporter(state)
    >>> exporter.save("my_analysis.ipynb")
    """

    def __init__(self, state: DeepLensState) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> dict[str, Any]:
        """Build and return an nbformat v4-compatible notebook dictionary.

        Returns
        -------
        dict
            A dictionary with ``nbformat``, ``nbformat_minor``, ``metadata``,
            and ``cells`` keys that can be written directly as a ``.ipynb``
            JSON file.
        """
        cells: list[dict[str, Any]] = []

        cells.append(self._cell_title())
        cells.append(self._cell_imports())
        cells.append(self._cell_dataset())
        cells.append(self._cell_data_overview())

        if self._state.has_model:
            cells.append(self._cell_model_training())

        if self._state.has_embeddings:
            cells.append(self._cell_embeddings())

        if self._state.has_shap:
            cells.append(self._cell_shap())

        cells.append(self._cell_summary_statistics())

        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": _NOTEBOOK_METADATA,
            "cells": cells,
        }

    def to_json(self) -> str:
        """Return the notebook as a JSON string.

        Returns
        -------
        str
            Pretty-printed JSON representation of the nbformat v4 notebook.
        """
        return json.dumps(self.generate(), indent=2)

    def save(self, path: str) -> None:
        """Write the notebook to *path*.

        Parameters
        ----------
        path:
            Destination file path.  Should end with ``.ipynb``.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    # ------------------------------------------------------------------
    # Cell builders (one private method per logical cell)
    # ------------------------------------------------------------------

    def _cell_title(self) -> dict[str, Any]:
        """Cell 1 — Markdown title and description."""
        state = self._state
        dataset = state.dataset_name or "custom"
        n_samples = state.n_samples
        n_features = state.n_features
        model_line = f"- **Model**: {state.model_name}" if state.has_model else ""
        embed_line = (
            f"- **Embeddings**: {state.embedding_method} → {state.reduction_method}"
            if state.has_embeddings
            else ""
        )
        shap_line = "- **SHAP**: computed" if state.has_shap else ""
        optional_lines = "\n".join(filter(None, [model_line, embed_line, shap_line]))
        if optional_lines:
            optional_lines = "\n" + optional_lines

        source = f"""\
# DeepLens Analysis — {dataset}

*Exported from a DeepLens session.*

## Session summary

- **Dataset**: {dataset}
- **Samples**: {n_samples}
- **Features**: {n_features}{optional_lines}

Run all cells in order to reproduce the analysis.
"""
        return _md_cell(source)

    def _cell_imports(self) -> dict[str, Any]:
        """Cell 2 — Standard library imports."""
        source = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
"""
        return _code_cell(source)

    def _cell_dataset(self) -> dict[str, Any]:
        """Cell 3 — Dataset loading code."""
        state = self._state
        dataset = (state.dataset_name or "").lower()

        if dataset in _SKLEARN_DATASETS:
            loader_fn = _SKLEARN_DATASETS[dataset]
            feature_cols = state.feature_columns
            label_col = state.label_column or "label"

            # Build the feature-column assignment
            if feature_cols:
                cols_repr = repr(feature_cols)
                feature_line = f"feature_columns = {cols_repr}"
            else:
                feature_line = (
                    "feature_columns = [c for c in df.columns "
                    f"if c not in ('{label_col}', 'target')]"
                )

            source = f"""\
# Load the '{dataset}' dataset from scikit-learn
from sklearn.datasets import {loader_fn}

bunch = {loader_fn}(as_frame=True)
df = bunch.frame.copy()

# Map integer target to class names when available
if hasattr(bunch, "target_names") and "target" in df.columns:
    mapping = {{i: name for i, name in enumerate(bunch.target_names)}}
    df["{label_col}"] = df["target"].map(mapping)

label_column = "{label_col}"
{feature_line}
class_names = sorted(df[label_column].unique().tolist())

print(f"Loaded {{len(df)}} samples, {{len(feature_columns)}} features, "
      f"{{len(class_names)}} classes: {{class_names}}")
"""
        elif dataset == "20newsgroups":
            label_col = state.label_column or "label"
            source = f"""\
# Load the 20 Newsgroups text dataset from scikit-learn
from sklearn.datasets import fetch_20newsgroups

bunch = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
)
df = pd.DataFrame({{"text": bunch.data,
                   "{label_col}": [bunch.target_names[t] for t in bunch.target]}})

label_column = "{label_col}"
text_column = "text"
class_names = sorted(df[label_column].unique().tolist())

print(f"Loaded {{len(df)}} samples, {{len(class_names)}} classes")
"""
        else:
            # Generic / custom dataset — emit a placeholder
            feature_cols = state.feature_columns
            label_col = state.label_column or "label"
            cols_repr = repr(feature_cols) if feature_cols else "[]"
            source = f"""\
# TODO: replace this cell with your actual data-loading code.
# The dataset '{dataset}' was loaded interactively in the DeepLens session.
#
# Example:
#   df = pd.read_csv("path/to/your_data.csv")

df = pd.DataFrame()  # placeholder — replace with real loading code

label_column = "{label_col}"
feature_columns = {cols_repr}
class_names = sorted(df[label_column].unique().tolist()) if label_column in df.columns else []
"""
        return _code_cell(source)

    def _cell_data_overview(self) -> dict[str, Any]:
        """Cell 4 — Quick data overview."""
        source = """\
# Data overview
print("Shape:", df.shape)
print("\\nFirst rows:")
display(df.head())

print("\\nDescriptive statistics:")
display(df.describe())
"""
        return _code_cell(source)

    def _cell_model_training(self) -> dict[str, Any]:
        """Cell 5 — Model training (only when a model was trained)."""
        state = self._state
        model_name = state.model_name or "LogisticRegression"
        label_col = state.label_column or "label"

        # Map human-readable model names to sklearn import paths
        _model_imports: dict[str, str] = {
            "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
            "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
            "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
            "SVC": "from sklearn.svm import SVC",
            "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
            "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
        }
        _model_constructors: dict[str, str] = {
            "LogisticRegression": "LogisticRegression(max_iter=500, random_state=42)",
            "RandomForestClassifier": "RandomForestClassifier(n_estimators=100, random_state=42)",
            "GradientBoostingClassifier": "GradientBoostingClassifier(random_state=42)",
            "SVC": "SVC(probability=True, random_state=42)",
            "KNeighborsClassifier": "KNeighborsClassifier()",
            "DecisionTreeClassifier": "DecisionTreeClassifier(random_state=42)",
        }

        import_line = _model_imports.get(
            model_name,
            f"# from sklearn... import {model_name}  # adjust as needed",
        )
        constructor = _model_constructors.get(model_name, f"{model_name}()")

        source = f"""\
# Model training — {model_name}
{import_line}
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X = df[feature_columns].values
le = LabelEncoder()
y = le.fit_transform(df["{label_col}"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = {constructor}
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {{accuracy:.4f}}")
print("\\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
"""
        return _code_cell(source)

    def _cell_embeddings(self) -> dict[str, Any]:
        """Cell 6 — Embedding computation and 2-D scatter plot."""
        state = self._state
        reduction = state.reduction_method or "pca"
        embedding_method = state.embedding_method or "features"
        label_col = state.label_column or "label"

        # Reducer code block
        if reduction == "pca":
            reducer_import = "from sklearn.decomposition import PCA"
            reducer_init = "reducer = PCA(n_components=2, random_state=42)"
        elif reduction == "umap":
            reducer_import = "from umap import UMAP  # pip install umap-learn"
            reducer_init = "reducer = UMAP(n_components=2, random_state=42)"
        elif reduction == "tsne":
            reducer_import = "from sklearn.manifold import TSNE"
            reducer_init = "reducer = TSNE(n_components=2, random_state=42, perplexity=30)"
        else:
            reducer_import = "from sklearn.decomposition import PCA"
            reducer_init = f"# Reduction method '{reduction}' not auto-mapped; defaulting to PCA\nreducer = PCA(n_components=2, random_state=42)"

        # Feature matrix block — depends on embedding method
        if embedding_method == "features":
            feature_matrix_code = "X_embed = df[feature_columns].values"
        elif embedding_method == "tfidf":
            text_col = state.text_column or "text"
            feature_matrix_code = (
                f"from sklearn.feature_extraction.text import TfidfVectorizer\n"
                f"tfidf = TfidfVectorizer(max_features=5000)\n"
                f'X_embed = tfidf.fit_transform(df["{text_col}"].astype(str)).toarray()'
            )
        else:
            feature_matrix_code = (
                "# Embedding method '{}' requires additional setup; "
                "using raw numeric features as fallback.\n"
                "X_embed = df[feature_columns].values".format(embedding_method)
            )

        source = f"""\
# Dimensionality reduction and 2-D scatter plot
# Embedding method: {embedding_method}  |  Reduction: {reduction}
from sklearn.preprocessing import StandardScaler
{reducer_import}

# Build feature matrix
{feature_matrix_code}

# Scale before reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_embed)

# Reduce to 2-D
{reducer_init}
embeddings_2d = reducer.fit_transform(X_scaled)

# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

if "{label_col}" in df.columns:
    labels_plot = df["{label_col}"].values
    unique_labels = sorted(set(labels_plot))
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(unique_labels):
        mask = labels_plot == lbl
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=str(lbl),
            alpha=0.7,
            s=20,
            color=cmap(i / max(len(unique_labels) - 1, 1)),
        )
    ax.legend(title="{label_col}", bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=20)

ax.set_title("{reduction.upper()} Embedding ({embedding_method})")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
plt.tight_layout()
plt.show()
"""
        return _code_cell(source)

    def _cell_shap(self) -> dict[str, Any]:
        """Cell 7 — SHAP values computation."""
        state = self._state
        label_col = state.label_column or "label"

        source = f"""\
# SHAP feature importance
# Requires: pip install shap
import shap

# Use the full feature matrix for SHAP
X_shap = df[feature_columns].values

# TreeExplainer is fastest for tree-based models;
# LinearExplainer for linear models; KernelExplainer as a fallback.
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)
    print("Using TreeExplainer")
except Exception:
    try:
        explainer = shap.LinearExplainer(model, X_shap)
        shap_values = explainer(X_shap)
        print("Using LinearExplainer")
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_shap, 100))
        shap_values = explainer(X_shap[:100])
        print("Using KernelExplainer (on first 100 samples)")

# Summary plot (bar — mean absolute SHAP)
shap.summary_plot(shap_values, X_shap, feature_names=feature_columns, plot_type="bar")

# Detailed dot plot
shap.summary_plot(shap_values, X_shap, feature_names=feature_columns)
"""
        return _code_cell(source)

    def _cell_summary_statistics(self) -> dict[str, Any]:
        """Cell 8 — Summary statistics block."""
        state = self._state
        label_col = state.label_column or "label"
        feature_cols = state.feature_columns or []
        cols_repr = repr(feature_cols)

        source = f"""\
# Summary statistics
print("=== Dataset summary ===")
print(f"Total samples : {{len(df)}}")
print(f"Feature columns ({len(feature_cols)}): {cols_repr}")

if "{label_col}" in df.columns:
    print(f"\\nClass distribution:")
    print(df["{label_col}"].value_counts().to_string())

if feature_columns:
    print("\\nFeature statistics:")
    display(df[feature_columns].describe().round(4))

# Correlation heat-map (numeric features only)
numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(4, len(numeric_cols) - 1)))
    corr = df[numeric_cols].corr()
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
"""
        return _code_cell(source)
