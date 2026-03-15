"""Custom Lumen Transforms for DeepLens.

Each transform implements ``apply(df) -> df`` following the Lumen
Transform API, enabling declarative data pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from lumen.transforms.base import Transform

    _HAS_LUMEN = True
except ImportError:
    import param

    class Transform(param.Parameterized):  # type: ignore[no-redef]
        transform_type = None
        def apply(self, table: pd.DataFrame) -> pd.DataFrame:
            return table

    _HAS_LUMEN = False


class NormalizeTransform(Transform):
    """StandardScaler or MinMaxScaler normalization on numeric columns."""

    transform_type = "normalize"

    def __init__(self, method: str = "standard", columns: list[str] | None = None, **params: Any):
        super().__init__(**params)
        self._method = method
        self._columns = columns

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        df = table.copy()
        cols = self._columns or list(df.select_dtypes(include=[np.number]).columns)
        if not cols:
            return df

        if self._method == "standard":
            from sklearn.preprocessing import StandardScaler

            df[cols] = StandardScaler().fit_transform(df[cols])
        elif self._method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            df[cols] = MinMaxScaler().fit_transform(df[cols])
        return df


class EmbeddingTransform(Transform):
    """Compute embeddings and append them as columns."""

    transform_type = "embed"

    def __init__(self, method: str = "tfidf", text_column: str = "text", n_components: int = 50, **params: Any):
        super().__init__(**params)
        self._method = method
        self._text_column = text_column
        self._n_components = n_components

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        from deeplens.embeddings.compute import EmbeddingComputer

        table = table.copy()
        computer = EmbeddingComputer(method=self._method)
        emb = computer.compute(table, text_col=self._text_column)
        n_dim = min(self._n_components, emb.shape[1])
        for i in range(n_dim):
            table[f"emb_{i}"] = emb[:, i]
        return table


class DimensionalityReductionTransform(Transform):
    """Apply UMAP / t-SNE / PCA and append 2-D coordinates."""

    transform_type = "reduce"

    def __init__(self, method: str = "pca", embedding_columns: list[str] | None = None, **params: Any):
        super().__init__(**params)
        self._method = method
        self._embedding_columns = embedding_columns

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        from deeplens.embeddings.reduce import DimensionalityReducer

        cols = self._embedding_columns or [c for c in table.columns if c.startswith("emb_")]
        if not cols:
            cols = list(table.select_dtypes(include=[np.number]).columns)
        arr = table[cols].values
        reducer = DimensionalityReducer(method=self._method)
        reduced = reducer.reduce(arr)
        table = table.copy()
        table["x"] = reduced[:, 0]
        table["y"] = reduced[:, 1]
        return table


class SHAPTransform(Transform):
    """Compute SHAP values and append as columns."""

    transform_type = "shap"

    def __init__(self, model: Any = None, feature_columns: list[str] | None = None, **params: Any):
        super().__init__(**params)
        self._model = model
        self._feature_columns = feature_columns

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            return table

        import shap

        cols = self._feature_columns or list(table.select_dtypes(include=[np.number]).columns)
        X = table[cols].values
        # Subsample background data for KernelExplainer to avoid O(n^2) cost
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.Explainer(self._model, background)
        sv = explainer(X)
        table = table.copy()
        for i, col in enumerate(cols):
            vals = sv.values[:, i]
            if vals.ndim > 1:
                vals = vals.mean(axis=-1)
            table[f"shap_{col}"] = vals
        return table
