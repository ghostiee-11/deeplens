"""Custom Lumen Sources for DeepLens.

These integrate DeepLens' data loaders with Lumen's declarative pipeline system.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from lumen.sources.base import Source

    _HAS_LUMEN = True
except ImportError:
    # Provide a stub so the module can still be imported without Lumen
    import param

    class Source(param.Parameterized):  # type: ignore[no-redef]
        source_type = None
        def get_tables(self): return []
        def get_schema(self, table=None): return {}
        def get(self, table, **query): return pd.DataFrame()

    _HAS_LUMEN = False

from deeplens.data.loaders import load_sklearn


class SklearnSource(Source):
    """Lumen ``Source`` backed by scikit-learn toy datasets.

    Usage in a Lumen YAML spec::

        sources:
          my_data:
            type: sklearn
            datasets:
              - iris
              - wine
    """

    source_type = "sklearn"

    def __init__(self, datasets: list[str] | None = None, **params: Any):
        super().__init__(**params)
        self._datasets = datasets or ["iris"]
        self._cache: dict[str, pd.DataFrame] = {}

    def get_tables(self) -> list[str]:
        return list(self._datasets)

    def get_schema(self, table: str | None = None) -> dict:
        tables = [table] if table else self._datasets
        schema = {}
        for t in tables:
            df = self._load(t)
            schema[t] = {col: str(df[col].dtype) for col in df.columns}
        return schema

    def get(self, table: str, **query: Any) -> pd.DataFrame:
        df = self._load(table)
        for col, val in query.items():
            if col in df.columns:
                if isinstance(val, (list, tuple)):
                    df = df[df[col].isin(val)]
                else:
                    df = df[df[col] == val]
        return df

    def _load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = load_sklearn(name)
        return self._cache[name]


class EmbeddingSource(Source):
    """Wraps another DataFrame and exposes its embeddings as a table."""

    source_type = "embedding"

    def __init__(self, df: pd.DataFrame | None = None, embeddings: Any | None = None, **params: Any):
        super().__init__(**params)
        self._df = df if df is not None else pd.DataFrame()
        self._embeddings = embeddings

    def get_tables(self) -> list[str]:
        tables = ["data"]
        if self._embeddings is not None:
            tables.append("embeddings")
        return tables

    def get_schema(self, table: str | None = None) -> dict:
        if table == "embeddings" and self._embeddings is not None:
            n_dim = self._embeddings.shape[1] if self._embeddings.ndim > 1 else 1
            return {"embeddings": {f"dim_{i}": "float64" for i in range(n_dim)}}
        return {"data": {col: str(self._df[col].dtype) for col in self._df.columns}}

    def get(self, table: str, **query: Any) -> pd.DataFrame:
        if table == "embeddings" and self._embeddings is not None:
            import numpy as np

            arr = self._embeddings
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return pd.DataFrame(arr, columns=[f"dim_{i}" for i in range(arr.shape[1])])
        return self._df
