"""Shared fixtures for DeepLens test suite."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pandas as pd
import pytest

from deeplens.config import DeepLensState
from deeplens.analyst.llm import LLMProvider


@pytest.fixture
def iris_state():
    """Load the iris dataset into a fully-populated DeepLensState."""
    from deeplens.data.loaders import load_sklearn, infer_columns

    df = load_sklearn("iris")
    cols = infer_columns(df)

    state = DeepLensState()
    state.dataset_name = "iris"
    state.df = df
    state.feature_columns = cols["features"]
    state.label_column = cols.get("label", "label")
    state.labels = np.array(df[state.label_column].tolist())
    state.class_names = sorted(df[state.label_column].unique().tolist())
    return state


@pytest.fixture
def iris_with_embeddings(iris_state):
    """Add raw and 2-D embeddings to an iris state."""
    from deeplens.embeddings.compute import EmbeddingComputer
    from deeplens.embeddings.reduce import DimensionalityReducer

    ec = EmbeddingComputer(method="features")
    raw = ec.compute(iris_state.df)
    iris_state.embeddings_raw = raw

    dr = DimensionalityReducer(method="pca")
    iris_state.embeddings_2d = dr.reduce(raw)
    return iris_state


@pytest.fixture
def iris_with_model(iris_with_embeddings):
    """Train a LogisticRegression on the iris state."""
    from sklearn.linear_model import LogisticRegression

    state = iris_with_embeddings
    X = state.df[state.feature_columns].values
    y = state.labels

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    state.trained_model = model
    state.model_name = "LogisticRegression"
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    return state


@pytest.fixture
def sample_df():
    """Small 20-row DataFrame for quick tests."""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "f1": rng.randn(20),
        "f2": rng.randn(20),
        "f3": rng.randn(20),
        "label": ["a", "b"] * 10,
    })


class MockLLMProvider(LLMProvider):
    """LLM provider that returns a fixed response without any API calls."""

    _fixed_response: str = "This is a mock LLM response."

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        yield self._fixed_response


@pytest.fixture
def mock_llm():
    """Return a MockLLMProvider instance."""
    return MockLLMProvider()
