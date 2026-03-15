"""Tests for deeplens.data.sources — SklearnSource and EmbeddingSource."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeplens.data.sources import EmbeddingSource, SklearnSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iris_source():
    """SklearnSource backed by the iris dataset."""
    return SklearnSource(datasets=["iris"])


@pytest.fixture
def multi_source():
    """SklearnSource backed by iris + wine."""
    return SklearnSource(datasets=["iris", "wine"])


@pytest.fixture
def iris_df():
    """Raw iris DataFrame for use in EmbeddingSource tests."""
    from deeplens.data.loaders import load_sklearn

    return load_sklearn("iris")


# ---------------------------------------------------------------------------
# SklearnSource.get_tables()
# ---------------------------------------------------------------------------


class TestSklearnSourceGetTables:
    def test_get_tables_single_dataset(self, iris_source):
        tables = iris_source.get_tables()
        assert tables == ["iris"]

    def test_get_tables_multiple_datasets(self, multi_source):
        tables = multi_source.get_tables()
        assert "iris" in tables
        assert "wine" in tables
        assert len(tables) == 2

    def test_get_tables_default_is_iris(self):
        src = SklearnSource()
        assert src.get_tables() == ["iris"]

    def test_get_tables_returns_list(self, iris_source):
        assert isinstance(iris_source.get_tables(), list)


# ---------------------------------------------------------------------------
# SklearnSource.get()
# ---------------------------------------------------------------------------


class TestSklearnSourceGet:
    def test_get_returns_dataframe(self, iris_source):
        df = iris_source.get("iris")
        assert isinstance(df, pd.DataFrame)

    def test_get_iris_row_count(self, iris_source):
        df = iris_source.get("iris")
        assert len(df) == 150

    def test_get_iris_has_label_column(self, iris_source):
        df = iris_source.get("iris")
        assert "label" in df.columns

    def test_get_iris_has_numeric_features(self, iris_source):
        df = iris_source.get("iris")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assert len(numeric_cols) >= 4

    def test_get_wine_returns_dataframe(self):
        src = SklearnSource(datasets=["wine"])
        df = src.get("wine")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100

    def test_get_breast_cancer_returns_dataframe(self):
        src = SklearnSource(datasets=["breast_cancer"])
        df = src.get("breast_cancer")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 500

    def test_get_digits_returns_dataframe(self):
        src = SklearnSource(datasets=["digits"])
        df = src.get("digits")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1000

    def test_get_diabetes_returns_dataframe(self):
        src = SklearnSource(datasets=["diabetes"])
        df = src.get("diabetes")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 400

    # Query filtering ----------------------------------------------------------

    def test_get_with_scalar_filter(self, iris_source):
        df_all = iris_source.get("iris")
        first_label = df_all["label"].iloc[0]
        df_filtered = iris_source.get("iris", label=first_label)
        assert len(df_filtered) < len(df_all)
        assert (df_filtered["label"] == first_label).all()

    def test_get_with_list_filter(self, iris_source):
        df = iris_source.get("iris", label=["setosa", "versicolor"])
        assert set(df["label"].unique()).issubset({"setosa", "versicolor"})

    def test_get_with_nonexistent_column_filter_ignored(self, iris_source):
        """Filtering on a column not in the DataFrame should be silently ignored."""
        df = iris_source.get("iris", nonexistent_col="value")
        assert len(df) == 150

    def test_get_does_not_mutate_cache(self, iris_source):
        """Repeated get() calls should return the same row count."""
        df1 = iris_source.get("iris")
        _ = iris_source.get("iris", label="setosa")
        df2 = iris_source.get("iris")
        assert len(df1) == len(df2)


# ---------------------------------------------------------------------------
# SklearnSource.get_schema()
# ---------------------------------------------------------------------------


class TestSklearnSourceGetSchema:
    def test_get_schema_with_table_returns_dict(self, iris_source):
        schema = iris_source.get_schema("iris")
        assert isinstance(schema, dict)

    def test_get_schema_contains_iris_key(self, iris_source):
        schema = iris_source.get_schema("iris")
        assert "iris" in schema

    def test_get_schema_column_entries_are_strings(self, iris_source):
        schema = iris_source.get_schema("iris")
        for col_type in schema["iris"].values():
            assert isinstance(col_type, str)

    def test_get_schema_no_table_returns_all_datasets(self, multi_source):
        schema = multi_source.get_schema()
        assert "iris" in schema
        assert "wine" in schema

    def test_get_schema_iris_has_expected_columns(self, iris_source):
        schema = iris_source.get_schema("iris")
        cols = set(schema["iris"].keys())
        # Iris should have at least the four flower measurement features
        assert cols.issuperset(
            {"sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"}
        )

    def test_get_schema_dtype_strings_are_valid(self, iris_source):
        schema = iris_source.get_schema("iris")
        for dtype_str in schema["iris"].values():
            # Each entry should be parseable by numpy/pandas
            assert len(dtype_str) > 0


# ---------------------------------------------------------------------------
# SklearnSource — caching
# ---------------------------------------------------------------------------


class TestSklearnSourceCaching:
    def test_repeated_get_returns_same_object(self, iris_source):
        """Second get() should hit the cache and return the identical DataFrame."""
        df1 = iris_source._load("iris")
        df2 = iris_source._load("iris")
        assert df1 is df2

    def test_cache_populated_after_get(self, iris_source):
        assert "iris" not in iris_source._cache
        iris_source.get("iris")
        assert "iris" in iris_source._cache


# ---------------------------------------------------------------------------
# SklearnSource — invalid dataset name
# ---------------------------------------------------------------------------


class TestSklearnSourceInvalidName:
    def test_invalid_dataset_raises_value_error(self):
        src = SklearnSource(datasets=["bogus_dataset"])
        with pytest.raises(ValueError, match="Unknown sklearn dataset"):
            src.get("bogus_dataset")

    def test_get_schema_invalid_raises_value_error(self):
        src = SklearnSource(datasets=["bogus_dataset"])
        with pytest.raises(ValueError, match="Unknown sklearn dataset"):
            src.get_schema("bogus_dataset")


# ---------------------------------------------------------------------------
# EmbeddingSource.get_tables()
# ---------------------------------------------------------------------------


class TestEmbeddingSourceGetTables:
    def test_get_tables_without_embeddings(self, iris_df):
        src = EmbeddingSource(df=iris_df)
        assert src.get_tables() == ["data"]

    def test_get_tables_with_embeddings(self, iris_df):
        embeddings = np.random.default_rng(0).random((len(iris_df), 8)).astype(np.float32)
        src = EmbeddingSource(df=iris_df, embeddings=embeddings)
        tables = src.get_tables()
        assert "data" in tables
        assert "embeddings" in tables

    def test_get_tables_empty_source(self):
        src = EmbeddingSource()
        assert src.get_tables() == ["data"]


# ---------------------------------------------------------------------------
# EmbeddingSource.get()
# ---------------------------------------------------------------------------


class TestEmbeddingSourceGet:
    def test_get_data_returns_original_dataframe(self, iris_df):
        src = EmbeddingSource(df=iris_df)
        result = src.get("data")
        pd.testing.assert_frame_equal(result, iris_df)

    def test_get_embeddings_returns_dataframe(self, iris_df):
        embeddings = np.random.default_rng(1).random((len(iris_df), 6)).astype(np.float32)
        src = EmbeddingSource(df=iris_df, embeddings=embeddings)
        emb_df = src.get("embeddings")
        assert isinstance(emb_df, pd.DataFrame)

    def test_get_embeddings_shape_matches(self, iris_df):
        n_dim = 10
        embeddings = np.random.default_rng(2).random((len(iris_df), n_dim)).astype(np.float32)
        src = EmbeddingSource(df=iris_df, embeddings=embeddings)
        emb_df = src.get("embeddings")
        assert emb_df.shape == (len(iris_df), n_dim)

    def test_get_embeddings_column_names(self, iris_df):
        n_dim = 4
        embeddings = np.ones((len(iris_df), n_dim), dtype=np.float32)
        src = EmbeddingSource(df=iris_df, embeddings=embeddings)
        emb_df = src.get("embeddings")
        assert list(emb_df.columns) == [f"dim_{i}" for i in range(n_dim)]

    def test_get_embeddings_1d_array_reshaped(self, iris_df):
        """1-D embedding array should be reshaped to (N, 1) automatically."""
        embeddings_1d = np.ones(len(iris_df), dtype=np.float32)
        src = EmbeddingSource(df=iris_df, embeddings=embeddings_1d)
        emb_df = src.get("embeddings")
        assert emb_df.shape == (len(iris_df), 1)
        assert list(emb_df.columns) == ["dim_0"]

    def test_get_data_without_embeddings_set(self, iris_df):
        """Calling get('embeddings') when none set falls back to data DataFrame."""
        src = EmbeddingSource(df=iris_df)
        result = src.get("embeddings")
        pd.testing.assert_frame_equal(result, iris_df)

    def test_get_empty_source(self):
        src = EmbeddingSource()
        result = src.get("data")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# EmbeddingSource.get_schema()
# ---------------------------------------------------------------------------


class TestEmbeddingSourceGetSchema:
    def test_get_schema_data_returns_dict(self, iris_df):
        src = EmbeddingSource(df=iris_df)
        schema = src.get_schema("data")
        assert isinstance(schema, dict)
        assert "data" in schema

    def test_get_schema_data_contains_columns(self, iris_df):
        src = EmbeddingSource(df=iris_df)
        schema = src.get_schema("data")
        for col in iris_df.columns:
            assert col in schema["data"]

    def test_get_schema_embeddings_returns_dim_keys(self, iris_df):
        n_dim = 5
        embeddings = np.ones((len(iris_df), n_dim), dtype=np.float32)
        src = EmbeddingSource(df=iris_df, embeddings=embeddings)
        schema = src.get_schema("embeddings")
        assert "embeddings" in schema
        assert list(schema["embeddings"].keys()) == [f"dim_{i}" for i in range(n_dim)]

    def test_get_schema_embeddings_dtype_is_float64(self, iris_df):
        embeddings = np.ones((len(iris_df), 3), dtype=np.float32)
        src = EmbeddingSource(df=iris_df, embeddings=embeddings)
        schema = src.get_schema("embeddings")
        for dtype_str in schema["embeddings"].values():
            assert dtype_str == "float64"

    def test_get_schema_no_table_returns_data_schema(self, iris_df):
        """get_schema() with no argument should return data schema."""
        src = EmbeddingSource(df=iris_df)
        schema = src.get_schema()
        assert "data" in schema


# ---------------------------------------------------------------------------
# EmbeddingSource wraps SklearnSource
# ---------------------------------------------------------------------------


class TestEmbeddingSourceWrapsSklearnSource:
    def test_wraps_sklearn_iris_with_feature_embeddings(self):
        """End-to-end: load iris via SklearnSource and wrap in EmbeddingSource."""
        from deeplens.embeddings.compute import EmbeddingComputer

        sklearn_src = SklearnSource(datasets=["iris"])
        df = sklearn_src.get("iris")
        ec = EmbeddingComputer(method="features")
        embeddings = ec.compute(df)

        emb_src = EmbeddingSource(df=df, embeddings=embeddings)

        assert "data" in emb_src.get_tables()
        assert "embeddings" in emb_src.get_tables()

        emb_df = emb_src.get("embeddings")
        assert emb_df.shape[0] == len(df)
        assert emb_df.shape[1] > 0

    def test_embedding_values_are_finite(self):
        from deeplens.embeddings.compute import EmbeddingComputer

        sklearn_src = SklearnSource(datasets=["iris"])
        df = sklearn_src.get("iris")
        ec = EmbeddingComputer(method="features")
        embeddings = ec.compute(df)

        emb_src = EmbeddingSource(df=df, embeddings=embeddings)
        emb_df = emb_src.get("embeddings")
        assert np.isfinite(emb_df.values).all()
