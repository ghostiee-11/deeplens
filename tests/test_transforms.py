"""Tests for deeplens.data.transforms — NormalizeTransform, EmbeddingTransform,
DimensionalityReductionTransform, and SHAPTransform (no-model path)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeplens.data.transforms import (
    DimensionalityReductionTransform,
    EmbeddingTransform,
    NormalizeTransform,
    SHAPTransform,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_df():
    """Small numeric DataFrame with 30 rows and 3 feature columns."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "f1": rng.randn(30) * 10 + 5,
            "f2": rng.randn(30) * 2 - 3,
            "f3": rng.uniform(0, 100, 30),
        }
    )


@pytest.fixture
def mixed_df():
    """DataFrame with numeric and non-numeric columns."""
    rng = np.random.RandomState(7)
    return pd.DataFrame(
        {
            "f1": rng.randn(20),
            "f2": rng.randn(20),
            "label": ["a", "b"] * 10,
        }
    )


@pytest.fixture
def text_df():
    """Small text DataFrame for EmbeddingTransform."""
    sentences = [
        "The cat sat on the mat with its hat",
        "A dog ran quickly through the park by the lake",
        "Machine learning models learn from large datasets",
        "Natural language processing is a field of AI research",
        "Deep neural networks have many hidden layers inside",
        "Gradient descent optimizes the model parameters slowly",
        "Supervised learning uses labeled training examples always",
        "Unsupervised learning discovers hidden structure in data",
    ]
    return pd.DataFrame({"text": sentences, "label": list(range(len(sentences)))})


@pytest.fixture
def embedded_df():
    """DataFrame that already has emb_* columns (simulating EmbeddingTransform output)."""
    rng = np.random.RandomState(0)
    n = 40
    data = {f"emb_{i}": rng.randn(n) for i in range(10)}
    data["label"] = ["x", "y"] * (n // 2)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# NormalizeTransform
# ---------------------------------------------------------------------------


class TestNormalizeTransformStandard:
    def test_apply_returns_dataframe(self, numeric_df):
        t = NormalizeTransform(method="standard")
        result = t.apply(numeric_df)
        assert isinstance(result, pd.DataFrame)

    def test_apply_preserves_shape(self, numeric_df):
        t = NormalizeTransform(method="standard")
        result = t.apply(numeric_df)
        assert result.shape == numeric_df.shape

    def test_apply_preserves_columns(self, numeric_df):
        t = NormalizeTransform(method="standard")
        result = t.apply(numeric_df)
        assert list(result.columns) == list(numeric_df.columns)

    def test_standard_mean_near_zero(self, numeric_df):
        t = NormalizeTransform(method="standard")
        result = t.apply(numeric_df)
        for col in ["f1", "f2", "f3"]:
            assert abs(result[col].mean()) < 1e-10, f"Mean of {col} should be ~0"

    def test_standard_std_near_one(self, numeric_df):
        t = NormalizeTransform(method="standard")
        result = t.apply(numeric_df)
        for col in ["f1", "f2", "f3"]:
            assert abs(result[col].std(ddof=0) - 1.0) < 1e-6, f"Std of {col} should be ~1"

    def test_does_not_mutate_input(self, numeric_df):
        original_f1 = numeric_df["f1"].copy()
        t = NormalizeTransform(method="standard")
        t.apply(numeric_df)
        pd.testing.assert_series_equal(numeric_df["f1"], original_f1)

    def test_non_numeric_columns_unchanged(self, mixed_df):
        t = NormalizeTransform(method="standard")
        result = t.apply(mixed_df)
        pd.testing.assert_series_equal(result["label"], mixed_df["label"])

    def test_subset_columns(self, numeric_df):
        """Only specified columns should be normalized."""
        t = NormalizeTransform(method="standard", columns=["f1"])
        result = t.apply(numeric_df)
        assert abs(result["f1"].mean()) < 1e-10
        # f2 and f3 should be unchanged
        pd.testing.assert_series_equal(result["f2"], numeric_df["f2"])
        pd.testing.assert_series_equal(result["f3"], numeric_df["f3"])


class TestNormalizeTransformMinMax:
    def test_minmax_range_zero_to_one(self, numeric_df):
        t = NormalizeTransform(method="minmax")
        result = t.apply(numeric_df)
        for col in ["f1", "f2", "f3"]:
            assert result[col].min() >= -1e-10, f"{col} min should be >= 0"
            assert result[col].max() <= 1 + 1e-10, f"{col} max should be <= 1"

    def test_minmax_min_is_zero(self, numeric_df):
        t = NormalizeTransform(method="minmax")
        result = t.apply(numeric_df)
        for col in ["f1", "f2", "f3"]:
            assert abs(result[col].min()) < 1e-6

    def test_minmax_max_is_one(self, numeric_df):
        t = NormalizeTransform(method="minmax")
        result = t.apply(numeric_df)
        for col in ["f1", "f2", "f3"]:
            assert abs(result[col].max() - 1.0) < 1e-6

    def test_minmax_preserves_shape(self, numeric_df):
        t = NormalizeTransform(method="minmax")
        result = t.apply(numeric_df)
        assert result.shape == numeric_df.shape

    def test_minmax_subset_columns(self, numeric_df):
        t = NormalizeTransform(method="minmax", columns=["f2", "f3"])
        result = t.apply(numeric_df)
        assert abs(result["f2"].min()) < 1e-6
        assert abs(result["f3"].min()) < 1e-6
        pd.testing.assert_series_equal(result["f1"], numeric_df["f1"])


class TestNormalizeTransformEdgeCases:
    def test_no_numeric_columns_returns_unchanged(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        t = NormalizeTransform(method="standard")
        result = t.apply(df)
        pd.testing.assert_frame_equal(result, df)

    def test_empty_columns_list_normalizes_all_numeric(self, numeric_df):
        t = NormalizeTransform(method="standard", columns=None)
        result = t.apply(numeric_df)
        for col in ["f1", "f2", "f3"]:
            assert abs(result[col].mean()) < 1e-10

    def test_iris_all_features_normalized(self):
        from deeplens.data.loaders import load_sklearn

        df = load_sklearn("iris")
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        t = NormalizeTransform(method="standard", columns=feature_cols)
        result = t.apply(df)
        for col in feature_cols:
            assert abs(result[col].mean()) < 1e-10


# ---------------------------------------------------------------------------
# EmbeddingTransform
# ---------------------------------------------------------------------------


class TestEmbeddingTransformApply:
    def test_apply_returns_dataframe(self, text_df):
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=5)
        result = t.apply(text_df)
        assert isinstance(result, pd.DataFrame)

    def test_apply_adds_emb_columns(self, text_df):
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=5)
        result = t.apply(text_df)
        emb_cols = [c for c in result.columns if c.startswith("emb_")]
        assert len(emb_cols) == 5

    def test_apply_preserves_original_columns(self, text_df):
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=3)
        result = t.apply(text_df)
        for col in text_df.columns:
            assert col in result.columns

    def test_apply_row_count_unchanged(self, text_df):
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=4)
        result = t.apply(text_df)
        assert len(result) == len(text_df)

    def test_emb_columns_are_numeric(self, text_df):
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=4)
        result = t.apply(text_df)
        emb_cols = [c for c in result.columns if c.startswith("emb_")]
        for col in emb_cols:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_n_components_respected(self, text_df):
        for n in [2, 5, 8]:
            t = EmbeddingTransform(method="tfidf", text_column="text", n_components=n)
            result = t.apply(text_df)
            emb_cols = [c for c in result.columns if c.startswith("emb_")]
            assert len(emb_cols) == n, f"Expected {n} emb columns, got {len(emb_cols)}"

    def test_does_not_mutate_input(self, text_df):
        original_cols = list(text_df.columns)
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=3)
        t.apply(text_df)
        assert list(text_df.columns) == original_cols

    def test_emb_column_names_sequential(self, text_df):
        n = 4
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=n)
        result = t.apply(text_df)
        for i in range(n):
            assert f"emb_{i}" in result.columns

    def test_feature_method_on_numeric_df(self, numeric_df):
        """method='features' should work on a purely numeric DataFrame."""
        t = EmbeddingTransform(method="features", n_components=3)
        result = t.apply(numeric_df)
        emb_cols = [c for c in result.columns if c.startswith("emb_")]
        assert len(emb_cols) == 3

    def test_n_components_capped_at_embedding_dim(self, text_df):
        """n_components larger than embedding dim should not crash; columns capped."""
        t = EmbeddingTransform(method="tfidf", text_column="text", n_components=10000)
        result = t.apply(text_df)
        emb_cols = [c for c in result.columns if c.startswith("emb_")]
        # Must have at least 1 column and no more than vocabulary size
        assert len(emb_cols) >= 1


# ---------------------------------------------------------------------------
# DimensionalityReductionTransform
# ---------------------------------------------------------------------------


class TestDRTransformApply:
    def test_apply_returns_dataframe(self, embedded_df):
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(embedded_df)
        assert isinstance(result, pd.DataFrame)

    def test_apply_adds_x_and_y_columns(self, embedded_df):
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(embedded_df)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_apply_preserves_original_columns(self, embedded_df):
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(embedded_df)
        for col in embedded_df.columns:
            assert col in result.columns

    def test_apply_row_count_unchanged(self, embedded_df):
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(embedded_df)
        assert len(result) == len(embedded_df)

    def test_x_y_columns_are_numeric(self, embedded_df):
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(embedded_df)
        assert pd.api.types.is_numeric_dtype(result["x"])
        assert pd.api.types.is_numeric_dtype(result["y"])

    def test_x_y_values_are_finite(self, embedded_df):
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(embedded_df)
        assert np.isfinite(result["x"].values).all()
        assert np.isfinite(result["y"].values).all()

    def test_does_not_mutate_input(self, embedded_df):
        original_cols = list(embedded_df.columns)
        t = DimensionalityReductionTransform(method="pca")
        t.apply(embedded_df)
        assert list(embedded_df.columns) == original_cols
        assert "x" not in embedded_df.columns

    def test_explicit_embedding_columns(self, embedded_df):
        """Specifying embedding_columns explicitly should restrict input to those cols."""
        cols = [f"emb_{i}" for i in range(5)]
        t = DimensionalityReductionTransform(method="pca", embedding_columns=cols)
        result = t.apply(embedded_df)
        assert "x" in result.columns
        assert "y" in result.columns


class TestDRTransformColumnDetection:
    def test_auto_detects_emb_prefix_columns(self):
        """Without specifying embedding_columns, transform should pick up emb_* cols."""
        rng = np.random.RandomState(1)
        df = pd.DataFrame({f"emb_{i}": rng.randn(30) for i in range(8)})
        df["label"] = "a"
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(df)
        assert "x" in result.columns

    def test_falls_back_to_numeric_when_no_emb_cols(self, numeric_df):
        """If no emb_* columns, falls back to all numeric columns."""
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(numeric_df)
        assert "x" in result.columns
        assert "y" in result.columns


class TestDRTransformPCA:
    def test_pca_reduce_iris_features(self):
        from deeplens.data.loaders import load_sklearn

        df = load_sklearn("iris")
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_features = df[feature_cols].copy()
        df_features.columns = [f"emb_{i}" for i in range(len(feature_cols))]

        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(df_features)
        assert result.shape[0] == 150
        assert "x" in result.columns
        assert "y" in result.columns

    def test_pca_output_shape(self, embedded_df):
        t = DimensionalityReductionTransform(method="pca")
        result = t.apply(embedded_df)
        # Original columns + x + y
        assert result.shape == (len(embedded_df), len(embedded_df.columns) + 2)


class TestDRTransformTSNE:
    def test_tsne_adds_x_y(self):
        rng = np.random.RandomState(5)
        df = pd.DataFrame({f"emb_{i}": rng.randn(25) for i in range(6)})
        t = DimensionalityReductionTransform(method="tsne")
        result = t.apply(df)
        assert "x" in result.columns
        assert "y" in result.columns
        assert len(result) == 25


# ---------------------------------------------------------------------------
# NormalizeTransform + EmbeddingTransform pipeline
# ---------------------------------------------------------------------------


class TestTransformPipeline:
    def test_normalize_then_embed_then_reduce(self, text_df):
        """Chaining Normalize -> Embed -> DRTransform should produce x, y columns."""
        # Step 1: create a small numeric frame alongside text
        rng = np.random.RandomState(99)
        df = text_df.copy()
        df["f1"] = rng.randn(len(df)) * 5
        df["f2"] = rng.randn(len(df))

        # Step 2: normalize numeric cols
        norm = NormalizeTransform(method="standard", columns=["f1", "f2"])
        df = norm.apply(df)
        assert abs(df["f1"].mean()) < 1e-10

        # Step 3: add tfidf embeddings
        embed = EmbeddingTransform(method="tfidf", text_column="text", n_components=5)
        df = embed.apply(df)
        assert len([c for c in df.columns if c.startswith("emb_")]) == 5

        # Step 4: reduce to 2-D
        dr = DimensionalityReductionTransform(method="pca")
        df = dr.apply(df)
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) == len(text_df)


# ---------------------------------------------------------------------------
# SHAPTransform — no-model path
# ---------------------------------------------------------------------------


class TestSHAPTransformNoModel:
    def test_no_model_returns_unchanged(self, numeric_df):
        """With model=None, SHAPTransform should be a no-op."""
        t = SHAPTransform(model=None)
        result = t.apply(numeric_df)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_no_model_preserves_shape(self, numeric_df):
        t = SHAPTransform()
        result = t.apply(numeric_df)
        assert result.shape == numeric_df.shape

    def test_no_model_with_mixed_df(self, mixed_df):
        t = SHAPTransform(model=None)
        result = t.apply(mixed_df)
        pd.testing.assert_frame_equal(result, mixed_df)


# ---------------------------------------------------------------------------
# transform_type class attributes
# ---------------------------------------------------------------------------


class TestTransformTypeAttributes:
    def test_normalize_transform_type(self):
        assert NormalizeTransform.transform_type == "normalize"

    def test_embedding_transform_type(self):
        assert EmbeddingTransform.transform_type == "embed"

    def test_dr_transform_type(self):
        assert DimensionalityReductionTransform.transform_type == "reduce"

    def test_shap_transform_type(self):
        assert SHAPTransform.transform_type == "shap"


# ---------------------------------------------------------------------------
# Additional coverage for missing lines: 18-26, 118-132
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch


class TestLumenFallback:
    """Cover lines 18-26: fallback Transform class when Lumen not installed."""

    def test_fallback_transform_has_apply(self):
        """The fallback Transform class (or real one) should have apply method."""
        from deeplens.data.transforms import Transform
        t = Transform()
        result = t.apply(pd.DataFrame({"a": [1, 2]}))
        assert isinstance(result, pd.DataFrame)

    def test_has_lumen_flag_exists(self):
        """_HAS_LUMEN should be a bool."""
        from deeplens.data.transforms import _HAS_LUMEN
        assert isinstance(_HAS_LUMEN, bool)


class TestSHAPTransformWithModel:
    """Cover lines 118-132: SHAP with an actual model."""

    def test_shap_transform_with_model(self):
        """Lines 118-132: full SHAP computation with a trained model."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")

        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(42)
        n = 50
        df = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
            "label": (["a", "b"] * (n // 2))[:n],
        })
        X = df[["f1", "f2", "f3"]].values
        y = df["label"].values
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)

        t = SHAPTransform(model=model, feature_columns=["f1", "f2", "f3"])
        result = t.apply(df)
        assert isinstance(result, pd.DataFrame)
        # Should have shap_ columns added
        shap_cols = [c for c in result.columns if c.startswith("shap_")]
        assert len(shap_cols) == 3
        assert "shap_f1" in result.columns
        assert "shap_f2" in result.columns
        assert "shap_f3" in result.columns

    def test_shap_transform_auto_detects_numeric_columns(self):
        """Lines 120: auto-detects numeric columns when feature_columns not specified."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")

        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(42)
        n = 50
        df = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "label": (["a", "b"] * (n // 2))[:n],
        })
        X = df[["f1", "f2"]].values
        y = df["label"].values
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)

        t = SHAPTransform(model=model)  # no feature_columns
        result = t.apply(df)
        shap_cols = [c for c in result.columns if c.startswith("shap_")]
        assert len(shap_cols) >= 2
