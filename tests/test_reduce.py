"""Tests for deeplens.embeddings.reduce."""

from __future__ import annotations

import numpy as np
import pytest

from deeplens.embeddings.reduce import DimensionalityReducer


@pytest.fixture
def high_dim_data():
    """50 samples x 20 features, reproducible."""
    rng = np.random.RandomState(42)
    return rng.randn(50, 20).astype(np.float32)


class TestPCAReduce:
    def test_pca_reduce(self, high_dim_data):
        dr = DimensionalityReducer(method="pca")
        result = dr.reduce(high_dim_data)
        assert result.shape == (50, 2)
        assert result.dtype == np.float32

    def test_pca_deterministic(self, high_dim_data):
        dr = DimensionalityReducer(method="pca", random_state=42)
        r1 = dr.reduce(high_dim_data)
        r2 = dr.reduce(high_dim_data)
        np.testing.assert_array_equal(r1, r2)


class TestTSNEReduce:
    def test_tsne_reduce(self, high_dim_data):
        dr = DimensionalityReducer(method="tsne", perplexity=10)
        result = dr.reduce(high_dim_data)
        assert result.shape == (50, 2)
        assert result.dtype == np.float32

    def test_tsne_perplexity_clamped(self):
        """Perplexity > n_samples-1 should be clamped internally."""
        small = np.random.RandomState(0).randn(15, 10).astype(np.float32)
        dr = DimensionalityReducer(method="tsne", perplexity=50)
        result = dr.reduce(small)
        assert result.shape == (15, 2)


class TestReduceShape:
    def test_reduce_shape_2d(self, high_dim_data):
        dr = DimensionalityReducer(method="pca", n_components=2)
        result = dr.reduce(high_dim_data)
        assert result.shape[1] == 2

    def test_reduce_rejects_1d(self):
        dr = DimensionalityReducer(method="pca")
        with pytest.raises(ValueError, match="Expected 2-D array"):
            dr.reduce(np.array([1.0, 2.0, 3.0]))

    def test_reduce_low_dim_passthrough(self):
        """If input already has <= n_components cols, return as-is."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dr = DimensionalityReducer(method="pca", n_components=2)
        result = dr.reduce(data)
        np.testing.assert_array_equal(result, data)


class TestQualityMetrics:
    def test_quality_metrics(self, high_dim_data):
        dr = DimensionalityReducer(method="pca")
        reduced = dr.reduce(high_dim_data)
        metrics = dr.quality_metrics(high_dim_data, reduced)
        assert "trustworthiness" in metrics
        assert "stress" in metrics
        assert 0.0 <= metrics["trustworthiness"] <= 1.0
        assert metrics["stress"] >= 0.0
        assert metrics["n_samples"] == 50

    def test_quality_metrics_k(self, high_dim_data):
        dr = DimensionalityReducer(method="pca")
        reduced = dr.reduce(high_dim_data)
        metrics = dr.quality_metrics(high_dim_data, reduced, k=5)
        assert metrics["k"] == 5


class TestReduceWithQuality:
    def test_reduce_with_quality(self, high_dim_data):
        dr = DimensionalityReducer(method="pca")
        reduced, metrics = dr.reduce_with_quality(high_dim_data)
        assert reduced.shape == (50, 2)
        assert isinstance(metrics, dict)
        assert "trustworthiness" in metrics
