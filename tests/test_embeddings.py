"""Tests for deeplens.embeddings.compute."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeplens.embeddings.compute import EmbeddingComputer


class TestTfidfEmbeddings:
    def test_tfidf_embeddings(self):
        df = pd.DataFrame({
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "A fast red fox leaps across a sleeping hound",
                "Machine learning is a branch of artificial intelligence",
                "Deep learning models use neural networks for predictions",
            ]
        })
        ec = EmbeddingComputer(method="tfidf")
        emb = ec.compute(df, text_col="text")
        assert emb.shape[0] == 4
        assert emb.ndim == 2
        assert emb.dtype == np.float32

    def test_tfidf_max_features(self):
        df = pd.DataFrame({"text": [f"word{i} sentence text" for i in range(50)]})
        ec = EmbeddingComputer(method="tfidf", max_features=200)
        emb = ec.compute(df, text_col="text")
        assert emb.shape[1] <= 200

    def test_tfidf_falls_back_to_features_if_no_text(self):
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        ec = EmbeddingComputer(method="tfidf")
        emb = ec.compute(df)
        # Falls back to _features since no text column found
        assert emb.shape == (3, 2)


class TestFeatureEmbeddings:
    def test_feature_embeddings(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "f3": [7.0, 8.0, 9.0],
            "label": ["a", "b", "a"],
        })
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        assert emb.shape == (3, 3)  # Only numeric cols
        assert emb.dtype == np.float32

    def test_feature_embeddings_excludes_strings(self):
        df = pd.DataFrame({
            "num": [1.0, 2.0],
            "cat": ["a", "b"],
        })
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        assert emb.shape == (2, 1)

    def test_feature_embeddings_no_numeric_raises(self):
        df = pd.DataFrame({"cat": ["a", "b"], "txt": ["hello", "world"]})
        ec = EmbeddingComputer(method="features")
        with pytest.raises(ValueError, match="No numeric columns"):
            ec.compute(df)


class TestEmbeddingShape:
    def test_embedding_shape_iris(self, iris_state):
        ec = EmbeddingComputer(method="features")
        # Pass only feature columns to get expected shape
        emb = ec.compute(iris_state.df[iris_state.feature_columns])
        assert emb.shape[0] == 150
        assert emb.shape[1] == len(iris_state.feature_columns)

    def test_embedding_shape_matches_df_rows(self, sample_df):
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(sample_df)
        assert emb.shape[0] == len(sample_df)


class TestEmbeddingNanHandling:
    def test_embedding_nan_handling(self):
        df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [np.nan, 2.0, np.nan, 4.0],
        })
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        assert not np.isnan(emb).any(), "NaN values should be imputed"
        assert emb.shape == (4, 2)

    def test_embedding_all_nan_column(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [np.nan, np.nan, np.nan],
        })
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        # All-NaN column should be filled with 0
        assert not np.isnan(emb).any()
        assert emb[:, 1].tolist() == [0.0, 0.0, 0.0]


class TestComputeDispatch:
    """Tests for compute() method dispatching to each backend."""

    def test_compute_tfidf_with_explicit_text_col(self):
        df = pd.DataFrame({
            "body": [
                "Natural language processing is a subfield of linguistics and computer science",
                "Machine learning algorithms learn from training data to make predictions",
                "Deep neural networks are powerful models for image recognition tasks",
            ]
        })
        ec = EmbeddingComputer(method="tfidf")
        emb = ec.compute(df, text_col="body")
        assert emb.shape[0] == 3
        assert emb.ndim == 2
        assert emb.dtype == np.float32

    def test_compute_tfidf_auto_detects_text_col(self):
        long_text = "This is a fairly long sentence that exceeds fifty characters for text detection."
        df = pd.DataFrame({"review": [long_text, long_text + " extra words here.", long_text + " more text."]})
        ec = EmbeddingComputer(method="tfidf")
        emb = ec.compute(df)
        assert emb.shape[0] == 3
        assert emb.dtype == np.float32

    def test_compute_features_method(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        assert emb.shape == (3, 2)
        assert emb.dtype == np.float32

    def test_compute_returns_float32(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0], "y": [1.0, 2.0, 3.0]})
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        assert emb.dtype == np.float32

    def test_default_method_is_tfidf(self):
        ec = EmbeddingComputer()
        assert ec.method == "tfidf"

    def test_method_selector_rejects_invalid(self):
        with pytest.raises(Exception):
            EmbeddingComputer(method="invalid_method")


class TestTfidfWithSparseMatrix:
    """Verify TF-IDF sparse output is properly densified."""

    def test_tfidf_output_is_dense_array(self):
        df = pd.DataFrame({
            "text": [
                "sparse matrix representation for text features in machine learning",
                "dense arrays are used after converting from sparse tfidf output",
                "scikit-learn tfidf vectorizer returns a sparse csr matrix object",
                "converting sparse to dense uses the toarray method in sklearn",
            ]
        })
        ec = EmbeddingComputer(method="tfidf", max_features=500)
        emb = ec.compute(df, text_col="text")
        # Result must be a plain numpy ndarray, not a sparse matrix
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 4

    def test_tfidf_sparse_values_are_non_negative(self):
        df = pd.DataFrame({
            "text": [
                "TF-IDF values are always non-negative because they represent term frequency weights",
                "inverse document frequency is also a non-negative quantity in all variants",
            ]
        })
        ec = EmbeddingComputer(method="tfidf")
        emb = ec.compute(df, text_col="text")
        assert (emb >= 0).all()


class TestNanHandlingExtended:
    """Additional NaN handling scenarios for _features backend."""

    def test_nan_in_single_row(self):
        df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0],
            "f2": [2.0, 3.0, 4.0],
        })
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        assert not np.isnan(emb).any()
        # NaN in row 1 of f1 should be replaced with column mean of f1 = (1+3)/2 = 2.0
        assert emb[1, 0] == pytest.approx(2.0)

    def test_nan_entire_row(self):
        df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0],
            "f2": [2.0, np.nan, 4.0],
        })
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        assert not np.isnan(emb).any()
        # Row with all NaN should be filled with column means
        assert emb[1, 0] == pytest.approx(2.0)
        assert emb[1, 1] == pytest.approx(3.0)

    def test_no_nan_passthrough(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        })
        ec = EmbeddingComputer(method="features")
        emb = ec.compute(df)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(emb, expected)


class TestMethodSelection:
    """Tests for switching between methods and default values."""

    def test_change_method_to_features(self):
        ec = EmbeddingComputer()
        ec.method = "features"
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        emb = ec.compute(df)
        assert emb.shape == (2, 2)

    def test_batch_size_param(self):
        ec = EmbeddingComputer(batch_size=16)
        assert ec.batch_size == 16

    def test_max_features_param(self):
        ec = EmbeddingComputer(max_features=1000)
        assert ec.max_features == 1000

    def test_tfidf_with_nan_text_values(self):
        # NaN values in text column cause sklearn TfidfVectorizer to raise
        df = pd.DataFrame({
            "text": [
                "The quick brown fox jumps over the lazy dog near the river",
                np.nan,
                "Machine learning and deep learning are subfields of artificial intelligence",
            ]
        })
        ec = EmbeddingComputer(method="tfidf")
        with pytest.raises(ValueError):
            ec.compute(df, text_col="text")


# ---------------------------------------------------------------------------
# Additional coverage for missing lines: 44, 46, 49, 69-83, 87-119
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch


class TestSentenceTransformersBackend:
    """Cover _sentence_transformers: lines 69-83."""

    def test_sentence_transformers_import_error(self):
        """Line 69-74: ImportError when sentence_transformers is not installed."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            ec = EmbeddingComputer(method="sentence-transformers")
            df = pd.DataFrame({
                "text": ["A long sentence that is definitely more than fifty characters for text detection."] * 5,
            })
            with pytest.raises(ImportError, match="sentence-transformers"):
                ec.compute(df, text_col="text")

    def test_sentence_transformers_no_text_column_raises(self):
        """Line 77-78: no text column found raises ValueError."""
        mock_st = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            ec = EmbeddingComputer(method="sentence-transformers")
            df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
            with pytest.raises(ValueError, match="No text column"):
                ec.compute(df)

    def test_sentence_transformers_success(self):
        """Lines 80-83: successful sentence-transformer encoding."""
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            ec = EmbeddingComputer(method="sentence-transformers", model_name="test-model", batch_size=16)
            df = pd.DataFrame({"text": ["sentence one is longer than fifty characters for detection",
                                        "sentence two is also long enough for the text detection heuristic",
                                        "sentence three is the last one in this test dataframe for coverage"]})
            emb = ec.compute(df, text_col="text")
            assert emb.shape == (3, 384)
            assert emb.dtype == np.float32
            mock_st_module.SentenceTransformer.assert_called_once_with("test-model")
            mock_model.encode.assert_called_once()


class TestClipBackend:
    """Cover _clip: lines 87-119."""

    def test_clip_import_error(self):
        """Lines 87-91: ImportError when open_clip is not installed."""
        with patch.dict("sys.modules", {"open_clip": None, "torch": None}):
            ec = EmbeddingComputer(method="clip")
            df = pd.DataFrame({"text": ["hello world"]})
            with pytest.raises(ImportError, match="open-clip-torch"):
                ec.compute(df, text_col="text")

    def test_clip_text_encoding(self):
        """Lines 107-115: CLIP text encoding path."""
        mock_open_clip = MagicMock()
        mock_torch = MagicMock()

        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer = MagicMock()
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer

        # encode_text returns a tensor-like object with .numpy()
        mock_encoded = MagicMock()
        mock_encoded.numpy.return_value = np.random.randn(2, 512).astype(np.float32)
        mock_model.encode_text.return_value = mock_encoded

        with patch.dict("sys.modules", {"open_clip": mock_open_clip, "torch": mock_torch}):
            ec = EmbeddingComputer(method="clip", batch_size=2)
            df = pd.DataFrame({"text": ["hello world", "foo bar"]})
            emb = ec.compute(df, text_col="text")
            assert emb.shape[0] == 2
            assert emb.dtype == np.float32

    def test_clip_no_text_or_image_raises(self):
        """Lines 116-117: no text_col or image_col raises ValueError."""
        mock_open_clip = MagicMock()
        mock_torch = MagicMock()

        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_open_clip.get_tokenizer.return_value = MagicMock()

        with patch.dict("sys.modules", {"open_clip": mock_open_clip, "torch": mock_torch}):
            ec = EmbeddingComputer(method="clip")
            df = pd.DataFrame({"f1": [1.0, 2.0]})
            with pytest.raises(ValueError, match="CLIP requires"):
                ec.compute(df)


class TestComputeDispatchExtended:
    """Cover compute() dispatch lines 44, 46, 49."""

    def test_compute_dispatches_to_sentence_transformers(self):
        """Line 44: method='sentence-transformers' dispatches correctly."""
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 128).astype(np.float32)
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            ec = EmbeddingComputer(method="sentence-transformers")
            df = pd.DataFrame({"text": ["A sentence that is quite long for proper detection above fifty chars",
                                        "Another lengthy sentence for text column auto detection purposes"]})
            emb = ec.compute(df, text_col="text")
            assert emb.shape == (2, 128)

    def test_compute_dispatches_to_clip(self):
        """Line 46: method='clip' dispatches correctly."""
        mock_open_clip = MagicMock()
        mock_torch = MagicMock()
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_open_clip.get_tokenizer.return_value = MagicMock()

        with patch.dict("sys.modules", {"open_clip": mock_open_clip, "torch": mock_torch}):
            ec = EmbeddingComputer(method="clip")
            df = pd.DataFrame({"f1": [1.0]})
            with pytest.raises(ValueError, match="CLIP requires"):
                ec.compute(df)

    def test_compute_dispatches_to_features(self):
        """Line 49: method='features' dispatches correctly (already tested but explicit)."""
        ec = EmbeddingComputer(method="features")
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        emb = ec.compute(df)
        assert emb.shape == (2, 2)
