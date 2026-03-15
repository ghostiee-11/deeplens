"""Tests for deeplens.export.notebook.NotebookExporter."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from deeplens.config import DeepLensState
from deeplens.export.notebook import NotebookExporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_state():
    """Bare-minimum state: just a name and a small DataFrame."""
    state = DeepLensState()
    state.dataset_name = "iris"
    state.df = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9, 6.3],
            "sepal width (cm)": [3.5, 3.0, 3.3],
            "label": ["setosa", "setosa", "virginica"],
        }
    )
    state.feature_columns = ["sepal length (cm)", "sepal width (cm)"]
    state.label_column = "label"
    state.labels = np.array(["setosa", "setosa", "virginica"])
    state.class_names = ["setosa", "virginica"]
    return state


@pytest.fixture
def state_with_model(minimal_state):
    """State that has a trained LogisticRegression model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    state = minimal_state
    X = state.df[state.feature_columns].values
    le = LabelEncoder()
    y = le.fit_transform(state.labels)

    model = LogisticRegression(max_iter=300, random_state=0)
    model.fit(X, y)

    state.trained_model = model
    state.model_name = "LogisticRegression"
    state.predictions = model.predict(X)
    state.probabilities = model.predict_proba(X)
    return state


@pytest.fixture
def state_with_embeddings(minimal_state):
    """State that has 2-D PCA embeddings."""
    from sklearn.decomposition import PCA

    state = minimal_state
    X = state.df[state.feature_columns].values
    state.embeddings_raw = X
    state.embeddings_2d = PCA(n_components=2).fit_transform(X)
    state.reduction_method = "pca"
    state.embedding_method = "features"
    return state


@pytest.fixture
def state_with_shap(state_with_model):
    """State that has a placeholder SHAP values object."""
    # Use a simple numpy array as a stand-in for shap.Explanation
    state = state_with_model
    # Mimic the shape: (n_samples, n_features)
    state.shap_values = np.zeros((3, 2))
    state.shap_expected = 0.0
    return state


@pytest.fixture
def custom_state():
    """State with a non-sklearn dataset name to exercise the generic branch."""
    state = DeepLensState()
    state.dataset_name = "my_custom_dataset"
    state.df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "label": ["a", "b"]})
    state.feature_columns = ["x", "y"]
    state.label_column = "label"
    return state


# ---------------------------------------------------------------------------
# 1. generate() returns a well-formed nbformat v4 dict
# ---------------------------------------------------------------------------


class TestGenerateStructure:
    def test_returns_dict_with_required_keys(self, minimal_state):
        nb = NotebookExporter(minimal_state).generate()
        assert isinstance(nb, dict)
        for key in ("nbformat", "nbformat_minor", "metadata", "cells"):
            assert key in nb, f"Missing top-level key: {key}"

    def test_nbformat_version(self, minimal_state):
        nb = NotebookExporter(minimal_state).generate()
        assert nb["nbformat"] == 4
        assert nb["nbformat_minor"] == 5

    def test_metadata_has_kernelspec(self, minimal_state):
        nb = NotebookExporter(minimal_state).generate()
        meta = nb["metadata"]
        assert "kernelspec" in meta
        assert meta["kernelspec"]["language"] == "python"

    def test_cells_is_list(self, minimal_state):
        nb = NotebookExporter(minimal_state).generate()
        assert isinstance(nb["cells"], list)

    def test_each_cell_has_required_fields(self, minimal_state):
        nb = NotebookExporter(minimal_state).generate()
        for cell in nb["cells"]:
            assert "cell_type" in cell
            assert "source" in cell
            assert "metadata" in cell
            assert cell["cell_type"] in ("markdown", "code")

    def test_code_cells_have_outputs_field(self, minimal_state):
        nb = NotebookExporter(minimal_state).generate()
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert "outputs" in cell
                assert isinstance(cell["outputs"], list)


# ---------------------------------------------------------------------------
# 2. Minimum cell count and mandatory cells
# ---------------------------------------------------------------------------


class TestCellCount:
    def test_minimal_state_has_four_base_cells(self, minimal_state):
        """Without model / embeddings / SHAP, we expect exactly 4 cells."""
        nb = NotebookExporter(minimal_state).generate()
        # title + imports + dataset + overview + summary = 5 cells
        assert len(nb["cells"]) >= 4

    def test_model_adds_one_cell(self, state_with_model):
        nb_base = NotebookExporter(
            _strip_model(state_with_model)
        ).generate()
        nb_full = NotebookExporter(state_with_model).generate()
        assert len(nb_full["cells"]) == len(nb_base["cells"]) + 1

    def test_embeddings_adds_one_cell(self):
        from sklearn.decomposition import PCA

        # Base state: no embeddings
        base = DeepLensState()
        base.dataset_name = "iris"
        base.df = pd.DataFrame(
            {
                "sepal length (cm)": [5.1, 4.9, 6.3],
                "sepal width (cm)": [3.5, 3.0, 3.3],
                "label": ["setosa", "setosa", "virginica"],
            }
        )
        base.feature_columns = ["sepal length (cm)", "sepal width (cm)"]
        base.label_column = "label"
        base.labels = np.array(["setosa", "setosa", "virginica"])
        base.class_names = ["setosa", "virginica"]

        # Extended state: same base data plus embeddings
        extended = DeepLensState()
        extended.dataset_name = base.dataset_name
        extended.df = base.df
        extended.feature_columns = base.feature_columns
        extended.label_column = base.label_column
        extended.labels = base.labels
        extended.class_names = base.class_names
        X = extended.df[extended.feature_columns].values
        extended.embeddings_raw = X
        extended.embeddings_2d = PCA(n_components=2).fit_transform(X)
        extended.reduction_method = "pca"
        extended.embedding_method = "features"

        nb_base = NotebookExporter(base).generate()
        nb_embed = NotebookExporter(extended).generate()
        assert len(nb_embed["cells"]) == len(nb_base["cells"]) + 1

    def test_shap_adds_one_cell(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        # State with model but no SHAP
        df = pd.DataFrame(
            {
                "sepal length (cm)": [5.1, 4.9, 6.3],
                "sepal width (cm)": [3.5, 3.0, 3.3],
                "label": ["setosa", "setosa", "virginica"],
            }
        )
        feature_columns = ["sepal length (cm)", "sepal width (cm)"]
        labels = np.array(["setosa", "setosa", "virginica"])

        no_shap = DeepLensState()
        no_shap.dataset_name = "iris"
        no_shap.df = df
        no_shap.feature_columns = feature_columns
        no_shap.label_column = "label"
        no_shap.labels = labels
        no_shap.class_names = ["setosa", "virginica"]
        X = df[feature_columns].values
        le = LabelEncoder()
        y = le.fit_transform(labels)
        model = LogisticRegression(max_iter=300, random_state=0)
        model.fit(X, y)
        no_shap.trained_model = model
        no_shap.model_name = "LogisticRegression"
        no_shap.predictions = model.predict(X)
        no_shap.probabilities = model.predict_proba(X)

        # State with model AND SHAP
        with_shap = DeepLensState()
        with_shap.dataset_name = no_shap.dataset_name
        with_shap.df = no_shap.df
        with_shap.feature_columns = no_shap.feature_columns
        with_shap.label_column = no_shap.label_column
        with_shap.labels = no_shap.labels
        with_shap.class_names = no_shap.class_names
        with_shap.trained_model = no_shap.trained_model
        with_shap.model_name = no_shap.model_name
        with_shap.predictions = no_shap.predictions
        with_shap.probabilities = no_shap.probabilities
        with_shap.shap_values = np.zeros((3, 2))
        with_shap.shap_expected = 0.0

        nb_no_shap = NotebookExporter(no_shap).generate()
        nb_shap = NotebookExporter(with_shap).generate()
        assert len(nb_shap["cells"]) == len(nb_no_shap["cells"]) + 1

    def test_all_features_present(self, state_with_shap):
        """Full state: title + imports + dataset + overview + model + embed + shap + summary."""
        from sklearn.decomposition import PCA

        state = state_with_shap
        X = state.df[state.feature_columns].values
        state.embeddings_raw = X
        state.embeddings_2d = PCA(n_components=2).fit_transform(X)
        nb = NotebookExporter(state).generate()
        assert len(nb["cells"]) == 8


def _strip_model(state: DeepLensState) -> DeepLensState:
    """Return a copy-like state with model attributes cleared."""
    bare = DeepLensState()
    bare.dataset_name = state.dataset_name
    bare.df = state.df
    bare.feature_columns = state.feature_columns
    bare.label_column = state.label_column
    bare.labels = state.labels
    bare.class_names = state.class_names
    return bare


# ---------------------------------------------------------------------------
# 3. Cell content checks
# ---------------------------------------------------------------------------


class TestCellContent:
    def _sources(self, state):
        nb = NotebookExporter(state).generate()
        return [c["source"] for c in nb["cells"]]

    def test_title_cell_is_markdown_and_contains_dataset_name(self, minimal_state):
        nb = NotebookExporter(minimal_state).generate()
        first = nb["cells"][0]
        assert first["cell_type"] == "markdown"
        assert "iris" in first["source"]

    def test_title_cell_contains_sample_count(self, minimal_state):
        sources = self._sources(minimal_state)
        title_src = sources[0]
        assert "3" in title_src  # 3 samples in minimal_state

    def test_imports_cell_includes_key_libraries(self, minimal_state):
        sources = self._sources(minimal_state)
        imports_src = sources[1]
        for lib in ("numpy", "pandas", "matplotlib"):
            assert lib in imports_src, f"Expected '{lib}' in imports cell"

    def test_dataset_cell_uses_correct_sklearn_loader(self, minimal_state):
        sources = self._sources(minimal_state)
        dataset_src = sources[2]
        assert "load_iris" in dataset_src

    def test_dataset_cell_generic_placeholder_for_custom(self, custom_state):
        sources = self._sources(custom_state)
        dataset_src = sources[2]
        assert "TODO" in dataset_src or "placeholder" in dataset_src.lower()

    def test_overview_cell_calls_head_and_describe(self, minimal_state):
        sources = self._sources(minimal_state)
        overview_src = sources[3]
        assert "head" in overview_src
        assert "describe" in overview_src

    def test_model_cell_references_model_name(self, state_with_model):
        sources = self._sources(state_with_model)
        # model cell is the 5th cell (index 4)
        model_src = sources[4]
        assert "LogisticRegression" in model_src

    def test_model_cell_contains_train_test_split(self, state_with_model):
        sources = self._sources(state_with_model)
        model_src = sources[4]
        assert "train_test_split" in model_src

    def test_embeddings_cell_references_reduction_method(self, state_with_embeddings):
        sources = self._sources(state_with_embeddings)
        embed_src = "\n".join(sources)
        assert "PCA" in embed_src

    def test_embeddings_cell_contains_scatter_plot(self, state_with_embeddings):
        sources = self._sources(state_with_embeddings)
        embed_src = "\n".join(sources)
        assert "scatter" in embed_src

    def test_shap_cell_references_shap_library(self, state_with_shap):
        sources = self._sources(state_with_shap)
        shap_src = "\n".join(sources)
        assert "import shap" in shap_src

    def test_shap_cell_contains_summary_plot(self, state_with_shap):
        sources = self._sources(state_with_shap)
        shap_src = "\n".join(sources)
        assert "summary_plot" in shap_src

    def test_summary_cell_references_feature_columns(self, minimal_state):
        sources = self._sources(minimal_state)
        summary_src = sources[-1]
        assert "feature_columns" in summary_src

    def test_summary_cell_contains_class_distribution(self, minimal_state):
        sources = self._sources(minimal_state)
        summary_src = sources[-1]
        assert "value_counts" in summary_src or "distribution" in summary_src.lower()


# ---------------------------------------------------------------------------
# 4. to_json() output
# ---------------------------------------------------------------------------


class TestToJson:
    def test_to_json_returns_string(self, minimal_state):
        result = NotebookExporter(minimal_state).to_json()
        assert isinstance(result, str)

    def test_to_json_is_valid_json(self, minimal_state):
        result = NotebookExporter(minimal_state).to_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_to_json_roundtrip_preserves_nbformat(self, minimal_state):
        result = NotebookExporter(minimal_state).to_json()
        parsed = json.loads(result)
        assert parsed["nbformat"] == 4

    def test_to_json_contains_dataset_name(self, minimal_state):
        result = NotebookExporter(minimal_state).to_json()
        assert "iris" in result

    def test_to_json_full_state_is_valid_json(self, state_with_shap):
        """Even a fully-populated state produces valid JSON."""
        result = NotebookExporter(state_with_shap).to_json()
        parsed = json.loads(result)
        assert len(parsed["cells"]) >= 6


# ---------------------------------------------------------------------------
# 5. save() writes a readable .ipynb file
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_creates_file(self, tmp_path, minimal_state):
        out = tmp_path / "test.ipynb"
        NotebookExporter(minimal_state).save(str(out))
        assert out.exists()

    def test_save_file_is_valid_json(self, tmp_path, minimal_state):
        out = tmp_path / "test.ipynb"
        NotebookExporter(minimal_state).save(str(out))
        with open(out, encoding="utf-8") as fh:
            parsed = json.load(fh)
        assert "cells" in parsed

    def test_save_and_generate_produce_same_content(self, tmp_path, minimal_state):
        out = tmp_path / "test.ipynb"
        exporter = NotebookExporter(minimal_state)
        exporter.save(str(out))
        with open(out, encoding="utf-8") as fh:
            from_file = json.load(fh)
        from_generate = exporter.generate()
        assert from_file == from_generate


# ---------------------------------------------------------------------------
# 6. Edge cases & alternative branches
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_state_generates_notebook(self):
        """An entirely default (empty) state should still produce a notebook."""
        state = DeepLensState()
        nb = NotebookExporter(state).generate()
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) >= 4

    def test_wine_dataset_uses_load_wine(self):
        state = DeepLensState()
        state.dataset_name = "wine"
        state.df = pd.DataFrame({"f": [1.0, 2.0], "label": ["a", "b"]})
        state.feature_columns = ["f"]
        state.label_column = "label"
        nb = NotebookExporter(state).generate()
        dataset_src = nb["cells"][2]["source"]
        assert "load_wine" in dataset_src

    def test_umap_reduction_emits_umap_import(self):
        state = DeepLensState()
        state.dataset_name = "iris"
        state.df = pd.DataFrame({"f": [1.0, 2.0, 3.0], "label": ["a", "b", "a"]})
        state.feature_columns = ["f"]
        state.label_column = "label"
        state.embeddings_2d = np.zeros((3, 2))
        state.reduction_method = "umap"
        state.embedding_method = "features"
        nb = NotebookExporter(state).generate()
        sources = "\n".join(c["source"] for c in nb["cells"])
        assert "UMAP" in sources

    def test_tsne_reduction_emits_tsne_import(self):
        state = DeepLensState()
        state.dataset_name = "iris"
        state.df = pd.DataFrame({"f": [1.0, 2.0, 3.0], "label": ["a", "b", "a"]})
        state.feature_columns = ["f"]
        state.label_column = "label"
        state.embeddings_2d = np.zeros((3, 2))
        state.reduction_method = "tsne"
        state.embedding_method = "features"
        nb = NotebookExporter(state).generate()
        sources = "\n".join(c["source"] for c in nb["cells"])
        assert "TSNE" in sources

    def test_tfidf_embedding_emits_tfidf_vectorizer(self):
        state = DeepLensState()
        state.dataset_name = "custom_text"
        state.df = pd.DataFrame({"text": ["hello world", "foo bar"], "label": ["a", "b"]})
        state.feature_columns = []
        state.label_column = "label"
        state.text_column = "text"
        state.embedding_method = "tfidf"
        state.embeddings_2d = np.zeros((2, 2))
        state.reduction_method = "pca"
        nb = NotebookExporter(state).generate()
        sources = "\n".join(c["source"] for c in nb["cells"])
        assert "TfidfVectorizer" in sources

    def test_unknown_model_name_still_generates_cell(self):
        from sklearn.linear_model import LogisticRegression

        state = DeepLensState()
        state.dataset_name = "iris"
        state.df = pd.DataFrame({"f": [1.0, 2.0, 3.0], "label": ["a", "b", "a"]})
        state.feature_columns = ["f"]
        state.label_column = "label"
        state.labels = np.array(["a", "b", "a"])

        model = LogisticRegression(max_iter=300).fit([[1], [2], [3]], [0, 1, 0])
        state.trained_model = model
        state.model_name = "MyCustomModel"  # not in the known map

        nb = NotebookExporter(state).generate()
        model_cell_src = nb["cells"][4]["source"]
        # constructor falls back to "MyCustomModel()"
        assert "MyCustomModel()" in model_cell_src

    def test_title_cell_shows_model_info_when_trained(self, state_with_model):
        nb = NotebookExporter(state_with_model).generate()
        title = nb["cells"][0]["source"]
        assert "LogisticRegression" in title

    def test_title_cell_shows_embedding_info_when_present(self, state_with_embeddings):
        nb = NotebookExporter(state_with_embeddings).generate()
        title = nb["cells"][0]["source"]
        assert "pca" in title.lower() or "features" in title.lower()

    def test_generate_is_idempotent(self, minimal_state):
        """Calling generate() twice produces identical output."""
        exporter = NotebookExporter(minimal_state)
        assert exporter.generate() == exporter.generate()
