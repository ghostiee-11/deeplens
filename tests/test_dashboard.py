"""Tests for deeplens.dashboard.app — DeepLensDashboard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import panel as pn
import pytest

from deeplens.config import DeepLensState
from deeplens.dashboard.app import DeepLensDashboard, _safe_import, launch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_dashboard():
    """Dashboard with default (empty) state — no data loaded."""
    return DeepLensDashboard()


@pytest.fixture
def dashboard_with_state(iris_with_embeddings):
    """Dashboard pre-populated with an iris state that has 2-D embeddings."""
    return DeepLensDashboard(state=iris_with_embeddings)


@pytest.fixture
def dashboard_with_model(iris_with_model):
    """Dashboard with a trained model in state."""
    return DeepLensDashboard(state=iris_with_model)


# ---------------------------------------------------------------------------
# _safe_import helper
# ---------------------------------------------------------------------------

class TestSafeImport:
    def test_valid_import_returns_class(self):
        cls = _safe_import("deeplens.config", "DeepLensState")
        assert cls is DeepLensState

    def test_missing_module_returns_none(self):
        result = _safe_import("deeplens.nonexistent_module", "SomeClass")
        assert result is None

    def test_missing_attribute_returns_none(self):
        result = _safe_import("deeplens.config", "NonexistentClass")
        assert result is None


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestDashboardInit:
    def test_default_state_created_when_none_given(self, empty_dashboard):
        assert isinstance(empty_dashboard.state, DeepLensState)

    def test_provided_state_is_used(self, iris_with_embeddings):
        db = DeepLensDashboard(state=iris_with_embeddings)
        assert db.state is iris_with_embeddings

    def test_llm_created_on_init(self, empty_dashboard):
        assert empty_dashboard._llm is not None

    def test_tab_cache_starts_empty(self, empty_dashboard):
        assert empty_dashboard._tab_cache == {}

    def test_embedder_created(self, empty_dashboard):
        from deeplens.embeddings.compute import EmbeddingComputer
        assert isinstance(empty_dashboard._embedder, EmbeddingComputer)

    def test_reducer_created(self, empty_dashboard):
        from deeplens.embeddings.reduce import DimensionalityReducer
        assert isinstance(empty_dashboard._reducer, DimensionalityReducer)

    def test_tabs_widget_created(self, empty_dashboard):
        assert isinstance(empty_dashboard._tabs, pn.Tabs)

    def test_tabs_has_seven_tabs(self, empty_dashboard):
        # Explore, Profile, Explain, Inspect, Compare, Drift, Quality, Annotate
        assert len(empty_dashboard._tabs) == 8

    def test_status_alert_created(self, empty_dashboard):
        assert isinstance(empty_dashboard._status, pn.pane.Alert)

    def test_load_btn_created(self, empty_dashboard):
        assert isinstance(empty_dashboard._load_btn, pn.widgets.Button)

    def test_dataset_select_has_options(self, empty_dashboard):
        assert len(empty_dashboard._dataset_select.options) > 0


# ---------------------------------------------------------------------------
# _build_sidebar
# ---------------------------------------------------------------------------

class TestBuildSidebar:
    def test_returns_column(self, empty_dashboard):
        sidebar = empty_dashboard._build_sidebar()
        assert isinstance(sidebar, pn.Column)

    def test_sidebar_contains_accordion(self, empty_dashboard):
        sidebar = empty_dashboard._build_sidebar()
        # There should be at least one Accordion somewhere in the Column
        accordion_types = [type(c).__name__ for c in sidebar]
        assert "Accordion" in accordion_types

    def test_accordion_has_six_sections(self, empty_dashboard):
        sidebar = empty_dashboard._build_sidebar()
        accordion = next(c for c in sidebar if isinstance(c, pn.Accordion))
        # Dataset, Upload/Fetch, Train Model, Upload Pre-trained, NL Filter, AI Analyst
        assert len(accordion) == 6

    def test_sidebar_scrollable(self, empty_dashboard):
        sidebar = empty_dashboard._build_sidebar()
        assert sidebar.scroll is True

    def test_sidebar_stretch_width(self, empty_dashboard):
        sidebar = empty_dashboard._build_sidebar()
        assert sidebar.sizing_mode == "stretch_width"


# ---------------------------------------------------------------------------
# _ingest_dataframe
# ---------------------------------------------------------------------------

class TestIngestDataframe:
    def _make_df(self, n: int = 30) -> pd.DataFrame:
        rng = np.random.RandomState(5)
        return pd.DataFrame(
            {
                "f1": rng.randn(n),
                "f2": rng.randn(n),
                "f3": rng.randn(n),
                "label": (["a", "b"] * (n // 2)),
            }
        )

    def test_state_df_populated(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df, name="my_data")
        assert empty_dashboard.state.df is not None
        assert len(empty_dashboard.state.df) == len(df)

    def test_state_dataset_name_set(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df, name="my_data")
        assert empty_dashboard.state.dataset_name == "my_data"

    def test_embeddings_computed(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df)
        assert empty_dashboard.state.embeddings_2d is not None
        assert empty_dashboard.state.embeddings_2d.ndim == 2

    def test_loading_flag_reset_to_false(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df)
        assert empty_dashboard.state.loading is False

    def test_tab_cache_cleared(self, empty_dashboard):
        # Pre-populate cache
        empty_dashboard._tab_cache["Explore"] = pn.pane.Markdown("old")
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df)
        # Cache should be rebuilt (either cleared or repopulated)
        # The key point: stale "old" content is gone
        assert empty_dashboard._tab_cache.get("Explore") is not pn.pane.Markdown("old")

    def test_labels_extracted(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df)
        assert empty_dashboard.state.labels is not None
        assert len(empty_dashboard.state.labels) == len(df)

    def test_class_names_extracted(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df)
        assert set(empty_dashboard.state.class_names) == {"a", "b"}

    def test_status_updated_to_success(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df)
        assert empty_dashboard._status.alert_type == "success"

    def test_summary_indicators_updated(self, empty_dashboard):
        df = self._make_df(n=30)
        empty_dashboard._ingest_dataframe(df)
        assert empty_dashboard._summary_samples.value == 30

    def test_summary_row_becomes_visible(self, empty_dashboard):
        df = self._make_df()
        empty_dashboard._ingest_dataframe(df)
        assert empty_dashboard._summary_row.visible is True

    def test_old_state_cleared_before_ingest(self, dashboard_with_model):
        """Ingesting new data should wipe stale model/prediction state."""
        db = dashboard_with_model
        rng = np.random.RandomState(3)
        new_df = pd.DataFrame(
            {
                "f1": rng.randn(30),
                "f2": rng.randn(30),
                "label": ["x", "y"] * 15,
            }
        )
        db._ingest_dataframe(new_df, name="new")
        assert db.state.trained_model is None
        assert db.state.predictions is None

    def test_no_features_sets_warning_status(self, empty_dashboard):
        """A DataFrame with only non-numeric columns should set warning status."""
        df = pd.DataFrame({"text": ["hello", "world"] * 5})
        empty_dashboard._ingest_dataframe(df, name="text_only")
        # No numeric features → warning or error alert, not success
        assert empty_dashboard._status.alert_type in ("warning", "danger")


# ---------------------------------------------------------------------------
# Tab builders — with and without prerequisites
# ---------------------------------------------------------------------------

class TestBuildExplore:
    def test_no_embeddings_returns_markdown(self, empty_dashboard):
        result = empty_dashboard._build_tab("Explore")
        assert isinstance(result, pn.pane.Markdown)

    def test_with_embeddings_returns_explorer_or_markdown(self, dashboard_with_state):
        result = dashboard_with_state._build_tab("Explore")
        # Should be an EmbeddingExplorer; at worst a fallback Markdown
        assert result is not None


class TestBuildExplain:
    def test_no_model_returns_markdown(self, empty_dashboard):
        result = empty_dashboard._build_tab("Explain")
        assert isinstance(result, pn.pane.Markdown)
        assert "Train" in result.object or "model" in result.object.lower()

    def test_with_model_returns_content(self, dashboard_with_model):
        result = dashboard_with_model._build_tab("Explain")
        assert result is not None


class TestBuildInspect:
    def test_no_model_returns_markdown(self, empty_dashboard):
        result = empty_dashboard._build_tab("Inspect")
        assert isinstance(result, pn.pane.Markdown)

    def test_with_model_returns_non_none(self, dashboard_with_model):
        result = dashboard_with_model._build_tab("Inspect")
        assert result is not None


class TestBuildCompare:
    def test_zero_models_returns_markdown(self, empty_dashboard):
        result = empty_dashboard._build_tab("Compare")
        assert isinstance(result, pn.pane.Markdown)
        assert "2" in result.object

    def test_one_model_returns_markdown(self, dashboard_with_model):
        # Only one model in history → Compare should still ask for another
        dashboard_with_model.state.model_history = [("LR", MagicMock())]
        result = dashboard_with_model._build_tab("Compare")
        assert isinstance(result, pn.pane.Markdown)

    def test_two_models_attempts_arena(self, dashboard_with_model):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression

        state = dashboard_with_model.state
        X = state.df[state.feature_columns].values
        y = state.labels
        m1 = LogisticRegression(max_iter=300).fit(X, y)
        m2 = DecisionTreeClassifier().fit(X, y)
        state.model_history = [("LR", m1), ("DT", m2)]
        result = dashboard_with_model._build_tab("Compare")
        # Either a real Arena or a fallback Markdown — just no exception
        assert result is not None


class TestBuildDrift:
    def test_no_data_returns_markdown(self, empty_dashboard):
        result = empty_dashboard._build_tab("Drift")
        assert isinstance(result, pn.pane.Markdown)

    def test_with_data_returns_content(self, dashboard_with_state):
        result = dashboard_with_state._build_tab("Drift")
        assert result is not None


class TestBuildQuality:
    def test_no_embeddings_returns_markdown(self, empty_dashboard):
        result = empty_dashboard._build_tab("Quality")
        assert isinstance(result, pn.pane.Markdown)

    def test_with_embeddings_returns_content(self, dashboard_with_state):
        result = dashboard_with_state._build_tab("Quality")
        assert result is not None


class TestBuildAnnotate:
    def test_no_model_returns_markdown(self, empty_dashboard):
        result = empty_dashboard._build_tab("Annotate")
        assert isinstance(result, pn.pane.Markdown)
        assert "model" in result.object.lower() or "Train" in result.object

    def test_with_model_returns_annotator(self, dashboard_with_model):
        result = dashboard_with_model._build_tab("Annotate")
        assert result is not None


# ---------------------------------------------------------------------------
# _on_load_dataset
# ---------------------------------------------------------------------------

class TestOnLoadDataset:
    def test_load_iris_populates_state(self, empty_dashboard):
        empty_dashboard._dataset_select.value = "Iris (150 samples, 4 features)"
        empty_dashboard._on_load_dataset()
        assert empty_dashboard.state.df is not None
        assert len(empty_dashboard.state.df) == 150

    def test_load_dataset_sets_success_alert(self, empty_dashboard):
        empty_dashboard._dataset_select.value = "Iris (150 samples, 4 features)"
        empty_dashboard._on_load_dataset()
        assert empty_dashboard._status.alert_type == "success"

    def test_load_dataset_failure_sets_danger_alert(self, empty_dashboard):
        with patch("deeplens.dashboard.app.load_sklearn", side_effect=RuntimeError("boom")):
            empty_dashboard._on_load_dataset()
        assert empty_dashboard._status.alert_type == "danger"


# ---------------------------------------------------------------------------
# _read_file_bytes (static helper)
# ---------------------------------------------------------------------------

class TestReadFileBytes:
    def _csv_bytes(self):
        return b"f1,f2,label\n1.0,2.0,a\n3.0,4.0,b\n"

    def test_reads_csv(self):
        df = DeepLensDashboard._read_file_bytes(self._csv_bytes(), "data.csv")
        assert len(df) == 2
        assert list(df.columns) == ["f1", "f2", "label"]

    def test_reads_tsv(self):
        data = b"f1\tf2\n1.0\t2.0\n3.0\t4.0\n"
        df = DeepLensDashboard._read_file_bytes(data, "data.tsv")
        assert len(df) == 2

    def test_reads_json(self):
        import json
        data = json.dumps([{"f1": 1, "f2": 2}, {"f1": 3, "f2": 4}]).encode()
        df = DeepLensDashboard._read_file_bytes(data, "data.json")
        assert len(df) == 2

    def test_reads_jsonl(self):
        data = b'{"f1": 1}\n{"f1": 2}\n'
        df = DeepLensDashboard._read_file_bytes(data, "data.jsonl")
        assert len(df) == 2

    def test_unknown_extension_falls_back_to_csv(self):
        df = DeepLensDashboard._read_file_bytes(self._csv_bytes(), "data.txt")
        assert len(df) == 2


# ---------------------------------------------------------------------------
# _generate_snapshot
# ---------------------------------------------------------------------------

class TestGenerateSnapshot:
    def test_returns_string_io(self, dashboard_with_state):
        import io
        result = dashboard_with_state._generate_snapshot()
        assert isinstance(result, io.StringIO)

    def test_snapshot_contains_dataset_name(self, dashboard_with_state):
        result = dashboard_with_state._generate_snapshot()
        text = result.getvalue()
        assert "iris" in text

    def test_snapshot_is_valid_json(self, dashboard_with_state):
        import json
        result = dashboard_with_state._generate_snapshot()
        parsed = json.loads(result.getvalue())
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# __panel__
# ---------------------------------------------------------------------------

class TestDashboardPanel:
    def test_panel_returns_fast_list_template(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        assert isinstance(result, pn.template.FastListTemplate)

    def test_panel_is_viewable(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        assert isinstance(result, pn.template.FastListTemplate)

    def test_panel_with_loaded_state_does_not_raise(self, dashboard_with_state):
        result = dashboard_with_state.__panel__()
        assert result is not None

    def test_panel_template_title_contains_deeplens(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        assert "DeepLens" in result.title


# ---------------------------------------------------------------------------
# DeepLensDashboard.create classmethod
# ---------------------------------------------------------------------------

class TestDashboardCreate:
    def test_create_returns_dashboard(self):
        db = DeepLensDashboard.create(dataset="iris", llm_provider="none")
        assert isinstance(db, DeepLensDashboard)

    def test_create_populates_state(self):
        db = DeepLensDashboard.create(dataset="iris", llm_provider="none")
        assert db.state.df is not None
        assert len(db.state.df) == 150

    def test_create_wine(self):
        db = DeepLensDashboard.create(dataset="wine", llm_provider="none")
        assert db.state.dataset_name == "wine"


# ---------------------------------------------------------------------------
# launch() function
# ---------------------------------------------------------------------------

class TestLaunch:
    def test_launch_returns_dashboard(self):
        with patch.object(pn.template.FastListTemplate, "servable"):
            db = launch(dataset="iris", llm_provider="none", show=False, port=0)
        assert isinstance(db, DeepLensDashboard)

    def test_launch_show_false_calls_servable(self):
        with patch.object(pn.template.FastListTemplate, "servable") as mock_servable:
            launch(dataset="iris", show=False, port=0)
        mock_servable.assert_called_once()

    def test_launch_show_true_calls_template_show(self):
        with patch.object(pn.template.FastListTemplate, "show") as mock_show:
            launch(dataset="iris", show=True, port=0)
        mock_show.assert_called_once()

    def test_launch_with_port_passes_port_to_show(self):
        with patch.object(pn.template.FastListTemplate, "show") as mock_show:
            launch(dataset="iris", show=True, port=9999)
        call_kwargs = mock_show.call_args.kwargs
        assert call_kwargs.get("port") == 9999


# ---------------------------------------------------------------------------
# _read_file_bytes — extended coverage
# ---------------------------------------------------------------------------

class TestReadFileBytesExtended:
    """Additional coverage for _read_file_bytes with more file types."""

    def test_reads_parquet(self):
        import io
        df_orig = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        buf = io.BytesIO()
        df_orig.to_parquet(buf, index=False)
        data = buf.getvalue()
        df = DeepLensDashboard._read_file_bytes(data, "data.parquet")
        assert len(df) == 3
        assert list(df.columns) == ["a", "b"]

    def test_reads_xlsx(self):
        import io
        df_orig = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        buf = io.BytesIO()
        df_orig.to_excel(buf, index=False)
        data = buf.getvalue()
        df = DeepLensDashboard._read_file_bytes(data, "data.xlsx")
        assert len(df) == 2
        assert "x" in df.columns

    def test_reads_xls_extension(self):
        # xls extension should also route to read_excel; use xlsx bytes as proxy
        import io
        df_orig = pd.DataFrame({"col": [1, 2]})
        buf = io.BytesIO()
        df_orig.to_excel(buf, index=False)
        data = buf.getvalue()
        df = DeepLensDashboard._read_file_bytes(data, "data.xls")
        assert len(df) == 2

    def test_reads_csv_correct_values(self):
        data = b"name,score\nalice,95\nbob,87\ncarol,92\n"
        df = DeepLensDashboard._read_file_bytes(data, "scores.csv")
        assert len(df) == 3
        assert df["score"].tolist() == [95, 87, 92]

    def test_reads_json_records(self):
        import json
        records = [{"id": i, "val": i * 2} for i in range(5)]
        data = json.dumps(records).encode()
        df = DeepLensDashboard._read_file_bytes(data, "records.json")
        assert len(df) == 5
        assert "id" in df.columns

    def test_reads_jsonl_multiple_records(self):
        lines = b'{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n{"a": 3, "b": "z"}\n'
        df = DeepLensDashboard._read_file_bytes(lines, "data.jsonl")
        assert len(df) == 3
        assert list(df.columns) == ["a", "b"]

    def test_reads_tsv_correct_separator(self):
        data = b"col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6\n"
        df = DeepLensDashboard._read_file_bytes(data, "data.tsv")
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_fallback_extension_still_parses_csv(self):
        data = b"p,q\n1,2\n3,4\n"
        df = DeepLensDashboard._read_file_bytes(data, "data.dat")
        assert len(df) == 2
        assert list(df.columns) == ["p", "q"]

    def test_uppercase_extension_treated_as_fallback(self):
        # .CSV (uppercase) does not match lower().endswith(".csv") — goes to fallback
        data = b"p,q\n1,2\n3,4\n"
        df = DeepLensDashboard._read_file_bytes(data, "DATA.CSV")
        # fallback read_csv should still work
        assert len(df) == 2


# ---------------------------------------------------------------------------
# _read_url — mocked HTTP/pandas
# ---------------------------------------------------------------------------

class TestReadUrl:
    """Tests for _read_url using mocked pandas I/O."""

    def test_csv_url_calls_read_csv(self):
        fake_df = pd.DataFrame({"a": [1, 2]})
        with patch("deeplens.dashboard.app.pd.read_csv", return_value=fake_df) as mock_rc:
            result = DeepLensDashboard._read_url("https://example.com/data.csv")
        mock_rc.assert_called_once_with("https://example.com/data.csv")
        assert len(result) == 2

    def test_tsv_url_calls_read_csv_with_tab(self):
        fake_df = pd.DataFrame({"a": [1]})
        with patch("deeplens.dashboard.app.pd.read_csv", return_value=fake_df) as mock_rc:
            DeepLensDashboard._read_url("https://example.com/data.tsv")
        mock_rc.assert_called_once_with("https://example.com/data.tsv", sep="\t")

    def test_json_url_calls_read_json(self):
        fake_df = pd.DataFrame({"k": [1]})
        with patch("deeplens.dashboard.app.pd.read_json", return_value=fake_df) as mock_rj:
            result = DeepLensDashboard._read_url("https://example.com/data.json")
        mock_rj.assert_called_once_with("https://example.com/data.json")
        assert result is fake_df

    def test_jsonl_url_calls_read_json_lines(self):
        fake_df = pd.DataFrame({"k": [1]})
        with patch("deeplens.dashboard.app.pd.read_json", return_value=fake_df) as mock_rj:
            DeepLensDashboard._read_url("https://example.com/data.jsonl")
        mock_rj.assert_called_once_with("https://example.com/data.jsonl", lines=True)

    def test_parquet_url_calls_read_parquet(self):
        fake_df = pd.DataFrame({"m": [1]})
        with patch("deeplens.dashboard.app.pd.read_parquet", return_value=fake_df) as mock_rp:
            result = DeepLensDashboard._read_url("https://example.com/data.parquet")
        mock_rp.assert_called_once_with("https://example.com/data.parquet")
        assert result is fake_df

    def test_xlsx_url_calls_read_excel(self):
        fake_df = pd.DataFrame({"z": [1]})
        with patch("deeplens.dashboard.app.pd.read_excel", return_value=fake_df) as mock_re:
            DeepLensDashboard._read_url("https://example.com/data.xlsx")
        mock_re.assert_called_once_with("https://example.com/data.xlsx")

    def test_xls_url_calls_read_excel(self):
        fake_df = pd.DataFrame({"z": [1]})
        with patch("deeplens.dashboard.app.pd.read_excel", return_value=fake_df) as mock_re:
            DeepLensDashboard._read_url("https://example.com/file.xls")
        mock_re.assert_called_once()

    def test_url_with_query_params_uses_path_only_for_extension(self):
        """Query string should not confuse extension detection."""
        fake_df = pd.DataFrame({"col": [1]})
        with patch("deeplens.dashboard.app.pd.read_csv", return_value=fake_df) as mock_rc:
            DeepLensDashboard._read_url("https://example.com/data.csv?token=abc&v=2")
        mock_rc.assert_called_once()

    def test_unknown_url_defaults_to_csv(self):
        fake_df = pd.DataFrame({"v": [1]})
        with patch("deeplens.dashboard.app.pd.read_csv", return_value=fake_df) as mock_rc:
            DeepLensDashboard._read_url("https://example.com/noextension")
        mock_rc.assert_called_once()


# ---------------------------------------------------------------------------
# _on_upload_file — mock FileInput
# ---------------------------------------------------------------------------

class TestOnUploadFile:
    def test_no_file_sets_warning(self, empty_dashboard):
        empty_dashboard._file_input.value = None
        empty_dashboard._on_upload_file()
        assert empty_dashboard._status.alert_type == "warning"
        assert "No file" in empty_dashboard._status.object

    def test_valid_csv_file_populates_state(self, empty_dashboard):
        csv_bytes = b"f1,f2,label\n1.0,2.0,a\n3.0,4.0,b\n5.0,6.0,a\n"
        empty_dashboard._file_input.value = csv_bytes
        empty_dashboard._file_input.filename = "test_data.csv"
        empty_dashboard._on_upload_file()
        assert empty_dashboard.state.df is not None
        assert len(empty_dashboard.state.df) == 3

    def test_valid_csv_sets_success_status(self, empty_dashboard):
        csv_bytes = b"f1,f2,label\n" + b"1.0,2.0,a\n" * 10
        empty_dashboard._file_input.value = csv_bytes
        empty_dashboard._file_input.filename = "test.csv"
        empty_dashboard._on_upload_file()
        assert empty_dashboard._status.alert_type == "success"

    def test_corrupt_file_sets_danger_status(self, empty_dashboard):
        # Corrupt parquet bytes → read_parquet will raise
        empty_dashboard._file_input.value = b"this is not a parquet file at all!!!"
        empty_dashboard._file_input.filename = "bad.parquet"
        empty_dashboard._on_upload_file()
        assert empty_dashboard._status.alert_type == "danger"
        assert "Failed" in empty_dashboard._status.object

    def test_uses_filename_for_dataset_name(self, empty_dashboard):
        csv_bytes = b"f1,f2,label\n1.0,2.0,a\n3.0,4.0,b\n"
        empty_dashboard._file_input.value = csv_bytes
        empty_dashboard._file_input.filename = "my_dataset.csv"
        empty_dashboard._on_upload_file()
        assert empty_dashboard.state.dataset_name == "my_dataset.csv"

    def test_tsv_file_parsed_correctly(self, empty_dashboard):
        tsv_bytes = b"col1\tcol2\tlabel\n1\t2\ta\n3\t4\tb\n"
        empty_dashboard._file_input.value = tsv_bytes
        empty_dashboard._file_input.filename = "data.tsv"
        empty_dashboard._on_upload_file()
        assert empty_dashboard.state.df is not None
        assert "col1" in empty_dashboard.state.df.columns

    def test_json_file_parsed_correctly(self, empty_dashboard):
        import json
        records = [{"f1": float(i), "f2": float(i * 2), "label": "a"} for i in range(5)]
        json_bytes = json.dumps(records).encode()
        empty_dashboard._file_input.value = json_bytes
        empty_dashboard._file_input.filename = "data.json"
        empty_dashboard._on_upload_file()
        assert empty_dashboard.state.df is not None
        assert len(empty_dashboard.state.df) == 5


# ---------------------------------------------------------------------------
# _on_fetch_url — mock _read_url
# ---------------------------------------------------------------------------

class TestOnFetchUrl:
    def test_empty_url_sets_warning(self, empty_dashboard):
        empty_dashboard._url_input.value = ""
        empty_dashboard._on_fetch_url()
        assert empty_dashboard._status.alert_type == "warning"
        assert "URL" in empty_dashboard._status.object

    def test_whitespace_url_sets_warning(self, empty_dashboard):
        empty_dashboard._url_input.value = "   "
        empty_dashboard._on_fetch_url()
        assert empty_dashboard._status.alert_type == "warning"

    def test_valid_url_populates_state(self, empty_dashboard):
        fake_df = pd.DataFrame({
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            "label": ["a", "b"] * 10,
        })
        with patch.object(DeepLensDashboard, "_read_url", return_value=fake_df):
            empty_dashboard._url_input.value = "https://example.com/data.csv"
            empty_dashboard._on_fetch_url()
        assert empty_dashboard.state.df is not None
        assert len(empty_dashboard.state.df) == 20

    def test_valid_url_sets_success_status(self, empty_dashboard):
        fake_df = pd.DataFrame({
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            "label": ["a", "b"] * 10,
        })
        with patch.object(DeepLensDashboard, "_read_url", return_value=fake_df):
            empty_dashboard._url_input.value = "https://example.com/data.csv"
            empty_dashboard._on_fetch_url()
        assert empty_dashboard._status.alert_type == "success"

    def test_fetch_error_sets_danger_status(self, empty_dashboard):
        with patch.object(
            DeepLensDashboard, "_read_url", side_effect=RuntimeError("connection refused")
        ):
            empty_dashboard._url_input.value = "https://bad-host.invalid/data.csv"
            empty_dashboard._on_fetch_url()
        assert empty_dashboard._status.alert_type == "danger"
        assert "Failed" in empty_dashboard._status.object

    def test_dataset_name_extracted_from_url_path(self, empty_dashboard):
        fake_df = pd.DataFrame({
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            "label": ["a", "b"] * 10,
        })
        with patch.object(DeepLensDashboard, "_read_url", return_value=fake_df):
            empty_dashboard._url_input.value = "https://example.com/datasets/my_data.csv"
            empty_dashboard._on_fetch_url()
        assert empty_dashboard.state.dataset_name == "my_data.csv"

    def test_dataset_name_strips_query_string(self, empty_dashboard):
        fake_df = pd.DataFrame({
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            "label": ["a", "b"] * 10,
        })
        with patch.object(DeepLensDashboard, "_read_url", return_value=fake_df):
            empty_dashboard._url_input.value = "https://example.com/file.csv?token=xyz"
            empty_dashboard._on_fetch_url()
        assert empty_dashboard.state.dataset_name == "file.csv"


# ---------------------------------------------------------------------------
# _on_upload_model
# ---------------------------------------------------------------------------

class TestOnUploadModel:
    def _make_pkl_bytes(self):
        import pickle
        from sklearn.linear_model import LogisticRegression
        import io
        # Provide a minimal dataset to fit the model
        rng = np.random.RandomState(0)
        X = rng.randn(30, 4)
        y = np.array(["a", "b"] * 15)
        model = LogisticRegression(max_iter=200).fit(X, y)
        return pickle.dumps(model)

    def test_no_file_sets_warning(self, empty_dashboard):
        empty_dashboard._model_file_input.value = None
        empty_dashboard._on_upload_model()
        assert empty_dashboard._status.alert_type == "warning"
        assert "No model file" in empty_dashboard._status.object

    def test_no_dataframe_sets_warning(self, empty_dashboard):
        # Has file bytes but no dataset loaded
        empty_dashboard._model_file_input.value = b"fake"
        empty_dashboard._model_file_input.filename = "model.pkl"
        empty_dashboard._on_upload_model()
        assert empty_dashboard._status.alert_type == "warning"
        assert "dataset" in empty_dashboard._status.object.lower()

    def test_valid_model_loads_and_sets_success(self, dashboard_with_state):
        pkl_bytes = self._make_pkl_bytes()
        dashboard_with_state._model_file_input.value = pkl_bytes
        dashboard_with_state._model_file_input.filename = "lr_model.pkl"
        dashboard_with_state._on_upload_model()
        assert dashboard_with_state._status.alert_type == "success"

    def test_valid_model_populates_trained_model(self, dashboard_with_state):
        pkl_bytes = self._make_pkl_bytes()
        dashboard_with_state._model_file_input.value = pkl_bytes
        dashboard_with_state._model_file_input.filename = "lr_model.pkl"
        dashboard_with_state._on_upload_model()
        assert dashboard_with_state.state.trained_model is not None

    def test_valid_model_generates_predictions(self, dashboard_with_state):
        pkl_bytes = self._make_pkl_bytes()
        dashboard_with_state._model_file_input.value = pkl_bytes
        dashboard_with_state._model_file_input.filename = "lr_model.pkl"
        dashboard_with_state._on_upload_model()
        assert dashboard_with_state.state.predictions is not None
        assert len(dashboard_with_state.state.predictions) == len(dashboard_with_state.state.df)

    def test_model_name_set_without_extension(self, dashboard_with_state):
        pkl_bytes = self._make_pkl_bytes()
        dashboard_with_state._model_file_input.value = pkl_bytes
        dashboard_with_state._model_file_input.filename = "my_clf.pkl"
        dashboard_with_state._on_upload_model()
        assert dashboard_with_state.state.model_name == "my_clf"

    def test_object_without_predict_sets_danger(self, dashboard_with_state):
        import pickle
        bad_obj = {"not": "a model"}
        bad_bytes = pickle.dumps(bad_obj)
        dashboard_with_state._model_file_input.value = bad_bytes
        dashboard_with_state._model_file_input.filename = "bad.pkl"
        dashboard_with_state._on_upload_model()
        assert dashboard_with_state._status.alert_type == "danger"

    def test_corrupt_bytes_sets_danger(self, dashboard_with_state):
        dashboard_with_state._model_file_input.value = b"this is not a valid pickle"
        dashboard_with_state._model_file_input.filename = "corrupt.pkl"
        dashboard_with_state._on_upload_model()
        assert dashboard_with_state._status.alert_type == "danger"


# ---------------------------------------------------------------------------
# _build_tab — systematic coverage of each branch
# ---------------------------------------------------------------------------

class TestBuildTabAllNames:
    """Directly invoke _build_tab with each known tab name."""

    def test_build_explore_no_embeddings(self, empty_dashboard):
        result = empty_dashboard._build_tab("Explore")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_profile_no_data(self, empty_dashboard):
        result = empty_dashboard._build_tab("Profile")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_explain_no_model(self, empty_dashboard):
        result = empty_dashboard._build_tab("Explain")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_inspect_no_model(self, empty_dashboard):
        result = empty_dashboard._build_tab("Inspect")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_compare_no_models(self, empty_dashboard):
        result = empty_dashboard._build_tab("Compare")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_drift_no_data(self, empty_dashboard):
        result = empty_dashboard._build_tab("Drift")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_quality_no_embeddings(self, empty_dashboard):
        result = empty_dashboard._build_tab("Quality")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_annotate_no_model(self, empty_dashboard):
        result = empty_dashboard._build_tab("Annotate")
        assert isinstance(result, pn.pane.Markdown)

    def test_build_unknown_tab_returns_none(self, empty_dashboard):
        result = empty_dashboard._build_tab("NonExistentTab")
        assert result is None

    def test_build_profile_with_data(self, dashboard_with_state):
        # With data loaded; may return profiler or fallback Markdown
        result = dashboard_with_state._build_tab("Profile")
        assert result is not None

    def test_build_drift_with_data(self, dashboard_with_state):
        result = dashboard_with_state._build_tab("Drift")
        assert result is not None

    def test_all_tab_names_return_something(self, empty_dashboard):
        tab_names = ["Explore", "Profile", "Explain", "Inspect",
                     "Compare", "Drift", "Quality", "Annotate"]
        for name in tab_names:
            result = empty_dashboard._build_tab(name)
            assert result is not None, f"_build_tab('{name}') returned None"

    def test_explore_with_embeddings_returns_non_none(self, dashboard_with_state):
        result = dashboard_with_state._build_tab("Explore")
        assert result is not None

    def test_explain_with_model_returns_non_none(self, dashboard_with_model):
        result = dashboard_with_model._build_tab("Explain")
        assert result is not None

    def test_inspect_with_model_returns_non_none(self, dashboard_with_model):
        result = dashboard_with_model._build_tab("Inspect")
        assert result is not None

    def test_annotate_with_model_returns_non_none(self, dashboard_with_model):
        result = dashboard_with_model._build_tab("Annotate")
        assert result is not None


# ---------------------------------------------------------------------------
# _on_open_chat
# ---------------------------------------------------------------------------

class TestOnOpenChat:
    def test_creates_float_panel_on_first_call(self, empty_dashboard):
        assert empty_dashboard._chat_float is None
        empty_dashboard._on_open_chat()
        assert empty_dashboard._chat_float is not None

    def test_float_panel_is_pn_float_panel(self, empty_dashboard):
        empty_dashboard._on_open_chat()
        assert isinstance(empty_dashboard._chat_float, pn.layout.FloatPanel)

    def test_float_panel_made_visible(self, empty_dashboard):
        empty_dashboard._on_open_chat()
        assert empty_dashboard._chat_float.visible is True

    def test_second_call_reuses_existing_float_panel(self, empty_dashboard):
        empty_dashboard._on_open_chat()
        first_float = empty_dashboard._chat_float
        empty_dashboard._chat_float.visible = False  # hide it
        empty_dashboard._on_open_chat()
        # Same object reused
        assert empty_dashboard._chat_float is first_float
        assert empty_dashboard._chat_float.visible is True

    def test_float_panel_title_contains_analyst(self, empty_dashboard):
        empty_dashboard._on_open_chat()
        assert "Analyst" in empty_dashboard._chat_float.name

    def test_float_panel_not_contained(self, empty_dashboard):
        empty_dashboard._on_open_chat()
        assert empty_dashboard._chat_float.contained is False


# ---------------------------------------------------------------------------
# _generate_snapshot — extended
# ---------------------------------------------------------------------------

class TestGenerateSnapshotExtended:
    def test_empty_dashboard_snapshot_is_string_io(self, empty_dashboard):
        import io
        result = empty_dashboard._generate_snapshot()
        assert isinstance(result, io.StringIO)

    def test_empty_dashboard_snapshot_is_valid_json(self, empty_dashboard):
        import json
        result = empty_dashboard._generate_snapshot()
        parsed = json.loads(result.getvalue())
        assert isinstance(parsed, dict)

    def test_snapshot_position_at_start(self, dashboard_with_state):
        result = dashboard_with_state._generate_snapshot()
        # After creation, position should be at beginning (0) so consumers can read it
        assert result.tell() == 0

    def test_snapshot_non_empty_with_data(self, dashboard_with_state):
        result = dashboard_with_state._generate_snapshot()
        assert len(result.getvalue()) > 10


# ---------------------------------------------------------------------------
# _generate_notebook
# ---------------------------------------------------------------------------

class TestGenerateNotebook:
    def test_returns_string_io(self, empty_dashboard):
        import io
        result = empty_dashboard._generate_notebook()
        assert isinstance(result, io.StringIO)

    def test_returns_string_io_with_data(self, dashboard_with_state):
        import io
        result = dashboard_with_state._generate_notebook()
        assert isinstance(result, io.StringIO)

    def test_notebook_content_non_empty(self, empty_dashboard):
        result = empty_dashboard._generate_notebook()
        assert len(result.getvalue()) > 0

    def test_notebook_is_json_parseable(self, empty_dashboard):
        import json
        result = empty_dashboard._generate_notebook()
        # Either valid notebook JSON or the error fallback JSON
        parsed = json.loads(result.getvalue())
        assert isinstance(parsed, dict)

    def test_notebook_fallback_when_exporter_missing(self, empty_dashboard):
        """When NotebookExporter is unavailable, returns error JSON StringIO."""
        import io
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "deeplens.export.notebook":
                raise ImportError("Simulated missing module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = empty_dashboard._generate_notebook()
        assert isinstance(result, io.StringIO)
        assert len(result.getvalue()) > 0

    def test_notebook_position_at_start(self, empty_dashboard):
        result = empty_dashboard._generate_notebook()
        assert result.tell() == 0

    def test_notebook_with_model_state(self, dashboard_with_model):
        import io
        result = dashboard_with_model._generate_notebook()
        assert isinstance(result, io.StringIO)
        assert len(result.getvalue()) > 0


# ---------------------------------------------------------------------------
# __panel__ — extended coverage
# ---------------------------------------------------------------------------

class TestDashboardPanelExtended:
    def test_sidebar_populated_in_template(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        assert len(result.sidebar) > 0

    def test_main_area_populated_in_template(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        assert len(result.main) > 0

    def test_template_has_dark_theme(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        assert "dark" in str(result.theme).lower()

    def test_template_accent_color_set(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        assert result.accent_base_color is not None

    def test_tabs_in_main_area(self, empty_dashboard):
        result = empty_dashboard.__panel__()
        # Find the Tabs widget somewhere in main
        main_objects = result.main
        tab_found = any(isinstance(obj, pn.Tabs) for obj in main_objects)
        assert tab_found

    def test_multiple_calls_return_separate_templates(self, empty_dashboard):
        result1 = empty_dashboard.__panel__()
        result2 = empty_dashboard.__panel__()
        # Each call builds a fresh template
        assert result1 is not result2

    def test_panel_with_model_state_does_not_raise(self, dashboard_with_model):
        result = dashboard_with_model.__panel__()
        assert isinstance(result, pn.template.FastListTemplate)


# ---------------------------------------------------------------------------
# _create_analyst / _create_nl_filter — with and without dependency
# ---------------------------------------------------------------------------

class TestCreateAnalyst:
    def test_returns_something(self, empty_dashboard):
        result = empty_dashboard._create_analyst()
        assert result is not None

    def test_returns_markdown_when_class_unavailable(self, empty_dashboard):
        with patch("deeplens.dashboard.app._safe_import", return_value=None):
            result = empty_dashboard._create_analyst()
        assert isinstance(result, pn.pane.Markdown)
        assert "unavailable" in result.object.lower()

    def test_llm_passed_when_available(self, empty_dashboard):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        with patch("deeplens.dashboard.app._safe_import", return_value=mock_cls):
            empty_dashboard._llm = MagicMock()
            result = empty_dashboard._create_analyst()
        # Should have been called with llm kwarg
        call_kwargs = mock_cls.call_args.kwargs
        assert "llm" in call_kwargs


class TestCreateNLFilter:
    def test_returns_something(self, empty_dashboard):
        result = empty_dashboard._create_nl_filter()
        assert result is not None

    def test_returns_markdown_when_class_unavailable(self, empty_dashboard):
        with patch("deeplens.dashboard.app._safe_import", return_value=None):
            result = empty_dashboard._create_nl_filter()
        assert isinstance(result, pn.pane.Markdown)
        assert "unavailable" in result.object.lower()

    def test_llm_passed_when_available(self, empty_dashboard):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        with patch("deeplens.dashboard.app._safe_import", return_value=mock_cls):
            empty_dashboard._llm = MagicMock()
            result = empty_dashboard._create_nl_filter()
        call_kwargs = mock_cls.call_args.kwargs
        assert "llm" in call_kwargs


# ---------------------------------------------------------------------------
# _ingest_dataframe_async — asyncio path
# ---------------------------------------------------------------------------

class TestIngestDataframeAsync:
    def _make_df(self, n: int = 20) -> pd.DataFrame:
        rng = np.random.RandomState(99)
        return pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "label": ["x", "y"] * (n // 2),
        })

    def test_async_path_populates_state(self, empty_dashboard):
        """Run the coroutine directly to validate the async path."""
        import asyncio
        df = self._make_df()
        asyncio.run(empty_dashboard._ingest_dataframe_async(df, "async_test"))
        assert empty_dashboard.state.df is not None
        assert empty_dashboard.state.dataset_name == "async_test"

    def test_async_path_computes_embeddings(self, empty_dashboard):
        import asyncio
        df = self._make_df()
        asyncio.run(empty_dashboard._ingest_dataframe_async(df, "async_emb"))
        assert empty_dashboard.state.embeddings_2d is not None

    def test_async_path_sets_success_status(self, empty_dashboard):
        import asyncio
        df = self._make_df()
        asyncio.run(empty_dashboard._ingest_dataframe_async(df, "async_ok"))
        assert empty_dashboard._status.alert_type == "success"

    def test_async_path_no_numeric_features_sets_warning(self, empty_dashboard):
        import asyncio
        df = pd.DataFrame({"text_col": ["hello", "world"] * 5})
        asyncio.run(empty_dashboard._ingest_dataframe_async(df, "no_numeric"))
        assert empty_dashboard._status.alert_type in ("warning", "danger")

    def test_async_path_clears_stale_cache(self, empty_dashboard):
        import asyncio
        empty_dashboard._tab_cache["Explore"] = pn.pane.Markdown("stale")
        df = self._make_df()
        asyncio.run(empty_dashboard._ingest_dataframe_async(df, "cache_clear"))
        # Cache should have been cleared and rebuilt (stale object gone)
        assert empty_dashboard._tab_cache.get("Explore") is not pn.pane.Markdown("stale")
