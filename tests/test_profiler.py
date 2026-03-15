"""Tests for deeplens.data.profiler.DatasetProfiler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv
import panel as pn

from deeplens.config import DeepLensState
from deeplens.data.profiler import DatasetProfiler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    """Small DataFrame with numeric features, a label, and some missing values."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "feat_a": rng.randn(40),
            "feat_b": rng.randn(40) * 2 + 1,
            "feat_c": rng.randn(40) * 0.5 - 3,
            "cat_col": ["x", "y"] * 20,
            "label": (["cat", "dog"] * 20),
        }
    )
    # Introduce some missing values
    df.loc[[2, 7, 15], "feat_a"] = np.nan
    df.loc[[5, 18], "feat_b"] = np.nan
    # Introduce duplicates
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    return df


@pytest.fixture
def numeric_features():
    return ["feat_a", "feat_b", "feat_c"]


@pytest.fixture
def standalone_profiler(simple_df, numeric_features):
    """Profiler instantiated in standalone mode (no state)."""
    return DatasetProfiler(
        df=simple_df,
        feature_columns=numeric_features,
        label_column="label",
    )


@pytest.fixture
def state_profiler(simple_df, numeric_features):
    """Profiler driven from a DeepLensState instance."""
    state = DeepLensState()
    state.df = simple_df
    state.feature_columns = numeric_features
    state.label_column = "label"
    return DatasetProfiler(state=state)


@pytest.fixture
def empty_profiler():
    """Profiler with no data attached."""
    return DatasetProfiler()


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_standalone_instantiation(self, standalone_profiler):
        assert isinstance(standalone_profiler, DatasetProfiler)
        assert isinstance(standalone_profiler, pn.viewable.Viewer)

    def test_state_instantiation(self, state_profiler):
        assert isinstance(state_profiler, DatasetProfiler)

    def test_empty_instantiation(self, empty_profiler):
        """Profiler should be constructable without any arguments."""
        assert empty_profiler.df is None
        assert empty_profiler.state is None


# ---------------------------------------------------------------------------
# 2. _resolve_df / _resolve_features / _resolve_label
# ---------------------------------------------------------------------------


class TestResolvers:
    def test_resolve_df_standalone(self, standalone_profiler, simple_df):
        resolved = standalone_profiler._resolve_df()
        assert resolved is not None
        assert len(resolved) == len(simple_df)

    def test_resolve_df_from_state(self, state_profiler, simple_df):
        resolved = state_profiler._resolve_df()
        assert resolved is not None
        assert len(resolved) == len(simple_df)

    def test_resolve_df_empty(self, empty_profiler):
        assert empty_profiler._resolve_df() is None

    def test_resolve_features_standalone(self, standalone_profiler, numeric_features):
        assert standalone_profiler._resolve_features() == numeric_features

    def test_resolve_features_from_state(self, state_profiler, numeric_features):
        assert state_profiler._resolve_features() == numeric_features

    def test_resolve_label_standalone(self, standalone_profiler):
        assert standalone_profiler._resolve_label() == "label"

    def test_resolve_label_from_state(self, state_profiler):
        assert state_profiler._resolve_label() == "label"

    def test_resolve_label_empty(self, empty_profiler):
        assert empty_profiler._resolve_label() == ""


# ---------------------------------------------------------------------------
# 3. Overview card
# ---------------------------------------------------------------------------


class TestOverviewCard:
    def test_returns_row_with_data(self, standalone_profiler):
        result = standalone_profiler._overview_card()
        assert isinstance(result, pn.Row)

    def test_returns_markdown_when_empty(self, empty_profiler):
        result = empty_profiler._overview_card()
        assert isinstance(result, pn.pane.Markdown)

    def test_overview_indicators_count(self, standalone_profiler):
        row = standalone_profiler._overview_card()
        # Should have 5 Number indicators (samples, features, missing%, dupes, memory)
        numbers = [obj for obj in row.objects if isinstance(obj, pn.indicators.Number)]
        assert len(numbers) == 5

    def test_sample_count_correct(self, standalone_profiler, simple_df):
        row = standalone_profiler._overview_card()
        samples_indicator = row.objects[0]
        assert samples_indicator.value == len(simple_df)

    def test_missing_pct_positive(self, standalone_profiler, simple_df):
        row = standalone_profiler._overview_card()
        missing_indicator = row.objects[2]
        assert missing_indicator.value > 0

    def test_duplicate_count(self, standalone_profiler, simple_df):
        row = standalone_profiler._overview_card()
        dup_indicator = row.objects[3]
        expected_dupes = int(simple_df.duplicated().sum())
        assert dup_indicator.value == expected_dupes


# ---------------------------------------------------------------------------
# 4. Missing-values heatmap
# ---------------------------------------------------------------------------


class TestMissingHeatmap:
    def test_returns_heatmap_with_data(self, standalone_profiler):
        result = standalone_profiler._missing_heatmap()
        assert isinstance(result, hv.HeatMap)

    def test_returns_placeholder_when_empty(self, empty_profiler):
        result = empty_profiler._missing_heatmap()
        assert isinstance(result, hv.Text)

    def test_heatmap_vdim_is_missing_pct(self, standalone_profiler):
        heatmap = standalone_profiler._missing_heatmap()
        vdim_names = [str(v) for v in heatmap.vdims]
        assert "Missing %" in vdim_names

    def test_heatmap_has_all_columns(self, standalone_profiler, simple_df):
        heatmap = standalone_profiler._missing_heatmap()
        # Extract unique "Column" values from the heatmap data
        data = heatmap.data
        if hasattr(data, "Column"):
            cols_in_heatmap = set(data["Column"].unique())
        else:
            cols_in_heatmap = set(np.unique(data[:, 1]))
        # Every DataFrame column should appear
        for col in simple_df.columns:
            assert col in cols_in_heatmap


# ---------------------------------------------------------------------------
# 5. Correlation matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_returns_heatmap(self, standalone_profiler):
        result = standalone_profiler._correlation_matrix()
        assert isinstance(result, hv.HeatMap)

    def test_placeholder_with_one_feature(self, simple_df):
        profiler = DatasetProfiler(df=simple_df, feature_columns=["feat_a"])
        result = profiler._correlation_matrix()
        assert isinstance(result, hv.Text)

    def test_placeholder_when_empty(self, empty_profiler):
        result = empty_profiler._correlation_matrix()
        assert isinstance(result, hv.Text)

    def test_diagonal_correlation_is_one(self, standalone_profiler, numeric_features):
        heatmap = standalone_profiler._correlation_matrix()
        df_data = heatmap.data
        # Find diagonal entries (X == Y)
        diag_mask = df_data["Column X"] == df_data["Column Y"]
        diag_vals = df_data.loc[diag_mask, "Correlation"]
        assert (diag_vals.round(2) == 1.0).all()

    def test_correlation_within_bounds(self, standalone_profiler):
        heatmap = standalone_profiler._correlation_matrix()
        vals = heatmap.data["Correlation"]
        assert (vals >= -1.001).all() and (vals <= 1.001).all()


# ---------------------------------------------------------------------------
# 6. Class balance
# ---------------------------------------------------------------------------


class TestClassBalance:
    def test_returns_bars_when_label_exists(self, standalone_profiler):
        result = standalone_profiler._class_balance()
        assert isinstance(result, hv.Bars)

    def test_returns_markdown_when_no_label(self, simple_df, numeric_features):
        profiler = DatasetProfiler(df=simple_df, feature_columns=numeric_features)
        result = profiler._class_balance()
        assert isinstance(result, pn.pane.Markdown)

    def test_returns_markdown_when_empty(self, empty_profiler):
        result = empty_profiler._class_balance()
        assert isinstance(result, pn.pane.Markdown)

    def test_bars_contain_all_classes(self, standalone_profiler, simple_df):
        bars = standalone_profiler._class_balance()
        classes_in_chart = set(bars.data["Class"].astype(str))
        expected = {str(c) for c in simple_df["label"].unique()}
        assert expected == classes_in_chart

    def test_counts_sum_to_n_samples(self, standalone_profiler, simple_df):
        bars = standalone_profiler._class_balance()
        total = bars.data["Count"].sum()
        assert total == len(simple_df)


# ---------------------------------------------------------------------------
# 7. Feature distributions
# ---------------------------------------------------------------------------


class TestFeatureDistributions:
    def test_returns_gridbox_with_data(self, standalone_profiler):
        result = standalone_profiler._feature_distributions()
        assert isinstance(result, pn.GridBox)

    def test_returns_markdown_when_empty(self, empty_profiler):
        result = empty_profiler._feature_distributions()
        assert isinstance(result, pn.pane.Markdown)

    def test_one_plot_per_feature(self, standalone_profiler, numeric_features):
        grid = standalone_profiler._feature_distributions()
        # Each feature should produce one HoloViews pane
        hv_panes = [obj for obj in grid.objects if isinstance(obj, pn.pane.HoloViews)]
        assert len(hv_panes) == len(numeric_features)

    def test_no_non_numeric_columns_plotted(self, simple_df):
        """cat_col is a string column and must not appear as a histogram."""
        profiler = DatasetProfiler(
            df=simple_df,
            feature_columns=["cat_col"],  # non-numeric
        )
        result = profiler._feature_distributions()
        # Should fall back to a Markdown placeholder
        assert isinstance(result, pn.pane.Markdown)


# ---------------------------------------------------------------------------
# 8. Outlier summary
# ---------------------------------------------------------------------------


class TestOutlierSummary:
    def test_returns_dataframe_widget(self, standalone_profiler):
        result = standalone_profiler._outlier_summary()
        assert isinstance(result, pn.widgets.DataFrame)

    def test_returns_markdown_when_empty(self, empty_profiler):
        result = empty_profiler._outlier_summary()
        assert isinstance(result, pn.pane.Markdown)

    def test_outlier_table_has_expected_columns(self, standalone_profiler):
        widget = standalone_profiler._outlier_summary()
        expected = {
            "Column", "Q1", "Q3", "IQR",
            "Lower fence", "Upper fence",
            "Outlier count", "Outlier %",
        }
        assert expected == set(widget.value.columns)

    def test_one_row_per_numeric_feature(self, standalone_profiler, numeric_features):
        widget = standalone_profiler._outlier_summary()
        assert len(widget.value) == len(numeric_features)

    def test_outlier_count_non_negative(self, standalone_profiler):
        widget = standalone_profiler._outlier_summary()
        assert (widget.value["Outlier count"] >= 0).all()

    def test_outlier_pct_between_0_and_100(self, standalone_profiler):
        widget = standalone_profiler._outlier_summary()
        assert (widget.value["Outlier %"] >= 0).all()
        assert (widget.value["Outlier %"] <= 100).all()

    def test_known_outlier_detected(self):
        """Inject a clear outlier and confirm it is counted."""
        rng = np.random.RandomState(0)
        data = rng.randn(100)
        # Append a clear outlier far beyond IQR bounds
        data = np.append(data, [500.0, -500.0])
        df = pd.DataFrame({"x": data})
        profiler = DatasetProfiler(df=df, feature_columns=["x"])
        widget = profiler._outlier_summary()
        row = widget.value[widget.value["Column"] == "x"].iloc[0]
        assert row["Outlier count"] >= 2


# ---------------------------------------------------------------------------
# 9. Data-types summary
# ---------------------------------------------------------------------------


class TestDtypeSummary:
    def test_returns_dataframe_widget(self, standalone_profiler):
        result = standalone_profiler._dtype_summary()
        assert isinstance(result, pn.widgets.DataFrame)

    def test_returns_markdown_when_empty(self, empty_profiler):
        result = empty_profiler._dtype_summary()
        assert isinstance(result, pn.pane.Markdown)

    def test_one_row_per_column(self, standalone_profiler, simple_df):
        widget = standalone_profiler._dtype_summary()
        assert len(widget.value) == len(simple_df.columns)

    def test_expected_columns_present(self, standalone_profiler):
        widget = standalone_profiler._dtype_summary()
        expected = {"Column", "Dtype", "Unique values", "Missing count", "Missing %"}
        assert expected == set(widget.value.columns)

    def test_missing_count_matches_df(self, standalone_profiler, simple_df):
        widget = standalone_profiler._dtype_summary()
        for _, row in widget.value.iterrows():
            col = row["Column"]
            expected_missing = int(simple_df[col].isnull().sum())
            assert row["Missing count"] == expected_missing


# ---------------------------------------------------------------------------
# 10. __panel__ layout
# ---------------------------------------------------------------------------


class TestPanelLayout:
    def test_panel_returns_column(self, standalone_profiler):
        layout = standalone_profiler.__panel__()
        assert isinstance(layout, pn.Column)

    def test_panel_contains_header(self, standalone_profiler):
        layout = standalone_profiler.__panel__()
        # Panel may store plain strings as-is or convert them to Markdown panes
        header_found = False
        for obj in layout.objects:
            if isinstance(obj, str) and "Dataset Profiler" in obj:
                header_found = True
                break
            if isinstance(obj, pn.pane.Markdown) and "Dataset Profiler" in obj.object:
                header_found = True
                break
        assert header_found

    def test_panel_state_driven(self, state_profiler):
        """Profiler driven from state should also render without error."""
        layout = state_profiler.__panel__()
        assert isinstance(layout, pn.Column)

    def test_panel_empty_no_error(self, empty_profiler):
        """Rendering with no data should not raise."""
        layout = empty_profiler.__panel__()
        assert isinstance(layout, pn.Column)


# ---------------------------------------------------------------------------
# 11. State reactivity
# ---------------------------------------------------------------------------


class TestStateReactivity:
    def test_profiler_updates_when_state_df_changes(self):
        """Assigning a new df to state should change what _resolve_df returns."""
        state = DeepLensState()
        profiler = DatasetProfiler(state=state)

        # Initially empty
        assert profiler._resolve_df() is None

        # Assign df to state
        rng = np.random.RandomState(1)
        new_df = pd.DataFrame({"a": rng.randn(10), "b": rng.randn(10)})
        state.df = new_df

        resolved = profiler._resolve_df()
        assert resolved is not None
        assert len(resolved) == 10

    def test_standalone_takes_precedence_over_empty_state(self):
        """When state has no df but standalone df is set, use standalone df."""
        state = DeepLensState()  # no df
        rng = np.random.RandomState(2)
        standalone_df = pd.DataFrame({"x": rng.randn(5)})

        profiler = DatasetProfiler(state=state, df=standalone_df)
        # state.df is None so _resolve_df falls through to self.df
        resolved = profiler._resolve_df()
        assert resolved is not None
        assert len(resolved) == 5


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_missing_column(self):
        """A column that is entirely NaN should not cause errors."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
        profiler = DatasetProfiler(df=df, feature_columns=["a", "b"])
        # Missing heatmap should render without crashing
        result = profiler._missing_heatmap()
        assert isinstance(result, hv.HeatMap)

    def test_single_class_label(self):
        """A label column with only one class should still render a bar chart."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["only"] * 3})
        profiler = DatasetProfiler(df=df, feature_columns=["x"], label_column="label")
        result = profiler._class_balance()
        assert isinstance(result, hv.Bars)
        assert len(result.data) == 1

    def test_no_numeric_features_outlier(self):
        """Outlier summary with zero numeric features returns Markdown."""
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        profiler = DatasetProfiler(df=df, feature_columns=[])
        result = profiler._outlier_summary()
        assert isinstance(result, pn.pane.Markdown)

    def test_large_dataframe_performance(self):
        """Profiler should handle a reasonably large DataFrame without hanging."""
        rng = np.random.RandomState(99)
        n = 5_000
        df = pd.DataFrame(
            {f"f{i}": rng.randn(n) for i in range(10)}
        )
        df["label"] = rng.choice(["a", "b", "c"], size=n)
        profiler = DatasetProfiler(
            df=df,
            feature_columns=[f"f{i}" for i in range(10)],
            label_column="label",
        )
        # These must complete in reasonable time
        assert isinstance(profiler._overview_card(), pn.Row)
        assert isinstance(profiler._outlier_summary(), pn.widgets.DataFrame)
        assert isinstance(profiler._correlation_matrix(), hv.HeatMap)
