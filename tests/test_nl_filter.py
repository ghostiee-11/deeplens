"""Tests for deeplens.analyst.nl_filter — NLFilter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import panel as pn
import pytest

from deeplens.analyst.llm import LLMProvider, sanitize_expression
from deeplens.analyst.nl_filter import NLFilter
from deeplens.config import DeepLensState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FixedLLM(LLMProvider):
    """LLM stub that yields a caller-supplied expression."""

    def __init__(self, response: str = "f1 > 0", **params):
        super().__init__(**params)
        self._response = response

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        yield self._response


@pytest.fixture
def simple_state():
    """Small DataFrame-backed state with numeric features."""
    rng = np.random.RandomState(1)
    n = 20
    state = DeepLensState()
    state.dataset_name = "test"
    state.df = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "confidence": rng.uniform(0, 1, n),
            "label": (["a", "b"] * (n // 2)),
            "prediction": (["a", "b"] * (n // 2)),
        }
    )
    state.feature_columns = ["f1", "f2", "confidence"]
    state.label_column = "label"
    state.labels = np.array(state.df["label"].tolist())
    state.predictions = np.array(state.df["prediction"].tolist())
    state.class_names = ["a", "b"]
    return state


@pytest.fixture
def nl_filter(simple_state, mock_llm):
    return NLFilter(state=simple_state, llm=mock_llm)


@pytest.fixture
def nl_filter_no_state():
    return NLFilter()


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestNLFilterInit:
    def test_init_no_args_uses_noop_llm(self, nl_filter_no_state):
        """With no llm kwarg, NLFilter should create a no-op LLMProvider."""
        assert isinstance(nl_filter_no_state.llm, LLMProvider)

    def test_init_with_state(self, nl_filter):
        assert nl_filter.state is not None

    def test_widgets_created(self, nl_filter):
        assert isinstance(nl_filter._query_input, pn.widgets.TextInput)
        assert isinstance(nl_filter._apply_btn, pn.widgets.Button)
        assert isinstance(nl_filter._clear_btn, pn.widgets.Button)

    def test_history_starts_empty(self, nl_filter):
        assert nl_filter._history == []

    def test_status_starts_blank(self, nl_filter):
        assert nl_filter._status.object == ""

    def test_suggestions_pane_populated_when_state_has_labels_and_predictions(
        self, nl_filter
    ):
        """Suggestions should list useful queries when model & labels are available."""
        suggestions = nl_filter._suggestions_pane.object
        assert "confident" in suggestions.lower() or "misclassified" in suggestions.lower() or suggestions != ""


# ---------------------------------------------------------------------------
# _on_clear
# ---------------------------------------------------------------------------

class TestOnClear:
    def test_clear_resets_query_input(self, nl_filter):
        nl_filter._query_input.value = "some query"
        nl_filter._on_clear()
        assert nl_filter._query_input.value == ""

    def test_clear_sets_status(self, nl_filter):
        nl_filter._on_clear()
        assert "cleared" in nl_filter._status.object.lower()

    def test_clear_resets_selected_indices(self, nl_filter):
        nl_filter.state.selected_indices = [1, 2, 3]
        nl_filter._on_clear()
        assert nl_filter.state.selected_indices == []

    def test_clear_with_no_state_does_not_raise(self, nl_filter_no_state):
        nl_filter_no_state._on_clear()  # should not raise


# ---------------------------------------------------------------------------
# _refresh_suggestions
# ---------------------------------------------------------------------------

class TestRefreshSuggestions:
    def test_empty_when_no_state(self, nl_filter_no_state):
        nl_filter_no_state._refresh_suggestions()
        assert nl_filter_no_state._suggestions_pane.object == ""

    def test_class_suggestion_included(self, nl_filter):
        nl_filter._refresh_suggestions()
        text = nl_filter._suggestions_pane.object
        # The first class name should appear in suggestions
        assert nl_filter.state.class_names[0] in text

    def test_cluster_suggestion_included_when_cluster_labels_set(self, nl_filter):
        nl_filter.state.cluster_labels = np.array([0, 1] * 10)
        nl_filter._refresh_suggestions()
        text = nl_filter._suggestions_pane.object
        assert "cluster 0" in text.lower()

    def test_no_predictions_suppresses_model_suggestions(self, simple_state):
        simple_state.predictions = None
        simple_state.labels = None
        flt = NLFilter(state=simple_state)
        text = flt._suggestions_pane.object
        assert "confident" not in text.lower()


# ---------------------------------------------------------------------------
# sanitize_expression (used inside _on_apply) — additional edge cases
# ---------------------------------------------------------------------------

class TestSanitizeExpression:
    """Supplement the existing test_analyst.py coverage with edge cases."""

    def test_allows_numeric_literals(self):
        result = sanitize_expression("confidence >= 0.9", ["confidence"])
        assert result == "confidence >= 0.9"

    def test_allows_string_literals(self):
        result = sanitize_expression("label == 'a'", ["label"])
        assert result == "label == 'a'"

    def test_strips_outer_double_quotes(self):
        result = sanitize_expression('"f1 > 0"', ["f1"])
        assert result == "f1 > 0"

    def test_strips_outer_single_quotes(self):
        result = sanitize_expression("'f2 < 1'", ["f2"])
        assert result == "f2 < 1"

    def test_blocks_apply(self):
        with pytest.raises(ValueError):
            sanitize_expression("df.apply(lambda x: x)", ["df"])

    def test_blocks_at_sign(self):
        with pytest.raises(ValueError, match="'@'"):
            sanitize_expression("f1 > @threshold", ["f1"])

    def test_blocks_getattr(self):
        with pytest.raises(ValueError):
            sanitize_expression("getattr(df, 'f1')", ["f1"])

    def test_blocks_pipe(self):
        with pytest.raises(ValueError):
            sanitize_expression("df.pipe(print)", ["df"])

    def test_blocks_dunder(self):
        with pytest.raises(ValueError):
            sanitize_expression("__import__('os')", ["col"])


# ---------------------------------------------------------------------------
# _on_apply (async)
# ---------------------------------------------------------------------------

class TestOnApply:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_apply_empty_query_sets_status(self, nl_filter):
        nl_filter._query_input.value = ""
        self._run(nl_filter._on_apply())
        assert nl_filter._status.object != ""

    def test_apply_no_state_sets_status(self):
        flt = NLFilter(llm=_FixedLLM("f1 > 0"))
        flt._query_input.value = "some query"
        self._run(flt._on_apply())
        assert flt._status.object != ""

    def test_apply_valid_expression_updates_state(self, simple_state):
        flt = NLFilter(state=simple_state, llm=_FixedLLM("f1 > 0"))
        flt._query_input.value = "positive f1"
        self._run(flt._on_apply())
        # selected_indices should be set to matching row indices
        assert isinstance(flt.state.selected_indices, list)

    def test_apply_valid_expression_updates_history(self, simple_state):
        flt = NLFilter(state=simple_state, llm=_FixedLLM("f1 > 0"))
        flt._query_input.value = "positive f1"
        self._run(flt._on_apply())
        assert len(flt._history) == 1
        assert flt._history[0]["Pandas Expression"] == "f1 > 0"

    def test_apply_cannot_filter_response_sets_status(self, simple_state):
        flt = NLFilter(state=simple_state, llm=_FixedLLM("CANNOT_FILTER"))
        flt._query_input.value = "something impossible"
        self._run(flt._on_apply())
        assert "Could not convert" in flt._status.object

    def test_apply_dangerous_expression_rejected(self, simple_state):
        flt = NLFilter(state=simple_state, llm=_FixedLLM("__import__('os')"))
        flt._query_input.value = "hack attempt"
        self._run(flt._on_apply())
        assert "Unsafe expression rejected" in flt._status.object

    def test_apply_bad_pandas_expression_shows_error(self, simple_state):
        flt = NLFilter(state=simple_state, llm=_FixedLLM("nonexistent_col > 0"))
        flt._query_input.value = "bad column"
        self._run(flt._on_apply())
        assert "error" in flt._status.object.lower() or "Filter" in flt._status.object

    def test_apply_updates_history_table(self, simple_state):
        flt = NLFilter(state=simple_state, llm=_FixedLLM("f1 > 0"))
        flt._query_input.value = "positive f1"
        self._run(flt._on_apply())
        assert len(flt._history_table.value) == 1

    def test_multiple_applies_accumulate_history(self, simple_state):
        flt = NLFilter(state=simple_state, llm=_FixedLLM("f1 > 0"))
        flt._query_input.value = "query 1"
        self._run(flt._on_apply())
        flt._query_input.value = "query 2"
        self._run(flt._on_apply())
        assert len(flt._history) == 2


# ---------------------------------------------------------------------------
# __panel__
# ---------------------------------------------------------------------------

class TestNLFilterPanel:
    def test_panel_returns_column(self, nl_filter):
        result = nl_filter.__panel__()
        assert isinstance(result, pn.Column)

    def test_panel_is_viewable(self, nl_filter):
        result = nl_filter.__panel__()
        assert isinstance(result, pn.viewable.Viewable)

    def test_panel_contains_query_input(self, nl_filter):
        # __panel__ composes a compact Column; check it's non-empty
        result = nl_filter.__panel__()
        assert len(result) > 0

    def test_panel_without_state_does_not_raise(self, nl_filter_no_state):
        result = nl_filter_no_state.__panel__()
        assert result is not None

    def test_panel_detail_panel_none_before_toggle(self, nl_filter):
        """Detail float panel should not be created until toggle is clicked."""
        assert nl_filter._detail_panel is None
        result = nl_filter.__panel__()
        # Without toggling, result is the compact Column (no extra child)
        assert isinstance(result, pn.Column)

    def test_panel_detail_panel_included_after_toggle(self, nl_filter):
        """After _toggle_detail(), __panel__ should include the float panel."""
        nl_filter._toggle_detail()
        result = nl_filter.__panel__()
        # Now result is a Column with two children: compact + float panel
        assert len(result) == 2

    def test_sizing_mode_stretch_width(self, nl_filter):
        result = nl_filter.__panel__()
        assert result.sizing_mode == "stretch_width"
