"""Tests for deeplens.analyst.chat — DeepLensAnalyst."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import panel as pn
import pytest

from deeplens.analyst.chat import DeepLensAnalyst, _MAX_HISTORY
from deeplens.analyst.llm import LLMProvider
from deeplens.config import DeepLensState


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------

class _FixedLLM(LLMProvider):
    """Returns a fixed string without calling any external API."""

    def __init__(self, response: str = "Test AI response.", **params):
        super().__init__(**params)
        self._response = response

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        yield self._response


class _EmptyLLM(LLMProvider):
    """Yields nothing (empty response)."""

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        return
        yield  # pragma: no cover  — makes it an async generator


@pytest.fixture
def full_state():
    """DeepLensState populated with iris-like data."""
    rng = np.random.RandomState(9)
    n = 30
    state = DeepLensState()
    state.dataset_name = "iris"
    state.df = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "label": (["a", "b", "c"] * (n // 3)),
        }
    )
    state.feature_columns = ["f1", "f2"]
    state.label_column = "label"
    state.labels = np.array(state.df["label"].tolist())
    state.class_names = ["a", "b", "c"]
    state.embeddings_2d = rng.randn(n, 2)
    return state


@pytest.fixture
def analyst(full_state):
    return DeepLensAnalyst(state=full_state, llm=_FixedLLM())


@pytest.fixture
def analyst_no_state():
    return DeepLensAnalyst(llm=_FixedLLM())


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestAnalystInit:
    def test_init_no_llm_creates_default(self):
        """When llm kwarg is absent, DeepLensAnalyst should create a no-op LLM."""
        a = DeepLensAnalyst()
        assert isinstance(a.llm, LLMProvider)

    def test_init_with_llm(self, analyst):
        assert isinstance(analyst.llm, _FixedLLM)

    def test_init_with_state(self, analyst, full_state):
        assert analyst.state is full_state

    def test_history_starts_empty(self, analyst):
        assert analyst._history == []

    def test_chat_interface_created(self, analyst):
        assert isinstance(analyst._chat, pn.chat.ChatInterface)


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_returns_string(self, analyst):
        ctx = analyst._build_context()
        assert isinstance(ctx, str)

    def test_includes_dataset_name(self, analyst):
        ctx = analyst._build_context()
        assert "iris" in ctx

    def test_no_state_returns_fallback(self, analyst_no_state):
        ctx = analyst_no_state._build_context()
        assert "No dataset" in ctx

    def test_context_mentions_sample_count(self, analyst):
        ctx = analyst._build_context()
        assert "30" in ctx

    def test_context_includes_class_names(self, analyst):
        ctx = analyst._build_context()
        assert "a" in ctx or "b" in ctx  # class names appear in context

    def test_context_includes_feature_count(self, analyst):
        ctx = analyst._build_context()
        assert "2" in ctx  # 2 features


# ---------------------------------------------------------------------------
# _respond (async streaming callback)
# ---------------------------------------------------------------------------

class TestRespond:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _collect_respond(self, analyst, contents: str) -> list[str]:
        """Exhaust the _respond async generator and return all yielded strings."""
        chunks = []

        async def _collect():
            async for chunk in analyst._respond(contents, "User", analyst._chat):
                chunks.append(chunk)

        self._run(_collect())
        return chunks

    def test_respond_yields_llm_output(self, analyst):
        chunks = self._collect_respond(analyst, "What is this dataset?")
        assert len(chunks) > 0
        assert "Test AI response." in "".join(chunks)

    def test_respond_appends_user_to_history(self, analyst):
        self._collect_respond(analyst, "Hello")
        user_msgs = [m for m in analyst._history if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Hello"

    def test_respond_appends_assistant_to_history(self, analyst):
        self._collect_respond(analyst, "Hello")
        assistant_msgs = [m for m in analyst._history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

    def test_history_trimmed_to_max(self, analyst):
        """After many turns, history must not exceed _MAX_HISTORY entries."""
        for i in range(_MAX_HISTORY + 5):
            self._collect_respond(analyst, f"message {i}")
        assert len(analyst._history) <= _MAX_HISTORY

    def test_respond_with_empty_llm_yields_empty_response(self):
        a = DeepLensAnalyst(llm=_EmptyLLM())
        chunks = []

        async def _collect():
            async for chunk in a._respond("hi", "User", a._chat):
                chunks.append(chunk)

        asyncio.get_event_loop().run_until_complete(_collect())
        # Empty LLM → empty response; history should still be appended
        assert any(m["role"] == "user" for m in a._history)


# ---------------------------------------------------------------------------
# auto_insight
# ---------------------------------------------------------------------------

class TestAutoInsight:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_returns_empty_string_without_selection(self, analyst):
        analyst.state.selected_indices = []
        result = self._run(analyst.auto_insight())
        assert result == ""

    def test_returns_empty_when_no_state(self, analyst_no_state):
        result = self._run(analyst_no_state.auto_insight())
        assert result == ""

    def test_returns_string_with_selection(self, analyst):
        analyst.state.selected_indices = [0, 1, 2]
        result = self._run(analyst.auto_insight())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_llm_output_with_selection(self, analyst):
        analyst.state.selected_indices = [0, 1]
        result = self._run(analyst.auto_insight())
        assert "Test AI response." in result


# ---------------------------------------------------------------------------
# cluster_story
# ---------------------------------------------------------------------------

class TestClusterStory:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_returns_string(self, analyst):
        result = self._run(analyst.cluster_story(0))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_no_data_without_state(self, analyst_no_state):
        result = self._run(analyst_no_state.cluster_story(1))
        assert "No data" in result

    def test_returns_llm_output(self, analyst):
        result = self._run(analyst.cluster_story(2))
        assert "Test AI response." in result

    def test_cluster_id_as_string(self, analyst):
        result = self._run(analyst.cluster_story("cluster_0"))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# __panel__
# ---------------------------------------------------------------------------

class TestAnalystPanel:
    def test_panel_returns_chat_interface(self, analyst):
        result = analyst.__panel__()
        assert isinstance(result, pn.chat.ChatInterface)

    def test_panel_is_same_as_chat_attr(self, analyst):
        result = analyst.__panel__()
        assert result is analyst._chat

    def test_panel_is_viewable(self, analyst):
        result = analyst.__panel__()
        assert isinstance(result, pn.viewable.Viewable)

    def test_panel_without_state_does_not_raise(self, analyst_no_state):
        result = analyst_no_state.__panel__()
        assert result is not None

    def test_panel_sizing_mode(self, analyst):
        result = analyst.__panel__()
        assert result.sizing_mode == "stretch_both"
