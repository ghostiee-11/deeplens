"""Tests for deeplens.analyst.llm."""

from __future__ import annotations

import asyncio

import pytest

from deeplens.analyst.llm import (
    LLMProvider,
    CompositeLLM,
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    create_llm,
    sanitize_expression,
    _is_error_response,
)


class TestSanitizeExpression:
    def test_safe_comparison(self):
        result = sanitize_expression("age > 30", ["age", "income"])
        assert result == "age > 30"

    def test_safe_equality(self):
        result = sanitize_expression("label == 'setosa'", ["label"])
        assert result == "label == 'setosa'"

    def test_safe_logical_and(self):
        result = sanitize_expression("f1 > 0 and f2 < 10", ["f1", "f2"])
        assert result == "f1 > 0 and f2 < 10"

    def test_safe_numeric_comparison(self):
        result = sanitize_expression("score >= 0.5", ["score"])
        assert result == "score >= 0.5"

    def test_strips_wrapping_quotes(self):
        result = sanitize_expression('"age > 30"', ["age"])
        assert result == "age > 30"

    def test_blocks_import(self):
        with pytest.raises(ValueError, match="Rejected unsafe expression"):
            sanitize_expression("import os", ["col1"])

    def test_blocks_exec(self):
        with pytest.raises(ValueError, match="Rejected unsafe expression"):
            sanitize_expression("exec('print(1)')", ["col1"])

    def test_blocks_eval(self):
        with pytest.raises(ValueError, match="Rejected unsafe expression"):
            sanitize_expression("eval('1+1')", ["col1"])

    def test_blocks_dunder(self):
        with pytest.raises(ValueError, match="Rejected unsafe expression"):
            sanitize_expression("__builtins__", ["col1"])

    def test_blocks_os_dot(self):
        with pytest.raises(ValueError, match="Rejected unsafe expression"):
            sanitize_expression("os.system('rm -rf /')", ["col1"])

    def test_blocks_subprocess(self):
        with pytest.raises(ValueError, match="Rejected unsafe expression"):
            sanitize_expression("subprocess.call(['ls'])", ["col1"])

    def test_blocks_lambda(self):
        with pytest.raises(ValueError, match="Rejected unsafe expression"):
            sanitize_expression("lambda x: x", ["col1"])


class TestIsErrorResponse:
    def test_detects_install_message(self):
        assert _is_error_response("Install google-generativeai: pip install ...")

    def test_detects_api_key_message(self):
        assert _is_error_response("Set GOOGLE_API_KEY environment variable")

    def test_detects_not_configured(self):
        assert _is_error_response("LLM not configured. Set provider.")

    def test_normal_text_not_error(self):
        assert not _is_error_response("The model accuracy is 0.95.")

    def test_empty_string_not_error(self):
        assert not _is_error_response("")


class TestCreateLLMFactory:
    def test_create_gemini(self):
        llm = create_llm("gemini", api_key="test-key")
        assert isinstance(llm, GeminiProvider)

    def test_create_groq(self):
        llm = create_llm("groq")
        assert isinstance(llm, GroqProvider)

    def test_create_ollama(self):
        llm = create_llm("ollama")
        assert isinstance(llm, OllamaProvider)

    def test_create_none(self):
        llm = create_llm("none")
        assert isinstance(llm, LLMProvider)
        assert not isinstance(llm, GeminiProvider)

    def test_create_with_model(self):
        llm = create_llm("gemini", model="gemini-1.5-pro")
        assert llm.model == "gemini-1.5-pro"

    def test_create_unknown_defaults_to_base(self):
        llm = create_llm("unknown_provider")
        assert isinstance(llm, LLMProvider)


class TestLLMProviderBase:
    def test_base_provider_yields_message(self):
        provider = LLMProvider()
        chunks = []

        async def collect():
            async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        asyncio.get_event_loop().run_until_complete(collect())
        assert len(chunks) == 1
        assert "not configured" in chunks[0].lower()


class TestCompositeLLM:
    def test_composite_falls_through_error_providers(self):
        """CompositeLLM should skip providers whose output looks like errors."""
        p1 = LLMProvider()  # yields "LLM not configured..."
        p2 = LLMProvider()  # same

        composite = CompositeLLM(providers=[p1, p2])
        chunks = []

        async def collect():
            async for chunk in composite.stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        asyncio.get_event_loop().run_until_complete(collect())
        full = "".join(chunks)
        # Both providers fail, so we get the last error
        assert len(full) > 0

    def test_composite_uses_first_working_provider(self, mock_llm):
        """If the first real provider works, use it."""
        composite = CompositeLLM(providers=[mock_llm])
        chunks = []

        async def collect():
            async for chunk in composite.stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        asyncio.get_event_loop().run_until_complete(collect())
        assert "mock LLM response" in "".join(chunks)
