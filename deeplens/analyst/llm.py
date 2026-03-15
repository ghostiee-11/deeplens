"""LLM provider abstraction — Gemini Flash, Groq, Ollama.

A simple async streaming interface with automatic fallback chain.
No LangChain dependency — direct API calls for minimal overhead.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator

import param


class LLMProvider(param.Parameterized):
    """Base LLM provider with async streaming."""

    provider = param.Selector(
        objects=["gemini", "groq", "ollama", "none"],
        default="none",
    )
    model = param.String(default="")
    _api_key = param.String(default="", precedence=-1)  # hidden from auto-widgets

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        """Yield text chunks from the LLM. Override in subclasses."""
        yield "LLM not configured. Set provider to 'gemini', 'groq', or 'ollama'."


class GeminiProvider(LLMProvider):
    """Google Gemini Flash — free tier: 15 RPM, 1M tokens/day."""

    provider = param.Selector(
        objects=["gemini", "groq", "ollama", "none"],
        default="gemini",
    )
    model = param.String(default="gemini-2.0-flash")

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        try:
            import google.generativeai as genai
        except ImportError:
            yield "Install google-generativeai: `pip install google-generativeai`"
            return

        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            yield "Set GOOGLE_API_KEY environment variable or pass api_key."
            return

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system if system else None,
        )

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [msg["content"]]})

        # Run blocking generate_content + iteration in executor
        import queue
        chunk_queue: queue.Queue = queue.Queue()
        sentinel = object()

        def _generate():
            response = model.generate_content(contents, stream=True)
            for chunk in response:
                if chunk.text:
                    chunk_queue.put(chunk.text)
            chunk_queue.put(sentinel)

        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, _generate)

        while True:
            try:
                item = await loop.run_in_executor(None, chunk_queue.get, True, 0.1)
            except queue.Empty:
                if fut.done():
                    break
                continue
            if item is sentinel:
                break
            yield item
        await fut  # propagate any exceptions


class GroqProvider(LLMProvider):
    """Groq — free tier: 30 RPM, blazing fast inference."""

    provider = param.Selector(
        objects=["gemini", "groq", "ollama", "none"],
        default="groq",
    )
    model = param.String(default="llama-3.3-70b-versatile")

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        try:
            from groq import AsyncGroq
        except ImportError:
            yield "Install groq: `pip install groq`"
            return

        api_key = self.api_key or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            yield "Set GROQ_API_KEY environment variable or pass api_key."
            return

        client = AsyncGroq(api_key=api_key)

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = await client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class OllamaProvider(LLMProvider):
    """Ollama — fully local, no API key needed, unlimited."""

    provider = param.Selector(
        objects=["gemini", "groq", "ollama", "none"],
        default="ollama",
    )
    model = param.String(default="llama3.2")

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        try:
            import ollama
        except ImportError:
            yield "Install ollama: `pip install ollama`"
            return

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        try:
            client = ollama.AsyncClient()
            response = await client.chat(
                model=self.model,
                messages=all_messages,
                stream=True,
            )
            async for chunk in response:
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text
        except Exception as e:
            yield f"Ollama error (is ollama running?): {e}"


class CompositeLLM(LLMProvider):
    """Fallback chain: tries providers in order until one succeeds."""

    provider = param.Selector(
        objects=["gemini", "groq", "ollama", "none", "composite"],
        default="composite",
    )

    def __init__(self, providers: list[LLMProvider] | None = None, **params):
        super().__init__(**params)
        self._providers = providers or []

    async def stream(self, messages: list[dict], system: str = "") -> AsyncIterator[str]:
        last_error = None
        for provider in self._providers:
            try:
                chunks: list[str] = []
                async for chunk in provider.stream(messages, system):
                    chunks.append(chunk)
                # Check if response looks like an error/setup message (not real LLM output)
                full = "".join(chunks)
                if _is_error_response(full):
                    last_error = full
                    continue  # Try next provider
                # Real response — yield it
                for chunk in chunks:
                    yield chunk
                return
            except Exception as e:
                last_error = str(e)
                continue

        yield last_error or "All LLM providers failed. Check your API keys or install ollama."


def _is_error_response(text: str) -> bool:
    """Detect error/setup messages from providers (not real LLM output)."""
    error_patterns = [
        "Install ",
        "Set GOOGLE_API_KEY",
        "Set GROQ_API_KEY",
        "LLM not configured",
        "Ollama error",
        "pip install",
    ]
    return any(pattern in text for pattern in error_patterns)


def sanitize_expression(expression: str, allowed_columns: list[str]) -> str:
    """Sanitize a pandas query expression to prevent code injection.

    Only allows: column names, comparison operators, logical operators,
    string/numeric literals, and parentheses.
    """
    # Strip only if the entire expression is wrapped in matching quotes
    expression = expression.strip()
    if len(expression) >= 2:
        if (expression[0] == '"' and expression[-1] == '"') or \
           (expression[0] == "'" and expression[-1] == "'"):
            expression = expression[1:-1]

    # Reject obviously dangerous patterns
    dangerous = [
        "__", "import", "exec", "eval", "compile", "open(",
        "os.", "sys.", "subprocess", "lambda", "def ", "class ",
        ".apply(", ".map(", ".pipe(", "globals", "locals",
        "getattr", "setattr", "delattr", "__builtins__",
    ]
    expr_lower = expression.lower()
    for pattern in dangerous:
        if pattern.lower() in expr_lower:
            raise ValueError(f"Rejected unsafe expression: contains '{pattern}'")

    # Block @ which allows pandas query to reference local variables
    if "@" in expression:
        raise ValueError("Rejected expression: '@' variable references not allowed")

    # Only allow known safe tokens (no @)
    safe_pattern = re.compile(
        r"""^[\s\w\.\+\-\*/<>=!&|~()'",.]+$"""
    )
    if not safe_pattern.match(expression):
        raise ValueError(f"Rejected expression with unexpected characters: {expression}")

    return expression


def create_llm(provider: str = "gemini", api_key: str = "", model: str = "") -> LLMProvider:
    """Factory function to create an LLM provider."""
    providers = {
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "ollama": OllamaProvider,
        "none": LLMProvider,
    }
    cls = providers.get(provider, LLMProvider)
    kwargs: dict = {}
    if api_key:
        kwargs["_api_key"] = api_key
    if model:
        kwargs["model"] = model
    return cls(**kwargs)
