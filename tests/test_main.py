"""Tests for the CLI entry point in deeplens/__main__.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_main(argv: list[str]):
    """Import and invoke main() with the given sys.argv."""
    with patch.object(sys, "argv", ["deeplens"] + argv):
        from deeplens.__main__ import main
        main()


# ---------------------------------------------------------------------------
# Argument parsing — ensure argparse is wired correctly
# ---------------------------------------------------------------------------

class TestArgParsing:
    """Verify that argparse definitions match the expected interface."""

    def _parse(self, argv: list[str]):
        """Return parsed args without actually launching anything."""
        import argparse

        # Replicate the parser from __main__ so we can test it in isolation.
        parser = argparse.ArgumentParser(prog="deeplens")
        parser.add_argument(
            "--dataset",
            default="iris",
            choices=["iris", "wine", "digits", "breast_cancer"],
        )
        parser.add_argument(
            "--llm",
            default="none",
            choices=["gemini", "groq", "ollama", "none"],
        )
        parser.add_argument("--port", type=int, default=0)
        parser.add_argument("--no-browser", action="store_true")
        return parser.parse_args(argv)

    def test_defaults(self):
        args = self._parse([])
        assert args.dataset == "iris"
        assert args.llm == "none"
        assert args.port == 0
        assert args.no_browser is False

    def test_dataset_iris(self):
        args = self._parse(["--dataset", "iris"])
        assert args.dataset == "iris"

    def test_dataset_wine(self):
        args = self._parse(["--dataset", "wine"])
        assert args.dataset == "wine"

    def test_dataset_digits(self):
        args = self._parse(["--dataset", "digits"])
        assert args.dataset == "digits"

    def test_dataset_breast_cancer(self):
        args = self._parse(["--dataset", "breast_cancer"])
        assert args.dataset == "breast_cancer"

    def test_invalid_dataset_raises_system_exit(self):
        with pytest.raises(SystemExit):
            self._parse(["--dataset", "invalid_ds"])

    def test_llm_gemini(self):
        args = self._parse(["--llm", "gemini"])
        assert args.llm == "gemini"

    def test_llm_groq(self):
        args = self._parse(["--llm", "groq"])
        assert args.llm == "groq"

    def test_llm_ollama(self):
        args = self._parse(["--llm", "ollama"])
        assert args.llm == "ollama"

    def test_llm_none(self):
        args = self._parse(["--llm", "none"])
        assert args.llm == "none"

    def test_invalid_llm_raises_system_exit(self):
        with pytest.raises(SystemExit):
            self._parse(["--llm", "openai"])

    def test_port_flag(self):
        args = self._parse(["--port", "9090"])
        assert args.port == 9090

    def test_port_zero_default(self):
        args = self._parse([])
        assert args.port == 0

    def test_no_browser_flag_sets_true(self):
        args = self._parse(["--no-browser"])
        assert args.no_browser is True

    def test_no_browser_absent_is_false(self):
        args = self._parse([])
        assert args.no_browser is False

    def test_combined_flags(self):
        args = self._parse(["--dataset", "wine", "--port", "8765", "--no-browser"])
        assert args.dataset == "wine"
        assert args.port == 8765
        assert args.no_browser is True


# ---------------------------------------------------------------------------
# main() integration — verify launch() is called with correct kwargs
# ---------------------------------------------------------------------------

class TestMainCallsLaunch:
    """main() should forward parsed args to deeplens.dashboard.app.launch."""

    def test_default_invocation_calls_launch(self):
        with patch("deeplens.dashboard.app.launch") as mock_launch:
            _run_main([])
        mock_launch.assert_called_once_with(
            dataset="iris",
            llm_provider="none",
            show=True,       # not --no-browser → show=True
            port=0,
        )

    def test_no_browser_flag_passes_show_false(self):
        with patch("deeplens.dashboard.app.launch") as mock_launch:
            _run_main(["--no-browser"])
        _, kwargs = mock_launch.call_args
        assert kwargs.get("show") is False or mock_launch.call_args[0][2] is False
        # Accept both positional and keyword
        call_args = mock_launch.call_args
        show_val = call_args.kwargs.get("show", call_args.args[2] if len(call_args.args) > 2 else None)
        assert show_val is False

    def test_port_flag_forwarded(self):
        with patch("deeplens.dashboard.app.launch") as mock_launch:
            _run_main(["--port", "7777"])
        call_args = mock_launch.call_args
        port_val = call_args.kwargs.get("port", call_args.args[3] if len(call_args.args) > 3 else None)
        assert port_val == 7777

    def test_dataset_flag_forwarded(self):
        with patch("deeplens.dashboard.app.launch") as mock_launch:
            _run_main(["--dataset", "wine"])
        call_args = mock_launch.call_args
        ds_val = call_args.kwargs.get("dataset", call_args.args[0] if call_args.args else None)
        assert ds_val == "wine"

    def test_llm_flag_forwarded(self):
        with patch("deeplens.dashboard.app.launch") as mock_launch:
            _run_main(["--llm", "groq"])
        call_args = mock_launch.call_args
        llm_val = call_args.kwargs.get("llm_provider", call_args.args[1] if len(call_args.args) > 1 else None)
        assert llm_val == "groq"

    def test_launch_not_called_on_bad_dataset(self):
        """Invalid choice should raise SystemExit before launch() is called."""
        with patch("deeplens.dashboard.app.launch") as mock_launch:
            with pytest.raises(SystemExit):
                _run_main(["--dataset", "bogus"])
        mock_launch.assert_not_called()

    def test_all_flags_combined(self):
        with patch("deeplens.dashboard.app.launch") as mock_launch:
            _run_main(["--dataset", "digits", "--llm", "ollama", "--port", "5050", "--no-browser"])
        mock_launch.assert_called_once_with(
            dataset="digits",
            llm_provider="ollama",
            show=False,
            port=5050,
        )
