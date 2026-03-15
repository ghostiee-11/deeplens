"""CLI entry point: ``python -m deeplens``."""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="deeplens",
        description="DeepLens — Interactive AI Model Interpretability & Embedding Explorer",
    )
    parser.add_argument(
        "--dataset",
        default="iris",
        choices=["iris", "wine", "digits", "breast_cancer"],
        help="Sklearn dataset to load (default: iris)",
    )
    parser.add_argument(
        "--llm",
        default="none",
        choices=["gemini", "groq", "ollama", "none"],
        help="LLM provider for the AI analyst (default: none)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (0 = auto-select)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open a browser tab automatically",
    )

    args = parser.parse_args()

    from deeplens.dashboard.app import launch

    launch(
        dataset=args.dataset,
        llm_provider=args.llm,
        show=not args.no_browser,
        port=args.port,
    )


if __name__ == "__main__":
    main()
