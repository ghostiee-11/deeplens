"""Dataset loaders — thin wrappers returning ``pandas.DataFrame``."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def load_sklearn(name: str = "iris") -> pd.DataFrame:
    """Load a scikit-learn toy dataset by name.

    Supported: iris, wine, breast_cancer, digits, diabetes,
    20newsgroups (subset).
    """
    from sklearn import datasets

    if name == "20newsgroups":
        bunch = datasets.fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        return pd.DataFrame({"text": bunch.data, "label": [bunch.target_names[t] for t in bunch.target]})

    loader = getattr(datasets, f"load_{name}", None)
    if loader is None:
        raise ValueError(
            f"Unknown sklearn dataset '{name}'. "
            f"Available: iris, wine, breast_cancer, digits, diabetes, 20newsgroups"
        )
    bunch = loader(as_frame=True)
    df = bunch.frame if hasattr(bunch, "frame") and bunch.frame is not None else bunch.data.copy()
    if "target" not in df.columns and hasattr(bunch, "target"):
        df["target"] = bunch.target
    # Replace integer targets with string names when available
    if hasattr(bunch, "target_names") and "target" in df.columns:
        mapping = {i: n for i, n in enumerate(bunch.target_names)}
        df["label"] = df["target"].map(mapping)
    return df


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def load_huggingface(dataset_id: str, split: str = "train", columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load a HuggingFace dataset as a DataFrame (requires ``datasets``)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' to use HuggingFace datasets: pip install datasets")

    ds = load_dataset(dataset_id, split=split)
    df = ds.to_pandas()
    if columns:
        df = df[list(columns)]
    return df


def load_images(directory: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")) -> pd.DataFrame:
    """Scan *directory* for image files and return a DataFrame with paths + metadata."""
    from pathlib import Path

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    rows: list[dict] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in extensions:
            rows.append({
                "path": str(p),
                "filename": p.name,
                "label": p.parent.name if p.parent != root else "unknown",
            })
    if not rows:
        raise ValueError(f"No images found in {directory} with extensions {extensions}")
    return pd.DataFrame(rows)


def infer_columns(df: pd.DataFrame) -> dict[str, str | list[str]]:
    """Heuristically detect label, text, image, and feature columns."""
    result: dict[str, str | list[str]] = {}

    # Label column — try exact matches first, then case-insensitive, then
    # heuristic: the first low-cardinality non-numeric or integer column that
    # looks like a classification target.
    _LABEL_NAMES = ("label", "target", "class", "y", "category", "survived",
                    "species", "diagnosis", "outcome", "sentiment", "type")
    for col in _LABEL_NAMES:
        if col in df.columns:
            result["label"] = col
            break
    else:
        # Case-insensitive search
        lower_map = {c.lower(): c for c in df.columns}
        for name in _LABEL_NAMES:
            if name in lower_map:
                result["label"] = lower_map[name]
                break
        else:
            # Heuristic: find first column with low cardinality (2-20 unique)
            # that is categorical, string, or integer — likely a label
            for col in df.columns:
                nunique = df[col].nunique()
                if 2 <= nunique <= 20:
                    if (pd.api.types.is_string_dtype(df[col])
                            or pd.api.types.is_categorical_dtype(df[col])
                            or (pd.api.types.is_integer_dtype(df[col])
                                and col not in ("id", "index"))):
                        result["label"] = col
                        break

    # Text column
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 50:
                result["text"] = col
                break

    # Image column
    for col in ("path", "image_path", "image", "file"):
        if col in df.columns:
            result["image"] = col
            break

    # Feature columns (numeric, excluding target)
    label_col = result.get("label", "")
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col and c != "target"]
    result["features"] = features

    return result
