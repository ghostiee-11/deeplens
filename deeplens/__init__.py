"""DeepLens — Interactive AI Model Interpretability & Embedding Explorer.

Powered by the HoloViz ecosystem (Panel, HoloViews, Datashader, hvPlot, Lumen).

Quick start::

    import deeplens
    deeplens.explore("iris")
    deeplens.explore(df, text_col="review", label_col="sentiment")
"""

from __future__ import annotations

import numpy as np

__version__ = "0.1.0"

# Convenience re-exports
from deeplens.config import DeepLensState  # noqa: F401


def explore(
    data,
    text_col: str | None = None,
    label_col: str | None = None,
    image_col: str | None = None,
    embedding_method: str = "tfidf",
    reduction_method: str = "pca",
    show: bool = True,
):
    """One-liner: open the Embedding Explorer on any DataFrame or named dataset.

    Parameters
    ----------
    data : DataFrame or str
        A pandas DataFrame, or a sklearn dataset name
        ('iris', 'wine', 'digits', 'breast_cancer').
    text_col : str, optional
        Column containing text data for embedding.
    label_col : str, optional
        Column containing labels / classes.
    image_col : str, optional
        Column containing image file paths.
    embedding_method : str
        One of 'tfidf', 'sentence-transformers', 'clip', 'features'.
    reduction_method : str
        One of 'pca', 'umap', 'tsne'.
    show : bool
        If True, open in browser via ``panel.serve``.
    """
    import pandas as pd
    from deeplens.config import DeepLensState
    from deeplens.data.loaders import load_sklearn, infer_columns
    from deeplens.embeddings.compute import EmbeddingComputer
    from deeplens.embeddings.reduce import DimensionalityReducer
    from deeplens.embeddings.explorer import EmbeddingExplorer

    # Accept string dataset names
    if isinstance(data, str):
        df = load_sklearn(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Expected str or pandas DataFrame, got {type(data)}")

    # Auto-detect columns if not provided
    info = infer_columns(df)
    label_col = label_col or info.get("label") or ""
    text_col = text_col or info.get("text") or ""
    image_col = image_col or info.get("image") or ""
    feature_cols = info.get("features", [])

    dataset_name = data if isinstance(data, str) else "custom"

    state = DeepLensState(
        df=df,
        dataset_name=dataset_name,
        label_column=label_col,
        text_column=text_col,
        image_column=image_col,
        feature_columns=feature_cols,
        embedding_method=embedding_method,
        reduction_method=reduction_method,
    )

    # Compute embeddings
    computer = EmbeddingComputer(method=embedding_method)
    state.embeddings_raw = computer.compute(
        df,
        text_col=text_col if text_col else None,
        image_col=image_col if image_col else None,
    )

    # Reduce
    reducer = DimensionalityReducer(method=reduction_method)
    state.embeddings_2d = reducer.reduce(state.embeddings_raw)

    # Labels
    if label_col and label_col in df.columns:
        state.labels = np.asarray(df[label_col])
        state.class_names = sorted(df[label_col].unique().tolist())

    explorer = EmbeddingExplorer(state=state)
    if show:
        explorer.show(title="DeepLens Explorer")
    return explorer


def _compare_fn(model_a, model_b, X, y, feature_names=None, show: bool = True):
    """Compare two models side-by-side with agreement zone visualization."""
    import importlib
    ModelArena = importlib.import_module("deeplens.compare.models").ModelArena

    arena = ModelArena(model_a=model_a, model_b=model_b, X=X, y=y, feature_names=feature_names)
    if show:
        arena.show(title="DeepLens Model Arena")
    return arena


def drift(reference_df, production_df, feature_columns=None, timestamp_col=None, show: bool = True):
    """Detect and visualize data drift between reference and production data."""
    import importlib
    DriftDetector = importlib.import_module("deeplens.compare.drift").DriftDetector

    detector = DriftDetector(
        reference_df=reference_df,
        production_df=production_df,
        feature_columns=feature_columns if feature_columns is not None else [],
        timestamp_col=timestamp_col if timestamp_col is not None else "",
    )
    if show:
        detector.show(title="DeepLens Drift Detector")
    return detector


def _dashboard_fn(df=None, model=None, dataset: str = "iris", show: bool = True):
    """Launch the full DeepLens dashboard with all modules."""
    import importlib
    mod = importlib.import_module("deeplens.dashboard.app")
    DeepLensDashboard = mod.DeepLensDashboard

    app = DeepLensDashboard.create(dataset=dataset)

    # Inject user-provided DataFrame if given
    if df is not None:
        app._ingest_dataframe(df, name="custom")

    # Inject pre-trained model if given
    if model is not None and app.state.df is not None:
        import numpy as np
        app.state.trained_model = model
        if app.state.feature_columns:
            X = app.state.df[app.state.feature_columns].values.astype(np.float64)
            app.state.predictions = model.predict(X)
            if hasattr(model, "predict_proba"):
                app.state.probabilities = model.predict_proba(X)

    if show:
        app.show(title="DeepLens")
    return app


# ---------------------------------------------------------------------------
# Guard against subpackage shadowing
# ---------------------------------------------------------------------------
# Python's import machinery sets ``sys.modules['deeplens'].compare`` and
# ``sys.modules['deeplens'].dashboard`` to their subpackage modules the first
# time ``deeplens.compare.*`` or ``deeplens.dashboard.*`` is imported.  This
# overwrites the callable names defined above.  Pre-importing the subpackages
# here and immediately re-assigning the function names prevents that race.

import deeplens.compare as _compare_pkg  # noqa: E402  (pre-load subpackage)
import deeplens.dashboard as _dashboard_pkg  # noqa: E402  (pre-load subpackage)

# Restore the callables — they were overwritten by the two imports above.
compare = _compare_fn
dashboard = _dashboard_fn
