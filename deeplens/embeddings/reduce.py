"""Dimensionality reduction with live parameter tuning and quality metrics."""

from __future__ import annotations

import numpy as np
import param


class DimensionalityReducer(param.Parameterized):
    """Reduce high-dimensional embeddings to 2-D for visualization.

    All parameters are exposed as ``param`` descriptors so Panel can
    auto-generate widgets for live tuning.
    """

    method = param.Selector(objects=["pca", "tsne", "umap"], default="pca")

    # UMAP params
    n_neighbors = param.Integer(default=15, bounds=(2, 200))
    min_dist = param.Number(default=0.1, bounds=(0.0, 1.0))

    # t-SNE params
    perplexity = param.Integer(default=30, bounds=(5, 100))
    learning_rate = param.Number(default=200.0, bounds=(10.0, 1000.0))

    # Common
    n_components = param.Integer(default=2, bounds=(2, 3))
    random_state = param.Integer(default=42)

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce ``(N, D)`` → ``(N, n_components)`` float32 array."""
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {embeddings.shape}")

        # If already low-dim, just return first n_components
        if embeddings.shape[1] <= self.n_components:
            return embeddings[:, : self.n_components].astype(np.float32)

        if self.method == "pca":
            return self._pca(embeddings)
        elif self.method == "tsne":
            return self._tsne(embeddings)
        elif self.method == "umap":
            return self._umap(embeddings)
        raise ValueError(f"Unknown method: {self.method}")

    def reduce_with_quality(self, embeddings: np.ndarray) -> tuple[np.ndarray, dict]:
        """Reduce and also compute quality metrics (trustworthiness, stress)."""
        reduced = self.reduce(embeddings)
        metrics = self.quality_metrics(embeddings, reduced)
        return reduced, metrics

    @staticmethod
    def quality_metrics(original: np.ndarray, reduced: np.ndarray, k: int = 10) -> dict:
        """Compute DR quality metrics.

        Returns
        -------
        dict with keys: trustworthiness, stress, n_samples
        """
        from sklearn.manifold import trustworthiness as tw
        from sklearn.metrics import pairwise_distances

        n = min(len(original), 5000)  # Subsample for speed
        if n < len(original):
            idx = np.random.RandomState(42).choice(len(original), n, replace=False)
            orig_sub = original[idx]
            red_sub = reduced[idx]
        else:
            orig_sub = original
            red_sub = reduced

        k_actual = min(k, max(1, n // 2 - 1))

        # Trustworthiness (0-1, higher = better)
        if k_actual < 1 or n < 4:
            trust = 1.0  # Too few samples to compute meaningfully
        else:
            trust = float(tw(orig_sub, red_sub, n_neighbors=k_actual))

        # Kruskal stress
        d_orig = pairwise_distances(orig_sub).ravel()
        d_red = pairwise_distances(red_sub).ravel()
        denom = np.sum(d_orig**2)
        stress = float(np.sqrt(np.sum((d_orig - d_red) ** 2) / denom)) if denom > 0 else 0.0

        return {
            "trustworthiness": trust,
            "stress": stress,
            "n_samples": n,
            "k": k_actual,
        }

    # ── Backends ────────────────────────────────────────────────────────

    def _pca(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        return pca.fit_transform(embeddings).astype(np.float32)

    def _tsne(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE

        # Pre-reduce with PCA if > 50 dims (recommended by sklearn docs)
        if embeddings.shape[1] > 50:
            from sklearn.decomposition import PCA

            embeddings = PCA(n_components=50, random_state=self.random_state).fit_transform(embeddings)

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=min(self.perplexity, len(embeddings) - 1),
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            init="pca",
        )
        return tsne.fit_transform(embeddings).astype(np.float32)

    def _umap(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, len(embeddings) - 1),
            min_dist=self.min_dist,
            random_state=self.random_state,
        )
        return reducer.fit_transform(embeddings).astype(np.float32)
