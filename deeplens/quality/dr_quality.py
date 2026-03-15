"""Dimensionality Reduction Quality Dashboard.

Interactive assessment of DR projection quality including
Shepard diagrams, trustworthiness/continuity curves, and
multi-method comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import param

import holoviews as hv
import panel as pn

hv.extension("bokeh")


def _available_methods() -> list[str]:
    """Return list of available DR methods."""
    methods = ["pca", "tsne"]
    try:
        from umap import UMAP  # noqa: F401
        methods.append("umap")
    except ImportError:
        pass
    return methods


class DRQualityDashboard(pn.viewable.Viewer):
    """Interactive DR quality assessment.

    Features
    --------
    - Shepard diagram: original vs reduced distances
    - Trustworthiness curve vs neighborhood size k
    - Multi-method comparison (PCA, t-SNE, UMAP)
    - Stress metrics as Panel indicators
    """

    state = param.ClassSelector(class_=object, default=None, doc="Optional DeepLensState")
    embeddings_raw = param.Array(doc="Original high-dimensional embeddings")
    k_max = param.Integer(default=50, bounds=(5, 200), doc="Max k for quality curves")
    sample_size = param.Integer(default=2000, bounds=(100, 10000), doc="Subsample for speed")

    def __init__(self, **params):
        super().__init__(**params)
        self._results_cache: dict[str, dict] = {}

    def _subsample(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Subsample embeddings for computation speed. Returns (subset, indices)."""
        if self.embeddings_raw is None:
            return np.array([]), None
        n = min(self.sample_size, len(self.embeddings_raw))
        if n < len(self.embeddings_raw):
            idx = np.random.RandomState(42).choice(len(self.embeddings_raw), n, replace=False)
            return self.embeddings_raw[idx], idx
        return self.embeddings_raw, None

    def _reduce_with_method(self, data: np.ndarray, method: str) -> np.ndarray:
        """Reduce data with a specific method."""
        from deeplens.embeddings.reduce import DimensionalityReducer
        return DimensionalityReducer(method=method).reduce(data)

    def _compute_quality(self, original: np.ndarray, reduced: np.ndarray, k_values: list[int]) -> dict:
        """Compute quality metrics for a range of k values."""
        from sklearn.manifold import trustworthiness
        from sklearn.neighbors import NearestNeighbors

        results = {"k": [], "trustworthiness": [], "continuity": []}

        # Pre-compute kNN once with max k for efficiency
        max_k = max(k_values)
        if max_k >= len(original):
            max_k = len(original) - 1
        nn_orig = NearestNeighbors(n_neighbors=max_k + 1).fit(original)
        nn_red = NearestNeighbors(n_neighbors=max_k + 1).fit(reduced)
        orig_nn_all = nn_orig.kneighbors(original, return_distance=False)[:, 1:]
        red_nn_all = nn_red.kneighbors(reduced, return_distance=False)[:, 1:]

        for k in k_values:
            if k >= len(original):
                break
            trust = trustworthiness(original, reduced, n_neighbors=k)
            cont = self._continuity_from_neighbors(orig_nn_all[:, :k], red_nn_all[:, :k], len(original), k)
            results["k"].append(k)
            results["trustworthiness"].append(trust)
            results["continuity"].append(cont)

        # Stress via sampled pairs (avoids O(n^2) full matrix)
        results["stress"] = self._sampled_stress(original, reduced)
        return results

    @staticmethod
    def _continuity_from_neighbors(
        orig_nn: np.ndarray, red_nn: np.ndarray, n: int, k: int
    ) -> float:
        """Compute continuity from pre-computed neighbor arrays.

        Continuity measures whether original neighbors remain neighbors
        after reduction. Uses unweighted set-difference with matching normalization.
        """
        missing = 0
        for i in range(n):
            orig_set = set(orig_nn[i])
            red_set = set(red_nn[i])
            missing += len(orig_set - red_set)

        max_missing = n * k  # Each point can miss at most k neighbors
        if max_missing == 0:
            return 1.0
        return float(1.0 - (missing / max_missing))

    @staticmethod
    def _sampled_stress(original: np.ndarray, reduced: np.ndarray, n_pairs: int = 5000) -> float:
        """Compute Kruskal stress from sampled pairs (O(n_pairs) not O(n^2))."""
        n = len(original)
        rng = np.random.RandomState(42)

        # Sample random pairs directly (simpler & avoids float precision issues)
        actual_pairs = min(n_pairs, n * (n - 1) // 2)
        i_idx = rng.randint(0, n, size=actual_pairs)
        j_idx = rng.randint(0, n - 1, size=actual_pairs)
        # Ensure i != j by shifting j where j >= i
        j_idx[j_idx >= i_idx] += 1

        # Compute distances only for sampled pairs
        d_orig = np.sqrt(np.sum((original[i_idx] - original[j_idx]) ** 2, axis=1))
        d_red = np.sqrt(np.sum((reduced[i_idx] - reduced[j_idx]) ** 2, axis=1))

        denom = np.sum(d_orig**2)
        if denom == 0:
            return 0.0
        return float(np.sqrt(np.sum((d_orig - d_red) ** 2) / denom))

    def _shepard_diagram(self, method: str = "pca") -> hv.Element:
        """Shepard diagram: sampled pairwise original vs reduced distances."""
        data, _ = self._subsample()
        if len(data) == 0:
            return hv.Text(0, 0, "No data")

        reduced = self._reduce_with_method(data, method)

        # Sample pairs from upper triangle
        n = len(data)
        n_pairs = min(5000, n * (n - 1) // 2)
        rng = np.random.RandomState(42)
        i_idx = rng.randint(0, n, n_pairs * 2)
        j_idx = rng.randint(0, n, n_pairs * 2)
        mask = i_idx < j_idx  # Upper triangle only, no duplicates
        i_idx, j_idx = i_idx[mask][:n_pairs], j_idx[mask][:n_pairs]

        d_orig = np.sqrt(np.sum((data[i_idx] - data[j_idx]) ** 2, axis=1))
        d_red = np.sqrt(np.sum((reduced[i_idx] - reduced[j_idx]) ** 2, axis=1))

        df = pd.DataFrame({
            "Original Distance": d_orig,
            "Reduced Distance": d_red,
        })

        scatter = hv.Scatter(df, "Original Distance", "Reduced Distance").opts(
            size=2, alpha=0.3, color="#3498db",
            width=450, height=400,
            title=f"Shepard Diagram ({method.upper()})",
            tools=["hover"],
        )

        max_val = max(df["Original Distance"].max(), df["Reduced Distance"].max())
        identity = hv.Curve([(0, 0), (max_val, max_val)]).opts(
            color="red", line_dash="dashed", line_width=2
        )

        return scatter * identity

    def _quality_curves(self) -> hv.Element:
        """Trustworthiness and continuity vs k for available methods."""
        data, _ = self._subsample()
        if len(data) == 0:
            return hv.Text(0, 0, "No data")

        k_values = list(range(5, min(self.k_max, len(data) - 1), 5))
        methods = _available_methods()

        overlays = []
        for method in methods:
            reduced = self._reduce_with_method(data, method)
            quality = self._compute_quality(data, reduced, k_values)
            self._results_cache[method] = quality

            trust_curve = hv.Curve(
                list(zip(quality["k"], quality["trustworthiness"])),
                kdims=["k"], vdims=["Score"],
                label=f"{method.upper()} Trust",
            )
            cont_curve = hv.Curve(
                list(zip(quality["k"], quality["continuity"])),
                kdims=["k"], vdims=["Score"],
                label=f"{method.upper()} Cont",
            ).opts(line_dash="dashed")
            overlays.extend([trust_curve, cont_curve])

        return hv.Overlay(overlays).opts(
            width=550, height=350,
            title="Quality vs Neighborhood Size (k)",
            legend_position="bottom_right",
        )

    def _stress_indicators(self) -> pn.Row:
        """Stress metric indicators for each method."""
        indicators = []
        for method, quality in self._results_cache.items():
            stress = quality.get("stress", 0)
            indicators.append(
                pn.indicators.Number(
                    name=f"{method.upper()} Stress",
                    value=round(stress, 4),
                    format="{value:.4f}",
                    colors=[(0.1, "green"), (0.3, "orange"), (1.0, "red")],
                    font_size="18pt",
                    title_size="10pt",
                )
            )
        return pn.Row(*indicators) if indicators else pn.pane.Markdown("")

    def _recommendation(self) -> pn.pane.Markdown:
        """Auto-recommend the best DR method."""
        if not self._results_cache:
            return pn.pane.Markdown("")

        best_method = None
        best_score = -1.0

        for method, quality in self._results_cache.items():
            if quality["trustworthiness"]:
                avg_trust = np.mean(quality["trustworthiness"])
                avg_cont = np.mean(quality["continuity"])
                score = (avg_trust + avg_cont) / 2 - quality["stress"] * 0.1
                if score > best_score:
                    best_score = score
                    best_method = method

        if best_method:
            return pn.pane.Markdown(
                f"### Recommendation\n"
                f"**Best method for this data: {best_method.upper()}**\n\n"
                f"Based on combined trustworthiness, continuity, and stress metrics."
            )
        return pn.pane.Markdown("")

    def __panel__(self):
        if self.embeddings_raw is None or len(self.embeddings_raw) == 0:
            return pn.pane.Markdown(
                "### DR Quality Dashboard\n*Provide embeddings to assess projection quality.*"
            )

        methods = _available_methods()
        method_select = pn.widgets.Select(
            name="Shepard Method", options=methods, value=methods[0]
        )

        @pn.depends(method_select)
        def shepard_view(method):
            return pn.pane.HoloViews(self._shepard_diagram(method))

        # Compute quality curves (populates _results_cache)
        quality_plot = pn.pane.HoloViews(self._quality_curves())

        return pn.Column(
            pn.Row(
                pn.Column(method_select, shepard_view),
                pn.Column(quality_plot, self._stress_indicators()),
            ),
            self._recommendation(),
            sizing_mode="stretch_width",
        )
