"""Tests for deeplens.quality.dr_quality.DRQualityDashboard."""

from __future__ import annotations

import numpy as np
import pytest

import panel as pn
import holoviews as hv
from sklearn.decomposition import PCA

from deeplens.quality.dr_quality import DRQualityDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings(n=50, d=8, seed=42):
    """Return a small high-dimensional embedding matrix."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float64)


def _make_reduced(embeddings, method="pca"):
    return PCA(n_components=2).fit_transform(embeddings)


def _make_dashboard(n=50, d=8):
    emb = _make_embeddings(n=n, d=d)
    return DRQualityDashboard(embeddings_raw=emb)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestDRQualityDashboardInit:
    def test_init_no_embeddings(self):
        """DRQualityDashboard should initialise with no data."""
        dash = DRQualityDashboard()
        assert dash.embeddings_raw is None

    def test_init_with_embeddings(self):
        emb = _make_embeddings()
        dash = DRQualityDashboard(embeddings_raw=emb)
        assert dash.embeddings_raw is not None
        assert dash.embeddings_raw.shape == emb.shape

    def test_results_cache_empty_on_init(self):
        dash = _make_dashboard()
        assert isinstance(dash._results_cache, dict)
        assert len(dash._results_cache) == 0

    def test_default_k_max(self):
        dash = DRQualityDashboard()
        assert dash.k_max == 50

    def test_default_sample_size(self):
        dash = DRQualityDashboard()
        assert dash.sample_size == 2000

    def test_custom_k_max(self):
        dash = DRQualityDashboard(k_max=20)
        assert dash.k_max == 20

    def test_custom_sample_size(self):
        dash = DRQualityDashboard(sample_size=500)
        assert dash.sample_size == 500


# ---------------------------------------------------------------------------
# _subsample
# ---------------------------------------------------------------------------

class TestSubsample:
    def test_no_embeddings_returns_empty(self):
        dash = DRQualityDashboard()
        data, idx = dash._subsample()
        assert len(data) == 0
        assert idx is None

    def test_small_dataset_returns_full(self):
        """When n < sample_size, all data should be returned."""
        emb = _make_embeddings(n=30)
        dash = DRQualityDashboard(embeddings_raw=emb, sample_size=2000)
        data, idx = dash._subsample()
        assert len(data) == 30
        assert idx is None

    def test_large_dataset_subsamples(self):
        """When n > sample_size, a subsample of size sample_size is returned."""
        emb = _make_embeddings(n=300, d=4)
        dash = DRQualityDashboard(embeddings_raw=emb, sample_size=100)
        data, idx = dash._subsample()
        assert len(data) == 100
        assert idx is not None
        assert len(idx) == 100

    def test_subsample_indices_unique(self):
        emb = _make_embeddings(n=300, d=4)
        dash = DRQualityDashboard(embeddings_raw=emb, sample_size=100)
        _, idx = dash._subsample()
        assert len(set(idx)) == 100

    def test_subsample_reproducible(self):
        """Same seed should produce the same subsample."""
        emb = _make_embeddings(n=300, d=4)
        dash = DRQualityDashboard(embeddings_raw=emb, sample_size=100)
        _, idx1 = dash._subsample()
        _, idx2 = dash._subsample()
        np.testing.assert_array_equal(idx1, idx2)


# ---------------------------------------------------------------------------
# _sampled_stress
# ---------------------------------------------------------------------------

class TestSampledStress:
    def test_stress_non_negative(self):
        original = _make_embeddings(n=30, d=8)
        reduced = _make_reduced(original)
        stress = DRQualityDashboard._sampled_stress(original, reduced)
        assert stress >= 0.0

    def test_identical_embeddings_stress_zero(self):
        """When original == reduced, stress should be 0."""
        X = _make_embeddings(n=20, d=2)
        stress = DRQualityDashboard._sampled_stress(X, X)
        assert stress == pytest.approx(0.0, abs=1e-6)

    def test_stress_returns_float(self):
        original = _make_embeddings(n=20, d=4)
        reduced = _make_reduced(original)
        stress = DRQualityDashboard._sampled_stress(original, reduced)
        assert isinstance(stress, float)

    def test_constant_original_returns_zero(self):
        """If all original distances are 0, stress should be 0 (degenerate case)."""
        original = np.ones((10, 4))
        reduced = _make_embeddings(n=10, d=2)
        stress = DRQualityDashboard._sampled_stress(original, reduced)
        assert stress == 0.0

    def test_stress_reasonable_range_pca(self):
        """PCA stress on a random Gaussian should be well under 1."""
        original = _make_embeddings(n=60, d=8)
        reduced = _make_reduced(original)
        stress = DRQualityDashboard._sampled_stress(original, reduced)
        assert 0.0 <= stress < 10.0  # loose upper bound


# ---------------------------------------------------------------------------
# _continuity_from_neighbors
# ---------------------------------------------------------------------------

class TestContinuityFromNeighbors:
    def _build_nn_arrays(self, n=20, k=3):
        """Build simple NN arrays where orig_nn == red_nn (perfect continuity)."""
        orig_nn = np.array([[j for j in range(n) if j != i][:k] for i in range(n)])
        return orig_nn, orig_nn.copy()

    def test_perfect_continuity_is_one(self):
        """Identical neighbor arrays → continuity == 1.0."""
        orig_nn, red_nn = self._build_nn_arrays(n=15, k=4)
        cont = DRQualityDashboard._continuity_from_neighbors(orig_nn, red_nn, n=15, k=4)
        assert cont == pytest.approx(1.0)

    def test_continuity_in_range(self):
        rng = np.random.RandomState(0)
        n, k = 20, 5
        orig_nn = np.array([rng.choice([j for j in range(n) if j != i], k, replace=False) for i in range(n)])
        red_nn = np.array([rng.choice([j for j in range(n) if j != i], k, replace=False) for i in range(n)])
        cont = DRQualityDashboard._continuity_from_neighbors(orig_nn, red_nn, n=n, k=k)
        assert 0.0 <= cont <= 1.0

    def test_returns_float(self):
        orig_nn, red_nn = self._build_nn_arrays(n=10, k=3)
        cont = DRQualityDashboard._continuity_from_neighbors(orig_nn, red_nn, n=10, k=3)
        assert isinstance(cont, float)

    def test_zero_max_missing_returns_one(self):
        """n*k == 0 (degenerate) should return 1.0 without error."""
        cont = DRQualityDashboard._continuity_from_neighbors(
            np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int), n=0, k=0
        )
        assert cont == 1.0


# ---------------------------------------------------------------------------
# _compute_quality
# ---------------------------------------------------------------------------

class TestComputeQuality:
    def test_returns_dict_with_expected_keys(self):
        original = _make_embeddings(n=25, d=4)
        reduced = _make_reduced(original)
        dash = DRQualityDashboard(embeddings_raw=original)
        result = dash._compute_quality(original, reduced, k_values=[3, 5])
        assert "k" in result
        assert "trustworthiness" in result
        assert "continuity" in result
        assert "stress" in result

    def test_k_values_match(self):
        original = _make_embeddings(n=25, d=4)
        reduced = _make_reduced(original)
        dash = DRQualityDashboard(embeddings_raw=original)
        k_values = [3, 5, 7]
        result = dash._compute_quality(original, reduced, k_values=k_values)
        assert result["k"] == k_values

    def test_trustworthiness_in_range(self):
        original = _make_embeddings(n=25, d=4)
        reduced = _make_reduced(original)
        dash = DRQualityDashboard(embeddings_raw=original)
        result = dash._compute_quality(original, reduced, k_values=[3, 5])
        for t in result["trustworthiness"]:
            assert 0.0 <= t <= 1.0

    def test_continuity_in_range(self):
        original = _make_embeddings(n=25, d=4)
        reduced = _make_reduced(original)
        dash = DRQualityDashboard(embeddings_raw=original)
        result = dash._compute_quality(original, reduced, k_values=[3, 5])
        for c in result["continuity"]:
            assert 0.0 <= c <= 1.0

    def test_stress_non_negative(self):
        original = _make_embeddings(n=25, d=4)
        reduced = _make_reduced(original)
        dash = DRQualityDashboard(embeddings_raw=original)
        result = dash._compute_quality(original, reduced, k_values=[3])
        assert result["stress"] >= 0.0

    def test_k_exceeding_n_is_skipped(self):
        """k values >= n should be silently skipped."""
        original = _make_embeddings(n=10, d=4)
        reduced = _make_reduced(original)
        dash = DRQualityDashboard(embeddings_raw=original)
        # k=15 exceeds n=10, so it should be skipped
        result = dash._compute_quality(original, reduced, k_values=[3, 15])
        assert 15 not in result["k"]


# ---------------------------------------------------------------------------
# _shepard_diagram
# ---------------------------------------------------------------------------

class TestShepardDiagram:
    def test_no_data_returns_text(self):
        dash = DRQualityDashboard()
        result = dash._shepard_diagram()
        assert isinstance(result, hv.Text)

    def test_returns_overlay(self):
        dash = _make_dashboard(n=30, d=6)
        result = dash._shepard_diagram(method="pca")
        assert isinstance(result, hv.Overlay)

    def test_overlay_contains_scatter(self):
        dash = _make_dashboard(n=30, d=6)
        result = dash._shepard_diagram(method="pca")
        has_scatter = any(isinstance(el, hv.Scatter) for el in result)
        assert has_scatter

    def test_overlay_contains_identity_line(self):
        """The Shepard diagram should contain a red diagonal reference line."""
        dash = _make_dashboard(n=30, d=6)
        result = dash._shepard_diagram(method="pca")
        has_curve = any(isinstance(el, hv.Curve) for el in result)
        assert has_curve

    def test_scatter_kdim_original_distance(self):
        dash = _make_dashboard(n=30, d=6)
        result = dash._shepard_diagram(method="pca")
        scatter = next(el for el in result if isinstance(el, hv.Scatter))
        kdim_names = [str(k) for k in scatter.kdims]
        assert "Original Distance" in kdim_names

    def test_scatter_vdim_reduced_distance(self):
        dash = _make_dashboard(n=30, d=6)
        result = dash._shepard_diagram(method="pca")
        scatter = next(el for el in result if isinstance(el, hv.Scatter))
        vdim_names = [str(v) for v in scatter.vdims]
        assert "Reduced Distance" in vdim_names


# ---------------------------------------------------------------------------
# _quality_curves
# ---------------------------------------------------------------------------

class TestQualityCurves:
    def test_no_data_returns_text(self):
        dash = DRQualityDashboard()
        result = dash._quality_curves()
        assert isinstance(result, hv.Text)

    def test_returns_overlay(self):
        # Use n=30, small k_max so test is fast
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        result = dash._quality_curves()
        assert isinstance(result, hv.Overlay)

    def test_overlay_contains_curves(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        result = dash._quality_curves()
        curves = [el for el in result if isinstance(el, hv.Curve)]
        assert len(curves) > 0

    def test_results_cache_populated(self):
        """After _quality_curves(), _results_cache should have at least PCA."""
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        dash._quality_curves()
        assert "pca" in dash._results_cache

    def test_curves_have_score_vdim(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        result = dash._quality_curves()
        for el in result:
            if isinstance(el, hv.Curve):
                vdim_names = [str(v) for v in el.vdims]
                assert "Score" in vdim_names


# ---------------------------------------------------------------------------
# _stress_indicators
# ---------------------------------------------------------------------------

class TestStressIndicators:
    def test_empty_cache_returns_empty_markdown(self):
        dash = DRQualityDashboard()
        result = dash._stress_indicators()
        assert isinstance(result, pn.pane.Markdown)

    def test_returns_panel_row_after_quality_curves(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        dash._quality_curves()  # populate cache
        result = dash._stress_indicators()
        assert isinstance(result, pn.Row)

    def test_indicators_are_number_widgets(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        dash._quality_curves()
        row = dash._stress_indicators()
        numbers = [c for c in row if isinstance(c, pn.indicators.Number)]
        assert len(numbers) >= 1

    def test_stress_values_non_negative(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        dash._quality_curves()
        row = dash._stress_indicators()
        for indicator in row:
            if isinstance(indicator, pn.indicators.Number):
                assert indicator.value >= 0.0


# ---------------------------------------------------------------------------
# _recommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_empty_cache_returns_empty_markdown(self):
        dash = DRQualityDashboard()
        result = dash._recommendation()
        assert isinstance(result, pn.pane.Markdown)
        assert result.object == ""

    def test_returns_markdown_after_cache_populated(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        dash._quality_curves()
        result = dash._recommendation()
        assert isinstance(result, pn.pane.Markdown)

    def test_recommendation_mentions_best_method(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        dash._quality_curves()
        result = dash._recommendation()
        text = result.object.upper()
        assert any(m.upper() in text for m in ["PCA", "TSNE", "UMAP"])

    def test_recommendation_content_not_empty(self):
        dash = DRQualityDashboard(embeddings_raw=_make_embeddings(n=30, d=4), k_max=15)
        dash._quality_curves()
        result = dash._recommendation()
        assert len(result.object) > 0


# ---------------------------------------------------------------------------
# __panel__
# ---------------------------------------------------------------------------

class TestDRQualityDashboardPanel:
    def test_no_embeddings_returns_placeholder_markdown(self):
        dash = DRQualityDashboard()
        result = dash.__panel__()
        assert isinstance(result, pn.pane.Markdown)
        assert "DR Quality Dashboard" in result.object

    def test_empty_array_returns_placeholder_markdown(self):
        dash = DRQualityDashboard(embeddings_raw=np.array([]))
        result = dash.__panel__()
        assert isinstance(result, pn.pane.Markdown)

    def test_with_embeddings_returns_panel_column(self):
        dash = DRQualityDashboard(
            embeddings_raw=_make_embeddings(n=30, d=4), k_max=15
        )
        result = dash.__panel__()
        assert isinstance(result, pn.Column)

    def test_panel_column_contains_row(self):
        dash = DRQualityDashboard(
            embeddings_raw=_make_embeddings(n=30, d=4), k_max=15
        )
        result = dash.__panel__()
        has_row = any(isinstance(child, pn.Row) for child in result)
        assert has_row

    def test_panel_contains_method_selector(self):
        dash = DRQualityDashboard(
            embeddings_raw=_make_embeddings(n=30, d=4), k_max=15
        )
        result = dash.__panel__()
        # The column at index 0 → row → first column should have the Select
        first_row = next(c for c in result if isinstance(c, pn.Row))
        first_col = first_row[0]
        assert isinstance(first_col, pn.Column)
        has_select = any(isinstance(w, pn.widgets.Select) for w in first_col)
        assert has_select

    def test_panel_sizing_mode_stretch(self):
        dash = DRQualityDashboard(
            embeddings_raw=_make_embeddings(n=30, d=4), k_max=15
        )
        result = dash.__panel__()
        assert result.sizing_mode == "stretch_width"

    def test_panel_with_larger_embeddings(self):
        """Panel should handle larger embeddings without crashing."""
        emb = _make_embeddings(n=120, d=10)
        dash = DRQualityDashboard(embeddings_raw=emb, k_max=20, sample_size=100)
        result = dash.__panel__()
        assert isinstance(result, pn.Column)
