"""Tests for the DriftDetector module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeplens.compare.drift import DriftDetector


@pytest.fixture
def ref_prod_dfs():
    """Create reference and production DataFrames with known drift."""
    rng = np.random.RandomState(42)
    ref = pd.DataFrame({
        "f1": rng.normal(0, 1, 100),
        "f2": rng.normal(0, 1, 100),
        "f3": rng.uniform(0, 1, 100),
    })
    # Production with drift on f1 and f3
    prod = pd.DataFrame({
        "f1": rng.normal(2, 1, 100),  # shifted mean
        "f2": rng.normal(0, 1, 100),  # no drift
        "f3": rng.uniform(0.5, 1.5, 100),  # shifted range
    })
    return ref, prod


class TestDriftDetectorInit:
    def test_auto_detects_feature_columns(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod)
        assert set(dd.feature_columns) == {"f1", "f2", "f3"}

    def test_respects_provided_feature_columns(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod, feature_columns=["f1"])
        assert dd.feature_columns == ["f1"]

    def test_excludes_timestamp_col(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        ref["ts"] = range(100)
        prod["ts"] = range(100)
        dd = DriftDetector(reference_df=ref, production_df=prod, timestamp_col="ts")
        assert "ts" not in dd.feature_columns


class TestComputePSI:
    def test_psi_no_drift(self):
        rng = np.random.RandomState(0)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0, 1, 1000)
        psi = DriftDetector._compute_psi(a, b)
        assert psi >= 0
        assert psi < 0.1  # low drift

    def test_psi_with_drift(self):
        rng = np.random.RandomState(0)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(3, 1, 1000)
        psi = DriftDetector._compute_psi(a, b)
        assert psi > 0.5  # significant drift

    def test_psi_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        psi = DriftDetector._compute_psi(a, a)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_psi_non_negative(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0.5, 2, 100)
        psi = DriftDetector._compute_psi(a, b)
        assert psi >= 0


class TestComputeDriftScores:
    def test_drift_scores_shape(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod)
        scores = dd._compute_drift_scores()
        assert len(scores) == 3
        assert "Feature" in scores.columns
        assert "KS Statistic" in scores.columns
        assert "Drift" in scores.columns

    def test_drift_detected_on_shifted_feature(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod)
        scores = dd._compute_drift_scores()
        f1_row = scores[scores["Feature"] == "f1"]
        assert f1_row["Drift"].values[0] == "Yes"

    def test_caching(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod)
        s1 = dd._get_drift_scores()
        s2 = dd._get_drift_scores()
        assert s1 is s2  # same object = cached

    def test_skips_missing_columns(self):
        ref = pd.DataFrame({"f1": [1, 2, 3]})
        prod = pd.DataFrame({"f2": [4, 5, 6]})
        dd = DriftDetector(reference_df=ref, production_df=prod, feature_columns=["f1", "f2"])
        scores = dd._compute_drift_scores()
        assert len(scores) == 0

    def test_skips_empty_columns(self):
        ref = pd.DataFrame({"f1": [np.nan, np.nan, np.nan]})
        prod = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        dd = DriftDetector(reference_df=ref, production_df=prod)
        scores = dd._compute_drift_scores()
        assert len(scores) == 0


class TestDriftPanel:
    def test_panel_no_data(self):
        dd = DriftDetector()
        panel = dd.__panel__()
        assert panel is not None

    def test_panel_no_features(self):
        ref = pd.DataFrame({"text": ["a", "b"]})
        prod = pd.DataFrame({"text": ["c", "d"]})
        dd = DriftDetector(reference_df=ref, production_df=prod)
        panel = dd.__panel__()
        assert panel is not None

    def test_panel_with_data(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod)
        panel = dd.__panel__()
        assert panel is not None


class TestKDEComparison:
    def test_kde_returns_overlay(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod)
        plot = dd._kde_comparison("f1")
        assert plot is not None


class TestTemporalAnimation:
    def test_no_timestamp(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        dd = DriftDetector(reference_df=ref, production_df=prod)
        result = dd._temporal_animation()
        assert result is not None

    def test_with_timestamp(self, ref_prod_dfs):
        ref, prod = ref_prod_dfs
        prod["ts"] = range(100)
        dd = DriftDetector(
            reference_df=ref, production_df=prod,
            feature_columns=["f1", "f2", "f3"], timestamp_col="ts"
        )
        result = dd._temporal_animation()
        assert result is not None
