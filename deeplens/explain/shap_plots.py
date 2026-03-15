"""Interactive SHAP visualizations as HoloViews elements.

Unlike the default SHAP library (static matplotlib), these plots are
fully interactive: hover, select, zoom, and cross-filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holoviews as hv

hv.extension("bokeh")


def waterfall(
    shap_values: np.ndarray,
    feature_names: list[str],
    base_value: float = 0.0,
    max_display: int = 15,
) -> hv.Layout:
    """Interactive waterfall plot showing per-feature SHAP contributions.

    Parameters
    ----------
    shap_values : array (n_features,)
        SHAP values for a single prediction.
    feature_names : list[str]
        Feature names corresponding to SHAP values.
    base_value : float
        Model expected value (base prediction).
    max_display : int
        Maximum number of features to display.
    """
    if shap_values.ndim > 1:
        shap_values = shap_values.mean(axis=-1)

    # Sort by absolute SHAP value, keep top features
    order = np.argsort(np.abs(shap_values))[::-1][:max_display]
    order = order[::-1]  # Reverse for bottom-to-top display

    names = [feature_names[i] for i in order]
    values = shap_values[order]

    colors = ["#ff4444" if v > 0 else "#4444ff" for v in values]

    bars = hv.Bars(
        [(name, val, c) for name, val, c in zip(names, values, colors)],
        kdims=["Feature"],
        vdims=["SHAP Value", "color"],
    ).opts(
        color="color",
        invert_axes=True,
        width=500,
        height=max(300, max_display * 25),
        tools=["hover"],
        title=f"SHAP Waterfall (base={base_value:.3f})",
        xlabel="SHAP Value",
        ylabel="Feature",
    )

    # Add base value line
    vline = hv.VLine(0).opts(color="black", line_dash="dashed", line_width=1)

    return bars * vline


def beeswarm(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> hv.Element:
    """Interactive beeswarm (summary) plot.

    Shows SHAP value distribution for each feature, colored by
    the feature's actual value.

    Parameters
    ----------
    shap_values : array (n_samples, n_features)
        SHAP values for all samples.
    feature_values : array (n_samples, n_features)
        Actual feature values for coloring.
    feature_names : list[str]
        Feature names.
    max_display : int
        Maximum features to show.
    """
    if shap_values.ndim > 2:
        shap_values = shap_values.mean(axis=-1)

    # Sort by mean |SHAP|
    importance_order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:max_display]

    # Vectorized construction for performance
    n_samples = shap_values.shape[0]
    n_display = len(importance_order)
    total = n_samples * n_display

    shap_col = np.empty(total, dtype=np.float32)
    pos_col = np.empty(total, dtype=np.float32)
    fv_col = np.empty(total, dtype=np.float32)
    raw_col = np.empty(total, dtype=np.float32)
    feat_col = np.empty(total, dtype=object)

    for rank, feat_idx in enumerate(importance_order):
        sl = slice(rank * n_samples, (rank + 1) * n_samples)
        sv = shap_values[:, feat_idx]
        fv = feature_values[:, feat_idx]
        fv_min, fv_max = np.nanmin(fv), np.nanmax(fv)
        fv_norm = (fv - fv_min) / (fv_max - fv_min + 1e-10)
        jitter = np.random.RandomState(feat_idx).uniform(-0.3, 0.3, n_samples)

        shap_col[sl] = sv
        pos_col[sl] = rank + jitter
        fv_col[sl] = fv_norm
        raw_col[sl] = fv
        feat_col[sl] = feature_names[feat_idx]

    df = pd.DataFrame({
        "SHAP Value": shap_col,
        "Feature Position": pos_col,
        "Feature Value": fv_col,
        "Feature": feat_col,
        "Raw Value": raw_col,
    })

    yticks = [(rank, feature_names[idx]) for rank, idx in enumerate(importance_order)]

    points = hv.Points(
        df,
        kdims=["SHAP Value", "Feature Position"],
        vdims=["Feature Value", "Feature", "Raw Value"],
    ).opts(
        color="Feature Value",
        cmap="RdBu_r",
        colorbar=True,
        clabel="Feature value (normalized)",
        size=3,
        alpha=0.6,
        width=650,
        height=max(350, max_display * 25),
        yticks=yticks,
        tools=["hover", "lasso_select", "box_select"],
        title="SHAP Beeswarm Plot",
        xlabel="SHAP Value (impact on prediction)",
    )

    vline = hv.VLine(0).opts(color="black", line_dash="dashed", line_width=1)

    return points * vline


def dependence(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_idx: int,
    feature_names: list[str],
    interaction_idx: int | None = None,
) -> hv.Element:
    """Interactive SHAP dependence plot.

    Shows how a feature's value affects the SHAP value,
    optionally colored by an interaction feature.

    Parameters
    ----------
    shap_values : array (n_samples, n_features)
    feature_values : array (n_samples, n_features)
    feature_idx : int
        Index of the feature to plot on x-axis.
    feature_names : list[str]
    interaction_idx : int, optional
        Index of feature to use for coloring (interaction effect).
    """
    if shap_values.ndim > 2:
        shap_values = shap_values.mean(axis=-1)

    x = feature_values[:, feature_idx]
    y = shap_values[:, feature_idx]
    fname = feature_names[feature_idx]

    df = pd.DataFrame({fname: x, "SHAP Value": y})

    opts = dict(
        size=5,
        alpha=0.6,
        width=550,
        height=400,
        tools=["hover", "lasso_select", "box_select"],
        title=f"SHAP Dependence: {fname}",
        xlabel=fname,
        ylabel="SHAP Value",
    )

    if interaction_idx is not None and interaction_idx != feature_idx:
        int_name = feature_names[interaction_idx]
        df[int_name] = feature_values[:, interaction_idx]
        scatter = hv.Scatter(df, kdims=[fname], vdims=["SHAP Value", int_name])
        opts["color"] = int_name
        opts["cmap"] = "coolwarm"
        opts["colorbar"] = True
    else:
        scatter = hv.Scatter(df, kdims=[fname], vdims=["SHAP Value"])
        opts["color"] = "#1f77b4"

    hline = hv.HLine(0).opts(color="black", line_dash="dashed", line_width=1)

    return scatter.opts(**opts) * hline


def importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> hv.Element:
    """Interactive global feature importance bar chart.

    Shows mean |SHAP value| for each feature, sorted descending.
    Supports ``Selection1D`` for cross-filtering.
    """
    if shap_values.ndim > 2:
        shap_values = shap_values.mean(axis=-1)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    order = order[::-1]  # Bottom-to-top

    names = [feature_names[i] for i in order]
    values = mean_abs[order]

    bars = hv.Bars(
        [(name, val) for name, val in zip(names, values)],
        kdims=["Feature"],
        vdims=["Mean |SHAP|"],
    ).opts(
        invert_axes=True,
        color="#1f77b4",
        width=450,
        height=max(300, max_display * 22),
        tools=["hover", "tap"],
        title="Global Feature Importance",
        xlabel="Mean |SHAP Value|",
        ylabel="Feature",
    )

    return bars
