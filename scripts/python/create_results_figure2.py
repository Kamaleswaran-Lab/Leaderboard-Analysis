#!/usr/bin/env python3
"""Generate Figure 2: Factor analysis methodology panels."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from pathlib import Path
import re

from plot_palette import (
    BASE_PALETTE,
    darken_color,
    get_metric_color,
    lighten_color,
)


ROOT = Path(__file__).resolve().parents[2]
PRELIM_DIR = ROOT / "optimal_threshold_results"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "figures" / "main_figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


plt.style.use("seaborn-v0_8-paper")
sns.set_palette(BASE_PALETTE)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


def compute_mcc(tp: pd.Series, fp: pd.Series, fn: pd.Series, tn: pd.Series) -> pd.Series:
    """Return Matthews Correlation Coefficient for confusion matrix inputs."""
    tp_f = tp.astype(float)
    fp_f = fp.astype(float)
    fn_f = fn.astype(float)
    tn_f = tn.astype(float)
    numerator = tp_f * tn_f - fp_f * fn_f
    denom = np.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
    with np.errstate(divide="ignore", invalid="ignore"):
        mcc = np.where(denom > 0, numerator / denom, np.nan)
    return pd.Series(mcc, index=tp.index)


def remove_outliers(series: pd.Series, iqr_factor: float = 1.5) -> pd.Series:
    """Return series with outliers removed using the IQR rule."""
    cleaned = series.dropna()
    if cleaned.empty:
        return cleaned
    q1, q3 = cleaned.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return cleaned[(cleaned >= lower) & (cleaned <= upper)]


def clean_team_label(raw: str) -> str:
    """Produce a readable, ASCII-only team label from raw identifiers."""
    if not isinstance(raw, str):
        return str(raw)
    label = raw.replace("output_", "").replace("outputs_", "").replace("output", "")
    label = label.split("_https")[0]
    label = label.replace("_", " ").strip()
    label = "".join(ch for ch in label if ord(ch) < 128)
    label = label.strip()
    label = re.sub(r"\d+$", "", label).strip()
    return label.title() if label else "Team"


def normalize_metric_name(name: str) -> str:
    """Normalize metric labels by lowercasing and stripping non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def annotate_without_overlap(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, labels: list[str]) -> None:
    """Label each point while avoiding overlapping text boxes."""
    texts: list[plt.Annotation] = []
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    order = np.argsort(ys)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    labels_sorted = [labels[idx] for idx in order]

    directions_left = [
        (-1.0, 0.0),
        (-0.9, 0.4),
        (-0.9, -0.4),
        (-0.7, 0.7),
        (-0.7, -0.7),
        (-0.5, 0.9),
        (-0.5, -0.9),
        (-0.3, 1.0),
        (-0.3, -1.0),
    ]
    directions_right = [
        (1.0, 0.0),
        (0.9, 0.4),
        (0.9, -0.4),
        (0.7, 0.7),
        (0.7, -0.7),
        (0.5, 0.9),
        (0.5, -0.9),
        (0.3, 1.0),
        (0.3, -1.0),
    ]
    radii = [12, 18, 24, 30, 36, 42, 48, 54, 60]

    y_top_threshold = 0.92
    y_bottom_threshold = 0.08

    for x_val, y_val, label in zip(xs_sorted, ys_sorted, labels_sorted):
        dirs = directions_right if x_val < 0.5 else directions_left
        ha = "left" if x_val < 0.5 else "right"

        placed = False
        for radius in radii:
            for dx_dir, dy_dir in dirs:
                if y_val > y_top_threshold and dy_dir > 0:
                    continue
                if y_val < y_bottom_threshold and dy_dir < 0:
                    continue

                dx = int(round(dx_dir * radius))
                dy = int(round(dy_dir * radius))

                text = ax.annotate(
                    label,
                    (x_val, y_val),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "edgecolor": "#777"},
                    arrowprops={"arrowstyle": "-", "color": "#777", "linewidth": 0.6},
                )
                fig.canvas.draw()
                bbox_new = text.get_window_extent(renderer=renderer)
                if not any(bbox_new.overlaps(t.get_window_extent(renderer=renderer)) for t in texts):
                    texts.append(text)
                    placed = True
                    break
                text.remove()
            if placed:
                break

        if not placed:
            # fall back to placing horizontally with gradual vertical adjustment until no overlap
            dy_options = [0, 8, -8, 16, -16, 24, -24, 32, -32]
            fallback_dx = radii[-1] if x_val < 0.5 else -radii[-1]
            for dy in dy_options:
                text = ax.annotate(
                    label,
                    (x_val, y_val),
                    xytext=(fallback_dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "edgecolor": "#777"},
                    arrowprops={"arrowstyle": "-", "color": "#777", "linewidth": 0.6},
                )
                fig.canvas.draw()
                bbox_new = text.get_window_extent(renderer=renderer)
                if not any(bbox_new.overlaps(t.get_window_extent(renderer=renderer)) for t in texts):
                    texts.append(text)
                    placed = True
                    break
                text.remove()

        if not placed:
            # Absolute last resort: no arrow, place directly atop with slight offset
            text = ax.text(
                x_val,
                y_val + 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "#777"},
                zorder=5,
            )
            fig.canvas.draw()
            texts.append(text)


def generate_figure2() -> Path:
    """Create Figure 2 and return the output path."""
    print("\n📈 Creating Figure 2: Factor Analysis Methodology...")

    final_scores = pd.read_csv(RAW_DIR / "final_scores.csv")
    prelim_sens = pd.read_csv(PRELIM_DIR / "evaluation_metrics_sensitivity_method.csv")
    dummy_phase = pd.read_csv(PROCESSED_DIR / "Dummy_phase_scores_new_full.csv")
    # Remove AUC outliers (AUC < 0.5) for consistency with Figure 1 pipeline
    if "AUC" in dummy_phase.columns:
        before_n = len(dummy_phase)
        dummy_phase = dummy_phase[dummy_phase["AUC"] > 0.5].copy()
        after_n = len(dummy_phase)
        print(f"Applied AUC > 0.5 filter to Dummy phase: {before_n} -> {after_n} submissions")
    if "AUC" in prelim_sens.columns:
        before_p = len(prelim_sens)
        prelim_sens = prelim_sens[prelim_sens["AUC"] > 0.5].copy()
        after_p = len(prelim_sens)
        print(f"Applied AUC > 0.5 filter to Preliminary phase: {before_p} -> {after_p} submissions")
    scale_params_df = pd.read_csv(PROCESSED_DIR / "scale_params.csv")
    scale_lookup = {
        row["feature"]: (row["center"], row["scale"])
        for _, row in scale_params_df.iterrows()
    }

    try:
        import json

        with open(PROCESSED_DIR / "factor_loadings.json", "r") as f:
            factor_loadings_dict = json.load(f)
        factor_loadings = pd.DataFrame(factor_loadings_dict)
    except Exception:
        factor_loadings = None

    factor_weights = pd.read_csv(PROCESSED_DIR / "factor_weights.csv")
    weight_map = {row["feature"]: row["weight"] for _, row in factor_weights.iterrows()}

    fig2 = plt.figure(figsize=(18, 10))
    gs2 = GridSpec(2, 4, figure=fig2, hspace=0.35, wspace=0.55, width_ratios=[0.9, 1.2, 1.2, 1.1])
    fig2.subplots_adjust(hspace=0.9, bottom=0.08, top=0.94, right=0.9, left=0.07, wspace=0.6)

    # Panel A
    ax2a = fig2.add_subplot(gs2[0, :2])
    core_metrics = ["F1", "AUPRC", "Net.Benefit", "ECE"]
    panel_a_weights = factor_weights[factor_weights["feature"].isin(core_metrics)].copy()
    colors_weight = [lighten_color(get_metric_color(feature), 0.25) for feature in panel_a_weights["feature"]]
    bars = ax2a.barh(
        panel_a_weights["feature"],
        panel_a_weights["weight"],
        color=colors_weight,
        alpha=0.7,
        edgecolor=[get_metric_color(feature) for feature in panel_a_weights["feature"]],
        linewidth=1.5,
    )
    ax2a.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax2a.set_xlabel("Weight Value", fontweight="bold", fontsize=12)
    ax2a.set_title("A. Multi-Criteria Factor Weights for Composite Scoring", fontweight="bold", fontsize=14)
    ax2a.grid(True, alpha=0.3, axis="x")
    ax2a.spines["top"].set_visible(False)
    ax2a.spines["right"].set_visible(False)

    for feature, weight in zip(panel_a_weights["feature"], panel_a_weights["weight"]):
        # For small weights (abs < 0.1), place text outside; otherwise inside
        if abs(weight) < 0.1:
            x_pos = weight + 0.03 if weight >= 0 else weight - 0.03
            ax2a.text(
                x_pos,
                feature,
                f"{weight:.3f}",
                va="center",
                ha="left" if weight >= 0 else "right",
                fontsize=11,
                fontweight="bold",
                color="black",
            )
        else:
            # Place text inside the bar
            x_pos = weight / 2  # Middle of the bar
            ax2a.text(
                x_pos,
                feature,
                f"{weight:.3f}",
                va="center",
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    # Panel B
    ax2b = fig2.add_subplot(gs2[0, 2:])
    panel_b_weights = panel_a_weights.copy()
    weights_abs = np.abs(panel_b_weights["weight"])
    total_core = weights_abs.sum()
    factor_target = 0.90
    if total_core > 0:
        panel_b_weights["share"] = weights_abs * (factor_target / total_core)
    else:
        panel_b_weights["share"] = 0.0

    # Append fixed 5% slices for simplicity and speed to mirror scoring pipeline
    panel_b_weights = pd.concat(
        [
            panel_b_weights,
            pd.DataFrame(
                {
                    "feature": ["Parsimony Score", "Inference Time"],
                    "weight": [weight_map.get("Parsimony Score", 0.0), weight_map.get("Inference Time", 0.0)],
                    "share": [0.05, 0.05],
                }
            ),
        ],
        ignore_index=True,
    )

    contribution = panel_b_weights["share"].values * 100
    colors_pie = [lighten_color(get_metric_color(feature), 0.25) for feature in panel_b_weights["feature"]]
    
    # Show percentages inside for all except small slices
    def autopct_func(_pct: float) -> str:
        """Keep interior labels empty; annotations added outside instead."""
        return ""
    
    wedges, texts, autotexts = ax2b.pie(
        contribution,
        labels=None,
        autopct=autopct_func,
        startangle=90,
        colors=colors_pie[: len(contribution)],
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    
    # Add labels outside the pie
    label_display_map = {
        "Net.Benefit": "Net Benefit",
        "Parsimony Score": "Model Simplicity",
        "Inference Time": "Inference Speed",
    }

    for wedge, pct, label in zip(wedges, contribution, panel_b_weights["feature"]):
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        
        # Get display label
        display_label = label_display_map.get(label, label)
        ax2b.text(
            1.2 * x,
            1.2 * y,
            f"{display_label}\n{pct:.1f}%",
            ha=horizontalalignment,
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    
    ax2b.set_title("B. Relative Weight Contribution", fontweight="bold", fontsize=13, pad=20)

    # Panel C - All metrics distributions from dummy phase (row 2 left)
    metric_definitions = [
        {"display": "F1", "aliases": ["F1"]},
        {"display": "AUPRC", "aliases": ["AUPRC"]},
        {"display": "AUC", "aliases": ["AUC"]},
        {"display": "ECE", "aliases": ["ECE"]},
        {"display": "Sensitivity", "aliases": ["Sensitivity"]},
        {"display": "Specificity", "aliases": ["Specificity"]},
        {"display": "Net Benefit", "aliases": ["Net.Benefit", "Net Benefit"]},
        {"display": "MCC", "aliases": ["MCC"]},
    ]

    phase_sources: list[tuple[str, pd.DataFrame]] = [
        ("Dummy Phase", dummy_phase.copy()),
        ("Preliminary Phase", prelim_sens.copy()),
    ]

    distribution_records: list[dict[str, object]] = []

    for phase_name, phase_df in phase_sources:
        numeric_df = phase_df.copy()
        for candidate in metric_definitions:
            for alias in candidate["aliases"]:
                if alias in numeric_df.columns:
                    numeric_df[alias] = pd.to_numeric(numeric_df[alias], errors="coerce")

        if {"TP", "FP", "FN", "TN"}.issubset(numeric_df.columns):
            numeric_df["MCC"] = compute_mcc(
                numeric_df["TP"],
                numeric_df["FP"],
                numeric_df["FN"],
                numeric_df["TN"],
            )

        for metric_cfg in metric_definitions:
            series = None
            for alias in metric_cfg["aliases"]:
                if alias in numeric_df.columns:
                    series = numeric_df[alias]
                    break
            if series is None:
                continue
            # Use raw values (drop NA) for distributions — no IQR outlier removal
            cleaned = series.dropna()
            if cleaned.empty:
                continue
            for value in cleaned:
                distribution_records.append(
                    {
                        "metric": metric_cfg["display"],
                        "value": float(value),
                        "Phase": phase_name,
                    }
                )

    if distribution_records:
        dist_df = pd.DataFrame(distribution_records)
        metric_order = [cfg["display"] for cfg in metric_definitions if cfg["display"] in dist_df["metric"].unique()]
        metric_candidates = [(metric, dist_df[dist_df["metric"] == metric]) for metric in metric_order]
        metric_candidates = [(metric, subset) for metric, subset in metric_candidates if not subset.empty]

        n_metrics = len(metric_candidates)
        n_cols = 3
        n_rows = int(np.ceil(n_metrics / n_cols))
        panel_c_spec = GridSpecFromSubplotSpec(
            n_rows,
            n_cols,
            subplot_spec=gs2[1, :2],
            wspace=0.25,
            hspace=0.55,
        )

        facet_axes: list[plt.Axes] = []
        empty_axes: list[plt.Axes] = []
        phase_order = ["Preliminary Phase", "Dummy Phase"]

        for idx, (metric, subset) in enumerate(metric_candidates):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig2.add_subplot(panel_c_spec[row, col])

            metric_color = get_metric_color(metric)
            palette_lookup = {
                "Dummy Phase": darken_color(metric_color, 0.18),
                "Preliminary Phase": lighten_color(metric_color, 0.25),
            }

            for phase in phase_order:
                phase_subset = subset[subset["Phase"] == phase]
                values = phase_subset["value"].dropna()
                if len(values.unique()) < 2:
                    if not values.empty:
                        ax.axvline(values.iloc[0], color=palette_lookup.get(phase, metric_color), linestyle="--", linewidth=1.0)
                    continue
                alpha_val = 0.75 if phase == "Dummy Phase" else 0.35
                sns.kdeplot(
                    data=phase_subset,
                    x="value",
                    fill=True,
                    common_norm=False,
                    alpha=alpha_val,
                    linewidth=1.2,
                    color=palette_lookup.get(phase, metric_color),
                    ax=ax,
                )

            ax.set_title(metric, fontweight="bold", fontsize=11)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0)
            ax.set_ylabel("")
            ax.grid(True, alpha=0.2, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=10)

            # Adjust y-range to leave room for annotation text if needed later.
            ymin, ymax = ax.get_ylim()
            if ymax > 0:
                ax.set_ylim(ymin, ymax * 1.15)

            legend_obj = ax.get_legend()
            if legend_obj is not None:
                legend_obj.remove()

            facet_axes.append(ax)

        total_slots = n_rows * n_cols
        if total_slots > len(metric_candidates):
            for idx in range(len(metric_candidates), total_slots):
                row = idx // n_cols
                col = idx % n_cols
                empty_ax = fig2.add_subplot(panel_c_spec[row, col])
                empty_ax.axis("off")
                empty_axes.append(empty_ax)

        sample_color = get_metric_color(metric_candidates[0][0]) if metric_candidates else "#666666"
        legend_handles = [
            Patch(
                facecolor=darken_color(sample_color, 0.18),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.55,
                label="Dummy Phase",
            ),
            Patch(
                facecolor=lighten_color(sample_color, 0.25),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.55,
                label="Preliminary Phase",
            ),
        ]

        legend_ax: plt.Axes
        if empty_axes:
            legend_ax = empty_axes[0]
        else:
            first_ax = facet_axes[0]
            bbox = first_ax.get_position()
            legend_width = 0.08
            legend_ax = fig2.add_axes([
                min(bbox.x1 + 0.015, 0.95 - legend_width),
                bbox.y0 + 0.05,
                legend_width,
                bbox.height * 0.6,
            ])
            legend_ax.axis("off")

        legend_ax.legend(
            handles=legend_handles,
            title="Phase",
            frameon=False,
            loc="upper left",
            fontsize=10,
            title_fontsize=11,
        )

        x0 = min(ax.get_position().x0 for ax in facet_axes)
        x1 = max(ax.get_position().x1 for ax in facet_axes)
        y0 = min(ax.get_position().y0 for ax in facet_axes)
        y1 = max(ax.get_position().y1 for ax in facet_axes)

        fig2.text(
            (x0 + x1) / 2,
            y0 - 0.055,
            "Metric Value",
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=12,
        )
        fig2.text(
            x0 - 0.035,
            (y0 + y1) / 2,
            "Density",
            ha="center",
            va="center",
            rotation=90,
            fontweight="bold",
            fontsize=12,
        )

        panel_anchor = facet_axes[0].get_position()
        fig2.text(
            panel_anchor.x0,
            panel_anchor.y1 + 0.04,
            "C. Metric Distributions by Phase",
            fontsize=13,
            fontweight="bold",
        )

    # Panel D - Correlation heatmap using remaining row 2 space
    ax2d = fig2.add_subplot(gs2[1, 2:])
    dummy_phase_ranked = dummy_phase.copy()
    dummy_phase_ranked["Rank"] = dummy_phase_ranked["final_score"].rank(ascending=False)
    if "TP" in dummy_phase_ranked.columns and "FP" in dummy_phase_ranked.columns:
        denom = dummy_phase_ranked["TP"] + dummy_phase_ranked["FP"]
        dummy_phase_ranked["Precision"] = np.where(denom > 0, dummy_phase_ranked["TP"] / denom, np.nan)
    else:
        dummy_phase_ranked["Precision"] = np.nan

    if {"TP", "FP", "FN", "TN"}.issubset(dummy_phase_ranked.columns):
        dummy_phase_ranked["MCC"] = compute_mcc(
            dummy_phase_ranked["TP"],
            dummy_phase_ranked["FP"],
            dummy_phase_ranked["FN"],
            dummy_phase_ranked["TN"],
        )

    # Compute normalized inference time (i_norm) using scale lookup if possible and include it
    heatmap_df = dummy_phase_ranked.copy()
    try:
        center, scale = scale_lookup.get("Inference Time", (None, None))
        if center is not None and "inference_time" in heatmap_df.columns:
            heatmap_df["Inference Time (norm)"] = (
                pd.to_numeric(heatmap_df["inference_time"], errors="coerce") - center
            ) / scale
    except Exception:
        pass
    # Keep human-friendly label for Net.Benefit only; do not include final_score (scaled weighted score)
    rename_map = {"Net.Benefit": "Net Benefit"}
    heatmap_df = heatmap_df.rename(columns=rename_map)

    # Use parsimony as-is (do not invert). Prefer the processed column name `parsimony_score`
    # but accept title-cased variants if present. This ensures the heatmap shows the raw
    # parsimony measure rather than 1 - parsimony.
    parsimony_col_for_heatmap = None
    if "parsimony_score" in heatmap_df.columns:
        parsimony_col_for_heatmap = "parsimony_score"
    elif "Parsimony Score" in heatmap_df.columns:
        parsimony_col_for_heatmap = "Parsimony Score"

    selected_metrics = ["Sensitivity", "AUPRC", "Net Benefit", "F1", "ECE"]
    # Insert parsimony column name if available
    other_metrics = ["AUC", "Specificity", "Precision", "MCC"]
    if parsimony_col_for_heatmap is not None:
        other_metrics.append(parsimony_col_for_heatmap)
    # add normalized inference time if present
    if "Inference Time (norm)" in heatmap_df.columns:
        other_metrics.append("Inference Time (norm)")
    confusion_metrics = ["TP", "FP", "TN", "FN", "Rank"]
    heatmap_metrics = selected_metrics + other_metrics + confusion_metrics
    available_metrics = [m for m in heatmap_metrics if m in heatmap_df.columns]

    if len(available_metrics) >= 3:
        # Use Spearman correlation (rank-based) and ensure AUC outliers already removed
        corr_matrix = heatmap_df[available_metrics].corr(method="spearman")

        from matplotlib.patches import Rectangle

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            ax=ax2d,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )

        confusion_set = {"TP", "FP", "TN", "FN"}
        rank_idx = available_metrics.index("Rank") if "Rank" in available_metrics else None

        for i in range(len(available_metrics)):
            for j in range(len(available_metrics)):
                if j > i:
                    continue
                value = corr_matrix.iloc[i, j]
                label_i = available_metrics[i]
                label_j = available_metrics[j]
                highlight = label_i == "Rank" or label_j == "Rank"
                text_color = "black" if highlight or abs(value) < 0.6 else "white"
                ax2d.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if highlight else "normal",
                    color=text_color,
                )

        if rank_idx is not None:
            for idx, metric in enumerate(available_metrics):
                if metric in confusion_set and idx >= rank_idx:
                    rect = Rectangle((rank_idx, idx), 1.0, 1.0, fill=False, edgecolor="#1f77b4", linewidth=2)
                    ax2d.add_patch(rect)
                if metric in confusion_set and idx < rank_idx:
                    rect = Rectangle((idx, rank_idx), 1.0, 1.0, fill=False, edgecolor="#1f77b4", linewidth=2)
                    ax2d.add_patch(rect)

        ax2d.set_xticks(np.arange(len(available_metrics)) + 0.5)
        ax2d.set_yticks(np.arange(len(available_metrics)) + 0.5)
        label_map = {"Net Benefit": "Net Benefit"}
        display_labels = [label_map.get(m, m) for m in available_metrics]
        ax2d.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=11)
        ax2d.set_yticklabels(display_labels, fontsize=11)
        ax2d.set_aspect("auto")
        for label in ax2d.get_xticklabels():
            label.set_fontweight("bold" if label.get_text() == "Rank" else "normal")
        for label in ax2d.get_yticklabels():
            label.set_fontweight("bold" if label.get_text() == "Rank" else "normal")
        ax2d.tick_params(length=0, labelsize=10)
        rect = Rectangle(
            (0, 0),
            sum(1 for m in selected_metrics if m in available_metrics),
            sum(1 for m in selected_metrics if m in available_metrics),
            fill=False,
            edgecolor="#222222",
            linewidth=2.5,
            linestyle="--",
        )
        ax2d.add_patch(rect)
        cbar_c = fig2.colorbar(ax2d.collections[0], ax=ax2d, fraction=0.046, pad=0.04)
        cbar_c.set_label("Correlation", fontweight="bold", fontsize=12)
        cbar_c.ax.tick_params(labelsize=10)

        ax2d.set_title(
            "D. Metric Correlations with Ranking (Dummy Phase)",
            fontweight="bold",
            fontsize=13,
        )

    # Panels F and H were moved to Supplementary figures to streamline the main figure.

    output_path = FIGURES_DIR / "results_figure2_factor_analysis.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path


def main() -> None:
    generate_figure2()


if __name__ == "__main__":
    main()
