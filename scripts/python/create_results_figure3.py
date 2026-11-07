#!/usr/bin/env python3
"""Generate Figure 3: Final-phase performance overview."""

from __future__ import annotations

from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, TwoSlopeNorm

from plot_palette import BASE_PALETTE, darken_color, get_metric_color, lighten_color


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PRELIM_DIR = ROOT / "optimal_threshold_results"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures" / "main_figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


plt.style.use("seaborn-v0_8-paper")
sns.set_palette(BASE_PALETTE)
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


def compute_mcc(tp: pd.Series, fp: pd.Series, fn: pd.Series, tn: pd.Series) -> pd.Series:
    """Return Matthews Correlation Coefficient given confusion matrix components."""
    tp_f = tp.astype(float)
    fp_f = fp.astype(float)
    fn_f = fn.astype(float)
    tn_f = tn.astype(float)
    numerator = tp_f * tn_f - fp_f * fn_f
    denom = np.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
    with np.errstate(divide="ignore", invalid="ignore"):
        mcc = np.where(denom > 0, numerator / denom, np.nan)
    return pd.Series(mcc, index=tp.index)


def compute_treat_all_nb(df: pd.DataFrame) -> pd.Series:
    """Return treat-all net benefit values for submissions with confusion data and thresholds."""
    required_cols = {"TP", "FP", "FN", "TN", "Optimal Threshold"}
    if not required_cols.issubset(df.columns):
        return pd.Series(dtype=float)

    numerator = df["TP"] + df["FN"]
    denominator = df["TP"] + df["FP"] + df["FN"] + df["TN"]
    prevalence = numerator / denominator.where(denominator != 0, np.nan)
    pt = df["Optimal Threshold"].astype(float).clip(1e-6, 1 - 1e-6)
    treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    treat_all = treat_all.replace([np.inf, -np.inf], np.nan)
    return treat_all.dropna()


def extract_team_name(raw: str) -> str:
    """Return a compact ASCII team label for plotting."""
    if pd.isna(raw):
        return "Unknown"
    name = raw.replace("output_", "").strip()
    name = name.split("_https")[0]
    name = name.split("_git")[0]
    # Drop emoji/flags and trim whitespace
    name = re.sub(r"[^\x00-\x7F]", "", name)
    return name.strip()


def build_contribution_table(top_scores: pd.DataFrame) -> pd.DataFrame:
    """Use the frozen primary formula, with components in displayed score units."""
    params = json.loads((PROCESSED_DIR / "scale_params.json").read_text())
    loadings = json.loads((PROCESSED_DIR / "factor_loadings.json").read_text())
    features = ["F1", "AUPRC", "Net.Benefit", "ECE"]
    centers = np.asarray(params["center"][:4], dtype=float)
    scales = np.asarray(params["scale"][:4], dtype=float)
    weights = np.asarray([loadings[f] for f in features])
    values = .9 * (top_scores[features].to_numpy(float) - centers) / scales * weights / 1.9692
    result = pd.DataFrame(values, columns=features, index=top_scores.index)
    result["Parsimony (1 - score)"] = .05 * (1 - top_scores["Parsimony Score"]) / 1.9692
    result["Inference Speed"] = -.05 * (top_scores["Inference Time"] - params["center"][4]) / params["scale"][4] / 1.9692
    np.testing.assert_allclose(result.sum(axis=1), top_scores["Scaled Weighted Score"], atol=1e-4)
    result.index = top_scores["Team_clean"].to_numpy()
    return result


def generate_figure3() -> Path:
    """Create Figure 3 and return the output path.

    Load final_scores once, merge any extended fields, compute MCC and FPR where
    possible, then build the figure panels (A-F). This removes duplicate loads
    and guarantees Panel D only shows individual alternative-metric rankings.
    """
    # Load and process final_scores (single canonical load)
    final_scores = pd.read_csv(RAW_DIR / "final_scores.csv")
    final_scores["sub_num"] = pd.to_numeric(final_scores["Submission"], errors="raise").astype(int)
    final_scores["Team_clean"] = final_scores["Team name"].apply(extract_team_name)

    if "Parsimony Score" in final_scores.columns:
        final_scores["parsimony_score"] = final_scores["Parsimony Score"]
    if "Inference Time" in final_scores.columns:
        final_scores["inference_time"] = final_scores["Inference Time"]
    # Preserve original column for plotting while adding a dot notation for compatibility
    if "Net Benefit" in final_scores.columns:
        final_scores["Net.Benefit"] = final_scores["Net Benefit"]

    if {"TP", "FP", "FN", "TN"}.issubset(final_scores.columns):
        final_scores["MCC"] = compute_mcc(
            final_scores["TP"], final_scores["FP"], final_scores["FN"], final_scores["TN"]
        )

    # Ensure FPR column exists and compute where possible
    if "FPR" not in final_scores.columns:
        final_scores["FPR"] = np.nan
    if final_scores["FPR"].isna().all():
        if {"FP", "TN"}.issubset(final_scores.columns):
            fp_tn_sum = final_scores["FP"] + final_scores["TN"]
            valid_mask = fp_tn_sum > 0
            final_scores.loc[valid_mask, "FPR"] = final_scores.loc[valid_mask, "FP"] / fp_tn_sum[valid_mask]
        elif "Specificity" in final_scores.columns:
            final_scores["FPR"] = 1 - final_scores["Specificity"]
    final_scores["FPR"] = pd.to_numeric(final_scores.get("FPR", pd.Series(dtype=float)), errors="coerce").clip(lower=0)
    final_scores["FPR_pct"] = final_scores["FPR"] * 100

    print("\nCreating Figure 3: Final-phase performance overview...")

    # (Panel E data loaded inline below)

    # Slightly wider figure; reduce the relative height of the first two rows
    fig3 = plt.figure(figsize=(26, 26))
    # Row 3 is a vertical CI chart (submissions on x-axis) — moderate height.
    gs3 = GridSpec(
        3,
        1,
        figure=fig3,
        height_ratios=[1.05, 1.10, 0.85],
        hspace=0.43,
    )
    top_spec    = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[0], width_ratios=[1, 1], wspace=0.28)
    middle_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[1], width_ratios=[1.10, 0.90], wspace=0.14)

    # Panel B — Rank correlations heatmap (lower triangle)
    heatmap_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=top_spec[0, 1], width_ratios=[24, 1], wspace=0.1)
    ax3a = fig3.add_subplot(heatmap_spec[0, 0])
    cbar_ax3a = fig3.add_subplot(heatmap_spec[0, 1])
    # Derive additional performance metrics for correlation analysis
    if {"TP", "FN"}.issubset(final_scores.columns):
        tp_fp = final_scores["TP"] + final_scores["FP"]
        final_scores["Precision"] = np.where(tp_fp > 0, final_scores["TP"] / tp_fp, np.nan)

    # Compute normalized inference time from scale params if available so it is included in the heatmap
    try:
        scale_df = pd.read_csv(PROCESSED_DIR / "scale_params.csv")
        scale_lookup = {row.feature: (row.center, row.scale) for _, row in scale_df.iterrows()}
        if "inference_time" in final_scores.columns:
            center, scale = scale_lookup.get("Inference Time", (0.0, 1.0))
            final_scores["Inference Time (norm)"] = (
                pd.to_numeric(final_scores["inference_time"], errors="coerce") - center
            ) / scale
        if 'prelim_scores' in locals() and "inference_time" in prelim_scores.columns:
            center, scale = scale_lookup.get("Inference Time", (0.0, 1.0))
            prelim_scores["Inference Time (norm)"] = (
                pd.to_numeric(prelim_scores["inference_time"], errors="coerce") - center
            ) / scale
    except Exception:
        pass

    # Derived metrics requested by the user:
    # - Parsimony (1 - score): treat lower parsimony_score as better by inverting
    # - Inference Speed: inverse of inference_time (higher is faster)
    if "parsimony_score" in final_scores.columns:
        final_scores["Parsimony (1-score)"] = 1.0 - pd.to_numeric(final_scores["parsimony_score"], errors="coerce")
    if "inference_time" in final_scores.columns:
        # avoid divide-by-zero; treat non-positive times as NaN
        inf_time = pd.to_numeric(final_scores["inference_time"], errors="coerce")
        final_scores["Inference Speed"] = np.where(inf_time > 0, 1.0 / inf_time, np.nan)

    selected_metrics = ["Sensitivity", "AUPRC", "Net Benefit", "F1", "ECE"]
    # Swap order: Inference Speed before Parsimony (1-score)
    other_metrics = ["AUC", "Specificity", "Precision", "MCC", "Inference Speed", "Parsimony (1-score)"]
    confusion_metrics = ["TP", "FP", "TN", "FN", "Rank"]
    heatmap_metrics = selected_metrics + other_metrics + confusion_metrics
    available_metrics = [m for m in heatmap_metrics if m in final_scores.columns]

    if len(available_metrics) >= 3:
        # Use Spearman rank correlation for robustness to non-normal metric distributions
        corr_matrix = final_scores[available_metrics].corr(method="spearman")
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            ax=ax3a,
            cbar=True,
            cbar_ax=cbar_ax3a,
            linewidths=0.5,
            linecolor="white",
        )

        confusion_set = {"TP", "FP", "FN", "TN"}
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
                ax3a.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold" if highlight else "normal",
                    color=text_color,
                )

        if rank_idx is not None:
            for idx, metric in enumerate(available_metrics):
                if metric in confusion_set and idx >= rank_idx:
                    rect = Rectangle((rank_idx, idx), 1.0, 1.0, fill=False, edgecolor="#1f77b4", linewidth=2)
                    ax3a.add_patch(rect)
                if metric in confusion_set and idx < rank_idx:
                    rect = Rectangle((idx, rank_idx), 1.0, 1.0, fill=False, edgecolor="#1f77b4", linewidth=2)
                    ax3a.add_patch(rect)

        ax3a.set_xticks(np.arange(len(available_metrics)) + 0.5)
        ax3a.set_yticks(np.arange(len(available_metrics)) + 0.5)
        label_map = {"Net Benefit": "Net Benefit"}
        display_labels = [label_map.get(m, m) for m in available_metrics]
        ax3a.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=13)
        # Set y-axis labels horizontal (0 degrees) as requested
        ax3a.set_yticklabels(display_labels, rotation=0, ha="right", fontsize=13)
        ax3a.set_aspect("auto")
        for label in ax3a.get_xticklabels():
            if label.get_text() == "Rank":
                label.set_fontweight("bold")
            else:
                label.set_fontweight("normal")
        for label in ax3a.get_yticklabels():
            if label.get_text() == "Rank":
                label.set_fontweight("bold")
            else:
                label.set_fontweight("normal")
        ax3a.tick_params(length=0, labelsize=10)
        rect = Rectangle(
            (0, 0),
            sum(1 for m in selected_metrics if m in available_metrics),
            sum(1 for m in selected_metrics if m in available_metrics),
            fill=False,
            edgecolor="#222222",
            linewidth=2.5,
            linestyle="--",
        )
        ax3a.add_patch(rect)
        cbar_ax3a.set_ylabel("Correlation", fontweight="bold", fontsize=14)
        cbar_ax3a.yaxis.set_label_position("right")
        cbar_ax3a.tick_params(length=0, labelsize=10)
        ax3a.set_title("B. Rank Correlations with Evaluation Metrics", fontweight="bold", fontsize=16)

    # Panel A — Metric distributions (Final vs Preliminary)
    metric_candidates = [
        {"display": "F1", "final": "F1", "prelim": ["F1"]},
        {"display": "AUPRC", "final": "AUPRC", "prelim": ["AUPRC"]},
        {"display": "Net Benefit", "final": "Net Benefit", "prelim": ["Net Benefit", "Net.Benefit"]},
        {"display": "ECE", "final": "ECE", "prelim": ["ECE"]},
        {"display": "AUC", "final": "AUC", "prelim": ["AUC"]},
        {"display": "Sensitivity", "final": "Sensitivity", "prelim": ["Sensitivity"]},
        {"display": "Specificity", "final": "Specificity", "prelim": ["Specificity"]},
        {"display": "MCC", "final": "MCC", "prelim": ["MCC"]},
    ]
    try:
        prelim_scores = pd.read_csv(PRELIM_DIR / "evaluation_metrics_sensitivity_method.csv")
        prelim_scores = prelim_scores.query("AUC >= 0.5 and AUC < 0.99").copy()
    except FileNotFoundError:
        prelim_scores = pd.DataFrame()
    else:
        if {"TP", "FP", "FN", "TN"}.issubset(prelim_scores.columns):
            prelim_scores["MCC"] = compute_mcc(
                prelim_scores["TP"],
                prelim_scores["FP"],
                prelim_scores["FN"],
                prelim_scores["TN"],
            )

    # (Normalized inference time and parsimony inclusion handled earlier before heatmap build)

    long_records: list[dict[str, str | float]] = []
    for candidate in metric_candidates:
        display = candidate["display"]
        final_col = candidate["final"]
        if final_col in final_scores.columns:
            for value in final_scores[final_col].dropna():
                long_records.append({"metric": display, "value": value, "Phase": "Final"})

        prelim_col = next((col for col in candidate["prelim"] if col in prelim_scores.columns), None)
        if prelim_col is not None:
            for value in prelim_scores[prelim_col].dropna():
                long_records.append({"metric": display, "value": value, "Phase": "Preliminary"})

    if long_records:
        dist_df = pd.DataFrame(long_records)
        nb_mask = dist_df["metric"] == "Net Benefit"
        # No IQR trimming for distribution plots — use raw values for panel A (consistent with Figure 2)

        phase_levels = dist_df["Phase"].unique().tolist()

        metrics_with_data: list[tuple[str, pd.DataFrame]] = []
        for candidate in metric_candidates:
            display = candidate["display"]
            subset = dist_df[dist_df["metric"] == display]
            if not subset.empty:
                metrics_with_data.append((display, subset))

        if metrics_with_data:
            n_metrics = len(metrics_with_data)
            n_cols = 3
            n_rows = int(np.ceil(n_metrics / n_cols))
            facet_spec = GridSpecFromSubplotSpec(
                n_rows,
                n_cols,
                subplot_spec=top_spec[0, 0],
                # tighten horizontal gaps between facet columns
                wspace=0.32,
                # reduce vertical spacing between facet rows to bring plots closer
                hspace=0.65,
            )

            facet_axes: list[Axes] = []
            empty_axes: list[Axes] = []
            for idx, (display, subset) in enumerate(metrics_with_data):
                row = idx // n_cols
                col = idx % n_cols
                ax = fig3.add_subplot(facet_spec[row, col])

                metric_color = get_metric_color(display)
                phase_palette = {
                    "Final": darken_color(metric_color, 0.18),
                    "Preliminary": lighten_color(metric_color, 0.25),
                }

                phase_order = [phase for phase in ["Final", "Preliminary"] if phase in subset["Phase"].unique()]
                if not phase_order:
                    phase_order = subset["Phase"].unique().tolist()

                for phase in phase_order:
                    phase_subset = subset[subset["Phase"] == phase]
                    values = phase_subset["value"].dropna()
                    # Re-enable IQR-based outlier trimming for Net Benefit only (as before)
                    if display == "Net Benefit" and not values.empty:
                        q1 = values.quantile(0.25)
                        q3 = values.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        trimmed = values[(values >= lower) & (values <= upper)]
                        # if trimming removed all points, fall back to untrimmed
                        if not trimmed.empty:
                            phase_subset = phase_subset[phase_subset["value"].between(lower, upper)]
                            values = trimmed
                    color = phase_palette.get(phase, metric_color)
                    if len(values.unique()) < 2:
                        if not values.empty:
                            ax.axvline(values.iloc[0], color=color, linestyle="--", linewidth=1.0)
                        continue

                    # Use same alpha for Final as Dummy in Figure 2 (Final=0.75), Preliminary=0.35
                    alpha_val = 0.75 if phase == "Final" else 0.35
                    sns.kdeplot(
                        data=phase_subset,
                        x="value",
                        fill=True,
                        common_norm=False,
                        linewidth=1.1,
                        alpha=alpha_val,
                        color=color,
                        ax=ax,
                    )

                legend_obj = ax.get_legend()
                if legend_obj is not None:
                    legend_obj.remove()

                ymin, ymax = ax.get_ylim()
                buffer = 0.18 if ymax > 0 else 0.05
                if ymax > 0:
                    ax.set_ylim(ymin, ymax * (1 + buffer))

                ax.set_title(display, fontweight="bold", fontsize=13)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)
                ax.set_ylabel("")
                ax.tick_params(axis="both", labelsize=10)

                ax.grid(True, alpha=0.2, axis="x")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                facet_axes.append(ax)

            total_slots = n_rows * n_cols
            if total_slots > len(metrics_with_data):
                for idx in range(len(metrics_with_data), total_slots):
                    row = idx // n_cols
                    col = idx % n_cols
                    empty_ax = fig3.add_subplot(facet_spec[row, col])
                    empty_ax.axis("off")
                    empty_axes.append(empty_ax)

            if phase_levels and facet_axes:
                sample_color = get_metric_color(metrics_with_data[0][0]) if metrics_with_data else "#666666"
                legend_palette = {
                    "Final": darken_color(sample_color, 0.18),
                    "Preliminary": lighten_color(sample_color, 0.25),
                }
                phase_label_map = {
                    "Final": "Final Phase",
                    "Preliminary": "Preliminary Phase",
                }
                legend_handles = [
                    Patch(
                        facecolor=legend_palette[level],
                        edgecolor="black",
                        linewidth=0.6,
                        alpha=0.6,
                        label=phase_label_map.get(level, level),
                    )
                    for level in ["Final", "Preliminary"]
                    if level in phase_levels
                ]

                if empty_axes:
                    legend_ax = empty_axes[0]
                else:
                    x1 = max(ax.get_position().x1 for ax in facet_axes)
                    y0 = min(ax.get_position().y0 for ax in facet_axes)
                    y1 = max(ax.get_position().y1 for ax in facet_axes)
                    legend_width = 0.07
                    legend_x = min(x1 + 0.015, 0.98 - legend_width)
                    legend_ax = fig3.add_axes([legend_x, y0, legend_width, y1 - y0])
                    legend_ax.axis("off")

                legend_ax.legend(
                    handles=legend_handles,
                    title="Phase",
                    loc="upper left",
                    frameon=False,
                    fontsize=12,
                    title_fontsize=13,
                )

                x0 = min(ax.get_position().x0 for ax in facet_axes)
                x1 = max(ax.get_position().x1 for ax in facet_axes)
                y0 = min(ax.get_position().y0 for ax in facet_axes)
                y1 = max(ax.get_position().y1 for ax in facet_axes)

                # increase vertical margin for the faceted x-axis label so it doesn't crowd the plots
                # place the faceted x-axis label slightly closer to the plots
                fig3.text(
                    (x0 + x1) / 2,
                    y0 - 0.02,
                    "Metric Value",
                    ha="center",
                    va="top",
                    fontweight="bold",
                    fontsize=14,
                )
                fig3.text(
                    x0 - 0.035,
                    (y0 + y1) / 2,
                    "Density",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontweight="bold",
                    fontsize=14,
                )

                panel_anchor = facet_axes[0].get_position()
                fig3.text(
                    panel_anchor.x0,
                    panel_anchor.y1 + 0.035,
                    "A. Metric Distributions by Phase",
                    fontsize=15,
                    fontweight="bold",
                )
        else:
            ax3b_placeholder = fig3.add_subplot(top_spec[0, 0])
            ax3b_placeholder.axis("off")
            ax3b_placeholder.text(
                0.5,
                0.5,
                "Metric distributions unavailable",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
    else:
        ax3b_placeholder = fig3.add_subplot(top_spec[0, 0])
        ax3b_placeholder.axis("off")
        ax3b_placeholder.text(
            0.5,
            0.5,
            "Metric distributions unavailable",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Panel C — Lollipop: weighted metric contributions (top 10)
    ax3c = fig3.add_subplot(middle_spec[0, 0])

    top_ranked   = final_scores.sort_values("Scaled Weighted Score", ascending=False, kind="stable").head(10).copy().reset_index(drop=True)
    contrib_df   = build_contribution_table(top_ranked).fillna(0.0)

    try:
        _ci_c = pd.read_csv(PROCESSED_DIR / "bootstrap_composite_ci.csv")
        ci_top10 = _ci_c.sort_values("scaled_orig", ascending=False).head(10).reset_index(drop=True)
    except Exception:
        ci_top10 = pd.DataFrame()

    _c_all_metrics  = ["F1", "AUPRC", "Net.Benefit", "ECE", "Parsimony (1 - score)", "Inference Speed"]
    _c_all_labels   = ["F1", "AUPRC", "Net Benefit", "ECE (inv.)", "Parsimony", "Inf. Speed"]
    _c_all_colors   = [
        get_metric_color("F1"),
        get_metric_color("AUPRC"),
        get_metric_color("Net Benefit"),
        "#D55E00",
        get_metric_color("Parsimony"),
        "#4B0082",
    ]
    _c_data         = contrib_df[_c_all_metrics].values  # shape: (submissions, metrics)

    _c_N   = len(_c_data)   # number of submissions
    _c_x   = np.arange(_c_N)  # best (left) → worst (right)
    _c_off = np.linspace(-0.30, 0.30, 6)

    # Uniform color for score annotations
    _c_dot_color = "#3a6cc4"

    if not ci_top10.empty:
        _c_raw    = _c_data.sum(axis=1)
        _c_scaled = ci_top10["scaled_orig"].values
        _c_lo     = ci_top10["scaled_lo_95"].values
        _c_hi     = ci_top10["scaled_hi_95"].values
        _c_labels = [f"{r.Team_clean} (sub{int(r.sub_num)})" for r in top_ranked.itertuples()]
    else:
        _c_raw    = np.zeros(_c_N)
        _c_scaled = np.zeros(_c_N)
        _c_lo     = np.zeros(_c_N)
        _c_hi     = np.zeros(_c_N)
        _c_labels = [f"{t}" for r, t in zip(top_ranked["Rank"], top_ranked["Team_clean"])]

    # Alternating column bands — one band per team
    for i, x in enumerate(_c_x):
        ax3c.axvspan(x - 0.5, x + 0.5, color="#f5f5f5" if i % 2 == 0 else "white", zorder=0)
    ax3c.axhline(0, color="#aaa", lw=1.0, linestyle="--", zorder=1)

    # Grouped vertical lollipops — 6 metrics per team, offset along x
    for m_idx, m_col in enumerate(_c_all_colors):
        xo = _c_off[m_idx]
        for i, x in enumerate(_c_x):
            v  = _c_data[i, m_idx]
            xx = x + xo
            ax3c.plot([xx, xx], [0, v], color=m_col, lw=1.2, alpha=0.55, zorder=2)
            ax3c.plot(xx, v, "o", markersize=6.5, color=m_col,
                      markeredgewidth=0.4, markeredgecolor="#333", alpha=0.9, zorder=3)

    # Weighted sum annotation (raw score) — placed inside the plot area to avoid legend overlap
    _c_ymax_data = float(np.nanmax(_c_data)) if np.isfinite(np.nanmax(_c_data)) else 0.0
    _c_ymin_data = float(np.nanmin(_c_data)) if np.isfinite(np.nanmin(_c_data)) else 0.0
    _c_y_pad = max(0.08, 0.10 * (_c_ymax_data - _c_ymin_data))
    # Place annotation above the data to avoid colliding with the lollipops
    _c_y_ann = _c_ymax_data + _c_y_pad * 0.25
    for x, rs in zip(_c_x, _c_raw):
        ax3c.text(x + 0.45, _c_y_ann, f"{rs:.2f}", va="bottom", ha="center",
                  fontsize=11, fontweight="bold", color=_c_dot_color)
    ax3c.text(_c_x[0] + 0.10, _c_y_ann, "Scaled\nsum", va="bottom", ha="right",
              fontsize=10.5, color="#555", style="italic")

    ax3c.set_xticks(_c_x)
    ax3c.set_xticklabels(_c_labels, rotation=30, ha="right", fontsize=12)
    ax3c.tick_params(length=0)
    ax3c.set_ylabel("Weighted Contribution to Composite Score", fontweight="bold", fontsize=13)
    ax3c.spines["top"].set_visible(False)
    ax3c.spines["right"].set_visible(False)
    ax3c.set_xlim(-0.65, _c_N - 0.05)
    _c_ylim = max(abs(_c_ymax_data), abs(_c_ymin_data)) + _c_y_pad * 2.0
    if _c_ylim <= 0:
        _c_ylim = 0.5
    ax3c.set_ylim(_c_ymin_data - _c_y_pad * 2.5, _c_y_ann + _c_y_pad * 0.55)
    ax3c.set_title(
        "C. Weighted Metric Contributions: Top 10 Finalist Submissions",
        fontweight="bold", fontsize=15, pad=10, loc="left",
    )

    # Place the metric legend in the lower-left open area of the panel.
    _c_leg_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_c_all_colors[j],
               markeredgecolor="#333", markeredgewidth=0.4, markersize=8.5,
               label=_c_all_labels[j])
        for j in range(6)
    ]
    ax3c.legend(
        handles=_c_leg_handles, loc="lower left", bbox_to_anchor=(0.01, 0.03),
        fontsize=10.5, framealpha=0.93, edgecolor="#ccc",
        title="Metric", title_fontsize=11.5, ncol=6,
        handletextpad=0.4, columnspacing=0.8, borderpad=0.6,
    )

    # Panel D — Individual Alt Ranking plots (X: Alt Rank by metric, Y: Composite Rank, color: FPR)
    individual_metrics = [
        {"name": "AUC", "title": "AUC"},
        {"name": "AUPRC", "title": "AUPRC"},
        {"name": "F1", "title": "F1"},
        {"name": "MCC", "title": "MCC"},
    ]
    n_panel_d = len(individual_metrics)
    n_rows_d = int(np.ceil(n_panel_d / 2))
    facet_d_spec = GridSpecFromSubplotSpec(n_rows_d, 2, subplot_spec=middle_spec[0, 1], wspace=0.28, hspace=0.35)
    # (removed duplicate _c_data line)
    # Ensure FPR column exists
    if "FPR" not in final_scores.columns:
        final_scores["FPR"] = np.nan
    # Compute FPR where possible
    if final_scores["FPR"].isna().all():
        if {"FP", "TN"}.issubset(final_scores.columns):
            fp_tn_sum = final_scores["FP"] + final_scores["TN"]
            final_scores.loc[fp_tn_sum > 0, "FPR"] = final_scores.loc[fp_tn_sum > 0, "FP"] / fp_tn_sum[fp_tn_sum > 0]
        elif "Specificity" in final_scores.columns:
            final_scores["FPR"] = 1 - final_scores["Specificity"]
    final_scores["FPR"] = pd.to_numeric(final_scores["FPR"], errors="coerce").clip(lower=0)
    final_scores["FPR_pct"] = final_scores["FPR"] * 100

    facet_axes_d: list[Axes] = []
    last_scatter = None
    for idx, metric_info in enumerate(individual_metrics):
        row = idx // 2
        col = idx % 2
        ax = fig3.add_subplot(facet_d_spec[row, col])
        facet_axes_d.append(ax)
        metric_name = metric_info["name"]

        if metric_name not in final_scores.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{metric_name} unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            continue

        # Compute Alt Rank using dense ranking (ECE lower is better; dense avoids gaps from ties)
        if metric_name == "ECE":
            alt_rank_series = final_scores[metric_name].rank(ascending=True, method="dense")
        else:
            alt_rank_series = final_scores[metric_name].rank(ascending=False, method="dense")

        plot_df = final_scores[["Team name", "Team_clean", "Rank", metric_name, "FPR_pct"]].copy()
        plot_df["Alt Rank"] = alt_rank_series
        plot_df = plot_df.dropna(subset=["Rank", metric_name])
        if plot_df.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            continue

        # Sort for plotting clarity
        plot_df = plot_df.sort_values("Alt Rank")

        # Plot: X = Alt Rank, Y = False Positive Rate (%), color = Composite Rank
        # Compute rank difference (Alt Rank - Composite Rank) and use it as color
        plot_df["Rank Diff"] = plot_df["Alt Rank"] - plot_df["Rank"]
        rd_min = plot_df["Rank Diff"].min()
        rd_max = plot_df["Rank Diff"].max()
        rd_bound = max(abs(rd_min) if not np.isnan(rd_min) else 0, abs(rd_max) if not np.isnan(rd_max) else 0)
        norm = TwoSlopeNorm(vcenter=0, vmin=-rd_bound, vmax=rd_bound) if rd_bound > 0 else None

        sc = ax.scatter(
            plot_df["Alt Rank"],
            plot_df["FPR_pct"],
            c=plot_df["Rank Diff"],
            cmap="coolwarm",
            norm=norm,
            edgecolor="#1a1a1a",
            linewidth=0.5,
            alpha=0.95,
            s=55,
        )
        last_scatter = sc

        # Highlight composite top-5 with rings at their FPR positions
        comp_top = plot_df[plot_df["Rank"] <= 5]
        if not comp_top.empty:
            ax.scatter(
                comp_top["Alt Rank"],
                comp_top["FPR_pct"],
                facecolors="none",
                edgecolors="#111111",
                linewidths=1.3,
                s=165,
                marker="o",
            )

        ax.set_xlim(0.5, plot_df["Alt Rank"].max() + 0.5)
        # Y axis: FPR percent (0..max)
        y_max = max(plot_df["FPR_pct"].max() * 1.15 if not plot_df["FPR_pct"].isnull().all() else 5, 5)
        ax.set_ylim(0, y_max)

        if col == 0:
            ax.set_ylabel("False Positive Rate (%)", fontweight="bold", fontsize=12)
        if row == n_rows_d - 1:
            ax.set_xlabel("Alt Rank (by metric)", fontweight="bold", fontsize=12)

        ax.set_title(f"{metric_info['title']} ranking", fontweight="bold", fontsize=13, pad=2)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(5.5, color="#666666", linestyle=":", linewidth=0.9, alpha=0.6)

    # Colorbar for Alt Rank - Composite Rank (Rank Diff)
    if last_scatter is not None and facet_axes_d:
        cbar = fig3.colorbar(last_scatter, ax=facet_axes_d, fraction=0.05, pad=0.03)
        cbar.set_label("Alt Rank - Composite Rank", fontweight="bold", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    if facet_axes_d:
        panel_d_x0 = min(ax.get_position().x0 for ax in facet_axes_d)
        panel_d_y1 = max(ax.get_position().y1 for ax in facet_axes_d)
        fig3.text(
            panel_d_x0 - 0.01,
            panel_d_y1 + 0.030,
            "D. Individual Alt Rankings vs False Positive Burden",
            fontweight="bold",
            fontsize=15,
        )
        fig3.text(
            panel_d_x0 - 0.01,
            panel_d_y1 + 0.018,
            "Shaded region highlights metric top-5; rings mark composite top-5",
            fontsize=11,
            color="#444444",
        )

    # ── Panel E — All-submissions composite CI forest plot ──────────────────────
    _COLOUR_MAP_E = [
        (0.40,  "#1f77b4"),
        (0.10,  "#2ca02c"),
        (-0.10, "#ff7f0e"),
        (-9999, "#d62728"),
    ]
    def _score_col_e(v):
        for thr, c in _COLOUR_MAP_E:
            if v >= thr:
                return c
        return _COLOUR_MAP_E[-1][1]

    try:
        _ci_all = pd.read_csv(PROCESSED_DIR / "bootstrap_composite_ci.csv")
        _DEADLINE_E = pd.Timestamp("2025-07-19 11:59")
        _ci_all["_st"] = pd.to_datetime(_ci_all["sub_time"], format="%m/%d/%y %H:%M", errors="coerce")
        _ci_all = _ci_all[_ci_all["_st"] <= _DEADLINE_E].copy()

        # All submissions now have real bootstrap CIs
        _ci_all["_plot_pt"] = _ci_all["scaled_orig"]
        _ci_all["_plot_lo"] = _ci_all["scaled_lo_95"]
        _ci_all["_plot_hi"] = _ci_all["scaled_hi_95"]
        _ci_all["_approx"]  = False

        _sub_cnts = _ci_all.groupby("team")["sub_num"].nunique().to_dict()
        def _elabel(team, sub_num):
            return f"{team} (sub{sub_num})" if _sub_cnts.get(team, 1) > 1 else team
        _ci_all["_label"] = [_elabel(r.team, r.sub_num) for _, r in _ci_all.iterrows()]
        _ci_all = _ci_all.sort_values("scaled_orig", ascending=False).reset_index(drop=True)

        _N_e  = len(_ci_all)
        _xe   = np.arange(_N_e)   # x positions: best rank (left) → worst (right)
        bottom_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[2], width_ratios=[1, 2.3], wspace=0.12)
        ax_e = fig3.add_subplot(bottom_spec[0, 1])

        for _i, (_, _row_e) in enumerate(_ci_all.iterrows()):
            _xx_e  = _xe[_i]
            _pt_e  = _row_e["_plot_pt"]
            _lo_e  = _row_e["_plot_lo"]
            _hi_e  = _row_e["_plot_hi"]
            _col_e = _score_col_e(float(_row_e["scaled_orig"]))
            _app_e = bool(_row_e["_approx"])
            # Vertical CI band
            ax_e.bar(_xx_e, _hi_e - _lo_e, bottom=_lo_e, width=0.55,
                     color=_col_e + "44", zorder=2)
            _ls_e = "--" if _app_e else "-"
            ax_e.plot([_xx_e, _xx_e], [_lo_e, _hi_e], color=_col_e, lw=1.4,
                      alpha=0.80, zorder=3, linestyle=_ls_e)
            _mfc_e = "white" if _app_e else _col_e
            _mew_e = 1.8 if _app_e else 0.6
            ax_e.plot(_xx_e, _pt_e, marker="D", markersize=8, color=_col_e,
                      markerfacecolor=_mfc_e, markeredgewidth=_mew_e,
                      markeredgecolor=_col_e, zorder=4)

        ax_e.set_xticks(_xe)
        ax_e.set_xticklabels(_ci_all["_label"], rotation=30, ha="right", fontsize=10)
        ax_e.axhline(0, color="#555", linestyle="--", lw=1.0, alpha=0.7, zorder=1)

        # ── Paired bootstrap difference p-values vs top submission ─────────────
        try:
            _pairwise = pd.read_csv(PROCESSED_DIR / "bootstrap_composite_pairwise_vs_top.csv")
            _pairwise_p = dict(zip(_pairwise["submission_key"], _pairwise["p_holm"]))
        except Exception:
            _pairwise_p = {}

        def _sig_star(p):
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return "ns"

        _sig_y = _ci_all["_plot_hi"].max() + 0.12
        for _si, (_idx, _row_s) in enumerate(_ci_all.iterrows()):
            if _si == 0:  # top team — no comparison against itself
                continue
            _submission_key = _row_s.get("submission_key")
            _p = _pairwise_p.get(_submission_key)
            if _p is None or pd.isna(_p):
                continue
            _star = _sig_star(_p)
            ax_e.text(
                _xe[_si], _sig_y, _star,
                ha="center", va="bottom", fontsize=12,
                color="#333" if _star != "ns" else "#aaa",
                fontweight="bold" if _star != "ns" else "normal",
            )
        # Label the annotation row
        ax_e.text(-0.25, _sig_y + 0.10, "Comparisons vs #1 (Holm)", ha="left", va="bottom",
                  fontsize=10, color="#555", style="italic")
        ax_e.set_ylim(top=_sig_y + 0.25)
        # ────────────────────────────────────────────────────────────────────
        ax_e.set_ylabel("Composite Score (z-normalised)", fontweight="bold", fontsize=16)
        ax_e.set_title(
            "F. Composite Score with Bootstrap 95% CI",
            fontweight="bold", fontsize=17, pad=8,
        )
        ax_e.grid(True, axis="y", alpha=0.3, zorder=0)
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)
        ax_e.tick_params(axis="x", length=0)
        ax_e.tick_params(axis="y", labelsize=11)
        ax_e.set_xlim(-0.6, _N_e - 0.4)

        _e_leg = [
            Line2D([0],[0], marker="D", color="w", markerfacecolor="#1f77b4",
                   markersize=8, label="Score > 0.40"),
            Line2D([0],[0], marker="D", color="w", markerfacecolor="#2ca02c",
                   markersize=8, label="0.10 – 0.40"),
            Line2D([0],[0], marker="D", color="w", markerfacecolor="#ff7f0e",
                   markersize=8, label="−0.10 – 0.10"),
            Line2D([0],[0], marker="D", color="w", markerfacecolor="#d62728",
                   markersize=8, label="Score < −0.10"),
        ]
        ax_e.legend(
            handles=_e_leg,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.02),
            fontsize=13,
            framealpha=0.9,
            edgecolor="#ccc",
            title="Composite score tier",
            title_fontsize=13,
        )
    except Exception as _e_exc:
        raise RuntimeError("Panel E failed") from _e_exc

    # Panel E: decision curves, on shared risk thresholds.
    from matplotlib.ticker import PercentFormatter
    dca = pd.read_csv(PROCESSED_DIR / "decision_curve_top3.csv")
    ax_f = fig3.add_subplot(bottom_spec[0, 0])
    for name, color in [("Badger v3", "#2874A6"), ("HELP v3", "#2E8B57"), ("Epiphany v2", "#E67E22")]:
        part = dca.loc[dca.model == name].sort_values("threshold")
        assert len(part) == 305 and part.N.eq(1151).all()
        np.testing.assert_allclose(part.NB, part.TP / part.N - part.FP / part.N * part.threshold / (1 - part.threshold), atol=1e-14)
        ax_f.plot(part.threshold, part.NB, color=color, lw=2.2, label=name)
    all_policy = dca.loc[dca.model == "Treat all"].sort_values("threshold")
    ax_f.plot(all_policy.threshold, all_policy.NB, color="#777777", ls="--", lw=1.7, label="Treat all")
    ax_f.axhline(0, color="#222222", lw=1.5, label="Treat none")
    ax_f.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_f.set_xlim(dca.threshold.min(), dca.threshold.max())
    ax_f.set_xlabel("Mortality-risk threshold", fontsize=15, fontweight="bold")
    ax_f.set_ylabel("Net benefit per patient", fontsize=15, fontweight="bold", labelpad=6)
    ax_f.set_title("E. Decision Curve Analysis", fontsize=17, fontweight="bold", pad=8)
    ax_f.legend(loc="lower left", ncol=2, fontsize=11, framealpha=.95)
    ax_f.tick_params(labelsize=12)
    ax_f.grid(axis="y", alpha=.25)
    ax_f.spines[["top", "right"]].set_visible(False)

    # Reduce the gap below A by moving C upward without changing its size.
    _c_box = ax3c.get_position()
    ax3c.set_position([_c_box.x0, _c_box.y0 + 0.025, _c_box.width, _c_box.height])

    # Enlarge all labels, ticks, annotations, and legend text by 32%.
    fig3.canvas.draw()
    _font_changes = []
    for _text in fig3.findobj(matplotlib.text.Text):
        if _text.get_text():
            _old_font = _text.get_fontsize()
            _text.set_fontsize(_old_font * 1.32)
            _font_changes.append(dict(text=_text.get_text(), old=_old_font, new=_text.get_fontsize()))
    # Keep each heatmap number within its own cell at the enlarged text size.
    fig3.canvas.draw()
    _renderer = fig3.canvas.get_renderer()
    _cell_width = abs(ax3a.transData.transform((1, 0))[0] - ax3a.transData.transform((0, 0))[0])
    _annotations = [t for t in ax3a.texts if t.get_text()]
    _fit = min(1., min(.95 * _cell_width / t.get_window_extent(_renderer).width for t in _annotations))
    for _text in _annotations:
        _text.set_fontsize(_text.get_fontsize() * _fit)
    _font_changes = [dict(text=t.get_text(), final=t.get_fontsize()) for t in fig3.findobj(matplotlib.text.Text) if t.get_text()]

    # Save Figure 3
    output_path = FIGURES_DIR / "results_figure3_final_phase.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"   ✓ Saved: {output_path.name}")
    return output_path
def main() -> None:
    generate_figure3()
    plt.close("all")


if __name__ == "__main__":
    main()
