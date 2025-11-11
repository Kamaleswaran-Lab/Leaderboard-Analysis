#!/usr/bin/env python3
"""Generate Figure 3: Final-phase performance overview."""

from __future__ import annotations

from pathlib import Path
import json
import re

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
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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


def load_extended_final_scores() -> pd.DataFrame:
    """Parse the extended final-phase CSV that includes parsimony and inference metrics."""
    eva_path = RAW_DIR / "Eva_csv - Sheet1.csv"
    if not eva_path.exists():
        return pd.DataFrame(columns=["Team_clean"])

    raw_df = pd.read_csv(eva_path)
    parsed_records: list[dict[str, object]] = []
    key_map = {
        "weighted_score": "Weighted Score",
        "scaled_weighted_score": "Scaled Weighted Score",
        "tp": "TP",
        "fp": "FP",
        "fn": "FN",
        "tn": "TN",
        "threshold_used": "Optimal Threshold",
    }

    for _, row in raw_df.iterrows():
        payload = row.get("Evaluated_1")
        if not isinstance(payload, str) or "score" not in payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        score = parsed.get("score", {})
        if not isinstance(score, dict) or not score:
            continue

        record: dict[str, object] = {
            "Team name": row.get("Team name"),
            "Team_clean": extract_team_name(row.get("Team name", "")),
        }
        for key, value in score.items():
            mapped = key_map.get(key, key)
            record[mapped] = value
        parsed_records.append(record)

    if not parsed_records:
        return pd.DataFrame(columns=["Team_clean"])

    parsed_df = pd.DataFrame(parsed_records)
    numeric_cols = [col for col in parsed_df.columns if col not in {"Team name", "Team_clean"}]
    parsed_df[numeric_cols] = parsed_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if "Scaled Weighted Score" in parsed_df.columns:
        parsed_df = parsed_df.sort_values("Scaled Weighted Score", ascending=False, na_position="last")
    parsed_df = parsed_df.dropna(subset=["Team_clean"])
    parsed_df = parsed_df.drop_duplicates(subset="Team_clean", keep="first")
    return parsed_df


def build_contribution_table(top_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted contributions for final-phase top performers."""
    scale_df = pd.read_csv(PROCESSED_DIR / "scale_params.csv")
    scale_lookup = {row.feature: (row.center, row.scale) for _, row in scale_df.iterrows()}

    weights_df = pd.read_csv(PROCESSED_DIR / "factor_weights.csv")
    weight_map = dict(zip(weights_df.feature, weights_df.weight))
    factor_share = 1.0 - weight_map.get("Parsimony Score", 0.0) - weight_map.get("Inference Time", 0.0)

    metric_columns = {
        "F1": "F1",
        "AUPRC": "AUPRC",
        "Net.Benefit": "Net.Benefit",
        "ECE": "ECE",
    }

    contribution_parts: dict[str, pd.Series] = {}
    for feature, column in metric_columns.items():
        if column not in top_scores.columns or feature not in weight_map:
            continue
        center, scale = scale_lookup.get(feature, (top_scores[column].mean(), top_scores[column].std()))
        if pd.isna(scale) or scale == 0:
            scale = 1.0
        z_values = (top_scores[column] - center) / scale
        contribution_parts[feature] = z_values * weight_map[feature] * factor_share

    # Add parsimony and inference components using original weights
    if "parsimony_score" in top_scores.columns and "Parsimony Score" in weight_map:
        contribution_parts["Parsimony (1 - score)"] = (1 - top_scores["parsimony_score"].clip(0, 1)) * weight_map["Parsimony Score"]
    if "inference_time" in top_scores.columns and "Inference Time" in weight_map:
        center, scale = scale_lookup.get("Inference Time", (top_scores["inference_time"].mean(), top_scores["inference_time"].std()))
        if pd.isna(scale) or scale == 0:
            scale = 1.0
        inf_norm = (top_scores["inference_time"] - center) / scale
        contribution_parts["Inference Speed"] = (-inf_norm) * weight_map["Inference Time"]

    contrib_df = pd.DataFrame(contribution_parts, index=top_scores.index)
    contrib_df.index = top_scores["Team_clean"].values
    return contrib_df


def generate_figure3() -> Path:
    """Create Figure 3 and return the output path.

    Load final_scores once, merge any extended fields, compute MCC and FPR where
    possible, then build the figure panels (A-D). This removes duplicate loads
    and guarantees Panel D only shows individual alternative-metric rankings.
    """
    # Load and process final_scores (single canonical load)
    final_scores = pd.read_csv(RAW_DIR / "final_scores.csv")
    final_scores["Team_clean"] = final_scores["Team name"].apply(extract_team_name)

    extended_scores = load_extended_final_scores()
    if not extended_scores.empty:
        merge_cols = [
            col
            for col in ["Parsimony Score", "Inference Time", "Sensitivity", "Specificity", "TP", "FP", "FN", "TN"]
            if col in extended_scores.columns
        ]
        final_scores = final_scores.merge(
            extended_scores[["Team_clean"] + merge_cols],
            on="Team_clean",
            how="left",
            suffixes=("", "_ext"),
        )
        for col in merge_cols:
            ext_col = f"{col}_ext"
            if ext_col in final_scores.columns:
                if col in final_scores.columns:
                    final_scores[col] = final_scores[col].fillna(final_scores[ext_col])
                else:
                    final_scores[col] = final_scores[ext_col]
                final_scores.drop(columns=[ext_col], inplace=True)

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

    # Increase the overall vertical height to add more vertical space between rows
    fig3 = plt.figure(figsize=(20, 20))
    # Increase vertical spacing (hspace) and slightly adjust height ratios so row titles
    # and x-axis labels do not overlap. Middle row gets a bit more height for Panel C/D.
    gs3 = GridSpec(
        3,
        1,
        figure=fig3,
        # give the top row (density plots) a bit more room and slightly reduce the
        # middle row so Panels C/D become a touch smaller vertically
        height_ratios=[1.25, 1.35, 1.0],
        # keep existing overall vertical gap
        hspace=0.62,
    )
    # Use partitions for left/right columns and make Panel C slightly wider than D
    # to visually reduce the Panel D footprint.
    top_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[0], width_ratios=[1, 1], wspace=0.20)
    middle_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[1], width_ratios=[1.10, 0.90], wspace=0.20)
    # Removed Panel E: No bottom_spec

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
                    fontsize=10,
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
        ax3a.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=11)
        # Set y-axis labels horizontal (0 degrees) as requested
        ax3a.set_yticklabels(display_labels, rotation=0, ha="right", fontsize=11)
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
        cbar_ax3a.set_ylabel("Correlation", fontweight="bold", fontsize=12)
        cbar_ax3a.yaxis.set_label_position("right")
        cbar_ax3a.tick_params(length=0, labelsize=10)
        ax3a.set_title("B. Rank Correlations with Evaluation Metrics", fontweight="bold", fontsize=13)

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
                hspace=0.85,
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

                ax.set_title(display, fontweight="bold", fontsize=11)
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
                    fontsize=10,
                    title_fontsize=11,
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
                    fontsize=12,
                )
                fig3.text(
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
                fig3.text(
                    panel_anchor.x0,
                    panel_anchor.y1 + 0.035,
                    "A. Metric Distributions by Phase",
                    fontsize=13,
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

    # Panel C — Top performers with metric contributions
    ax3c = fig3.add_subplot(middle_spec[0, 0])
    contribution_metrics = ["F1", "AUPRC", "Net.Benefit", "ECE"]
    available_contrib = [m for m in contribution_metrics if m in final_scores.columns]
    if available_contrib:
        top_ranked = final_scores.sort_values("Rank").head(10).copy()
        contrib_df = build_contribution_table(top_ranked)
        contrib_df = contrib_df.reindex(top_ranked["Team_clean"].values).fillna(0.0)

    # Increase spacing between bars for better text separation
    x_gap = 1.6  # Further increase gap between bars
    x_pos = np.arange(len(contrib_df)) * x_gap
    pos_bottom = np.zeros(len(contrib_df))
    neg_bottom = np.zeros(len(contrib_df))
    metric_palette = {metric: get_metric_color(metric) for metric in contrib_df.columns}
    legend_entries: list[tuple[str, str]] = []
    negative_annotations: list[tuple[float, float, str, float, str]] = []

    for metric in contrib_df.columns:
        values = contrib_df[metric].values
        display_name = metric.replace("Net.Benefit", "Net Benefit")
        base_color = metric_palette.get(metric, "#666666")
        positive_color = lighten_color(base_color, 0.5)
        negative_color = darken_color(base_color, 0.2)

        if display_name not in {label for label, _ in legend_entries}:
            legend_entries.append((display_name, base_color))

        positive = np.where(values > 0, values, 0.0)
        negative = np.where(values < 0, values, 0.0)

        bar_width = 0.8
        if np.any(positive):
            ax3c.bar(
                x_pos,
                positive,
                bottom=pos_bottom,
                color=positive_color,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
                width=bar_width,
            )
            pos_bottom += positive

        if np.any(negative):
            neg_base = neg_bottom.copy()
            ax3c.bar(
                x_pos,
                negative,
                bottom=neg_base,
                color=negative_color,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
                width=bar_width,
            )
            for idx, val in enumerate(negative):
                if val < 0:
                    negative_annotations.append(
                        (
                            x_pos[idx],
                            neg_base[idx] + val / 2.0,
                            display_name,
                            val,
                            negative_color,
                            base_color,
                        )
                    )
            neg_bottom += negative

    ax3c.axhline(0, color="#555", linewidth=1.0, linestyle="--", alpha=0.6)

    # Increase vertical spacing for text annotations
    for x_coord, y_coord, metric_name, val, neg_color, base_color in negative_annotations:
        ax3c.text(
            x_coord,
            y_coord,
            f"{metric_name}\n{val:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=base_color,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": neg_color, "boxstyle": "round,pad=0.45"},
        )

    # Annotate with final scaled scores just beyond the dominant contribution side
    if "Scaled Weighted Score" in top_ranked.columns:
        totals = contrib_df.sum(axis=1).values
        annotation_base = np.where(totals >= 0, pos_bottom, neg_bottom)
        offsets = np.where(totals >= 0, 0.04, -0.04)
        vertical_align = np.where(totals >= 0, "bottom", "top")

        for idx, x_coord in enumerate(x_pos):
            score = top_ranked.iloc[idx]["Scaled Weighted Score"]
            ax3c.text(
                x_coord,
                annotation_base[idx] + offsets[idx],
                f"{score:.2f}",
                ha="center",
                va=vertical_align[idx],
                fontsize=10,
                fontweight="bold",
            )

    labels = [f"Rank {int(rank)}\n{team}" for rank, team in zip(top_ranked["Rank"], top_ranked["Team_clean"])]
    ax3c.set_xticks(x_pos)
    ax3c.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax3c.set_ylabel("Weighted Contribution", fontweight="bold", fontsize=12)
    # Move the panel title a bit farther from the plot for clarity
    ax3c.set_title(
        "C. Weighted Metric Contributions for Top Finalist Submissions",
        fontweight="bold",
        fontsize=13,
        pad=25,
    )
    ax3c.grid(True, alpha=0.3, axis="y")
    ax3c.spines["top"].set_visible(False)
    ax3c.spines["right"].set_visible(False)
    ax3c.tick_params(axis="both", labelsize=10)
    if available_contrib:
        legend_handles = [
            Patch(
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                label=label,
            )
            for label, color in legend_entries
        ]
        # Place legend just outside the upper-right of the panel to avoid overlap
        # Move legend into the top-right corner of the panel
        ax3c.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.08),
            frameon=False,
            title="Metric",
            fontsize=9,
            title_fontsize=11,
            handletextpad=0.6,
            borderaxespad=0.2,
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
    facet_d_spec = GridSpecFromSubplotSpec(n_rows_d, 2, subplot_spec=middle_spec[0, 1], wspace=0.32, hspace=0.45)

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
            ax.set_ylabel("False Positive Rate (%)", fontweight="bold", fontsize=10)
        if row == n_rows_d - 1:
            ax.set_xlabel("Alt Rank (by metric)", fontweight="bold", fontsize=10)

        ax.set_title(f"{metric_info['title']} ranking", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(5.5, color="#666666", linestyle=":", linewidth=0.9, alpha=0.6)

    # Colorbar for Alt Rank - Composite Rank (Rank Diff)
    if last_scatter is not None and facet_axes_d:
        cbar = fig3.colorbar(last_scatter, ax=facet_axes_d, fraction=0.05, pad=0.03)
        cbar.set_label("Alt Rank - Composite Rank", fontweight="bold", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    if facet_axes_d:
        top_left_ax = facet_axes_d[0]
        panel_d_bbox = top_left_ax.get_position()
        fig3.text(
            panel_d_bbox.x0 - 0.01,
            panel_d_bbox.y1 + 0.04,
            "D. Individual Alt Rankings vs False Positive Burden",
            fontweight="bold",
            fontsize=13,
        )
        fig3.text(
            panel_d_bbox.x0 - 0.01,
            panel_d_bbox.y1 + 0.02,
            "Shaded region highlights metric top-5; rings mark composite top-5",
            fontsize=9,
            color="#444444",
        )
    # Save Figure 3
    output_path = FIGURES_DIR / "results_figure3_final_phase.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path
def generate_figure4_composite_alt_rankings(final_scores: pd.DataFrame) -> Path:
    # Ensure MCC is computed as in Figure 3
    if "MCC" not in final_scores.columns and {"TP", "FP", "FN", "TN"}.issubset(final_scores.columns):
        final_scores["MCC"] = compute_mcc(final_scores["TP"], final_scores["FP"], final_scores["FN"], final_scores["TN"])
    """Create Figure 4: Composite Alt Ranking metrics for explanation."""
    print("\nCreating Figure 4: Composite Alt Ranking metrics...")
    # Only show composite combinations and their trade-offs with FPR, TPR (Sensitivity), ECE, and Net Benefit.
    composite_metrics = [
        {"name": "AUC+ECE", "title": "AUC + ECE"},
        {"name": "AUPRC+ECE", "title": "AUPRC + ECE"},
        {"name": "AUC+NetBenefit", "title": "AUC + Net Benefit"},
        {"name": "AUPRC+NetBenefit", "title": "AUPRC + Net Benefit"},
        {"name": "AUC+ECE+NetBenefit", "title": "AUC + ECE + Net Benefit"},
    ]

    # Ensure FPR and Sensitivity (TPR) are available
    if {"FP", "TN"}.issubset(final_scores.columns):
        fp_tn_sum = final_scores["FP"] + final_scores["TN"]
        final_scores.loc[fp_tn_sum > 0, "FPR"] = final_scores.loc[fp_tn_sum > 0, "FP"] / fp_tn_sum[fp_tn_sum > 0]
    if "FPR" not in final_scores.columns or final_scores["FPR"].isna().all():
        if "Specificity" in final_scores.columns:
            final_scores["FPR"] = 1 - final_scores["Specificity"]
    final_scores["FPR"] = pd.to_numeric(final_scores.get("FPR", pd.Series(dtype=float)), errors="coerce").clip(lower=0)
    final_scores["FPR_pct"] = final_scores["FPR"] * 100

    if "Sensitivity" not in final_scores.columns and {"TP", "FN"}.issubset(final_scores.columns):
        denom = final_scores["TP"] + final_scores["FN"]
        final_scores.loc[denom > 0, "Sensitivity"] = final_scores.loc[denom > 0, "TP"] / denom[denom > 0]

    # Metrics to display as Y columns in the trade-off panels
    y_metrics = [
        ("FPR_pct", "False Positive Rate (%)"),
        ("Sensitivity", "True Positive Rate (Sensitivity)"),
        ("ECE", "Calibration (ECE)"),
        ("Net.Benefit", "Net Benefit"),
    ]

    n_rows = len(composite_metrics)
    n_cols = len(y_metrics)
    fig4 = plt.figure(figsize=(4.2 * n_cols, 2.6 * n_rows))
    gs4 = GridSpec(n_rows, n_cols, figure=fig4, wspace=0.28, hspace=0.38)

    scatter_refs = []
    for r_idx, metric_info in enumerate(composite_metrics):
        metric_name = metric_info["name"]
        # parse composite parts (map NetBenefit token to actual column name if present)
        metric_parts = metric_name.replace("+", " ").replace("NetBenefit", "Net.Benefit").split()
        metric_cols = [m for m in metric_parts if m in final_scores.columns]
        if not metric_cols:
            # For the whole row, place unavailable markers
            for c_idx in range(n_cols):
                ax = fig4.add_subplot(gs4[r_idx, c_idx])
                ax.axis("off")
                if c_idx == 0:
                    ax.text(0.5, 0.5, f"{metric_info['title']} unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            continue

        # Build rank_compare using the composite sorting logic (ECE ascending)
        rank_compare = final_scores.dropna(subset=["Rank"] + metric_cols).copy()
        sort_cols = []
        sort_orders = []
        for col in metric_cols:
            sort_cols.append(col)
            sort_orders.append(True if col == "ECE" else False)
        rank_compare = rank_compare.sort_values(sort_cols, ascending=sort_orders)
        rank_compare["Alt Rank"] = range(1, len(rank_compare) + 1)
        # For plotting convenience compute Rank Diff and FPR_pct
        rank_compare["Rank Diff"] = rank_compare["Alt Rank"] - rank_compare["Rank"]
        rank_compare["FPR_pct"] = rank_compare.get("FPR", np.nan) * 100

        for c_idx, (y_col, y_label) in enumerate(y_metrics):
            ax = fig4.add_subplot(gs4[r_idx, c_idx])
            if y_col not in rank_compare.columns:
                ax.axis("off")
                if c_idx == 0:
                    ax.text(0.5, 0.5, f"{metric_info['title']} (no data)", transform=ax.transAxes, ha="center", va="center", fontsize=10)
                continue

            plot_df = rank_compare.dropna(subset=[y_col, "Alt Rank"]).copy()
            if plot_df.empty:
                ax.axis("off")
                continue

            sc = ax.scatter(
                plot_df["Alt Rank"],
                plot_df[y_col],
                c=plot_df["Rank Diff"],
                cmap="coolwarm",
                edgecolor="#1a1a1a",
                linewidth=0.45,
                alpha=0.9,
                s=48,
            )
            scatter_refs.append(sc)

            # highlight composite top-5
            comp_top = plot_df[plot_df["Rank"] <= 5]
            if not comp_top.empty:
                ax.scatter(
                    comp_top["Alt Rank"],
                    comp_top[y_col],
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=1.2,
                    s=120,
                    marker="o",
                )

            ax.set_xlim(0.5, plot_df["Alt Rank"].max() + 0.5)
            # reasonable y-limits per metric
            if y_col == "FPR_pct":
                y_max = max(plot_df[y_col].max() * 1.15 if not plot_df[y_col].isnull().all() else 5, 5)
                ax.set_ylim(0, y_max)
            elif y_col == "Sensitivity":
                ax.set_ylim(0, 1.02)
            elif y_col == "ECE":
                y_max = plot_df[y_col].max() * 1.15 if not plot_df[y_col].isnull().all() else 0.1
                ax.set_ylim(0, max(y_max, 0.01))
            ax.set_title(y_label if r_idx == 0 else "", fontsize=10 if r_idx == 0 else 9)
            if c_idx == 0:
                ax.set_ylabel(metric_info["title"], fontweight="bold", fontsize=10)
            if r_idx == n_rows - 1:
                ax.set_xlabel("Alt Rank (by composite)", fontweight="bold", fontsize=10)
            ax.grid(True, alpha=0.25)

    # Shared colorbar for Rank Diff
    if scatter_refs:
        cbar = fig4.colorbar(scatter_refs[-1], ax=fig4.axes, fraction=0.04, pad=0.02)
        cbar.set_label("Alt Rank - Composite Rank", fontweight="bold", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    output_path = Path("figures/main_figures/results_figure4_composite_alt_rankings.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path

    # Panel E removed
    output_path = FIGURES_DIR / "results_figure3_final_phase.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path


def main() -> None:
    output_path = generate_figure3()
    # Also generate Figure 4 for composite alt rankings
    final_scores = pd.read_csv(ROOT / "data" / "raw" / "final_scores.csv")
    generate_figure4_composite_alt_rankings(final_scores)


if __name__ == "__main__":
    main()
