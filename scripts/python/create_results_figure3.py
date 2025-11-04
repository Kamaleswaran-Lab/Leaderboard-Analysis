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
    """Create Figure 3 and return the output path."""
    print("\nCreating Figure 3: Final-phase performance overview...")

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
            extended_scores[ ["Team_clean"] + merge_cols ],
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
        final_scores["MCC"] = compute_mcc(final_scores["TP"], final_scores["FP"], final_scores["FN"], final_scores["TN"])

    fig3 = plt.figure(figsize=(20, 12))
    gs3 = GridSpec(
        2,
        1,
        figure=fig3,
        height_ratios=[1.0, 1.35],
        hspace=0.32,
    )

    top_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[0], width_ratios=[55, 45], wspace=0.25)
    bottom_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[1], width_ratios=[1.85, 0.75], wspace=0.25)

    # Panel B — Rank correlations heatmap (lower triangle)
    heatmap_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=top_spec[0, 1], width_ratios=[24, 1], wspace=0.1)
    ax3a = fig3.add_subplot(heatmap_spec[0, 0])
    cbar_ax3a = fig3.add_subplot(heatmap_spec[0, 1])
    # Derive additional performance metrics for correlation analysis
    if {"TP", "FN"}.issubset(final_scores.columns):
        tp_fp = final_scores["TP"] + final_scores["FP"]
        final_scores["Precision"] = np.where(tp_fp > 0, final_scores["TP"] / tp_fp, np.nan)

    selected_metrics = ["Sensitivity", "AUPRC", "Net Benefit", "F1", "ECE"]
    other_metrics = ["AUC", "Specificity", "Precision", "MCC", "Scaled Weighted Score"]
    confusion_metrics = ["TP", "FP", "TN", "FN", "Rank"]
    heatmap_metrics = selected_metrics + other_metrics + confusion_metrics
    available_metrics = [m for m in heatmap_metrics if m in final_scores.columns]

    if len(available_metrics) >= 3:
        corr_matrix = final_scores[available_metrics].corr()
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
        ax3a.set_yticklabels(display_labels, fontsize=11)
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
        if nb_mask.any():
            nb_values = dist_df.loc[nb_mask, "value"].astype(float)
            if len(nb_values) > 2:
                q1, q3 = nb_values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                dist_df = dist_df[~(nb_mask & ((dist_df["value"] < lower) | (dist_df["value"] > upper)))]

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
                wspace=0.25,
                hspace=0.55,
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
                    color = phase_palette.get(phase, metric_color)
                    if len(values.unique()) < 2:
                        if not values.empty:
                            ax.axvline(values.iloc[0], color=color, linestyle="--", linewidth=1.0)
                        continue

                    sns.kdeplot(
                        data=phase_subset,
                        x="value",
                        fill=True,
                        common_norm=False,
                        linewidth=1.1,
                        alpha=0.45,
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

                fig3.text(
                    (x0 + x1) / 2,
                    y0 - 0.055,
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
    ax3c = fig3.add_subplot(bottom_spec[0, 0])
    contribution_metrics = ["F1", "AUPRC", "Net.Benefit", "ECE"]
    available_contrib = [m for m in contribution_metrics if m in final_scores.columns]
    if available_contrib:
        top_ranked = final_scores.sort_values("Rank").head(10).copy()
        contrib_df = build_contribution_table(top_ranked)
        contrib_df = contrib_df.reindex(top_ranked["Team_clean"].values).fillna(0.0)

        x_pos = np.arange(len(contrib_df))
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

            if np.any(positive):
                ax3c.bar(
                    x_pos,
                    positive,
                    bottom=pos_bottom,
                    color=positive_color,
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.6,
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
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": neg_color, "boxstyle": "round,pad=0.22"},
            )

        # Annotate with final scaled scores just beyond the dominant contribution side
        if "Scaled Weighted Score" in top_ranked.columns:
            totals = contrib_df.sum(axis=1).values
            annotation_base = np.where(totals >= 0, pos_bottom, neg_bottom)
            offsets = np.where(totals >= 0, 0.02, -0.02)
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
    ax3c.set_title("C. Weighted Metric Contributions for Top Finalist Submissions", fontweight="bold", fontsize=13)
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
        ax3c.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.08),
            frameon=False,
            title="Metric",
            fontsize=10,
            title_fontsize=11,
        )

    # Panel D — Rank difference between composite and MCC ordering
    ax3d = fig3.add_subplot(bottom_spec[0, 1])
    if {"Rank", "MCC"}.issubset(final_scores.columns):
        rank_compare = final_scores.dropna(subset=["Rank", "MCC"]).copy()
        rank_compare["MCC Rank"] = rank_compare["MCC"].rank(ascending=False, method="min")
        rank_compare["Rank Diff"] = rank_compare["MCC Rank"] - rank_compare["Rank"]

        scatter = ax3d.scatter(
            rank_compare["Rank"],
            rank_compare["Rank Diff"],
            c=rank_compare["Rank Diff"].abs(),
            cmap="YlOrRd",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            s=80,
        )
        ax3d.axhline(0, color="#1f77b4", linestyle="--", linewidth=1.2)
        ax3d.set_xlabel("Competition Rank", fontweight="bold", fontsize=12)
        ax3d.set_ylabel("Rank Difference (MCC - Final)", fontweight="bold", fontsize=12)
        ax3d.set_title("D. Rank Stability: Composite vs MCC", fontweight="bold", fontsize=13)
        ax3d.set_xlim(0.5, rank_compare["Rank"].max() + 0.5)
        diff_limit = max(abs(rank_compare["Rank Diff"]).max(), 1)
        ax3d.set_ylim(-diff_limit - 0.5, diff_limit + 0.5)
        max_rank = int(rank_compare["Rank"].max())
        ax3d.set_xticks(np.arange(5, max_rank + 5, 5))
        y_tick_min = int(-diff_limit)
        y_tick_max = int(diff_limit)
        ax3d.set_yticks(np.arange((y_tick_min // 5) * 5, y_tick_max + 5, 5))
        ax3d.grid(True, alpha=0.3)
        cbar = fig3.colorbar(scatter, ax=ax3d, fraction=0.046, pad=0.04)
        ax3d.tick_params(axis="both", labelsize=10)
        cbar.set_label("|Rank Difference|", fontweight="bold", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    output_path = FIGURES_DIR / "results_figure3_final_phase.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path


def main() -> None:
    generate_figure3()


if __name__ == "__main__":
    main()
