#!/usr/bin/env python3
"""Generate Figure 2: Factor analysis methodology panels."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
PRELIM_DIR = ROOT / "optimal_threshold_results"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "figures" / "main_figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


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

    fig2 = plt.figure(figsize=(18, 13))
    gs2 = GridSpec(3, 4, figure=fig2, hspace=0.35, wspace=0.55, width_ratios=[0.9, 1.2, 1.2, 1.1])
    fig2.subplots_adjust(hspace=1.2, bottom=0.06, top=0.95, right=0.9, left=0.07, wspace=0.6)

    # Panel A
    ax2a = fig2.add_subplot(gs2[0, :2])
    core_metrics = ["F1", "AUPRC", "Net.Benefit", "ECE"]
    panel_a_weights = factor_weights[factor_weights["feature"].isin(core_metrics)].copy()
    colors_weight = ["#2ECC71" if w > 0 else "#E74C3C" for w in panel_a_weights["weight"]]
    bars = ax2a.barh(
        panel_a_weights["feature"],
        panel_a_weights["weight"],
        color=colors_weight,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2a.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax2a.set_xlabel("Weight Value", fontweight="bold")
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
                fontsize=10,
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
                fontsize=10,
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
    colors_pie = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    
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
        textprops={"fontsize": 9, "fontweight": "bold"},
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
            fontsize=9,
            fontweight="bold",
        )
    
    ax2b.set_title("B. Relative Weight Contribution", fontweight="bold", fontsize=13, pad=20)

    # Panel C - All metrics distributions from dummy phase (row 2 left)
    ax2c = fig2.add_subplot(gs2[1, :2])
    base_metrics = ["F1", "AUPRC", "AUC", "ECE", "Sensitivity", "Specificity"]
    metric_labels_dummy: list[str] = []
    metric_data_dummy: list[pd.Series] = []
    label_map = {
        "F1": "F1",
        "AUPRC": "AUPRC",
        "AUC": "AUC",
        "ECE": "ECE",
        "Sensitivity": "Sensitivity",
        "Specificity": "Specificity",
    }

    for metric in base_metrics:
        if metric in dummy_phase.columns:
            cleaned = remove_outliers(dummy_phase[metric])
            if not cleaned.empty:
                metric_data_dummy.append(cleaned)
                metric_labels_dummy.append(label_map.get(metric, metric))

    if "Net.Benefit" in dummy_phase.columns:
        nb_clean = remove_outliers(dummy_phase["Net.Benefit"])
        if not nb_clean.empty:
            metric_data_dummy.append(nb_clean)
            metric_labels_dummy.append("Net Benefit")

    if metric_data_dummy:
        metric_arrays = [series.values for series in metric_data_dummy]
        parts = ax2c.violinplot(
            metric_arrays,
            positions=range(len(metric_labels_dummy)),
            showmeans=True,
            showmedians=True,
        )
        palette_c = sns.color_palette("husl", len(metric_labels_dummy))
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette_c[i])
            pc.set_alpha(0.75)
        ax2c.set_xticks(range(len(metric_labels_dummy)))
        ax2c.set_xticklabels(metric_labels_dummy, rotation=25, ha="right")
        ax2c.set_ylabel("Metric Value", fontweight="bold")
        ax2c.set_title(
            "C. Distribution of Ranking Metrics\n(Dummy Phase, outliers removed)",
            fontweight="bold",
            fontsize=13,
        )
        ax2c.grid(True, alpha=0.3, axis="y")
        ax2c.spines["top"].set_visible(False)
        ax2c.spines["right"].set_visible(False)

    # Panel D - Correlation heatmap using remaining row 2 space
    ax2d = fig2.add_subplot(gs2[1, 2:])
    dummy_phase_ranked = dummy_phase.copy()
    dummy_phase_ranked["Rank"] = dummy_phase_ranked["final_score"].rank(ascending=False)
    if "TP" in dummy_phase_ranked.columns and "FP" in dummy_phase_ranked.columns:
        denom = dummy_phase_ranked["TP"] + dummy_phase_ranked["FP"]
        dummy_phase_ranked["Precision"] = np.where(denom > 0, dummy_phase_ranked["TP"] / denom, np.nan)
    else:
        dummy_phase_ranked["Precision"] = np.nan

    selected_metrics = ["AUC", "AUPRC", "F1", "Precision", "Sensitivity"]
    other_metrics = ["Specificity", "ECE", "Net.Benefit", "TP", "FP", "TN", "FN", "Rank"]
    metric_order = selected_metrics + other_metrics
    available_metrics = [m for m in metric_order if m in dummy_phase_ranked.columns]

    if len(available_metrics) > 2:
        corr_matrix_dummy = dummy_phase_ranked[available_metrics].corr()

        from matplotlib.patches import Rectangle

        mask = np.triu(np.ones_like(corr_matrix_dummy, dtype=bool), k=1)
        heatmap_palette = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(
            corr_matrix_dummy,
            mask=mask,
            cmap=heatmap_palette,
            vmin=-1,
            vmax=1,
            ax=ax2d,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )

        confusion_metrics = ["TP", "FP", "TN", "FN"]
        confusion_indices = [available_metrics.index(m) for m in confusion_metrics if m in available_metrics]
        rank_idx = available_metrics.index("Rank") if "Rank" in available_metrics else None
        f1_idx = available_metrics.index("F1") if "F1" in available_metrics else None
        weighted_metrics_norm = {normalize_metric_name(feat) for feat in factor_weights["feature"]}

        # Add text annotations only for visible (lower triangular) cells
        for i in range(len(available_metrics)):
            for j in range(len(available_metrics)):
                if j > i:
                    continue
                value = corr_matrix_dummy.iloc[i, j]
                highlight = False
                label_i_norm = normalize_metric_name(available_metrics[i])
                label_j_norm = normalize_metric_name(available_metrics[j])
                if rank_idx is not None:
                    if (i == rank_idx and j in confusion_indices) or (j == rank_idx and i in confusion_indices):
                        highlight = True
                    if f1_idx is not None and ((i == rank_idx and j == f1_idx) or (j == rank_idx and i == f1_idx)):
                        highlight = True
                    if i == rank_idx and label_j_norm in weighted_metrics_norm:
                        highlight = True
                    if j == rank_idx and label_i_norm in weighted_metrics_norm:
                        highlight = True
                if label_i_norm in weighted_metrics_norm and label_j_norm in weighted_metrics_norm:
                    highlight = True
                ax2d.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(value) < 0.6 else "white",
                    fontsize=8,
                    fontweight="bold" if highlight or abs(value) >= 0.5 else "normal",
                )

        # Highlight association between ranking and confusion metrics (lower triangle)
        if rank_idx is not None and confusion_indices:
            for metric_idx in confusion_indices:
                row_idx, col_idx = (metric_idx, rank_idx) if metric_idx >= rank_idx else (rank_idx, metric_idx)
                rect = Rectangle(
                    (col_idx, row_idx),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#1f77b4",
                    linewidth=2,
                )
                ax2d.add_patch(rect)

        ax2d.set_xticks(np.arange(len(available_metrics)) + 0.5)
        ax2d.set_yticks(np.arange(len(available_metrics)) + 0.5)
        ax2d.set_xticklabels(available_metrics, rotation=25, ha="right", fontsize=9)
        ax2d.set_yticklabels(available_metrics, fontsize=9)

        for tick, label in zip(ax2d.get_xticklabels(), available_metrics):
            label_norm = normalize_metric_name(label)
            if label_norm in weighted_metrics_norm or label_norm == "rank":
                tick.set_fontweight("bold")

        for tick, label in zip(ax2d.get_yticklabels(), available_metrics):
            label_norm = normalize_metric_name(label)
            if label_norm in weighted_metrics_norm or label_norm == "rank":
                tick.set_fontweight("bold")
        ax2d.tick_params(length=0)

        cbar_c = fig2.colorbar(ax2d.collections[0], ax=ax2d, fraction=0.046, pad=0.04)
        cbar_c.set_label("Correlation", fontweight="bold", fontsize=10)

        ax2d.set_title(
            "D. Metric Correlations with Ranking (Dummy Phase)",
            fontweight="bold",
            fontsize=13,
        )

    # Panel F - Model complexity vs final score (dummy phase)
    ax2f = fig2.add_subplot(gs2[2, :2])
    resource_cols = ["parsimony_score", "final_score", "inference_time"]
    if all(col in dummy_phase.columns for col in resource_cols):
        final_scores_clean = remove_outliers(dummy_phase["final_score"])
        filtered = dummy_phase.loc[final_scores_clean.index].copy() if not final_scores_clean.empty else dummy_phase.copy()

        parsimony_complement = np.clip(1 - filtered["parsimony_score"], 0, 1)
        inference_ms = filtered["inference_time"] * 1000

        scatter = ax2f.scatter(
            parsimony_complement,
            filtered["final_score"],
            c=inference_ms,
            cmap="viridis",
            alpha=0.8,
            s=60,
            edgecolor="black",
            linewidth=0.4,
        )

        # Annotate top three unique teams by highest filtered final score
        if "Team" in filtered.columns:
            label_series = filtered["Team"].apply(clean_team_label)
        elif "id" in filtered.columns:
            label_series = filtered["id"].apply(clean_team_label)
        else:
            label_series = pd.Series(filtered.index.astype(str), index=filtered.index)
        filtered = filtered.assign(label=label_series)
        annotate_without_overlap(
            ax2f,
            parsimony_complement.values,
            filtered["final_score"].values,
            filtered["label"].tolist(),
        )

        ax2f.set_xlabel("Model Simplicity (1 − Parsimony Score)", fontweight="bold")
        ax2f.set_ylabel("Final Score (Dummy Phase)", fontweight="bold")
        ax2f.set_title("F. Model Complexity vs Final Score (Dummy Phase)", fontweight="bold", fontsize=13)
        ax2f.grid(True, alpha=0.3)
        ax2f.spines["top"].set_visible(False)
        ax2f.spines["right"].set_visible(False)

        cbar_f = fig2.colorbar(scatter, ax=ax2f, fraction=0.046, pad=0.04)
        cbar_f.set_label("Inference Time (ms)", fontweight="bold", fontsize=9)

    # Panel H - Weighted metric contributions for top dummy-phase teams
    ax2h = fig2.add_subplot(gs2[2, 2:])
    if "Team" in dummy_phase.columns:
        base_team_series = dummy_phase["Team"]
    elif "id" in dummy_phase.columns:
        base_team_series = dummy_phase["id"]
    else:
        base_team_series = pd.Series(dummy_phase.index.astype(str), index=dummy_phase.index)

    dummy_phase_labeled = dummy_phase.copy()
    dummy_phase_labeled["team_label"] = base_team_series.apply(clean_team_label)
    dummy_phase_labeled["team_label_base"] = dummy_phase_labeled["team_label"].str.replace(r"\d+$", "", regex=True).str.strip()

    top_dummy_all = dummy_phase_labeled.sort_values(by="final_score", ascending=False)
    top_dummy = top_dummy_all.drop_duplicates(subset="team_label_base", keep="first").head(5)

    if not top_dummy.empty:
        weight_map = {row["feature"]: row["weight"] for _, row in factor_weights.iterrows()}
        # Factor-analysis metrics share a combined 90% weight after allocating 5% each
        # to parsimony and inference time (see Factor_analysis_for_scores_weights.Rmd).
        factor_share = 1.0 - weight_map.get("Parsimony Score", 0.0) - weight_map.get("Inference Time", 0.0)
        metrics_for_contrib = ["F1", "AUPRC", "Net.Benefit", "ECE"]
        normalized_contrib: list[tuple[pd.Series, str]] = []

        for metric in metrics_for_contrib:
            if metric in top_dummy.columns and metric in weight_map:
                center, scale = scale_lookup.get(metric, (top_dummy[metric].mean(), top_dummy[metric].std()))
                if scale == 0 or np.isnan(scale):
                    scale = 1.0
                scaled_values = (top_dummy[metric] - center) / scale
                contrib = scaled_values * weight_map[metric] * factor_share
                normalized_contrib.append((contrib, metric))

        if "parsimony_score" in top_dummy.columns and "Parsimony Score" in weight_map:
            parsimony_component = 1 - top_dummy["parsimony_score"].clip(0, 1)
            contrib = parsimony_component * weight_map["Parsimony Score"]
            normalized_contrib.append((contrib, "Parsimony (1 - score)"))

        if "inference_time" in top_dummy.columns and "Inference Time" in weight_map:
            center, scale = scale_lookup.get("Inference Time", (top_dummy["inference_time"].mean(), top_dummy["inference_time"].std()))
            if scale == 0 or np.isnan(scale):
                scale = 1.0
            inf_norm = (top_dummy["inference_time"] - center) / scale
            contrib = inf_norm * weight_map["Inference Time"]
            normalized_contrib.append((contrib, "Inference Time"))

        if normalized_contrib:
            x_pos = np.arange(len(top_dummy))
            bottom = np.zeros(len(top_dummy))
            colors_comp = sns.color_palette("Set2", len(normalized_contrib))
            for i, (contrib_series, label) in enumerate(normalized_contrib):
                contrib_ordered = contrib_series.reindex(top_dummy.index).values
                ax2h.bar(
                    x_pos,
                    contrib_ordered,
                    bottom=bottom,
                    label=label,
                    color=colors_comp[i],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottom += contrib_ordered

            team_labels = top_dummy["team_label"].tolist()
            ax2h.set_xticks(x_pos)
            ax2h.set_xticklabels([f"Rank {i + 1}\n{team_labels[i]}" for i in range(len(team_labels))], fontsize=9)
            ax2h.set_ylabel("Weighted Contribution", fontweight="bold")
            ax2h.set_title(
                "H. Weighted Metric Contributions\n(Top Dummy-Phase Teams)",
                fontweight="bold",
                fontsize=13,
            )
            ax2h.tick_params(axis="x", pad=6)
            ax2h.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
            ax2h.grid(True, alpha=0.3, axis="y")
            ax2h.spines["top"].set_visible(False)
            ax2h.spines["right"].set_visible(False)

    output_path = FIGURES_DIR / "results_figure2_factor_analysis.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path


def main() -> None:
    generate_figure2()


if __name__ == "__main__":
    main()
