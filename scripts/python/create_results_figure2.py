#!/usr/bin/env python3
"""Generate Figure 2: Factor analysis methodology panels."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRELIM_DIR = ROOT / "optimal_threshold_results"
VALIDATION_DIR = ROOT / "optimal_threshold_results_validation"
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


def generate_figure2() -> Path:
    """Create Figure 2 and return the output path."""
    print("\n📈 Creating Figure 2: Factor Analysis Methodology...")

    final_scores = pd.read_csv(RAW_DIR / "final_scores.csv")
    prelim_sens = pd.read_csv(PRELIM_DIR / "evaluation_metrics_sensitivity_method.csv")
    validation_sens = pd.read_csv(VALIDATION_DIR / "evaluation_metrics_sensitivity_method.csv")

    try:
        import json

        with open(PROCESSED_DIR / "factor_loadings.json", "r") as f:
            factor_loadings_dict = json.load(f)
        factor_loadings = pd.DataFrame(factor_loadings_dict)
    except Exception:
        factor_loadings = None

    factor_weights = pd.read_csv(PROCESSED_DIR / "factor_weights.csv")

    fig2 = plt.figure(figsize=(16, 12))
    gs2 = GridSpec(3, 3, figure=fig2, hspace=0.4, wspace=0.35)

    # Panel A
    ax2a = fig2.add_subplot(gs2[0, :2])
    colors_weight = ["#2ECC71" if w > 0 else "#E74C3C" for w in factor_weights["weight"]]
    bars = ax2a.barh(
        factor_weights["feature"],
        factor_weights["weight"],
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

    for feature, weight in zip(factor_weights["feature"], factor_weights["weight"]):
        ax2a.text(
            weight + 0.03 if weight >= 0 else weight - 0.03,
            feature,
            f"{weight:.3f}",
            va="center",
            ha="left" if weight >= 0 else "right",
            fontsize=10,
            fontweight="bold",
        )

    # Panel B
    ax2b = fig2.add_subplot(gs2[0, 2])
    weights_abs = np.abs(factor_weights["weight"])
    contribution = weights_abs / weights_abs.sum() * 100
    colors_pie = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    ax2b.pie(
        contribution,
        labels=factor_weights["feature"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors_pie[: len(contribution)],
        textprops={"fontsize": 9},
    )
    ax2b.set_title("B. Relative Weight Contribution", fontweight="bold", fontsize=13)

    # Panel C
    ax2c = fig2.add_subplot(gs2[1, 0])
    key_metrics = ["F1", "AUPRC", "ECE"]
    metric_data_prelim = []
    metric_labels_prelim = []
    for metric in key_metrics:
        if metric in prelim_sens.columns:
            metric_data_prelim.append(prelim_sens[metric].dropna())
            metric_labels_prelim.append(metric)

    if metric_data_prelim:
        parts = ax2c.violinplot(
            metric_data_prelim,
            positions=range(len(metric_labels_prelim)),
            showmeans=True,
            showmedians=True,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors_pie[i])
            pc.set_alpha(0.7)
        ax2c.set_xticks(range(len(metric_labels_prelim)))
        ax2c.set_xticklabels(metric_labels_prelim)
        ax2c.set_ylabel("Metric Value", fontweight="bold")
        ax2c.set_title("C. Key Metrics Distribution\n(Preliminary Phase)", fontweight="bold", fontsize=13)
        ax2c.grid(True, alpha=0.3, axis="y")
        ax2c.spines["top"].set_visible(False)
        ax2c.spines["right"].set_visible(False)

    # Panel D
    ax2d = fig2.add_subplot(gs2[1, 1])
    corr_metrics_prelim = ["AUC", "AUPRC", "F1", "ECE", "Net Benefit"]
    available_corr = [m for m in corr_metrics_prelim if m in prelim_sens.columns]
    if len(available_corr) > 2:
        corr_matrix_prelim = prelim_sens[available_corr].corr()
        sns.heatmap(
            corr_matrix_prelim,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax2d,
            cbar_kws={"label": "Correlation"},
            square=True,
            linewidths=1,
            linecolor="white",
        )
        ax2d.set_title("D. Metric Correlations\n(Preliminary Phase)", fontweight="bold", fontsize=13)

    # Panel E
    ax2e = fig2.add_subplot(gs2[1, 2])
    if "Weighted Score" in final_scores.columns:
        weighted_scores = final_scores["Weighted Score"].dropna()
        q1, q99 = weighted_scores.quantile([0.01, 0.99])
        weighted_clean = weighted_scores[(weighted_scores >= q1) & (weighted_scores <= q99)]

        ax2e.hist(
            weighted_clean,
            bins=25,
            color="#3498DB",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )
        ax2e.axvline(
            weighted_clean.median(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Median: {weighted_clean.median():.2f}",
        )
        ax2e.set_xlabel("Weighted Score", fontweight="bold")
        ax2e.set_ylabel("Number of Teams", fontweight="bold")
        ax2e.set_title("E. Final Weighted Score Distribution", fontweight="bold", fontsize=13)
        ax2e.legend()
        ax2e.grid(True, alpha=0.3, axis="y")
        ax2e.spines["top"].set_visible(False)
        ax2e.spines["right"].set_visible(False)

    # Panel F
    ax2f = fig2.add_subplot(gs2[2, 0])
    if all(m in final_scores.columns for m in ["F1", "AUPRC", "Net Benefit"]):
        metrics_for_contrib = ["F1", "AUPRC", "Net Benefit"]
        top_5_teams = (
            final_scores.nlargest(5, "Weighted Score")
            if "Weighted Score" in final_scores.columns
            else final_scores.head(5)
        )
        normalized_contrib = []
        for metric in metrics_for_contrib:
            if metric in top_5_teams.columns:
                values = top_5_teams[metric].values
                metric_weight = factor_weights[factor_weights["feature"] == metric]["weight"].values
                if len(metric_weight) > 0:
                    weighted_values = values * metric_weight[0]
                    normalized_contrib.append((weighted_values, metric))

        if normalized_contrib:
            x_pos = np.arange(len(top_5_teams))
            bottom = np.zeros(len(top_5_teams))
            colors_comp = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C"]
            for i, (contrib, label) in enumerate(normalized_contrib):
                ax2f.bar(
                    x_pos,
                    contrib,
                    bottom=bottom,
                    label=label,
                    color=colors_comp[i % len(colors_comp)],
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottom += contrib

            ax2f.set_xticks(x_pos)
            ax2f.set_xticklabels([f"Rank {i + 1}" for i in range(len(top_5_teams))])
            ax2f.set_ylabel("Weighted Contribution", fontweight="bold")
            ax2f.set_title("F. Metric Contributions\n(Top 5 Teams)", fontweight="bold", fontsize=13)
            ax2f.legend(loc="best", fontsize=8)
            ax2f.grid(True, alpha=0.3, axis="y")
            ax2f.spines["top"].set_visible(False)
            ax2f.spines["right"].set_visible(False)

    # Panel G
    ax2g = fig2.add_subplot(gs2[2, 1])
    if "ECE" in prelim_sens.columns:
        ece_prelim = prelim_sens["ECE"].dropna()
        q1, q99 = ece_prelim.quantile([0.01, 0.99])
        ece_clean = ece_prelim[(ece_prelim >= q1) & (ece_prelim <= q99)]

        ax2g.hist(
            ece_clean,
            bins=20,
            color="#E74C3C",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )
        ax2g.axvline(
            ece_clean.median(),
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"Median: {ece_clean.median():.3f}",
        )
        ax2g.set_xlabel("Expected Calibration Error", fontweight="bold")
        ax2g.set_ylabel("Number of Teams", fontweight="bold")
        ax2g.set_title("G. Calibration Quality\n(Preliminary Phase)", fontweight="bold", fontsize=13)
        ax2g.legend()
        ax2g.grid(True, alpha=0.3, axis="y")
        ax2g.spines["top"].set_visible(False)
        ax2g.spines["right"].set_visible(False)

    # Panel H
    ax2h = fig2.add_subplot(gs2[2, 2])
    stability_metrics = ["AUC", "F1", "AUPRC"]
    prelim_medians = [prelim_sens[m].median() for m in stability_metrics if m in prelim_sens.columns]
    valid_medians = [validation_sens[m].median() for m in stability_metrics if m in validation_sens.columns]
    x_pos = np.arange(len(stability_metrics))
    width = 0.35

    ax2h.bar(
        x_pos - width / 2,
        prelim_medians,
        width,
        label="Preliminary",
        color="#3498DB",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    ax2h.bar(
        x_pos + width / 2,
        valid_medians,
        width,
        label="Validation",
        color="#2ECC71",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    ax2h.set_xticks(x_pos)
    ax2h.set_xticklabels(stability_metrics)
    ax2h.set_ylabel("Median Value", fontweight="bold")
    ax2h.set_title("H. Performance Stability\nAcross Phases", fontweight="bold", fontsize=13)
    ax2h.legend(loc="best", fontsize=9)
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
