#!/usr/bin/env python3
"""Generate Figure 3: Performance rankings and comparisons."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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


def extract_team_name(full_name: str) -> str:
    """Extract clean team name from filename conventions."""
    if pd.isna(full_name):
        return full_name
    name = full_name.replace("output_", "")
    if "_https" in name:
        team_name = name.split("_https")[0]
    elif "_git" in name:
        team_name = name.split("_git")[0]
    else:
        team_name = name
    if team_name == "TeamAIMS2":
        team_name = "TeamAIMS"
    return team_name


def generate_figure3() -> Path:
    """Create Figure 3 and return the output path."""
    print("\n📈 Creating Figure 3: Performance Rankings...")

    final_scores = pd.read_csv(RAW_DIR / "final_scores.csv")
    final_scores["Team_clean"] = final_scores["Team name"].apply(extract_team_name)

    fig3 = plt.figure(figsize=(16, 10))
    gs3 = GridSpec(2, 2, figure=fig3, hspace=0.35, wspace=0.3)

    # Panel A
    ax3a = fig3.add_subplot(gs3[0, :])
    if "Weighted Score" in final_scores.columns:
        top_10 = final_scores.nlargest(10, "Weighted Score").copy()
        top_10["Rank"] = range(1, 11)
        team_labels = top_10["Team_clean"].str[:40]
        colors_rank = plt.cm.viridis(np.linspace(0, 1, 10))
        bars = ax3a.barh(
            range(len(top_10)),
            top_10["Weighted Score"].values,
            color=colors_rank,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )
        ax3a.set_yticks(range(len(top_10)))
        ax3a.set_yticklabels([f"{i + 1}. {label}" for i, label in enumerate(team_labels)], fontsize=10)
        ax3a.set_xlabel("Composite Weighted Score", fontweight="bold")
        ax3a.set_title("A. Top 10 Teams by Final Composite Score", fontweight="bold", fontsize=14)
        ax3a.invert_yaxis()
        ax3a.grid(True, alpha=0.3, axis="x")
        ax3a.spines["top"].set_visible(False)
        ax3a.spines["right"].set_visible(False)
        for i, (bar, score) in enumerate(zip(bars, top_10["Weighted Score"].values)):
            width = bar.get_width()
            ax3a.text(width + 0.02, i, f"{score:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")

    # Panel B
    ax3b = fig3.add_subplot(gs3[1, 0])
    if "Weighted Score" in final_scores.columns:
        top_5 = final_scores.nlargest(5, "Weighted Score")
        component_metrics = ["F1", "AUPRC", "Net Benefit"]
        available_components = [m for m in component_metrics if m in top_5.columns]
        if available_components:
            x_pos = np.arange(len(top_5))
            bottom = np.zeros(len(top_5))
            colors_comp = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C"]
            for i, metric in enumerate(available_components):
                values = np.abs(top_5[metric].values)
                ax3b.bar(
                    x_pos,
                    values,
                    bottom=bottom,
                    label=metric,
                    color=colors_comp[i % len(colors_comp)],
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottom += values
            ax3b.set_xticks(x_pos)
            ax3b.set_xticklabels([f"Rank {i + 1}" for i in range(len(top_5))])
            ax3b.set_ylabel("Metric Value (Stacked)", fontweight="bold")
            ax3b.set_title("B. Performance Component Breakdown\n(Top 5 Teams)", fontweight="bold", fontsize=13)
            ax3b.legend(loc="upper right", fontsize=9)
            ax3b.grid(True, alpha=0.3, axis="y")
            ax3b.spines["top"].set_visible(False)
            ax3b.spines["right"].set_visible(False)

    # Panel C
    ax3c = fig3.add_subplot(gs3[1, 1])
    if "ECE" in final_scores.columns:
        ece_values = final_scores["ECE"].dropna()
        bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
        labels = [
            "Excellent\n(<0.05)",
            "Good\n(0.05-0.10)",
            "Fair\n(0.10-0.15)",
            "Poor\n(0.15-0.20)",
            "Very Poor\n(>0.20)",
        ]
        ece_categories = pd.cut(ece_values, bins=bins, labels=labels)
        category_counts = ece_categories.value_counts().sort_index()
        colors_cal = ["#2ECC71", "#3498DB", "#F39C12", "#E67E22", "#E74C3C"]
        bars = ax3c.bar(
            range(len(category_counts)),
            category_counts.values,
            color=colors_cal,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax3c.set_xticks(range(len(category_counts)))
        ax3c.set_xticklabels(category_counts.index, fontsize=9)
        ax3c.set_ylabel("Number of Teams", fontweight="bold")
        ax3c.set_title(
            "C. Model Calibration Quality\n(Expected Calibration Error)",
            fontweight="bold",
            fontsize=13,
        )
        ax3c.grid(True, alpha=0.3, axis="y")
        ax3c.spines["top"].set_visible(False)
        ax3c.spines["right"].set_visible(False)
        for bar, count in zip(bars, category_counts.values):
            height = bar.get_height()
            ax3c.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(count)}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    output_path = FIGURES_DIR / "results_figure3_rankings.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path


def main() -> None:
    generate_figure3()


if __name__ == "__main__":
    main()
