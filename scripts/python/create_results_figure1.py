#!/usr/bin/env python3
"""Generate Figure 1: Study design overview and decision analytics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
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


def remove_outliers_nb(data: np.ndarray) -> np.ndarray:
    """Remove extreme net benefit outliers via median ± 3*IQR."""
    data = np.asarray(data)
    if data.size == 0:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    mask = (data >= lower_bound) & (data <= upper_bound)
    filtered = data[mask]
    return filtered if filtered.size > 0 else data


def compute_treat_all_nb(df: pd.DataFrame) -> np.ndarray:
    """Compute treat-all net benefit per submission."""
    required_cols = {"TP", "FP", "FN", "TN", "Optimal Threshold"}
    if not required_cols.issubset(df.columns):
        return np.array([])
    totals = df["TP"] + df["FP"] + df["FN"] + df["TN"]
    totals = totals.replace(0, np.nan)
    prevalence = (df["TP"] + df["FN"]) / totals
    pt = df["Optimal Threshold"].clip(1e-6, 1 - 1e-6)
    treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    treat_all = treat_all.replace([np.inf, -np.inf], np.nan).dropna()
    return treat_all.to_numpy()


def filter_nb_outliers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return subset with extreme net benefit rows removed."""
    if "Net Benefit" not in df.columns or df["Net Benefit"].dropna().empty:
        return df
    nb_series = df["Net Benefit"].dropna()
    q1, q3 = np.percentile(nb_series, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    mask = (df["Net Benefit"] >= lower_bound) & (df["Net Benefit"] <= upper_bound)
    return df[mask]


def compute_treat_all_series(df: pd.DataFrame) -> pd.Series:
    """Compute treat-all NB aligned with dataframe entries."""
    required_cols = {"TP", "FP", "FN", "TN", "Optimal Threshold"}
    if not required_cols.issubset(df.columns):
        return pd.Series(dtype=float)
    totals = df["TP"] + df["FP"] + df["FN"] + df["TN"]
    totals = totals.replace(0, np.nan)
    prevalence = (df["TP"] + df["FN"]) / totals
    pt = df["Optimal Threshold"].clip(1e-6, 1 - 1e-6)
    treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    return treat_all.replace([np.inf, -np.inf], np.nan)


def aggregate_dca_curve(df: pd.DataFrame, threshold_bins_pct: np.ndarray) -> pd.DataFrame:
    """Aggregate decision-curve net benefit across threshold bins."""
    cols = ["Optimal Threshold", "Net Benefit", "TP", "FP", "FN", "TN"]
    df_curve = df[cols].dropna(subset=["Optimal Threshold", "Net Benefit"]).copy()
    if df_curve.empty:
        return pd.DataFrame(columns=["threshold_pct", "net_benefit"])
    df_curve = df_curve[(df_curve["Optimal Threshold"] > 0) & (df_curve["Optimal Threshold"] < 1)]
    df_curve = filter_nb_outliers_df(df_curve)
    if df_curve.empty:
        return pd.DataFrame(columns=["threshold_pct", "net_benefit"])
    df_curve["threshold_pct"] = df_curve["Optimal Threshold"] * 100
    df_curve["bin"] = pd.cut(df_curve["threshold_pct"], bins=threshold_bins_pct, include_lowest=True)
    curve_stats = df_curve.groupby("bin", observed=False).agg(
        threshold_pct=("threshold_pct", "mean"),
        net_benefit=("Net Benefit", "median"),
    ).dropna()
    return curve_stats


def generate_figure1() -> Path:
    """Create Figure 1 and return the output path."""
    print("\n📈 Creating Figure 1: Study Design and Participant Flow...")

    final_scores = pd.read_csv(RAW_DIR / "final_scores.csv")
    prelim_sens = pd.read_csv(PRELIM_DIR / "evaluation_metrics_sensitivity_method.csv")
    prelim_f1 = pd.read_csv(PRELIM_DIR / "evaluation_metrics_f1_method.csv")
    prelim_youden = pd.read_csv(PRELIM_DIR / "evaluation_metrics_youden_method.csv")
    validation_sens = pd.read_csv(VALIDATION_DIR / "evaluation_metrics_sensitivity_method.csv")

    final_scores["Team_clean"] = final_scores["Team name"].str.encode("ascii", errors="ignore").str.decode("ascii")
    validation_sens["Team_clean"] = validation_sens["Team"].apply(extract_team_name)
    prelim_sens["Team_clean"] = prelim_sens["Team"].apply(extract_team_name)
    prelim_f1["Team_clean"] = prelim_f1["Team"].apply(extract_team_name)
    prelim_youden["Team_clean"] = prelim_youden["Team"].apply(extract_team_name)

    fig1 = plt.figure(figsize=(18, 18))
    gs1 = GridSpec(4, 4, figure=fig1, hspace=0.32, wspace=0.35)

    # Panel A
    ax1a = fig1.add_subplot(gs1[0, 0])
    n_prelim = len(prelim_sens)
    n_prelim_teams = prelim_sens["Team_clean"].nunique()
    n_valid = len(validation_sens)
    n_valid_teams = validation_sens["Team_clean"].nunique()
    n_final_subs = len(final_scores)
    n_final_teams = final_scores["Team name"].nunique()

    print(
        "  Panel A - Phase participation: Prelim={0} ({1} teams), Validation={2} ({3} teams), Final={4} ({5} teams)".format(
            n_prelim,
            n_prelim_teams,
            n_valid,
            n_valid_teams,
            n_final_subs,
            n_final_teams,
        )
    )

    categories = ["Preliminary", "Dummy/Validation", "Final"]
    total_values = [n_prelim, n_valid, n_final_subs]
    unique_values = [n_prelim_teams, n_valid_teams, n_final_teams]
    colors_bar = ["#3498DB", "#2ECC71", "#9B59B6"]
    bars = ax1a.bar(categories, total_values, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)

    for bar, total, unique in zip(bars, total_values, unique_values):
        height = bar.get_height()
        ax1a.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(total)}", ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax1a.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2.0,
            f"{int(unique)} unique\nteams",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
        )

    ax1a.set_ylabel("Number of Submissions", fontweight="bold")
    ax1a.set_title("A. Multi-Phase Participation", fontweight="bold", fontsize=13)
    ax1a.set_ylim(0, max(total_values) * 1.15)
    ax1a.spines["top"].set_visible(False)
    ax1a.spines["right"].set_visible(False)
    ax1a.grid(True, alpha=0.3, axis="y")

    # Panel B
    ax1b = fig1.add_subplot(gs1[0, 1:])
    sens_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    methods_data = {"Sensitivity": prelim_sens, "Youden": prelim_youden, "F1 Opt": prelim_f1}
    method_colors = {"Sensitivity": "#1F77B4", "Youden": "#E74C3C", "F1 Opt": "#2ECC71"}

    fractions_by_method = {}
    nb_by_method = {}

    for method, df_method in methods_data.items():
        fractions = []
        nb_values = []
        total = len(df_method)
        for thresh in sens_thresholds:
            subset = df_method[df_method["Sensitivity"] >= thresh]
            fractions.append(len(subset) / total if total else 0)
            if len(subset) > 0:
                subset_nb = subset[["Net Benefit"]].dropna()
                if not subset_nb.empty:
                    nb_filtered = remove_outliers_nb(subset_nb["Net Benefit"].values)
                    nb_values.append(np.median(nb_filtered) if nb_filtered.size > 0 else np.median(subset_nb["Net Benefit"]))
                else:
                    nb_values.append(np.nan)
            else:
                nb_values.append(np.nan)
        fractions_by_method[method] = np.array(fractions)
        nb_by_method[method] = np.array(nb_values, dtype=float)

    x_positions = np.arange(len(sens_thresholds))
    fraction_lines = []
    nb_lines = []
    chosen_idx = sens_thresholds.index(0.80)

    for method, fractions in fractions_by_method.items():
        color = method_colors[method]
        line, = ax1b.plot(x_positions, fractions, marker="o", linewidth=2.2, color=color, label=f"{method}: fraction ≥ threshold")
        fraction_lines.append(line)
        ax1b.fill_between(x_positions, fractions, color=color, alpha=0.10)
        ax1b.scatter(x_positions[chosen_idx], fractions[chosen_idx], color=color, s=70, edgecolor="black", zorder=5)
        ax1b.annotate(
            f"{method}: {fractions[chosen_idx] * 100:.0f}%",
            xy=(x_positions[chosen_idx], fractions[chosen_idx]),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    ax1b.set_ylabel("Fraction of Models Meeting Sensitivity Threshold", fontweight="bold")
    ax1b.set_ylim(0, 1.05)

    ax1b_secondary = ax1b.twinx()
    nb_percent_all = []
    for method, nb_vals in nb_by_method.items():
        color = method_colors[method]
        nb_percent = nb_vals * 100.0
        nb_percent_all.append(nb_percent[np.isfinite(nb_percent)])
        line_nb, = ax1b_secondary.plot(
            x_positions,
            nb_percent,
            linestyle="--",
            linewidth=1.8,
            marker="s",
            color=color,
            label=f"{method}: median NB (%)",
        )
        nb_lines.append(line_nb)

    if nb_percent_all:
        nb_concat = np.hstack(nb_percent_all)
        nb_range = np.nanmax(nb_concat) - np.nanmin(nb_concat)
        lower = np.nanmin(nb_concat) - 0.1 * nb_range if nb_range > 0 else np.nanmin(nb_concat) - 1
        upper = np.nanmax(nb_concat) + 0.1 * nb_range if nb_range > 0 else np.nanmax(nb_concat) + 1
        ax1b_secondary.set_ylim(lower, upper)
    else:
        ax1b_secondary.set_ylim(-5, 5)

    ax1b_secondary.set_ylabel("Median Net Benefit (%)", fontweight="bold")
    ax1b_secondary.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))

    ax1b.axvline(x=chosen_idx, color="#34495E", linestyle="--", linewidth=2.5, alpha=0.7)
    ax1b.text(
        0.02,
        0.92,
        "Vertical line marks ≥80% sensitivity requirement\nCompare coverage fractions across methods",
        transform=ax1b.transAxes,
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#34495E", alpha=0.85),
    )

    ax1b.set_xticks(x_positions)
    ax1b.set_xticklabels([f"{int(t * 100)}%" for t in sens_thresholds])
    ax1b.set_xlabel("Sensitivity Threshold Requirement", fontweight="bold")
    ax1b.set_title(
        "B. Why Require ≥80% Sensitivity?\nCoverage vs. utility across threshold optimization methods",
        fontweight="bold",
        fontsize=13,
    )
    ax1b.grid(True, alpha=0.3, axis="y")
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_visible(False)
    ax1b_secondary.spines["top"].set_visible(False)

    handles = fraction_lines + nb_lines
    labels = [h.get_label() for h in handles]
    ax1b.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)

    # Panel C
    ax1c = fig1.add_subplot(gs1[1, 0:2])
    methods_nb = {
        "Sensitivity": prelim_sens["Net Benefit"].dropna().values,
        "Youden": prelim_youden["Net Benefit"].dropna().values,
        "F1 Opt": prelim_f1["Net Benefit"].dropna().values,
    }
    methods_nb_clean = {k: remove_outliers_nb(v) for k, v in methods_nb.items()}
    print(
        "  Panel C - Net Benefit data points after filtering: Sensitivity={0}, Youden={1}, F1={2}".format(
            len(methods_nb_clean["Sensitivity"]),
            len(methods_nb_clean["Youden"]),
            len(methods_nb_clean["F1 Opt"]),
        )
    )

    treat_all_arrays = [compute_treat_all_nb(prelim_sens), compute_treat_all_nb(prelim_youden), compute_treat_all_nb(prelim_f1)]
    treat_all_arrays = [arr for arr in treat_all_arrays if arr.size > 0]
    treat_all_baseline = None
    if treat_all_arrays:
        treat_all_combined = np.concatenate(treat_all_arrays)
        treat_all_filtered = remove_outliers_nb(treat_all_combined)
        if treat_all_filtered.size > 0:
            treat_all_baseline = np.median(treat_all_filtered)

    methods_nb_percent = {k: v * 100.0 for k, v in methods_nb_clean.items()}
    positions = [0, 1, 2]
    parts = ax1c.violinplot(list(methods_nb_percent.values()), positions=positions, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_bar[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.2)

    ax1c.set_xticks(positions)
    ax1c.set_xticklabels(list(methods_nb_clean.keys()))
    ax1c.set_ylabel("Net Benefit (%)", fontweight="bold")
    ax1c.set_title("C. Net Benefit by Threshold Method\n(Preliminary Phase)", fontweight="bold", fontsize=13)
    ax1c.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Treat-none baseline")
    if treat_all_baseline is not None:
        treat_all_percent = treat_all_baseline * 100.0
        ax1c.axhline(
            y=treat_all_percent,
            color="#8E44AD",
            linestyle=":",
            linewidth=2,
            label=f"Treat-all baseline (median NB = {treat_all_percent:.1f}%)",
        )
    ax1c.legend(loc="best", fontsize=8)
    ax1c.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax1c.grid(True, alpha=0.3, axis="y")
    ax1c.spines["top"].set_visible(False)
    ax1c.spines["right"].set_visible(False)

    # Panel D
    ax1d = fig1.add_subplot(gs1[1, 2:])
    threshold_bins_pct = np.linspace(5, 95, 19)
    for method, df_method in methods_data.items():
        curve = aggregate_dca_curve(df_method, threshold_bins_pct)
        if not curve.empty:
            ax1d.plot(
                curve["threshold_pct"],
                curve["net_benefit"] * 100,
                marker="o",
                linewidth=2,
                color=method_colors[method],
                label=f"{method} net benefit",
            )

    prelim_for_dca = filter_nb_outliers_df(
        prelim_sens[["Optimal Threshold", "TP", "FP", "FN", "TN", "Net Benefit"]]
        .dropna(subset=["Optimal Threshold", "Net Benefit"])
        .copy()
    )
    if not prelim_for_dca.empty:
        treat_all_series = compute_treat_all_series(prelim_for_dca)
        prelim_for_dca["Treat All NB"] = treat_all_series
        prelim_for_dca = prelim_for_dca.dropna(subset=["Treat All NB"])
        prelim_for_dca["threshold_pct"] = prelim_for_dca["Optimal Threshold"] * 100
        prelim_for_dca["bin"] = pd.cut(prelim_for_dca["threshold_pct"], bins=threshold_bins_pct, include_lowest=True)
        treat_all_curve = prelim_for_dca.groupby("bin", observed=False).agg(
            threshold_pct=("threshold_pct", "mean"),
            treat_all=("Treat All NB", "median"),
        ).dropna()
        if not treat_all_curve.empty:
            ax1d.plot(
                treat_all_curve["threshold_pct"],
                treat_all_curve["treat_all"] * 100,
                linestyle="--",
                linewidth=2,
                color="#8E44AD",
                label="Treat-all strategy",
            )

    ax1d.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Treat-none strategy")
    ax1d.set_xlabel("Threshold Probability (%)", fontweight="bold")
    ax1d.set_ylabel("Net Benefit (%)", fontweight="bold")
    ax1d.set_title("D. Decision Curve Analysis\n(Aggregated across submissions)", fontweight="bold", fontsize=13)
    ax1d.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax1d.grid(True, alpha=0.3)
    ax1d.spines["top"].set_visible(False)
    ax1d.spines["right"].set_visible(False)
    ax1d.legend(loc="best", fontsize=8, framealpha=0.9)

    # Panel E
    ax1e = fig1.add_subplot(gs1[2, 0:4])

    prelim_sens["Precision"] = prelim_sens["TP"] / (prelim_sens["TP"] + prelim_sens["FP"])
    prelim_sens["FPR"] = prelim_sens["FP"] / (prelim_sens["FP"] + prelim_sens["TN"])
    metrics_to_plot = ["F1", "Net Benefit", "ECE", "AUPRC", "AUC", "Sensitivity", "Specificity", "Precision"]
    metrics_display = ["F1", "Net Benefit", "ECE", "AUPRC", "AUC", "Sensitivity", "Specificity", "Precision"]

    def remove_outliers_iqr_series(series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        mask = (series >= lower_bound) & (series <= upper_bound)
        filtered = series[mask]
        return filtered if len(filtered) > 0 else series

    metrics_data = []
    for metric in metrics_to_plot:
        if metric in prelim_sens.columns:
            clean_data = remove_outliers_iqr_series(prelim_sens[metric].dropna())
            metrics_data.append(clean_data.values)
        else:
            metrics_data.append([])

    positions = list(range(len(metrics_display)))
    parts = ax1e.violinplot(metrics_data, positions=positions, showmeans=True, showmedians=True, widths=0.7)

    colors_gradient = plt.cm.Set3(np.linspace(0, 1, len(metrics_display)))
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_gradient[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1.5)

    ax1e.set_xticks(positions)
    ax1e.set_xticklabels(metrics_display, rotation=45, ha="right")
    ax1e.set_ylabel("Metric Value", fontweight="bold")
    ax1e.set_title("E. Key Performance Metrics Distribution\n(Preliminary Phase, Outliers Removed)", fontweight="bold", fontsize=13)
    ax1e.grid(True, alpha=0.3, axis="y")
    ax1e.spines["top"].set_visible(False)
    ax1e.spines["right"].set_visible(False)
    ax1e.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1e.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Panel F
    ax1f = fig1.add_subplot(gs1[3, 0:4])
    metrics_for_corr = ["AUC", "AUPRC", "F1", "Sensitivity", "Specificity", "Precision", "Net Benefit", "ECE"]
    corr_data = prelim_sens[metrics_for_corr].corr()
    selected_metrics = ["Net Benefit", "F1", "AUPRC", "ECE", "Sensitivity"]
    other_metrics = ["AUC", "Specificity", "Precision"]
    metric_order = selected_metrics + other_metrics
    corr_reordered = corr_data.loc[metric_order, metric_order]

    from matplotlib.patches import Rectangle

    im = ax1f.imshow(corr_reordered.abs(), cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    for i in range(len(metric_order)):
        for j in range(len(metric_order)):
            ax1f.text(
                j,
                i,
                f"{corr_reordered.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold" if abs(corr_reordered.iloc[i, j]) < 0.7 else "normal",
            )

    for i in range(len(selected_metrics)):
        for j in range(len(selected_metrics)):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=3)
            ax1f.add_patch(rect)

    ax1f.set_xticks(np.arange(len(metric_order)))
    ax1f.set_yticks(np.arange(len(metric_order)))
    ax1f.set_xticklabels(metric_order, rotation=45, ha="right", fontsize=10)
    ax1f.set_yticklabels(metric_order, fontsize=10)

    cbar_f = plt.colorbar(im, ax=ax1f, fraction=0.046, pad=0.04)
    cbar_f.set_label("Absolute Correlation", fontweight="bold", fontsize=10)
    ax1f.set_title(
        "F. Metric Selection Rationale for Final Ranking\n(Blue box: Selected metrics show complementary information with lower inter-correlation)",
        fontweight="bold",
        fontsize=13,
        pad=20,
    )

    output_path = FIGURES_DIR / "results_figure1_overview.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"   ✓ Saved: {output_path.name}")
    return output_path


def main() -> None:
    generate_figure1()


if __name__ == "__main__":
    main()
