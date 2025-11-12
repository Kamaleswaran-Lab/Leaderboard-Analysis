#!/usr/bin/env python3
"""Generate Figure 1: Study design overview and decision analytics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex
from pathlib import Path

from plot_palette import BASE_PALETTE, get_metric_color, lighten_color

ROOT = Path(__file__).resolve().parents[2]
PRELIM_DIR = ROOT / "optimal_threshold_results"
VALIDATION_DIR = ROOT / "optimal_threshold_results_validation"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "figures" / "main_figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


plt.style.use("seaborn-v0_8-paper")
sns.set_palette(BASE_PALETTE)


def pastelize(color: str, blend: float = 0.45) -> str:
    """Blend a base color toward white to recreate the prior pastel tones."""
    rgb = np.array(to_rgb(color))
    pastel_rgb = rgb + (1 - rgb) * blend
    return to_hex(pastel_rgb)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


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


def generate_figure1() -> Path:
    """Create Figure 1 and return the output path."""
    print("\n📈 Creating Figure 1: Study Design and Participant Flow...")


    final_scores = pd.read_csv(RAW_DIR / "final_scores.csv")
    prelim_sens = pd.read_csv(PRELIM_DIR / "evaluation_metrics_sensitivity_method.csv")
    prelim_f1 = pd.read_csv(PRELIM_DIR / "evaluation_metrics_f1_method.csv")
    prelim_youden = pd.read_csv(PRELIM_DIR / "evaluation_metrics_youden_method.csv")
    validation_sens = pd.read_csv(VALIDATION_DIR / "evaluation_metrics_sensitivity_method.csv")

    # Keep unfiltered copies to display true participant counts in Panel A
    prelim_sens_all = prelim_sens.copy()
    prelim_f1_all = prelim_f1.copy()
    prelim_youden_all = prelim_youden.copy()
    validation_sens_all = validation_sens.copy()
    final_scores_all = final_scores.copy()

    # Remove outlier submissions with AUC <= 0.5
    if "AUC" in prelim_sens.columns:
        prelim_sens = prelim_sens[prelim_sens["AUC"] > 0.5]
    if "AUC" in prelim_f1.columns:
        prelim_f1 = prelim_f1[prelim_f1["AUC"] > 0.5]
    if "AUC" in prelim_youden.columns:
        prelim_youden = prelim_youden[prelim_youden["AUC"] > 0.5]
    if "AUC" in validation_sens.columns:
        validation_sens = validation_sens[validation_sens["AUC"] > 0.5]
    if "AUC" in final_scores.columns:
        final_scores = final_scores[final_scores["AUC"] > 0.5]

    # create cleaned team labels for the unfiltered copies (used in Panel A)
    final_scores_all["Team_clean"] = final_scores_all["Team name"].str.encode("ascii", errors="ignore").str.decode("ascii").str.strip()
    validation_sens_all["Team_clean"] = validation_sens_all["Team"].apply(extract_team_name)
    prelim_sens_all["Team_clean"] = prelim_sens_all["Team"].apply(extract_team_name)
    prelim_f1_all["Team_clean"] = prelim_f1_all["Team"].apply(extract_team_name)
    prelim_youden_all["Team_clean"] = prelim_youden_all["Team"].apply(extract_team_name)

    # Now apply existing cleaning/filtering to the working dataframes (these may be filtered by AUC later)
    final_scores["Team_clean"] = final_scores["Team name"].str.encode("ascii", errors="ignore").str.decode("ascii")
    validation_sens["Team_clean"] = validation_sens["Team"].apply(extract_team_name)
    prelim_sens["Team_clean"] = prelim_sens["Team"].apply(extract_team_name)
    prelim_f1["Team_clean"] = prelim_f1["Team"].apply(extract_team_name)
    prelim_youden["Team_clean"] = prelim_youden["Team"].apply(extract_team_name)
    final_scores["Team_clean"] = final_scores["Team_clean"].str.strip()

    # Do not filter by AUC; include all teams
    # Do NOT filter validation_sens by AUC

    if {"TP", "FP", "FN", "TN"}.issubset(prelim_sens.columns):
        prelim_sens["MCC"] = compute_mcc(
            prelim_sens["TP"],
            prelim_sens["FP"],
            prelim_sens["FN"],
            prelim_sens["TN"],
        )

    fig1 = plt.figure(figsize=(17, 16))
    gs1 = GridSpec(
        3,
        4,
        figure=fig1,
        hspace=0.42,
        wspace=0.36,
        width_ratios=[1.9, 0.9, 0.9, 0.9],
        height_ratios=[1.0, 1.1, 1.25],
    )
    fig1.subplots_adjust(right=0.85)

    # Panel A
    ax1a = fig1.add_subplot(gs1[0, 0])
    # counts and unique teams per phase — use unfiltered copies so Panel A shows all participants
    n_prelim = len(prelim_sens_all)
    n_prelim_teams = prelim_sens_all["Team_clean"].nunique()
    n_valid = len(validation_sens_all)
    n_valid_teams = validation_sens_all["Team_clean"].nunique()
    n_final_subs = len(final_scores_all)
    n_final_teams = final_scores_all["Team_clean"].nunique()

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
    colors_bar = [pastelize(BASE_PALETTE[i % len(BASE_PALETTE)], 0.5) for i in range(len(categories))]
    bars = ax1a.bar(categories, total_values, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=1.5)

    for idx, (bar, total, unique) in enumerate(zip(bars, total_values, unique_values)):
        if idx == 0:
            bar.set_width(bar.get_width() * 1.2)
            bar.set_x(bar.get_x() - bar.get_width() * 0.1)
        height = bar.get_height()
        ax1a.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(total)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
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
    ax1a.grid(True, alpha=0.3, axis="y")
    # Panel B
    ax1b = fig1.add_subplot(gs1[0, 1:])
    sens_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    methods_data = {"Sensitivity": prelim_sens, "Youden": prelim_youden, "F1 Opt": prelim_f1}
    method_color_alias = {"Sensitivity": "Sensitivity", "Youden": "Youden", "F1 Opt": "F1"}
    method_colors = {
        method: pastelize(get_metric_color(alias), 0.35)
        for method, alias in method_color_alias.items()
    }

    method_offsets = {"Sensitivity": -0.08, "Youden": 0.0, "F1 Opt": 0.08}
    fractions_by_method = {}
    nb_by_method = {}
    nb_lower_bounds = {}

    for method, df_method in methods_data.items():
        nb_method_vals = df_method["Net Benefit"].dropna().values
        if nb_method_vals.size > 0:
            nb_lower_bounds[method] = np.percentile(nb_method_vals, 5)
        else:
            nb_lower_bounds[method] = None

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
                    nb_values_raw = subset_nb["Net Benefit"].values
                    nb_filtered = remove_outliers_nb(nb_values_raw)
                    lower_bound = nb_lower_bounds.get(method)
                    if lower_bound is not None:
                        filtered_custom = nb_filtered[nb_filtered >= lower_bound]
                        if filtered_custom.size > 0:
                            nb_filtered = filtered_custom
                    nb_filtered = nb_filtered[np.isfinite(nb_filtered)]
                    median_nb = np.median(nb_filtered) if nb_filtered.size > 0 else np.nan
                    nb_values.append(median_nb)
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
        x_positions_offset = x_positions + method_offsets.get(method, 0.0)
        line, = ax1b.plot(x_positions_offset, fractions, marker="o", linewidth=2.2, color=color, label=f"{method}: fraction ≥ threshold")
        fraction_lines.append(line)
        ax1b.fill_between(
            x_positions_offset,
            fractions,
            color=color,
            alpha=0.25,
            zorder=1,
        )
        ax1b.scatter(x_positions_offset[chosen_idx], fractions[chosen_idx], color=color, s=80, edgecolor="black", zorder=5)

    ax1b.set_ylabel("Fraction of models", fontweight="bold")
    ax1b.set_ylim(0, 1.12)
    ax1b.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

    ax1b_secondary = ax1b.twinx()
    for method, nb_vals in nb_by_method.items():
        color = method_colors[method]
        nb_percent = nb_vals * 100.0
        x_positions_offset = x_positions + method_offsets.get(method, 0.0)
        line_nb, = ax1b_secondary.plot(
            x_positions_offset,
            nb_percent,
            linestyle="--",
            linewidth=1.8,
            marker="s",
            color=color,
            label=f"{method}: median NB (%)",
        )
        nb_lines.append(line_nb)

    ax1b_secondary.set_ylim(-25, 7)
    ax1b_secondary.set_yticks([-20, -15, -10, -5, 0, 5])

    ax1b_secondary.set_ylabel("Median Net Benefit (percent)", fontweight="bold")
    ax1b_secondary.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax1b_secondary.axhline(0, color="gray", linestyle=":", linewidth=1)

    ax1b.axvline(x=chosen_idx, color="#34495E", linestyle="--", linewidth=2.0, alpha=0.6)

    ax1b.set_xticks(x_positions)
    ax1b.set_xticklabels([f"{int(t * 100)}" for t in sens_thresholds])
    ax1b.set_xlabel("Sensitivity Threshold Requirement (percent)", fontweight="bold")
    ax1b.set_title(
        "B. Why Require ≥80% Sensitivity?\nCoverage vs. utility across threshold optimization methods",
        fontweight="bold",
        fontsize=13,
    )
    ax1b.grid(True, alpha=0.3, axis="y")
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_visible(False)
    ax1b_secondary.spines["top"].set_visible(False)

    legend_handles = []
    for method in methods_data.keys():
        legend_handles.append(
            Line2D([0], [0], color=method_colors[method], marker="o", linestyle="-", linewidth=2.2, label=f"{method}: coverage")
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=method_colors[method],
                marker="s",
                linestyle="--",
                linewidth=1.8,
                dashes=(4, 2),
                label=f"{method}: median NB",
            )
        )

    ax1b.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper left",
        bbox_to_anchor=(0.73, 0.8),
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0.4,
        handlelength=2.5,
    )
    ax1b.margins(x=0.05)

    # Panel C
    ax1c = fig1.add_subplot(gs1[1, 0])
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

    # Statistical diff test p-values between each method (Mann-Whitney U)
    from scipy.stats import mannwhitneyu
    method_names = list(methods_nb_clean.keys())
    pval_matrix = np.full((3, 3), np.nan)
    stat_matrix = np.full((3, 3), np.nan)
    for i in range(3):
        for j in range(i+1, 3):
            x = methods_nb_clean[method_names[i]]
            y = methods_nb_clean[method_names[j]]
            if len(x) > 0 and len(y) > 0:
                stat, pval = mannwhitneyu(x, y, alternative="two-sided")
                pval_matrix[i, j] = pval
                pval_matrix[j, i] = pval
                stat_matrix[i, j] = stat
                stat_matrix[j, i] = stat



    # Compute treat-all baseline for each method
    treat_all_sens = compute_treat_all_nb(prelim_sens)
    treat_all_youden = compute_treat_all_nb(prelim_youden)
    treat_all_f1 = compute_treat_all_nb(prelim_f1)
    treat_all_sens_median = np.median(treat_all_sens) * 100.0
    treat_all_youden_median = np.median(treat_all_youden) * 100.0
    treat_all_f1_median = np.median(treat_all_f1) * 100.0

    methods_nb_percent = {k: v * 100.0 for k, v in methods_nb_clean.items()}
    positions = [0, 1, 2]
    parts = ax1c.violinplot(list(methods_nb_percent.values()), positions=positions, showmeans=True, showmedians=True)
    violin_colors = [lighten_color(get_metric_color(method_color_alias.get(name, name)), 0.25) for name in methods_nb_clean.keys()]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.2)

    ax1c.set_xticks(positions)
    ax1c.set_xticklabels(list(methods_nb_clean.keys()))
    ax1c.set_ylabel("Net Benefit (percent)", fontweight="bold")
    ax1c.set_title("C. Net Benefit by Threshold Method\n(Preliminary Phase)", fontweight="bold", fontsize=13)
    ax1c.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Treat-none baseline")
    ax1c.axhline(y=treat_all_sens_median, color="#8E44AD", linestyle=":", linewidth=2, label=f"Treat-all Sensitivity (median NB = {treat_all_sens_median:.1f}%)")
    ax1c.axhline(y=treat_all_youden_median, color="#27AE60", linestyle=":", linewidth=2, label=f"Treat-all Youden (median NB = {treat_all_youden_median:.1f}%)")
    ax1c.axhline(y=treat_all_f1_median, color="#2980B9", linestyle=":", linewidth=2, label=f"Treat-all F1 (median NB = {treat_all_f1_median:.1f}%)")
    # Move legend slightly inside the plot area at bottom-left to keep it visible but within axes
    ax1c.legend(loc="lower left", bbox_to_anchor=(0.03, 0.03), fontsize=8, ncol=1, frameon=True)
    ax1c.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax1c.grid(True, alpha=0.3, axis="y")
    ax1c.spines["top"].set_visible(False)
    ax1c.spines["right"].set_visible(False)

    # Draw lines between each pair of methods and annotate with p-value
    # Place lines and p-values at the very top of the plot
    # Find the current y-axis max and extend if needed
    y_max = max([np.max(v) if len(v) > 0 else 0 for v in methods_nb_percent.values()])
    # Place each comparison line just above the top of the violins but not overlapping.
    n_comparisons = 3
    # base just above the data max, then stagger so lines are separated but not overlapping
    y_base = y_max + 1.0
    # loosen spacing per request (increase vertical separation between comparison lines)
    stagger = 2.1
    # compute offsets but ensure we stay under the requested display cap (15)
    y_offsets = [min(y_base + stagger * k, 14.0 - stagger * (n_comparisons - 1 - k)) for k in range(n_comparisons)]
    ax1c.set_ylim(None, 15)
    # Assign y_offsets so the longest horizontal comparison (largest x1-x0)
    # is placed at the highest y (so its connector is longest).
    pair_list = []
    for i in range(3):
        for j in range(i + 1, 3):
            pair_list.append((i, j))
    # compute widths and sort pairs by width (ascending)
    pair_widths = {pair: abs(positions[pair[1]] - positions[pair[0]]) for pair in pair_list}
    pair_list_sorted = sorted(pair_list, key=lambda p: pair_widths[p])
    # map each pair to a y_offset so smallest width -> lowest y, largest width -> highest y
    pair_to_y = {pair: y for pair, y in zip(pair_list_sorted, y_offsets)}

    # draw comparisons in arbitrary order but use assigned offsets
    for (i, j), y_line in pair_to_y.items():
        x0, x1 = positions[i], positions[j]
        # Draw a horizontal line at y_line between the two methods (draw first, behind p-value box)
        ax1c.plot([x0, x1], [y_line, y_line], color="#555", linestyle="--", linewidth=1.2, alpha=0.8, zorder=6)
        # Add short vertical connector ticks from top of each violin up to the comparison line
        try:
            name_i = method_names[i]
            name_j = method_names[j]
            top_i = np.max(methods_nb_percent.get(name_i, np.array([0]))) if len(methods_nb_percent.get(name_i, [])) > 0 else 0
            top_j = np.max(methods_nb_percent.get(name_j, np.array([0]))) if len(methods_nb_percent.get(name_j, [])) > 0 else 0
        except Exception:
            top_i = top_j = 0
        vpad = 0.45  # increased vertical padding so connectors sit clearly below p-value text and are more separated vertically
        # draw vertical ticks if they won't overlap (start a bit above top)
        yi0 = top_i + vpad
        yi1 = top_j + vpad
        # draw connectors behind the p-value box
        ax1c.plot([x0, x0], [yi0, y_line], color="#555", linewidth=0.9, alpha=0.6, zorder=5)
        ax1c.plot([x1, x1], [yi1, y_line], color="#555", linewidth=0.9, alpha=0.6, zorder=5)
        # Annotate with p-value in the middle of the comparison line
        xm = (x0 + x1) / 2
        pval = pval_matrix[i, j]
        if not np.isnan(pval):
            # calculate significance stars
            if pval < 0.001:
                star = "***"
            elif pval < 0.01:
                star = "**"
            elif pval < 0.05:
                star = "*"
            else:
                star = ""
            # place p-value annotation centered on the comparison line, with opaque bbox so the line passes visually behind it
            text_label = f"p={pval:.2g}" + (f" {star}" if star else "")
            ax1c.text(
                xm,
                y_line,
                text_label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#222",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=1.0, edgecolor="#888"),
                zorder=12,
            )
    # Write statistical test info to stats TXT file
    stats_txt_path = FIGURES_DIR / "figure1_stats.txt"
    with open(stats_txt_path, "a") as f:
        f.write("\n--- Panel C: Pairwise Mann-Whitney U test results ---\n")
        for i in range(3):
            for j in range(i+1, 3):
                stat = stat_matrix[i, j]
                pval = pval_matrix[i, j]
                f.write(f"{method_names[i]} vs {method_names[j]}: U={stat:.2f}, p={pval:.3g}\n")

    # Panel D
    ax1d = fig1.add_subplot(gs1[1, 1:4])

    prelim_sens["Precision"] = prelim_sens["TP"] / (prelim_sens["TP"] + prelim_sens["FP"])
    prelim_sens["FPR"] = prelim_sens["FP"] / (prelim_sens["FP"] + prelim_sens["TN"])
    metrics_to_plot = ["F1", "ECE", "AUPRC", "AUC", "Sensitivity", "Specificity", "Precision"]
    metrics_display = ["F1", "ECE", "AUPRC", "AUC", "Sensitivity", "Specificity", "Precision"]

    def remove_outliers_iqr_series(series: pd.Series) -> pd.Series:
        # Outlier trimming disabled per user request: return raw (non-missing) series.
        # Previous behaviour used a 3*IQR trimming; that removed valid values for
        # metrics with low uniqueness (e.g., Sensitivity). Use the raw data so
        # distributions reflect the post-AUC-filter universe directly.
        return series

    metrics_data = []
    for metric in metrics_to_plot:
        if metric in prelim_sens.columns:
            series = prelim_sens[metric].dropna()
            clean_data = remove_outliers_iqr_series(series)
            metrics_data.append(clean_data.values)
        else:
            metrics_data.append([])

    positions = list(range(len(metrics_display)))
    parts = ax1d.violinplot(metrics_data, positions=positions, showmeans=True, showmedians=True, widths=0.7)

    pastel_metrics = [lighten_color(get_metric_color(metric), 0.25) for metric in metrics_display]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(pastel_metrics[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1.5)

    ax1d.set_xticks(positions)
    ax1d.set_xticklabels(metrics_display, rotation=45, ha="right")
    ax1d.set_ylabel("Metric Value", fontweight="bold")
    # Note: outlier trimming removed; distributions are plotted from raw post-AUC values
    ax1d.set_title("D. Key Performance Metrics Distribution\n(Preliminary Phase)", fontweight="bold", fontsize=13)
    ax1d.grid(True, alpha=0.3, axis="y")
    ax1d.spines["top"].set_visible(False)
    ax1d.spines["right"].set_visible(False)
    ax1d.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1d.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Panel E
    ax1e = fig1.add_subplot(gs1[2, 0:4])
    metrics_for_corr = ["AUC", "AUPRC", "F1", "Sensitivity", "Specificity", "Precision", "Net Benefit", "ECE", "MCC"]
    corr_data = prelim_sens[metrics_for_corr].corr(method="spearman")
    selected_metrics = ["Sensitivity", "AUPRC", "Net Benefit", "F1", "ECE"]
    other_metrics = ["AUC", "Specificity", "Precision", "MCC"]
    metric_order = selected_metrics + other_metrics
    corr_reordered = corr_data.loc[metric_order, metric_order]

    from matplotlib.patches import Rectangle

    heatmap_palette = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.zeros_like(corr_reordered, dtype=bool)
    heatmap = sns.heatmap(
        corr_reordered,
        mask=mask,
        cmap=heatmap_palette,
        vmin=-1,
        vmax=1,
        ax=ax1e,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )

    for i in range(corr_reordered.shape[0]):
        for j in range(corr_reordered.shape[1]):
            value = corr_reordered.iloc[i, j]
            ax1e.text(
                j + 0.5,
                i + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if abs(value) >= 0.6 else "normal",
                color="black" if abs(value) < 0.6 else "white",
            )

    rect = Rectangle(
        (0, 0),
        len(selected_metrics),
        len(selected_metrics),
        fill=False,
        edgecolor="#222222",
        linewidth=2.5,
        linestyle="--",
    )
    ax1e.add_patch(rect)

    ax1e.set_xticks(np.arange(len(metric_order)) + 0.5)
    ax1e.set_yticks(np.arange(len(metric_order)) + 0.5)
    ax1e.set_xticklabels(metric_order, rotation=45, ha="right", fontsize=11)
    ax1e.set_yticklabels(metric_order, fontsize=11)
    ax1e.tick_params(length=0, labelsize=10)

    cbar_f = fig1.colorbar(heatmap.collections[0], ax=ax1e, fraction=0.046, pad=0.04)
    cbar_f.set_label("Correlation", fontweight="bold", fontsize=12)
    cbar_f.ax.tick_params(labelsize=10)

    ax1e.set_title(
        "E. Metric Selection Rationale for Final Ranking\n(Outlined cells: Selected metrics display complementary correlations)",
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
