#!/usr/bin/env python3
"""
Create comprehensive panel figures for the Results section of the paper.
This script generates publication-ready multi-panel figures that tell the story
of the pediatric sepsis challenge analysis.

Updated to use only:
- Preliminary phase metrics (optimal_threshold_results/)
- Validation phase metrics (optimal_threshold_results_validation/)
- Factor weights (data/processed/)
- Final leaderboard (data/raw/final_scores.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
import json
from pathlib import Path

# Set up paths
ROOT = Path(__file__).resolve().parents[2]
PRELIM_DIR = ROOT / "optimal_threshold_results"
VALIDATION_DIR = ROOT / "optimal_threshold_results_validation"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "figures" / "main_figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

print("📊 CREATING RESULTS SECTION PANEL FIGURES")
print("=" * 60)

# ============================================================================
# FIGURE 1: STUDY DESIGN AND PARTICIPANT FLOW
# ============================================================================
print("\n📈 Creating Figure 1: Study Design and Participant Flow...")

fig1 = plt.figure(figsize=(18, 14))
gs1 = GridSpec(3, 4, figure=fig1, hspace=0.32, wspace=0.35)

# Load data from new sources
final_scores = pd.read_csv(RAW_DIR / 'final_scores.csv')
prelim_sens = pd.read_csv(PRELIM_DIR / 'evaluation_metrics_sensitivity_method.csv')
prelim_f1 = pd.read_csv(PRELIM_DIR / 'evaluation_metrics_f1_method.csv')
prelim_youden = pd.read_csv(PRELIM_DIR / 'evaluation_metrics_youden_method.csv')
validation_sens = pd.read_csv(VALIDATION_DIR / 'evaluation_metrics_sensitivity_method.csv')

# Clean team names - extract team name from the full string (format: output_TeamName_https___...)
def extract_team_name(full_name):
    """Extract clean team name from output_TeamName_https___..._timestamp format"""
    if pd.isna(full_name):
        return full_name
    # Remove 'output_' prefix if present
    name = full_name.replace('output_', '')
    # Split by '_https' to get team name before the URL
    if '_https' in name:
        team_name = name.split('_https')[0]
    # Split by '_git' if no https
    elif '_git' in name:
        team_name = name.split('_git')[0]
    else:
        team_name = name
    
    # TeamAIMS and TeamAIMS2 are the same team
    if team_name == 'TeamAIMS2':
        team_name = 'TeamAIMS'
    
    return team_name

# Clean team names in all datasets
final_scores['Team_clean'] = final_scores['Team name'].str.encode('ascii', errors='ignore').str.decode('ascii')
validation_sens['Team_clean'] = validation_sens['Team'].apply(extract_team_name)
prelim_sens['Team_clean'] = prelim_sens['Team'].apply(extract_team_name) if 'Team' in prelim_sens.columns else prelim_sens['Team']

def remove_outliers_nb(data):
    """Remove extreme Net Benefit outliers using median ± 3*IQR."""
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

def compute_treat_all_nb(df):
    """Compute treat-all net benefit per submission."""
    required_cols = {'TP', 'FP', 'FN', 'TN', 'Optimal Threshold'}
    if not required_cols.issubset(df.columns):
        return np.array([])
    totals = df['TP'] + df['FP'] + df['FN'] + df['TN']
    totals = totals.replace(0, np.nan)
    prevalence = (df['TP'] + df['FN']) / totals
    pt = df['Optimal Threshold'].clip(1e-6, 1 - 1e-6)
    treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    treat_all = treat_all.replace([np.inf, -np.inf], np.nan).dropna()
    return treat_all.to_numpy()

# Panel A: Multi-phase Participation with Unique Teams as Proportion
ax1a = fig1.add_subplot(gs1[0, 0])
n_prelim = len(prelim_sens)
n_prelim_teams = prelim_sens['Team_clean'].nunique() if 'Team_clean' in prelim_sens.columns else n_prelim
n_valid = len(validation_sens)
n_valid_teams = validation_sens['Team_clean'].nunique()
n_final_subs = len(final_scores)
n_final_teams = final_scores['Team name'].nunique()

categories = ['Preliminary', 'Dummy/Validation', 'Final']
total_values = [n_prelim, n_valid, n_final_subs]
unique_values = [n_prelim_teams, n_valid_teams, n_final_teams]

print(f"  Panel A - Phase participation: Prelim={n_prelim} ({n_prelim_teams} teams), Validation={n_valid} ({n_valid_teams} teams), Final={n_final_subs} ({n_final_teams} teams)")

colors_bar = ['#3498DB', '#2ECC71', '#9B59B6']
bars = ax1a.bar(categories, total_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add numbers inside bars
for i, (bar, total, unique) in enumerate(zip(bars, total_values, unique_values)):
    height = bar.get_height()
    # Total number at top
    ax1a.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(total)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    # Unique teams inside bar (show for all phases where total != unique)
    if total != unique:
        ax1a.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{int(unique)} unique\nteams', ha='center', va='center', 
                 fontweight='bold', fontsize=10, color='white',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    else:
        # For preliminary phase, show it's all unique
        ax1a.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{int(unique)} unique\nteams', ha='center', va='center', 
                 fontweight='bold', fontsize=10, color='white',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

ax1a.set_ylabel('Number of Submissions', fontweight='bold')
ax1a.set_title('A. Multi-Phase Participation', fontweight='bold', fontsize=13)
ax1a.set_ylim(0, max(total_values) * 1.15)
ax1a.spines['top'].set_visible(False)
ax1a.spines['right'].set_visible(False)
ax1a.grid(True, alpha=0.3, axis='y')

# Panel B: Sensitivity Threshold Trade-off Across Optimization Methods
ax1b = fig1.add_subplot(gs1[0, 1:])

sens_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
methods_data = {
    'Sensitivity': prelim_sens,
    'Youden': prelim_youden,
    'F1 Opt': prelim_f1
}
method_colors = {
    'Sensitivity': '#1F77B4',
    'Youden': '#E74C3C',
    'F1 Opt': '#2ECC71'
}

fractions_by_method = {}
nb_by_method = {}
spec_by_method = {}

for method, df_method in methods_data.items():
    fractions = []
    nb_values = []
    spec_values = []
    total = len(df_method)
    for thresh in sens_thresholds:
        subset = df_method[df_method['Sensitivity'] >= thresh]
        fractions.append(len(subset) / total if total else 0)
        if len(subset) > 0:
            nb_values.append(subset['Net Benefit'].median())
            spec_values.append(subset['Specificity'].median())
        else:
            nb_values.append(np.nan)
            spec_values.append(np.nan)
    fractions_by_method[method] = np.array(fractions)
    nb_by_method[method] = np.array(nb_values, dtype=float)
    spec_by_method[method] = np.array(spec_values, dtype=float)

x_positions = np.arange(len(sens_thresholds))

fraction_lines = []
nb_lines = []
chosen_idx = sens_thresholds.index(0.80) if 0.80 in sens_thresholds else None

for method, fractions in fractions_by_method.items():
    color = method_colors[method]
    line, = ax1b.plot(x_positions, fractions, marker='o', linewidth=2.2, color=color,
                      label=f'{method}: fraction ≥ threshold')
    fraction_lines.append(line)
    ax1b.fill_between(x_positions, fractions, color=color, alpha=0.10)
    if chosen_idx is not None:
        ax1b.scatter(x_positions[chosen_idx], fractions[chosen_idx], color=color,
                     s=70, edgecolor='black', zorder=5)
        ax1b.annotate(f"{method}: {fractions[chosen_idx]*100:.0f}%",
                      xy=(x_positions[chosen_idx], fractions[chosen_idx]),
                      xytext=(5, 10), textcoords='offset points', fontsize=8,
                      color=color, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax1b.set_ylabel('Fraction of Models Meeting Sensitivity Threshold', fontweight='bold')
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
        linestyle='--',
        linewidth=1.8,
        marker='s',
        color=color,
        label=f'{method}: median NB (%)'
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

ax1b_secondary.set_ylabel('Median Net Benefit (%)', fontweight='bold')
ax1b_secondary.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))

if chosen_idx is not None:
    ax1b.axvline(x=chosen_idx, color='#34495E', linestyle='--', linewidth=2.5, alpha=0.7)
    ax1b.text(0.02, 0.92,
              "Vertical line marks ≥80% sensitivity requirement\n"
              "Compare coverage fractions across methods",
              transform=ax1b.transAxes, fontsize=8, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#34495E', alpha=0.85))

ax1b.set_xticks(x_positions)
ax1b.set_xticklabels([f'{int(t*100)}%' for t in sens_thresholds])
ax1b.set_xlabel('Sensitivity Threshold Requirement', fontweight='bold')
ax1b.set_title('B. Why Require ≥80% Sensitivity?\nCoverage vs. utility across threshold optimization methods',
              fontweight='bold', fontsize=13)
ax1b.grid(True, alpha=0.3, axis='y')
ax1b.spines['top'].set_visible(False)
ax1b.spines['right'].set_visible(False)
ax1b_secondary.spines['top'].set_visible(False)

handles = fraction_lines + nb_lines
labels = [h.get_label() for h in handles]
ax1b.legend(handles, labels, loc='upper right', fontsize=8, framealpha=0.9)

# Panel C: Net Benefit Comparison Across Threshold Methods (Preliminary Phase)
ax1c = fig1.add_subplot(gs1[1, 0:2])

methods_nb = {
    'Sensitivity': prelim_sens['Net Benefit'].dropna().values,
    'Youden': prelim_youden['Net Benefit'].dropna().values,
    'F1 Opt': prelim_f1['Net Benefit'].dropna().values
}

methods_nb_clean = {k: remove_outliers_nb(v) for k, v in methods_nb.items()}
print(
    "  Panel C - Net Benefit data points after filtering: "
    f"Sensitivity={len(methods_nb_clean['Sensitivity'])}, "
    f"Youden={len(methods_nb_clean['Youden'])}, "
    f"F1={len(methods_nb_clean['F1 Opt'])}"
)

treat_all_arrays = [compute_treat_all_nb(prelim_sens),
                    compute_treat_all_nb(prelim_youden),
                    compute_treat_all_nb(prelim_f1)]
treat_all_arrays = [arr for arr in treat_all_arrays if arr.size > 0]
treat_all_baseline = None
if treat_all_arrays:
    treat_all_combined = np.concatenate(treat_all_arrays)
    treat_all_filtered = remove_outliers_nb(treat_all_combined)
    if treat_all_filtered.size > 0:
        treat_all_baseline = np.median(treat_all_filtered)

methods_nb_percent = {k: v * 100.0 for k, v in methods_nb_clean.items()}
positions = [0, 1, 2]
parts = ax1c.violinplot(list(methods_nb_percent.values()), positions=positions,
                        showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_bar[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.2)

ax1c.set_xticks(positions)
ax1c.set_xticklabels(list(methods_nb_clean.keys()))
ax1c.set_ylabel('Net Benefit (%)', fontweight='bold')
ax1c.set_title('C. Net Benefit by Threshold Method\n(Preliminary Phase)', fontweight='bold', fontsize=13)
ax1c.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Treat-none baseline')
if treat_all_baseline is not None:
    treat_all_percent = treat_all_baseline * 100.0
    ax1c.axhline(y=treat_all_percent, color='#8E44AD', linestyle=':', linewidth=2,
                 label=f'Treat-all baseline (median NB = {treat_all_percent:.1f}%)')
ax1c.legend(loc='best', fontsize=8)
ax1c.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax1c.grid(True, alpha=0.3, axis='y')
ax1c.spines['top'].set_visible(False)
ax1c.spines['right'].set_visible(False)

# Panel D: Key Metrics Distribution (Preliminary Phase)
ax1d = fig1.add_subplot(gs1[1, 2:])  # Row 1, use remaining columns to tighten layout

# Calculate Precision (PPV) from TP and FP
prelim_sens['Precision'] = prelim_sens['TP'] / (prelim_sens['TP'] + prelim_sens['FP'])
prelim_youden['Precision'] = prelim_youden['TP'] / (prelim_youden['TP'] + prelim_youden['FP'])
prelim_f1['Precision'] = prelim_f1['TP'] / (prelim_f1['TP'] + prelim_f1['FP'])

# Calculate False Positive Rate (FPR = FP / (FP + TN))
prelim_sens['FPR'] = prelim_sens['FP'] / (prelim_sens['FP'] + prelim_sens['TN'])
prelim_youden['FPR'] = prelim_youden['FP'] / (prelim_youden['FP'] + prelim_youden['TN'])
prelim_f1['FPR'] = prelim_f1['FP'] / (prelim_f1['FP'] + prelim_f1['TN'])

# Select key metrics for visualization
metrics_to_plot = ['F1', 'Net Benefit', 'ECE', 'AUPRC', 'AUC', 'Sensitivity', 'Specificity', 'Precision']
metrics_display = ['F1', 'Net Benefit', 'ECE', 'AUPRC', 'AUC', 'Sensitivity', 'Specificity', 'Precision']

# Remove outliers using IQR method for each metric
def remove_outliers_iqr_series(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    mask = (series >= lower_bound) & (series <= upper_bound)
    filtered = series[mask]
    return filtered if len(filtered) > 0 else series

# Prepare data for violin plot
metrics_data = []
for metric in metrics_to_plot:
    if metric in prelim_sens.columns:
        clean_data = remove_outliers_iqr_series(prelim_sens[metric].dropna())
        metrics_data.append(clean_data.values)
    else:
        metrics_data.append([])

# Create violin plot
positions = list(range(len(metrics_display)))
parts = ax1d.violinplot(metrics_data, positions=positions, showmeans=True, showmedians=True, widths=0.7)

# Color the violins with a gradient
colors_gradient = plt.cm.Set3(np.linspace(0, 1, len(metrics_display)))
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_gradient[i])
    pc.set_alpha(0.8)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Customize appearance
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
    if partname in parts:
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1.5)

ax1d.set_xticks(positions)
ax1d.set_xticklabels(metrics_display, rotation=45, ha='right')
ax1d.set_ylabel('Metric Value', fontweight='bold')
ax1d.set_title('D. Key Performance Metrics Distribution\n(Preliminary Phase, Outliers Removed)', 
              fontweight='bold', fontsize=13)
ax1d.grid(True, alpha=0.3, axis='y')
ax1d.spines['top'].set_visible(False)
ax1d.spines['right'].set_visible(False)

# Add a horizontal line at 0 for reference
ax1d.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1d.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Panel E: Metric Selection Rationale - Why NB, F1, AUPRC, ECE for Final Ranking
ax1e = fig1.add_subplot(gs1[2, 0:3])  # Row 2, span all columns

# Calculate correlation matrix for all metrics
metrics_for_corr = ['AUC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Precision', 'Net Benefit', 'ECE']
corr_data = prelim_sens[metrics_for_corr].corr()

# Create a custom visualization showing metric independence and clinical relevance
# Split into two parts: correlation heatmap (left) and metric properties (right)

# Left side: Correlation heatmap with selected metrics highlighted
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Create subset focusing on selected vs non-selected metrics
selected_metrics = ['Net Benefit', 'F1', 'AUPRC', 'ECE', 'Sensitivity']
other_metrics = ['AUC', 'Specificity', 'Precision']

# Reorder for better visualization
metric_order = selected_metrics + other_metrics
corr_reordered = corr_data.loc[metric_order, metric_order]

# Create heatmap
im = ax1e.imshow(corr_reordered.abs(), cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

# Add correlation values as text
for i in range(len(metric_order)):
    for j in range(len(metric_order)):
        text = ax1e.text(j, i, f'{corr_reordered.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=9,
                        fontweight='bold' if abs(corr_reordered.iloc[i, j]) < 0.7 else 'normal')

# Highlight selected metrics with rectangles
for i in range(len(selected_metrics)):
    for j in range(len(selected_metrics)):
        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, 
                        edgecolor='blue', linewidth=3)
        ax1e.add_patch(rect)

# Set ticks and labels
ax1e.set_xticks(np.arange(len(metric_order)))
ax1e.set_yticks(np.arange(len(metric_order)))
ax1e.set_xticklabels(metric_order, rotation=45, ha='right', fontsize=10)
ax1e.set_yticklabels(metric_order, fontsize=10)

# Add colorbar
cbar_e = plt.colorbar(im, ax=ax1e, fraction=0.046, pad=0.04)
cbar_e.set_label('Absolute Correlation', fontweight='bold', fontsize=10)

# Title and annotations
ax1e.set_title('E. Metric Selection Rationale for Final Ranking\n(Blue box: Selected metrics show complementary information with lower inter-correlation)', 
              fontweight='bold', fontsize=13, pad=20)

# Remove the text box - information will go in figure caption instead

plt.savefig(FIGURES_DIR / 'results_figure1_overview.png', bbox_inches='tight', dpi=300)
print("   ✓ Saved: results_figure1_overview.png")

# ============================================================================
# FIGURE 2: FACTOR ANALYSIS METHODOLOGY
# ============================================================================
print("\n📈 Creating Figure 2: Factor Analysis Methodology...")

fig2 = plt.figure(figsize=(16, 12))
gs2 = GridSpec(3, 3, figure=fig2, hspace=0.4, wspace=0.35)

# Load factor analysis data from new locations
try:
    import json
    with open(PROCESSED_DIR / 'factor_loadings.json', 'r') as f:
        factor_loadings_dict = json.load(f)
    # Convert to DataFrame format
    factor_loadings = pd.DataFrame(factor_loadings_dict)
except:
    # Fallback: try CSV files if they exist
    factor_loadings = None

factor_weights = pd.read_csv(PROCESSED_DIR / 'factor_weights.csv')

# Panel A: Factor Weights (Primary Display)
ax2a = fig2.add_subplot(gs2[0, :2])
colors_weight = ['#2ECC71' if w > 0 else '#E74C3C' for w in factor_weights['weight']]
bars = ax2a.barh(factor_weights['feature'], factor_weights['weight'],
                color=colors_weight, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2a.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2a.set_xlabel('Weight Value', fontweight='bold')
ax2a.set_title('A. Multi-Criteria Factor Weights for Composite Scoring', fontweight='bold', fontsize=14)
ax2a.grid(True, alpha=0.3, axis='x')
ax2a.spines['top'].set_visible(False)
ax2a.spines['right'].set_visible(False)

# Add value labels
for i, (feature, weight) in enumerate(zip(factor_weights['feature'], factor_weights['weight'])):
    ax2a.text(weight + 0.03 if weight >= 0 else weight - 0.03, i,
             f'{weight:.3f}', va='center',
             ha='left' if weight >= 0 else 'right', fontsize=10, fontweight='bold')

# Panel B: Weight Contribution Pie Chart
ax2b = fig2.add_subplot(gs2[0, 2])
weights_abs = np.abs(factor_weights['weight'])
contribution = weights_abs / weights_abs.sum() * 100
colors_pie = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
wedges, texts, autotexts = ax2b.pie(contribution, labels=factor_weights['feature'],
                                    autopct='%1.1f%%', startangle=90,
                                    colors=colors_pie[:len(contribution)],
                                    textprops={'fontsize': 9})
ax2b.set_title('B. Relative Weight Contribution', fontweight='bold', fontsize=13)

# Panel C: Preliminary Phase Metric Distribution
ax2c = fig2.add_subplot(gs2[1, 0])
# Select key metrics from preliminary phase
key_metrics = ['F1', 'AUPRC', 'ECE']
metric_data_prelim = []
metric_labels_prelim = []
for metric in key_metrics:
    if metric in prelim_sens.columns:
        metric_data_prelim.append(prelim_sens[metric].dropna())
        metric_labels_prelim.append(metric)

if metric_data_prelim:
    parts = ax2c.violinplot(metric_data_prelim, positions=range(len(metric_labels_prelim)),
                            showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_pie[i])
        pc.set_alpha(0.7)
    ax2c.set_xticks(range(len(metric_labels_prelim)))
    ax2c.set_xticklabels(metric_labels_prelim)
    ax2c.set_ylabel('Metric Value', fontweight='bold')
    ax2c.set_title('C. Key Metrics Distribution\n(Preliminary Phase)', fontweight='bold', fontsize=13)
    ax2c.grid(True, alpha=0.3, axis='y')
    ax2c.spines['top'].set_visible(False)
    ax2c.spines['right'].set_visible(False)

# Panel D: Metric Correlation Matrix (Preliminary Phase)
ax2d = fig2.add_subplot(gs2[1, 1])
corr_metrics_prelim = ['AUC', 'AUPRC', 'F1', 'ECE', 'Net Benefit']
available_corr = [m for m in corr_metrics_prelim if m in prelim_sens.columns]
if len(available_corr) > 2:
    corr_matrix_prelim = prelim_sens[available_corr].corr()
    sns.heatmap(corr_matrix_prelim, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax2d, cbar_kws={'label': 'Correlation'},
                square=True, linewidths=1, linecolor='white')
    ax2d.set_title('D. Metric Correlations\n(Preliminary Phase)', fontweight='bold', fontsize=13)

# Panel E: Weighted Score Impact on Final Rankings
ax2e = fig2.add_subplot(gs2[1, 2])
if 'Weighted Score' in final_scores.columns:
    weighted_scores = final_scores['Weighted Score'].dropna()
    # Remove extreme outliers
    q1, q99 = weighted_scores.quantile([0.01, 0.99])
    weighted_clean = weighted_scores[(weighted_scores >= q1) & (weighted_scores <= q99)]
    
    ax2e.hist(weighted_clean, bins=25, color='#3498DB',
             alpha=0.7, edgecolor='black', linewidth=1)
    ax2e.axvline(weighted_clean.median(), color='red',
                linestyle='--', linewidth=2, label=f'Median: {weighted_clean.median():.2f}')
    ax2e.set_xlabel('Weighted Score', fontweight='bold')
    ax2e.set_ylabel('Number of Teams', fontweight='bold')
    ax2e.set_title('E. Final Weighted Score Distribution', fontweight='bold', fontsize=13)
    ax2e.legend()
    ax2e.grid(True, alpha=0.3, axis='y')
    ax2e.spines['top'].set_visible(False)
    ax2e.spines['right'].set_visible(False)

# Panel F: Component Contribution Analysis
ax2f = fig2.add_subplot(gs2[2, 0])
# Show how each metric contributes based on its weight
if all(m in final_scores.columns for m in ['F1', 'AUPRC', 'Net Benefit']):
    # Normalize metrics for comparison
    metrics_for_contrib = ['F1', 'AUPRC', 'Net Benefit']
    top_5_teams = final_scores.nlargest(5, 'Weighted Score') if 'Weighted Score' in final_scores.columns else final_scores.head(5)
    
    normalized_contrib = []
    for metric in metrics_for_contrib:
        if metric in top_5_teams.columns:
            values = top_5_teams[metric].values
            # Get corresponding weight
            metric_weight = factor_weights[factor_weights['feature'] == metric]['weight'].values
            if len(metric_weight) > 0:
                weighted_values = values * metric_weight[0]
                normalized_contrib.append(weighted_values)
    
    if normalized_contrib:
        x_pos = np.arange(5)
        bottom = np.zeros(5)
        colors_comp = ['#2ECC71', '#3498DB', '#F39C12']
        
        for i, (contrib, label) in enumerate(zip(normalized_contrib, metrics_for_contrib)):
            ax2f.bar(x_pos, contrib, bottom=bottom, label=label,
                    color=colors_comp[i % len(colors_comp)], alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            bottom += contrib
        
        ax2f.set_xticks(x_pos)
        ax2f.set_xticklabels([f'Rank {i+1}' for i in range(5)])
        ax2f.set_ylabel('Weighted Contribution', fontweight='bold')
        ax2f.set_title('F. Metric Contributions\n(Top 5 Teams)', fontweight='bold', fontsize=13)
        ax2f.legend(loc='best', fontsize=8)
        ax2f.grid(True, alpha=0.3, axis='y')
        ax2f.spines['top'].set_visible(False)
        ax2f.spines['right'].set_visible(False)

# Panel G: Calibration Error (ECE) Analysis
ax2g = fig2.add_subplot(gs2[2, 1])
if 'ECE' in prelim_sens.columns:
    ece_prelim = prelim_sens['ECE'].dropna()
    # Remove extreme outliers
    q1, q99 = ece_prelim.quantile([0.01, 0.99])
    ece_clean = ece_prelim[(ece_prelim >= q1) & (ece_prelim <= q99)]
    
    ax2g.hist(ece_clean, bins=20, color='#E74C3C',
             alpha=0.7, edgecolor='black', linewidth=1)
    ax2g.axvline(ece_clean.median(), color='darkred',
                linestyle='--', linewidth=2, label=f'Median: {ece_clean.median():.3f}')
    ax2g.set_xlabel('Expected Calibration Error', fontweight='bold')
    ax2g.set_ylabel('Number of Teams', fontweight='bold')
    ax2g.set_title('G. Calibration Quality\n(Preliminary Phase)', fontweight='bold', fontsize=13)
    ax2g.legend()
    ax2g.grid(True, alpha=0.3, axis='y')
    ax2g.spines['top'].set_visible(False)
    ax2g.spines['right'].set_visible(False)

# Panel H: Performance Stability (Preliminary vs Validation)
ax2h = fig2.add_subplot(gs2[2, 2])
# Show stability across phases
stability_metrics = ['AUC', 'F1', 'AUPRC']
prelim_medians = [prelim_sens[m].median() for m in stability_metrics if m in prelim_sens.columns]
valid_medians = [validation_sens[m].median() for m in stability_metrics if m in validation_sens.columns]

x_pos = np.arange(len(stability_metrics))
width = 0.35

bars1 = ax2h.bar(x_pos - width/2, prelim_medians, width, label='Preliminary',
                color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax2h.bar(x_pos + width/2, valid_medians, width, label='Validation',
                color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1)

ax2h.set_xticks(x_pos)
ax2h.set_xticklabels(stability_metrics)
ax2h.set_ylabel('Median Value', fontweight='bold')
ax2h.set_title('H. Performance Stability\nAcross Phases', fontweight='bold', fontsize=13)
ax2h.legend(loc='best', fontsize=9)
ax2h.grid(True, alpha=0.3, axis='y')
ax2h.spines['top'].set_visible(False)
ax2h.spines['right'].set_visible(False)

plt.savefig(FIGURES_DIR / 'results_figure2_factor_analysis.png', bbox_inches='tight', dpi=300)
print("   ✓ Saved: results_figure2_factor_analysis.png")

# ============================================================================
# FIGURE 3: PERFORMANCE RANKINGS AND COMPARISONS
# ============================================================================
print("\n📈 Creating Figure 3: Performance Rankings...")

fig3 = plt.figure(figsize=(16, 10))
gs3 = GridSpec(2, 2, figure=fig3, hspace=0.35, wspace=0.3)

# Panel A: Top 10 Teams from Final Leaderboard
ax3a = fig3.add_subplot(gs3[0, :])
if 'Weighted Score' in final_scores.columns:
    top_10 = final_scores.nlargest(10, 'Weighted Score').copy()
    top_10['Rank'] = range(1, 11)
    
    # Use cleaned team names
    team_labels = top_10['Team_clean'].str[:40]  # Truncate long names
    
    colors_rank = plt.cm.viridis(np.linspace(0, 1, 10))
    bars = ax3a.barh(range(10), top_10['Weighted Score'].values, 
                    color=colors_rank, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3a.set_yticks(range(10))
    ax3a.set_yticklabels([f'{i+1}. {label}' for i, label in enumerate(team_labels)], 
                         fontsize=10)
    ax3a.set_xlabel('Composite Weighted Score', fontweight='bold')
    ax3a.set_title('A. Top 10 Teams by Final Composite Score', fontweight='bold', fontsize=14)
    ax3a.invert_yaxis()
    ax3a.grid(True, alpha=0.3, axis='x')
    ax3a.spines['top'].set_visible(False)
    ax3a.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_10['Weighted Score'].values)):
        width = bar.get_width()
        ax3a.text(width + 0.02, i, f'{score:.2f}',
                 va='center', ha='left', fontsize=10, fontweight='bold')

# Panel B: Performance Component Breakdown (Top 5)
ax3b = fig3.add_subplot(gs3[1, 0])
if 'Weighted Score' in final_scores.columns:
    top_5 = final_scores.nlargest(5, 'Weighted Score')
    
    # Get key metric components
    component_metrics = ['F1', 'AUPRC', 'Net Benefit']
    available_components = [m for m in component_metrics if m in top_5.columns]
    
    if available_components:
        x_pos = np.arange(5)
        bottom = np.zeros(5)
        colors_comp = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        
        for i, metric in enumerate(available_components):
            # Normalize values to positive for stacking
            values = np.abs(top_5[metric].values)
            ax3b.bar(x_pos, values, bottom=bottom, label=metric,
                    color=colors_comp[i % len(colors_comp)], alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            bottom += values
        
        ax3b.set_xticks(x_pos)
        ax3b.set_xticklabels([f'Rank {i+1}' for i in range(5)])
        ax3b.set_ylabel('Metric Value (Stacked)', fontweight='bold')
        ax3b.set_title('B. Performance Component Breakdown\n(Top 5 Teams)', 
                      fontweight='bold', fontsize=13)
        ax3b.legend(loc='upper right', fontsize=9)
        ax3b.grid(True, alpha=0.3, axis='y')
        ax3b.spines['top'].set_visible(False)
        ax3b.spines['right'].set_visible(False)

# Panel C: Calibration Quality Distribution (Final Phase)
ax3c = fig3.add_subplot(gs3[1, 1])
if 'ECE' in final_scores.columns:
    ece_values = final_scores['ECE'].dropna()
    
    # Create bins for ECE categories
    bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
    labels = ['Excellent\n(<0.05)', 'Good\n(0.05-0.10)', 
             'Fair\n(0.10-0.15)', 'Poor\n(0.15-0.20)', 
             'Very Poor\n(>0.20)']
    ece_categories = pd.cut(ece_values, bins=bins, labels=labels)
    category_counts = ece_categories.value_counts().sort_index()
    
    colors_cal = ['#2ECC71', '#3498DB', '#F39C12', '#E67E22', '#E74C3C']
    bars = ax3c.bar(range(len(category_counts)), category_counts.values,
                   color=colors_cal, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax3c.set_xticks(range(len(category_counts)))
    ax3c.set_xticklabels(category_counts.index, fontsize=9)
    ax3c.set_ylabel('Number of Teams', fontweight='bold')
    ax3c.set_title('C. Model Calibration Quality\n(Expected Calibration Error)', 
                  fontweight='bold', fontsize=13)
    ax3c.grid(True, alpha=0.3, axis='y')
    ax3c.spines['top'].set_visible(False)
    ax3c.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, count in zip(bars, category_counts.values):
        height = bar.get_height()
        ax3c.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(count)}', ha='center', va='bottom', 
                 fontsize=11, fontweight='bold')

plt.savefig(FIGURES_DIR / 'results_figure3_rankings.png', bbox_inches='tight', dpi=300)
print("   ✓ Saved: results_figure3_rankings.png")

print("\n" + "=" * 60)
print("✅ ALL RESULTS FIGURES CREATED SUCCESSFULLY!")
print("\nGenerated files in figures/main_figures/:")
print("  • results_figure1_overview.png - Multi-phase study overview and metrics")
print("  • results_figure2_factor_analysis.png - Factor analysis and weighting methodology")
print("  • results_figure3_rankings.png - Final phase performance rankings")
print("\nData sources used:")
print("  ✓ Preliminary phase: optimal_threshold_results/")
print("  ✓ Validation phase: optimal_threshold_results_validation/")
print("  ✓ Factor weights: data/processed/factor_weights.csv")
print("  ✓ Final leaderboard: data/raw/final_scores.csv")
print("  ✗ No final-phase probability data used (per specification)")

