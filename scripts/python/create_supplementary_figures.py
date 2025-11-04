#!/usr/bin/env python3
"""
Create supplementary figures for the Results section.
Includes detailed analyses and multi-phase comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
import json
import re

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("📊 CREATING SUPPLEMENTARY RESULTS FIGURES")
print("=" * 60)

# ============================================================================
# SUPPLEMENTARY FIGURE 1: Detailed Statistical Analyses
# ============================================================================
print("\n📈 Creating Supplementary Figure 1: Statistical Analyses...")

# Resolve repository root for robust file paths
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / 'data' / 'processed'
RAW_DIR = ROOT / 'data' / 'raw'
FIGURES_SUPP_DIR = ROOT / 'figures' / 'supplementary'
FIGURES_SUPP_DIR.mkdir(parents=True, exist_ok=True)

fig_s1 = plt.figure(figsize=(16, 12))
gs_s1 = GridSpec(3, 3, figure=fig_s1, hspace=0.4, wspace=0.35)

# Load final-phase leaderboard scores from data/raw
final_scores = pd.read_csv(RAW_DIR / 'final_scores.csv')


def canonicalize_team(name: str) -> str:
    """Normalize team strings by removing emoji/punctuation and lowercasing."""
    if not isinstance(name, str):
        return ""
    ascii_name = name.encode('ascii', 'ignore').decode()
    ascii_name = ascii_name.lower()
    return re.sub(r'[^a-z0-9]', '', ascii_name)


# Standardize column names and compute derived metrics
final_scores = final_scores.rename(columns=lambda col: col.strip())
final_scores = final_scores.rename(
    columns={
        'Team name': 'team_name',
        'Affiliation': 'affiliation',
        'Weighted Score': 'weighted_score',
        'Scaled Weighted Score': 'scaled_weighted_score',
        'Net Benefit': 'net_benefit',
        'Submission time': 'submission_time'
    }
)

numeric_cols = [
    'Rank', 'weighted_score', 'scaled_weighted_score', 'AUPRC', 'net_benefit',
    'ECE', 'F1', 'TP', 'FP', 'FN', 'TN', 'AUC'
]
for col in numeric_cols:
    if col in final_scores.columns:
        final_scores[col] = pd.to_numeric(final_scores[col], errors='coerce')

final_scores['raw_score'] = final_scores['weighted_score']
final_scores['final_score'] = final_scores['scaled_weighted_score']

sens_denom = final_scores['TP'] + final_scores['FN']
spec_denom = final_scores['TN'] + final_scores['FP']
final_scores['sensitivity'] = np.where(sens_denom > 0, final_scores['TP'] / sens_denom, np.nan)
final_scores['specificity'] = np.where(spec_denom > 0, final_scores['TN'] / spec_denom, np.nan)
final_scores['team_key'] = final_scores['team_name'].apply(canonicalize_team)

# Parse detailed evaluation metrics (inference time, parsimony) from raw exports
eval_records = []
eval_path = RAW_DIR / 'Eva_csv - Sheet1.csv'
if eval_path.exists():
    eval_raw = pd.read_csv(eval_path)
    for _, row in eval_raw.iterrows():
        blob = row.get('Evaluated_1')
        if not isinstance(blob, str) or not blob.strip().startswith('{'):
            continue
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        score = payload.get('score') or {}
        record = {
            'team_name_eval': row.get('Team name', ''),
            'submission_time_eval': row.get('Submission time'),
        }
        for key, value in score.items():
            key_norm = key.lower().replace(' ', '_')
            record[key_norm] = value
        eval_records.append(record)

    if eval_records:
        eval_df = pd.DataFrame(eval_records)
        eval_df['team_key'] = eval_df['team_name_eval'].apply(canonicalize_team)
        numeric_eval_cols = [
            'weighted_score',
            'scaled_weighted_score',
            'inference_time',
            'parsimony_score'
        ]
        for col in numeric_eval_cols:
            if col in eval_df.columns:
                eval_df[col] = pd.to_numeric(eval_df[col], errors='coerce')

        if not eval_df.empty:
            eval_df = eval_df.sort_values(
                by=['team_key', 'scaled_weighted_score', 'weighted_score'],
                ascending=[True, False, False]
            )
            best_eval = eval_df.groupby('team_key', as_index=False).first()
            final_scores = final_scores.merge(
                best_eval[['team_key', 'inference_time', 'parsimony_score']],
                on='team_key',
                how='left'
            )

final_scores['inference_time'] = pd.to_numeric(final_scores.get('inference_time'), errors='coerce')
final_scores['parsimony_score'] = pd.to_numeric(final_scores.get('parsimony_score'), errors='coerce')

# Panel A: Residual Analysis
ax_s1a = fig_s1.add_subplot(gs_s1[0, 0])
if 'raw_score' in final_scores.columns and 'final_score' in final_scores.columns:
    # Remove extreme outliers for visualization
    mask = (final_scores['raw_score'] > -100) & (final_scores['final_score'] > -100)
    clean_data = final_scores[mask]
    
    residuals = clean_data['final_score'] - clean_data['raw_score']
    ax_s1a.scatter(clean_data['raw_score'], residuals, alpha=0.6, s=80,
                  edgecolors='black', linewidth=0.5, c='#3498DB')
    ax_s1a.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax_s1a.set_xlabel('Raw Score', fontweight='bold')
    ax_s1a.set_ylabel('Residual (Final - Raw)', fontweight='bold')
    ax_s1a.set_title('A. Residual Analysis\n(Z-score Transformation)', 
                     fontweight='bold', fontsize=12)
    ax_s1a.grid(True, alpha=0.3)
    ax_s1a.spines['top'].set_visible(False)
    ax_s1a.spines['right'].set_visible(False)

# Panel B: QQ Plot for Normality
ax_s1b = fig_s1.add_subplot(gs_s1[0, 1])
if 'final_score' in final_scores.columns:
    clean_scores = final_scores[final_scores['final_score'] > -100]['final_score']
    stats.probplot(clean_scores, dist="norm", plot=ax_s1b)
    ax_s1b.set_title('B. Q-Q Plot (Normality Check)', fontweight='bold', fontsize=12)
    ax_s1b.grid(True, alpha=0.3)
    ax_s1b.spines['top'].set_visible(False)
    ax_s1b.spines['right'].set_visible(False)

# Panel C: Performance Metric Pairwise Relationships
ax_s1c = fig_s1.add_subplot(gs_s1[0, 2])
if 'AUC' in final_scores.columns and 'AUPRC' in final_scores.columns:
    scatter = ax_s1c.scatter(final_scores['AUC'], final_scores['AUPRC'],
                            c=final_scores['final_score'], cmap='RdYlGn',
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    mask = final_scores['AUC'].notna() & final_scores['AUPRC'].notna()
    if mask.sum() > 2:
        z = np.polyfit(final_scores.loc[mask, 'AUC'], 
                      final_scores.loc[mask, 'AUPRC'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(final_scores['AUC'].min(), 
                            final_scores['AUC'].max(), 100)
        ax_s1c.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8,
                   label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax_s1c.set_xlabel('AUC', fontweight='bold')
    ax_s1c.set_ylabel('AUPRC', fontweight='bold')
    ax_s1c.set_title('C. AUC vs AUPRC Relationship', fontweight='bold', fontsize=12)
    ax_s1c.legend()
    cbar = plt.colorbar(scatter, ax=ax_s1c)
    cbar.set_label('Final Score', fontweight='bold')
    ax_s1c.grid(True, alpha=0.3)
    ax_s1c.spines['top'].set_visible(False)
    ax_s1c.spines['right'].set_visible(False)

# Panel D: Rank Stability Analysis
ax_s1d = fig_s1.add_subplot(gs_s1[1, 0])
if 'AUC' in final_scores.columns and 'final_score' in final_scores.columns:
    # Rank by different metrics
    rank_auc = final_scores['AUC'].rank(ascending=False)
    rank_final = final_scores['final_score'].rank(ascending=False)
    rank_diff = rank_auc - rank_final
    
    ax_s1d.scatter(rank_final, rank_diff, alpha=0.6, s=100,
                  c=np.abs(rank_diff), cmap='YlOrRd',
                  edgecolors='black', linewidth=0.5)
    ax_s1d.axhline(y=0, color='blue', linestyle='--', linewidth=2)
    ax_s1d.set_xlabel('Rank by Final Score', fontweight='bold')
    ax_s1d.set_ylabel('Rank Difference (AUC - Final)', fontweight='bold')
    ax_s1d.set_title('D. Rank Stability Analysis\n(AUC vs Composite)', 
                     fontweight='bold', fontsize=12)
    ax_s1d.grid(True, alpha=0.3)
    ax_s1d.spines['top'].set_visible(False)
    ax_s1d.spines['right'].set_visible(False)

# Panel E: Calibration Belt Analysis
ax_s1e = fig_s1.add_subplot(gs_s1[1, 1])
if 'ECE' in final_scores.columns:
    ece_sorted = final_scores.sort_values('ECE')
    ece_clean = ece_sorted[ece_sorted['ECE'] < 0.5]['ECE']  # Remove extreme outliers
    
    ax_s1e.plot(range(len(ece_clean)), ece_clean.values, 'o-', 
               linewidth=2, markersize=6, alpha=0.7, color='#E74C3C')
    ax_s1e.axhline(y=0.05, color='green', linestyle='--', 
                  linewidth=2, label='Excellent (<0.05)', alpha=0.7)
    ax_s1e.axhline(y=0.10, color='orange', linestyle='--', 
                  linewidth=2, label='Good (<0.10)', alpha=0.7)
    ax_s1e.set_xlabel('Team Rank by ECE', fontweight='bold')
    ax_s1e.set_ylabel('Expected Calibration Error', fontweight='bold')
    ax_s1e.set_title('E. Calibration Error Landscape', 
                     fontweight='bold', fontsize=12)
    ax_s1e.legend(fontsize=8)
    ax_s1e.grid(True, alpha=0.3)
    ax_s1e.spines['top'].set_visible(False)
    ax_s1e.spines['right'].set_visible(False)

# Panel F: Net Benefit vs F1 Trade-off
ax_s1f = fig_s1.add_subplot(gs_s1[1, 2])
if 'net_benefit' in final_scores.columns and 'F1' in final_scores.columns:
    mask = final_scores['net_benefit'].between(-1, 0.1)
    plot_data = final_scores[mask]

    scatter = ax_s1f.scatter(plot_data['F1'], plot_data['net_benefit'],
                            c=plot_data['ECE'], cmap='viridis_r',
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax_s1f.set_xlabel('F1 Score', fontweight='bold')
    ax_s1f.set_ylabel('Net Benefit', fontweight='bold')
    ax_s1f.set_title('F. Net Benefit vs F1 Trade-off\n(colored by ECE)', 
                     fontweight='bold', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax_s1f)
    cbar.set_label('ECE', fontweight='bold')
    ax_s1f.grid(True, alpha=0.3)
    ax_s1f.spines['top'].set_visible(False)
    ax_s1f.spines['right'].set_visible(False)

# Panel G: Computational Efficiency vs Performance
ax_s1g = fig_s1.add_subplot(gs_s1[2, 0])
if 'inference_time' in final_scores.columns and 'AUC' in final_scores.columns:
    scatter = ax_s1g.scatter(np.log10(final_scores['inference_time'] + 1e-6),
                            final_scores['AUC'],
                            c=final_scores['final_score'], cmap='RdYlGn',
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax_s1g.set_xlabel('Log10(Inference Time + 1e-6)', fontweight='bold')
    ax_s1g.set_ylabel('AUC', fontweight='bold')
    ax_s1g.set_title('G. Efficiency-Performance Trade-off', 
                     fontweight='bold', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax_s1g)
    cbar.set_label('Final Score', fontweight='bold')
    ax_s1g.grid(True, alpha=0.3)
    ax_s1g.spines['top'].set_visible(False)
    ax_s1g.spines['right'].set_visible(False)

# Panel H: Parsimony Impact
ax_s1h = fig_s1.add_subplot(gs_s1[2, 1])
if 'parsimony_score' in final_scores.columns:
    parsimony_fraction = final_scores['parsimony_score'].clip(lower=0.0, upper=1.0)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    intervals = pd.cut(parsimony_fraction, bins=bins, include_lowest=True, right=False)
    parsimony_groups = final_scores.groupby(intervals, observed=False)

    def _format_interval(interval: pd.Interval) -> str:
        if pd.isna(interval):
            return "Missing"
        left_bracket = "[" if interval.closed_left else "("
        right_bracket = ")"
        if np.isclose(interval.right, bins[-1]):
            right_bracket = "]"
        left_val = max(interval.left, 0.0)
        right_val = min(interval.right, 1.0)
        return f"{left_bracket}{left_val:.1f}, {right_val:.1f}{right_bracket}"
    
    group_data = []
    group_labels = []
    for name, group in parsimony_groups:
        if len(group) > 0 and 'final_score' in group.columns:
            clean = group[group['final_score'] > -100]['final_score']
            if len(clean) > 0:
                group_data.append(clean.values)
                group_labels.append(_format_interval(name))
    
    if group_data:
        colors = sns.color_palette('Reds', len(group_data))
        bp = ax_s1h.boxplot(group_data, tick_labels=group_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_s1h.set_xlabel('Fraction of Variables Used (Parsimony Score)', fontweight='bold')
        ax_s1h.set_ylabel('Final Score', fontweight='bold')
        ax_s1h.set_title('H. Final Score vs. Fraction of Variables Used', 
                         fontweight='bold', fontsize=12)
        ax_s1h.grid(True, alpha=0.3, axis='y')
        ax_s1h.spines['top'].set_visible(False)
        ax_s1h.spines['right'].set_visible(False)

# Panel I: Score Distribution by Performance Tier
ax_s1i = fig_s1.add_subplot(gs_s1[2, 2])
if 'final_score' in final_scores.columns:
    clean_scores = final_scores[final_scores['final_score'] > -100]

    score_bins = [-np.inf, -0.2, 0.0, 0.1, np.inf]
    tiers = pd.cut(clean_scores['final_score'], bins=score_bins, include_lowest=True, right=False)

    def _format_score_interval(interval: pd.Interval) -> str:
        if pd.isna(interval):
            return "Missing"
        left_bracket = "[" if interval.closed_left else "("
        right_bracket = ")"
        left_val = interval.left
        right_val = interval.right
        if np.isneginf(left_val):
            left_str = "-∞"
        else:
            left_str = f"{left_val:.2f}"
        if np.isposinf(right_val):
            right_str = "∞"
            right_bracket = ")"
        else:
            right_str = f"{right_val:.2f}"
        return f"{left_bracket}{left_str}, {right_str}{right_bracket}"

    tier_counts = tiers.value_counts().sort_index()
    tier_labels = [_format_score_interval(interval) for interval in tier_counts.index]
    
    colors_tier = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']
    bars = ax_s1i.bar(range(len(tier_counts)), tier_counts.values,
                     color=colors_tier[:len(tier_counts)], alpha=0.7,
                     edgecolor='black', linewidth=1.5)
    
    ax_s1i.set_xticks(range(len(tier_counts)))
    ax_s1i.set_xticklabels(tier_labels, rotation=25, ha='right')
    ax_s1i.set_ylabel('Number of Teams', fontweight='bold')
    ax_s1i.set_ylim(0, tier_counts.max() * 1.25)
    ax_s1i.set_title('I. Performance Tier Distribution', 
                     fontweight='bold', fontsize=12)
    ax_s1i.grid(True, alpha=0.3, axis='y')
    ax_s1i.spines['top'].set_visible(False)
    ax_s1i.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, count in zip(bars, tier_counts.values):
        height = bar.get_height()
        ax_s1i.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({100*count/len(clean_scores):.1f}%)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

figure1_path = FIGURES_SUPP_DIR / 'results_supplementary_figure1.png'
plt.savefig(figure1_path, bbox_inches='tight', dpi=300)
print(f"   ✓ Saved: {figure1_path.relative_to(ROOT)}")

# ============================================================================
# SUPPLEMENTARY TABLE: Detailed Correlations
# ============================================================================
print("\n📊 Creating Supplementary Correlation Table...")

metrics_for_corr = ['AUC', 'AUPRC', 'F1', 'sensitivity', 'specificity',
                    'net_benefit', 'ECE', 'inference_time', 'final_score']
available_metrics = [m for m in metrics_for_corr if m in final_scores.columns]

if len(available_metrics) > 1:
    # Calculate Spearman correlations
    corr_matrix = final_scores[available_metrics].corr(method='spearman')
    
    # Save to CSV
    corr_path = PROCESSED_DIR / 'correlation_matrix_spearman_detailed.csv'
    corr_matrix.to_csv(corr_path)
    print(f"   ✓ Saved: {corr_path.relative_to(ROOT)}")
    
    # Create visualization
    fig_corr = plt.figure(figsize=(12, 10))
    ax_corr = fig_corr.add_subplot(111)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
               center=0, square=True, linewidths=1, linecolor='white',
               cbar_kws={'label': 'Spearman ρ', 'shrink': 0.8},
               ax=ax_corr, vmin=-1, vmax=1)
    
    ax_corr.set_title('Supplementary: Comprehensive Metric Correlation Matrix\n(Spearman)',
                     fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    corr_fig_path = FIGURES_SUPP_DIR / 'results_supplementary_correlations.png'
    plt.savefig(corr_fig_path, bbox_inches='tight', dpi=300)
    print(f"   ✓ Saved: {corr_fig_path.relative_to(ROOT)}")

print("\n" + "=" * 60)
print("✅ SUPPLEMENTARY FIGURES CREATED SUCCESSFULLY!")

# =============================================================================
# SUPPLEMENTARY FIGURE 2: Panels moved from Figure 2 (F and H)
# =============================================================================
print("\n📈 Creating Supplementary Figure 2: Additional Development Diagnostics...")

try:
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT / "data" / "processed"

    dummy_phase = pd.read_csv(PROCESSED_DIR / "Dummy_phase_scores_new_full.csv")
    factor_weights = pd.read_csv(PROCESSED_DIR / "factor_weights.csv")
    scale_params_df = pd.read_csv(PROCESSED_DIR / "scale_params.csv")
    scale_lookup = {row["feature"]: (row["center"], row["scale"]) for _, row in scale_params_df.iterrows()}

    fig_s2 = plt.figure(figsize=(14, 6))
    gs_s2 = GridSpec(1, 2, figure=fig_s2, wspace=0.4)

    # S2-A: Model Complexity vs Final Score (moved from Fig 2F)
    ax_s2a = fig_s2.add_subplot(gs_s2[0, 0])
    cols = ["parsimony_score", "final_score", "inference_time"]
    if all(col in dummy_phase.columns for col in cols):
        clean = dummy_phase["final_score"].dropna()
        if not clean.empty:
            q1, q3 = clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (dummy_phase["final_score"] >= q1 - 1.5 * iqr) & (dummy_phase["final_score"] <= q3 + 1.5 * iqr)
            filtered = dummy_phase[mask].copy()
        else:
            filtered = dummy_phase.copy()
        parsimony_complement = np.clip(1 - filtered["parsimony_score"], 0, 1)
        inference_ms = filtered["inference_time"] * 1000
        sc = ax_s2a.scatter(parsimony_complement, filtered["final_score"], c=inference_ms, cmap="viridis",
                            alpha=0.8, s=60, edgecolors='black', linewidth=0.4)
        ax_s2a.set_xlabel("Model Simplicity (1 − Parsimony Score)", fontweight='bold')
        ax_s2a.set_ylabel("Final Score (Dummy Phase)", fontweight='bold')
        ax_s2a.set_title("A. Model Complexity vs Final Score (Dummy Phase)", fontweight='bold')
        ax_s2a.grid(True, alpha=0.3)
        ax_s2a.spines['top'].set_visible(False)
        ax_s2a.spines['right'].set_visible(False)
        cbar = fig_s2.colorbar(sc, ax=ax_s2a, fraction=0.046, pad=0.04)
        cbar.set_label("Inference Time (ms)", fontweight='bold')

    # S2-B: Weighted Metric Contributions (moved from Fig 2H)
    ax_s2b = fig_s2.add_subplot(gs_s2[0, 1])
    base_team_series = (
        dummy_phase["Team"] if "Team" in dummy_phase.columns else (
            dummy_phase["id"] if "id" in dummy_phase.columns else pd.Series(dummy_phase.index.astype(str), index=dummy_phase.index)
        )
    )
    def _clean(lbl: str) -> str:
        if not isinstance(lbl, str):
            return str(lbl)
        s = lbl.replace("output_", "").replace("outputs_", "").replace("output", "")
        s = s.split("_https")[0].replace("_", " ").strip()
        s = "".join(ch for ch in s if ord(ch) < 128).strip()
        return s

    df_lab = dummy_phase.copy()
    df_lab["team_label"] = base_team_series.apply(_clean)
    df_lab["team_label_base"] = df_lab["team_label"].str.replace(r"\d+$", "", regex=True).str.strip()
    top_all = df_lab.sort_values(by="final_score", ascending=False)
    top_dummy = top_all.drop_duplicates(subset="team_label_base", keep="first").head(5)

    if not top_dummy.empty:
        wmap = {row["feature"]: row["weight"] for _, row in factor_weights.iterrows()}
        factor_share = 1.0 - wmap.get("Parsimony Score", 0.0) - wmap.get("Inference Time", 0.0)
        parts = []
        for metric in ["F1", "AUPRC", "Net.Benefit", "ECE"]:
            if metric in top_dummy.columns and metric in wmap:
                cen, scl = scale_lookup.get(metric, (top_dummy[metric].mean(), top_dummy[metric].std()))
                scl = 1.0 if (scl == 0 or np.isnan(scl)) else scl
                z = (top_dummy[metric] - cen) / scl
                parts.append((z * wmap[metric] * factor_share, metric))
        if "parsimony_score" in top_dummy.columns and "Parsimony Score" in wmap:
            parts.append(((1 - top_dummy["parsimony_score"].clip(0, 1)) * wmap["Parsimony Score"], "Parsimony (1 - score)"))
        if "inference_time" in top_dummy.columns and "Inference Time" in wmap:
            cen, scl = scale_lookup.get("Inference Time", (top_dummy["inference_time"].mean(), top_dummy["inference_time"].std()))
            scl = 1.0 if (scl == 0 or np.isnan(scl)) else scl
            z_inf = (top_dummy["inference_time"] - cen) / scl
            parts.append((-z_inf * wmap["Inference Time"], "Inference Speed"))
        if parts:
            x = np.arange(len(top_dummy))
            bottom = np.zeros(len(top_dummy))
            colors = sns.color_palette('Set2', len(parts))
            for i, (ser, lab) in enumerate(parts):
                vals = ser.reindex(top_dummy.index).values
                ax_s2b.bar(x, vals, bottom=bottom, label=lab, color=colors[i], alpha=0.85,
                           edgecolor='black', linewidth=0.6)
                bottom += vals
            labels = top_dummy["team_label"].tolist()
            ax_s2b.set_xticks(x)
            ax_s2b.set_xticklabels([f"Rank {i+1}\n{labels[i]}" for i in range(len(labels))], fontsize=9)
            ax_s2b.set_ylabel('Weighted Contribution', fontweight='bold')
            ax_s2b.set_title('B. Weighted Metric Contributions (Top Dummy-Phase Teams)', fontweight='bold')
            ax_s2b.legend(loc='upper right', frameon=False, fontsize=8)
            ax_s2b.grid(True, alpha=0.3, axis='y')
            ax_s2b.spines['top'].set_visible(False)
            ax_s2b.spines['right'].set_visible(False)

    figure2_path = FIGURES_SUPP_DIR / 'results_supplementary_figure2.png'
    plt.savefig(figure2_path, bbox_inches='tight', dpi=300)
    print(f"   ✓ Saved: {figure2_path.relative_to(ROOT)}")
except Exception as e:
    print(f"[WARN] Could not create Supplementary Figure 2: {e}")
