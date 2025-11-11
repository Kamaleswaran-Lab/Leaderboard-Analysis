#!/usr/bin/env python3
# results_figure1_stats.py

import os
import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------
SENS_CSV   = 'optimal_threshold_results/evaluation_metrics_sensitivity_method.csv'
YOUDEN_CSV = 'optimal_threshold_results/evaluation_metrics_youden_method.csv'
F1_CSV     = 'optimal_threshold_results/evaluation_metrics_f1_method.csv'

SENS_THRESHOLD = 0.80

OUT_DIR = 'figures/main_figures'
os.makedirs(OUT_DIR, exist_ok=True)
PANEL_E_CSV          = os.path.join(OUT_DIR, 'figure1_panelE_spearman_correlation_table.csv')
PANEL_E_CSV_ROUNDED  = os.path.join(OUT_DIR, 'figure1_panelE_spearman_correlation_table_rounded.csv')
PANEL_E_NPAIRS_CSV   = os.path.join(OUT_DIR, 'figure1_panelE_spearman_npairs.csv')
STATS_TXT            = os.path.join(OUT_DIR, 'figure1_stats.txt')

MIN_PERIODS_CORR = 5      # require at least this many pairwise non-missing obs for a correlation
ROUND_DECIMALS   = 2      # for the rounded correlation CSV & prints


# ----------------------------
# Helpers
# ----------------------------
def _safe_div(n, d):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where((d == 0) | (~np.isfinite(d)), np.nan, n / d)
    return out

def compute_mcc(tp, fp, fn, tn):
    tp = tp.astype(float); fp = fp.astype(float)
    fn = fn.astype(float); tn = tn.astype(float)
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide='ignore', invalid='ignore'):
        mcc = np.where(den == 0, np.nan, num / den)
    return mcc

def nb_median_sd(df):
    return float(np.nanmedian(df['Net Benefit'])) * 100.0, float(np.nanstd(df['Net Benefit'])) * 100.0

def compute_treat_all_nb(df):
    """
    Decision curve treat-all NB:
        NB_all = prevalence - (1 - prevalence) * pt/(1-pt)
    Uses 'Optimal Threshold' if present; else defaults to 0.5.
    """
    pt = df.get('Optimal Threshold', pd.Series(0.5, index=df.index)).clip(1e-6, 1-1e-6)
    totals = df['TP'] + df['FP'] + df['FN'] + df['TN']
    prev   = (df['TP'] + df['FN']) / totals
    nb     = prev - (1 - prev) * (pt / (1 - pt))
    nb     = nb.replace([np.inf, -np.inf], np.nan).dropna()
    return nb * 100.0

def fraction_and_nb(df, thresh=SENS_THRESHOLD):
    elig = df['Sensitivity'] >= thresh
    frac = float(elig.sum()) / float(len(df)) if len(df) > 0 else np.nan
    if elig.any():
        nb_med = float(np.nanmedian(df.loc[elig, 'Net Benefit'])) * 100.0
        nb_sd  = float(np.nanstd(df.loc[elig, 'Net Benefit'])) * 100.0
    else:
        nb_med, nb_sd = np.nan, np.nan
    return frac, nb_med, nb_sd


# ----------------------------
# Load
# ----------------------------
df_sens   = pd.read_csv(SENS_CSV)
df_youden = pd.read_csv(YOUDEN_CSV)
df_f1     = pd.read_csv(F1_CSV)

print("Loaded:")
for name, df in [('Sensitivity', df_sens), ('Youden', df_youden), ('F1', df_f1)]:
    print(f"  {name}: shape={df.shape}")

# ----------------------------
# Filter & Align (AUC > 0.5 universe defined by Sensitivity set)
# ----------------------------
df_sens = df_sens[df_sens['AUC'] > 0.5].copy()
valid_teams = set(df_sens['Team'])

df_youden = df_youden[(df_youden['Team'].isin(valid_teams)) & (df_youden['AUC'] > 0.5)].copy()
df_f1     = df_f1[(df_f1['Team'].isin(valid_teams)) & (df_f1['AUC'] > 0.5)].copy()

print("After filtering (AUC>0.5) and aligning teams:")
print(f"  Sensitivity: n={len(df_sens)}, teams={df_sens['Team'].nunique()}")
print(f"  Youden:      n={len(df_youden)}, teams={df_youden['Team'].nunique()}")
print(f"  F1:          n={len(df_f1)}, teams={df_f1['Team'].nunique()}")

# ----------------------------
# Panel A
# ----------------------------
prelim_n = len(df_sens)
prelim_teams = df_sens['Team'].nunique()
print("\nPanel A:")
print(f"  Submissions: {prelim_n}, Unique teams: {prelim_teams}")

# ----------------------------
# Panel B (≥80% sensitivity)
# ----------------------------
sens_frac, sens_nb, sens_nb_sd = fraction_and_nb(df_sens)
youden_frac, youden_nb, youden_nb_sd = fraction_and_nb(df_youden)
f1_frac, f1_nb, f1_nb_sd = fraction_and_nb(df_f1)

print("\nPanel B (≥80% sensitivity):")
print(f"  Sensitivity method: fraction={sens_frac:.3f}, median NB={sens_nb:.2f}%, SD={sens_nb_sd:.2f}%")
print(f"  Youden method:      fraction={youden_frac:.3f}, median NB={youden_nb:.2f}%, SD={youden_nb_sd:.2f}%")
print(f"  F1 method:          fraction={f1_frac:.3f}, median NB={f1_nb:.2f}%, SD={f1_nb_sd:.2f}%")

# ----------------------------
# Panel C (all models) + Treat-all baselines
# ----------------------------
sens_nb_med_all, sens_nb_sd_all = nb_median_sd(df_sens)
youden_nb_med_all, youden_nb_sd_all = nb_median_sd(df_youden)
f1_nb_med_all, f1_nb_sd_all = nb_median_sd(df_f1)

sens_treat_all   = compute_treat_all_nb(df_sens)
youden_treat_all = compute_treat_all_nb(df_youden)
f1_treat_all     = compute_treat_all_nb(df_f1)

print("\nPanel C (all models):")
print(f"  Sensitivity method: median NB={sens_nb_med_all:.2f}%, SD={sens_nb_sd_all:.2f}%")
print(f"  Youden method:      median NB={youden_nb_med_all:.2f}%, SD={youden_nb_sd_all:.2f}%")
print(f"  F1 method:          median NB={f1_nb_med_all:.2f}%, SD={f1_nb_sd_all:.2f}%")

print("\nTreat-all baseline (median net benefit):")
print(f"  Sensitivity method: {np.nanmedian(sens_treat_all):.2f}%")
print(f"  Youden method:      {np.nanmedian(youden_treat_all):.2f}%")
print(f"  F1 method:          {np.nanmedian(f1_treat_all):.2f}%")

# ----------------------------
# Panel D (key metrics: median ± SD)
# ----------------------------
metrics = ['F1','ECE','AUPRC','AUC','Sensitivity','Specificity']
medians = {m: float(np.nanmedian(df_sens[m])) for m in metrics}
sds     = {m: float(np.nanstd(df_sens[m]))     for m in metrics}

precision_vals = _safe_div(df_sens['TP'], (df_sens['TP'] + df_sens['FP']))
medians['Precision'] = float(np.nanmedian(precision_vals))
sds['Precision']     = float(np.nanstd(precision_vals))

print("\nPanel D (key metrics, median ± SD):")
for m in ['F1','ECE','AUPRC','AUC','Sensitivity','Specificity','Precision']:
    print(f"  {m}: {medians[m]:.4f} ± {sds[m]:.4f}")

# ----------------------------
# Panel E (Spearman, pairwise-complete; keep NaN MCC but ignore pairwise)
# ----------------------------
dfE = df_sens.copy()

# Compute derived columns if counts present
if {'TP','FP','FN','TN'}.issubset(dfE.columns):
  
    dfE['Precision_calc']   = _safe_div(dfE['TP'], (dfE['TP'] + dfE['FP']))
    dfE['MCC_calc']         = compute_mcc(dfE['TP'], dfE['FP'], dfE['FN'], dfE['TN'])
else:
    # fallbacks if counts missing
    dfE['Precision_calc']   = dfE.get('Precision',   np.nan)
    dfE['MCC_calc']         = dfE.get('MCC',         np.nan)

corr_cols = [
    'Sensitivity', 'AUPRC', 'Net Benefit', 'F1', 'ECE', 'AUC',
    'Specificity_calc', 'Precision_calc', 'MCC_calc'
]

df_corr_all = dfE.loc[:, [c for c in corr_cols if c in dfE.columns]] \
                 .replace([np.inf, -np.inf], np.nan)

# Pairwise-complete Spearman with min_periods
corr_spear = df_corr_all.corr(method='spearman', min_periods=MIN_PERIODS_CORR)

# Number of valid pairs per cell
n_pairs = df_corr_all.notna().astype(int).T @ df_corr_all.notna().astype(int)

# Save outputs
corr_spear.to_csv(PANEL_E_CSV)
corr_spear.round(ROUND_DECIMALS).to_csv(PANEL_E_CSV_ROUNDED)
n_pairs.to_csv(PANEL_E_NPAIRS_CSV)

print("\nPanel E (Spearman correlations, rounded):")
print(corr_spear.round(ROUND_DECIMALS).to_string())
print("\nValid pair counts for each correlation cell:")
print(n_pairs.to_string())

# ----------------------------
# Append manuscript-style summary to text file
# ----------------------------
with open(STATS_TXT, 'a') as sf:
    sf.write("\n=== Figure 1 Stats ===\n")
    sf.write(f"Panel A: Submissions={prelim_n}, Unique teams={prelim_teams}\n")
    sf.write(f"Panel B: Sens: frac={sens_frac:.3f}, NB={sens_nb:.2f}%, SD={sens_nb_sd:.2f}%; "
             f"Youden: frac={youden_frac:.3f}, NB={youden_nb:.2f}%, SD={youden_nb_sd:.2f}%; "
             f"F1: frac={f1_frac:.3f}, NB={f1_nb:.2f}%, SD={f1_nb_sd:.2f}%\n")
    sf.write(f"Panel C: Sens NB={sens_nb_med_all:.2f}%±{sens_nb_sd_all:.2f}%; "
             f"Youden NB={youden_nb_med_all:.2f}%±{youden_nb_sd_all:.2f}%; "
             f"F1 NB={f1_nb_med_all:.2f}%±{f1_nb_sd_all:.2f}%\n")
    sf.write(f"Treat-all median NB: Sens={np.nanmedian(sens_treat_all):.2f}%; "
             f"Youden={np.nanmedian(youden_treat_all):.2f}%; "
             f"F1={np.nanmedian(f1_treat_all):.2f}%\n")
    sf.write("Panel D (median ± SD): " +
             ", ".join([f"{m}={medians[m]:.4f}±{sds[m]:.4f}" for m in
                        ['F1','ECE','AUPRC','AUC','Sensitivity','Specificity','Precision']]) + "\n")
    sf.write(f"Panel E: saved to {PANEL_E_CSV} (rounded: {PANEL_E_CSV_ROUNDED}); n-pairs: {PANEL_E_NPAIRS_CSV}\n")
    sf.write("\n--- Spearman correlation matrix (rounded) ---\n")
    sf.write(corr_spear.round(ROUND_DECIMALS).to_string())
    sf.write("\n\n--- Valid pair counts for each correlation cell ---\n")
    sf.write(n_pairs.to_string())
    sf.write("\n")

print(f"\nSaved:")
print(f"  Panel E correlations:       {PANEL_E_CSV}")
print(f"  Panel E correlations (rnd): {PANEL_E_CSV_ROUNDED}")
print(f"  Panel E n-pairs:            {PANEL_E_NPAIRS_CSV}")
print(f"  Summary text:               {STATS_TXT}")