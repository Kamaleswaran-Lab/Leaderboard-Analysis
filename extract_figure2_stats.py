import pandas as pd
import numpy as np
import json

# Panel B: Relative Weight Contribution
factor_weights = pd.read_csv("data/processed/factor_weights.csv")
core_metrics = ["F1", "AUPRC", "Net.Benefit", "ECE"]
panel_b_weights = factor_weights[factor_weights["feature"].isin(core_metrics)].copy()
weights_abs = np.abs(panel_b_weights["weight"])
factor_target = 0.90
panel_b_weights["share"] = weights_abs * (factor_target / weights_abs.sum())
panel_b_weights = pd.concat([
    panel_b_weights,
    pd.DataFrame({
        "feature": ["Parsimony Score", "Inference Time"],
        "weight": [factor_weights.set_index("feature").at["Parsimony Score", "weight"],
                   factor_weights.set_index("feature").at["Inference Time", "weight"]],
        "share": [0.05, 0.05]
    })
], ignore_index=True)
print("Panel B - Percent Contribution:")
print(panel_b_weights[["feature", "share"]].assign(percent=lambda df: df["share"] * 100))

# Panel C: Metric Distributions
dummy_phase = pd.read_csv("data/processed/Dummy_phase_scores_new_full.csv")
# Use preliminary metrics from the original optimal_threshold_results (keeps expected 30->28 behavior)
prelim_sens = pd.read_csv("optimal_threshold_results/evaluation_metrics_sensitivity_method.csv")
# Remove AUC outliers (AUC <= 0.5) to match Figure 2 filtering
if "AUC" in dummy_phase.columns:
    before = len(dummy_phase)
    dummy_phase = dummy_phase[dummy_phase["AUC"] > 0.5].copy()
    after = len(dummy_phase)
    print(f"Filtered Dummy phase by AUC>0.5: {before} -> {after}")
if "AUC" in prelim_sens.columns:
    before_p = len(prelim_sens)
    prelim_sens = prelim_sens[prelim_sens["AUC"] > 0.5].copy()
    after_p = len(prelim_sens)
    print(f"Filtered Preliminary phase by AUC>0.5: {before_p} -> {after_p}")
metric_list = ["F1", "AUPRC", "AUC", "ECE", "Sensitivity", "Specificity", "Net.Benefit", "MCC"]
# NOTE: outlier removal (IQR trimming) intentionally disabled for distribution summaries

print("\nPanel C - Metric Distributions (mean ± std, Dummy Phase):")
for metric in metric_list:
    if metric in dummy_phase.columns:
        series = pd.to_numeric(dummy_phase[metric], errors="coerce").dropna()
        vals = series
        if not vals.empty:
            print(f"{metric}: {vals.mean():.3f} ± {vals.std():.3f} (n={len(vals)})")

print("\nPanel C - Metric Distributions (mean ± std, Preliminary Phase):")
for metric in metric_list:
    if metric in prelim_sens.columns:
        series = pd.to_numeric(prelim_sens[metric], errors="coerce").dropna()
        vals = series
        if not vals.empty:
            print(f"{metric}: {vals.mean():.3f} ± {vals.std():.3f} (n={len(vals)})")

# Panel D: Metric Correlations with Ranking
dummy_phase["Rank"] = dummy_phase["final_score"].rank(ascending=False)
if {"TP", "FP", "FN", "TN"}.issubset(dummy_phase.columns):
    tp, fp, fn, tn = [pd.to_numeric(dummy_phase[x], errors="coerce") for x in ["TP", "FP", "FN", "TN"]]
    numerator = tp * tn - fp * fn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    dummy_phase["MCC"] = np.where(denom > 0, numerator / denom, np.nan)

# Use metrics consistent with Figure 2 (exclude scaled weighted score)
selected_metrics = ["Sensitivity", "AUPRC", "Net.Benefit", "F1", "ECE"]
other_metrics = ["AUC", "Specificity", "Precision", "MCC"]
confusion_metrics = ["TP", "FP", "TN", "FN", "Rank"]
# If a parsimony column exists, include it (use raw parsimony, not inverted)
parsimony_variants = ["parsimony_score", "Parsimony Score", "Parsimony"]
parsimony_col = None
for col in parsimony_variants:
    if col in dummy_phase.columns:
        parsimony_col = col
        break
if parsimony_col is not None:
    other_metrics.append(parsimony_col)

# Compute normalized inference time (i_norm) using scale params if available
try:
    scale_df = pd.read_csv("data/processed/scale_params.csv")
    scale_lookup = {row["feature"]: (row["center"], row["scale"]) for _, row in scale_df.iterrows()}
    if "inference_time" in dummy_phase.columns:
        center, scale = scale_lookup.get("Inference Time", (0.0, 1.0))
        dummy_phase["Inference Time (norm)"] = (pd.to_numeric(dummy_phase["inference_time"], errors="coerce") - center) / scale
    if "inference_time" in prelim_sens.columns:
        center, scale = scale_lookup.get("Inference Time", (0.0, 1.0))
        prelim_sens["Inference Time (norm)"] = (pd.to_numeric(prelim_sens["inference_time"], errors="coerce") - center) / scale
    # add normalized inference to other_metrics for heatmap inclusion
    other_metrics.append("Inference Time (norm)")
except Exception:
    # if scale params not available, skip adding normalized inference
    pass

metrics_for_corr = [m for m in (selected_metrics + other_metrics + confusion_metrics) if m in dummy_phase.columns]
if "Rank" not in metrics_for_corr:
    metrics_for_corr.append("Rank")
if len(metrics_for_corr) >= 3:
    corr_matrix = dummy_phase[metrics_for_corr].corr(method="spearman")
    print("\nPanel D - Spearman Correlations with Rank:")
    for metric in metrics_for_corr:
        if metric != "Rank" and "Rank" in corr_matrix.columns:
            print(f"{metric}: {corr_matrix.at[metric, 'Rank']:.2f}")
else:
    print("Not enough metrics to compute Spearman correlation matrix for Panel D.")
    
# Prepare output directory for stats TXT
from pathlib import Path
OUT_DIR = Path("figures") / "main_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare distribution summaries for both phases
def summarize_df(df, metrics):
    rows = []
    for m in metrics:
        if m in df.columns:
            s_clean = pd.to_numeric(df[m], errors="coerce").dropna()
            rows.append({
                "metric": m,
                "n": int(s_clean.count()),
                "mean": float(s_clean.mean()) if not s_clean.empty else np.nan,
                "std": float(s_clean.std()) if not s_clean.empty else np.nan,
                "min": float(s_clean.min()) if not s_clean.empty else np.nan,
                "max": float(s_clean.max()) if not s_clean.empty else np.nan,
            })
    return pd.DataFrame(rows)

metrics_for_summary = [m for m in metric_list if m in dummy_phase.columns or m in prelim_sens.columns]
# If any parsimony column variants exist, include them in summaries
parsimony_variants = ["parsimony_score", "Parsimony Score", "Parsimony"]
parsimony_col = None
for col in parsimony_variants:
    if col in dummy_phase.columns or col in prelim_sens.columns:
        parsimony_col = col
        break
if parsimony_col is not None and parsimony_col not in metrics_for_summary:
    metrics_for_summary.append(parsimony_col)
# Include normalized inference time in summaries if present
if "Inference Time (norm)" in dummy_phase.columns or "Inference Time (norm)" in prelim_sens.columns:
    if "Inference Time (norm)" not in metrics_for_summary:
        metrics_for_summary.append("Inference Time (norm)")
summary_dummy = summarize_df(dummy_phase, metrics_for_summary)
summary_prelim = summarize_df(prelim_sens, metrics_for_summary)

stats_path = OUT_DIR / "figure2_stats.txt"
with open(stats_path, "w") as fh:
    fh.write("Figure 2 - Extracted Statistics\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Panel B - Percent Contribution:\n")
    for _, row in panel_b_weights.iterrows():
        fh.write(f" - {row['feature']}: {row['share']*100:.2f}%\n")
    fh.write("\n")
    # AUC filter counts (if variables exist)
    if 'before' in globals() and 'after' in globals():
        fh.write(f"Dummy phase: {before} -> {after} (AUC>0.5 filter)\n")
    if 'before_p' in globals() and 'after_p' in globals():
        fh.write(f"Preliminary phase: {before_p} -> {after_p} (AUC>0.5 filter)\n")
    fh.write("\nAvailable metrics included in heatmap:\n")
    fh.write(", ".join(metrics_for_corr) + "\n\n")
    if corr_matrix is not None:
        fh.write("Spearman correlation matrix:\n")
        fh.write(corr_matrix.to_string() + "\n\n")
    fh.write("Distribution summaries - Dummy phase:\n")
    fh.write(summary_dummy.to_string(index=False) + "\n\n")
    fh.write("Distribution summaries - Preliminary phase:\n")
    fh.write(summary_prelim.to_string(index=False) + "\n\n")

    # Parsimony summary (mean ± sd) for both phases if available — append to TXT only
    if parsimony_col is not None:
        fh.write("Parsimony summaries:\n")
        for phase_name, df in [("Dummy phase", dummy_phase), ("Preliminary phase", prelim_sens)]:
            if parsimony_col in df.columns:
                s = pd.to_numeric(df[parsimony_col], errors="coerce").dropna()
                n = int(s.count())
                mean = float(s.mean()) if n > 0 else np.nan
                sd = float(s.std()) if n > 0 else np.nan
                fh.write(f" - {phase_name} ({parsimony_col}): {mean:.4f} ± {sd:.4f} (n={n})\n")
            else:
                fh.write(f" - {phase_name}: {parsimony_col} not available\n")
        fh.write("\n")
    # Normalized inference time summary
    if "Inference Time (norm)" in dummy_phase.columns or "Inference Time (norm)" in prelim_sens.columns:
        fh.write("Inference Time (norm) summaries:\n")
        for phase_name, df in [("Dummy phase", dummy_phase), ("Preliminary phase", prelim_sens)]:
            if "Inference Time (norm)" in df.columns:
                s = pd.to_numeric(df["Inference Time (norm)"], errors="coerce").dropna()
                n = int(s.count())
                mean = float(s.mean()) if n > 0 else np.nan
                sd = float(s.std()) if n > 0 else np.nan
                fh.write(f" - {phase_name}: {mean:.4f} ± {sd:.4f} (n={n})\n")
            else:
                fh.write(f" - {phase_name}: Inference Time (norm) not available\n")
        fh.write("\n")
    # Top 5 by final_score
    if 'final_score' in dummy_phase.columns:
        fh.write("Top 5 Dummy-phase submissions by final_score:\n")
        top5 = dummy_phase.sort_values('final_score', ascending=False).head(5)
        for _, r in top5.iterrows():
            team = r.get('Team', r.get('id', ''))
            fh.write(f" - {team}: final_score={r.get('final_score')}, Rank={r.get('Rank')}\n")

print(f"\nSaved Figure 2 stats summary to: {stats_path}")
