
import pandas as pd
import numpy as np
import json

# Load final and preliminary scores
final_scores = pd.read_csv("data/raw/final_scores.csv")
prelim_scores = pd.read_csv("optimal_threshold_results/evaluation_metrics_sensitivity_method.csv")

# Merge extended fields (parsimony, inference_time) from Eva_csv - Sheet1.csv if present
try:
    eva_path = "data/raw/Eva_csv - Sheet1.csv"
    ext = pd.read_csv(eva_path)
    parsed = []
    def normalize_key(k: str) -> str:
        # turn keys like 'Parsimony Score' -> 'parsimony_score'
        return (
            k.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    def normalize_team(s: str) -> str:
        if not isinstance(s, str):
            return ""
        # remove common prefixes like 'output_' and non-alphanumeric chars (including emoji)
        t = s
        if t.startswith("output_"):
            t = t[len("output_"):]
        # keep ascii letters/numbers and underscores
        import re

        t = re.sub(r"[^0-9a-zA-Z_]+", "", t)
        return t.lower()

    for _, row in ext.iterrows():
        payload = row.get("Evaluated_1")
        if not isinstance(payload, str) or "score" not in payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        score = obj.get("score", {})
        if not isinstance(score, dict):
            continue
        team_raw = row.get("Team name") or row.get("Team") or ""
        rec = {"Team name": team_raw, "team_key": normalize_team(team_raw)}
        for k, v in score.items():
            nn = normalize_key(k)
            rec[nn] = v
        parsed.append(rec)
    if parsed:
        ext_df = pd.DataFrame(parsed)
        # normalize keys to expected column names (snake_case)
        if "parsimony_score" in ext_df.columns:
            ext_df["parsimony_score"] = pd.to_numeric(ext_df["parsimony_score"], errors="coerce")
        if "inference_time" in ext_df.columns:
            ext_df["inference_time"] = pd.to_numeric(ext_df["inference_time"], errors="coerce")

        # build a team_key in final_scores as well so we can match even if display names differ
        def make_team_key_col(df, col_names):
            import re

            # try several candidate columns
            for c in col_names:
                if c in df.columns:
                    def nk(x):
                        if not isinstance(x, str):
                            return ""
                        v = x
                        if v.startswith("output_"):
                            v = v[len("output_"):]
                        return re.sub(r"[^0-9a-zA-Z_]+", "", v).lower()
                    df["team_key"] = df[c].apply(nk)
                    return True
            return False

        make_team_key_col(final_scores, ["Team name", "Team", "team", "Team_clean"])

        # merge on team_key when available, falling back to Team name exact match
        merge_cols = [c for c in ["parsimony_score", "Parsimony Score", "inference_time", "Inference Time"] if c in ext_df.columns or c.lower() in ext_df.columns]
        # prefer normalized column names in ext_df (parsimony_score, inference_time)
        available_ext_cols = []
        if "parsimony_score" in ext_df.columns:
            available_ext_cols.append("parsimony_score")
        if "inference_time" in ext_df.columns:
            available_ext_cols.append("inference_time")

        if "team_key" in final_scores.columns and "team_key" in ext_df.columns:
            if available_ext_cols:
                final_scores = final_scores.merge(ext_df[["team_key"] + available_ext_cols], on="team_key", how="left", suffixes=("", "_ext"))
                for col in available_ext_cols:
                    ext_col = f"{col}_ext"
                    if ext_col in final_scores.columns:
                        if col in final_scores.columns:
                            final_scores[col] = final_scores[col].fillna(final_scores[ext_col])
                        else:
                            final_scores[col] = final_scores[ext_col]
                        final_scores.drop(columns=[ext_col], inplace=True)
        else:
            # fallback to merge on Team name exact match
            if "Team name" in final_scores.columns and "Team name" in ext_df.columns:
                merge_cols_fallback = [c for c in ["parsimony_score", "Parsimony Score", "inference_time", "Inference Time"] if c in ext_df.columns]
                if merge_cols_fallback:
                    final_scores = final_scores.merge(ext_df[["Team name"] + merge_cols_fallback], on="Team name", how="left", suffixes=("", "_ext"))
                    for col in merge_cols_fallback:
                        ext_col = f"{col}_ext"
                        if ext_col in final_scores.columns:
                            if col in final_scores.columns:
                                final_scores[col] = final_scores[col].fillna(final_scores[ext_col])
                            else:
                                final_scores[col] = final_scores[ext_col]
                            final_scores.drop(columns=[ext_col], inplace=True)
except Exception:
    pass

# Compute MCC for both phases if possible
def compute_mcc(tp, fp, fn, tn):
    tp_f = tp.astype(float)
    fp_f = fp.astype(float)
    fn_f = fn.astype(float)
    tn_f = tn.astype(float)
    numerator = tp_f * tn_f - fp_f * fn_f
    denom = np.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
    with np.errstate(divide="ignore", invalid="ignore"):
        mcc = np.where(denom > 0, numerator / denom, np.nan)
    return mcc

if set(["TP", "FP", "FN", "TN"]).issubset(final_scores.columns):
    final_scores["MCC"] = compute_mcc(final_scores["TP"], final_scores["FP"], final_scores["FN"], final_scores["TN"])
if set(["TP", "FP", "FN", "TN"]).issubset(prelim_scores.columns):
    prelim_scores["MCC"] = compute_mcc(prelim_scores["TP"], prelim_scores["FP"], prelim_scores["FN"], prelim_scores["TN"])

# Panel A: Metric distributions and difference between phases
metrics = ["F1", "AUPRC", "Net Benefit", "ECE", "AUC", "Sensitivity", "Specificity", "MCC"]
print("Panel A - Metric Distributions (mean ± std, n) and Final-Prelim Difference:")
for metric in metrics:
    final_vals = pd.to_numeric(final_scores.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    prelim_vals = pd.to_numeric(prelim_scores.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    if not final_vals.empty and not prelim_vals.empty:
        final_mean, final_std, final_n = final_vals.mean(), final_vals.std(), len(final_vals)
        prelim_mean, prelim_std, prelim_n = prelim_vals.mean(), prelim_vals.std(), len(prelim_vals)
        diff = final_mean - prelim_mean
        print(f"{metric}: Final {final_mean:.3f} ± {final_std:.3f} (n={final_n}), Prelim {prelim_mean:.3f} ± {prelim_std:.3f} (n={prelim_n}), Diff: {diff:.3f}")
    elif not final_vals.empty:
        print(f"{metric}: Final {final_vals.mean():.3f} ± {final_vals.std():.3f} (n={len(final_vals)}), Prelim NA, Diff: NA")
    elif not prelim_vals.empty:
        print(f"{metric}: Final NA, Prelim {prelim_vals.mean():.3f} ± {prelim_vals.std():.3f} (n={len(prelim_vals)}), Diff: NA")

# Compute normalized inference time (if scale params available)
try:
    scale_df = pd.read_csv("data/processed/scale_params.csv")
    scale_lookup = {row["feature"]: (row["center"], row["scale"]) for _, row in scale_df.iterrows()}
    if "inference_time" in final_scores.columns:
        c, s = scale_lookup.get("Inference Time", (0.0, 1.0))
        final_scores["Inference Time (norm)"] = (pd.to_numeric(final_scores["inference_time"], errors="coerce") - c) / s
except Exception:
    pass

# Panel B: Spearman correlations with Rank using selected metric list
heatmap_metrics = ["Sensitivity", "AUPRC", "Net Benefit", "F1", "ECE", "AUC", "Specificity", "Precision", "MCC", "TP", "FP", "TN", "FN", "Rank"]
# include parsimony and normalized inference time if present
if "parsimony_score" in final_scores.columns:
    heatmap_metrics.insert(8, "parsimony_score")
elif "Parsimony Score" in final_scores.columns:
    heatmap_metrics.insert(8, "Parsimony Score")
if "Inference Time (norm)" in final_scores.columns:
    heatmap_metrics.insert(9, "Inference Time (norm)")

# Add Precision if possible
if set(["TP", "FP"]).issubset(final_scores.columns):
    tp_fp = final_scores["TP"] + final_scores["FP"]
    final_scores["Precision"] = np.where(tp_fp > 0, final_scores["TP"] / tp_fp, np.nan)

available_metrics = [m for m in heatmap_metrics if m in final_scores.columns]
corr_matrix = None
if len(available_metrics) >= 3:
    corr_matrix = final_scores[available_metrics].corr(method="spearman")
    print("\nPanel B - Spearman Correlations with Composite Rank:")
    for metric in available_metrics:
        if metric != "Rank" and "Rank" in (corr_matrix.columns if corr_matrix is not None else []):
            print(f"{metric}: {corr_matrix.at[metric, 'Rank']:.2f}")
else:
    print("Not enough metrics to compute Spearman correlation matrix for Panel B.")

# Panel D: Alt Rank, FPR, and Rank Diff for each metric
print("\nPanel D - Alt Rank, FPR, and Rank Diff (Top 5 for each metric):")
stability_metrics = ["MCC", "AUC", "AUPRC", "F1"]
for metric in stability_metrics:
    if metric in final_scores.columns:
        df = final_scores.dropna(subset=["Rank", metric, "FP", "TN"]).copy()
        df["Alt Rank"] = df[metric].rank(ascending=False, method="min")
        df["FPR"] = df["FP"] / (df["FP"] + df["TN"])
        df["Rank Diff"] = df["Alt Rank"] - df["Rank"]
        top5 = df.nsmallest(5, "Alt Rank")
        print(f"\n{metric} Top 5:")
        for _, row in top5.iterrows():
            print(f"Team: {row['Team name']}, Alt Rank: {int(row['Alt Rank'])}, Composite Rank: {int(row['Rank'])}, FPR: {row['FPR']:.2f}, Rank Diff: {row['Rank Diff']:.0f}")

        # Save stats to TXT including Spearman correlation matrix and summaries
        from pathlib import Path
        OUT_DIR = Path("figures") / "main_figures"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "figure3_stats.txt"
        with open(out_path, "w") as fh:
            fh.write("Figure 3 - Extracted Statistics\n")
            fh.write("=" * 60 + "\n\n")
            fh.write("Available metrics included in heatmap:\n")
            fh.write(", ".join(available_metrics) + "\n\n")
            # Panel A - write metric diffs (Final vs Preliminary) if prelim available
            fh.write("Panel A - Metric distributions (Final vs Preliminary) diffs:\n")
            for metric in metrics:
                final_vals = pd.to_numeric(final_scores.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
                prelim_vals = pd.to_numeric(prelim_scores.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
                if not final_vals.empty and not prelim_vals.empty:
                    final_mean, final_std, final_n = final_vals.mean(), final_vals.std(), len(final_vals)
                    prelim_mean, prelim_std, prelim_n = prelim_vals.mean(), prelim_vals.std(), len(prelim_vals)
                    diff = final_mean - prelim_mean
                    fh.write(f"{metric}: Final {final_mean:.3f} ± {final_std:.3f} (n={final_n}), Prelim {prelim_mean:.3f} ± {prelim_std:.3f} (n={prelim_n}), Diff: {diff:.3f}\n")
                elif not final_vals.empty:
                    fh.write(f"{metric}: Final {final_vals.mean():.3f} ± {final_vals.std():.3f} (n={len(final_vals)}), Prelim NA, Diff: NA\n")
                elif not prelim_vals.empty:
                    fh.write(f"{metric}: Final NA, Prelim {prelim_vals.mean():.3f} ± {prelim_vals.std():.3f} (n={len(prelim_vals)}), Diff: NA\n")
            fh.write("\n")
            # Panel B: Spearman matrix and per-metric correlation with Rank
            if corr_matrix is not None:
                fh.write("Spearman correlation matrix:\n")
                fh.write(corr_matrix.to_string() + "\n\n")
                fh.write("Spearman correlations with Rank:\n")
                for metric in available_metrics:
                    if metric != "Rank" and "Rank" in corr_matrix.columns:
                        fh.write(f" - {metric}: {corr_matrix.at[metric, 'Rank']:.2f}\n")
                fh.write("\n")
            else:
                fh.write("Not enough metrics to compute Spearman correlation matrix for Panel B.\n\n")
            fh.write("Distribution summaries - Final phase:\n")
            # build distribution table
            rows = []
            for m in metrics:
                if m in final_scores.columns:
                    s = pd.to_numeric(final_scores[m], errors="coerce").dropna()
                    rows.append({"metric": m, "n": int(s.count()), "mean": float(s.mean()) if not s.empty else np.nan, "std": float(s.std()) if not s.empty else np.nan, "min": float(s.min()) if not s.empty else np.nan, "max": float(s.max()) if not s.empty else np.nan})
            # include parsimony & inference norm if present
            if "parsimony_score" in final_scores.columns and "parsimony_score" not in [r['metric'] for r in rows]:
                s = pd.to_numeric(final_scores['parsimony_score'], errors="coerce").dropna()
                rows.append({"metric": 'parsimony_score', "n": int(s.count()), "mean": float(s.mean()) if not s.empty else np.nan, "std": float(s.std()) if not s.empty else np.nan, "min": float(s.min()) if not s.empty else np.nan, "max": float(s.max()) if not s.empty else np.nan})
            if "Inference Time (norm)" in final_scores.columns and 'Inference Time (norm)' not in [r['metric'] for r in rows]:
                s = pd.to_numeric(final_scores['Inference Time (norm)'], errors="coerce").dropna()
                rows.append({"metric": 'Inference Time (norm)', "n": int(s.count()), "mean": float(s.mean()) if not s.empty else np.nan, "std": float(s.std()) if not s.empty else np.nan, "min": float(s.min()) if not s.empty else np.nan, "max": float(s.max()) if not s.empty else np.nan})
            summary_df = pd.DataFrame(rows)
            if not summary_df.empty:
                fh.write(summary_df.to_string(index=False) + "\n\n")
            # parsimony mean±sd
            if 'parsimony_score' in final_scores.columns or 'Parsimony Score' in final_scores.columns:
                col = 'parsimony_score' if 'parsimony_score' in final_scores.columns else 'Parsimony Score'
                s = pd.to_numeric(final_scores[col], errors='coerce').dropna()
                fh.write('Parsimony summaries:\n')
                fh.write(f" - Final ({col}): {s.mean():.4f} ± {s.std():.4f} (n={s.count()})\n\n")
            if 'Inference Time (norm)' in final_scores.columns:
                s = pd.to_numeric(final_scores['Inference Time (norm)'], errors='coerce').dropna()
                fh.write(f"Inference Time (norm) summaries:\n - Final: {s.mean():.4f} ± {s.std():.4f} (n={s.count()})\n\n")
            # Panel D: Top5 per stability metric
            fh.write("Panel D - Top 5 by alternative metrics:\n")
            for metric in ["MCC", "AUC", "AUPRC", "F1"]:
                if metric in final_scores.columns:
                    df = final_scores.dropna(subset=["Rank", metric, "FP", "TN"]).copy()
                    if df.empty:
                        continue
                    df["Alt Rank"] = df[metric].rank(ascending=False, method="min")
                    df["FPR"] = df["FP"] / (df["FP"] + df["TN"]) if ((df["FP"] + df["TN"]) != 0).any() else np.nan
                    df["Rank Diff"] = df["Alt Rank"] - df["Rank"]
                    top5 = df.nsmallest(5, "Alt Rank")
                    fh.write(f"\n{metric} Top 5:\n")
                    for _, row in top5.iterrows():
                        team = row.get('Team name', row.get('Team_clean', ''))
                        fh.write(f"Team: {team}, Alt Rank: {int(row['Alt Rank'])}, Composite Rank: {int(row['Rank'])}, FPR: {row.get('FPR', np.nan):.2f}, Rank Diff: {row['Rank Diff']:.0f}\n")
            fh.write("\n")

        print(f"Saved Figure 3 stats summary to: {out_path}")
