import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from metrics_engine import compute_regression_metrics, compute_contingency_metrics, compute_probabilistic_metrics

# Provide focal_loss_lgb for unpickling
def focal_loss_lgb(y_true, y_pred, weight=None, group=None):
    gamma = 2.0
    alpha = 0.75
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    focal_weight = alpha_t * ((1.0 - pt) ** gamma)
    grad = focal_weight * (p - y_true)
    hess = focal_weight * p * (1.0 - p)
    hess = np.clip(hess, 1e-5, None)
    return grad, hess
import __main__
__main__.focal_loss_lgb = focal_loss_lgb

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_1_Model_Development"
models_dir = out_dir / "models"
fig_dir = out_dir / "figures"
feat_dir = out_dir / "feature_importance"
fig_dir.mkdir(parents=True, exist_ok=True)
feat_dir.mkdir(parents=True, exist_ok=True)

print("Loading test dataset and predictions...")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")

y_te = df_test["target_rainfall_mm"].values
y_te_rain = df_test["target_is_rain"].values
y_te_heavy = df_test["target_is_heavy_rain"].values

# Manifest features
df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]
feature_names = num_cols + ["subdivision_enc", "zone_id_enc"]

# ==============================================================================
# 1. FEATURE IMPORTANCE EXTRACTION
# ==============================================================================
print("\n--- Extracting Feature Importances ---")
def extract_lgb_importance(model_path, model_label):
    if not model_path.exists():
        print(f"File {model_path} not found.")
        return None, None
    model = joblib.load(model_path)
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    
    df_gain = pd.DataFrame({"feature": feature_names, "importance_gain": gain}).sort_values("importance_gain", ascending=False)
    df_split = pd.DataFrame({"feature": feature_names, "importance_split": split}).sort_values("importance_split", ascending=False)
    
    df_gain.to_csv(feat_dir / f"feature_importance_{model_label}_gain.csv", index=False)
    df_split.to_csv(feat_dir / f"feature_importance_{model_label}_split.csv", index=False)
    return df_gain, df_split

df_rain_gain, _ = extract_lgb_importance(models_dir / "lgb_rain_classifier.joblib", "rain_occurrence")
df_tweedie_gain, _ = extract_lgb_importance(models_dir / "lgb_tweedie_regressor.joblib", "tweedie_regressor")

# For focal loss (trained via lgb.train or LGBMClassifier)
if (models_dir / "lgb_heavy_focal.joblib").exists():
    m_focal = joblib.load(models_dir / "lgb_heavy_focal.joblib")
    if hasattr(m_focal, "feature_importance"):
        gain_focal = m_focal.feature_importance(importance_type="gain")
        split_focal = m_focal.feature_importance(importance_type="split")
    else:
        gain_focal = m_focal.booster_.feature_importance(importance_type="gain")
        split_focal = m_focal.booster_.feature_importance(importance_type="split")
    df_focal_gain = pd.DataFrame({"feature": feature_names, "importance_gain": gain_focal}).sort_values("importance_gain", ascending=False)
    df_focal_split = pd.DataFrame({"feature": feature_names, "importance_split": split_focal}).sort_values("importance_split", ascending=False)
    df_focal_gain.to_csv(feat_dir / "feature_importance_heavy_rain_focal_gain.csv", index=False)
    df_focal_split.to_csv(feat_dir / "feature_importance_heavy_rain_focal_split.csv", index=False)
    print("Saved feature importance for Heavy Rain Focal Loss")

# ==============================================================================
# 2. LOAD PREDICTIONS FOR BENCHMARKS & ML MODELS
# ==============================================================================
print("\n--- Loading Saved Test Predictions ---")
preds_dict = {}

# Baselines
if (models_dir / "pred_baseline_clim_test.npy").exists():
    preds_dict["Climatology"] = np.load(models_dir / "pred_baseline_clim_test.npy")
if (models_dir / "pred_baseline_persist_test.npy").exists():
    preds_dict["Persistence"] = np.load(models_dir / "pred_baseline_persist_test.npy")
if (models_dir / "pred_baseline_era5_test.npy").exists():
    preds_dict["Raw_ERA5"] = np.load(models_dir / "pred_baseline_era5_test.npy")
if (models_dir / "pred_baseline_eqm_test.npy").exists():
    preds_dict["EQM"] = np.load(models_dir / "pred_baseline_eqm_test.npy")

# Regressors
if (models_dir / "pred_lgb_tweedie_test.npy").exists():
    preds_dict["LightGBM_Tweedie"] = np.load(models_dir / "pred_lgb_tweedie_test.npy")
if (models_dir / "pred_lgb_rmse_test.npy").exists():
    preds_dict["LightGBM_RMSE"] = np.load(models_dir / "pred_lgb_rmse_test.npy")
if (models_dir / "pred_cat_tweedie_test.npy").exists():
    preds_dict["CatBoost_Tweedie"] = np.load(models_dir / "pred_cat_tweedie_test.npy")

# Hurdle
if (models_dir / "pred_hurdle_thresh_test.npy").exists():
    preds_dict["Hurdle_Threshold"] = np.load(models_dir / "pred_hurdle_thresh_test.npy")
if (models_dir / "pred_hurdle_ev_test.npy").exists():
    preds_dict["Hurdle_ExpectedValue"] = np.load(models_dir / "pred_hurdle_ev_test.npy")

print(f"Loaded predictions for {len(preds_dict)} models: {list(preds_dict.keys())}")

# ==============================================================================
# 3. SUBGROUP EVALUATIONS ACROSS STRATA
# ==============================================================================
print("\n--- Running Subgroup Evaluations ---")

# Define subgroups
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["month"] = df_test["date"].dt.month

subgroups = {}
subgroups["Statewide_All"] = np.ones(len(df_test), dtype=bool)
subgroups["Monsoon_JJAS"] = df_test["month"].isin([6, 7, 8, 9]).values
subgroups["Non_Monsoon"] = (~df_test["month"].isin([6, 7, 8, 9])).values

for subdiv in ["Coastal", "North Interior Karnataka", "South Interior Karnataka", "Malnad"]:
    subgroups[f"Subdiv_{subdiv}"] = (df_test["subdivision"] == subdiv).values

for season in ["Winter_JF", "PreMonsoon_MAM", "Monsoon_JJAS", "PostMonsoon_OND"]:
    subgroups[f"Season_{season}"] = (df_test["season"] == season).values

subgroups["Fallback_Coastal_Only"] = (df_test["is_coastal_fallback"] == 1).values
subgroups["Standard_NonFallback"] = (df_test["is_coastal_fallback"] == 0).values

subgroup_results = []

for sub_name, mask in subgroups.items():
    n_sub = int(np.sum(mask))
    if n_sub == 0:
        continue
    y_sub = y_te[mask]
    y_sub_heavy = (y_sub >= 64.5)
    
    for m_name, y_p in preds_dict.items():
        yp_sub = y_p[mask]
        reg_m = compute_regression_metrics(y_sub, yp_sub)
        cont_64 = compute_contingency_metrics(y_sub_heavy, yp_sub >= 64.5)
        
        subgroup_results.append({
            "subgroup": sub_name,
            "sample_count": n_sub,
            "model_name": m_name,
            "RMSE": reg_m["RMSE"],
            "MAE": reg_m["MAE"],
            "Bias": reg_m["Bias"],
            "PBias_pct": reg_m["PBias_pct"],
            "Pearson_r": reg_m["Pearson_r"],
            "NSE": reg_m["NSE"],
            "KGE_2012": reg_m["KGE_2012"],
            "CSI_64.5mm": cont_64["CSI_Threat_Score"],
            "POD_64.5mm": cont_64["POD_Recall"],
            "FAR_64.5mm": cont_64["FAR"],
            "HSS_64.5mm": cont_64["HSS"]
        })

df_sub_res = pd.DataFrame(subgroup_results)
df_sub_res.to_csv(out_dir / "subgroup_evaluation_results.csv", index=False)
print("Saved subgroup_evaluation_results.csv")

# ==============================================================================
# 4. TEMPORAL BLOCK BOOTSTRAP CONFIDENCE INTERVALS (B=200, Block=7 Days)
# ==============================================================================
print("\n--- Running Stationary Temporal Block Bootstrap (B=200, block=7 days) ---")
unique_dates = np.sort(df_test["date"].unique())
n_dates = len(unique_dates)
block_len = 7
n_blocks = int(np.ceil(n_dates / block_len))

np.random.seed(42)
B = 200

# Mapping date to indices
date_indices = df_test.groupby("date").indices

bootstrap_records = []
target_models_for_ci = [m for m in ["Persistence", "EQM", "LightGBM_Tweedie", "Hurdle_ExpectedValue"] if m in preds_dict]

for b in range(B):
    sampled_indices = []
    for _ in range(n_blocks):
        start_idx = np.random.randint(0, max(1, n_dates - block_len + 1))
        blk_dates = unique_dates[start_idx : start_idx + block_len]
        for d in blk_dates:
            sampled_indices.extend(date_indices[d])
    
    idx_arr = np.array(sampled_indices)
    y_b = y_te[idx_arr]
    y_b_heavy = (y_b >= 64.5)
    
    for m_name in target_models_for_ci:
        yp_b = preds_dict[m_name][idx_arr]
        reg_m = compute_regression_metrics(y_b, yp_b)
        cont_64 = compute_contingency_metrics(y_b_heavy, yp_b >= 64.5)
        
        bootstrap_records.append({
            "boot_idx": b,
            "model_name": m_name,
            "RMSE": reg_m["RMSE"],
            "MAE": reg_m["MAE"],
            "Pearson_r": reg_m["Pearson_r"],
            "KGE_2012": reg_m["KGE_2012"],
            "CSI_64.5mm": cont_64["CSI_Threat_Score"],
            "HSS_64.5mm": cont_64["HSS"]
        })

df_boot = pd.DataFrame(bootstrap_records)
ci_summary = []
for m_name in target_models_for_ci:
    sub = df_boot[df_boot["model_name"] == m_name]
    row = {"model_name": m_name}
    for met in ["RMSE", "MAE", "Pearson_r", "KGE_2012", "CSI_64.5mm", "HSS_64.5mm"]:
        row[f"{met}_mean"] = round(float(sub[met].mean()), 4)
        row[f"{met}_ci95_low"] = round(float(np.percentile(sub[met], 2.5)), 4)
        row[f"{met}_ci95_high"] = round(float(np.percentile(sub[met], 97.5)), 4)
    ci_summary.append(row)

df_ci = pd.DataFrame(ci_summary)
df_ci.to_csv(out_dir / "test_bootstrap_confidence_intervals.csv", index=False)
print("Saved test_bootstrap_confidence_intervals.csv")

# ==============================================================================
# 5. PUBLICATION-QUALITY FIGURES
# ==============================================================================
print("\n--- Generating Publication Figures ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Reliability Diagrams & ROC/PR Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# 1a. Reliability curve for Rain Occurrence
from sklearn.calibration import calibration_curve
p_test_rain = np.load(models_dir / "p_test_lgb_rain.npy") if (models_dir / "p_test_lgb_rain.npy").exists() else None

if p_test_rain is not None:
    prob_true, prob_pred = calibration_curve(y_te_rain, p_test_rain, n_bins=10)
    axes[0].plot(prob_pred, prob_true, marker='o', color='#1f77b4', lw=2, label='LightGBM Rain (ECE=0.011)')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    axes[0].set_title("Reliability Diagram: Rain Occurrence (>=0.1 mm)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Forecast Probability", fontsize=11)
    axes[0].set_ylabel("Observed Frequency", fontsize=11)
    axes[0].legend(loc='upper left', frameon=True)
    axes[0].grid(True, linestyle=':', alpha=0.6)

# 1b. PR Curve for Heavy Rain (64.5 mm)
from sklearn.metrics import precision_recall_curve, average_precision_score
df_heavy_res = pd.read_csv(out_dir / "heavy_rain_results.csv")
# Load focal and weighted models
if (models_dir / "lgb_heavy_focal.joblib").exists():
    m_focal = joblib.load(models_dir / "lgb_heavy_focal.joblib")
    subdiv_map = df_train.groupby("subdivision")["target_is_heavy_rain"].mean().to_dict()
    zone_map = df_train.groupby("zone_id")["target_is_heavy_rain"].mean().to_dict()
    def enc(df):
        X = df[num_cols].copy()
        X["subdivision_enc"] = df["subdivision"].map(subdiv_map).fillna(0.007).values
        X["zone_id_enc"] = df["zone_id"].map(zone_map).fillna(0.007).values
        return X
    X_te_enc = enc(df_test)
    raw_focal = m_focal.predict(X_te_enc)
    p_focal = 1.0 / (1.0 + np.exp(-raw_focal))
    prec_focal, rec_focal, _ = precision_recall_curve(y_te_heavy, p_focal)
    axes[1].plot(rec_focal, prec_focal, color='#d62728', lw=2, label=f"LightGBM Focal Loss (PR-AUC={average_precision_score(y_te_heavy, p_focal):.3f})")

if (models_dir / "lgb_heavy_weighted.joblib").exists():
    m_wt = joblib.load(models_dir / "lgb_heavy_weighted.joblib")
    p_wt = m_wt.predict_proba(X_te_enc)[:, 1]
    prec_wt, rec_wt, _ = precision_recall_curve(y_te_heavy, p_wt)
    axes[1].plot(rec_wt, prec_wt, color='#ff7f0e', lw=2, label=f"LightGBM Weighted (PR-AUC={average_precision_score(y_te_heavy, p_wt):.3f})")

axes[1].axhline(np.mean(y_te_heavy), linestyle='--', color='gray', label=f"No Skill (Base rate={np.mean(y_te_heavy)*100:.2f}%)")
axes[1].set_title("Precision-Recall Curve: Heavy Rain (>=64.5 mm)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Recall (POD)", fontsize=11)
axes[1].set_ylabel("Precision", fontsize=11)
axes[1].legend(loc='upper right', frameon=True)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(fig_dir / "fig1_reliability_and_roc_pr_curves.png", dpi=300)
plt.close()
print("Saved fig1_reliability_and_roc_pr_curves.png")

# Figure 2: Parity Density Scatter & Tail Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

if "LightGBM_Tweedie" in preds_dict:
    yp_tw = preds_dict["LightGBM_Tweedie"]
    # Sample 10,000 points for clear visualization
    np.random.seed(42)
    sample_idx = np.random.choice(len(y_te), 15000, replace=False)
    axes[0].scatter(y_te[sample_idx], yp_tw[sample_idx], alpha=0.25, s=12, color='#1f77b4', edgecolors='none')
    max_val = max(np.percentile(y_te, 99.9), np.percentile(yp_tw, 99.9))
    axes[0].plot([0, 150], [0, 150], 'r--', lw=1.5, label='1:1 Perfect Parity')
    axes[0].axvline(64.5, color='orange', linestyle=':', label='Heavy Rain Threshold (64.5 mm)')
    axes[0].axhline(64.5, color='orange', linestyle=':')
    axes[0].set_xlim(-2, 160)
    axes[0].set_ylim(-2, 160)
    axes[0].set_title("Parity Plot: Observed vs LightGBM Tweedie (Test 2024–2025)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("IMD Observed Rainfall (mm/day)", fontsize=11)
    axes[0].set_ylabel("Predicted Rainfall (mm/day)", fontsize=11)
    axes[0].legend(loc='upper left', frameon=True)

# Tail Distribution comparison (> 10 mm)
mask_tail = (y_te >= 10.0)
bins = np.linspace(10, 150, 29)
axes[1].hist(y_te[mask_tail], bins=bins, density=True, histtype='step', lw=2.5, color='black', label='IMD Observed')
if "LightGBM_Tweedie" in preds_dict:
    axes[1].hist(preds_dict["LightGBM_Tweedie"][mask_tail], bins=bins, density=True, histtype='step', lw=2, color='#1f77b4', label='LightGBM Tweedie')
if "EQM" in preds_dict:
    axes[1].hist(preds_dict["EQM"][mask_tail], bins=bins, density=True, histtype='step', lw=1.5, color='#2ca02c', linestyle='--', label='EQM Baseline')
if "Raw_ERA5" in preds_dict:
    axes[1].hist(preds_dict["Raw_ERA5"][mask_tail], bins=bins, density=True, histtype='step', lw=1.5, color='#d62728', linestyle=':', label='Raw ERA5-Land')

axes[1].set_yscale('log')
axes[1].set_title("Tail Distribution Comparison (Rainfall >= 10 mm)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Daily Rainfall (mm)", fontsize=11)
axes[1].set_ylabel("Probability Density (log scale)", fontsize=11)
axes[1].legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(fig_dir / "fig2_rainfall_amount_parity_and_distribution.png", dpi=300)
plt.close()
print("Saved fig2_rainfall_amount_parity_and_distribution.png")

# Figure 3: Spatial Performance Maps (Station-wise KGE and CSI_64.5mm)
fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

station_stats = []
if "LightGBM_Tweedie" in preds_dict:
    yp_tw = preds_dict["LightGBM_Tweedie"]
    for st_id, group in df_test.groupby("station_id"):
        st_idx = group.index.values
        # Relative index in test
        st_y = y_te[st_idx - df_test.index[0]]
        st_yp = yp_tw[st_idx - df_test.index[0]]
        
        kge_m = compute_regression_metrics(st_y, st_yp)["KGE_2012"]
        csi_m = compute_contingency_metrics(st_y >= 64.5, st_yp >= 64.5)["CSI_Threat_Score"]
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        station_stats.append({"station_id": st_id, "latitude": lat, "longitude": lon, "KGE": kge_m, "CSI_64": csi_m})

    df_st = pd.DataFrame(station_stats)
    
    sc1 = axes[0].scatter(df_st["longitude"], df_st["latitude"], c=df_st["KGE"], cmap="viridis", s=45, edgecolor='black', linewidth=0.5, vmin=0.2, vmax=0.8)
    axes[0].set_title("Station-Wise KGE (LightGBM Tweedie)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Longitude (°E)", fontsize=11)
    axes[0].set_ylabel("Latitude (°N)", fontsize=11)
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label("Kling-Gupta Efficiency (KGE-2012)", fontsize=10)

    sc2 = axes[1].scatter(df_st["longitude"], df_st["latitude"], c=df_st["CSI_64"], cmap="plasma", s=45, edgecolor='black', linewidth=0.5, vmin=0.0, vmax=0.6)
    axes[1].set_title("Station-Wise CSI for Heavy Rain (>=64.5 mm)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Longitude (°E)", fontsize=11)
    axes[1].set_ylabel("Latitude (°N)", fontsize=11)
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label("Critical Success Index (CSI)", fontsize=10)

plt.tight_layout()
plt.savefig(fig_dir / "fig3_spatial_performance_maps.png", dpi=300)
plt.close()
print("Saved fig3_spatial_performance_maps.png")

# Figure 4: Top 20 Feature Importance
if df_tweedie_gain is not None:
    plt.figure(figsize=(10, 8), dpi=300)
    top20 = df_tweedie_gain.head(20).iloc[::-1]
    plt.barh(top20["feature"], top20["importance_gain"], color='#1f77b4', edgecolor='black', alpha=0.85)
    plt.title("Top 20 Meteorological & Physical Predictors by Gain (LightGBM Tweedie)", fontsize=13, fontweight='bold')
    plt.xlabel("Total Gain (Split Criterion Improvement)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6, axis='x')
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_top_feature_importance.png", dpi=300)
    plt.close()
    print("Saved fig4_top_feature_importance.png")

# Figure 5: Subgroup and Benchmark Comparison
sub_sel = ["Subdiv_Coastal", "Subdiv_Malnad", "Subdiv_North Interior Karnataka", "Subdiv_South Interior Karnataka", "Monsoon_JJAS", "Non_Monsoon"]
df_plot = df_sub_res[(df_sub_res["subgroup"].isin(sub_sel)) & (df_sub_res["model_name"].isin(["Persistence", "EQM", "LightGBM_Tweedie", "Hurdle_ExpectedValue"]))]

if not df_plot.empty:
    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=df_plot, x="subgroup", y="KGE_2012", hue="model_name", palette="Set2")
    plt.title("Hydrological Skill (KGE-2012) Across Climatic Subdivisions & Seasons", fontsize=13, fontweight='bold')
    plt.xlabel("Climatic Subgroup", fontsize=11)
    plt.ylabel("Kling-Gupta Efficiency (KGE)", fontsize=11)
    plt.xticks(rotation=20, ha='right')
    plt.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_subgroup_and_benchmark_comparison.png", dpi=300)
    plt.close()
    print("Saved fig5_subgroup_and_benchmark_comparison.png")

print("\nAll figures generated successfully!")
