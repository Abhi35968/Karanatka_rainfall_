import os
import joblib
import shap
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
models_dir = base_dir / "Stage4_1_Model_Development/models"
stage5_dir = base_dir / "Stage5_Model_Interpretation"
fig_dir = stage5_dir / "stage5_figures"
stage5_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

print("Loading dataset metadata and partitions...")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]
feature_names = num_cols + ["subdivision_enc", "zone_id_enc"]

# Target encodings strictly from training data
subdiv_map_reg = df_train.groupby("subdivision")["target_rainfall_mm"].mean().to_dict()
zone_map_reg = df_train.groupby("zone_id")["target_rainfall_mm"].mean().to_dict()

subdiv_map_heavy = df_train.groupby("subdivision")["target_is_heavy_rain"].mean().to_dict()
zone_map_heavy = df_train.groupby("zone_id")["target_is_heavy_rain"].mean().to_dict()

def encode_features_reg(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_map_reg).fillna(3.5).values
    X["zone_id_enc"] = df["zone_id"].map(zone_map_reg).fillna(3.5).values
    return X

def encode_features_heavy(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_map_heavy).fillna(0.007).values
    X["zone_id_enc"] = df["zone_id"].map(zone_map_heavy).fillna(0.007).values
    return X

X_tr_reg = encode_features_reg(df_train)
X_va_reg = encode_features_reg(df_val)
y_va = df_val["target_rainfall_mm"].values

X_va_heavy = encode_features_heavy(df_val)
y_va_heavy = df_val["target_is_heavy_rain"].values

# Load trained models
print("Loading trained models...")
m_lgb_tweedie = joblib.load(models_dir / "lgb_tweedie_regressor.joblib")
m_xgb_tweedie = joblib.load(models_dir / "xgb_tweedie_regressor.joblib")
m_cat_tweedie = joblib.load(models_dir / "cat_tweedie_regressor.joblib")
m_lgb_heavy_focal = joblib.load(models_dir / "lgb_heavy_focal.joblib")

# ==============================================================================
# 1. CROSS-MODEL FEATURE IMPORTANCE RECONCILIATION
# ==============================================================================
print("\n--- 1. Reconciling Feature Importance Across Models ---")
# LightGBM Tweedie Gain
lgb_gain = m_lgb_tweedie.booster_.feature_importance(importance_type="gain")
lgb_split = m_lgb_tweedie.booster_.feature_importance(importance_type="split")

# XGBoost Tweedie Gain
xgb_booster = m_xgb_tweedie.get_booster()
xgb_score_dict = xgb_booster.get_score(importance_type="gain")
xgb_gain = np.array([xgb_score_dict.get(col, 0.0) for col in feature_names])

# CatBoost Tweedie Importance
cat_importance = m_cat_tweedie.get_feature_importance()

# LightGBM Focal Loss Heavy Rain Gain
if hasattr(m_lgb_heavy_focal, "feature_importance"):
    lgb_focal_gain = m_lgb_heavy_focal.feature_importance(importance_type="gain")
else:
    lgb_focal_gain = m_lgb_heavy_focal.booster_.feature_importance(importance_type="gain")

df_imp = pd.DataFrame({
    "feature": feature_names,
    "lgb_tweedie_gain": lgb_gain,
    "lgb_tweedie_gain_pct": 100.0 * lgb_gain / np.sum(lgb_gain),
    "lgb_tweedie_split": lgb_split,
    "xgb_tweedie_gain": xgb_gain,
    "xgb_tweedie_gain_pct": 100.0 * xgb_gain / (np.sum(xgb_gain) if np.sum(xgb_gain) > 0 else 1.0),
    "cat_tweedie_importance": cat_importance,
    "cat_tweedie_importance_pct": 100.0 * cat_importance / np.sum(cat_importance),
    "lgb_focal_heavy_gain": lgb_focal_gain,
    "lgb_focal_heavy_gain_pct": 100.0 * lgb_focal_gain / np.sum(lgb_focal_gain)
})

# Add individual ranks and mean rank across regression models
df_imp["rank_lgb"] = df_imp["lgb_tweedie_gain"].rank(ascending=False).astype(int)
df_imp["rank_xgb"] = df_imp["xgb_tweedie_gain"].rank(ascending=False).astype(int)
df_imp["rank_cat"] = df_imp["cat_tweedie_importance"].rank(ascending=False).astype(int)
df_imp["rank_focal"] = df_imp["lgb_focal_heavy_gain"].rank(ascending=False).astype(int)
df_imp["mean_rank_regressors"] = (df_imp["rank_lgb"] + df_imp["rank_xgb"] + df_imp["rank_cat"]) / 3.0

df_imp = df_imp.sort_values("mean_rank_regressors", ascending=True).reset_index(drop=True)
df_imp.to_csv(stage5_dir / "feature_importance_comparison.csv", index=False)
print("Saved feature_importance_comparison.csv")

# ==============================================================================
# 2. PERMUTATION IMPORTANCE ON FULL VALIDATION DATASET (N=91,980)
# ==============================================================================
print("\n--- 2. Computing Permutation Importance on Full Validation Data ---")
base_pred_va = np.clip(m_lgb_tweedie.predict(X_va_reg), 0.0, None)
base_rmse = np.sqrt(np.mean((y_va - base_pred_va) ** 2))
base_mae = np.mean(np.abs(y_va - base_pred_va))
base_r = np.corrcoef(y_va, base_pred_va)[0, 1]
print(f"Baseline Validation: RMSE = {base_rmse:.4f} mm, MAE = {base_mae:.4f} mm, r = {base_r:.4f}")

perm_records = []
seeds = [42, 101, 2024]
n_feats = len(feature_names)

X_va_mat = X_va_reg.copy()

for i, feat in enumerate(feature_names):
    t_feat_start = time.time()
    delta_rmse_list = []
    delta_mae_list = []
    delta_r_list = []
    
    orig_vals = X_va_mat[feat].values.copy()
    
    for s in seeds:
        np.random.seed(s)
        perm_vals = np.random.permutation(orig_vals)
        X_va_mat[feat] = perm_vals
        
        y_pred_perm = np.clip(m_lgb_tweedie.predict(X_va_mat), 0.0, None)
        rmse_p = np.sqrt(np.mean((y_va - y_pred_perm) ** 2))
        mae_p = np.mean(np.abs(y_va - y_pred_perm))
        r_p = np.corrcoef(y_va, y_pred_perm)[0, 1]
        
        delta_rmse_list.append(rmse_p - base_rmse)
        delta_mae_list.append(mae_p - base_mae)
        delta_r_list.append(base_r - r_p)
    
    # Restore original column
    X_va_mat[feat] = orig_vals
    
    mean_d_rmse = float(np.mean(delta_rmse_list))
    std_d_rmse = float(np.std(delta_rmse_list))
    mean_d_mae = float(np.mean(delta_mae_list))
    mean_d_r = float(np.mean(delta_r_list))
    
    perm_records.append({
        "feature": feat,
        "delta_rmse_mean": mean_d_rmse,
        "delta_rmse_std": std_d_rmse,
        "delta_mae_mean": mean_d_mae,
        "delta_r_mean": mean_d_r,
        "percent_rmse_increase": 100.0 * mean_d_rmse / base_rmse
    })
    
    if (i + 1) % 10 == 0 or (i + 1) == n_feats:
        print(f"  Processed {i+1}/{n_feats} features (last took {time.time()-t_feat_start:.2f}s)...")

df_perm = pd.DataFrame(perm_records).sort_values("delta_rmse_mean", ascending=False).reset_index(drop=True)
df_perm["permutation_rank"] = np.arange(1, len(df_perm) + 1)
df_perm.to_csv(stage5_dir / "permutation_importance.csv", index=False)
print("Saved permutation_importance.csv")

# ==============================================================================
# 3. SHAP EXPLANATION (TREEEXPLAINER)
# ==============================================================================
print("\n--- 3. Computing SHAP Values (TreeExplainer) on Validation Sample ---")
# To guarantee dense representation of heavy rainfall and seasonal variation:
# Sample 10,000 validation records: all 332 positive heavy rain events + 9,668 stratified by season
pos_idx = np.where(y_va_heavy == 1)[0]
neg_idx = np.where(y_va_heavy == 0)[0]
np.random.seed(42)
neg_sampled_idx = np.random.choice(neg_idx, size=10000 - len(pos_idx), replace=False)
shap_sample_idx = np.sort(np.concatenate([pos_idx, neg_sampled_idx]))

sample_X_reg = X_va_reg.iloc[shap_sample_idx].reset_index(drop=True)
sample_X_heavy = X_va_heavy.iloc[shap_sample_idx].reset_index(drop=True)
sample_y = y_va[shap_sample_idx]
sample_y_heavy = y_va_heavy[shap_sample_idx]

print(f"SHAP sample size: {len(shap_sample_idx)} (Heavy rain positive instances: {len(pos_idx)})")

# SHAP for LightGBM Tweedie
print("Calculating SHAP values for LightGBM Tweedie...")
t0 = time.time()
explainer_tweedie = shap.TreeExplainer(m_lgb_tweedie)
shap_tweedie = explainer_tweedie.shap_values(sample_X_reg)
print(f"Tweedie SHAP completed in {time.time()-t0:.2f}s, shape: {shap_tweedie.shape}")

# SHAP for LightGBM Focal Loss
print("Calculating SHAP values for LightGBM Heavy Rain Focal Loss...")
t0 = time.time()
explainer_focal = shap.TreeExplainer(m_lgb_heavy_focal)
shap_focal = explainer_focal.shap_values(sample_X_heavy)
if isinstance(shap_focal, list):
    shap_focal = shap_focal[1]
print(f"Focal SHAP completed in {time.time()-t0:.2f}s, shape: {shap_focal.shape}")

# Mean absolute SHAP values
mean_abs_tweedie = np.mean(np.abs(shap_tweedie), axis=0)
mean_abs_focal = np.mean(np.abs(shap_focal), axis=0)

df_shap = pd.DataFrame({
    "feature": feature_names,
    "shap_mean_abs_tweedie": mean_abs_tweedie,
    "shap_mean_abs_heavy_focal": mean_abs_focal
})
df_shap["rank_shap_tweedie"] = df_shap["shap_mean_abs_tweedie"].rank(ascending=False).astype(int)
df_shap["rank_shap_focal"] = df_shap["shap_mean_abs_heavy_focal"].rank(ascending=False).astype(int)
df_shap = df_shap.sort_values("rank_shap_tweedie", ascending=True).reset_index(drop=True)
df_shap.to_csv(stage5_dir / "shap_summary_values.csv", index=False)
print("Saved shap_summary_values.csv")

# ==============================================================================
# 4. PUBLICATION FIGURES: FIG 1, FIG 2, FIG 3
# ==============================================================================
print("\n--- 4. Generating Publication Figures (Fig 1, 2, 3) ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------------------------
# Figure 1: Cross-Model Feature Importance
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)

# 1a. LightGBM Tweedie Gain %
top_lgb = df_imp.sort_values("lgb_tweedie_gain_pct", ascending=False).head(15)
sns.barplot(data=top_lgb, x="lgb_tweedie_gain_pct", y="feature", ax=axes[0, 0], color="#2b5c8f")
axes[0, 0].set_title("A. LightGBM Tweedie (Gain Importance %)", fontsize=13, fontweight="bold")
axes[0, 0].set_xlabel("Relative Gain Importance (%)", fontsize=11)
axes[0, 0].set_ylabel("")

# 1b. CatBoost Tweedie Importance %
top_cat = df_imp.sort_values("cat_tweedie_importance_pct", ascending=False).head(15)
sns.barplot(data=top_cat, x="cat_tweedie_importance_pct", y="feature", ax=axes[0, 1], color="#e07a5f")
axes[0, 1].set_title("B. CatBoost Tweedie (Feature Importance %)", fontsize=13, fontweight="bold")
axes[0, 1].set_xlabel("Relative Importance (%)", fontsize=11)
axes[0, 1].set_ylabel("")

# 1c. XGBoost Tweedie Gain %
top_xgb = df_imp.sort_values("xgb_tweedie_gain_pct", ascending=False).head(15)
sns.barplot(data=top_xgb, x="xgb_tweedie_gain_pct", y="feature", ax=axes[1, 0], color="#3d405b")
axes[1, 0].set_title("C. XGBoost Tweedie (Gain Importance %)", fontsize=13, fontweight="bold")
axes[1, 0].set_xlabel("Relative Gain Importance (%)", fontsize=11)
axes[1, 0].set_ylabel("")

# 1d. Validation Permutation Importance (Delta RMSE mm)
top_perm = df_perm.sort_values("delta_rmse_mean", ascending=False).head(15)
axes[1, 1].barh(range(len(top_perm)), top_perm["delta_rmse_mean"], xerr=top_perm["delta_rmse_std"],
                color="#81b29a", align="center", capsize=3)
axes[1, 1].set_yticks(range(len(top_perm)))
axes[1, 1].set_yticklabels(top_perm["feature"])
axes[1, 1].invert_yaxis()
axes[1, 1].set_title("D. Validation Permutation Importance (ΔRMSE mm, N=91,980)", fontsize=13, fontweight="bold")
axes[1, 1].set_xlabel("Increase in Validation RMSE (mm)", fontsize=11)
axes[1, 1].set_ylabel("")

plt.tight_layout()
fig.savefig(fig_dir / "fig1_cross_model_feature_importance.png", dpi=300)
plt.close(fig)
print("Saved fig1_cross_model_feature_importance.png")

# ------------------------------------------------------------------------------
# Figure 2: SHAP Beeswarm Summary Plot
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=300)

plt.sca(axes[0])
shap.summary_plot(shap_tweedie, sample_X_reg, max_display=15, show=False, plot_size=None)
axes[0].set_title("A. LightGBM Tweedie Regressor (SHAP Attribution, mm)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("SHAP Value (Impact on Estimated Daily Rainfall, mm)", fontsize=11)

plt.sca(axes[1])
shap.summary_plot(shap_focal, sample_X_heavy, max_display=15, show=False, plot_size=None)
axes[1].set_title("B. LightGBM Heavy Rain Focal Loss Classifier (Logit Impact)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("SHAP Value (Impact on Heavy Rain Log-Odds Margin)", fontsize=11)

plt.tight_layout()
fig.savefig(fig_dir / "fig2_shap_summary_tweedie_and_heavy_rain.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_shap_summary_tweedie_and_heavy_rain.png")

# ------------------------------------------------------------------------------
# Figure 3: Hydrometeorological Dependence Plots
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
axes = axes.flatten()

dep_configs = [
    ("swvl1_lag_1d", "Antecedent Soil Moisture L1 (m³/m³)", "era5_tp_lag_1d", "ERA5 Precip Lag 1 (mm)", 0),
    ("ro_lag_1d", "Total Runoff Lag 1 (mm)", "swvl1_lag_1d", "Soil Moisture L1 (m³/m³)", 1),
    ("dewpoint_depression_lag_1d", "Dewpoint Depression (°C)", "t2m_mean_lag_1d", "2m Temp (°C)", 2),
    ("wind_speed_lag_1d", "10m Wind Speed Lag 1 (m/s)", "dewpoint_depression_lag_1d", "Dewpoint Dep (°C)", 3),
    ("elevation_dem_m", "Copernicus DEM Elevation (m)", "dist_to_coast_km", "Dist to Coast (km)", 4),
    ("dist_to_coast_km", "Distance to Coast (km)", "elevation_dem_m", "Elevation (m)", 5),
]

for feat, label, col_feat, col_label, idx in dep_configs:
    feat_idx = list(sample_X_reg.columns).index(feat)
    col_idx = list(sample_X_reg.columns).index(col_feat)
    
    x_vals = sample_X_reg[feat].values
    shap_vals = shap_tweedie[:, feat_idx]
    c_vals = sample_X_reg[col_feat].values
    
    sc = axes[idx].scatter(x_vals, shap_vals, c=c_vals, cmap="viridis", alpha=0.6, s=12, edgecolors="none")
    cbar = plt.colorbar(sc, ax=axes[idx])
    cbar.set_label(col_label, fontsize=9)
    
    # Add a LOWESS or running-mean trend line
    df_temp = pd.DataFrame({"x": x_vals, "y": shap_vals}).sort_values("x")
    df_temp["y_smooth"] = df_temp["y"].rolling(window=max(50, len(df_temp)//50), min_periods=10, center=True).mean()
    axes[idx].plot(df_temp["x"], df_temp["y_smooth"], color="crimson", linewidth=2.0, label="Mean Attribution Trend")
    
    axes[idx].set_title(f"{chr(65+idx)}. SHAP Dependence: {label}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel(label, fontsize=10)
    axes[idx].set_ylabel("SHAP Value (Rainfall Impact, mm)", fontsize=10)
    axes[idx].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[idx].legend(loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig(fig_dir / "fig3_hydrometeorological_dependence_plots.png", dpi=300)
plt.close(fig)
print("Saved fig3_hydrometeorological_dependence_plots.png")

print("\n--- Task 1 Completed Successfully ---")

