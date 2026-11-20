import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

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
stage6_dir = base_dir / "Stage6_Robustness_Generalisation"
fig_dir = stage6_dir / "stage6_figures"
stage6_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

print("Loading test dataset and predictions...")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["year"] = df_test["date"].dt.year
df_test["month"] = df_test["date"].dt.month

y_true = df_test["target_rainfall_mm"].values

# Load continuous predictions
models = {
    "LightGBM_Tweedie": np.load(models_dir / "pred_lgb_tweedie_test.npy"),
    "CatBoost_Tweedie": np.load(models_dir / "pred_cat_tweedie_test.npy"),
    "Hurdle_ExpectedValue": np.load(models_dir / "pred_hurdle_ev_test.npy"),
    "Persistence": np.load(models_dir / "pred_baseline_persist_test.npy"),
    "Raw_ERA5": np.load(models_dir / "pred_baseline_era5_test.npy"),
    "EQM": np.load(models_dir / "pred_baseline_eqm_test.npy")
}

# Load classifiers
df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

subdiv_map_heavy = df_train.groupby("subdivision")["target_is_heavy_rain"].mean().to_dict()
zone_map_heavy = df_train.groupby("zone_id")["target_is_heavy_rain"].mean().to_dict()

X_te_heavy = df_test[num_cols].copy()
X_te_heavy["subdivision_enc"] = df_test["subdivision"].map(subdiv_map_heavy).fillna(0.007).values
X_te_heavy["zone_id_enc"] = df_test["zone_id"].map(zone_map_heavy).fillna(0.007).values

m_lgb_focal = joblib.load(models_dir / "lgb_heavy_focal.joblib")
m_lgb_weighted = joblib.load(models_dir / "lgb_heavy_weighted.joblib")
m_cat_weighted = joblib.load(models_dir / "cat_heavy_weighted.joblib")

z_focal = m_lgb_focal.predict(X_te_heavy, raw_score=True)
p_focal = 1.0 / (1.0 + np.exp(-z_focal))
p_lgb_weighted = m_lgb_weighted.predict_proba(X_te_heavy)[:, 1]
p_cat_weighted = m_cat_weighted.predict_proba(X_te_heavy)[:, 1]

# Frozen decision thresholds
thresh_focal = 0.53
thresh_lgb_wt = 0.96
thresh_cat_wt = 0.98

classifiers_pred_binary = {
    "LightGBM_Focal_Loss (p*=0.53)": (p_focal >= thresh_focal).astype(int),
    "LightGBM_Class_Weighted (p*=0.96)": (p_lgb_weighted >= thresh_lgb_wt).astype(int),
    "CatBoost_Class_Weighted (p*=0.98)": (p_cat_weighted >= thresh_cat_wt).astype(int),
    "LightGBM_Tweedie (Direct Thresh)": None, # Will be set per threshold
    "Persistence (Direct Thresh)": None,
    "Raw_ERA5 (Direct Thresh)": None,
    "EQM (Direct Thresh)": None
}

def compute_regression_metrics(y_obs, y_hat):
    y_hat = np.clip(y_hat, 0.0, None)
    n = len(y_obs)
    if n == 0:
        return {}
    
    mae = float(np.mean(np.abs(y_hat - y_obs)))
    rmse = float(np.sqrt(np.mean((y_hat - y_obs) ** 2)))
    bias = float(np.mean(y_hat - y_obs))
    
    mu_o = float(np.mean(y_obs))
    mu_p = float(np.mean(y_hat))
    std_o = float(np.std(y_obs))
    std_p = float(np.std(y_hat))
    
    pbias = float(100.0 * bias / mu_o) if mu_o > 1e-6 else 0.0
    r = float(np.corrcoef(y_obs, y_hat)[0, 1]) if (std_o > 1e-6 and std_p > 1e-6) else 0.0
    
    denom_nse = float(np.sum((y_obs - mu_o) ** 2))
    nse = float(1.0 - (np.sum((y_hat - y_obs) ** 2) / denom_nse)) if denom_nse > 1e-6 else -999.0
    
    beta = mu_p / mu_o if mu_o > 1e-6 else 1.0
    alpha = std_p / std_o if std_o > 1e-6 else 1.0
    kge_2009 = float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)) if (std_o > 1e-6 and std_p > 1e-6) else -999.0
    
    cv_o = std_o / mu_o if mu_o > 1e-6 else 1.0
    cv_p = std_p / mu_p if mu_p > 1e-6 else 1.0
    gamma = cv_p / cv_o if cv_o > 1e-6 else 1.0
    kge_2012 = float(1.0 - np.sqrt((r - 1.0)**2 + (gamma - 1.0)**2 + (beta - 1.0)**2)) if (std_o > 1e-6 and std_p > 1e-6) else -999.0
    
    return {
        "sample_count": n,
        "obs_mean": round(mu_o, 4),
        "pred_mean": round(mu_p, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Bias": round(bias, 4),
        "PBias_pct": round(pbias, 2),
        "Pearson_r": round(r, 4),
        "NSE": round(nse, 4),
        "KGE_2009": round(kge_2009, 4),
        "KGE_2012": round(kge_2012, 4)
    }

def compute_contingency(obs_bin, pred_bin):
    h = int(np.sum((obs_bin == 1) & (pred_bin == 1)))
    m = int(np.sum((obs_bin == 1) & (pred_bin == 0)))
    f = int(np.sum((obs_bin == 0) & (pred_bin == 1)))
    c = int(np.sum((obs_bin == 0) & (pred_bin == 0)))
    n = len(obs_bin)
    
    prec = float(h / (h + f)) if (h + f) > 0 else 0.0
    pod = float(h / (h + m)) if (h + m) > 0 else 0.0
    far = float(f / (h + f)) if (h + f) > 0 else 0.0
    csi = float(h / (h + m + f)) if (h + m + f) > 0 else 0.0
    
    h_rand = ((h + m) * (h + f)) / n if n > 0 else 0.0
    denom_ets = (h + m + f - h_rand)
    ets = float((h - h_rand) / denom_ets) if denom_ets > 0 else 0.0
    
    hss_num = 2.0 * (h * c - f * m)
    hss_den = ((h + m) * (m + c) + (h + f) * (f + c))
    hss = float(hss_num / hss_den) if hss_den > 0 else 0.0
    
    return {
        "Total_Events": int(np.sum(obs_bin == 1)),
        "Predicted_Events": int(np.sum(pred_bin == 1)),
        "Hits": h,
        "Misses": m,
        "False_Alarms": f,
        "Correct_Negatives": c,
        "Precision": round(prec, 4),
        "POD_Recall": round(pod, 4),
        "FAR": round(far, 4),
        "CSI": round(csi, 4),
        "HSS": round(hss, 4),
        "ETS": round(ets, 4)
    }

# ==============================================================================
# 1. TEMPORAL GENERALISATION (YEARLY & YEAR-SEASONAL BREAKDOWN)
# ==============================================================================
print("\n--- 1. Evaluating Yearly and Seasonal Consistency ---")
temporal_strata = {
    "Year_2024_Full": (df_test["year"] == 2024).values,
    "Year_2025_Full": (df_test["year"] == 2025).values,
    "2024_Monsoon_JJAS": ((df_test["year"] == 2024) & (df_test["month"].isin([6, 7, 8, 9]))).values,
    "2025_Monsoon_JJAS": ((df_test["year"] == 2025) & (df_test["month"].isin([6, 7, 8, 9]))).values,
    "2024_PostMonsoon_OND": ((df_test["year"] == 2024) & (df_test["month"].isin([10, 11, 12]))).values,
    "2025_PostMonsoon_OND": ((df_test["year"] == 2025) & (df_test["month"].isin([10, 11, 12]))).values,
    "2024_PreMonsoon_MAM": ((df_test["year"] == 2024) & (df_test["month"].isin([3, 4, 5]))).values,
    "2025_PreMonsoon_MAM": ((df_test["year"] == 2025) & (df_test["month"].isin([3, 4, 5]))).values,
    "2024_Winter_JF": ((df_test["year"] == 2024) & (df_test["month"].isin([1, 2]))).values,
    "2025_Winter_JF": ((df_test["year"] == 2025) & (df_test["month"].isin([1, 2]))).values,
    "Test_Period_Overall": np.ones(len(df_test), dtype=bool)
}

yearly_records = []
for t_name, mask in temporal_strata.items():
    y_sub = y_true[mask]
    for m_name, y_pred in models.items():
        yp_sub = y_pred[mask]
        met = compute_regression_metrics(y_sub, yp_sub)
        yearly_records.append({
            "temporal_stratum": t_name,
            "model": m_name,
            **met
        })

df_yearly = pd.DataFrame(yearly_records)
df_yearly.to_csv(stage6_dir / "yearly_performance_results.csv", index=False)
print("Saved yearly_performance_results.csv")

# ==============================================================================
# 2. RAINFALL INTENSITY SENSITIVITY
# ==============================================================================
print("\n--- 2. Evaluating Rainfall Intensity Sensitivity Across 7 IMD Bins ---")
intensity_bins = [
    ("No Rain (<0.1 mm)", -0.01, 0.1),
    ("Very Light (0.1–2.4 mm)", 0.1, 2.5),
    ("Light (2.5–15.5 mm)", 2.5, 15.6),
    ("Moderate (15.6–64.4 mm)", 15.6, 64.5),
    ("Heavy (64.5–115.5 mm)", 64.5, 115.6),
    ("Very Heavy (115.6–204.4 mm)", 115.6, 204.5),
    ("Extremely Heavy (≥204.5 mm)", 204.5, 10000.0)
]

intensity_records = []
for cat_name, low, high in intensity_bins:
    mask = (y_true >= low) & (y_true < high)
    n_cat = int(np.sum(mask))
    if n_cat == 0:
        continue
    y_sub = y_true[mask]
    obs_m = float(np.mean(y_sub))
    obs_s = float(np.std(y_sub))
    
    for m_name, y_pred in models.items():
        yp_sub = np.clip(y_pred[mask], 0.0, None)
        pred_m = float(np.mean(yp_sub))
        pred_s = float(np.std(yp_sub))
        mae = float(np.mean(np.abs(yp_sub - y_sub)))
        rmse = float(np.sqrt(np.mean((yp_sub - y_sub)**2)))
        bias = float(np.mean(yp_sub - y_sub))
        pbias = float(100.0 * bias / obs_m) if obs_m > 1e-4 else 0.0
        
        # Percentage of observations underestimated
        pct_under = float(100.0 * np.sum(yp_sub < y_sub) / n_cat)
        
        intensity_records.append({
            "category": cat_name,
            "sample_count": n_cat,
            "pct_total_dataset": round(100.0 * n_cat / len(y_true), 2),
            "obs_mean_mm": round(obs_m, 2),
            "obs_std_mm": round(obs_s, 2),
            "model": m_name,
            "pred_mean_mm": round(pred_m, 2),
            "pred_std_mm": round(pred_s, 2),
            "MAE_mm": round(mae, 2),
            "RMSE_mm": round(rmse, 2),
            "Mean_Bias_mm": round(bias, 2),
            "PBias_pct": round(pbias, 1),
            "Percent_Underestimated": round(pct_under, 1)
        })

df_intensity = pd.DataFrame(intensity_records)
df_intensity.to_csv(stage6_dir / "rainfall_intensity_sensitivity.csv", index=False)
print("Saved rainfall_intensity_sensitivity.csv")

# ==============================================================================
# 3. THRESHOLD SENSITIVITY (35.5, 64.5, 115.6, 204.5 mm)
# ==============================================================================
print("\n--- 3. Evaluating Multi-Threshold Sensitivity ---")
eval_thresholds = [
    ("Custom_Diagnostic_35.5mm", 35.5),
    ("Official_Heavy_Rain_64.5mm", 64.5),
    ("Official_Very_Heavy_Rain_115.6mm", 115.6),
    ("Official_Extremely_Heavy_Rain_204.5mm", 204.5)
]

threshold_records = []
for t_label, thresh_val in eval_thresholds:
    obs_bin = (y_true >= thresh_val).astype(int)
    
    # 1. Classifiers (Focal Loss, Class Weighted LGBM, Class Weighted CatBoost)
    # Note: Focal loss and class weighted models were trained at the 64.5 mm threshold in Stage 4.1.
    # At 64.5 mm, we use the frozen calibrated thresholds (0.53, 0.96, 0.98).
    # For other thresholds, we evaluate the continuous regressors and baselines directly.
    # For 64.5 mm, we also report the classifiers.
    if thresh_val == 64.5:
        c_models = {
            "LightGBM_Focal_Loss (p*=0.53)": (p_focal >= thresh_focal).astype(int),
            "LightGBM_Class_Weighted (p*=0.96)": (p_lgb_weighted >= thresh_lgb_wt).astype(int),
            "CatBoost_Class_Weighted (p*=0.98)": (p_cat_weighted >= thresh_cat_wt).astype(int),
        }
        for cm_name, p_b in c_models.items():
            stats = compute_contingency(obs_bin, p_b)
            threshold_records.append({
                "threshold_label": t_label,
                "threshold_mm": thresh_val,
                "model": cm_name,
                "model_type": "Binary_Classifier",
                **stats
            })
            
    # Continuous models thresholded at thresh_val
    for m_name in ["LightGBM_Tweedie", "CatBoost_Tweedie", "Hurdle_ExpectedValue", "Persistence", "Raw_ERA5", "EQM"]:
        pred_b = (models[m_name] >= thresh_val).astype(int)
        stats = compute_contingency(obs_bin, pred_b)
        threshold_records.append({
            "threshold_label": t_label,
            "threshold_mm": thresh_val,
            "model": m_name,
            "model_type": "Continuous_Thresholded",
            **stats
        })

df_threshold = pd.DataFrame(threshold_records)
df_threshold.to_csv(stage6_dir / "threshold_sensitivity_results.csv", index=False)
print("Saved threshold_sensitivity_results.csv")

# ==============================================================================
# 4. REGIONAL ROBUSTNESS ACROSS SUBDIVISIONS & FALLBACK CELLS
# ==============================================================================
print("\n--- 4. Evaluating Regional Robustness and Terrain Coupling ---")
regional_strata = {
    "Coastal_Karnataka_All_13_cells": (df_test["subdivision"] == "Coastal Karnataka").values,
    "Malnad_35_cells": (df_test["subdivision"] == "Malnad").values,
    "North_Interior_Karnataka_118_cells": (df_test["subdivision"] == "North Interior Karnataka").values,
    "South_Interior_Karnataka_86_cells": (df_test["subdivision"] == "South Interior Karnataka").values,
    "Coastal_Fallback_Only_3_cells_IDs_52_68_125": (df_test["is_coastal_fallback"] == 1).values,
    "Coastal_Native_Land_Only_10_cells": ((df_test["subdivision"] == "Coastal Karnataka") & (df_test["is_coastal_fallback"] == 0)).values,
    "Statewide_All_252_cells": np.ones(len(df_test), dtype=bool)
}

regional_records = []
for r_name, mask in regional_strata.items():
    y_sub = y_true[mask]
    n_sub = len(y_sub)
    
    # Subgroup geographic characteristics
    sub_df = df_test[mask]
    mean_elev = float(sub_df["elevation_dem_m"].mean())
    std_elev = float(sub_df["elevation_dem_m"].std())
    mean_dist_coast = float(sub_df["dist_to_coast_km"].mean())
    heavy_count = int(np.sum(y_sub >= 64.5))
    heavy_pct = round(100.0 * heavy_count / n_sub, 3)
    
    for m_name, y_pred in models.items():
        yp_sub = y_pred[mask]
        met = compute_regression_metrics(y_sub, yp_sub)
        
        # Heavy rain contingency for this region
        cont_64 = compute_contingency((y_sub >= 64.5).astype(int), (yp_sub >= 64.5).astype(int))
        
        regional_records.append({
            "region": r_name,
            "sample_count": n_sub,
            "mean_elevation_m": round(mean_elev, 1),
            "std_elevation_m": round(std_elev, 1),
            "mean_dist_coast_km": round(mean_dist_coast, 1),
            "heavy_events_obs": heavy_count,
            "heavy_events_pct": heavy_pct,
            "model": m_name,
            **met,
            "CSI_64.5mm": cont_64["CSI"],
            "POD_64.5mm": cont_64["POD_Recall"],
            "FAR_64.5mm": cont_64["FAR"]
        })

df_regional = pd.DataFrame(regional_records)
df_regional.to_csv(stage6_dir / "regional_robustness_results.csv", index=False)
print("Saved regional_robustness_results.csv")

# ==============================================================================
# 5. PUBLICATION FIGURES: FIG 1, 2, 3, 4
# ==============================================================================
print("\n--- 5. Generating Publication Figures (Fig 1, 2, 3, 4) ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------------------------
# Figure 1: Temporal Consistency (2024 vs 2025 Yearly and Seasonal Breakdowns)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

yearly_main = df_yearly[df_yearly["temporal_stratum"].isin(["Year_2024_Full", "Year_2025_Full"])].copy()
yearly_main["Year"] = yearly_main["temporal_stratum"].str.extract(r"Year_(\d+)_Full")

# 1a. Annual RMSE by Year
sns.barplot(data=yearly_main, x="Year", y="RMSE", hue="model", ax=axes[0, 0], palette="tab10")
axes[0, 0].set_title("A. Annual RMSE Comparison (2024 vs. 2025)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("RMSE (mm)", fontsize=11)
axes[0, 0].set_xlabel("Calendar Year", fontsize=11)
axes[0, 0].get_legend().remove()

# 1b. Annual KGE-2012 by Year
sns.barplot(data=yearly_main, x="Year", y="KGE_2012", hue="model", ax=axes[0, 1], palette="tab10")
axes[0, 1].set_title("B. Modified Kling-Gupta Efficiency (KGE-2012)", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("KGE-2012", fontsize=11)
axes[0, 1].set_xlabel("Calendar Year", fontsize=11)
axes[0, 1].get_legend().remove()

# 1c. Seasonal RMSE in 2024 vs 2025 (LightGBM Tweedie vs Persistence vs Raw ERA5)
seas_compare = df_yearly[df_yearly["temporal_stratum"].str.contains("Monsoon|Winter") & 
                         df_yearly["model"].isin(["LightGBM_Tweedie", "Persistence", "Raw_ERA5"])].copy()
seas_compare["Season_Clean"] = seas_compare["temporal_stratum"].apply(lambda s: s[5:])
seas_compare["Year"] = seas_compare["temporal_stratum"].apply(lambda s: s[:4])

sns.barplot(data=seas_compare[seas_compare["model"] == "LightGBM_Tweedie"],
            x="Season_Clean", y="RMSE", hue="Year", ax=axes[1, 0], palette=["#2a9d8f", "#e76f51"])
axes[1, 0].set_title("C. LightGBM Tweedie Seasonal RMSE (2024 vs. 2025)", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("RMSE (mm)", fontsize=11)
axes[1, 0].set_xlabel("Season", fontsize=11)

# 1d. Seasonal Pearson Correlation (2024 vs 2025)
sns.barplot(data=seas_compare[seas_compare["model"] == "LightGBM_Tweedie"],
            x="Season_Clean", y="Pearson_r", hue="Year", ax=axes[1, 1], palette=["#2a9d8f", "#e76f51"])
axes[1, 1].set_title("D. LightGBM Tweedie Pearson Correlation (2024 vs. 2025)", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Correlation (r)", fontsize=11)
axes[1, 1].set_xlabel("Season", fontsize=11)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
fig.savefig(fig_dir / "fig1_temporal_consistency_yearly_seasonal.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_temporal_consistency_yearly_seasonal.png")

# ------------------------------------------------------------------------------
# Figure 2: Rainfall Intensity Error Sensitivity
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

df_int_plot = df_intensity[df_intensity["model"].isin(["LightGBM_Tweedie", "Raw_ERA5", "EQM"])].copy()
cat_order = [c[0] for c in intensity_bins]

# 2a. Mean Bias by Intensity Category
sns.barplot(data=df_int_plot, x="category", y="Mean_Bias_mm", hue="model",
            order=cat_order, ax=axes[0], palette=["#1d3557", "#e63946", "#457b9d"])
axes[0].set_title("A. Conditional Mean Bias across 7 IMD Intensity Categories", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Mean Bias (mm)", fontsize=11)
axes[0].set_xlabel("IMD Rainfall Intensity Category", fontsize=11)
axes[0].tick_params(axis='x', rotation=25)
axes[0].axhline(0, color="black", linestyle="-", linewidth=0.8)
axes[0].legend(fontsize=9)

# 2b. Percentage Underestimated
sns.barplot(data=df_int_plot, x="category", y="Percent_Underestimated", hue="model",
            order=cat_order, ax=axes[1], palette=["#1d3557", "#e63946", "#457b9d"])
axes[1].set_title("B. Proportion of Instances Underestimated (% of bin count)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Underestimated Observations (%)", fontsize=11)
axes[1].set_xlabel("IMD Rainfall Intensity Category", fontsize=11)
axes[1].tick_params(axis='x', rotation=25)
axes[1].axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% Parity Line")
axes[1].legend(fontsize=9)

plt.tight_layout()
fig.savefig(fig_dir / "fig2_rainfall_intensity_error_distribution.png", dpi=300)
plt.close(fig)
print("Saved fig2_rainfall_intensity_error_distribution.png")

# ------------------------------------------------------------------------------
# Figure 3: Heavy Rain Threshold Sensitivity
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

df_thresh_plot = df_threshold[df_threshold["model"].isin([
    "LightGBM_Tweedie", "Hurdle_ExpectedValue", "Raw_ERA5", "EQM", "LightGBM_Focal_Loss (p*=0.53)"
])].copy()

# 3a. CSI across Thresholds
thresh_labels = ["35.5 mm", "64.5 mm", "115.6 mm", "204.5 mm"]
t_vals = [35.5, 64.5, 115.6, 204.5]

for m_name, color, style in [
    ("LightGBM_Tweedie", "#1d3557", "-o"),
    ("Hurdle_ExpectedValue", "#2a9d8f", "-s"),
    ("Raw_ERA5", "#e76f51", "--^"),
    ("EQM", "#457b9d", ":d")
]:
    sub = df_thresh_plot[df_thresh_plot["model"] == m_name]
    axes[0].plot(range(len(t_vals)), sub["CSI"], style, color=color, linewidth=2.0, label=m_name)

# Overlay Focal Loss at 64.5 mm
focal_csi = df_threshold[(df_threshold["model"].str.startswith("LightGBM_Focal")) & (df_threshold["threshold_mm"] == 64.5)]["CSI"].values
if len(focal_csi) > 0:
    axes[0].scatter([1], focal_csi, color="crimson", s=150, zorder=5, label=f"LightGBM Focal Loss (CSI={focal_csi[0]:.4f})")

axes[0].set_xticks(range(len(t_vals)))
axes[0].set_xticklabels(thresh_labels, fontsize=10)
axes[0].set_title("A. Critical Success Index (CSI) across Rainfall Thresholds", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Operational Threshold", fontsize=11)
axes[0].set_ylabel("Critical Success Index (CSI)", fontsize=11)
axes[0].set_ylim(0, 0.45)
axes[0].legend(fontsize=9)

# 3b. Precision vs Recall Trade-off across Thresholds
for m_name, color in [("LightGBM_Tweedie", "#1d3557"), ("Raw_ERA5", "#e76f51"), ("EQM", "#457b9d")]:
    sub = df_thresh_plot[df_thresh_plot["model"] == m_name]
    axes[1].plot(sub["POD_Recall"], sub["Precision"], "-o", color=color, linewidth=2.0, label=m_name)
    for _, r in sub.iterrows():
        axes[1].annotate(f"{r['threshold_mm']}mm", (r["POD_Recall"], r["Precision"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)

# Add Focal Loss point
focal_row = df_threshold[(df_threshold["model"].str.startswith("LightGBM_Focal")) & (df_threshold["threshold_mm"] == 64.5)]
if len(focal_row) > 0:
    axes[1].scatter(focal_row["POD_Recall"], focal_row["Precision"], color="crimson", s=150, zorder=5,
                    label="Focal Loss (64.5mm, P=0.58, R=0.47)")

axes[1].set_title("B. Precision vs. Recall across Rainfall Thresholds", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Recall / Probability of Detection (POD)", fontsize=11)
axes[1].set_ylabel("Precision (1 - FAR)", fontsize=11)
axes[1].set_xlim(0, 0.7)
axes[1].set_ylim(0, 0.8)
axes[1].legend(fontsize=9, loc="upper right")

plt.tight_layout()
fig.savefig(fig_dir / "fig3_heavy_rain_threshold_sensitivity.png", dpi=300)
plt.close(fig)
print("Saved fig3_heavy_rain_threshold_sensitivity.png")

# ------------------------------------------------------------------------------
# Figure 4: Regional Robustness Metrics
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

plot_regs = [
    "Coastal_Karnataka_All_13_cells",
    "Malnad_35_cells",
    "North_Interior_Karnataka_118_cells",
    "South_Interior_Karnataka_86_cells",
    "Coastal_Fallback_Only_3_cells_IDs_52_68_125"
]
plot_labels = [
    "Coastal (13 cells)",
    "Malnad (35 cells)",
    "North Interior (118 cells)",
    "South Interior (86 cells)",
    "Coastal Fallback (3 cells)"
]
df_reg_p = df_regional[df_regional["region"].isin(plot_regs) & 
                       df_regional["model"].isin(["LightGBM_Tweedie", "Persistence", "Raw_ERA5"])].copy()
df_reg_p["Region_Label"] = df_reg_p["region"].map(dict(zip(plot_regs, plot_labels)))

# 4a. Regional RMSE
sns.barplot(data=df_reg_p, x="Region_Label", y="RMSE", hue="model", ax=axes[0], palette="tab10")
axes[0].set_title("A. Regional RMSE Comparison", fontsize=12, fontweight="bold")
axes[0].set_ylabel("RMSE (mm)", fontsize=11)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', rotation=15)
axes[0].legend(fontsize=9)

# 4b. Regional KGE-2012
sns.barplot(data=df_reg_p, x="Region_Label", y="KGE_2012", hue="model", ax=axes[1], palette="tab10")
axes[1].set_title("B. Regional Modified KGE-2012 Comparison", fontsize=12, fontweight="bold")
axes[1].set_ylabel("KGE-2012", fontsize=11)
axes[1].set_xlabel("")
axes[1].tick_params(axis='x', rotation=15)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].legend(fontsize=9)

plt.tight_layout()
fig.savefig(fig_dir / "fig4_regional_robustness_metrics.png", dpi=300)
plt.close(fig)
print("Saved fig4_regional_robustness_metrics.png")

print("\n--- Task 1 Completed Successfully ---")

