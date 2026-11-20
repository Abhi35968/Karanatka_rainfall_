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
stage5_dir = base_dir / "Stage5_Model_Interpretation"
fig_dir = stage5_dir / "stage5_figures"

print("Loading test datasets and feature manifest...")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["month"] = df_test["date"].dt.month

y_true = df_test["target_rainfall_mm"].values
y_true_heavy = (y_true >= 64.5).astype(int)

df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

subdiv_map_heavy = df_train.groupby("subdivision")["target_is_heavy_rain"].mean().to_dict()
zone_map_heavy = df_train.groupby("zone_id")["target_is_heavy_rain"].mean().to_dict()

X_te_heavy = df_test[num_cols].copy()
X_te_heavy["subdivision_enc"] = df_test["subdivision"].map(subdiv_map_heavy).fillna(0.007).values
X_te_heavy["zone_id_enc"] = df_test["zone_id"].map(zone_map_heavy).fillna(0.007).values

# Load classifiers and threshold predictions
print("Loading classifiers...")
m_lgb_focal = joblib.load(models_dir / "lgb_heavy_focal.joblib")
m_lgb_weighted = joblib.load(models_dir / "lgb_heavy_weighted.joblib")
m_cat_weighted = joblib.load(models_dir / "cat_heavy_weighted.joblib")

# Frozen decision thresholds calibrated strictly on 2023 Validation set in Stage 4.1
thresh_lgb_focal = 0.53
thresh_lgb_weighted = 0.96
thresh_cat_weighted = 0.98

# Get predicted probabilities
# Custom objective LightGBM outputs raw margin z -> apply sigmoid
z_focal = m_lgb_focal.predict(X_te_heavy, raw_score=True)
p_focal = 1.0 / (1.0 + np.exp(-z_focal))

p_lgb_weighted = m_lgb_weighted.predict_proba(X_te_heavy)[:, 1]
p_cat_weighted = m_cat_weighted.predict_proba(X_te_heavy)[:, 1]

# Binary predictions
pred_bin_focal = (p_focal >= thresh_lgb_focal).astype(int)
pred_bin_lgb_wt = (p_lgb_weighted >= thresh_lgb_weighted).astype(int)
pred_bin_cat_wt = (p_cat_weighted >= thresh_cat_weighted).astype(int)

# Continuous regressor predictions
pred_lgb_tweedie = np.load(models_dir / "pred_lgb_tweedie_test.npy")
pred_cat_tweedie = np.load(models_dir / "pred_cat_tweedie_test.npy")
pred_hurdle_ev = np.load(models_dir / "pred_hurdle_ev_test.npy")
pred_persist = np.load(models_dir / "pred_baseline_persist_test.npy")
pred_era5 = np.load(models_dir / "pred_baseline_era5_test.npy")
pred_eqm = np.load(models_dir / "pred_baseline_eqm_test.npy")

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
    
    # ETS (Equitable Threat Score)
    h_random = ((h + m) * (h + f)) / n if n > 0 else 0.0
    denom_ets = (h + m + f - h_random)
    ets = float((h - h_random) / denom_ets) if denom_ets > 0 else 0.0

    # HSS (Heidke Skill Score)
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
# 1. HEAVY RAIN EVENT ANALYSIS (DELIVERABLE 7)
# ==============================================================================
print("\n--- 1. Generating Heavy Rain Event Analysis Table ---")
heavy_models_eval = {
    "LightGBM_Focal_Loss (p*=0.53)": pred_bin_focal,
    "LightGBM_Class_Weighted (p*=0.96)": pred_bin_lgb_wt,
    "CatBoost_Class_Weighted (p*=0.98)": pred_bin_cat_wt,
    "LightGBM_Tweedie (Rain >= 64.5mm)": (pred_lgb_tweedie >= 64.5).astype(int),
    "CatBoost_Tweedie (Rain >= 64.5mm)": (pred_cat_tweedie >= 64.5).astype(int),
    "Hurdle_ExpectedValue (Rain >= 64.5mm)": (pred_hurdle_ev >= 64.5).astype(int),
    "Persistence (Rain >= 64.5mm)": (pred_persist >= 64.5).astype(int),
    "Raw_ERA5 (Precip >= 64.5mm)": (pred_era5 >= 64.5).astype(int),
    "EQM (Bias-Corrected >= 64.5mm)": (pred_eqm >= 64.5).astype(int)
}

heavy_event_rows = []

# Statewide overall
for m_label, p_bin in heavy_models_eval.items():
    stats = compute_contingency(y_true_heavy, p_bin)
    heavy_event_rows.append({
        "Stratum": "Statewide_All (184,212 instances)",
        "Model": m_label,
        **stats
    })

# Subdivisions for champion focal loss vs weighted vs baselines
subdiv_names = ["Coastal Karnataka", "Malnad", "North Interior Karnataka", "South Interior Karnataka"]
for subdiv in subdiv_names:
    mask = (df_test["subdivision"] == subdiv).values
    for m_label in ["LightGBM_Focal_Loss (p*=0.53)", "LightGBM_Class_Weighted (p*=0.96)", "CatBoost_Class_Weighted (p*=0.98)", "Raw_ERA5 (Precip >= 64.5mm)", "EQM (Bias-Corrected >= 64.5mm)"]:
        stats = compute_contingency(y_true_heavy[mask], heavy_models_eval[m_label][mask])
        heavy_event_rows.append({
            "Stratum": f"Subdivision: {subdiv}",
            "Model": m_label,
            **stats
        })

# Coastal Fallback vs Non-Fallback
mask_fb = (df_test["is_coastal_fallback"] == 1).values
mask_nonfb_coastal = ((df_test["subdivision"] == "Coastal Karnataka") & (df_test["is_coastal_fallback"] == 0)).values

for m_label in ["LightGBM_Focal_Loss (p*=0.53)", "LightGBM_Class_Weighted (p*=0.96)", "Raw_ERA5 (Precip >= 64.5mm)"]:
    stats_fb = compute_contingency(y_true_heavy[mask_fb], heavy_models_eval[m_label][mask_fb])
    heavy_event_rows.append({
        "Stratum": "Coastal_Fallback_Only_3_cells (2,193 instances)",
        "Model": m_label,
        **stats_fb
    })
    stats_nonfb = compute_contingency(y_true_heavy[mask_nonfb_coastal], heavy_models_eval[m_label][mask_nonfb_coastal])
    heavy_event_rows.append({
        "Stratum": "Coastal_Native_Land_10_cells (7,310 instances)",
        "Model": m_label,
        **stats_nonfb
    })

# Seasons
season_names = {
    "Monsoon_JJAS": df_test["month"].isin([6, 7, 8, 9]).values,
    "PostMonsoon_OND": df_test["month"].isin([10, 11, 12]).values,
    "PreMonsoon_MAM": df_test["month"].isin([3, 4, 5]).values,
    "Winter_JF": df_test["month"].isin([1, 2]).values
}
for s_label, mask in season_names.items():
    for m_label in ["LightGBM_Focal_Loss (p*=0.53)", "LightGBM_Class_Weighted (p*=0.96)", "CatBoost_Class_Weighted (p*=0.98)", "Raw_ERA5 (Precip >= 64.5mm)"]:
        stats = compute_contingency(y_true_heavy[mask], heavy_models_eval[m_label][mask])
        heavy_event_rows.append({
            "Stratum": f"Season: {s_label}",
            "Model": m_label,
            **stats
        })

df_heavy_res = pd.DataFrame(heavy_event_rows)
df_heavy_res.to_csv(stage5_dir / "heavy_rain_event_analysis.csv", index=False)
print("Saved heavy_rain_event_analysis.csv")

# ==============================================================================
# 2. PREDICTION ERROR ANALYSIS (DELIVERABLE 8)
# ==============================================================================
print("\n--- 2. Generating Prediction Error Analysis Table ---")
# Standard IMD Rainfall Intensity Categories
intensity_bins = [
    ("No Rain (<0.1 mm)", -0.01, 0.1),
    ("Very Light (0.1–2.4 mm)", 0.1, 2.5),
    ("Light (2.5–15.5 mm)", 2.5, 15.6),
    ("Moderate (15.6–64.4 mm)", 15.6, 64.5),
    ("Heavy (64.5–115.5 mm)", 64.5, 115.6),
    ("Very Heavy (115.6–204.4 mm)", 115.6, 204.5),
    ("Extremely Heavy (≥204.5 mm)", 204.5, 10000.0)
]

error_rows = []
for cat_name, low, high in intensity_bins:
    mask = (y_true >= low) & (y_true < high)
    n_cat = int(np.sum(mask))
    pct_cat = 100.0 * n_cat / len(y_true)
    
    if n_cat == 0:
        continue
        
    y_sub = y_true[mask]
    obs_mean = float(np.mean(y_sub))
    obs_std = float(np.std(y_sub))
    
    for m_label, y_pred in [
        ("LightGBM_Tweedie", pred_lgb_tweedie),
        ("CatBoost_Tweedie", pred_cat_tweedie),
        ("Hurdle_ExpectedValue", pred_hurdle_ev),
        ("Raw_ERA5", pred_era5),
        ("EQM", pred_eqm),
        ("Persistence", pred_persist)
    ]:
        yp_sub = np.clip(y_pred[mask], 0.0, None)
        pred_mean = float(np.mean(yp_sub))
        pred_std = float(np.std(yp_sub))
        mae = float(np.mean(np.abs(yp_sub - y_sub)))
        rmse = float(np.sqrt(np.mean((yp_sub - y_sub) ** 2)))
        bias = float(np.mean(yp_sub - y_sub))
        pbias = float(100.0 * bias / obs_mean) if obs_mean > 1e-4 else 0.0
        
        error_rows.append({
            "Intensity_Category": cat_name,
            "Sample_Count": n_cat,
            "Percent_Test_Data": round(pct_cat, 2),
            "Observed_Mean_mm": round(obs_mean, 2),
            "Observed_Std_mm": round(obs_std, 2),
            "Model": m_label,
            "Predicted_Mean_mm": round(pred_mean, 2),
            "Predicted_Std_mm": round(pred_std, 2),
            "MAE_mm": round(mae, 2),
            "RMSE_mm": round(rmse, 2),
            "Mean_Bias_mm": round(bias, 2),
            "PBias_pct": round(pbias, 1)
        })

df_err_res = pd.DataFrame(error_rows)
df_err_res.to_csv(stage5_dir / "prediction_error_analysis.csv", index=False)
print("Saved prediction_error_analysis.csv")

# Compute Quantile Summary Comparison
quantiles = [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
q_obs = np.quantile(y_true, quantiles)
q_lgb = np.quantile(pred_lgb_tweedie, quantiles)
q_cat = np.quantile(pred_cat_tweedie, quantiles)
q_hurdle = np.quantile(pred_hurdle_ev, quantiles)
q_era5 = np.quantile(pred_era5, quantiles)
q_eqm = np.quantile(pred_eqm, quantiles)

df_quantiles = pd.DataFrame({
    "Quantile": [f"{int(q*100) if q*100 == int(q*100) else q*100}th Percentile" for q in quantiles],
    "Observed_IMD_mm": np.round(q_obs, 2),
    "LightGBM_Tweedie_mm": np.round(q_lgb, 2),
    "CatBoost_Tweedie_mm": np.round(q_cat, 2),
    "Hurdle_ExpectedValue_mm": np.round(q_hurdle, 2),
    "Raw_ERA5_mm": np.round(q_era5, 2),
    "EQM_mm": np.round(q_eqm, 2)
})
df_quantiles.to_csv(stage5_dir / "observed_vs_predicted_quantiles.csv", index=False)
print("Saved observed_vs_predicted_quantiles.csv")

# Grid-cell level spatial error summary across 252 IMD grid cells
station_summary = []
for sid, grp in df_test.groupby("station_id"):
    idx = grp.index.values
    y_s = y_true[idx]
    yp_s = pred_lgb_tweedie[idx]
    
    station_summary.append({
        "station_id": int(sid),
        "latitude": grp["latitude"].iloc[0],
        "longitude": grp["longitude"].iloc[0],
        "subdivision": grp["subdivision"].iloc[0],
        "elevation_dem_m": grp["elevation_dem_m"].iloc[0],
        "dist_to_coast_km": grp["dist_to_coast_km"].iloc[0],
        "is_coastal_fallback": grp["is_coastal_fallback"].iloc[0],
        "sample_count": len(y_s),
        "obs_mean": float(np.mean(y_s)),
        "pred_mean": float(np.mean(yp_s)),
        "RMSE": float(np.sqrt(np.mean((yp_s - y_s)**2))),
        "MAE": float(np.mean(np.abs(yp_s - y_s))),
        "Bias": float(np.mean(yp_s - y_s)),
        "Pearson_r": float(np.corrcoef(y_s, yp_s)[0, 1]) if np.std(y_s) > 1e-6 and np.std(yp_s) > 1e-6 else 0.0,
        "Heavy_Events_Obs": int(np.sum(y_s >= 64.5)),
        "Heavy_Events_Pred": int(np.sum(yp_s >= 64.5))
    })
df_station_err = pd.DataFrame(station_summary)
df_station_err.to_csv(stage5_dir / "grid_cell_spatial_error_summary.csv", index=False)
print("Saved grid_cell_spatial_error_summary.csv")

# ==============================================================================
# 3. PUBLICATION FIGURES: FIG 6, 7, 8, 9
# ==============================================================================
print("\n--- 3. Generating Figures 6, 7, 8, 9 ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------------------------
# Figure 6: Heavy Rain Contingency & Detection
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

models_fig6 = [
    "LightGBM Focal Loss",
    "LightGBM Weighted",
    "CatBoost Weighted",
    "Raw ERA5-Land",
    "Empirical QM"
]
hits = [595, 688, 608, 231, 372]
misses = [672, 579, 659, 1036, 895]
false_alarms = [427, 722, 515, 606, 622]

x_idx = np.arange(len(models_fig6))
width = 0.25

# 6a. Contingency Counts
axes[0, 0].bar(x_idx - width, hits, width=width, label="Hits (Correct Detections)", color="#2a9d8f")
axes[0, 0].bar(x_idx, misses, width=width, label="Misses (Undetected)", color="#e76f51")
axes[0, 0].bar(x_idx + width, false_alarms, width=width, label="False Alarms (Spurious)", color="#f4a261")
axes[0, 0].set_xticks(x_idx)
axes[0, 0].set_xticklabels(models_fig6, rotation=15, fontsize=10)
axes[0, 0].set_title("A. Heavy-Rain Contingency Frequencies (≥64.5 mm, Test N=184,212)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Event Count", fontsize=11)
axes[0, 0].legend(fontsize=9)

# 6b. Precision vs Recall vs CSI
precision = [0.5822, 0.4879, 0.5414, 0.2760, 0.3742]
recall = [0.4696, 0.5430, 0.4799, 0.1823, 0.2936]
csi = [0.3512, 0.3459, 0.3412, 0.1233, 0.1969]

axes[0, 1].bar(x_idx - width, precision, width=width, label="Precision (1 - FAR)", color="#264653")
axes[0, 1].bar(x_idx, recall, width=width, label="Recall / POD", color="#457b9d")
axes[0, 1].bar(x_idx + width, csi, width=width, label="Critical Success Index (CSI)", color="#e63946")
axes[0, 1].set_xticks(x_idx)
axes[0, 1].set_xticklabels(models_fig6, rotation=15, fontsize=10)
axes[0, 1].set_title("B. Heavy-Rain Skill Metric Comparison", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Metric Score", fontsize=11)
axes[0, 1].set_ylim(0, 0.7)
axes[0, 1].legend(fontsize=9)

# 6c. Heavy Rain CSI by Subdivision
subdiv_csi_data = df_heavy_res[df_heavy_res["Stratum"].str.startswith("Subdivision:")].copy()
subdiv_csi_data["Subdiv_Clean"] = subdiv_csi_data["Stratum"].str.replace("Subdivision: ", "")
sns.barplot(data=subdiv_csi_data, x="Subdiv_Clean", y="CSI", hue="Model", ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("C. Heavy-Rain CSI by Meteorological Subdivision", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("CSI Score", fontsize=11)
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis='x', rotation=15)
axes[1, 0].legend(fontsize=8, loc="upper right")

# 6d. Heavy Rain CSI by Season
seas_csi_data = df_heavy_res[df_heavy_res["Stratum"].str.startswith("Season:")].copy()
seas_csi_data["Season_Clean"] = seas_csi_data["Stratum"].str.replace("Season: ", "")
sns.barplot(data=seas_csi_data, x="Season_Clean", y="CSI", hue="Model", ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("D. Heavy-Rain CSI across Seasons", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("CSI Score", fontsize=11)
axes[1, 1].set_xlabel("")
axes[1, 1].legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(fig_dir / "fig6_heavy_rain_contingency_and_detection.png", dpi=300)
plt.close(fig)
print("Saved fig6_heavy_rain_contingency_and_detection.png")

# ------------------------------------------------------------------------------
# Figure 7: Observed vs Predicted Rainfall across Intensity Categories
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

# 7a. Observed vs Predicted Means across IMD Bins
df_binned = df_err_res[df_err_res["Model"].isin(["LightGBM_Tweedie", "Raw_ERA5", "EQM"])].copy()
cat_order = [c[0] for c in intensity_bins]

sns.barplot(data=df_binned, x="Intensity_Category", y="Predicted_Mean_mm", hue="Model",
            order=cat_order, ax=axes[0], palette=["#1d3557", "#e63946", "#457b9d"])

# Overlay observed mean as points/line
obs_points = df_err_res[df_err_res["Model"] == "LightGBM_Tweedie"].set_index("Intensity_Category").loc[cat_order, "Observed_Mean_mm"]
axes[0].plot(range(len(cat_order)), obs_points.values, marker="o", color="crimson", linewidth=2.5, markersize=8, label="Ground Truth IMD Mean")

axes[0].set_title("A. Conditional Mean Estimation across IMD Rainfall Intensity Categories", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Rainfall Mean (mm)", fontsize=11)
axes[0].set_xlabel("IMD Intensity Category", fontsize=11)
axes[0].tick_params(axis='x', rotation=25)
axes[0].legend(fontsize=9)

# 7b. Empirical Quantile-Quantile (Q-Q) Plot on Physical Scale
q_levels = np.linspace(0.01, 0.9995, 300)
q_obs_fine = np.quantile(y_true, q_levels)
q_lgb_fine = np.quantile(pred_lgb_tweedie, q_levels)
q_cat_fine = np.quantile(pred_cat_tweedie, q_levels)
q_era5_fine = np.quantile(pred_era5, q_levels)
q_eqm_fine = np.quantile(pred_eqm, q_levels)

axes[1].plot([0, 200], [0, 200], "k--", linewidth=1.5, label="1:1 Perfect Quantile Parity")
axes[1].plot(q_obs_fine, q_lgb_fine, color="#1d3557", linewidth=2.2, label="LightGBM Tweedie")
axes[1].plot(q_obs_fine, q_cat_fine, color="#2a9d8f", linewidth=2.0, linestyle="-.", label="CatBoost Tweedie")
axes[1].plot(q_obs_fine, q_eqm_fine, color="#e76f51", linewidth=1.8, linestyle=":", label="Empirical Quantile Mapping (EQM)")
axes[1].plot(q_obs_fine, q_era5_fine, color="#7f7f7f", linewidth=1.5, linestyle="--", label="Raw ERA5-Land")

axes[1].axvline(64.5, color="red", linestyle=":", alpha=0.7, label="Official Heavy Rain (64.5 mm)")
axes[1].set_xlim(0, 160)
axes[1].set_ylim(0, 160)
axes[1].set_title("B. Quantile Parity Plot: Tail Shrinkage in Predictions", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Observed IMD Quantiles (mm)", fontsize=11)
axes[1].set_ylabel("Predicted Quantiles (mm)", fontsize=11)
axes[1].legend(fontsize=9, loc="upper left")

plt.tight_layout()
fig.savefig(fig_dir / "fig7_observed_vs_predicted_intensity_binned.png", dpi=300)
plt.close(fig)
print("Saved fig7_observed_vs_predicted_intensity_binned.png")

# ------------------------------------------------------------------------------
# Figure 8: Spatial Error Distribution Map across 252 IMD Grid Cells
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

# Western Ghats crest approximate boundary line for reference
# Ghats crest runs approx lon 74.8 to 75.8, lat 12.0 to 15.5
ghats_lat = np.linspace(11.8, 15.5, 50)
ghats_lon = 75.8 - 0.25 * (ghats_lat - 12.0)

# 8a. Spatial RMSE Map
sc1 = axes[0].scatter(df_station_err["longitude"], df_station_err["latitude"],
                      c=df_station_err["RMSE"], cmap="YlOrRd", s=70, edgecolors="k", linewidth=0.5)
cbar1 = plt.colorbar(sc1, ax=axes[0])
cbar1.set_label("Test RMSE (mm)", fontsize=10)
axes[0].plot(ghats_lon, ghats_lat, "k--", linewidth=1.5, label="Western Ghats Orographic Crest")
axes[0].set_title("A. Spatial Distribution of Test RMSE (mm) across 252 IMD Cells", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Longitude (°E)", fontsize=11)
axes[0].set_ylabel("Latitude (°N)", fontsize=11)
axes[0].legend(loc="upper left", fontsize=9)

# Highlight coastal fallback cells
fallback_cells = df_station_err[df_station_err["is_coastal_fallback"] == 1]
axes[0].scatter(fallback_cells["longitude"], fallback_cells["latitude"],
                facecolors="none", edgecolors="blue", s=180, linewidth=2.0, label="Coastal Fallback Cells (IDs 52, 68, 125)")
axes[0].legend(loc="upper left", fontsize=8)

# 8b. Spatial Mean Bias Map
max_abs_bias = max(abs(df_station_err["Bias"].min()), abs(df_station_err["Bias"].max()))
sc2 = axes[1].scatter(df_station_err["longitude"], df_station_err["latitude"],
                      c=df_station_err["Bias"], cmap="coolwarm", s=70, edgecolors="k", linewidth=0.5,
                      vmin=-max_abs_bias, vmax=max_abs_bias)
cbar2 = plt.colorbar(sc2, ax=axes[1])
cbar2.set_label("Test Mean Bias (mm)", fontsize=10)
axes[1].plot(ghats_lon, ghats_lat, "k--", linewidth=1.5, label="Western Ghats Orographic Crest")
axes[1].set_title("B. Spatial Distribution of Test Mean Bias (mm) across 252 IMD Cells", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Longitude (°E)", fontsize=11)
axes[1].set_ylabel("Latitude (°N)", fontsize=11)
axes[1].legend(loc="upper left", fontsize=9)

plt.tight_layout()
fig.savefig(fig_dir / "fig8_spatial_error_distribution_map.png", dpi=300)
plt.close(fig)
print("Saved fig8_spatial_error_distribution_map.png")

# ------------------------------------------------------------------------------
# Figure 9: Representative Heavy-Rain Case Studies
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Case 1: 2024-07-19 (Monsoonal Orographic Downpour)
date_case1 = "2024-07-19"
df_c1 = df_test[df_test["date"] == date_case1].copy()
idx_c1 = df_c1.index.values
obs_c1 = y_true[idx_c1]
pred_c1 = pred_lgb_tweedie[idx_c1]

vmax_c1 = max(obs_c1.max(), pred_c1.max())

sc11 = axes[0, 0].scatter(df_c1["longitude"], df_c1["latitude"], c=obs_c1,
                          cmap="Blues", s=80, edgecolors="k", linewidth=0.5, vmin=0, vmax=vmax_c1)
cbar11 = plt.colorbar(sc11, ax=axes[0, 0])
cbar11.set_label("Observed IMD Rainfall (mm)", fontsize=10)
axes[0, 0].set_title(f"A. Observed IMD Rainfall: {date_case1} (Active Monsoon)", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel("Longitude (°E)", fontsize=10)
axes[0, 0].set_ylabel("Latitude (°N)", fontsize=10)

sc12 = axes[0, 1].scatter(df_c1["longitude"], df_c1["latitude"], c=pred_c1,
                          cmap="Blues", s=80, edgecolors="k", linewidth=0.5, vmin=0, vmax=vmax_c1)
cbar12 = plt.colorbar(sc12, ax=axes[0, 1])
cbar12.set_label("LightGBM Tweedie Prediction (mm)", fontsize=10)
axes[0, 1].set_title(f"B. Predicted Rainfall: {date_case1} (LightGBM Tweedie)", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel("Longitude (°E)", fontsize=10)
axes[0, 1].set_ylabel("Latitude (°N)", fontsize=10)

# Case 2: 2024-10-21 (Post-Monsoon Convective Downpour)
date_case2 = "2024-10-21"
df_c2 = df_test[df_test["date"] == date_case2].copy()
idx_c2 = df_c2.index.values
obs_c2 = y_true[idx_c2]
pred_c2 = pred_lgb_tweedie[idx_c2]

vmax_c2 = max(obs_c2.max(), pred_c2.max())

sc21 = axes[1, 0].scatter(df_c2["longitude"], df_c2["latitude"], c=obs_c2,
                          cmap="Blues", s=80, edgecolors="k", linewidth=0.5, vmin=0, vmax=vmax_c2)
cbar21 = plt.colorbar(sc21, ax=axes[1, 0])
cbar21.set_label("Observed IMD Rainfall (mm)", fontsize=10)
axes[1, 0].set_title(f"C. Observed IMD Rainfall: {date_case2} (Post-Monsoon Storm)", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Longitude (°E)", fontsize=10)
axes[1, 0].set_ylabel("Latitude (°N)", fontsize=10)

sc22 = axes[1, 1].scatter(df_c2["longitude"], df_c2["latitude"], c=pred_c2,
                          cmap="Blues", s=80, edgecolors="k", linewidth=0.5, vmin=0, vmax=vmax_c2)
cbar22 = plt.colorbar(sc22, ax=axes[1, 1])
cbar22.set_label("LightGBM Tweedie Prediction (mm)", fontsize=10)
axes[1, 1].set_title(f"D. Predicted Rainfall: {date_case2} (LightGBM Tweedie)", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel("Longitude (°E)", fontsize=10)
axes[1, 1].set_ylabel("Latitude (°N)", fontsize=10)

plt.tight_layout()
fig.savefig(fig_dir / "fig9_representative_heavy_rain_case_studies.png", dpi=300)
plt.close(fig)
print("Saved fig9_representative_heavy_rain_case_studies.png")

print("\n--- Task 3 Completed Successfully ---")
