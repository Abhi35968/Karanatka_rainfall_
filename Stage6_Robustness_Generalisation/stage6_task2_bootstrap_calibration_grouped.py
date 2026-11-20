import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.calibration import calibration_curve

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
stage6_dir = base_dir / "Stage6_Robustness_Generalisation"
fig_dir = stage6_dir / "stage6_figures"

print("Loading test dataset and predictions...")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_test["date"] = pd.to_datetime(df_test["date"])

y_true = df_test["target_rainfall_mm"].values
y_true_rain = df_test["target_is_rain"].values
y_true_heavy = df_test["target_is_heavy_rain"].values

# Continuous model predictions
pred_lgb = np.load(models_dir / "pred_lgb_tweedie_test.npy")
pred_cat = np.load(models_dir / "pred_cat_tweedie_test.npy")
pred_hurdle = np.load(models_dir / "pred_hurdle_ev_test.npy")
pred_persist = np.load(models_dir / "pred_baseline_persist_test.npy")
pred_era5 = np.load(models_dir / "pred_baseline_era5_test.npy")
pred_eqm = np.load(models_dir / "pred_baseline_eqm_test.npy")

# Classification probabilities
p_rain = np.load(models_dir / "p_test_lgb_rain.npy")

# Heavy rain models
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
p_heavy_focal = 1.0 / (1.0 + np.exp(-z_focal))
p_heavy_lgb_wt = m_lgb_weighted.predict_proba(X_te_heavy)[:, 1]
p_heavy_cat_wt = m_cat_weighted.predict_proba(X_te_heavy)[:, 1]

# ==============================================================================
# 1. PAIRED TEMPORAL BLOCK BOOTSTRAP (B=2,000, 7-DAY VS 14-DAY BLOCK LENGTH)
# ==============================================================================
print("\n--- 1. Running Paired Block Bootstrap (B=2,000, L=7 and L=14 days) ---")
unique_dates = np.sort(df_test["date"].unique())
n_dates = len(unique_dates)
n_instances = len(df_test)

# Map date to instance indices
date_indices = df_test.groupby("date").indices

# Pre-aggregate sum of squared errors per date for fast vectorized bootstrapping
all_models_dict = {
    "LightGBM_Tweedie": pred_lgb,
    "CatBoost_Tweedie": pred_cat,
    "Hurdle_ExpectedValue": pred_hurdle,
    "Persistence": pred_persist,
    "Raw_ERA5": pred_era5,
    "EQM": pred_eqm
}

date_sse = {}
for m_name, y_p in all_models_dict.items():
    sq_err = (np.clip(y_p, 0.0, None) - y_true) ** 2
    # Group sum per date
    df_temp = pd.DataFrame({"date": df_test["date"], "se": sq_err})
    date_sse[m_name] = df_temp.groupby("date")["se"].sum().loc[unique_dates].values

B = 2000
np.random.seed(42)

bootstrap_results = []
bootstrap_replicates_dict = {}

comparison_pairs = [
    ("LightGBM_Tweedie", "Persistence"),
    ("LightGBM_Tweedie", "EQM"),
    ("LightGBM_Tweedie", "Raw_ERA5"),
    ("CatBoost_Tweedie", "Persistence"),
    ("CatBoost_Tweedie", "EQM"),
    ("Hurdle_ExpectedValue", "Persistence"),
    ("Hurdle_ExpectedValue", "EQM")
]

for block_len in [7, 14]:
    print(f"  Computing bootstrap for block length L={block_len} days...")
    n_blocks = int(np.ceil(n_dates / block_len))
    
    # Store replicate differences
    pair_diffs = {f"{m1}_vs_{m2}": np.zeros(B) for m1, m2 in comparison_pairs}
    
    for b in range(B):
        # Sample date blocks
        sampled_date_indices = []
        for _ in range(n_blocks):
            start_d = np.random.randint(0, max(1, n_dates - block_len + 1))
            sampled_date_indices.extend(range(start_d, start_d + block_len))
        
        # Trim to exact date length
        sampled_date_indices = sampled_date_indices[:n_dates]
        
        # Calculate RMSE for each model
        rmse_b = {}
        for m_name in all_models_dict:
            total_sse = np.sum(date_sse[m_name][sampled_date_indices])
            rmse_b[m_name] = np.sqrt(total_sse / n_instances)
            
        for m1, m2 in comparison_pairs:
            pair_diffs[f"{m1}_vs_{m2}"][b] = rmse_b[m1] - rmse_b[m2]
            
    # Store for figure
    bootstrap_replicates_dict[block_len] = pair_diffs
    
    # Summarize stats
    for m1, m2 in comparison_pairs:
        k = f"{m1}_vs_{m2}"
        diffs = pair_diffs[k]
        
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        ci_low = float(np.percentile(diffs, 2.5))
        ci_high = float(np.percentile(diffs, 97.5))
        n_nonneg = int(np.sum(diffs >= 0))
        p_val_bound = f"p < {1.0 / (B + 1):.4f}" if n_nonneg == 0 else f"p = {n_nonneg / B:.4f}"
        
        bootstrap_results.append({
            "block_length_days": block_len,
            "replicates": B,
            "model_A": m1,
            "baseline_B": m2,
            "mean_delta_rmse_mm": round(mean_diff, 4),
            "std_error_mm": round(std_diff, 4),
            "ci95_low_mm": round(ci_low, 4),
            "ci95_high_mm": round(ci_high, 4),
            "replicates_delta_ge_zero": n_nonneg,
            "empirical_p_value": p_val_bound,
            "assumptions": f"Stationary block bootstrap (L={block_len}d), statewide 252 grid cells kept intact"
        })

df_boot = pd.DataFrame(bootstrap_results)
df_boot.to_csv(stage6_dir / "paired_bootstrap_robustness.csv", index=False)
print("Saved paired_bootstrap_robustness.csv")

# ==============================================================================
# 2. CLASSIFICATION RELIABILITY & PROBABILITY CALIBRATION
# ==============================================================================
print("\n--- 2. Evaluating Classification Reliability and Calibration ---")

def compute_calibration_metrics(y_true_bin, p_pred, model_name, task_name, thresh=0.5):
    yt = np.asarray(y_true_bin, dtype=int)
    p = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
    n = len(yt)
    
    # Brier Score
    bs = float(np.mean((p - yt) ** 2))
    
    # Climatology Brier Score
    base_rate = float(np.mean(yt))
    bs_clim = float(np.mean((base_rate - yt) ** 2))
    bss = float(1.0 - (bs / bs_clim)) if bs_clim > 1e-6 else 0.0
    
    # 10-bin calibration curve
    prob_true, prob_pred = calibration_curve(yt, p, n_bins=10, strategy="uniform")
    
    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, 11)
    bin_indices = np.digitize(p, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, 9)
    
    ece = 0.0
    mce = 0.0
    for b in range(10):
        in_bin = (bin_indices == b)
        count_b = np.sum(in_bin)
        if count_b > 0:
            avg_p = float(np.mean(p[in_bin]))
            avg_y = float(np.mean(yt[in_bin]))
            gap = abs(avg_p - avg_y)
            ece += (count_b / n) * gap
            if gap > mce:
                mce = gap
                
    # Contingency at decision threshold
    p_bin = (p >= thresh).astype(int)
    h = int(np.sum((yt == 1) & (p_bin == 1)))
    m = int(np.sum((yt == 1) & (p_bin == 0)))
    f = int(np.sum((yt == 0) & (p_bin == 1)))
    c = int(np.sum((yt == 0) & (p_bin == 0)))
    
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
        "task": task_name,
        "model": model_name,
        "calibrated_threshold": thresh,
        "base_event_rate": round(base_rate, 4),
        "Brier_Score": round(bs, 6),
        "Brier_Skill_Score": round(bss, 4),
        "ECE": round(ece, 4),
        "MCE": round(mce, 4),
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

calib_records = [
    compute_calibration_metrics(y_true_rain, p_rain, "LightGBM_Rain_Classifier", "Rain_Occurrence (>=0.1mm)", thresh=0.44),
    compute_calibration_metrics(y_true_heavy, p_heavy_focal, "LightGBM_Focal_Loss", "Heavy_Rain (>=64.5mm)", thresh=0.53),
    compute_calibration_metrics(y_true_heavy, p_heavy_lgb_wt, "LightGBM_Class_Weighted", "Heavy_Rain (>=64.5mm)", thresh=0.96),
    compute_calibration_metrics(y_true_heavy, p_heavy_cat_wt, "CatBoost_Class_Weighted", "Heavy_Rain (>=64.5mm)", thresh=0.98),
]

df_calib = pd.DataFrame(calib_records)
df_calib.to_csv(stage6_dir / "classification_calibration_results.csv", index=False)
print("Saved classification_calibration_results.csv")

# ==============================================================================
# 3. GROUPED FEATURE SENSITIVITY
# ==============================================================================
print("\n--- 3. Evaluating Grouped Physical Feature Sensitivities ---")
# Load permutation and SHAP values from Stage 5
df_perm = pd.read_csv(stage5_dir / "permutation_importance.csv")
df_shap = pd.read_csv(stage5_dir / "shap_summary_values.csv")

# Define physical domain groupings
domain_map = {}
for col in primary_cols:
    if "target_rain" in col or col == "api_14d":
        domain_map[col] = "1. Antecedent IMD Rainfall Memory"
    elif "era5_tp" in col or col in ["ro_lag_1d", "sro_lag_1d", "src_lag_1d", "e_lag_1d", "pev_lag_1d", "evabs_lag_1d", "evatc_lag_1d", "evavt_lag_1d", "fal_lag_1d"]:
        domain_map[col] = "2. ERA5 Runoff & Hydrological Fluxes"
    elif col in ["t2m_mean_lag_1d", "t2m_min_lag_1d", "t2m_max_lag_1d", "temp_range_lag_1d", "d2m_mean_lag_1d", "dewpoint_depression_lag_1d", "t2m_roll_3d_mean", "t2m_roll_7d_mean", "sp_lag_1d", "wind_speed_lag_1d", "wind_roll_3d_mean", "wind_roll_7d_mean"]:
        domain_map[col] = "3. Atmospheric Thermodynamic State & Wind"
    elif "swvl" in col or col in ["ssrd_lag_1d", "strd_lag_1d", "slhf_lag_1d", "sshf_lag_1d"]:
        domain_map[col] = "4. Soil Moisture & Energy Radiation Fluxes"
    elif col in ["elevation_dem_m", "slope_dem_deg", "latitude", "longitude", "dist_to_coast_km", "dist_to_ghats_crest_km"]:
        domain_map[col] = "5. Topography, Elevation & Spatial Proximity"
    elif col in ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "is_monsoon"]:
        domain_map[col] = "6. Temporal Ephemeris & Seasonality"
    else:
        domain_map[col] = "7. Categorical & Regional Encodings"

domain_map["subdivision_enc"] = "7. Categorical & Regional Encodings"
domain_map["zone_id_enc"] = "7. Categorical & Regional Encodings"
domain_map["is_coastal_fallback"] = "7. Categorical & Regional Encodings"

df_perm["domain"] = df_perm["feature"].map(domain_map).fillna("7. Categorical & Regional Encodings")
df_shap["domain"] = df_shap["feature"].map(domain_map).fillna("7. Categorical & Regional Encodings")

# Aggregate by physical domain
group_summary = []
for dom, grp_p in df_perm.groupby("domain"):
    grp_s = df_shap[df_shap["domain"] == dom]
    
    feat_count = len(grp_p)
    sum_delta_rmse = float(grp_p["delta_rmse_mean"].sum())
    mean_delta_rmse = float(grp_p["delta_rmse_mean"].mean())
    max_delta_rmse = float(grp_p["delta_rmse_mean"].max())
    lead_feat_perm = grp_p.sort_values("delta_rmse_mean", ascending=False)["feature"].iloc[0]
    
    sum_shap_tweedie = float(grp_s["shap_mean_abs_tweedie"].sum())
    sum_shap_focal = float(grp_s["shap_mean_abs_heavy_focal"].sum())
    
    group_summary.append({
        "physical_domain": dom,
        "feature_count": feat_count,
        "sum_delta_rmse_mm": round(sum_delta_rmse, 4),
        "mean_delta_rmse_mm": round(mean_delta_rmse, 4),
        "max_delta_rmse_mm": round(max_delta_rmse, 4),
        "leading_feature": lead_feat_perm,
        "sum_shap_tweedie_mm": round(sum_shap_tweedie, 4),
        "sum_shap_heavy_focal": round(sum_shap_focal, 4),
        "interpretative_caution": "Features within domain are physically and statistically correlated; values indicate joint predictive association, not orthogonal causality."
    })

df_group = pd.DataFrame(group_summary).sort_values("sum_delta_rmse_mm", ascending=False).reset_index(drop=True)
df_group.to_csv(stage6_dir / "grouped_feature_sensitivity.csv", index=False)
print("Saved grouped_feature_sensitivity.csv")

# ==============================================================================
# 4. PUBLICATION FIGURES: FIG 5, 6, 7
# ==============================================================================
print("\n--- 4. Generating Publication Figures (Fig 5, 6, 7) ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------------------------
# Figure 5: Paired Block Bootstrap Sensitivity (7-Day vs 14-Day Blocks)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

diffs_persist_7 = bootstrap_replicates_dict[7]["LightGBM_Tweedie_vs_Persistence"]
diffs_persist_14 = bootstrap_replicates_dict[14]["LightGBM_Tweedie_vs_Persistence"]

diffs_eqm_7 = bootstrap_replicates_dict[7]["LightGBM_Tweedie_vs_EQM"]
diffs_eqm_14 = bootstrap_replicates_dict[14]["LightGBM_Tweedie_vs_EQM"]

# 5a. Tweedie vs Persistence
sns.kdeplot(diffs_persist_7, ax=axes[0], label=f"L = 7 days (Mean: {np.mean(diffs_persist_7):.2f} mm)", color="#2a9d8f", linewidth=2.2)
sns.kdeplot(diffs_persist_14, ax=axes[0], label=f"L = 14 days (Mean: {np.mean(diffs_persist_14):.2f} mm)", color="#e76f51", linewidth=2.2, linestyle="--")
axes[0].axvline(0, color="black", linestyle="-", linewidth=1.0, label="Zero Difference (Parity)")
axes[0].axvline(np.percentile(diffs_persist_7, 97.5), color="#2a9d8f", linestyle=":", alpha=0.8, label="97.5% CI Limit (L=7)")
axes[0].axvline(np.percentile(diffs_persist_14, 97.5), color="#e76f51", linestyle=":", alpha=0.8, label="97.5% CI Limit (L=14)")
axes[0].set_title("A. Paired ΔRMSE: LightGBM Tweedie vs. Persistence (B=2,000)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Paired RMSE Difference: Model - Persistence (mm)", fontsize=10)
axes[0].set_ylabel("Kernel Density", fontsize=10)
axes[0].legend(fontsize=8, loc="upper left")

# 5b. Tweedie vs EQM
sns.kdeplot(diffs_eqm_7, ax=axes[1], label=f"L = 7 days (Mean: {np.mean(diffs_eqm_7):.2f} mm)", color="#2a9d8f", linewidth=2.2)
sns.kdeplot(diffs_eqm_14, ax=axes[1], label=f"L = 14 days (Mean: {np.mean(diffs_eqm_14):.2f} mm)", color="#e76f51", linewidth=2.2, linestyle="--")
axes[1].axvline(0, color="black", linestyle="-", linewidth=1.0, label="Zero Difference (Parity)")
axes[1].axvline(np.percentile(diffs_eqm_7, 97.5), color="#2a9d8f", linestyle=":", alpha=0.8, label="97.5% CI Limit (L=7)")
axes[1].axvline(np.percentile(diffs_eqm_14, 97.5), color="#e76f51", linestyle=":", alpha=0.8, label="97.5% CI Limit (L=14)")
axes[1].set_title("B. Paired ΔRMSE: LightGBM Tweedie vs. EQM (B=2,000)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Paired RMSE Difference: Model - EQM (mm)", fontsize=10)
axes[1].set_ylabel("Kernel Density", fontsize=10)
axes[1].legend(fontsize=8, loc="upper left")

plt.tight_layout()
fig.savefig(fig_dir / "fig5_paired_block_bootstrap_sensitivity.png", dpi=300)
plt.close(fig)
print("Saved fig5_paired_block_bootstrap_sensitivity.png")

# ------------------------------------------------------------------------------
# Figure 6: Probability Calibration and Reliability Diagrams
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# 6a. Rain Occurrence Reliability Diagram
prob_true_rain, prob_pred_rain = calibration_curve(y_true_rain, p_rain, n_bins=10, strategy="uniform")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect Calibration")
axes[0].plot(prob_pred_rain, prob_true_rain, "s-", color="#1d3557", linewidth=2.0, markersize=7,
             label=f"LightGBM Rain (ECE={calib_records[0]['ECE']:.4f}, BSS={calib_records[0]['Brier_Skill_Score']:.4f})")
axes[0].axvline(0.44, color="crimson", linestyle=":", alpha=0.8, label="Operating Threshold p*=0.44")
axes[0].set_title("A. Reliability Diagram: Rain Occurrence (≥0.1 mm)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Forecast Probability Bin Mean", fontsize=11)
axes[0].set_ylabel("Empirical Event Frequency", fontsize=11)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=9, loc="upper left")

# 6b. Heavy Rain Reliability Diagram (Focal Loss vs Class-Weighted)
prob_true_foc, prob_pred_foc = calibration_curve(y_true_heavy, p_heavy_focal, n_bins=10, strategy="uniform")
prob_true_wt, prob_pred_wt = calibration_curve(y_true_heavy, p_heavy_lgb_wt, n_bins=10, strategy="uniform")

axes[1].plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect Calibration")
axes[1].plot(prob_pred_foc, prob_true_foc, "o-", color="crimson", linewidth=2.2, markersize=7,
             label=f"LightGBM Focal Loss (ECE={calib_records[1]['ECE']:.4f}, CSI=0.3512)")
axes[1].plot(prob_pred_wt, prob_true_wt, "^-", color="#457b9d", linewidth=1.8, markersize=7,
             label=f"LightGBM Weighted (ECE={calib_records[2]['ECE']:.4f}, CSI=0.3459)")
axes[1].set_title("B. Reliability Diagram: Heavy Rain (≥64.5 mm)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Forecast Probability Bin Mean", fontsize=11)
axes[1].set_ylabel("Empirical Event Frequency", fontsize=11)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=9, loc="upper left")

plt.tight_layout()
fig.savefig(fig_dir / "fig6_probability_calibration_and_reliability.png", dpi=300)
plt.close(fig)
print("Saved fig6_probability_calibration_and_reliability.png")

# ------------------------------------------------------------------------------
# Figure 7: Grouped Feature Domain Sensitivity
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

df_group_plot = df_group.sort_values("sum_delta_rmse_mm", ascending=True).copy()

# 7a. Sum of Permutation Delta RMSE
axes[0].barh(range(len(df_group_plot)), df_group_plot["sum_delta_rmse_mm"], color="#2b5c8f", align="center")
axes[0].set_yticks(range(len(df_group_plot)))
axes[0].set_yticklabels([d.split(". ")[-1] for d in df_group_plot["physical_domain"]], fontsize=10)
axes[0].set_title("A. Total Permutation Impact (Sum of ΔRMSE, mm)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Aggregate Validation ΔRMSE Increase (mm)", fontsize=11)

# Annotate leading feature
for idx, row in df_group_plot.reset_index(drop=True).iterrows():
    axes[0].text(row["sum_delta_rmse_mm"] + 0.05, idx, f"Lead: {row['leading_feature']}", va="center", fontsize=8, color="#333333")

# 7b. Total SHAP Value Attribution
axes[1].barh(range(len(df_group_plot)), df_group_plot["sum_shap_tweedie_mm"], color="#e07a5f", align="center")
axes[1].set_yticks(range(len(df_group_plot)))
axes[1].set_yticklabels([d.split(". ")[-1] for d in df_group_plot["physical_domain"]], fontsize=10)
axes[1].set_title("B. Total SHAP Attribution (Sum of Mean |SHAP|, mm)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Aggregate Validation SHAP Magnitude (mm)", fontsize=11)

plt.tight_layout()
fig.savefig(fig_dir / "fig7_grouped_feature_sensitivity.png", dpi=300)
plt.close(fig)
print("Saved fig7_grouped_feature_sensitivity.png")

print("\n--- Task 2 Completed Successfully ---")

