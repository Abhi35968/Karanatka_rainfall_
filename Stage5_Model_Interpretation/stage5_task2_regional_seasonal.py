import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
models_dir = base_dir / "Stage4_1_Model_Development/models"
stage5_dir = base_dir / "Stage5_Model_Interpretation"
fig_dir = stage5_dir / "stage5_figures"

print("Loading test dataset and predictions...")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["month"] = df_test["date"].dt.month

y_true = df_test["target_rainfall_mm"].values
y_true_heavy = (y_true >= 64.5)

# Load frozen predictions
models = {
    "LightGBM_Tweedie": np.load(models_dir / "pred_lgb_tweedie_test.npy"),
    "CatBoost_Tweedie": np.load(models_dir / "pred_cat_tweedie_test.npy"),
    "Hurdle_ExpectedValue": np.load(models_dir / "pred_hurdle_ev_test.npy"),
    "Persistence": np.load(models_dir / "pred_baseline_persist_test.npy"),
    "Raw_ERA5": np.load(models_dir / "pred_baseline_era5_test.npy"),
    "EQM": np.load(models_dir / "pred_baseline_eqm_test.npy")
}

def calc_all_metrics(y_obs, y_hat):
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
    
    if std_o > 1e-6 and std_p > 1e-6:
        r = float(np.corrcoef(y_obs, y_hat)[0, 1])
    else:
        r = 0.0
        
    denom_nse = float(np.sum((y_obs - mu_o) ** 2))
    nse = float(1.0 - (np.sum((y_hat - y_obs) ** 2) / denom_nse)) if denom_nse > 1e-6 else -999.0
    
    # Classic KGE-2009
    beta = mu_p / mu_o if mu_o > 1e-6 else 1.0
    alpha = std_p / std_o if std_o > 1e-6 else 1.0
    kge_2009 = float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)) if (std_o > 1e-6 and std_p > 1e-6) else -999.0
    
    # Modified KGE-2012
    cv_o = std_o / mu_o if mu_o > 1e-6 else 1.0
    cv_p = std_p / mu_p if mu_p > 1e-6 else 1.0
    gamma = cv_p / cv_o if cv_o > 1e-6 else 1.0
    kge_2012 = float(1.0 - np.sqrt((r - 1.0)**2 + (gamma - 1.0)**2 + (beta - 1.0)**2)) if (std_o > 1e-6 and std_p > 1e-6) else -999.0
    
    # Contingency at 64.5 mm
    obs_heavy = (y_obs >= 64.5)
    pred_heavy = (y_hat >= 64.5)
    
    h = int(np.sum(obs_heavy & pred_heavy))
    m = int(np.sum(obs_heavy & (~pred_heavy)))
    f = int(np.sum((~obs_heavy) & pred_heavy))
    c = int(np.sum((~obs_heavy) & (~pred_heavy)))
    
    csi = float(h / (h + m + f)) if (h + m + f) > 0 else 0.0
    pod = float(h / (h + m)) if (h + m) > 0 else 0.0
    far = float(f / (h + f)) if (h + f) > 0 else 0.0
    
    expected_hits = ((h + m) * (h + f)) / n if n > 0 else 0.0
    denom_hss = (2 * h + m + f) - 2 * expected_hits
    hss = float(2 * (h * c - m * f) / denom_hss) if abs(denom_hss) > 1e-6 else 0.0
    
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
        "KGE_2012": round(kge_2012, 4),
        "Heavy_Obs_Count": int(np.sum(obs_heavy)),
        "Heavy_Pred_Count": int(np.sum(pred_heavy)),
        "Hits": h,
        "Misses": m,
        "False_Alarms": f,
        "Correct_Negatives": c,
        "CSI_64.5mm": round(csi, 4),
        "POD_64.5mm": round(pod, 4),
        "FAR_64.5mm": round(far, 4),
        "HSS_64.5mm": round(hss, 4)
    }

# ==============================================================================
# 1. REGIONAL EVALUATION (Subdivisions + Coastal Fallback Specifics)
# ==============================================================================
print("\n--- 1. Evaluating Regional Strata ---")
regional_masks = {
    "Coastal_Karnataka_All_13_cells": (df_test["subdivision"] == "Coastal Karnataka").values,
    "Malnad_35_cells": (df_test["subdivision"] == "Malnad").values,
    "North_Interior_Karnataka_118_cells": (df_test["subdivision"] == "North Interior Karnataka").values,
    "South_Interior_Karnataka_86_cells": (df_test["subdivision"] == "South Interior Karnataka").values,
    "Coastal_Fallback_Only_3_cells_IDs_52_68_125": (df_test["is_coastal_fallback"] == 1).values,
    "Coastal_Native_Land_Only_10_cells": ((df_test["subdivision"] == "Coastal Karnataka") & (df_test["is_coastal_fallback"] == 0)).values,
    "Statewide_All_252_cells": np.ones(len(df_test), dtype=bool)
}

regional_records = []
for reg_name, mask in regional_masks.items():
    y_sub = y_true[mask]
    for m_name, y_hat in models.items():
        yh_sub = y_hat[mask]
        met = calc_all_metrics(y_sub, yh_sub)
        row = {"region": reg_name, "model": m_name, **met}
        regional_records.append(row)

df_reg = pd.DataFrame(regional_records)
df_reg.to_csv(stage5_dir / "regional_performance_analysis.csv", index=False)
print("Saved regional_performance_analysis.csv")

# ==============================================================================
# 2. SEASONAL EVALUATION
# ==============================================================================
print("\n--- 2. Evaluating Seasonal Strata ---")
seasonal_masks = {
    "Southwest_Monsoon_JJAS": df_test["month"].isin([6, 7, 8, 9]).values,
    "Post_Monsoon_Northeast_OND": df_test["month"].isin([10, 11, 12]).values,
    "Pre_Monsoon_MAM": df_test["month"].isin([3, 4, 5]).values,
    "Winter_JF": df_test["month"].isin([1, 2]).values,
    "Annual_Full_Period": np.ones(len(df_test), dtype=bool)
}

seasonal_records = []
for seas_name, mask in seasonal_masks.items():
    y_sub = y_true[mask]
    for m_name, y_hat in models.items():
        yh_sub = y_hat[mask]
        met = calc_all_metrics(y_sub, yh_sub)
        row = {"season": seas_name, "model": m_name, **met}
        seasonal_records.append(row)

df_seas = pd.DataFrame(seasonal_records)
df_seas.to_csv(stage5_dir / "seasonal_performance_analysis.csv", index=False)
print("Saved seasonal_performance_analysis.csv")

# ==============================================================================
# 3. PUBLICATION FIGURES: FIG 4 & FIG 5
# ==============================================================================
print("\n--- 3. Generating Figures 4 and 5 ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------------------------
# Figure 4: Regional Performance Comparison
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

plot_regions = [
    "Coastal_Karnataka_All_13_cells",
    "Malnad_35_cells",
    "North_Interior_Karnataka_118_cells",
    "South_Interior_Karnataka_86_cells",
    "Coastal_Fallback_Only_3_cells_IDs_52_68_125"
]
plot_region_labels = [
    "Coastal (13 cells)",
    "Malnad (35 cells)",
    "North Interior (118 cells)",
    "South Interior (86 cells)",
    "Coastal Fallback (3 cells)"
]

df_reg_plot = df_reg[df_reg["region"].isin(plot_regions)].copy()
region_label_map = dict(zip(plot_regions, plot_region_labels))
df_reg_plot["region_label"] = df_reg_plot["region"].map(region_label_map)

# 4a. RMSE by Region
sns.barplot(data=df_reg_plot, x="region_label", y="RMSE", hue="model", ax=axes[0, 0], palette="tab10")
axes[0, 0].set_title("A. Root Mean Squared Error (RMSE, mm) by Agroclimatic Region", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("RMSE (mm)", fontsize=11)
axes[0, 0].set_xlabel("")
axes[0, 0].tick_params(axis='x', rotation=15)
axes[0, 0].get_legend().remove()

# 4b. Modified KGE-2012 by Region
sns.barplot(data=df_reg_plot, x="region_label", y="KGE_2012", hue="model", ax=axes[0, 1], palette="tab10")
axes[0, 1].set_title("B. Modified Kling-Gupta Efficiency (KGE-2012) by Region", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("KGE-2012", fontsize=11)
axes[0, 1].set_xlabel("")
axes[0, 1].tick_params(axis='x', rotation=15)
axes[0, 1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0, 1].get_legend().remove()

# 4c. Heavy Rain CSI (>= 64.5 mm) by Region
sns.barplot(data=df_reg_plot, x="region_label", y="CSI_64.5mm", hue="model", ax=axes[1, 0], palette="tab10")
axes[1, 0].set_title("C. Heavy Rain Threat Score (CSI, ≥64.5 mm) by Region", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Critical Success Index (CSI)", fontsize=11)
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis='x', rotation=15)
axes[1, 0].get_legend().remove()

# 4d. Percentage Bias (%) by Region
sns.barplot(data=df_reg_plot, x="region_label", y="PBias_pct", hue="model", ax=axes[1, 1], palette="tab10")
axes[1, 1].set_title("D. Percentage Bias (PBias, %) by Region", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Percentage Bias (%)", fontsize=11)
axes[1, 1].set_xlabel("")
axes[1, 1].tick_params(axis='x', rotation=15)
axes[1, 1].axhline(0, color="black", linestyle="-", linewidth=0.8)

# Unified legend at bottom
handles, labels = axes[1, 1].get_legend_handles_labels()
axes[1, 1].get_legend().remove()
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
fig.savefig(fig_dir / "fig4_regional_performance_breakdown.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_regional_performance_breakdown.png")

# ------------------------------------------------------------------------------
# Figure 5: Seasonal Performance Comparison
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

plot_seasons = [
    "Southwest_Monsoon_JJAS",
    "Post_Monsoon_Northeast_OND",
    "Pre_Monsoon_MAM",
    "Winter_JF"
]
plot_season_labels = [
    "SW Monsoon (JJAS)",
    "Post-Monsoon (OND)",
    "Pre-Monsoon (MAM)",
    "Winter (JF)"
]

df_seas_plot = df_seas[df_seas["season"].isin(plot_seasons)].copy()
season_label_map = dict(zip(plot_seasons, plot_season_labels))
df_seas_plot["season_label"] = df_seas_plot["season"].map(season_label_map)

# 5a. RMSE by Season
sns.barplot(data=df_seas_plot, x="season_label", y="RMSE", hue="model", ax=axes[0, 0], palette="tab10")
axes[0, 0].set_title("A. Root Mean Squared Error (RMSE, mm) across Seasons", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("RMSE (mm)", fontsize=11)
axes[0, 0].set_xlabel("")
axes[0, 0].get_legend().remove()

# 5b. Modified KGE-2012 by Season
sns.barplot(data=df_seas_plot, x="season_label", y="KGE_2012", hue="model", ax=axes[0, 1], palette="tab10")
axes[0, 1].set_title("B. Modified Kling-Gupta Efficiency (KGE-2012) across Seasons", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("KGE-2012", fontsize=11)
axes[0, 1].set_xlabel("")
axes[0, 1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0, 1].get_legend().remove()

# 5c. Pearson Correlation by Season
sns.barplot(data=df_seas_plot, x="season_label", y="Pearson_r", hue="model", ax=axes[1, 0], palette="tab10")
axes[1, 0].set_title("C. Pearson Correlation (r) across Seasons", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Correlation (r)", fontsize=11)
axes[1, 0].set_xlabel("")
axes[1, 0].get_legend().remove()

# 5d. Heavy Rain CSI (>= 64.5 mm) by Season
sns.barplot(data=df_seas_plot, x="season_label", y="CSI_64.5mm", hue="model", ax=axes[1, 1], palette="tab10")
axes[1, 1].set_title("D. Heavy Rain CSI (≥64.5 mm) across Seasons", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Critical Success Index (CSI)", fontsize=11)
axes[1, 1].set_xlabel("")

handles, labels = axes[1, 1].get_legend_handles_labels()
axes[1, 1].get_legend().remove()
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
fig.savefig(fig_dir / "fig5_seasonal_performance_breakdown.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_seasonal_performance_breakdown.png")

print("\n--- Task 2 Completed Successfully ---")

