import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as patches

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
models_dir = base_dir / "Stage4_1_Model_Development/models"
stage5_dir = base_dir / "Stage5_Model_Interpretation"
stage6_dir = base_dir / "Stage6_Robustness_Generalisation"

out_dir = base_dir / "Stage7_Publication_Ready_Package"
fig_dir = out_dir / "publication_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set global publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans', 'Arial', 'Helvetica'
matplotlib.rcParams['axes.edgecolor'] = '#333333'
matplotlib.rcParams['axes.linewidth'] = 0.8

print("Loading test dataset and predictions...")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
y_true = df_test["target_rainfall_mm"].values
y_true_heavy = (y_true >= 64.5).astype(int)

pred_lgb = np.load(models_dir / "pred_lgb_tweedie_test.npy")
pred_cat = np.load(models_dir / "pred_cat_tweedie_test.npy")
pred_hurdle = np.load(models_dir / "pred_hurdle_ev_test.npy")
pred_persist = np.load(models_dir / "pred_baseline_persist_test.npy")
pred_era5 = np.load(models_dir / "pred_baseline_era5_test.npy")
pred_eqm = np.load(models_dir / "pred_baseline_eqm_test.npy")

# Helper to save both PNG (300 dpi) and vector PDF
def save_pub_fig(fig, base_name):
    png_path = fig_dir / f"{base_name}.png"
    pdf_path = fig_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base_name}.png and {base_name}.pdf")

# ==============================================================================
# FIGURE 1: STUDY AREA AND DATA COVERAGE
# ==============================================================================
print("\n--- Generating Figure 1: Study Area and Data Coverage ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

cells = df_test[['station_id', 'latitude', 'longitude', 'subdivision', 'elevation_dem_m', 'dist_to_coast_km', 'is_coastal_fallback']].drop_duplicates().copy()

# Western Ghats ridge approximation
ghats_lat = np.linspace(11.8, 15.5, 50)
ghats_lon = 75.8 - 0.25 * (ghats_lat - 12.0)

# 1a. Map of 252 IMD Grid Cells
subdiv_palette = {
    "Coastal Karnataka": "#1f77b4",
    "Malnad": "#2ca02c",
    "North Interior Karnataka": "#ff7f0e",
    "South Interior Karnataka": "#9467bd"
}

for subdiv, grp in cells.groupby("subdivision"):
    axes[0].scatter(grp["longitude"], grp["latitude"], color=subdiv_palette[subdiv], s=65, label=f"{subdiv} ({len(grp)} cells)", edgecolors="k", linewidth=0.5)

# Plot Western Ghats crest
axes[0].plot(ghats_lon, ghats_lat, color="black", linestyle="--", linewidth=1.8, label="Western Ghats Orographic Crest")

# Highlight 3 Coastal Fallback Cells
fallback_cells = cells[cells["is_coastal_fallback"] == 1]
axes[0].scatter(fallback_cells["longitude"], fallback_cells["latitude"], facecolors="none", edgecolors="red", s=180, linewidth=2.2, label="Coastal Fallback Cells (IDs 52, 68, 125)")

for _, r in fallback_cells.iterrows():
    axes[0].annotate(f"Cell {int(r['station_id'])}", (r["longitude"], r["latitude"]), textcoords="offset points", xytext=(-35, 5), fontsize=8, fontweight="bold", color="red")

axes[0].set_title("A. Spatial Distribution of 252 IMD Grid Cells Across Subdivisions", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Longitude (°E)", fontsize=11)
axes[0].set_ylabel("Latitude (°N)", fontsize=11)
axes[0].legend(loc="upper left", fontsize=8.5, frameon=True)

# 1b. Topographic Elevation and Coast Distance Profile
sc = axes[1].scatter(cells["dist_to_coast_km"], cells["elevation_dem_m"], c=cells["subdivision"].map(subdiv_palette), s=70, edgecolors="k", linewidth=0.5)
axes[1].axvline(50, color="gray", linestyle=":", label="Coastal Plain / Scarp Transition (~50 km)")
axes[1].set_title("B. Orographic Profile: Elevation vs. Distance to Arabian Sea", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Distance to Arabian Sea Coast (km)", fontsize=11)
axes[1].set_ylabel("Copernicus DEM Elevation (m)", fontsize=11)

# Annotate coastal fallback cells
for _, r in fallback_cells.iterrows():
    axes[1].scatter(r["dist_to_coast_km"], r["elevation_dem_m"], facecolors="none", edgecolors="red", s=180, linewidth=2.0)
    axes[1].annotate(f"Cell {int(r['station_id'])}", (r["dist_to_coast_km"], r["elevation_dem_m"]), textcoords="offset points", xytext=(8, -3), fontsize=8, color="red", fontweight="bold")

axes[1].legend(loc="upper right", fontsize=9)

plt.tight_layout()
save_pub_fig(fig, "figure1_study_area")

# ==============================================================================
# FIGURE 2: METHODOLOGICAL WORKFLOW
# ==============================================================================
print("\n--- Generating Figure 2: Methodological Workflow ---")
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.axis("off")

boxes = [
    ("1. Raw Data Ingestion", "• ERA5-Land Reanalysis (0.10°, hourly, 2019–2025)\n• IMD Gridded Rainfall (0.25°, daily, 2019–2025)\n• Copernicus DEM GLO-90 Topography (90m)", 0.05, 0.70, 0.26, 0.22, "#dbe9f6"),
    ("2. Temporal Alignment & De-accumulation", "• ERA5 forecast steps de-accumulated to hourly\n• 24-hr sum synchronized to 03:00–03:00 UTC cycle\n• Aligned exactly to IMD 08:30 IST reporting window", 0.37, 0.70, 0.26, 0.22, "#dbe9f6"),
    ("3. Spatial Matching & Fallback Protocol", "• 252 in-state 0.25° IMD grid cell centroids\n• Nearest-inland extraction for 3 fallback cells\n• Topographic macro-slope across 27.75 km baseline", 0.69, 0.70, 0.26, 0.22, "#dbe9f6"),
    
    ("4. Leakage-Free Feature Engineering", "• 63 primary physical predictors\n• 14-day antecedent IMD rainfall & API\n• Surface runoff, soil moisture, radiation, wind, temp\n• Target encoding strictly on Training partition", 0.05, 0.38, 0.26, 0.22, "#e2f0d9"),
    ("5. Chronological Dataset Partitioning", "• Training (2019-01-15 to 2022-12-31, N=364,644)\n• Validation (2023-01-01 to 2023-12-31, N=91,980)\n• Test [FROZEN] (2024-01-01 to 2025-12-31, N=184,212)", 0.37, 0.38, 0.26, 0.22, "#e2f0d9"),
    ("6. Model Development Hierarchy", "• Baseline Benchmarks (Persistence, Climatology, EQM)\n• Tweedie Gradient Boosters (LightGBM, CatBoost)\n• Two-Stage Hurdle Architecture (Occurrence + Amount)\n• Extreme Classifiers (LightGBM Focal Loss)", 0.69, 0.38, 0.26, 0.22, "#e2f0d9"),
    
    ("7. Independent Statistical Audit", "• 100% metric recalculation & reconciliation\n• Analytical & finite-diff focal loss Hessian audit\n• Paired stationary block bootstrap (B=5,000, L=7d)\n• Duan smearing residual heteroscedasticity audit", 0.15, 0.06, 0.32, 0.22, "#fce4d6"),
    ("8. Robustness & Reproducibility", "• Yearly consistency & multi-threshold sensitivity\n• Regional & seasonal breakdown across subdivisions\n• Extreme quantile shrinkage boundary profiling\n• Publication package with SHA-256 manifest", 0.53, 0.06, 0.32, 0.22, "#fce4d6")
]

for title, desc, x, y, w, h, bg in boxes:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", edgecolor="#2b5c8f", facecolor=bg, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.035, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#1d3557")
    ax.text(x + 0.015, y + h - 0.075, desc, ha="left", va="top", fontsize=8.5, color="#222222", linespacing=1.4)

# Draw arrows between steps
arrow_props = dict(arrowstyle="->", color="#1d3557", lw=2, mutation_scale=15)
ax.annotate("", xy=(0.37, 0.81), xytext=(0.31, 0.81), arrowprops=arrow_props)
ax.annotate("", xy=(0.69, 0.81), xytext=(0.63, 0.81), arrowprops=arrow_props)
ax.annotate("", xy=(0.82, 0.60), xytext=(0.82, 0.70), arrowprops=arrow_props)
ax.annotate("", xy=(0.63, 0.49), xytext=(0.69, 0.49), arrowprops=arrow_props)
ax.annotate("", xy=(0.31, 0.49), xytext=(0.37, 0.49), arrowprops=arrow_props)
ax.annotate("", xy=(0.18, 0.28), xytext=(0.18, 0.38), arrowprops=arrow_props)
ax.annotate("", xy=(0.53, 0.17), xytext=(0.47, 0.17), arrowprops=arrow_props)

plt.tight_layout()
save_pub_fig(fig, "figure2_methodology")

# ==============================================================================
# FIGURE 3: OBSERVED VERSUS PREDICTED RAINFALL
# ==============================================================================
print("\n--- Generating Figure 3: Observed Versus Predicted Rainfall ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Subsample 25,000 points for scatter rendering
np.random.seed(42)
sample_idx = np.random.choice(len(y_true), size=30000, replace=False)
y_sub_true = y_true[sample_idx]
y_sub_pred = pred_lgb[sample_idx]

# 3a. Parity Scatter with Density Hexbin
hb = axes[0].hexbin(y_true, pred_lgb, gridsize=60, cmap="Blues", mincnt=1, bins="log", extent=[0, 150, 0, 150])
axes[0].plot([0, 150], [0, 150], "r--", linewidth=1.5, label="1:1 Perfect Parity Line")
cbar0 = plt.colorbar(hb, ax=axes[0])
cbar0.set_label("log10(Instance Density)", fontsize=10)

# Regression fit line
m_slope, m_int = np.polyfit(y_true, pred_lgb, 1)
axes[0].plot([0, 150], [m_int, m_slope * 150 + m_int], color="black", linestyle="-", linewidth=1.8, label=f"Linear Fit (Slope: {m_slope:.2f})")

axes[0].set_title("A. Density Parity: Observed vs. LightGBM Tweedie (Test N=184,212)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Observed IMD Rainfall (mm)", fontsize=10)
axes[0].set_ylabel("Predicted Rainfall (mm)", fontsize=10)
axes[0].set_xlim(0, 150)
axes[0].set_ylim(0, 150)
axes[0].legend(loc="upper left", fontsize=9)

# 3b. Quantile-Quantile (Q-Q) Parity Curve
q_levels = np.linspace(0.01, 0.9995, 300)
q_obs = np.quantile(y_true, q_levels)
q_lgb_fine = np.quantile(pred_lgb, q_levels)
q_cat_fine = np.quantile(pred_cat, q_levels)
q_hurdle_fine = np.quantile(pred_hurdle, q_levels)
q_era5_fine = np.quantile(pred_era5, q_levels)

axes[1].plot([0, 180], [0, 180], "k--", linewidth=1.5, label="1:1 Perfect Quantile Line")
axes[1].plot(q_obs, q_lgb_fine, color="#1d3557", linewidth=2.2, label="LightGBM Tweedie")
axes[1].plot(q_obs, q_cat_fine, color="#2a9d8f", linewidth=2.0, linestyle="-.", label="CatBoost Tweedie")
axes[1].plot(q_obs, q_hurdle_fine, color="#e76f51", linewidth=1.8, linestyle=":", label="Hurdle Expected Value")
axes[1].plot(q_obs, q_era5_fine, color="gray", linewidth=1.5, linestyle="--", label="Raw ERA5-Land")

axes[1].axvline(64.5, color="crimson", linestyle=":", label="Official Heavy Rain (64.5 mm)")
axes[1].axvline(115.6, color="darkred", linestyle=":", label="Official Very Heavy Rain (115.6 mm)")
axes[1].set_title("B. Quantile Parity Curve: Extreme Tail Quantile Deficit", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Observed IMD Quantiles (mm)", fontsize=10)
axes[1].set_ylabel("Predicted Quantiles (mm)", fontsize=10)
axes[1].set_xlim(0, 180)
axes[1].set_ylim(0, 180)
axes[1].legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
save_pub_fig(fig, "figure3_observed_predicted")

# ==============================================================================
# FIGURE 4: MODEL BENCHMARKING
# ==============================================================================
print("\n--- Generating Figure 4: Model Benchmarking ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

df_bench = pd.read_csv(out_dir / "publication_tables/table3_model_benchmarking.csv")
# Filter down to the 6 core comparison models
bench_models = [
    "LightGBM Tweedie p1.5",
    "CatBoost Tweedie p1.5",
    "Hurdle ExpectedValue",
    "Rainfall Persistence",
    "Antecedent ERA5 Land",
    "Empirical Quantile Mapping EQM"
]
bench_labels = [
    "LightGBM Tweedie",
    "CatBoost Tweedie",
    "Hurdle Expected Value",
    "Persistence",
    "Raw ERA5-Land",
    "EQM"
]
df_b_plot = df_bench[df_bench["Model_Name"].isin(bench_models)].copy()
df_b_plot["Clean_Label"] = df_b_plot["Model_Name"].map(dict(zip(bench_models, bench_labels)))

model_colors = ["#1d3557", "#2a9d8f", "#e76f51", "#f4a261", "#6c757d", "#457b9d"]

# 4a. RMSE (mm)
axes[0, 0].bar(range(len(df_b_plot)), df_b_plot["RMSE_mm"], color=model_colors)
axes[0, 0].set_xticks(range(len(df_b_plot)))
axes[0, 0].set_xticklabels(df_b_plot["Clean_Label"], rotation=20, ha="right", fontsize=9.5)
axes[0, 0].set_title("A. Root Mean Squared Error (RMSE, mm)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("RMSE (mm)", fontsize=11)
axes[0, 0].set_ylim(0, 13)

# 4b. MAE (mm)
axes[0, 1].bar(range(len(df_b_plot)), df_b_plot["MAE_mm"], color=model_colors)
axes[0, 1].set_xticks(range(len(df_b_plot)))
axes[0, 1].set_xticklabels(df_b_plot["Clean_Label"], rotation=20, ha="right", fontsize=9.5)
axes[0, 1].set_title("B. Mean Absolute Error (MAE, mm)", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("MAE (mm)", fontsize=11)
axes[0, 1].set_ylim(0, 5)

# 4c. Pearson Correlation (r)
axes[1, 0].bar(range(len(df_b_plot)), df_b_plot["Pearson_r"], color=model_colors)
axes[1, 0].set_xticks(range(len(df_b_plot)))
axes[1, 0].set_xticklabels(df_b_plot["Clean_Label"], rotation=20, ha="right", fontsize=9.5)
axes[1, 0].set_title("C. Pearson Correlation Coefficient (r)", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Correlation (r)", fontsize=11)
axes[1, 0].set_ylim(0, 0.9)

# 4d. Modified KGE-2012
axes[1, 1].bar(range(len(df_b_plot)), df_b_plot["Modified_KGE_2012"], color=model_colors)
axes[1, 1].set_xticks(range(len(df_b_plot)))
axes[1, 1].set_xticklabels(df_b_plot["Clean_Label"], rotation=20, ha="right", fontsize=9.5)
axes[1, 1].set_title("D. Modified Kling-Gupta Efficiency (KGE-2012)", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("KGE-2012", fontsize=11)
axes[1, 1].set_ylim(0, 0.8)

plt.tight_layout()
save_pub_fig(fig, "figure4_model_benchmarking")

# ==============================================================================
# FIGURE 5: HEAVY RAIN DETECTION (>= 64.5 mm)
# ==============================================================================
print("\n--- Generating Figure 5: Heavy Rain Detection ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

df_h_src = pd.read_csv(out_dir / "publication_tables/table4_heavy_rain_classification.csv")

# 5a. Contingency Counts (Hits, Misses, False Alarms)
models_h = ["LightGBM Focal Loss", "LightGBM Class-Weighted", "CatBoost Class-Weighted", "Rainfall Persistence", "Raw ERA5-Land", "Empirical Quantile Mapping"]
clean_labels_h = ["LGBM Focal", "LGBM Weighted", "CatBoost Weighted", "Persistence", "Raw ERA5", "EQM"]

hits_h = df_h_src["Hits"].values
misses_h = df_h_src["Misses"].values
fa_h = df_h_src["False_Alarms"].values

x_idx = np.arange(len(hits_h))
w = 0.26

axes[0].bar(x_idx - w, hits_h, width=w, label="Hits (Correct Detections)", color="#2a9d8f")
axes[0].bar(x_idx, misses_h, width=w, label="Misses (Undetected Events)", color="#e76f51")
axes[0].bar(x_idx + w, fa_h, width=w, label="False Alarms (Spurious Warnings)", color="#f4a261")
axes[0].set_xticks(x_idx)
axes[0].set_xticklabels(clean_labels_h, rotation=15, fontsize=10)
axes[0].set_title("A. Heavy-Rain Contingency Table Frequencies (≥64.5 mm, Test N=184,212)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Event Count", fontsize=10)
axes[0].legend(fontsize=9)

# 5b. Categorical Skill Metrics
prec_h = df_h_src["Precision"].values
pod_h = df_h_src["POD_Recall"].values
csi_h = df_h_src["CSI"].values

axes[1].bar(x_idx - w, prec_h, width=w, label="Precision (1 - FAR)", color="#264653")
axes[1].bar(x_idx, pod_h, width=w, label="Recall / POD", color="#457b9d")
axes[1].bar(x_idx + w, csi_h, width=w, label="Critical Success Index (CSI)", color="#e63946")
axes[1].set_xticks(x_idx)
axes[1].set_xticklabels(clean_labels_h, rotation=15, fontsize=10)
axes[1].set_title("B. Heavy-Rain Skill Scores: Precision, Recall and Threat Score (CSI)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Score (0 to 1)", fontsize=10)
axes[1].set_ylim(0, 0.75)
axes[1].legend(fontsize=9)

plt.tight_layout()
save_pub_fig(fig, "figure5_heavy_rain_detection")

# ==============================================================================
# FIGURE 6: FEATURE IMPORTANCE AND PHYSICAL INTERPRETATION
# ==============================================================================
print("\n--- Generating Figure 6: Feature Importance and Physical Interpretation ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

df_perm = pd.read_csv(stage5_dir / "permutation_importance.csv").head(12)
df_grp = pd.read_csv(stage6_dir / "grouped_feature_sensitivity.csv").sort_values("sum_delta_rmse_mm", ascending=True)

# 6a. Top 12 Permutation Features
axes[0].barh(range(len(df_perm)), df_perm["delta_rmse_mean"], xerr=df_perm["delta_rmse_std"], color="#2b5c8f", align="center", capsize=3)
axes[0].set_yticks(range(len(df_perm)))
axes[0].set_yticklabels(df_perm["feature"], fontsize=10)
axes[0].invert_yaxis()
axes[0].set_title("A. Validation Permutation Importance: Individual Predictors (N=91,980)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Increase in Validation RMSE (mm)", fontsize=10)

# 6b. Grouped Physical Domains
clean_domains = [d.split(". ")[-1] for d in df_grp["physical_domain"]]
axes[1].barh(range(len(df_grp)), df_grp["sum_delta_rmse_mm"], color="#e07a5f", align="center")
axes[1].set_yticks(range(len(df_grp)))
axes[1].set_yticklabels(clean_domains, fontsize=10)
axes[1].set_title("B. Grouped Physical Domain Sensitivity (Aggregate Permutation ΔRMSE)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Aggregate Validation ΔRMSE Increase (mm)", fontsize=10)

for idx, r in df_grp.reset_index(drop=True).iterrows():
    axes[1].text(r["sum_delta_rmse_mm"] + 0.08, idx, f"Lead: {r['leading_feature']}", va="center", fontsize=8.5, color="#333333")

plt.tight_layout()
save_pub_fig(fig, "figure6_feature_interpretation")

# ==============================================================================
# FIGURE 7: REGIONAL AND SEASONAL PERFORMANCE
# ==============================================================================
print("\n--- Generating Figure 7: Regional and Seasonal Performance ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

df_t5_src = pd.read_csv(out_dir / "publication_tables/table5_regional_seasonal_performance.csv")
df_t5_reg = df_t5_src[df_t5_src["Stratum_Type"] == "Meteorological Subdivision"].copy()
df_t5_seas = df_t5_src[df_t5_src["Stratum_Type"] == "Seasonal Regime"].copy()

clean_reg_labels = ["Coastal (13 cells)", "Malnad (35 cells)", "North Interior (118 cells)", "South Interior (86 cells)", "Fallback (3 cells)", "Native Coastal (10 cells)"]
df_t5_reg["Clean_Label"] = clean_reg_labels

# 7a. Regional RMSE
w = 0.28
x_r = np.arange(len(df_t5_reg))
axes[0, 0].bar(x_r - w/2, df_t5_reg["ML_RMSE_mm"], width=w, label="LightGBM Tweedie", color="#1d3557")
axes[0, 0].bar(x_r + w/2, df_t5_reg["Raw_ERA5_RMSE_mm"], width=w, label="Raw ERA5-Land", color="#e76f51")
axes[0, 0].set_xticks(x_r)
axes[0, 0].set_xticklabels(df_t5_reg["Clean_Label"], rotation=20, ha="right", fontsize=9.5)
axes[0, 0].set_title("A. Regional RMSE Comparison (ML vs. Raw ERA5)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("RMSE (mm)", fontsize=11)
axes[0, 0].legend(fontsize=9)

# 7b. Regional KGE-2012
axes[0, 1].bar(x_r, df_t5_reg["ML_KGE_2012"], width=0.5, color="#2a9d8f")
axes[0, 1].set_xticks(x_r)
axes[0, 1].set_xticklabels(df_t5_reg["Clean_Label"], rotation=20, ha="right", fontsize=9.5)
axes[0, 1].set_title("B. Modified Kling-Gupta Efficiency (KGE-2012) by Region", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("KGE-2012", fontsize=11)
axes[0, 1].set_ylim(0, 0.9)

# 7c. Seasonal RMSE
clean_seas_labels = ["SW Monsoon (JJAS)", "Post-Monsoon (OND)", "Pre-Monsoon (MAM)", "Winter (JF)"]
df_t5_seas["Clean_Label"] = clean_seas_labels
x_s = np.arange(len(df_t5_seas))

axes[1, 0].bar(x_s - w/2, df_t5_seas["ML_RMSE_mm"], width=w, label="LightGBM Tweedie", color="#1d3557")
axes[1, 0].bar(x_s + w/2, df_t5_seas["Raw_ERA5_RMSE_mm"], width=w, label="Raw ERA5-Land", color="#e76f51")
axes[1, 0].set_xticks(x_s)
axes[1, 0].set_xticklabels(df_t5_seas["Clean_Label"], fontsize=10)
axes[1, 0].set_title("C. Seasonal RMSE Comparison across Annual Cycle", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("RMSE (mm)", fontsize=11)
axes[1, 0].legend(fontsize=9)

# 7d. Seasonal Correlation (r)
axes[1, 1].bar(x_s, df_t5_seas["ML_Pearson_r"], width=0.5, color="#457b9d")
axes[1, 1].set_xticks(x_s)
axes[1, 1].set_xticklabels(df_t5_seas["Clean_Label"], fontsize=10)
axes[1, 1].set_title("D. Pearson Correlation (r) across Seasonal Regimes", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Correlation (r)", fontsize=11)
axes[1, 1].set_ylim(0, 0.9)

plt.tight_layout()
save_pub_fig(fig, "figure7_regional_seasonal")

# ==============================================================================
# FIGURE 8: EXTREME RAINFALL ERROR ANALYSIS
# ==============================================================================
print("\n--- Generating Figure 8: Extreme Rainfall Error Analysis ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

df_t8_src = pd.read_csv(out_dir / "publication_tables/table8_rainfall_intensity_limitations.csv")

clean_cats = [
    "No Rain\n(<0.1mm)", "Very Light\n(0.1–2.4mm)", "Light\n(2.5–15.5mm)",
    "Moderate\n(15.6–64.4mm)", "Heavy\n(64.5–115.5mm)", "Very Heavy\n(115.6–204.4mm)", "Extremely Heavy\n(≥204.5mm)"
]

# 8a. Mean Bias by Category
axes[0].bar(range(len(df_t8_src)), df_t8_src["Mean_Bias_mm"], color=["#457b9d" if b >= 0 else "#e63946" for b in df_t8_src["Mean_Bias_mm"]])
axes[0].axhline(0, color="black", linestyle="-", linewidth=0.8)
axes[0].set_xticks(range(len(df_t8_src)))
axes[0].set_xticklabels(clean_cats, fontsize=9)
axes[0].set_title("A. Conditional Mean Bias (mm) across 7 IMD Intensity Categories", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Mean Bias (Predicted - Observed, mm)", fontsize=11)

for idx, r in df_t8_src.iterrows():
    val_str = f"{r['Mean_Bias_mm']:+.1f}mm\n({r['Percentage_Bias_pct']:+.0f}%)" if pd.notna(r['Percentage_Bias_pct']) else f"{r['Mean_Bias_mm']:+.2f}mm"
    axes[0].text(idx, r["Mean_Bias_mm"] + (4 if r["Mean_Bias_mm"] >= 0 else -18), val_str, ha="center", fontsize=8, fontweight="bold")

# 8b. Percentage of Instances Underestimated
axes[1].bar(range(len(df_t8_src)), df_t8_src["Percent_Underestimated"], color="#d62828")
axes[1].axhline(50, color="gray", linestyle="--", linewidth=1.2, label="50% Median Parity Line")
axes[1].set_xticks(range(len(df_t8_src)))
axes[1].set_xticklabels(clean_cats, fontsize=9)
axes[1].set_title("B. Proportion of Instances Underestimated (%) by Intensity Category", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Underestimated Observations (%)", fontsize=11)
axes[1].set_ylim(0, 115)
axes[1].legend(loc="upper left", fontsize=9)

for idx, r in df_t8_src.iterrows():
    axes[1].text(idx, r["Percent_Underestimated"] + 2.5, f"{r['Percent_Underestimated']:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

plt.tight_layout()
save_pub_fig(fig, "figure8_extreme_rainfall_errors")

# ==============================================================================
# FIGURE 9: TEMPORAL ROBUSTNESS & BOOTSTRAP UNCERTAINTY
# ==============================================================================
print("\n--- Generating Figure 9: Temporal Robustness ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

df_yearly = pd.read_csv(stage6_dir / "yearly_performance_results.csv")
y_compare = df_yearly[df_yearly["temporal_stratum"].isin(["Year_2024_Full", "Year_2025_Full"])].copy()
y_compare["Year"] = y_compare["temporal_stratum"].str.extract(r"Year_(\d+)_Full")
y_compare["Clean_Model"] = y_compare["model"].map({
    "LightGBM_Tweedie": "LightGBM Tweedie",
    "CatBoost_Tweedie": "CatBoost Tweedie",
    "Hurdle_ExpectedValue": "Hurdle EV",
    "Persistence": "Persistence",
    "Raw_ERA5": "Raw ERA5",
    "EQM": "EQM"
})

# 9a. Yearly Consistency (2024 vs 2025)
sns.barplot(data=y_compare, x="Clean_Model", y="RMSE", hue="Year", ax=axes[0], palette=["#2a9d8f", "#e76f51"])
axes[0].set_title("A. Annual RMSE Consistency: 2024 vs. 2025 Forward Evaluation", fontsize=12, fontweight="bold")
axes[0].set_ylabel("RMSE (mm)", fontsize=11)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend(title="Calendar Year", fontsize=9)

# 9b. Paired Bootstrap Confidence Intervals (7-day vs 14-day)
df_boot = pd.read_csv(out_dir / "publication_tables/table6_statistical_robustness.csv")
df_b_tweedie = df_boot[df_boot["Model_Pair"].str.startswith("LightGBM_Tweedie")].copy()

pair_labels = ["vs. Persistence (L=7d)", "vs. Persistence (L=14d)", "vs. EQM (L=7d)", "vs. EQM (L=14d)", "vs. Raw ERA5 (L=7d)", "vs. Raw ERA5 (L=14d)"]
y_pos = np.arange(len(df_b_tweedie))

axes[1].errorbar(df_b_tweedie["Mean_Delta_RMSE_mm"], y_pos,
                xerr=[df_b_tweedie["Mean_Delta_RMSE_mm"] - df_b_tweedie["CI_95_Low_mm"],
                      df_b_tweedie["CI_95_High_mm"] - df_b_tweedie["Mean_Delta_RMSE_mm"]],
                fmt="o", color="#1d3557", ecolor="#e63946", elinewidth=2.2, capsize=4, markersize=7)

axes[1].axvline(0, color="black", linestyle="--", linewidth=1.0, label="Zero Difference (Parity)")
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(pair_labels, fontsize=9.5)
axes[1].invert_yaxis()
axes[1].set_title("B. Paired Block Bootstrap ΔRMSE (95% CI, B=2,000 Replicates)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Paired RMSE Reduction: Model - Baseline (mm)", fontsize=11)
axes[1].legend(loc="lower right", fontsize=9)

plt.tight_layout()
save_pub_fig(fig, "figure9_temporal_robustness")

print("\n--- All Publication Figures Successfully Generated in PNG & PDF ---")

