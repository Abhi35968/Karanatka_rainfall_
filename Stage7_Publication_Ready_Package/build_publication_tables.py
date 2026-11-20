import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
stage3_dir = base_dir / "Stage3_4_Final_Release_Audit"
stage4_1_dir = base_dir / "Stage4_1_Model_Development"
stage4_2_dir = base_dir / "Stage4_2_Independent_Model_Audit"
stage5_dir = base_dir / "Stage5_Model_Interpretation"
stage6_dir = base_dir / "Stage6_Robustness_Generalisation"

out_dir = base_dir / "Stage7_Publication_Ready_Package"
tables_dir = out_dir / "publication_tables"
tables_dir.mkdir(parents=True, exist_ok=True)

print("--- 1. Building Master Results Table ---")
# Load audited recalculations from Stage 4.2
df_audit_recalc = pd.read_csv(stage4_2_dir / "independent_metric_recalculation.csv")

master_rows = []
for _, row in df_audit_recalc.iterrows():
    task_map = {
        "baseline_results.csv": "Baseline Benchmarking",
        "regression_model_results.csv": "Continuous Rainfall Regression",
        "hurdle_model_results.csv": "Two-Stage Hurdle Architecture",
        "rain_occurrence_results.csv": "Rain Occurrence Classification (>=0.1mm)",
        "heavy_rain_results.csv": "Heavy Rain Classification (>=64.5mm)"
    }
    task_desc = task_map.get(row["source_file"], "Rainfall Modeling")
    
    # Calculate percentage bias if possible
    pbias = None
    if pd.notna(row["recalculated_Bias"]):
        # Mean observed in val is ~2.35, test is ~3.51
        denom = 2.3571 if "Validation" in str(row["partition"]) else 3.5085
        pbias = round(100.0 * float(row["recalculated_Bias"]) / denom, 2)
        
    master_rows.append({
        "Model_Name": row["model_name"],
        "Prediction_Task": task_desc,
        "Dataset_Split": row["partition"],
        "Sample_Count": row["sample_count"],
        "RMSE_mm": row["recalculated_RMSE"],
        "MAE_mm": row["recalculated_MAE"],
        "Mean_Bias_mm": row["recalculated_Bias"],
        "PBias_pct": pbias,
        "Pearson_r": row["recalculated_Pearson_r"],
        "NSE": row["recalculated_NSE"],
        "Classic_KGE_2009": row["recalculated_KGE_classic_2009"],
        "Modified_KGE_2012": row["recalculated_KGE_2012"],
        "Heavy_CSI_64.5mm": row["recalculated_CSI_64.5"],
        "Audited_Status": row["reconciled_status"]
    })

df_master = pd.DataFrame(master_rows)
df_master.to_csv(out_dir / "master_results_table.csv", index=False)
print(f"Saved master_results_table.csv ({len(df_master)} rows)")

# ==============================================================================
# TABLE 1 — DATASET CHARACTERISTICS
# ==============================================================================
print("\n--- 2. Building Table 1: Dataset Characteristics ---")
table1_data = [
    {
        "Data_Source": "IMD Gridded Daily Rainfall (Target)",
        "Provider": "India Meteorological Department (National Climate Centre, Pune)",
        "Spatial_Resolution": "0.25° × 0.25° (~27.75 km)",
        "Temporal_Resolution": "Daily accumulation (03:00 UTC Day d-1 to 03:00 UTC Day d / 08:30 IST)",
        "Spatial_Coverage": "Karnataka State (252 in-state land cells, Lat 11.75°N–18.50°N, Lon 74.25°E–78.50°E)",
        "Temporal_Span": "2019-01-01 to 2025-12-31 (7 calendar years, 2,557 days)",
        "Total_Grid_Cells": 252,
        "Total_Records": 640836,
        "Training_Records": "364,644 (2019-01-15 to 2022-12-31, 1,447 days; 14-day lag warmup buffer excluded)",
        "Validation_Records": "91,980 (2023-01-01 to 2023-12-31, 365 days; regional drought year)",
        "Test_Records": "184,212 (2024-01-01 to 2025-12-31, 731 days; 366d in 2024 leap year + 365d in 2025)"
    },
    {
        "Data_Source": "ERA5-Land Global Atmospheric Reanalysis (Predictors)",
        "Provider": "ECMWF / Copernicus Climate Change Service (C3S)",
        "Spatial_Resolution": "0.10° × 0.10° (~9 km) aggregated to IMD 0.25° centroids",
        "Temporal_Resolution": "Hourly raw, de-accumulated and temporally aligned to IMD 03:00–03:00 UTC cycle",
        "Spatial_Coverage": "Karnataka state domain plus coastal maritime fallback buffer",
        "Temporal_Span": "2019-01-01 to 2025-12-31 (84 monthly NetCDF files)",
        "Total_Grid_Cells": 252,
        "Total_Records": 640836,
        "Training_Records": "364,644 (Zero missing or infinite values across all 63 primary features)",
        "Validation_Records": "91,980 (Zero missing values; strictly out-of-sample temporal holdout)",
        "Test_Records": "184,212 (Zero missing values; frozen out-of-sample temporal evaluation)"
    },
    {
        "Data_Source": "Copernicus GLO-90 Digital Elevation Model (Terrain)",
        "Provider": "European Space Agency (ESA) Copernicus Programme",
        "Spatial_Resolution": "90 m native resolution, resampled to 0.25° grid centroids and macro-gradient slope",
        "Temporal_Resolution": "Static topographic surface",
        "Spatial_Coverage": "Centroid point elevation (m) and 27.75 km baseline macro-slope (degrees)",
        "Temporal_Span": "Static baseline",
        "Total_Grid_Cells": 252,
        "Total_Records": 252,
        "Training_Records": "Identical static properties across all 252 IMD grid cells",
        "Validation_Records": "Identical static properties across all 252 IMD grid cells",
        "Test_Records": "Identical static properties across all 252 IMD grid cells"
    }
]
df_table1 = pd.DataFrame(table1_data)
df_table1.to_csv(tables_dir / "table1_dataset_characteristics.csv", index=False)
print("Saved table1_dataset_characteristics.csv")

# ==============================================================================
# TABLE 2 — FEATURE GROUPS
# ==============================================================================
print("\n--- 3. Building Table 2: Feature Groups ---")
table2_data = [
    {
        "Feature_Group": "1. Antecedent IMD Rainfall Memory",
        "Predictor_Count": 10,
        "Representative_Variables": "target_rain_lag_1d, target_rain_lag_2d, target_rain_lag_3d, target_rain_lag_7d, target_rain_lag_14d, target_rain_roll_3d_sum, target_rain_roll_7d_sum, target_rain_roll_14d_sum, api_14d",
        "Temporal_Alignment": "03:00 UTC Day d-1 (Lag 1) to Day d-14",
        "Scientific_Relevance": "Ground-truth antecedent precipitation persistence, catchment moisture memory, and antecedent precipitation index (decay k=0.85)"
    },
    {
        "Feature_Group": "2. ERA5-Land Precipitation & Hydrological Fluxes",
        "Predictor_Count": 15,
        "Representative_Variables": "era5_tp_lag_1d, era5_tp_roll_3d_sum, era5_tp_roll_7d_sum, ro_lag_1d (total runoff), sro_lag_1d (surface runoff), src_lag_1d (skin reservoir content), e_lag_1d, pev_lag_1d, evabs_lag_1d, evatc_lag_1d, evavt_lag_1d, fal_lag_1d",
        "Temporal_Alignment": "De-accumulated daily sum 03:00 UTC Day d-2 to 03:00 UTC Day d-1",
        "Scientific_Relevance": "Numerical model precipitation forecast memory, surface saturation overland flow, canopy interception storage, and evapotranspiration demand"
    },
    {
        "Feature_Group": "3. Atmospheric Thermodynamic State & Wind",
        "Predictor_Count": 12,
        "Representative_Variables": "t2m_mean_lag_1d, t2m_min_lag_1d, t2m_max_lag_1d, temp_range_lag_1d, d2m_mean_lag_1d, dewpoint_depression_lag_1d, t2m_roll_3d_mean, t2m_roll_7d_mean, sp_lag_1d, wind_speed_lag_1d, wind_roll_3d_mean, wind_roll_7d_mean",
        "Temporal_Alignment": "Daily mean, min, max over 24-hr cycle ending 03:00 UTC Day d-1",
        "Scientific_Relevance": "Boundary-layer thermal structure, moisture saturation deficit (sub-cloud relative humidity), synoptic surface pressure, and low-level jet (LLJ) moisture transport"
    },
    {
        "Feature_Group": "4. Soil Moisture Profiles & Energy Radiation Fluxes",
        "Predictor_Count": 9,
        "Representative_Variables": "swvl1_lag_1d (0-7 cm), swvl1_lag_3d, swvl1_lag_7d, swvl2_lag_1d (7-28 cm), swvl3_lag_1d (28-100 cm), ssrd_lag_1d (solar downwards), strd_lag_1d (thermal downwards), slhf_lag_1d, sshf_lag_1d",
        "Temporal_Alignment": "Layer volumetric water content and daily accumulated radiative/turbulent energy fluxes",
        "Scientific_Relevance": "Root-zone soil water storage, cloud-cover radiative trapping (downward longwave), clear-sky solar insolation, and surface sensible/latent heat partition"
    },
    {
        "Feature_Group": "5. Topography, Elevation & Spatial Proximity",
        "Predictor_Count": 6,
        "Representative_Variables": "elevation_dem_m, slope_dem_deg, latitude, longitude, dist_to_coast_km, dist_to_ghats_crest_km",
        "Temporal_Alignment": "Static spatial coordinates and Copernicus DEM GLO-90 terrain metrics",
        "Scientific_Relevance": "Orographic uplift forcing along the Western Ghats barrier, maritime-continental continentality gradient, and regional terrain macro-slope"
    },
    {
        "Feature_Group": "6. Temporal Ephemeris & Astronomical Cycles",
        "Predictor_Count": 7,
        "Representative_Variables": "month, day_of_year, sin_day_of_year, cos_day_of_year, sin_month, cos_month, is_monsoon",
        "Temporal_Alignment": "Harmonic cyclical transformation of Julian date and calendar indicators",
        "Scientific_Relevance": "Synoptic seasonal climatology, solar declination angle, and Southwest Monsoon boundary definition"
    },
    {
        "Feature_Group": "7. Categorical Regional Encodings & Auxiliary Flags",
        "Predictor_Count": 4,
        "Representative_Variables": "subdivision (Coastal, Malnad, North Interior, South Interior), zone_id (10 agroclimatic zones), subdivision_enc, zone_id_enc, is_coastal_fallback",
        "Temporal_Alignment": "Target-encoded strictly on Training partition (2019-2022); coastal fallback indicator (0/1)",
        "Scientific_Relevance": "Regional agroclimatic baselines and forensic identification of marine water nearest-neighbor extraction cells"
    }
]
df_table2 = pd.DataFrame(table2_data)
df_table2.to_csv(tables_dir / "table2_feature_groups.csv", index=False)
print("Saved table2_feature_groups.csv")

# ==============================================================================
# TABLE 3 — MODEL BENCHMARKING (TEST PARTITION 2024–2025)
# ==============================================================================
print("\n--- 4. Building Table 3: Model Benchmarking ---")
df_reg_bench = pd.read_csv(stage4_2_dir / "independent_metric_recalculation.csv")
test_bench = df_reg_bench[df_reg_bench["partition"] == "Test_2024_2025"].copy()

table3_models = [
    "Rainfall_Persistence",
    "Station_DOY_Climatology",
    "Antecedent_ERA5_Land",
    "Empirical_Quantile_Mapping_EQM",
    "LightGBM_RMSE",
    "LightGBM_Log_OrdinaryInverse",
    "LightGBM_Log_DuanSmearing",
    "LightGBM_Tweedie_p1.5",
    "CatBoost_Tweedie_p1.5",
    "Hurdle_ExpectedValue"
]

t3_rows = []
for m in table3_models:
    sub = test_bench[test_bench["model_name"] == m]
    if len(sub) > 0:
        r = sub.iloc[0]
        t3_rows.append({
            "Model_Hierarchy_Tier": "Tier 0 (Naive Baselines)" if "Persistence" in m or "Climatology" in m else ("Tier 1 (Raw Reanalysis)" if "Antecedent_ERA5" in m else ("Tier 2 (Statistical Correction)" if "EQM" in m else "Tier 4 (Machine Learning)")),
            "Model_Name": m.replace("_", " "),
            "Loss_Objective": "Zero Learning" if "Persistence" in m or "ERA5" in m or "EQM" in m or "Climatology" in m else ("RMSE (L2)" if "RMSE" in m else ("Log MSE" if "Log" in m else ("Compound Poisson-Gamma (p=1.5)" if "Tweedie" in m else "Two-Stage Hurdle Expected Value"))),
            "RMSE_mm": r["recalculated_RMSE"],
            "MAE_mm": r["recalculated_MAE"],
            "Mean_Bias_mm": r["recalculated_Bias"],
            "PBias_pct": round(100.0 * float(r["recalculated_Bias"]) / 3.5085, 2) if pd.notna(r["recalculated_Bias"]) else None,
            "Pearson_r": r["recalculated_Pearson_r"],
            "NSE": r["recalculated_NSE"],
            "Classic_KGE_2009": r["recalculated_KGE_classic_2009"],
            "Modified_KGE_2012": r["recalculated_KGE_2012"],
            "Heavy_CSI_64.5mm": r["recalculated_CSI_64.5"]
        })

df_table3 = pd.DataFrame(t3_rows)
df_table3.to_csv(tables_dir / "table3_model_benchmarking.csv", index=False)
print("Saved table3_model_benchmarking.csv")

# ==============================================================================
# TABLE 4 — HEAVY RAIN CLASSIFICATION CONTINGENCY (TEST PARTITION 2024–2025)
# ==============================================================================
print("\n--- 5. Building Table 4: Heavy Rain Classification ---")
df_t4_src = pd.read_csv(stage4_2_dir / "heavy_rain_confusion_matrices.csv")
table4_data = []
for _, r in df_t4_src.iterrows():
    table4_data.append({
        "Model_Name": r["model_name"],
        "Decision_Threshold": r["calibrated_decision_threshold"],
        "Threshold_Calibration_Source": r["threshold_selection_protocol"],
        "Hits": int(r["Hits"]),
        "Misses": int(r["Misses"]),
        "False_Alarms": int(r["False_Alarms"]),
        "Correct_Negatives": int(r["Correct_Negatives"]),
        "Precision": r["Precision"],
        "POD_Recall": r["POD_Recall"],
        "FAR": r["FAR"],
        "CSI": r["CSI_Threat_Score"],
        "HSS": r["HSS"],
        "ETS": r["ETS"],
        "PR_AUC": r["PR_AUC"]
    })

df_table4 = pd.DataFrame(table4_data)
df_table4.to_csv(tables_dir / "table4_heavy_rain_classification.csv", index=False)
print("Saved table4_heavy_rain_classification.csv")

# ==============================================================================
# TABLE 5 — REGIONAL AND SEASONAL PERFORMANCE
# ==============================================================================
print("\n--- 6. Building Table 5: Regional and Seasonal Performance ---")
df_reg_src = pd.read_csv(stage5_dir / "regional_performance_analysis.csv")
df_seas_src = pd.read_csv(stage5_dir / "seasonal_performance_analysis.csv")

t5_rows = []
# Regional
for r_name in [
    "Coastal_Karnataka_All_13_cells",
    "Malnad_35_cells",
    "North_Interior_Karnataka_118_cells",
    "South_Interior_Karnataka_86_cells",
    "Coastal_Fallback_Only_3_cells_IDs_52_68_125",
    "Coastal_Native_Land_Only_10_cells"
]:
    sub = df_reg_src[(df_reg_src["region"] == r_name) & (df_reg_src["model"] == "LightGBM_Tweedie")]
    sub_era5 = df_reg_src[(df_reg_src["region"] == r_name) & (df_reg_src["model"] == "Raw_ERA5")]
    sub_pers = df_reg_src[(df_reg_src["region"] == r_name) & (df_reg_src["model"] == "Persistence")]
    
    if len(sub) > 0:
        r = sub.iloc[0]
        re = sub_era5.iloc[0]
        rp = sub_pers.iloc[0]
        t5_rows.append({
            "Stratum_Type": "Meteorological Subdivision",
            "Stratum_Name": r_name.replace("_", " "),
            "Sample_Count": r["sample_count"],
            "Observed_Mean_mm": r["obs_mean"],
            "ML_RMSE_mm": r["RMSE"],
            "Persistence_RMSE_mm": rp["RMSE"],
            "Raw_ERA5_RMSE_mm": re["RMSE"],
            "ML_Mean_Bias_mm": r["Bias"],
            "ML_PBias_pct": r["PBias_pct"],
            "ML_Pearson_r": r["Pearson_r"],
            "ML_KGE_2012": r["KGE_2012"],
            "ML_Heavy_CSI_64.5mm": r["CSI_64.5mm"]
        })

# Seasonal
for s_name in [
    "Southwest_Monsoon_JJAS",
    "Post_Monsoon_Northeast_OND",
    "Pre_Monsoon_MAM",
    "Winter_JF"
]:
    sub = df_seas_src[(df_seas_src["season"] == s_name) & (df_seas_src["model"] == "LightGBM_Tweedie")]
    sub_era5 = df_seas_src[(df_seas_src["season"] == s_name) & (df_seas_src["model"] == "Raw_ERA5")]
    sub_pers = df_seas_src[(df_seas_src["season"] == s_name) & (df_seas_src["model"] == "Persistence")]
    
    if len(sub) > 0:
        r = sub.iloc[0]
        re = sub_era5.iloc[0]
        rp = sub_pers.iloc[0]
        t5_rows.append({
            "Stratum_Type": "Seasonal Regime",
            "Stratum_Name": s_name.replace("_", " "),
            "Sample_Count": r["sample_count"],
            "Observed_Mean_mm": r["obs_mean"],
            "ML_RMSE_mm": r["RMSE"],
            "Persistence_RMSE_mm": rp["RMSE"],
            "Raw_ERA5_RMSE_mm": re["RMSE"],
            "ML_Mean_Bias_mm": r["Bias"],
            "ML_PBias_pct": r["PBias_pct"],
            "ML_Pearson_r": r["Pearson_r"],
            "ML_KGE_2012": r["KGE_2012"],
            "ML_Heavy_CSI_64.5mm": r["CSI_64.5mm"]
        })

df_table5 = pd.DataFrame(t5_rows)
df_table5.to_csv(tables_dir / "table5_regional_seasonal_performance.csv", index=False)
print("Saved table5_regional_seasonal_performance.csv")

# ==============================================================================
# TABLE 6 — ROBUSTNESS AND STATISTICAL UNCERTAINTY
# ==============================================================================
print("\n--- 7. Building Table 6: Robustness and Statistical Uncertainty ---")
df_t6_src = pd.read_csv(stage6_dir / "paired_bootstrap_robustness.csv")
table6_data = []
for _, r in df_t6_src.iterrows():
    table6_data.append({
        "Model_Pair": f"{r['model_A']} vs. {r['baseline_B']}",
        "Block_Length_Days": r["block_length_days"],
        "Temporal_Memory_Assumption": "Synoptic weather persistence (7d)" if r["block_length_days"] == 7 else "Bi-weekly intraseasonal / MJO persistence (14d)",
        "Bootstrap_Replicates": r["replicates"],
        "Mean_Delta_RMSE_mm": r["mean_delta_rmse_mm"],
        "Standard_Error_mm": r["std_error_mm"],
        "CI_95_Low_mm": r["ci95_low_mm"],
        "CI_95_High_mm": r["ci95_high_mm"],
        "Replicates_Delta_ge_Zero": r["replicates_delta_ge_zero"],
        "Empirical_p_Value_Resolution": r["empirical_p_value"]
    })

df_table6 = pd.DataFrame(table6_data)
df_table6.to_csv(tables_dir / "table6_statistical_robustness.csv", index=False)
print("Saved table6_statistical_robustness.csv")

# ==============================================================================
# TABLE 7 — FEATURE ABLATION AND SENSITIVITY
# ==============================================================================
print("\n--- 8. Building Table 7: Feature Ablation ---")
df_abl_val = pd.read_csv(stage4_2_dir / "feature_ablation_validation_results.csv")
df_abl_te = pd.read_csv(stage4_2_dir / "feature_ablation_test_results.csv")

t7_rows = []
for _, r_te in df_abl_te.iterrows():
    sc_name = r_te["ablation_scenario"]
    sub_val = df_abl_val[df_abl_val["ablation_scenario"] == sc_name]
    r_val = sub_val.iloc[0] if len(sub_val) > 0 else {}
    
    clean_desc = sc_name.replace("Scenario_", "").replace("_", " ")
    sc_letter = clean_desc[0]
    
    t7_rows.append({
        "Scenario_ID": f"Scenario {sc_letter}",
        "Ablation_Description": clean_desc,
        "Features_Retained_Count": r_te["feature_count"],
        "Validation_RMSE_mm": r_val.get("RMSE", np.nan),
        "Validation_KGE_2012": r_val.get("KGE_2012", np.nan),
        "Validation_Heavy_CSI": r_val.get("CSI_64.5mm", np.nan),
        "Test_RMSE_mm": r_te["RMSE"],
        "Test_KGE_2012": r_te["KGE_2012"],
        "Test_Heavy_CSI": r_te["CSI_64.5mm"],
        "Delta_Test_RMSE_vs_Full_mm": round(r_te["RMSE"] - 7.4427, 4),
        "Scientific_Interpretation": "Champion full configuration" if sc_letter == "A" else (
            "Hydrological land fluxes provide critical moisture saturation proxies" if sc_letter == "B" else (
            "Local gauge memory essential for extreme threshold alerting" if sc_letter == "C" else (
            "Terrain captures spatial orographic boundaries" if sc_letter == "D" else (
            "ERA5 atmospheric predictors provide measurable incremental predictive value" if sc_letter == "E" else "Baseline autoregressive floor"
            )
        )))
    })

df_table7 = pd.DataFrame(t7_rows)
df_table7.to_csv(tables_dir / "table7_feature_ablation.csv", index=False)
print("Saved table7_feature_ablation.csv")

# ==============================================================================
# TABLE 8 — RAINFALL INTENSITY LIMITATIONS
# ==============================================================================
print("\n--- 9. Building Table 8: Rainfall Intensity Limitations ---")
df_t8_src = pd.read_csv(stage6_dir / "rainfall_intensity_sensitivity.csv")
t8_tweedie = df_t8_src[df_t8_src["model"] == "LightGBM_Tweedie"].copy()

table8_data = []
for _, r in t8_tweedie.iterrows():
    table8_data.append({
        "IMD_Rainfall_Intensity_Category": r["category"],
        "Sample_Count": int(r["sample_count"]),
        "Percent_Total_Test_Data": r["pct_total_dataset"],
        "Observed_Mean_mm": r["obs_mean_mm"],
        "Observed_Std_mm": r["obs_std_mm"],
        "Predicted_Mean_mm": r["pred_mean_mm"],
        "Predicted_Std_mm": r["pred_std_mm"],
        "MAE_mm": r["MAE_mm"],
        "RMSE_mm": r["RMSE_mm"],
        "Mean_Bias_mm": r["Mean_Bias_mm"],
        "Percentage_Bias_pct": r["PBias_pct"],
        "Percent_Underestimated": r["Percent_Underestimated"],
        "Structural_Limitation_Note": "Near-zero background drizzle inflation" if "No Rain" in r["category"] else (
            "Overestimation due to exponential link penalty" if "Very Light" in r["category"] else (
            "Optimal continuous calibration sweet spot" if "Light" in r["category"] else (
            "Progressive underestimation onset" if "Moderate" in r["category"] else (
            "Severe extreme quantile shrinkage (conditional expectation cap)"
            )
        )))
    })

df_table8 = pd.DataFrame(table8_data)
df_table8.to_csv(tables_dir / "table8_rainfall_intensity_limitations.csv", index=False)
print("Saved table8_rainfall_intensity_limitations.csv")

print("\n--- All Publication Tables Created Successfully ---")
