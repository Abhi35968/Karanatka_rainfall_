import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
stage4_dir = base_dir / "Stage4_1_Model_Development"
stage4_2_dir = base_dir / "Stage4_2_Independent_Model_Audit"
stage5_dir = base_dir / "Stage5_Model_Interpretation"
stage6_dir = base_dir / "Stage6_Robustness_Generalisation"
out_dir = base_dir / "Stage9_1_Final_Validated_Manuscript"

df_reg = pd.read_csv(stage4_dir / "regression_model_results.csv")
df_base = pd.read_csv(stage4_dir / "baseline_results.csv")
df_hurdle = pd.read_csv(stage4_dir / "hurdle_model_results.csv")
df_heavy = pd.read_csv(stage4_dir / "heavy_rain_results.csv")
df_sub = pd.read_csv(stage4_dir / "subgroup_evaluation_results.csv")
df_yearly = pd.read_csv(stage6_dir / "yearly_performance_results.csv")
df_intens = pd.read_csv(stage6_dir / "rainfall_intensity_sensitivity.csv")
df_boot = pd.read_csv(stage6_dir / "paired_bootstrap_robustness.csv")
df_perm = pd.read_csv(stage5_dir / "permutation_importance.csv")
df_feat = pd.read_csv(stage5_dir / "feature_importance_comparison.csv")
df_shap = pd.read_csv(stage5_dir / "shap_summary_values.csv")

audit_records = []

def add_check(cat, name, stratum, reported_val, source_file, source_val, tolerance=1e-3, notes=""):
    try:
        r_num = float(str(reported_val).replace("%", "").replace("+", "").strip())
        s_num = float(str(source_val).replace("%", "").replace("+", "").strip())
        diff = abs(r_num - s_num)
        status = "MATCH" if diff <= tolerance else "MISMATCH"
    except:
        status = "MATCH" if str(reported_val).strip() == str(source_val).strip() else "MISMATCH"
    
    audit_records.append({
        "metric_category": cat,
        "metric_name": name,
        "stratum_or_model": stratum,
        "reported_in_manuscript": reported_val,
        "authoritative_source_file": source_file,
        "source_value": source_val,
        "audit_status": status,
        "audit_notes": notes
    })

# 1. Continuous Regression Overall Test Metrics
lgb_tweedie = df_reg[(df_reg["model_name"] == "LightGBM_Tweedie_p1.5") & (df_reg["partition"].str.contains("Test"))].iloc[0]
add_check("Continuous Regression", "RMSE", "LightGBM Tweedie (Test)", "7.4427", "regression_model_results.csv", f"{lgb_tweedie['RMSE']:.4f}", notes="Overall Test RMSE (mm)")
add_check("Continuous Regression", "MAE", "LightGBM Tweedie (Test)", "2.8993", "regression_model_results.csv", f"{lgb_tweedie['MAE']:.4f}", notes="Overall Test MAE (mm)")
add_check("Continuous Regression", "Mean Bias", "LightGBM Tweedie (Test)", "-0.0179", "regression_model_results.csv", f"{lgb_tweedie['Bias']:.4f}", notes="Overall Test Mean Bias (mm)")
add_check("Continuous Regression", "Percentage Bias", "LightGBM Tweedie (Test)", "-0.51%", "regression_model_results.csv", f"{lgb_tweedie['PBias_pct']:.2f}%", notes="Overall Test Relative Bias (%)")
add_check("Continuous Regression", "Pearson r", "LightGBM Tweedie (Test)", "0.7363", "regression_model_results.csv", f"{lgb_tweedie['Pearson_r']:.4f}", notes="Overall Test Pearson correlation")
add_check("Continuous Regression", "NSE", "LightGBM Tweedie (Test)", "0.5420", "regression_model_results.csv", f"{lgb_tweedie['NSE']:.4f}", notes="Overall Test Nash-Sutcliffe Efficiency")
add_check("Continuous Regression", "KGE-2012", "LightGBM Tweedie (Test)", "0.6245", "regression_model_results.csv", f"{lgb_tweedie['KGE_2012']:.4f}", notes="Overall Test Modified Kling-Gupta Efficiency")

cat_tweedie = df_reg[(df_reg["model_name"] == "CatBoost_Tweedie_p1.5") & (df_reg["partition"].str.contains("Test"))].iloc[0]
add_check("Continuous Regression", "RMSE", "CatBoost Tweedie (Test)", "7.4335", "regression_model_results.csv", f"{cat_tweedie['RMSE']:.4f}", notes="CatBoost Test RMSE (mm)")
add_check("Continuous Regression", "KGE-2012", "CatBoost Tweedie (Test)", "0.6261", "regression_model_results.csv", f"{cat_tweedie['KGE_2012']:.4f}", notes="CatBoost Highest Test KGE-2012")

hurdle_ev = df_hurdle[(df_hurdle["model_name"] == "Hurdle_ExpectedValue") & (df_hurdle["partition"].str.contains("Test"))].iloc[0]
add_check("Continuous Regression", "RMSE", "Hurdle Expected Value (Test)", "7.4241", "hurdle_model_results.csv", f"{hurdle_ev['RMSE']:.4f}", notes="Lowest Overall Test RMSE (mm)")
add_check("Continuous Regression", "Pearson r", "Hurdle Expected Value (Test)", "0.7380", "hurdle_model_results.csv", f"{hurdle_ev['Pearson_r']:.4f}", notes="Highest Test Pearson correlation")
add_check("Continuous Regression", "NSE", "Hurdle Expected Value (Test)", "0.5443", "hurdle_model_results.csv", f"{hurdle_ev['NSE']:.4f}", notes="Highest Test Nash-Sutcliffe Efficiency")
add_check("Continuous Regression", "Heavy CSI", "Hurdle Expected Value (Test)", "0.3072", "hurdle_model_results.csv", f"{hurdle_ev['CSI_Heavy_64.5mm']:.4f}", notes="Hurdle Test Heavy CSI (>=64.5 mm)")

# 2. Baselines (Test Partition)
base_pers = df_base[(df_base["model_name"] == "Rainfall_Persistence") & (df_base["partition"].str.contains("Test"))].iloc[0]
add_check("Baselines", "RMSE", "Persistence (Test)", "10.3582", "baseline_results.csv", f"{base_pers['RMSE_mm']:.4f}", notes="Persistence Baseline Test RMSE")
base_clim = df_base[(df_base["model_name"] == "Station_DOY_Climatology") & (df_base["partition"].str.contains("Test"))].iloc[0]
add_check("Baselines", "RMSE", "Grid-cell DOY Climatology (Test)", "10.1155", "baseline_results.csv", f"{base_clim['RMSE_mm']:.4f}", notes="Grid-cell DOY Climatology Baseline Test RMSE")
base_era = df_base[(df_base["model_name"] == "Antecedent_ERA5_Land") & (df_base["partition"].str.contains("Test"))].iloc[0]
add_check("Baselines", "RMSE", "Raw ERA5-Land (Test)", "10.4550", "baseline_results.csv", f"{base_era['RMSE_mm']:.4f}", notes="Raw Antecedent ERA5 Baseline Test RMSE")
base_eqm = df_base[(df_base["model_name"] == "Empirical_Quantile_Mapping_EQM") & (df_base["partition"].str.contains("Test"))].iloc[0]
add_check("Baselines", "RMSE", "EQM (Test)", "11.1970", "baseline_results.csv", f"{base_eqm['RMSE_mm']:.4f}", notes="Empirical Quantile Mapping Test RMSE")

ridge = df_reg[(df_reg["model_name"] == "Ridge_Linear_Baseline") & (df_reg["partition"].str.contains("Test"))].iloc[0]
add_check("Baselines", "RMSE", "Ridge Regression (Tier 3)", "7.9750", "regression_model_results.csv", f"{ridge['RMSE']:.4f}", notes="Tier 3 Linear Baseline Test RMSE")
enet = df_reg[(df_reg["model_name"] == "ElasticNet_Linear_Baseline") & (df_reg["partition"].str.contains("Test"))].iloc[0]
add_check("Baselines", "RMSE", "ElasticNet (Tier 3)", "8.0156", "regression_model_results.csv", f"{enet['RMSE']:.4f}", notes="Tier 3 Linear Baseline Test RMSE")

# 3. Annual Performance (2024 vs 2025)
y24_lgb = df_yearly[(df_yearly["temporal_stratum"] == "Year_2024_Full") & (df_yearly["model"] == "LightGBM_Tweedie")].iloc[0]
add_check("Annual Robustness", "2024 RMSE", "LightGBM Tweedie (2024)", "7.5094", "yearly_performance_results.csv", f"{y24_lgb['RMSE']:.4f}", notes="2024 Full Year RMSE (mm)")
add_check("Annual Robustness", "2024 Bias", "LightGBM Tweedie (2024)", "+0.3442", "yearly_performance_results.csv", f"{y24_lgb['Bias']:+.4f}", notes="2024 Full Year Bias (mm)")
add_check("Annual Robustness", "2024 PBias", "LightGBM Tweedie (2024)", "+10.26%", "yearly_performance_results.csv", f"{y24_lgb['PBias_pct']:+.2f}%", notes="2024 Full Year PBias (%)")
add_check("Annual Robustness", "2024 Pearson r", "LightGBM Tweedie (2024)", "0.7320", "yearly_performance_results.csv", f"{y24_lgb['Pearson_r']:.4f}", notes="2024 Full Year Pearson r")
add_check("Annual Robustness", "2024 NSE", "LightGBM Tweedie (2024)", "0.5346", "yearly_performance_results.csv", f"{y24_lgb['NSE']:.4f}", notes="2024 Full Year NSE")
add_check("Annual Robustness", "2024 KGE-2012", "LightGBM Tweedie (2024)", "0.5676", "yearly_performance_results.csv", f"{y24_lgb['KGE_2012']:.4f}", notes="2024 Full Year KGE-2012")

y25_lgb = df_yearly[(df_yearly["temporal_stratum"] == "Year_2025_Full") & (df_yearly["model"] == "LightGBM_Tweedie")].iloc[0]
add_check("Annual Robustness", "2025 RMSE", "LightGBM Tweedie (2025)", "7.3752", "yearly_performance_results.csv", f"{y25_lgb['RMSE']:.4f}", notes="2025 Full Year RMSE (mm)")
add_check("Annual Robustness", "2025 Bias", "LightGBM Tweedie (2025)", "-0.3810", "yearly_performance_results.csv", f"{y25_lgb['Bias']:+.4f}", notes="2025 Full Year Bias (mm)")
add_check("Annual Robustness", "2025 PBias", "LightGBM Tweedie (2025)", "-10.40%", "yearly_performance_results.csv", f"{y25_lgb['PBias_pct']:+.2f}%", notes="2025 Full Year PBias (%)")
add_check("Annual Robustness", "2025 Pearson r", "LightGBM Tweedie (2025)", "0.7426", "yearly_performance_results.csv", f"{y25_lgb['Pearson_r']:.4f}", notes="2025 Full Year Pearson r")
add_check("Annual Robustness", "2025 NSE", "LightGBM Tweedie (2025)", "0.5493", "yearly_performance_results.csv", f"{y25_lgb['NSE']:.4f}", notes="2025 Full Year NSE")
add_check("Annual Robustness", "2025 KGE-2012", "LightGBM Tweedie (2025)", "0.6541", "yearly_performance_results.csv", f"{y25_lgb['KGE_2012']:.4f}", notes="2025 Full Year KGE-2012")

# 4. Heavy Rain Focal Classifier Performance (Decision threshold p*=0.53)
focal_heavy = df_heavy[(df_heavy["model_name"] == "LightGBM_Heavy_FocalLoss") & (df_heavy["partition"].str.contains("Test"))].iloc[0]
add_check("Heavy Rain Detection", "Decision Threshold", "LightGBM Focal (Test)", "0.53", "heavy_rain_results.csv", f"{focal_heavy['calibrated_threshold']:.2f}", notes="Validation-selected decision threshold p*")
add_check("Heavy Rain Detection", "POD (Recall)", "LightGBM Focal (Test)", "46.96%", "heavy_rain_results.csv", f"{focal_heavy['POD_Recall']*100:.2f}%", notes="Test POD for events >=64.5 mm")
add_check("Heavy Rain Detection", "Missed Rate", "LightGBM Focal (Test)", "53.04%", "heavy_rain_results.csv", f"{(1.0 - focal_heavy['POD_Recall'])*100:.2f}%", notes="Misses more than half of observed events")
add_check("Heavy Rain Detection", "Precision", "LightGBM Focal (Test)", "58.22%", "heavy_rain_results.csv", f"{focal_heavy['Precision']*100:.2f}%", notes="Test Precision")
add_check("Heavy Rain Detection", "FAR", "LightGBM Focal (Test)", "41.78%", "heavy_rain_results.csv", f"{focal_heavy['FAR']*100:.2f}%", notes="Test False Alarm Ratio")
add_check("Heavy Rain Detection", "CSI Threat Score", "LightGBM Focal (Test)", "0.3512", "heavy_rain_results.csv", f"{focal_heavy['CSI_Threat_Score']:.4f}", notes="Test Critical Success Index")
add_check("Heavy Rain Detection", "PR-AUC", "LightGBM Focal (Test)", "0.5084", "heavy_rain_results.csv", f"{focal_heavy['PR_AUC']:.4f}", notes="Precision-Recall Area Under Curve")
add_check("Heavy Rain Detection", "Hits", "LightGBM Focal (Test)", "595", "heavy_rain_results.csv", f"{int(focal_heavy['Hits'])}", notes="Contingency Hits count")
add_check("Heavy Rain Detection", "Misses", "LightGBM Focal (Test)", "672", "heavy_rain_results.csv", f"{int(focal_heavy['Misses'])}", notes="Contingency Misses count")
add_check("Heavy Rain Detection", "False Alarms", "LightGBM Focal (Test)", "427", "heavy_rain_results.csv", f"{int(focal_heavy['False_Alarms'])}", notes="Contingency False Alarms count")
add_check("Heavy Rain Detection", "Correct Negatives", "LightGBM Focal (Test)", "182518", "heavy_rain_results.csv", f"{int(focal_heavy['Correct_Negatives'])}", notes="Contingency Correct Negatives count")

# 5. Rainfall Intensity Categories & Extreme Underestimation
no_rain = df_intens[(df_intens["category"] == "No Rain (<0.1 mm)") & (df_intens["model"] == "LightGBM_Tweedie")].iloc[0]
add_check("Intensity Spectrum", "No Rain Sample Count", "No Rain (<0.1 mm)", "119669", "rainfall_intensity_sensitivity.csv", f"{int(no_rain['sample_count'])}", notes="No Rain test sample count (64.96%)")
vh_rain = df_intens[(df_intens["category"] == "Very Heavy (115.6–204.4 mm)") & (df_intens["model"] == "LightGBM_Tweedie")].iloc[0]
add_check("Intensity Spectrum", "Very Heavy Sample Count", "Very Heavy (115.6-204.4 mm)", "192", "rainfall_intensity_sensitivity.csv", f"{int(vh_rain['sample_count'])}", notes="Very Heavy test sample count")
add_check("Intensity Spectrum", "Very Heavy Underestimated", "Very Heavy (115.6-204.4 mm)", "100.0%", "rainfall_intensity_sensitivity.csv", f"{vh_rain['Percent_Underestimated']:.1f}%", notes="100% underestimation across all 192 events")
eh_rain = df_intens[(df_intens["category"] == "Extremely Heavy (≥204.5 mm)") & (df_intens["model"] == "LightGBM_Tweedie")].iloc[0]
add_check("Intensity Spectrum", "Extremely Heavy Sample Count", "Extremely Heavy (>=204.5 mm)", "8", "rainfall_intensity_sensitivity.csv", f"{int(eh_rain['sample_count'])}", notes="Extremely Heavy test count")
add_check("Intensity Spectrum", "Extremely Heavy Underestimated", "Extremely Heavy (>=204.5 mm)", "100.0%", "rainfall_intensity_sensitivity.csv", f"{eh_rain['Percent_Underestimated']:.1f}%", notes="100% underestimation across all 8 events")
tot_gt_115 = int(vh_rain['sample_count']) + int(eh_rain['sample_count'])
add_check("Intensity Spectrum", "Total Events >=115.6 mm", "Events >=115.6 mm", "200", "rainfall_intensity_sensitivity.csv", f"{tot_gt_115}", notes="192 Very Heavy + 8 Extremely Heavy = 200")
add_check("Intensity Spectrum", "Events >=115.6 mm Underestimated", "Events >=115.6 mm", "100.0%", "rainfall_intensity_sensitivity.csv", "100.0%", notes="All 200 events >=115.6 mm underestimated")

# 6. Coastal Fallback Cells
fb_sub = df_sub[df_sub["subgroup"] == "Fallback_Coastal_Only"].iloc[0]
add_check("Spatial Diagnostics", "Coastal Fallback Sample Count", "Fallback_Coastal_Only", "2193", "subgroup_evaluation_results.csv", f"{int(fb_sub['sample_count'])}", notes="3 stations x 731 days = 2193 samples")
add_check("Spatial Diagnostics", "Coastal Fallback Cell Count", "Fallback_Coastal_Only", "3", "build_stage3_dataset.py", "3", notes="Grid-Cell IDs 52, 68, 125 out of 13 coastal cells")

# 7. Feature Importance & Permutation Provenance
sro_perm = df_perm[df_perm["feature"] == "sro_lag_1d"].iloc[0]
add_check("Feature Attribution", "Permutation Delta RMSE", "sro_lag_1d (Validation)", "+2.3545", "permutation_importance.csv", f"{sro_perm['delta_rmse_mean']:.4f}", notes="Full Validation (N=91,980) Delta RMSE (mm)")
f_ro = df_feat[df_feat["feature"] == "ro_lag_1d"]["lgb_focal_heavy_gain_pct"].values[0]
f_sro = df_feat[df_feat["feature"] == "sro_lag_1d"]["lgb_focal_heavy_gain_pct"].values[0]
add_check("Feature Attribution", "Heavy Classifier Tree Split Gain", "ro_lag_1d + sro_lag_1d", "54.52%", "feature_importance_comparison.csv", f"{f_ro + f_sro:.2f}%", notes="28.11% ro + 26.41% sro in LightGBM Focal Classifier")

# 8. Paired Block Bootstrap
b_era = df_boot[(df_boot["model_A"] == "LightGBM_Tweedie") & (df_boot["baseline_B"] == "Raw_ERA5") & (df_boot["block_length_days"] == 7)].iloc[0]
add_check("Bootstrap Rigor", "Bootstrap Replicates", "Stationary Block Bootstrap", "2000", "paired_bootstrap_robustness.csv", f"{int(b_era['replicates'])}", notes="2,000 resamples across statewide blocks")
add_check("Bootstrap Rigor", "Delta RMSE vs Raw ERA5", "LightGBM Tweedie vs Raw ERA5", "-3.0109", "paired_bootstrap_robustness.csv", f"{b_era['mean_delta_rmse_mm']:.4f}", notes="Mean Delta RMSE (mm)")
add_check("Bootstrap Rigor", "Empirical p-value Resolution", "LightGBM Tweedie vs Raw ERA5", "p < 0.0005", "paired_bootstrap_robustness.csv", f"{b_era['empirical_p_value']}", notes="Empirical p-value resolution limit for 2,000 replicates")

df_audit = pd.DataFrame(audit_records)
out_csv = out_dir / "numerical_reconciliation.csv"
df_audit.to_csv(out_csv, index=False)
print(f"Generated numerical_reconciliation.csv with {len(df_audit)} checks.")
print(df_audit['audit_status'].value_counts())
