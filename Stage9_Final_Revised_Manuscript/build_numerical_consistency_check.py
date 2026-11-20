import pandas as pd
from pathlib import Path

stage9_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage9_Final_Revised_Manuscript")
stage6_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage6_Robustness_Generalisation")
stage7_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage7_Publication_Ready_Package")
stage5_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage5_Model_Interpretation")

# 1. Load source tables
df_t1 = pd.read_csv(stage7_dir / "publication_tables/table1_dataset_characteristics.csv")
df_t3 = pd.read_csv(stage7_dir / "publication_tables/table3_model_benchmarking.csv")
df_t4 = pd.read_csv(stage7_dir / "publication_tables/table4_heavy_rain_classification.csv")
df_t5 = pd.read_csv(stage7_dir / "publication_tables/table5_regional_seasonal_performance.csv")
df_t6 = pd.read_csv(stage7_dir / "publication_tables/table6_statistical_robustness.csv")
df_t7 = pd.read_csv(stage7_dir / "publication_tables/table7_feature_ablation.csv")
df_t8 = pd.read_csv(stage9_dir / "manuscript_tables/table8_rainfall_intensity_limitations.csv")
df_yearly = pd.read_csv(stage6_dir / "yearly_performance_results.csv")
df_feat = pd.read_csv(stage5_dir / "feature_importance_comparison.csv")
df_perm = pd.read_csv(stage5_dir / "permutation_importance.csv")

audit_rows = []

def record(metric_category, metric_name, stratum_or_model, reported_value, authoritative_source_file, source_value, notes):
    match_status = "MATCH" if str(reported_value).strip() == str(source_value).strip() or abs(float(str(reported_value).replace("%","").replace("Undefined*","0")) - float(str(source_value).replace("%","").replace("Undefined*","0"))) < 1e-3 else "MISMATCH"
    audit_rows.append({
        "metric_category": metric_category,
        "metric_name": metric_name,
        "stratum_or_model": stratum_or_model,
        "reported_in_manuscript": reported_value,
        "authoritative_source_file": authoritative_source_file,
        "source_value": source_value,
        "audit_status": match_status,
        "audit_notes": notes
    })

# Benchmark Overall (Test)
t3_lgb = df_t3[df_t3["Model_Name"].str.contains("LightGBM Tweedie")].iloc[0]
record("Continuous Regression", "RMSE", "LightGBM_Tweedie", 7.4427, "table3_model_benchmarking.csv", t3_lgb["RMSE_mm"], "Overall Test RMSE (mm)")
record("Continuous Regression", "MAE", "LightGBM_Tweedie", 2.8993, "table3_model_benchmarking.csv", t3_lgb["MAE_mm"], "Overall Test MAE (mm)")
record("Continuous Regression", "Mean Bias", "LightGBM_Tweedie", -0.0179, "table3_model_benchmarking.csv", t3_lgb["Mean_Bias_mm"], "Overall Test Bias (mm)")
record("Continuous Regression", "PBias (%)", "LightGBM_Tweedie", -0.51, "table3_model_benchmarking.csv", t3_lgb["PBias_pct"], "Overall Test Percentage Bias (%)")
record("Continuous Regression", "Pearson r", "LightGBM_Tweedie", 0.7363, "table3_model_benchmarking.csv", t3_lgb["Pearson_r"], "Overall Test Pearson correlation")
record("Continuous Regression", "NSE", "LightGBM_Tweedie", 0.5420, "table3_model_benchmarking.csv", t3_lgb["NSE"], "Overall Test Nash-Sutcliffe Efficiency")
record("Continuous Regression", "KGE-2012", "LightGBM_Tweedie", 0.6245, "table3_model_benchmarking.csv", t3_lgb["Modified_KGE_2012"], "Overall Test Modified KGE")

t3_cat = df_t3[df_t3["Model_Name"].str.contains("CatBoost Tweedie")].iloc[0]
record("Continuous Regression", "RMSE", "CatBoost_Tweedie", 7.4335, "table3_model_benchmarking.csv", t3_cat["RMSE_mm"], "CatBoost Test RMSE (mm)")
record("Continuous Regression", "MAE", "CatBoost_Tweedie", 2.9186, "table3_model_benchmarking.csv", t3_cat["MAE_mm"], "CatBoost Test MAE (mm)")
record("Continuous Regression", "Mean Bias", "CatBoost_Tweedie", 0.0388, "table3_model_benchmarking.csv", t3_cat["Mean_Bias_mm"], "CatBoost Test Bias (mm)")
record("Continuous Regression", "KGE-2012", "CatBoost_Tweedie", 0.6261, "table3_model_benchmarking.csv", t3_cat["Modified_KGE_2012"], "CatBoost Test Modified KGE")

t3_hurdle = df_t3[df_t3["Model_Name"].str.contains("Hurdle ExpectedValue")].iloc[0]
record("Continuous Regression", "RMSE", "Hurdle_ExpectedValue", 7.4241, "table3_model_benchmarking.csv", t3_hurdle["RMSE_mm"], "Lowest RMSE among all continuous models")
record("Continuous Regression", "MAE", "Hurdle_ExpectedValue", 2.9300, "table3_model_benchmarking.csv", t3_hurdle["MAE_mm"], "Hurdle Test MAE (mm)")
record("Continuous Regression", "Pearson r", "Hurdle_ExpectedValue", 0.7380, "table3_model_benchmarking.csv", t3_hurdle["Pearson_r"], "Highest Pearson r among continuous models")
record("Continuous Regression", "NSE", "Hurdle_ExpectedValue", 0.5443, "table3_model_benchmarking.csv", t3_hurdle["NSE"], "Highest NSE among continuous models")
record("Continuous Regression", "Heavy CSI", "Hurdle_ExpectedValue", 0.3072, "table3_model_benchmarking.csv", t3_hurdle["Heavy_CSI_64.5mm"], "Highest CSI among continuous models")

t3_pers = df_t3[df_t3["Model_Name"].str.contains("Rainfall Persistence")].iloc[0]
record("Baselines", "RMSE", "Persistence", 10.3582, "table3_model_benchmarking.csv", t3_pers["RMSE_mm"], "Persistence Test RMSE")
t3_era5 = df_t3[df_t3["Model_Name"].str.contains("Antecedent ERA5")].iloc[0]
record("Baselines", "RMSE", "Raw_ERA5", 10.4550, "table3_model_benchmarking.csv", t3_era5["RMSE_mm"], "Raw ERA5-Land Test RMSE")
t3_eqm = df_t3[df_t3["Model_Name"].str.contains("Empirical Quantile")].iloc[0]
record("Baselines", "RMSE", "EQM", 11.1970, "table3_model_benchmarking.csv", t3_eqm["RMSE_mm"], "EQM Test RMSE")

# Heavy Rain Classification (64.5 mm)
t4_focal = df_t4[df_t4["Model_Name"].str.contains("FocalLoss")].iloc[0]
record("Extreme Classification", "Decision Threshold", "LightGBM_FocalLoss", 0.53, "table4_heavy_rain_classification.csv", t4_focal["Decision_Threshold"], "Operating decision threshold")
record("Extreme Classification", "Hits", "LightGBM_FocalLoss", 595, "table4_heavy_rain_classification.csv", t4_focal["Hits"], "Contingency True Positives")
record("Extreme Classification", "Misses", "LightGBM_FocalLoss", 672, "table4_heavy_rain_classification.csv", t4_focal["Misses"], "Contingency False Negatives")
record("Extreme Classification", "False Alarms", "LightGBM_FocalLoss", 427, "table4_heavy_rain_classification.csv", t4_focal["False_Alarms"], "Contingency False Positives")
record("Extreme Classification", "Precision (%)", "LightGBM_FocalLoss", 58.22, "table4_heavy_rain_classification.csv", round(t4_focal["Precision"]*100, 2), "Precision percentage")
record("Extreme Classification", "Recall / POD (%)", "LightGBM_FocalLoss", 46.96, "table4_heavy_rain_classification.csv", round(t4_focal["POD_Recall"]*100, 2), "Probability of detection")
record("Extreme Classification", "FAR (%)", "LightGBM_FocalLoss", 41.78, "table4_heavy_rain_classification.csv", round(t4_focal["FAR"]*100, 2), "False alarm ratio")
record("Extreme Classification", "CSI", "LightGBM_FocalLoss", 0.3512, "table4_heavy_rain_classification.csv", t4_focal["CSI"], "Critical success index")
record("Extreme Classification", "HSS", "LightGBM_FocalLoss", 0.5169, "table4_heavy_rain_classification.csv", t4_focal["HSS"], "Heidke skill score")

# Annual Breakdown Reconciled (Stage 6)
y24_lgb = df_yearly[(df_yearly["temporal_stratum"] == "Year_2024_Full") & (df_yearly["model"] == "LightGBM_Tweedie")].iloc[0]
record("Annual Breakdown", "RMSE", "LightGBM_Tweedie_2024", 7.5094, "yearly_performance_results.csv", y24_lgb["RMSE"], "Reconciled 2024 RMSE")
record("Annual Breakdown", "MAE", "LightGBM_Tweedie_2024", 3.0074, "yearly_performance_results.csv", y24_lgb["MAE"], "Reconciled 2024 MAE")
record("Annual Breakdown", "Bias", "LightGBM_Tweedie_2024", 0.3442, "yearly_performance_results.csv", y24_lgb["Bias"], "Reconciled 2024 Bias")
record("Annual Breakdown", "Pearson r", "LightGBM_Tweedie_2024", 0.7320, "yearly_performance_results.csv", y24_lgb["Pearson_r"], "Reconciled 2024 Pearson r")
record("Annual Breakdown", "NSE", "LightGBM_Tweedie_2024", 0.5346, "yearly_performance_results.csv", y24_lgb["NSE"], "Reconciled 2024 NSE")
record("Annual Breakdown", "KGE-2012", "LightGBM_Tweedie_2024", 0.5676, "yearly_performance_results.csv", y24_lgb["KGE_2012"], "Reconciled 2024 KGE-2012")

y25_lgb = df_yearly[(df_yearly["temporal_stratum"] == "Year_2025_Full") & (df_yearly["model"] == "LightGBM_Tweedie")].iloc[0]
record("Annual Breakdown", "RMSE", "LightGBM_Tweedie_2025", 7.3752, "yearly_performance_results.csv", y25_lgb["RMSE"], "Reconciled 2025 RMSE")
record("Annual Breakdown", "MAE", "LightGBM_Tweedie_2025", 2.7909, "yearly_performance_results.csv", y25_lgb["MAE"], "Reconciled 2025 MAE")
record("Annual Breakdown", "Bias", "LightGBM_Tweedie_2025", -0.3810, "yearly_performance_results.csv", y25_lgb["Bias"], "Reconciled 2025 Bias")
record("Annual Breakdown", "Pearson r", "LightGBM_Tweedie_2025", 0.7426, "yearly_performance_results.csv", y25_lgb["Pearson_r"], "Reconciled 2025 Pearson r")
record("Annual Breakdown", "NSE", "LightGBM_Tweedie_2025", 0.5493, "yearly_performance_results.csv", y25_lgb["NSE"], "Reconciled 2025 NSE")
record("Annual Breakdown", "KGE-2012", "LightGBM_Tweedie_2025", 0.6541, "yearly_performance_results.csv", y25_lgb["KGE_2012"], "Reconciled 2025 KGE-2012")

# Feature Importance Origins
ro_focal_gain = df_feat[df_feat["feature"] == "ro_lag_1d"]["lgb_focal_heavy_gain_pct"].values[0]
sro_focal_gain = df_feat[df_feat["feature"] == "sro_lag_1d"]["lgb_focal_heavy_gain_pct"].values[0]
comb_gain = round(ro_focal_gain + sro_focal_gain, 2)
record("Feature Attribution", "Combined Tree Gain Pct", "Runoff_ro_plus_sro_focal", 54.52, "feature_importance_comparison.csv", comb_gain, "Tree split gain in Focal Loss heavy rain classifier (28.11% + 26.41%)")

sro_tweedie_gain = round(df_feat[df_feat["feature"] == "sro_lag_1d"]["lgb_tweedie_gain_pct"].values[0], 2)
record("Feature Attribution", "Tree Gain Pct", "sro_lag_1d_tweedie", 34.61, "feature_importance_comparison.csv", sro_tweedie_gain, "Tree split gain in continuous LightGBM Tweedie")

sro_perm_delta = round(df_perm[df_perm["feature"] == "sro_lag_1d"]["delta_rmse_mean"].values[0], 4)
record("Feature Attribution", "Permutation Delta RMSE", "sro_lag_1d_perm", 2.3545, "permutation_importance.csv", sro_perm_delta, "Validation permutation importance delta RMSE (Rank 1, +42.16% increase)")

# Extreme Rainfall Quantile Shrinkage
t8_very_heavy = df_t8[df_t8["IMD_Rainfall_Intensity_Category"].str.contains("Very Heavy")].iloc[0]
record("Extreme Limitations", "Percent Underestimated", "Very Heavy (115.6-204.4mm)", 100.0, "table8_rainfall_intensity_limitations.csv", t8_very_heavy["Percent_Underestimated"], "100% of 192 Very Heavy events underestimated")

t8_ext_heavy = df_t8[df_t8["IMD_Rainfall_Intensity_Category"].str.contains("Extremely Heavy")].iloc[0]
record("Extreme Limitations", "Percent Underestimated", "Extremely Heavy (>=204.5mm)", 100.0, "table8_rainfall_intensity_limitations.csv", t8_ext_heavy["Percent_Underestimated"], "100% of 8 Extremely Heavy events underestimated")
record("Extreme Limitations", "Prevalence Pct", "Extremely Heavy (>=204.5mm)", 0.0043, "table8_rainfall_intensity_limitations.csv", t8_ext_heavy["Percent_Total_Test_Data"], "8 events out of 184,212 instances")

df_audit = pd.DataFrame(audit_rows)
df_audit.to_csv(stage9_dir / "final_numerical_consistency_check.csv", index=False)
print(f"Generated final_numerical_consistency_check.csv with {len(df_audit)} audited entries.")
print(df_audit["audit_status"].value_counts())

