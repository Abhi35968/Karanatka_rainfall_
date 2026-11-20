import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
audit_dir = base_dir / "Stage4_2_Independent_Model_Audit"
audit_dir.mkdir(parents=True, exist_ok=True)

print("Loading datasets for Task 5: Feature Ablation Experiments...")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

y_tr = df_train["target_rainfall_mm"].values.astype(np.float64)
y_va = df_val["target_rainfall_mm"].values.astype(np.float64)
y_te = df_test["target_rainfall_mm"].values.astype(np.float64)

# Manifest features
df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

# Target encodings computed strictly on Train
subdiv_map = df_train.groupby("subdivision")["target_rainfall_mm"].mean().to_dict()
zone_map = df_train.groupby("zone_id")["target_rainfall_mm"].mean().to_dict()

def prepare_base_matrix(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_map).fillna(3.5).values
    X["zone_id_enc"] = df["zone_id"].map(zone_map).fillna(3.5).values
    return X

X_tr_all = prepare_base_matrix(df_train)
X_va_all = prepare_base_matrix(df_val)
X_te_all = prepare_base_matrix(df_test)

all_feature_list = list(X_tr_all.columns)
print(f"Total Base Features: {len(all_feature_list)}")

# Feature grouping for ablation
hydro_runoff_cols = [
    "sro_lag_1d", "ro_lag_1d", "src_lag_1d", "evabs_lag_1d", "evatc_lag_1d", "pev_lag_1d",
    "sro_roll_3d_sum", "ro_roll_3d_sum", "sro_roll_7d_sum", "ro_roll_7d_sum", "fal_lag_1d", "swvl2_lag_1d"
]

imd_rainfall_cols = [
    "target_rain_lag_1d", "target_rain_lag_2d", "target_rain_lag_3d", "target_rain_lag_7d", "target_rain_lag_14d",
    "target_rain_roll_3d_sum", "target_rain_roll_3d_mean", "target_rain_roll_7d_sum", "target_rain_roll_7d_mean",
    "target_rain_roll_14d_sum", "target_rain_roll_14d_mean", "api_14d", "api_30d"
]

terrain_spatial_cols = [
    "elevation_dem_m", "slope_dem_deg", "dist_to_coast_km", "dist_to_ghats_crest_km",
    "latitude", "longitude", "subdivision_enc", "zone_id_enc"
]

temporal_cols = [
    "day_of_year", "month", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "quarter"
]

era5_cols = [c for c in all_feature_list if c not in (imd_rainfall_cols + terrain_spatial_cols + temporal_cols)]

# Define Ablation Scenarios
ablation_scenarios = {
    "Scenario_A_All_63_Primary_Predictors": all_feature_list,
    "Scenario_B_Without_Runoff_and_Hydrological_Fluxes": [c for c in all_feature_list if c not in hydro_runoff_cols],
    "Scenario_C_Without_Antecedent_IMD_Rainfall": [c for c in all_feature_list if c not in imd_rainfall_cols],
    "Scenario_D_Without_Terrain_and_Spatial_Predictors": [c for c in all_feature_list if c not in terrain_spatial_cols],
    "Scenario_E_Without_ERA5_Predictors": [c for c in all_feature_list if c not in era5_cols],
    "Scenario_F_Only_Antecedent_IMD_and_Temporal": [c for c in (imd_rainfall_cols + temporal_cols) if c in all_feature_list]
}

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    
    sum_true = float(np.sum(y_true))
    pbias = float((np.sum(diff) / sum_true * 100.0) if sum_true > 1e-12 else 0.0)
    
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    ss_res = float(np.sum(diff ** 2))
    nse = float(1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0)
    
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    
    if std_true > 1e-8 and std_pred > 1e-8:
        cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        r = float(cov / (std_true * std_pred))
        r = float(np.clip(r, -1.0, 1.0))
        beta = float(mean_pred / mean_true if mean_true > 1e-8 else 1.0)
        cv_true = std_true / mean_true if mean_true > 1e-8 else 1.0
        cv_pred = std_pred / mean_pred if mean_pred > 1e-8 else 1.0
        gamma = cv_pred / cv_true if cv_true > 1e-8 else 1.0
        kge_2012 = float(1.0 - np.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2))
        alpha = float(std_pred / std_true if std_true > 1e-8 else 1.0)
        kge_classic = float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))
    else:
        r, kge_2012, kge_classic = 0.0, -999.0, -999.0
        
    # Contingency 64.5 mm
    yt_h = (y_true >= 64.5)
    yp_h = (y_pred >= 64.5)
    h = int(np.sum(yt_h & yp_h))
    f = int(np.sum((~yt_h) & yp_h))
    m = int(np.sum(yt_h & (~yp_h)))
    c = int(np.sum((~yt_h) & (~yp_h)))
    n = len(yt_h)
    
    pod = (h / (h + m)) if (h + m) > 0 else 0.0
    far = (f / (h + f)) if (h + f) > 0 else 0.0
    csi = (h / (h + m + f)) if (h + m + f) > 0 else 0.0
    hss_num = 2.0 * (h * c - f * m)
    hss_den = ((h + m) * (m + c) + (h + f) * (f + c))
    hss = (hss_num / hss_den) if hss_den > 0 else 0.0
    
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Bias": round(bias, 4),
        "PBias_pct": round(pbias, 2),
        "Pearson_r": round(r, 4),
        "NSE": round(nse, 4),
        "KGE_2012": round(kge_2012, 4),
        "KGE_classic_2009": round(kge_classic, 4),
        "CSI_64.5mm": round(csi, 4),
        "POD_64.5mm": round(pod, 4),
        "FAR_64.5mm": round(far, 4),
        "HSS_64.5mm": round(hss, 4),
        "Hits": h,
        "False_Alarms": f,
        "Misses": m
    }

val_records = []
test_records = []

for sc_name, cols in ablation_scenarios.items():
    print(f"\nTraining Tweedie Regressor on {sc_name} ({len(cols)} features)...")
    
    X_tr_sub = X_tr_all[cols]
    X_va_sub = X_va_all[cols]
    X_te_sub = X_te_all[cols]
    
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Early stopping strictly on Validation
    model.fit(
        X_tr_sub, y_tr,
        eval_set=[(X_va_sub, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Predict on Validation
    y_val_pred = model.predict(X_va_sub)
    m_val = calc_metrics(y_va, y_val_pred)
    
    # Predict on Test
    y_te_pred = model.predict(X_te_sub)
    m_te = calc_metrics(y_te, y_te_pred)
    
    row_val = {
        "ablation_scenario": sc_name,
        "feature_count": len(cols),
        "best_iteration": model.best_iteration_,
        "partition": "Validation_2023",
        "sample_count": len(y_va),
        **m_val
    }
    val_records.append(row_val)
    
    row_te = {
        "ablation_scenario": sc_name,
        "feature_count": len(cols),
        "best_iteration": model.best_iteration_,
        "partition": "Test_2024_2025",
        "sample_count": len(y_te),
        **m_te
    }
    test_records.append(row_te)

df_val_ablation = pd.DataFrame(val_records)
df_val_ablation.to_csv(audit_dir / "feature_ablation_validation_results.csv", index=False)
print("\nSaved feature_ablation_validation_results.csv")
print("\nValidation Ablation Results (2023):")
print(df_val_ablation[["ablation_scenario", "feature_count", "best_iteration", "RMSE", "MAE", "Bias", "Pearson_r", "NSE", "KGE_2012", "CSI_64.5mm"]].to_string())

df_test_ablation = pd.DataFrame(test_records)
df_test_ablation.to_csv(audit_dir / "feature_ablation_test_results.csv", index=False)
print("\nSaved feature_ablation_test_results.csv")
print("\nTest Ablation Results (2024–2025, Frozen Evaluation):")
print(df_test_ablation[["ablation_scenario", "feature_count", "RMSE", "MAE", "Bias", "Pearson_r", "NSE", "KGE_2012", "CSI_64.5mm"]].to_string())

