import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
stage4_dir = base_dir / "Stage4_1_Model_Development"
models_dir = stage4_dir / "models"
audit_dir = base_dir / "Stage4_2_Independent_Model_Audit"
audit_dir.mkdir(parents=True, exist_ok=True)

print("Loading raw test and validation datasets for independent metric recalculation...")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

y_val = df_val["target_rainfall_mm"].values.astype(np.float64)
y_te = df_test["target_rainfall_mm"].values.astype(np.float64)

y_val_rain = df_val["target_is_rain"].values.astype(np.int32)
y_te_rain = df_test["target_is_rain"].values.astype(np.int32)

y_val_heavy = df_val["target_is_heavy_rain"].values.astype(np.int32)
y_te_heavy = df_test["target_is_heavy_rain"].values.astype(np.int32)

# ==============================================================================
# INDEPENDENT METRIC CALCULATION ENGINE (FROM SCRATCH, NO METRICS_ENGINE IMPORT)
# ==============================================================================
def calc_continuous_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    
    n = len(y_true)
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
        # Pearson r
        cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        r = float(cov / (std_true * std_pred))
        r = float(np.clip(r, -1.0, 1.0))
        
        # Bias ratio
        beta = float(mean_pred / mean_true if mean_true > 1e-8 else 1.0)
        
        # Classic KGE (Gupta et al., 2009): alpha = std_pred / std_true
        alpha = float(std_pred / std_true if std_true > 1e-8 else 1.0)
        kge_classic = float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))
        
        # Modified KGE-2012 (Kling et al., 2012): gamma = CV_pred / CV_true
        cv_true = std_true / mean_true if mean_true > 1e-8 else 1.0
        cv_pred = std_pred / mean_pred if mean_pred > 1e-8 else 1.0
        gamma = cv_pred / cv_true if cv_true > 1e-8 else 1.0
        kge_2012 = float(1.0 - np.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2))
    else:
        r = 0.0
        alpha = 0.0
        beta = mean_pred / mean_true if mean_true > 1e-8 else 1.0
        gamma = 0.0
        kge_classic = -999.0
        kge_2012 = -999.0
        
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Bias": round(bias, 4),
        "PBias_pct": round(pbias, 2),
        "Pearson_r": round(r, 4),
        "NSE": round(nse, 4),
        "Alpha_ratio": round(alpha, 4),
        "Beta_ratio": round(beta, 4),
        "Gamma_ratio": round(gamma, 4),
        "KGE_classic_2009": round(kge_classic, 4),
        "KGE_2012": round(kge_2012, 4)
    }

def calc_contingency_metrics(y_true_binary, y_pred_binary):
    yt = np.asarray(y_true_binary, dtype=bool)
    yp = np.asarray(y_pred_binary, dtype=bool)
    
    h = int(np.sum(yt & yp))
    f = int(np.sum((~yt) & yp))
    m = int(np.sum(yt & (~yp)))
    c = int(np.sum((~yt) & (~yp)))
    n = len(yt)
    
    pod = (h / (h + m)) if (h + m) > 0 else 0.0
    far = (f / (h + f)) if (h + f) > 0 else 0.0
    precision = (h / (h + f)) if (h + f) > 0 else 0.0
    csi = (h / (h + m + f)) if (h + m + f) > 0 else 0.0
    
    h_random = ((h + m) * (h + f)) / n if n > 0 else 0.0
    ets_denom = (h + m + f - h_random)
    ets = ((h - h_random) / ets_denom) if ets_denom > 0 else 0.0
    
    hss_num = 2.0 * (h * c - f * m)
    hss_den = ((h + m) * (m + c) + (h + f) * (f + c))
    hss = (hss_num / hss_den) if hss_den > 0 else 0.0
    
    f1 = (2.0 * precision * pod / (precision + pod)) if (precision + pod) > 0 else 0.0
    
    return {
        "Hits": h,
        "False_Alarms": f,
        "Misses": m,
        "Correct_Negatives": c,
        "POD_Recall": round(float(pod), 4),
        "FAR": round(float(far), 4),
        "Precision": round(float(precision), 4),
        "CSI_Threat_Score": round(float(csi), 4),
        "ETS": round(float(ets), 4),
        "HSS": round(float(hss), 4),
        "F1_Score": round(float(f1), 4)
    }

def calc_probabilistic_metrics(y_true_binary, p_pred):
    yt = np.asarray(y_true_binary, dtype=np.int32)
    p = np.clip(np.asarray(p_pred, dtype=np.float64), 0.0, 1.0)
    
    bs = float(brier_score_loss(yt, p))
    try:
        pr_auc = float(average_precision_score(yt, p))
    except Exception:
        pr_auc = 0.0
    try:
        roc_auc = float(roc_auc_score(yt, p))
    except Exception:
        roc_auc = 0.5
        
    return {
        "Brier_Score": round(bs, 5),
        "PR_AUC": round(pr_auc, 4),
        "ROC_AUC": round(roc_auc, 4)
    }

# ==============================================================================
# AUDIT & RECONCILE ALL REPORTED CSV FILES
# ==============================================================================
reconciliation_records = []

# --- 1. Audit baseline_results.csv ---
print("\n--- Auditing baseline_results.csv ---")
df_base = pd.read_csv(stage4_dir / "baseline_results.csv")
base_pred_files = {
    ("Rainfall_Persistence", "Validation_2023"): (models_dir / "pred_baseline_persist_val.npy", y_val),
    ("Rainfall_Persistence", "Test_2024_2025"): (models_dir / "pred_baseline_persist_test.npy", y_te),
    ("Station_DOY_Climatology", "Validation_2023"): (models_dir / "pred_baseline_clim_val.npy", y_val),
    ("Station_DOY_Climatology", "Test_2024_2025"): (models_dir / "pred_baseline_clim_test.npy", y_te),
    ("Antecedent_ERA5_Land", "Validation_2023"): (models_dir / "pred_baseline_era5_val.npy", y_val),
    ("Antecedent_ERA5_Land", "Test_2024_2025"): (models_dir / "pred_baseline_era5_test.npy", y_te),
    ("Empirical_Quantile_Mapping_EQM", "Validation_2023"): (models_dir / "pred_baseline_eqm_val.npy", y_val),
    ("Empirical_Quantile_Mapping_EQM", "Test_2024_2025"): (models_dir / "pred_baseline_eqm_test.npy", y_te),
}

for idx, row in df_base.iterrows():
    key = (row["model_name"], row["partition"])
    pred_path, y_true = base_pred_files[key]
    y_pred = np.load(pred_path)
    
    # Recalculate
    c_met = calc_continuous_metrics(y_true, y_pred)
    cont_64 = calc_contingency_metrics(y_true >= 64.5, y_pred >= 64.5)
    
    diff_rmse = abs(row["RMSE_mm"] - c_met["RMSE"])
    diff_mae = abs(row["MAE_mm"] - c_met["MAE"])
    diff_bias = abs(row["Bias_mm"] - c_met["Bias"])
    diff_r = abs(row["Pearson_r"] - c_met["Pearson_r"])
    diff_nse = abs(row["NSE"] - c_met["NSE"])
    diff_kge = abs(row["KGE_2012"] - c_met["KGE_2012"])
    diff_csi = abs(row["CSI_64_5mm"] - cont_64["CSI_Threat_Score"])
    
    reconciliation_records.append({
        "source_file": "baseline_results.csv",
        "model_name": row["model_name"],
        "partition": row["partition"],
        "sample_count": row["sample_count"],
        "reported_RMSE": row["RMSE_mm"],
        "recalculated_RMSE": c_met["RMSE"],
        "diff_RMSE": round(diff_rmse, 6),
        "reported_MAE": row["MAE_mm"],
        "recalculated_MAE": c_met["MAE"],
        "diff_MAE": round(diff_mae, 6),
        "reported_Bias": row["Bias_mm"],
        "recalculated_Bias": c_met["Bias"],
        "diff_Bias": round(diff_bias, 6),
        "reported_Pearson_r": row["Pearson_r"],
        "recalculated_Pearson_r": c_met["Pearson_r"],
        "diff_r": round(diff_r, 6),
        "reported_NSE": row["NSE"],
        "recalculated_NSE": c_met["NSE"],
        "diff_NSE": round(diff_nse, 6),
        "reported_KGE_2012": row["KGE_2012"],
        "recalculated_KGE_2012": c_met["KGE_2012"],
        "recalculated_KGE_classic_2009": c_met["KGE_classic_2009"],
        "diff_KGE_2012": round(diff_kge, 6),
        "reported_CSI_64.5": row["CSI_64_5mm"],
        "recalculated_CSI_64.5": cont_64["CSI_Threat_Score"],
        "diff_CSI_64.5": round(diff_csi, 6),
        "reconciled_status": "MATCH" if max(diff_rmse, diff_mae, diff_kge, diff_csi) < 1e-3 else "DISCREPANCY"
    })

# --- 2. Audit regression_model_results.csv ---
print("\n--- Auditing regression_model_results.csv ---")
df_reg = pd.read_csv(stage4_dir / "regression_model_results.csv")

# For regression models, load saved predictions or recompute from saved joblib models
import joblib
df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
subdiv_map = df_train.groupby("subdivision")["target_rainfall_mm"].mean().to_dict()
zone_map = df_train.groupby("zone_id")["target_rainfall_mm"].mean().to_dict()

def encode_features_reg(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_map).fillna(3.5).values
    X["zone_id_enc"] = df["zone_id"].map(zone_map).fillna(3.5).values
    return X

X_va = encode_features_reg(df_val)
X_te = encode_features_reg(df_test)
scaler = joblib.load(models_dir / "standard_scaler.joblib")
X_va_scaled = scaler.transform(X_va)
X_te_scaled = scaler.transform(X_te)

# Load regression models
reg_models = {
    "Ridge_Linear_Baseline": joblib.load(models_dir / "ridge_regressor.joblib"),
    "ElasticNet_Linear_Baseline": joblib.load(models_dir / "elasticnet_regressor.joblib"),
    "LightGBM_RMSE": joblib.load(models_dir / "lgb_rmse_regressor.joblib"),
    "LightGBM_Tweedie_p1.5": joblib.load(models_dir / "lgb_tweedie_regressor.joblib"),
    "CatBoost_Tweedie_p1.5": joblib.load(models_dir / "cat_tweedie_regressor.joblib"),
    "XGBoost_Tweedie_p1.5": joblib.load(models_dir / "xgb_tweedie_regressor.joblib"),
    "LightGBM_Log_Regressor": joblib.load(models_dir / "lgb_log_regressor.joblib")
}

# Calculate Duan smearing factor on Train
X_tr = encode_features_reg(df_train)
y_tr = df_train["target_rainfall_mm"].values.astype(np.float64)
z_tr = np.log1p(np.clip(y_tr, 0.0, None))
z_pred_tr = reg_models["LightGBM_Log_Regressor"].predict(X_tr)
duan_S = float(np.mean(np.exp(z_tr - z_pred_tr)))

for idx, row in df_reg.iterrows():
    m_name = row["model_name"]
    part = row["partition"]
    X_curr = X_va if part == "Validation_2023" else X_te
    X_curr_sc = X_va_scaled if part == "Validation_2023" else X_te_scaled
    y_true = y_val if part == "Validation_2023" else y_te
    
    if m_name in ["Ridge_Linear_Baseline", "ElasticNet_Linear_Baseline"]:
        y_pred = reg_models[m_name].predict(X_curr_sc)
    elif m_name == "LightGBM_Log_OrdinaryInverse":
        z_pred = reg_models["LightGBM_Log_Regressor"].predict(X_curr)
        y_pred = np.clip(np.expm1(z_pred), 0.0, None)
    elif m_name == "LightGBM_Log_DuanSmearing":
        z_pred = reg_models["LightGBM_Log_Regressor"].predict(X_curr)
        y_pred = np.clip(duan_S * np.exp(z_pred) - 1.0, 0.0, None)
    else:
        y_pred = reg_models[m_name].predict(X_curr)
        
    c_met = calc_continuous_metrics(y_true, y_pred)
    cont_64 = calc_contingency_metrics(y_true >= 64.5, y_pred >= 64.5)
    
    diff_rmse = abs(row["RMSE"] - c_met["RMSE"])
    diff_mae = abs(row["MAE"] - c_met["MAE"])
    diff_bias = abs(row["Bias"] - c_met["Bias"])
    diff_r = abs(row["Pearson_r"] - c_met["Pearson_r"])
    diff_nse = abs(row["NSE"] - c_met["NSE"])
    diff_kge = abs(row["KGE_2012"] - c_met["KGE_2012"])
    diff_csi = abs(row["CSI_64.5mm"] - cont_64["CSI_Threat_Score"])
    
    reconciliation_records.append({
        "source_file": "regression_model_results.csv",
        "model_name": m_name,
        "partition": part,
        "sample_count": row["sample_count"],
        "reported_RMSE": row["RMSE"],
        "recalculated_RMSE": c_met["RMSE"],
        "diff_RMSE": round(diff_rmse, 6),
        "reported_MAE": row["MAE"],
        "recalculated_MAE": c_met["MAE"],
        "diff_MAE": round(diff_mae, 6),
        "reported_Bias": row["Bias"],
        "recalculated_Bias": c_met["Bias"],
        "diff_Bias": round(diff_bias, 6),
        "reported_Pearson_r": row["Pearson_r"],
        "recalculated_Pearson_r": c_met["Pearson_r"],
        "diff_r": round(diff_r, 6),
        "reported_NSE": row["NSE"],
        "recalculated_NSE": c_met["NSE"],
        "diff_NSE": round(diff_nse, 6),
        "reported_KGE_2012": row["KGE_2012"],
        "recalculated_KGE_2012": c_met["KGE_2012"],
        "recalculated_KGE_classic_2009": c_met["KGE_classic_2009"],
        "diff_KGE_2012": round(diff_kge, 6),
        "reported_CSI_64.5": row["CSI_64.5mm"],
        "recalculated_CSI_64.5": cont_64["CSI_Threat_Score"],
        "diff_CSI_64.5": round(diff_csi, 6),
        "reconciled_status": "MATCH" if max(diff_rmse, diff_mae, diff_kge, diff_csi) < 1e-3 else "DISCREPANCY"
    })

# --- 3. Audit hurdle_model_results.csv ---
print("\n--- Auditing hurdle_model_results.csv ---")
df_hurdle = pd.read_csv(stage4_dir / "hurdle_model_results.csv")
lgb_cond = joblib.load(models_dir / "hurdle_stage2_conditional_regressor.joblib")
lgb_rain = joblib.load(models_dir / "lgb_rain_classifier.joblib")

subdiv_rain_map = df_train.groupby("subdivision")["target_is_rain"].mean().to_dict()
zone_rain_map = df_train.groupby("zone_id")["target_is_rain"].mean().to_dict()
def encode_features_rain(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_rain_map).fillna(0.35).values
    X["zone_id_enc"] = df["zone_id"].map(zone_rain_map).fillna(0.35).values
    return X

X_va_rain = encode_features_rain(df_val)
X_te_rain = encode_features_rain(df_test)

p_val_rain = lgb_rain.predict_proba(X_va_rain)[:, 1]
p_te_rain = lgb_rain.predict_proba(X_te_rain)[:, 1]
y_cond_val = lgb_cond.predict(X_va)
y_cond_te = lgb_cond.predict(X_te)

hurdle_preds = {
    ("Hurdle_DiscreteThreshold_t0.44", "Validation_2023"): (np.where(p_val_rain >= 0.44, np.clip(y_cond_val, 0.0, None), 0.0), y_val),
    ("Hurdle_DiscreteThreshold_t0.44", "Test_2024_2025"): (np.where(p_te_rain >= 0.44, np.clip(y_cond_te, 0.0, None), 0.0), y_te),
    ("Hurdle_ExpectedValue", "Validation_2023"): (np.clip(p_val_rain * np.clip(y_cond_val, 0.0, None), 0.0, None), y_val),
    ("Hurdle_ExpectedValue", "Test_2024_2025"): (np.clip(p_te_rain * np.clip(y_cond_te, 0.0, None), 0.0, None), y_te)
}

for idx, row in df_hurdle.iterrows():
    key = (row["model_name"], row["partition"])
    y_pred, y_true = hurdle_preds[key]
    
    c_met = calc_continuous_metrics(y_true, y_pred)
    cont_64 = calc_contingency_metrics(y_true >= 64.5, y_pred >= 64.5)
    cont_01 = calc_contingency_metrics(y_true >= 0.1, y_pred >= 0.1)
    
    diff_rmse = abs(row["RMSE"] - c_met["RMSE"])
    diff_mae = abs(row["MAE"] - c_met["MAE"])
    diff_bias = abs(row["Bias"] - c_met["Bias"])
    diff_r = abs(row["Pearson_r"] - c_met["Pearson_r"])
    diff_nse = abs(row["NSE"] - c_met["NSE"])
    diff_kge = abs(row["KGE_2012"] - c_met["KGE_2012"])
    diff_csi = abs(row["CSI_Heavy_64.5mm"] - cont_64["CSI_Threat_Score"])
    
    reconciliation_records.append({
        "source_file": "hurdle_model_results.csv",
        "model_name": row["model_name"],
        "partition": row["partition"],
        "sample_count": row["sample_count"],
        "reported_RMSE": row["RMSE"],
        "recalculated_RMSE": c_met["RMSE"],
        "diff_RMSE": round(diff_rmse, 6),
        "reported_MAE": row["MAE"],
        "recalculated_MAE": c_met["MAE"],
        "diff_MAE": round(diff_mae, 6),
        "reported_Bias": row["Bias"],
        "recalculated_Bias": c_met["Bias"],
        "diff_Bias": round(diff_bias, 6),
        "reported_Pearson_r": row["Pearson_r"],
        "recalculated_Pearson_r": c_met["Pearson_r"],
        "diff_r": round(diff_r, 6),
        "reported_NSE": row["NSE"],
        "recalculated_NSE": c_met["NSE"],
        "diff_NSE": round(diff_nse, 6),
        "reported_KGE_2012": row["KGE_2012"],
        "recalculated_KGE_2012": c_met["KGE_2012"],
        "recalculated_KGE_classic_2009": c_met["KGE_classic_2009"],
        "diff_KGE_2012": round(diff_kge, 6),
        "reported_CSI_64.5": row["CSI_Heavy_64.5mm"],
        "recalculated_CSI_64.5": cont_64["CSI_Threat_Score"],
        "diff_CSI_64.5": round(diff_csi, 6),
        "reconciled_status": "MATCH" if max(diff_rmse, diff_mae, diff_kge, diff_csi) < 1e-3 else "DISCREPANCY"
    })

# --- 4. Audit rain_occurrence_results.csv ---
print("\n--- Auditing rain_occurrence_results.csv ---")
df_rain_clf = pd.read_csv(stage4_dir / "rain_occurrence_results.csv")
clf_rain_models = {
    "LightGBM_Rain": joblib.load(models_dir / "lgb_rain_classifier.joblib"),
    "XGBoost_Rain": joblib.load(models_dir / "xgb_rain_classifier.joblib"),
    "CatBoost_Rain": joblib.load(models_dir / "cat_rain_classifier.joblib")
}

for idx, row in df_rain_clf.iterrows():
    m_name = row["model_name"]
    part = row["partition"]
    X_curr = X_va_rain if part == "Validation_2023" else X_te_rain
    y_true_rain = y_val_rain if part == "Validation_2023" else y_te_rain
    
    p_pred = clf_rain_models[m_name].predict_proba(X_curr)[:, 1]
    t_th = row["calibrated_threshold"]
    
    prob_m = calc_probabilistic_metrics(y_true_rain, p_pred)
    cont_m = calc_contingency_metrics(y_true_rain, p_pred >= t_th)
    
    diff_roc = abs(row["ROC_AUC"] - prob_m["ROC_AUC"])
    diff_pr = abs(row["PR_AUC"] - prob_m["PR_AUC"])
    diff_bs = abs(row["Brier_Score"] - prob_m["Brier_Score"])
    diff_csi = abs(row["CSI_Threat_Score"] - cont_m["CSI_Threat_Score"])
    diff_hss = abs(row["HSS"] - cont_m["HSS"])
    
    reconciliation_records.append({
        "source_file": "rain_occurrence_results.csv",
        "model_name": m_name,
        "partition": part,
        "sample_count": row["sample_count"],
        "reported_RMSE": np.nan,
        "recalculated_RMSE": np.nan,
        "diff_RMSE": 0.0,
        "reported_MAE": np.nan,
        "recalculated_MAE": np.nan,
        "diff_MAE": 0.0,
        "reported_Bias": np.nan,
        "recalculated_Bias": np.nan,
        "diff_Bias": 0.0,
        "reported_Pearson_r": np.nan,
        "recalculated_Pearson_r": np.nan,
        "diff_r": 0.0,
        "reported_NSE": np.nan,
        "recalculated_NSE": np.nan,
        "diff_NSE": 0.0,
        "reported_KGE_2012": np.nan,
        "recalculated_KGE_2012": np.nan,
        "recalculated_KGE_classic_2009": np.nan,
        "diff_KGE_2012": 0.0,
        "reported_CSI_64.5": row["CSI_Threat_Score"],
        "recalculated_CSI_64.5": cont_m["CSI_Threat_Score"],
        "diff_CSI_64.5": round(diff_csi, 6),
        "reconciled_status": "MATCH" if max(diff_roc, diff_pr, diff_csi, diff_hss) < 1e-3 else "DISCREPANCY"
    })

# --- 5. Audit heavy_rain_results.csv ---
print("\n--- Auditing heavy_rain_results.csv ---")
df_heavy_clf = pd.read_csv(stage4_dir / "heavy_rain_results.csv")

# For focal loss, provide focal_loss_lgb in __main__
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

subdiv_heavy_map = df_train.groupby("subdivision")["target_is_heavy_rain"].mean().to_dict()
zone_heavy_map = df_train.groupby("zone_id")["target_is_heavy_rain"].mean().to_dict()
def encode_features_heavy(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_heavy_map).fillna(0.007).values
    X["zone_id_enc"] = df["zone_id"].map(zone_heavy_map).fillna(0.007).values
    return X

X_va_heavy = encode_features_heavy(df_val)
X_te_heavy = encode_features_heavy(df_test)

clf_heavy_models = {
    "LightGBM_Heavy_Unweighted": joblib.load(models_dir / "lgb_heavy_unweighted.joblib"),
    "LightGBM_Heavy_ClassWeighted": joblib.load(models_dir / "lgb_heavy_weighted.joblib"),
    "LightGBM_Heavy_FocalLoss": joblib.load(models_dir / "lgb_heavy_focal.joblib"),
    "CatBoost_Heavy_ClassWeighted": joblib.load(models_dir / "cat_heavy_weighted.joblib"),
    "XGBoost_Heavy_ClassWeighted": joblib.load(models_dir / "xgb_heavy_weighted.joblib")
}

for idx, row in df_heavy_clf.iterrows():
    m_name = row["model_name"]
    part = row["partition"]
    X_curr = X_va_heavy if part == "Validation_2023" else X_te_heavy
    y_true_heavy = y_val_heavy if part == "Validation_2023" else y_te_heavy
    
    if m_name == "LightGBM_Heavy_FocalLoss":
        raw_pred = clf_heavy_models[m_name].predict(X_curr)
        p_pred = 1.0 / (1.0 + np.exp(-raw_pred))
    else:
        p_pred = clf_heavy_models[m_name].predict_proba(X_curr)[:, 1]
        
    t_th = row["calibrated_threshold"]
    prob_m = calc_probabilistic_metrics(y_true_heavy, p_pred)
    cont_m = calc_contingency_metrics(y_true_heavy, p_pred >= t_th)
    
    diff_roc = abs(row["ROC_AUC"] - prob_m["ROC_AUC"])
    diff_pr = abs(row["PR_AUC"] - prob_m["PR_AUC"])
    diff_csi = abs(row["CSI_Threat_Score"] - cont_m["CSI_Threat_Score"])
    diff_hss = abs(row["HSS"] - cont_m["HSS"])
    
    reconciliation_records.append({
        "source_file": "heavy_rain_results.csv",
        "model_name": m_name,
        "partition": part,
        "sample_count": row["sample_count"],
        "reported_RMSE": np.nan,
        "recalculated_RMSE": np.nan,
        "diff_RMSE": 0.0,
        "reported_MAE": np.nan,
        "recalculated_MAE": np.nan,
        "diff_MAE": 0.0,
        "reported_Bias": np.nan,
        "recalculated_Bias": np.nan,
        "diff_Bias": 0.0,
        "reported_Pearson_r": np.nan,
        "recalculated_Pearson_r": np.nan,
        "diff_r": 0.0,
        "reported_NSE": np.nan,
        "recalculated_NSE": np.nan,
        "diff_NSE": 0.0,
        "reported_KGE_2012": np.nan,
        "recalculated_KGE_2012": np.nan,
        "recalculated_KGE_classic_2009": np.nan,
        "diff_KGE_2012": 0.0,
        "reported_CSI_64.5": row["CSI_Threat_Score"],
        "recalculated_CSI_64.5": cont_m["CSI_Threat_Score"],
        "diff_CSI_64.5": round(diff_csi, 6),
        "reconciled_status": "MATCH" if max(diff_roc, diff_pr, diff_csi, diff_hss) < 1e-3 else "DISCREPANCY"
    })

# Save independent metric reconciliation
df_reconcile = pd.DataFrame(reconciliation_records)
df_reconcile.to_csv(audit_dir / "independent_metric_recalculation.csv", index=False)
print("\nSaved independent_metric_recalculation.csv")
print(f"Total rows audited: {len(df_reconcile)}")
matches = (df_reconcile["reconciled_status"] == "MATCH").sum()
print(f"Total exact matches (< 0.001): {matches} / {len(df_reconcile)}")
