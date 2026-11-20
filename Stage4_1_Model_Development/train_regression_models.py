import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from metrics_engine import compute_regression_metrics, compute_contingency_metrics
import joblib

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_1_Model_Development"
models_dir = out_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

print("Loading datasets for Task C (Regression)...")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()

cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

# Target encoding for regression using training mean target
subdiv_map = df_train.groupby("subdivision")["target_rainfall_mm"].mean().to_dict()
zone_map = df_train.groupby("zone_id")["target_rainfall_mm"].mean().to_dict()

def encode_features(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_map).fillna(3.5).values
    X["zone_id_enc"] = df["zone_id"].map(zone_map).fillna(3.5).values
    return X

X_tr = encode_features(df_train)
X_va = encode_features(df_val)
X_te = encode_features(df_test)

y_tr = df_train["target_rainfall_mm"].values
y_va = df_val["target_rainfall_mm"].values
y_te = df_test["target_rainfall_mm"].values

# Standard scaler fitted strictly on Train for linear baselines
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)
X_te_scaled = scaler.transform(X_te)
joblib.dump(scaler, models_dir / "standard_scaler.joblib")

def record_reg_metrics(model_name, partition, y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0, None)
    c_met = compute_regression_metrics(y_true, y_pred)
    cont_64 = compute_contingency_metrics(y_true >= 64.5, y_pred >= 64.5)
    
    return {
        "model_name": model_name,
        "partition": partition,
        "sample_count": len(y_true),
        "MAE": c_met["MAE"],
        "RMSE": c_met["RMSE"],
        "Bias": c_met["Bias"],
        "PBias_pct": c_met["PBias_pct"],
        "Pearson_r": c_met["Pearson_r"],
        "NSE": c_met["NSE"],
        "KGE_2012": c_met["KGE_2012"],
        "CSI_64.5mm": cont_64["CSI_Threat_Score"],
        "POD_64.5mm": cont_64["POD_Recall"],
        "FAR_64.5mm": cont_64["FAR"],
        "HSS_64.5mm": cont_64["HSS"]
    }

reg_results = []

# 1. Ridge Regression Baseline
print("\n--- 1. Ridge Regression Baseline ---")
ridge = Ridge(alpha=100.0, random_state=42)
ridge.fit(X_tr_scaled, y_tr)
y_val_ridge = ridge.predict(X_va_scaled)
y_te_ridge = ridge.predict(X_te_scaled)
joblib.dump(ridge, models_dir / "ridge_regressor.joblib")
reg_results.append(record_reg_metrics("Ridge_Linear_Baseline", "Validation_2023", y_va, y_val_ridge))
reg_results.append(record_reg_metrics("Ridge_Linear_Baseline", "Test_2024_2025", y_te, y_te_ridge))

# 2. ElasticNet Baseline
print("\n--- 2. ElasticNet Baseline ---")
enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
enet.fit(X_tr_scaled, y_tr)
y_val_enet = enet.predict(X_va_scaled)
y_te_enet = enet.predict(X_te_scaled)
joblib.dump(enet, models_dir / "elasticnet_regressor.joblib")
reg_results.append(record_reg_metrics("ElasticNet_Linear_Baseline", "Validation_2023", y_va, y_val_enet))
reg_results.append(record_reg_metrics("ElasticNet_Linear_Baseline", "Test_2024_2025", y_te, y_te_enet))

# 3. LightGBM RMSE Regressor
print("\n--- 3. LightGBM RMSE Regressor ---")
lgb_rmse = lgb.LGBMRegressor(
    objective="regression",
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
lgb_rmse.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
y_val_lgb_rmse = lgb_rmse.predict(X_va)
y_te_lgb_rmse = lgb_rmse.predict(X_te)
joblib.dump(lgb_rmse, models_dir / "lgb_rmse_regressor.joblib")
reg_results.append(record_reg_metrics("LightGBM_RMSE", "Validation_2023", y_va, y_val_lgb_rmse))
reg_results.append(record_reg_metrics("LightGBM_RMSE", "Test_2024_2025", y_te, y_te_lgb_rmse))

# 4. Retransformation Bias Experiment: Log-Transformed LightGBM
print("\n--- 4. Retransformation Bias Experiment (Log-Transformed LightGBM) ---")
z_tr = np.log1p(np.clip(y_tr, 0.0, None))
z_va = np.log1p(np.clip(y_va, 0.0, None))
z_te = np.log1p(np.clip(y_te, 0.0, None))

lgb_log = lgb.LGBMRegressor(
    objective="regression",
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
lgb_log.fit(
    X_tr, z_tr,
    eval_set=[(X_va, z_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
joblib.dump(lgb_log, models_dir / "lgb_log_regressor.joblib")

# Predictions on log scale
z_pred_tr = lgb_log.predict(X_tr)
z_pred_va = lgb_log.predict(X_va)
z_pred_te = lgb_log.predict(X_te)

# 4a. Ordinary inverse: exp(z) - 1
y_val_log_ord = np.clip(np.expm1(z_pred_va), 0.0, None)
y_te_log_ord = np.clip(np.expm1(z_pred_te), 0.0, None)
reg_results.append(record_reg_metrics("LightGBM_Log_OrdinaryInverse", "Validation_2023", y_va, y_val_log_ord))
reg_results.append(record_reg_metrics("LightGBM_Log_OrdinaryInverse", "Test_2024_2025", y_te, y_te_log_ord))

# 4b. Duan's Smearing correction: S * exp(z) - 1, S = mean(exp(e_train))
train_residuals = z_tr - z_pred_tr
duan_smearing_factor = float(np.mean(np.exp(train_residuals)))
print(f"Duan Smearing Factor S calculated on Train: {duan_smearing_factor:.5f}")

y_val_log_duan = np.clip(duan_smearing_factor * np.exp(z_pred_va) - 1.0, 0.0, None)
y_te_log_duan = np.clip(duan_smearing_factor * np.exp(z_pred_te) - 1.0, 0.0, None)
reg_results.append(record_reg_metrics("LightGBM_Log_DuanSmearing", "Validation_2023", y_va, y_val_log_duan))
reg_results.append(record_reg_metrics("LightGBM_Log_DuanSmearing", "Test_2024_2025", y_te, y_te_log_duan))

# 5. Raw Tweedie Regressors (Bypassing log-retransformation bias)
print("\n--- 5. LightGBM Tweedie Regressor (p=1.5) ---")
lgb_tweedie = lgb.LGBMRegressor(
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
lgb_tweedie.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
y_val_lgb_tweedie = lgb_tweedie.predict(X_va)
y_te_lgb_tweedie = lgb_tweedie.predict(X_te)
joblib.dump(lgb_tweedie, models_dir / "lgb_tweedie_regressor.joblib")
reg_results.append(record_reg_metrics("LightGBM_Tweedie_p1.5", "Validation_2023", y_va, y_val_lgb_tweedie))
reg_results.append(record_reg_metrics("LightGBM_Tweedie_p1.5", "Test_2024_2025", y_te, y_te_lgb_tweedie))

# 6. CatBoost Tweedie Regressor
print("\n--- 6. CatBoost Tweedie Regressor (p=1.5) ---")
cat_tweedie = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=1000,
    learning_rate=0.03,
    depth=7,
    random_seed=42,
    thread_count=-1,
    verbose=False
)
cat_tweedie.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
y_val_cat_tweedie = cat_tweedie.predict(X_va)
y_te_cat_tweedie = cat_tweedie.predict(X_te)
joblib.dump(cat_tweedie, models_dir / "cat_tweedie_regressor.joblib")
reg_results.append(record_reg_metrics("CatBoost_Tweedie_p1.5", "Validation_2023", y_va, y_val_cat_tweedie))
reg_results.append(record_reg_metrics("CatBoost_Tweedie_p1.5", "Test_2024_2025", y_te, y_te_cat_tweedie))

# 7. XGBoost Tweedie Regressor
print("\n--- 7. XGBoost Tweedie Regressor (p=1.5) ---")
xgb_tweedie = xgb.XGBRegressor(
    objective="reg:tweedie",
    tweedie_variance_power=1.5,
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)
xgb_tweedie.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
y_val_xgb_tweedie = xgb_tweedie.predict(X_va)
y_te_xgb_tweedie = xgb_tweedie.predict(X_te)
joblib.dump(xgb_tweedie, models_dir / "xgb_tweedie_regressor.joblib")
reg_results.append(record_reg_metrics("XGBoost_Tweedie_p1.5", "Validation_2023", y_va, y_val_xgb_tweedie))
reg_results.append(record_reg_metrics("XGBoost_Tweedie_p1.5", "Test_2024_2025", y_te, y_te_xgb_tweedie))

# Save predictions for test evaluation and figures
np.save(models_dir / "pred_lgb_tweedie_test.npy", y_te_lgb_tweedie)
np.save(models_dir / "pred_lgb_rmse_test.npy", y_te_lgb_rmse)
np.save(models_dir / "pred_cat_tweedie_test.npy", y_te_cat_tweedie)

df_reg = pd.DataFrame(reg_results)
df_reg.to_csv(out_dir / "regression_model_results.csv", index=False)
print("\n" + "="*80)
print("FINAL TASK C REGRESSION MODEL RESULTS")
print("="*80)
print(df_reg[["model_name", "partition", "RMSE", "MAE", "Bias", "Pearson_r", "NSE", "KGE_2012", "CSI_64.5mm"]].to_string())

