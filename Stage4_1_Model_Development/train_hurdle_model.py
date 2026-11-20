import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from metrics_engine import compute_regression_metrics, compute_contingency_metrics
import joblib

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_1_Model_Development"
models_dir = out_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

print("Loading datasets for Task D (Hurdle Model)...")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()

cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

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

# Filter for wet days strictly on Train partition
mask_wet_tr = (y_tr >= 0.1)
X_tr_wet = X_tr.loc[mask_wet_tr]
y_tr_wet = y_tr[mask_wet_tr]

# Filter for wet days on Validation partition for early stopping
mask_wet_va = (y_va >= 0.1)
X_va_wet = X_va.loc[mask_wet_va]
y_va_wet = y_va[mask_wet_va]

print(f"Stage 2 Conditional Training Set: {len(X_tr_wet):,} wet records ({np.mean(mask_wet_tr)*100:.1f}%)")

# Train Stage 2 Conditional Regressor (Tweedie p=1.5 on wet days)
print("\n--- Training Stage 2 Conditional Regressor (LightGBM Tweedie on Wet Days) ---")
lgb_cond = lgb.LGBMRegressor(
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
lgb_cond.fit(
    X_tr_wet, y_tr_wet,
    eval_set=[(X_va_wet, y_va_wet)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
joblib.dump(lgb_cond, models_dir / "hurdle_stage2_conditional_regressor.joblib")

# Predictions of conditional regressor across all samples
y_cond_val = lgb_cond.predict(X_va)
y_cond_te = lgb_cond.predict(X_te)

# Load Stage 1 Occurrence Classifier predictions
# Note: X_tr encoding for rain occurrence used rain rate mapping, but lgb_rain uses primary features
clf_rain = joblib.load(models_dir / "lgb_rain_classifier.joblib")
subdiv_rain_map = df_train.groupby("subdivision")["target_is_rain"].mean().to_dict()
zone_rain_map = df_train.groupby("zone_id")["target_is_rain"].mean().to_dict()

def encode_features_rain(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_rain_map).fillna(0.35).values
    X["zone_id_enc"] = df["zone_id"].map(zone_rain_map).fillna(0.35).values
    return X

X_va_rain = encode_features_rain(df_val)
X_te_rain = encode_features_rain(df_test)

p_val_rain = clf_rain.predict_proba(X_va_rain)[:, 1]
p_te_rain = clf_rain.predict_proba(X_te_rain)[:, 1]
t_opt = 0.44  # Frozen calibrated threshold from Task A

# Two Hurdle formulations:
# 1. Discrete Threshold Hurdle: y_hat = y_cond if p >= t_opt else 0.0
y_val_hurdle_thresh = np.where(p_val_rain >= t_opt, np.clip(y_cond_val, 0.0, None), 0.0)
y_te_hurdle_thresh = np.where(p_te_rain >= t_opt, np.clip(y_cond_te, 0.0, None), 0.0)

# 2. Expected Value Hurdle: y_hat = p * y_cond
y_val_hurdle_ev = np.clip(p_val_rain * np.clip(y_cond_val, 0.0, None), 0.0, None)
y_te_hurdle_ev = np.clip(p_te_rain * np.clip(y_cond_te, 0.0, None), 0.0, None)

def record_hurdle_metrics(model_name, partition, y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0, None)
    c_met = compute_regression_metrics(y_true, y_pred)
    cont_64 = compute_contingency_metrics(y_true >= 64.5, y_pred >= 64.5)
    cont_01 = compute_contingency_metrics(y_true >= 0.1, y_pred >= 0.1)
    
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
        "CSI_Rain_0.1mm": cont_01["CSI_Threat_Score"],
        "HSS_Rain_0.1mm": cont_01["HSS"],
        "CSI_Heavy_64.5mm": cont_64["CSI_Threat_Score"],
        "POD_Heavy_64.5mm": cont_64["POD_Recall"],
        "FAR_Heavy_64.5mm": cont_64["FAR"],
        "HSS_Heavy_64.5mm": cont_64["HSS"]
    }

hurdle_results = [
    record_hurdle_metrics("Hurdle_DiscreteThreshold_t0.44", "Validation_2023", y_va, y_val_hurdle_thresh),
    record_hurdle_metrics("Hurdle_DiscreteThreshold_t0.44", "Test_2024_2025", y_te, y_te_hurdle_thresh),
    record_hurdle_metrics("Hurdle_ExpectedValue", "Validation_2023", y_va, y_val_hurdle_ev),
    record_hurdle_metrics("Hurdle_ExpectedValue", "Test_2024_2025", y_te, y_te_hurdle_ev)
]

# Save hurdle predictions
np.save(models_dir / "pred_hurdle_thresh_test.npy", y_te_hurdle_thresh)
np.save(models_dir / "pred_hurdle_ev_test.npy", y_te_hurdle_ev)

df_hurdle = pd.DataFrame(hurdle_results)
df_hurdle.to_csv(out_dir / "hurdle_model_results.csv", index=False)
print("\n" + "="*80)
print("FINAL TASK D HURDLE MODEL RESULTS")
print("="*80)
print(df_hurdle[["model_name", "partition", "RMSE", "MAE", "Bias", "KGE_2012", "CSI_Rain_0.1mm", "CSI_Heavy_64.5mm"]].to_string())

