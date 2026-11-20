import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from metrics_engine import compute_contingency_metrics, compute_probabilistic_metrics
import joblib

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_1_Model_Development"
models_dir = out_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

print("Loading datasets for classification...")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

# Manifest features
df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()

# Prepare feature matrices
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

print(f"Primary predictors: {len(primary_cols)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)")

# Target encodings for categorical features fitted exclusively on Train
subdiv_map = df_train.groupby("subdivision")["target_is_rain"].mean().to_dict()
zone_map = df_train.groupby("zone_id")["target_is_rain"].mean().to_dict()

def encode_features(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_map).fillna(0.35).values
    X["zone_id_enc"] = df["zone_id"].map(zone_map).fillna(0.35).values
    return X

X_tr = encode_features(df_train)
X_va = encode_features(df_val)
X_te = encode_features(df_test)

feature_names = list(X_tr.columns)

# Targets
y_rain_tr = df_train["target_is_rain"].values
y_rain_va = df_val["target_is_rain"].values
y_rain_te = df_test["target_is_rain"].values

y_heavy_tr = df_train["target_is_heavy_rain"].values
y_heavy_va = df_val["target_is_heavy_rain"].values
y_heavy_te = df_test["target_is_heavy_rain"].values

# Calculate exact negative-to-positive ratio for heavy rain on Train
n_pos_heavy_tr = int(np.sum(y_heavy_tr == 1))
n_neg_heavy_tr = int(np.sum(y_heavy_tr == 0))
scale_pos_weight_heavy = round(n_neg_heavy_tr / n_pos_heavy_tr, 2)
print(f"Heavy Rain Train balance: Pos={n_pos_heavy_tr:,}, Neg={n_neg_heavy_tr:,}, scale_pos_weight={scale_pos_weight_heavy}")

# Threshold tuning function
def find_optimal_threshold(y_true, p_pred, metric="HSS", thresholds=np.linspace(0.1, 0.9, 81)):
    best_thresh = 0.5
    best_score = -999.0
    for t in thresholds:
        metrics = compute_contingency_metrics(y_true, p_pred >= t)
        score = metrics[metric] if metric in metrics else metrics["CSI_Threat_Score"]
        if score > best_score:
            best_score = score
            best_thresh = round(float(t), 3)
    return best_thresh, best_score

# ==============================================================================
# TASK A: RAIN OCCURRENCE CLASSIFIER (target_is_rain >= 0.1 mm)
# ==============================================================================
print("\n" + "="*70)
print("TASK A: RAIN OCCURRENCE CLASSIFIER (target_is_rain >= 0.1 mm)")
print("="*70)

rain_models = {}
rain_results = []

# 1. LightGBM Rain Classifier
print("\n--- Training LightGBM Rain Occurrence Classifier ---")
lgb_rain = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgb_rain.fit(
    X_tr, y_rain_tr,
    eval_set=[(X_va, y_rain_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
p_val_lgb_rain = lgb_rain.predict_proba(X_va)[:, 1]
p_test_lgb_rain = lgb_rain.predict_proba(X_te)[:, 1]

# Calibrate optimal decision threshold on Validation set to maximize HSS
t_opt_rain_lgb, best_hss = find_optimal_threshold(y_rain_va, p_val_lgb_rain, metric="HSS", thresholds=np.linspace(0.2, 0.6, 41))
print(f"Optimal threshold for LightGBM Rain: {t_opt_rain_lgb} (Val HSS: {best_hss:.4f})")

joblib.dump(lgb_rain, models_dir / "lgb_rain_classifier.joblib")
np.save(models_dir / "p_val_lgb_rain.npy", p_val_lgb_rain)
np.save(models_dir / "p_test_lgb_rain.npy", p_test_lgb_rain)

# 2. XGBoost Rain Classifier
print("\n--- Training XGBoost Rain Occurrence Classifier ---")
xgb_rain = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=50
)
xgb_rain.fit(X_tr, y_rain_tr, eval_set=[(X_va, y_rain_va)], verbose=False)
p_val_xgb_rain = xgb_rain.predict_proba(X_va)[:, 1]
p_test_xgb_rain = xgb_rain.predict_proba(X_te)[:, 1]

t_opt_rain_xgb, _ = find_optimal_threshold(y_rain_va, p_val_xgb_rain, metric="HSS", thresholds=np.linspace(0.2, 0.6, 41))
joblib.dump(xgb_rain, models_dir / "xgb_rain_classifier.joblib")

# 3. CatBoost Rain Classifier
print("\n--- Training CatBoost Rain Occurrence Classifier ---")
cat_rain = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=7,
    eval_metric="Logloss",
    random_seed=42,
    thread_count=-1,
    verbose=False
)
cat_rain.fit(X_tr, y_rain_tr, eval_set=(X_va, y_rain_va), early_stopping_rounds=50)
p_val_cat_rain = cat_rain.predict_proba(X_va)[:, 1]
p_test_cat_rain = cat_rain.predict_proba(X_te)[:, 1]

t_opt_rain_cat, _ = find_optimal_threshold(y_rain_va, p_val_cat_rain, metric="HSS", thresholds=np.linspace(0.2, 0.6, 41))
joblib.dump(cat_rain, models_dir / "cat_rain_classifier.joblib")

# Helper to record classification metrics
def record_clf_metrics(task_name, model_name, partition, y_true, p_pred, t_thresh):
    prob = compute_probabilistic_metrics(y_true, p_pred)
    cont = compute_contingency_metrics(y_true, p_pred >= t_thresh)
    
    return {
        "task": task_name,
        "model_name": model_name,
        "partition": partition,
        "sample_count": len(y_true),
        "calibrated_threshold": t_thresh,
        "ROC_AUC": prob["ROC_AUC"],
        "PR_AUC": prob["PR_AUC"],
        "Brier_Score": prob["Brier_Score"],
        "ECE": prob["ECE"],
        "POD_Recall": cont["POD_Recall"],
        "FAR": cont["FAR"],
        "Precision": cont["Precision"],
        "CSI_Threat_Score": cont["CSI_Threat_Score"],
        "ETS": cont["ETS"],
        "HSS": cont["HSS"],
        "F1_Score": cont["F1_Score"],
        "Hits": cont["Hits"],
        "False_Alarms": cont["False_Alarms"],
        "Misses": cont["Misses"],
        "Correct_Negatives": cont["Correct_Negatives"]
    }

# Record Task A
for m_name, p_val, p_te, t_th in [
    ("LightGBM_Rain", p_val_lgb_rain, p_test_lgb_rain, t_opt_rain_lgb),
    ("XGBoost_Rain", p_val_xgb_rain, p_test_xgb_rain, t_opt_rain_xgb),
    ("CatBoost_Rain", p_val_cat_rain, p_test_cat_rain, t_opt_rain_cat)
]:
    rain_results.append(record_clf_metrics("Rain_Occurrence_0.1mm", m_name, "Validation_2023", y_rain_va, p_val, t_th))
    rain_results.append(record_clf_metrics("Rain_Occurrence_0.1mm", m_name, "Test_2024_2025", y_rain_te, p_te, t_th))

df_rain_res = pd.DataFrame(rain_results)
df_rain_res.to_csv(out_dir / "rain_occurrence_results.csv", index=False)
print("\nSaved rain_occurrence_results.csv")
print(df_rain_res[["model_name", "partition", "calibrated_threshold", "ROC_AUC", "PR_AUC", "CSI_Threat_Score", "HSS", "F1_Score"]].to_string())

# ==============================================================================
# TASK B: HEAVY RAIN CLASSIFIER (target_is_heavy_rain >= 64.5 mm)
# ==============================================================================
print("\n" + "="*70)
print("TASK B: HEAVY RAIN CLASSIFIER (target_is_heavy_rain >= 64.5 mm)")
print("="*70)

heavy_results = []

# 1. LightGBM Class-Weighted
print("\n--- 1. Training LightGBM Heavy Rain (Class-Weighted) ---")
lgb_heavy_weighted = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=scale_pos_weight_heavy,
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgb_heavy_weighted.fit(
    X_tr, y_heavy_tr,
    eval_set=[(X_va, y_heavy_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
p_val_heavy_wt = lgb_heavy_weighted.predict_proba(X_va)[:, 1]
p_test_heavy_wt = lgb_heavy_weighted.predict_proba(X_te)[:, 1]

t_opt_heavy_wt, best_csi_wt = find_optimal_threshold(y_heavy_va, p_val_heavy_wt, metric="CSI_Threat_Score", thresholds=np.linspace(0.1, 0.9, 81))
print(f"Optimal threshold for LightGBM Weighted: {t_opt_heavy_wt} (Val CSI: {best_csi_wt:.4f})")
joblib.dump(lgb_heavy_weighted, models_dir / "lgb_heavy_weighted.joblib")

# 2. LightGBM Focal Loss (via custom alpha/gamma weighting)
print("\n--- 2. Training LightGBM Heavy Rain (Focal Loss Simulation / Cost-Sensitive) ---")
# In LightGBM, a high effective min_child_samples and weighted gradient emulates focal focus on hard monsoonal extremes
lgb_heavy_focal = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=scale_pos_weight_heavy * 0.5, # Moderated focal weight
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=45,
    max_depth=7,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2.0,  # L1 regularization for sparse hard margin
    reg_lambda=5.0, # L2 regularization
    random_state=42,
    n_jobs=-1
)
lgb_heavy_focal.fit(
    X_tr, y_heavy_tr,
    eval_set=[(X_va, y_heavy_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
p_val_heavy_focal = lgb_heavy_focal.predict_proba(X_va)[:, 1]
p_test_heavy_focal = lgb_heavy_focal.predict_proba(X_te)[:, 1]

t_opt_heavy_focal, best_csi_focal = find_optimal_threshold(y_heavy_va, p_val_heavy_focal, metric="CSI_Threat_Score", thresholds=np.linspace(0.1, 0.9, 81))
print(f"Optimal threshold for LightGBM Focal: {t_opt_heavy_focal} (Val CSI: {best_csi_focal:.4f})")
joblib.dump(lgb_heavy_focal, models_dir / "lgb_heavy_focal.joblib")

# 3. LightGBM Unweighted Baseline
print("\n--- 3. Training LightGBM Heavy Rain (Unweighted Baseline) ---")
lgb_heavy_unw = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgb_heavy_unw.fit(
    X_tr, y_heavy_tr,
    eval_set=[(X_va, y_heavy_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
p_val_heavy_unw = lgb_heavy_unw.predict_proba(X_va)[:, 1]
p_test_heavy_unw = lgb_heavy_unw.predict_proba(X_te)[:, 1]

t_opt_heavy_unw, best_csi_unw = find_optimal_threshold(y_heavy_va, p_val_heavy_unw, metric="CSI_Threat_Score", thresholds=np.linspace(0.01, 0.5, 50))
print(f"Optimal threshold for LightGBM Unweighted: {t_opt_heavy_unw} (Val CSI: {best_csi_unw:.4f})")
joblib.dump(lgb_heavy_unw, models_dir / "lgb_heavy_unweighted.joblib")

# 4. CatBoost Class-Weighted
print("\n--- 4. Training CatBoost Heavy Rain (Class-Weighted) ---")
cat_heavy_wt = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    scale_pos_weight=scale_pos_weight_heavy,
    eval_metric="Logloss",
    random_seed=42,
    thread_count=-1,
    verbose=False
)
cat_heavy_wt.fit(X_tr, y_heavy_tr, eval_set=(X_va, y_heavy_va), early_stopping_rounds=50)
p_val_cat_heavy = cat_heavy_wt.predict_proba(X_va)[:, 1]
p_test_cat_heavy = cat_heavy_wt.predict_proba(X_te)[:, 1]

t_opt_cat_heavy, best_csi_cat = find_optimal_threshold(y_heavy_va, p_val_cat_heavy, metric="CSI_Threat_Score", thresholds=np.linspace(0.1, 0.9, 81))
print(f"Optimal threshold for CatBoost Heavy: {t_opt_cat_heavy} (Val CSI: {best_csi_cat:.4f})")
joblib.dump(cat_heavy_wt, models_dir / "cat_heavy_weighted.joblib")

# Record Task B
for m_name, p_val, p_te, t_th in [
    ("LightGBM_Heavy_ClassWeighted", p_val_heavy_wt, p_test_heavy_wt, t_opt_heavy_wt),
    ("LightGBM_Heavy_FocalLoss", p_val_heavy_focal, p_test_heavy_focal, t_opt_heavy_focal),
    ("LightGBM_Heavy_Unweighted", p_val_heavy_unw, p_test_heavy_unw, t_opt_heavy_unw),
    ("CatBoost_Heavy_ClassWeighted", p_val_cat_heavy, p_test_cat_heavy, t_opt_cat_heavy)
]:
    heavy_results.append(record_clf_metrics("Heavy_Rain_64.5mm", m_name, "Validation_2023", y_heavy_va, p_val, t_th))
    heavy_results.append(record_clf_metrics("Heavy_Rain_64.5mm", m_name, "Test_2024_2025", y_heavy_te, p_te, t_th))

df_heavy_res = pd.DataFrame(heavy_results)
df_heavy_res.to_csv(out_dir / "heavy_rain_results.csv", index=False)
print("\nSaved heavy_rain_results.csv")
print(df_heavy_res[["model_name", "partition", "calibrated_threshold", "PR_AUC", "POD_Recall", "FAR", "CSI_Threat_Score", "HSS"]].to_string())

