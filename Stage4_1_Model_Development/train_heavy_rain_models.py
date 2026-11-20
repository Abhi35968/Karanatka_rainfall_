import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from metrics_engine import compute_contingency_metrics, compute_probabilistic_metrics
import joblib

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_1_Model_Development"
models_dir = out_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

print("Loading datasets for Task B (Heavy Rain)...")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()

cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

subdiv_map = df_train.groupby("subdivision")["target_is_heavy_rain"].mean().to_dict()
zone_map = df_train.groupby("zone_id")["target_is_heavy_rain"].mean().to_dict()

def encode_features(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_map).fillna(0.007).values
    X["zone_id_enc"] = df["zone_id"].map(zone_map).fillna(0.007).values
    return X

X_tr = encode_features(df_train)
X_va = encode_features(df_val)
X_te = encode_features(df_test)

y_heavy_tr = df_train["target_is_heavy_rain"].values
y_heavy_va = df_val["target_is_heavy_rain"].values
y_heavy_te = df_test["target_is_heavy_rain"].values

n_pos = int(np.sum(y_heavy_tr == 1))
n_neg = int(np.sum(y_heavy_tr == 0))
scale_pos_weight = round(n_neg / n_pos, 2)
print(f"Heavy Rain Train Balance: Pos={n_pos:,}, Neg={n_neg:,}, scale_pos_weight={scale_pos_weight}")

def find_optimal_threshold(y_true, p_pred, metric="CSI_Threat_Score", thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 97)
    best_thresh = 0.5
    best_score = -999.0
    for t in thresholds:
        metrics = compute_contingency_metrics(y_true, p_pred >= t)
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_thresh = round(float(t), 3)
    return best_thresh, best_score

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

heavy_results = []

# 1. LightGBM Unweighted Baseline (with threshold tuning)
print("\n--- 1. LightGBM Unweighted Baseline ---")
lgb_unw = lgb.LGBMClassifier(
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
lgb_unw.fit(
    X_tr, y_heavy_tr,
    eval_set=[(X_va, y_heavy_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
p_val_unw = lgb_unw.predict_proba(X_va)[:, 1]
p_te_unw = lgb_unw.predict_proba(X_te)[:, 1]
t_opt_unw, best_csi_unw = find_optimal_threshold(y_heavy_va, p_val_unw, thresholds=np.linspace(0.01, 0.60, 60))
print(f"LGBM Unweighted: it={lgb_unw.best_iteration_}, Val CSI={best_csi_unw:.4f} at t={t_opt_unw}")
joblib.dump(lgb_unw, models_dir / "lgb_heavy_unweighted.joblib")

# 2. LightGBM Class-Weighted (scale_pos_weight, stopping on AUC)
print("\n--- 2. LightGBM Class-Weighted ---")
lgb_wt = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    scale_pos_weight=scale_pos_weight,
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgb_wt.fit(
    X_tr, y_heavy_tr,
    eval_set=[(X_va, y_heavy_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
p_val_wt = lgb_wt.predict_proba(X_va)[:, 1]
p_te_wt = lgb_wt.predict_proba(X_te)[:, 1]
t_opt_wt, best_csi_wt = find_optimal_threshold(y_heavy_va, p_val_wt, thresholds=np.linspace(0.2, 0.98, 79))
print(f"LGBM Weighted: it={lgb_wt.best_iteration_}, Val CSI={best_csi_wt:.4f} at t={t_opt_wt}")
joblib.dump(lgb_wt, models_dir / "lgb_heavy_weighted.joblib")

# 3. LightGBM Focal Loss (Custom Objective)
print("\n--- 3. LightGBM Focal Loss ---")
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

lgb_focal = lgb.LGBMClassifier(
    objective=focal_loss_lgb,
    metric="auc",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgb_focal.fit(
    X_tr, y_heavy_tr,
    eval_set=[(X_va, y_heavy_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
raw_val_focal = lgb_focal.predict(X_va)
raw_te_focal = lgb_focal.predict(X_te)
p_val_focal = 1.0 / (1.0 + np.exp(-raw_val_focal))
p_te_focal = 1.0 / (1.0 + np.exp(-raw_te_focal))
t_opt_focal, best_csi_focal = find_optimal_threshold(y_heavy_va, p_val_focal, thresholds=np.linspace(0.01, 0.95, 95))
print(f"LGBM Focal: it={lgb_focal.best_iteration_}, Val CSI={best_csi_focal:.4f} at t={t_opt_focal}")
joblib.dump(lgb_focal, models_dir / "lgb_heavy_focal.joblib")

# 4. CatBoost Class-Weighted
print("\n--- 4. CatBoost Class-Weighted ---")
cat_wt = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    scale_pos_weight=scale_pos_weight,
    eval_metric="Logloss",
    random_seed=42,
    thread_count=-1,
    verbose=False
)
cat_wt.fit(X_tr, y_heavy_tr, eval_set=(X_va, y_heavy_va), early_stopping_rounds=50)
p_val_cat = cat_wt.predict_proba(X_va)[:, 1]
p_te_cat = cat_wt.predict_proba(X_te)[:, 1]
t_opt_cat, best_csi_cat = find_optimal_threshold(y_heavy_va, p_val_cat, thresholds=np.linspace(0.1, 0.98, 89))
print(f"CatBoost Weighted: Val CSI={best_csi_cat:.4f} at t={t_opt_cat}")
joblib.dump(cat_wt, models_dir / "cat_heavy_weighted.joblib")

# 5. XGBoost Class-Weighted
print("\n--- 5. XGBoost Class-Weighted ---")
xgb_wt = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="auc",
    early_stopping_rounds=50
)
xgb_wt.fit(X_tr, y_heavy_tr, eval_set=[(X_va, y_heavy_va)], verbose=False)
p_val_xgb = xgb_wt.predict_proba(X_va)[:, 1]
p_te_xgb = xgb_wt.predict_proba(X_te)[:, 1]
t_opt_xgb, best_csi_xgb = find_optimal_threshold(y_heavy_va, p_val_xgb, thresholds=np.linspace(0.2, 0.98, 79))
print(f"XGBoost Weighted: Val CSI={best_csi_xgb:.4f} at t={t_opt_xgb}")
joblib.dump(xgb_wt, models_dir / "xgb_heavy_weighted.joblib")

# Record all models
model_runs = [
    ("LightGBM_Heavy_Unweighted", p_val_unw, p_te_unw, t_opt_unw),
    ("LightGBM_Heavy_ClassWeighted", p_val_wt, p_te_wt, t_opt_wt),
    ("LightGBM_Heavy_FocalLoss", p_val_focal, p_te_focal, t_opt_focal),
    ("CatBoost_Heavy_ClassWeighted", p_val_cat, p_te_cat, t_opt_cat),
    ("XGBoost_Heavy_ClassWeighted", p_val_xgb, p_te_xgb, t_opt_xgb)
]

for m_name, p_val, p_te, t_th in model_runs:
    heavy_results.append(record_clf_metrics("Heavy_Rain_64.5mm", m_name, "Validation_2023", y_heavy_va, p_val, t_th))
    heavy_results.append(record_clf_metrics("Heavy_Rain_64.5mm", m_name, "Test_2024_2025", y_heavy_te, p_te, t_th))

df_heavy = pd.DataFrame(heavy_results)
df_heavy.to_csv(out_dir / "heavy_rain_results.csv", index=False)
print("\n" + "="*80)
print("FINAL TASK B HEAVY RAIN RESULTS")
print("="*80)
print(df_heavy[["model_name", "partition", "calibrated_threshold", "PR_AUC", "POD_Recall", "FAR", "CSI_Threat_Score", "HSS", "Hits", "False_Alarms", "Misses"]].to_string())
