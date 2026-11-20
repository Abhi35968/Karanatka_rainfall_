import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
stage4_dir = base_dir / "Stage4_1_Model_Development"
models_dir = stage4_dir / "models"
audit_dir = base_dir / "Stage4_2_Independent_Model_Audit"
audit_dir.mkdir(parents=True, exist_ok=True)

print("Running Task 3: Heavy Rain Classifier Verification on Test Partition (2024–2025)...")

df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
y_te_raw = df_test["target_rainfall_mm"].values.astype(np.float64)
y_te_binary = (y_te_raw >= 64.5).astype(np.int32)

n_total = len(y_te_binary)
n_pos = int(np.sum(y_te_binary == 1))
n_neg = int(np.sum(y_te_binary == 0))
base_rate_pct = (n_pos / n_total) * 100.0

print(f"Test Partition Size: {n_total:,} rows")
print(f"Official Heavy Rain Events (>= 64.5 mm): {n_pos:,} ({base_rate_pct:.4f}%)")
print(f"Non-Heavy Events (< 64.5 mm): {n_neg:,} ({100.0 - base_rate_pct:.4f}%)")

# Feature preparation for ML models
df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")
primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
subdiv_heavy_map = df_train.groupby("subdivision")["target_is_heavy_rain"].mean().to_dict()
zone_heavy_map = df_train.groupby("zone_id")["target_is_heavy_rain"].mean().to_dict()

def encode_features_heavy(df):
    X = df[num_cols].copy()
    X["subdivision_enc"] = df["subdivision"].map(subdiv_heavy_map).fillna(0.007).values
    X["zone_id_enc"] = df["zone_id"].map(zone_heavy_map).fillna(0.007).values
    return X

X_te_heavy = encode_features_heavy(df_test)

# Confusion matrix calculation
def compute_detailed_contingency(y_true, y_pred_bin, p_pred=None, model_name="", threshold_note=""):
    yt = (y_true == 1)
    yp = (y_pred_bin == 1)
    
    hits = int(np.sum(yt & yp))
    false_alarms = int(np.sum((~yt) & yp))
    misses = int(np.sum(yt & (~yp)))
    correct_negatives = int(np.sum((~yt) & (~yp)))
    n = len(yt)
    
    pod = (hits / (hits + misses)) if (hits + misses) > 0 else 0.0
    far = (false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    precision = (hits / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    csi = (hits / (hits + misses + false_alarms)) if (hits + misses + false_alarms) > 0 else 0.0
    
    # ETS
    h_rand = ((hits + misses) * (hits + false_alarms)) / n if n > 0 else 0.0
    ets_den = (hits + misses + false_alarms - h_rand)
    ets = ((hits - h_rand) / ets_den) if ets_den > 0 else 0.0
    
    # HSS
    hss_num = 2.0 * (hits * correct_negatives - false_alarms * misses)
    hss_den = ((hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (false_alarms + correct_negatives))
    hss = (hss_num / hss_den) if hss_den > 0 else 0.0
    
    # F1
    f1 = (2.0 * precision * pod / (precision + pod)) if (precision + pod) > 0 else 0.0
    
    # Probabilistic if available
    pr_auc = float(average_precision_score(yt, p_pred)) if p_pred is not None else np.nan
    roc_auc = float(roc_auc_score(yt, p_pred)) if p_pred is not None else np.nan
    brier = float(brier_score_loss(yt, p_pred)) if p_pred is not None else np.nan
    
    return {
        "model_name": model_name,
        "evaluation_partition": "Test_2024_2025",
        "sample_count": n,
        "positive_event_count": hits + misses,
        "threshold_selection_protocol": threshold_note,
        "calibrated_decision_threshold": p_pred is not None,
        "Hits": hits,
        "False_Alarms": false_alarms,
        "Misses": misses,
        "Correct_Negatives": correct_negatives,
        "Precision": round(precision, 4),
        "POD_Recall": round(pod, 4),
        "FAR": round(far, 4),
        "CSI_Threat_Score": round(csi, 4),
        "ETS": round(ets, 4),
        "HSS": round(hss, 4),
        "F1_Score": round(f1, 4),
        "PR_AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else "N/A",
        "ROC_AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
        "Brier_Score": round(brier, 5) if not np.isnan(brier) else "N/A"
    }

records = []

# 1. LightGBM Focal Loss
# Function for focal loss
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

m_focal = joblib.load(models_dir / "lgb_heavy_focal.joblib")
raw_focal = m_focal.predict(X_te_heavy)
p_focal = 1.0 / (1.0 + np.exp(-raw_focal))
rec_focal = compute_detailed_contingency(
    y_te_binary, (p_focal >= 0.53).astype(int), p_pred=p_focal,
    model_name="LightGBM_Heavy_FocalLoss",
    threshold_note="Frozen threshold p*=0.53 tuned exclusively on 2023 Validation to maximize CSI"
)
rec_focal["calibrated_decision_threshold"] = 0.53
records.append(rec_focal)

# 2. LightGBM Class Weighted
m_lgb_wt = joblib.load(models_dir / "lgb_heavy_weighted.joblib")
p_lgb_wt = m_lgb_wt.predict_proba(X_te_heavy)[:, 1]
rec_lgb_wt = compute_detailed_contingency(
    y_te_binary, (p_lgb_wt >= 0.96).astype(int), p_pred=p_lgb_wt,
    model_name="LightGBM_Heavy_ClassWeighted",
    threshold_note="Frozen threshold p*=0.96 tuned exclusively on 2023 Validation to maximize CSI"
)
rec_lgb_wt["calibrated_decision_threshold"] = 0.96
records.append(rec_lgb_wt)

# 3. CatBoost Class Weighted
m_cat_wt = joblib.load(models_dir / "cat_heavy_weighted.joblib")
p_cat_wt = m_cat_wt.predict_proba(X_te_heavy)[:, 1]
rec_cat_wt = compute_detailed_contingency(
    y_te_binary, (p_cat_wt >= 0.98).astype(int), p_pred=p_cat_wt,
    model_name="CatBoost_Heavy_ClassWeighted",
    threshold_note="Frozen threshold p*=0.98 tuned exclusively on 2023 Validation to maximize CSI"
)
rec_cat_wt["calibrated_decision_threshold"] = 0.98
records.append(rec_cat_wt)

# 4. Rainfall Persistence
pred_persist = np.load(models_dir / "pred_baseline_persist_test.npy")
bin_persist = (pred_persist >= 64.5).astype(int)
rec_persist = compute_detailed_contingency(
    y_te_binary, bin_persist, p_pred=None,
    model_name="Rainfall_Persistence",
    threshold_note="Physical 64.5 mm threshold applied directly to y_{t-1} observations"
)
rec_persist["calibrated_decision_threshold"] = "64.5 mm (fixed physical)"
records.append(rec_persist)

# 5. Raw ERA5-Land
pred_era5 = np.load(models_dir / "pred_baseline_era5_test.npy")
bin_era5 = (pred_era5 >= 64.5).astype(int)
rec_era5 = compute_detailed_contingency(
    y_te_binary, bin_era5, p_pred=None,
    model_name="Raw_ERA5_Land",
    threshold_note="Physical 64.5 mm threshold applied directly to raw antecedent total precipitation"
)
rec_era5["calibrated_decision_threshold"] = "64.5 mm (fixed physical)"
records.append(rec_era5)

# 6. Empirical Quantile Mapping (EQM)
pred_eqm = np.load(models_dir / "pred_baseline_eqm_test.npy")
bin_eqm = (pred_eqm >= 64.5).astype(int)
rec_eqm = compute_detailed_contingency(
    y_te_binary, bin_eqm, p_pred=None,
    model_name="Empirical_Quantile_Mapping_EQM",
    threshold_note="Physical 64.5 mm threshold applied to station-specific quantile-mapped rainfall"
)
rec_eqm["calibrated_decision_threshold"] = "64.5 mm (fixed physical)"
records.append(rec_eqm)

df_out = pd.DataFrame(records)
df_out.to_csv(audit_dir / "heavy_rain_confusion_matrices.csv", index=False)
print("\nSaved heavy_rain_confusion_matrices.csv")
print(df_out[["model_name", "calibrated_decision_threshold", "Hits", "False_Alarms", "Misses", "Correct_Negatives", "Precision", "POD_Recall", "FAR", "CSI_Threat_Score", "HSS", "ETS"]].to_string())

