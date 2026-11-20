import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

def compute_regression_metrics(y_true, y_pred):
    """
    Computes standard hydrometeorological continuous regression metrics:
    MAE, RMSE, Bias, Percent Bias (PBias), R2, Pearson r, NSE, KGE-2012.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # Clip negative predictions to zero
    y_pred = np.clip(y_pred, 0.0, None)
    
    n = len(y_true)
    if n == 0:
        return {}
        
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    
    sum_true = np.sum(y_true)
    pbias = (np.sum(diff) / sum_true * 100.0) if sum_true > 0 else 0.0
    
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    ss_res = np.sum(diff ** 2)
    
    nse = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = nse  # In hydrological modeling, NSE is mathematically equivalent to Nash-Sutcliffe R2
    
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    
    if std_true > 1e-8 and std_pred > 1e-8:
        r = np.corrcoef(y_true, y_pred)[0, 1]
        r = 0.0 if np.isnan(r) else r
        beta = mean_pred / mean_true if mean_true > 1e-8 else 1.0
        cv_true = std_true / mean_true if mean_true > 1e-8 else 1.0
        cv_pred = std_pred / mean_pred if mean_pred > 1e-8 else 1.0
        gamma = cv_pred / cv_true if cv_true > 1e-8 else 1.0
        kge = 1.0 - np.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2)
    else:
        r = 0.0
        kge = -999.0
        
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "Bias": round(float(bias), 4),
        "PBias_pct": round(float(pbias), 2),
        "R2": round(float(r2), 4),
        "Pearson_r": round(float(r), 4),
        "NSE": round(float(nse), 4),
        "KGE_2012": round(float(kge), 4)
    }

def compute_contingency_metrics(y_true_binary, y_pred_binary):
    """
    Computes categorical 2x2 contingency table metrics:
    Hits (H), False Alarms (F), Misses (M), Correct Negatives (C),
    POD (Recall), FAR, Precision, CSI (Threat Score), ETS, HSS, F1.
    """
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
    
    # ETS (Equitable Threat Score)
    h_random = ((h + m) * (h + f)) / n if n > 0 else 0.0
    ets_denom = (h + m + f - h_random)
    ets = ((h - h_random) / ets_denom) if ets_denom > 0 else 0.0
    
    # HSS (Heidke Skill Score)
    hss_num = 2.0 * (h * c - f * m)
    hss_den = ((h + m) * (m + c) + (h + f) * (f + c))
    hss = (hss_num / hss_den) if hss_den > 0 else 0.0
    
    # F1 Score
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

def evaluate_at_thresholds(y_true, y_pred, thresholds=[0.1, 15.6, 35.5, 64.5]):
    """
    Evaluates categorical performance across multiple operational rainfall thresholds.
    """
    results = {}
    for t in thresholds:
        t_key = f"T_{t}mm"
        yt_b = (y_true >= t)
        yp_b = (y_pred >= t)
        res = compute_contingency_metrics(yt_b, yp_b)
        results[t_key] = res
    return results

def compute_probabilistic_metrics(y_true_binary, p_pred):
    """
    Computes probabilistic calibration metrics: Brier score, PR-AUC, ROC-AUC, ECE.
    """
    yt = np.asarray(y_true_binary, dtype=np.int32)
    p = np.asarray(p_pred, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    
    bs = brier_score_loss(yt, p)
    try:
        pr_auc = average_precision_score(yt, p)
    except Exception:
        pr_auc = 0.0
    try:
        roc_auc = roc_auc_score(yt, p)
    except Exception:
        roc_auc = 0.5
        
    # Expected Calibration Error (ECE) with 10 bins
    bin_limits = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    n = len(yt)
    for b in range(10):
        in_bin = (p >= bin_limits[b]) & (p < bin_limits[b+1]) if b < 9 else (p >= bin_limits[b]) & (p <= bin_limits[b+1])
        n_bin = np.sum(in_bin)
        if n_bin > 0:
            acc_bin = np.mean(yt[in_bin])
            conf_bin = np.mean(p[in_bin])
            ece += (n_bin / n) * np.abs(acc_bin - conf_bin)
            
    return {
        "Brier_Score": round(float(bs), 5),
        "PR_AUC": round(float(pr_auc), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "ECE": round(float(ece), 4)
    }

