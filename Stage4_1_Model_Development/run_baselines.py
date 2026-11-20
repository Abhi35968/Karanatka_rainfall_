import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from metrics_engine import compute_regression_metrics, evaluate_at_thresholds

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_1_Model_Development"
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading final datasets...")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

y_train = df_train["target_rainfall_mm"].values
y_val = df_val["target_rainfall_mm"].values
y_test = df_test["target_rainfall_mm"].values

results = []

def record_baseline_metrics(model_name, partition_name, y_true, y_pred):
    reg = compute_regression_metrics(y_true, y_pred)
    cat = evaluate_at_thresholds(y_true, y_pred)
    
    row = {
        "model_name": model_name,
        "partition": partition_name,
        "sample_count": len(y_true),
        "MAE_mm": reg["MAE"],
        "RMSE_mm": reg["RMSE"],
        "Bias_mm": reg["Bias"],
        "PBias_pct": reg["PBias_pct"],
        "R2": reg["R2"],
        "Pearson_r": reg["Pearson_r"],
        "NSE": reg["NSE"],
        "KGE_2012": reg["KGE_2012"],
        # Categorical metrics at 0.1 mm (Rain occurrence)
        "POD_0_1mm": cat["T_0.1mm"]["POD_Recall"],
        "FAR_0_1mm": cat["T_0.1mm"]["FAR"],
        "CSI_0_1mm": cat["T_0.1mm"]["CSI_Threat_Score"],
        "HSS_0_1mm": cat["T_0.1mm"]["HSS"],
        # Categorical metrics at 15.6 mm (Moderate rain)
        "POD_15_6mm": cat["T_15.6mm"]["POD_Recall"],
        "FAR_15_6mm": cat["T_15.6mm"]["FAR"],
        "CSI_15_6mm": cat["T_15.6mm"]["CSI_Threat_Score"],
        "HSS_15_6mm": cat["T_15.6mm"]["HSS"],
        # Categorical metrics at 35.5 mm (Custom diagnostic)
        "POD_35_5mm": cat["T_35.5mm"]["POD_Recall"],
        "FAR_35_5mm": cat["T_35.5mm"]["FAR"],
        "CSI_35_5mm": cat["T_35.5mm"]["CSI_Threat_Score"],
        "HSS_35_5mm": cat["T_35.5mm"]["HSS"],
        # Categorical metrics at 64.5 mm (Official IMD Heavy Rain)
        "POD_64_5mm": cat["T_64.5mm"]["POD_Recall"],
        "FAR_64_5mm": cat["T_64.5mm"]["FAR"],
        "CSI_64_5mm": cat["T_64.5mm"]["CSI_Threat_Score"],
        "HSS_64_5mm": cat["T_64.5mm"]["HSS"]
    }
    results.append(row)
    print(f"[{model_name} | {partition_name}] RMSE={reg['RMSE']:.2f}, MAE={reg['MAE']:.2f}, r={reg['Pearson_r']:.3f}, KGE={reg['KGE_2012']:.3f}, CSI_64.5={cat['T_64.5mm']['CSI_Threat_Score']:.4f}")

# -------------------------------------------------------------
# 1. Baseline 1: Rainfall Persistence (y_hat = lag1_imd_rain)
# -------------------------------------------------------------
print("\n--- 1. Evaluating Rainfall Persistence ---")
pred_persist_val = df_val["target_rain_lag_1d"].values
pred_persist_test = df_test["target_rain_lag_1d"].values

record_baseline_metrics("Rainfall_Persistence", "Validation_2023", y_val, pred_persist_val)
record_baseline_metrics("Rainfall_Persistence", "Test_2024_2025", y_test, pred_persist_test)

# -------------------------------------------------------------
# 2. Baseline 2: Station Day-of-Year Climatology (Train only)
# -------------------------------------------------------------
print("\n--- 2. Evaluating Station Day-of-Year Climatology ---")
# Compute DOY mean for each station on Train
doy_means = df_train.groupby(["station_id", "day_of_year"])["target_rainfall_mm"].mean().reset_index()

# Apply 7-day circular Gaussian smoothing for each station across days 1..366
smoothed_doy_dict = {}
for st_id, group in doy_means.groupby("station_id"):
    s_series = group.set_index("day_of_year")["target_rainfall_mm"].reindex(range(1, 367)).fillna(0.0).values
    # Circular boundary padding
    padded = np.concatenate([s_series[-7:], s_series, s_series[:7]])
    smoothed = gaussian_filter1d(padded, sigma=1.5, mode="wrap")[7:-7]
    for d, val in enumerate(smoothed, start=1):
        smoothed_doy_dict[(st_id, d)] = max(0.0, float(val))

pred_clim_val = np.array([smoothed_doy_dict.get((s, d), 0.0) for s, d in zip(df_val["station_id"], df_val["day_of_year"])])
pred_clim_test = np.array([smoothed_doy_dict.get((s, d), 0.0) for s, d in zip(df_test["station_id"], df_test["day_of_year"])])

record_baseline_metrics("Station_DOY_Climatology", "Validation_2023", y_val, pred_clim_val)
record_baseline_metrics("Station_DOY_Climatology", "Test_2024_2025", y_test, pred_clim_test)

# -------------------------------------------------------------
# 3. Baseline 3: Antecedent ERA5 Precipitation (lag1_era5_tp)
# -------------------------------------------------------------
print("\n--- 3. Evaluating Antecedent ERA5-Land Reanalysis ---")
pred_era5_val = df_val["era5_tp_lag_1d"].values
pred_era5_test = df_test["era5_tp_lag_1d"].values

record_baseline_metrics("Antecedent_ERA5_Land", "Validation_2023", y_val, pred_era5_val)
record_baseline_metrics("Antecedent_ERA5_Land", "Test_2024_2025", y_test, pred_era5_test)

# -------------------------------------------------------------
# 4. Baseline 4: Empirical Quantile Mapping (Fitted on Train)
# -------------------------------------------------------------
print("\n--- 4. Evaluating Empirical Quantile Mapping (EQM) ---")
# Fit EQM per subdivision on Train
eqm_models = {}
for subdiv in df_train["subdivision"].unique():
    sub_tr = df_train[df_train["subdivision"] == subdiv]
    x_tr = sub_tr["era5_tp_lag_1d"].values
    y_tr = sub_tr["target_rainfall_mm"].values
    
    # Wet day threshold matching: P(y < 0.1 mm)
    p0_obs = np.mean(y_tr < 0.1)
    x0_thresh = np.quantile(x_tr, p0_obs)
    
    # Positive wet day quantiles
    wet_x = x_tr[x_tr > x0_thresh]
    wet_y = y_tr[y_tr >= 0.1]
    
    if len(wet_x) > 50 and len(wet_y) > 50:
        pcts = np.linspace(0.01, 0.99, 100)
        q_x = np.quantile(wet_x, pcts)
        q_y = np.quantile(wet_y, pcts)
        max_x = np.max(wet_x)
        max_y = np.max(wet_y)
    else:
        q_x, q_y, max_x, max_y = None, None, 0.0, 0.0
        
    eqm_models[subdiv] = {
        "x0_thresh": x0_thresh,
        "q_x": q_x,
        "q_y": q_y,
        "max_x": max_x,
        "max_y": max_y
    }

def apply_eqm(df):
    preds = np.zeros(len(df), dtype=np.float64)
    for subdiv, model in eqm_models.items():
        idx = (df["subdivision"] == subdiv).values
        if not np.any(idx):
            continue
        x_vals = df.loc[idx, "era5_tp_lag_1d"].values
        sub_preds = np.zeros(len(x_vals), dtype=np.float64)
        
        x0 = model["x0_thresh"]
        q_x = model["q_x"]
        q_y = model["q_y"]
        max_x = model["max_x"]
        max_y = model["max_y"]
        
        wet_mask = (x_vals > x0)
        if q_x is not None and np.any(wet_mask):
            wx = x_vals[wet_mask]
            # Interpolate for normal range
            mapped = np.interp(wx, q_x, q_y)
            # Extrapolate beyond training maximum
            extrap_mask = (wx > max_x)
            mapped[extrap_mask] = max_y + (wx[extrap_mask] - max_x)
            sub_preds[wet_mask] = mapped
            
        preds[idx] = sub_preds
    return preds

pred_eqm_val = apply_eqm(df_val)
pred_eqm_test = apply_eqm(df_test)

record_baseline_metrics("Empirical_Quantile_Mapping_EQM", "Validation_2023", y_val, pred_eqm_val)
record_baseline_metrics("Empirical_Quantile_Mapping_EQM", "Test_2024_2025", y_test, pred_eqm_test)

# Save baseline predictions for ensemble/hurdle synthesis
np.save(out_dir / "models/pred_baseline_persist_val.npy", pred_persist_val)
np.save(out_dir / "models/pred_baseline_persist_test.npy", pred_persist_test)
np.save(out_dir / "models/pred_baseline_clim_val.npy", pred_clim_val)
np.save(out_dir / "models/pred_baseline_clim_test.npy", pred_clim_test)
np.save(out_dir / "models/pred_baseline_era5_val.npy", pred_era5_val)
np.save(out_dir / "models/pred_baseline_era5_test.npy", pred_era5_test)
np.save(out_dir / "models/pred_baseline_eqm_val.npy", pred_eqm_val)
np.save(out_dir / "models/pred_baseline_eqm_test.npy", pred_eqm_test)

df_res = pd.DataFrame(results)
out_csv = out_dir / "baseline_results.csv"
df_res.to_csv(out_csv, index=False)
print(f"\nSaved baseline_results.csv with {len(df_res)} rows.")
print(df_res[["model_name", "partition", "RMSE_mm", "MAE_mm", "Pearson_r", "KGE_2012", "CSI_0_1mm", "CSI_64_5mm"]].to_string())

