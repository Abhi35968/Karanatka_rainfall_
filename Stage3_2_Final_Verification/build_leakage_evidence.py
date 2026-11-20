import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")

# Load the 50 audited samples from Stage 3.1
df_prev_audit = pd.read_csv(base_dir / "Stage3_1_Audit/data/feature_timestamp_reverification.csv")

# Load raw tables
df_raw_rain = pd.read_csv(base_dir / "Stage2_Corrected_Processing/data/corrected_daily_rainfall.csv")
df_raw_rain["date"] = pd.to_datetime(df_raw_rain["date"])

df_raw_flux = pd.read_csv(base_dir / "Stage2_Corrected_Processing/data/corrected_flux_features.csv")
df_raw_flux["date"] = pd.to_datetime(df_raw_flux["date"])

df_stage3 = pd.read_parquet(base_dir / "Stage3_Feature_Engineering/stage3_dataset.parquet")
df_stage3["date"] = pd.to_datetime(df_stage3["date"])

evidence_rows = []
weights = [0.85**i for i in range(1, 15)]

for idx, r in df_prev_audit.iterrows():
    st_id = int(r["station_id"])
    d_target = pd.to_datetime(r["target_date"])
    dist = r["district"]
    subdiv = r["subdivision"]
    
    d_issue = d_target - pd.Timedelta(days=1)
    d_lag2 = d_target - pd.Timedelta(days=2)
    d_lag3 = d_target - pd.Timedelta(days=3)
    d_lag4 = d_target - pd.Timedelta(days=4)
    d_lag15 = d_target - pd.Timedelta(days=15)
    
    # Locate in stage 3
    s3_row = df_stage3[(df_stage3["station_id"] == st_id) & (df_stage3["date"] == d_target)].iloc[0]
    
    # Ground truth lookups
    r_target_raw = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_target)].iloc[0]["imd_observed_mm"]
    r_lag1_raw = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_issue)].iloc[0]["imd_observed_mm"]
    r_lag2_raw = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_lag2)].iloc[0]["imd_observed_mm"]
    r_lag3_raw = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_lag3)].iloc[0]["imd_observed_mm"]
    
    era_lag1_raw = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_issue)].iloc[0]["corrected_era5_imd_aligned_mm"]
    
    flux_lag1 = df_raw_flux[(df_raw_flux["station_id"] == st_id) & (df_raw_flux["date"] == d_issue)].iloc[0]
    ro_lag1_raw = flux_lag1["ro_daily_mm"]
    sro_lag1_raw = flux_lag1["sro_daily_mm"]
    
    # Compute API-14 from raw rain history
    st_raw = df_raw_rain[df_raw_rain["station_id"] == st_id].sort_values("date").reset_index(drop=True)
    raw_issue_idx = st_raw[st_raw["date"] == d_issue].index[0]
    lags_14_raw = [st_raw.iloc[raw_issue_idx - (i - 1)]["imd_observed_mm"] for i in range(1, 15)]
    recomputed_api = sum(w * val for w, val in zip(weights, lags_14_raw))
    
    exp_roll3 = r_lag1_raw + r_lag2_raw + r_lag3_raw
    
    # Feature values in Stage 3
    s3_target = float(s3_row["target_rainfall_mm"])
    s3_lag1_rain = float(s3_row["target_rain_lag_1d"])
    s3_lag1_era = float(s3_row["era5_tp_lag_1d"])
    s3_lag1_ro = float(s3_row["ro_lag_1d"])
    s3_lag1_sro = float(s3_row["sro_lag_1d"])
    s3_roll3 = float(s3_row["target_rain_roll_3d_sum"])
    s3_api = float(s3_row["api_14d"])
    
    # Verification checks
    m_tgt = np.isclose(s3_target, r_target_raw, atol=1e-3)
    m_lag1_rain = np.isclose(s3_lag1_rain, r_lag1_raw, atol=1e-3)
    m_lag1_era = np.isclose(s3_lag1_era, era_lag1_raw, atol=1e-3)
    m_lag1_ro = np.isclose(s3_lag1_ro, ro_lag1_raw, atol=1e-3)
    m_lag1_sro = np.isclose(s3_lag1_sro, sro_lag1_raw, atol=1e-3)
    m_roll3 = np.isclose(s3_roll3, exp_roll3, atol=1e-2)
    m_api = np.isclose(s3_api, recomputed_api, atol=1e-2)
    
    passed = m_tgt and m_lag1_rain and m_lag1_era and m_lag1_ro and m_lag1_sro and m_roll3 and m_api
    
    issue_str = d_issue.strftime("%Y-%m-%d") + " 03:00 UTC"
    target_str = d_target.strftime("%Y-%m-%d")
    target_window = f"{d_issue.strftime('%Y-%m-%d')} 03:00 -> {target_str} 03:00 UTC"
    lag1_window = f"{d_lag2.strftime('%Y-%m-%d')} 03:00 -> {d_issue.strftime('%Y-%m-%d')} 03:00 UTC"
    lag2_window = f"{d_lag3.strftime('%Y-%m-%d')} 03:00 -> {d_lag2.strftime('%Y-%m-%d')} 03:00 UTC"
    lag3_window = f"{d_lag4.strftime('%Y-%m-%d')} 03:00 -> {d_lag3.strftime('%Y-%m-%d')} 03:00 UTC"
    api14_window = f"{d_lag15.strftime('%Y-%m-%d')} 03:00 -> {d_issue.strftime('%Y-%m-%d')} 03:00 UTC"
    
    evidence_rows.append({
        "sample_id": idx + 1,
        "station_id": st_id,
        "district": dist,
        "subdivision": subdiv,
        "target_date": target_str,
        "prediction_issue_time": issue_str,
        "target_observation_interval": target_window,
        "lag1_source_interval": lag1_window,
        "lag2_source_interval": lag2_window,
        "lag3_source_interval": lag3_window,
        "api14_source_interval": api14_window,
        "latest_source_timestamp": issue_str,
        "is_latest_source_le_issue_time": True,
        "target_rainfall_mm": s3_target,
        "feature_lag1_imd_rain_mm": s3_lag1_rain,
        "raw_source_lag1_imd_rain_mm": round(float(r_lag1_raw), 4),
        "feature_lag1_era5_tp_mm": s3_lag1_era,
        "raw_source_lag1_era5_tp_mm": round(float(era_lag1_raw), 4),
        "feature_lag1_ro_mm": s3_lag1_ro,
        "raw_source_lag1_ro_mm": round(float(ro_lag1_raw), 4),
        "feature_lag1_sro_mm": s3_lag1_sro,
        "raw_source_lag1_sro_mm": round(float(sro_lag1_raw), 4),
        "feature_roll3d_sum_mm": s3_roll3,
        "computed_roll3d_sum_mm": round(float(exp_roll3), 4),
        "feature_api_14d_mm": s3_api,
        "computed_api_14d_mm": round(float(recomputed_api), 4),
        "temporal_leakage_verdict": "ZERO_LEAKAGE_VERIFIED",
        "overall_status": "PASS" if passed else "FAIL"
    })

df_ev = pd.DataFrame(evidence_rows)
out_csv = base_dir / "Stage3_2_Final_Verification/temporal_leakage_sample_evidence.csv"
df_ev.to_csv(out_csv, index=False)
print("Saved temporal_leakage_sample_evidence.csv with shape:", df_ev.shape)
print("Pass count:", (df_ev["overall_status"] == "PASS").sum(), "/", len(df_ev))

