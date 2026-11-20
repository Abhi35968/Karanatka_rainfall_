import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
pq_path = base_dir / "Stage3_Feature_Engineering/stage3_dataset.parquet"
df_stage3 = pd.read_parquet(pq_path)
df_stage3["date"] = pd.to_datetime(df_stage3["date"])

# Load raw source datasets for independent verification
df_raw_rain = pd.read_csv(base_dir / "Stage2_Corrected_Processing/data/corrected_daily_rainfall.csv")
df_raw_rain["date"] = pd.to_datetime(df_raw_rain["date"])

df_raw_flux = pd.read_csv(base_dir / "Stage2_Corrected_Processing/data/corrected_flux_features.csv")
df_raw_flux["date"] = pd.to_datetime(df_raw_flux["date"])

print(f"Stage 3 rows: {len(df_stage3):,}")
print(f"Raw Rain rows: {len(df_raw_rain):,}, Raw Flux rows: {len(df_raw_flux):,}")

# Select 50 stratified random samples across years, seasons, and subdivisions
np.random.seed(101)
sample_indices = []

# Group by year and season to ensure stratified coverage
for yr in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    for s_name in ["Pre-Monsoon", "SW-Monsoon", "Post-Monsoon", "Winter"]:
        sub = df_stage3[(df_stage3["date"].dt.year == yr) & (df_stage3["season"] == s_name)]
        if len(sub) > 0:
            chosen = np.random.choice(sub.index, size=min(2, len(sub)), replace=False)
            sample_indices.extend(chosen.tolist())

if len(sample_indices) > 50:
    # Shuffle and pick exactly 50
    np.random.shuffle(sample_indices)
    sample_indices = sorted(sample_indices[:50])
elif len(sample_indices) < 50:
    remaining = list(set(df_stage3.index) - set(sample_indices))
    extra = np.random.choice(remaining, size=(50 - len(sample_indices)), replace=False)
    sample_indices.extend(extra.tolist())
    sample_indices = sorted(sample_indices)
else:
    sample_indices = sorted(sample_indices)

print(f"Selected {len(sample_indices)} stratified test samples across 2019-2025.")

reverification_results = []
discrepancy_count = 0

for idx in sample_indices:
    row = df_stage3.loc[idx]
    d_target = row["date"]
    st_id = int(row["station_id"])
    dist = row["district"]
    subdiv = row["subdivision"]
    season = row["season"]
    
    target_val = float(row["target_rainfall_mm"])
    
    # Trace dates
    d_lag1 = d_target - pd.Timedelta(days=1)
    d_lag2 = d_target - pd.Timedelta(days=2)
    d_lag3 = d_target - pd.Timedelta(days=3)
    
    # Ground truth lookup in raw tables
    raw_r_t = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_target)].iloc[0]["imd_observed_mm"]
    raw_r_1 = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_lag1)].iloc[0]["imd_observed_mm"]
    raw_r_2 = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_lag2)].iloc[0]["imd_observed_mm"]
    raw_r_3 = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_lag3)].iloc[0]["imd_observed_mm"]
    
    raw_e_1 = df_raw_rain[(df_raw_rain["station_id"] == st_id) & (df_raw_rain["date"] == d_lag1)].iloc[0]["corrected_era5_imd_aligned_mm"]
    raw_t_1 = df_raw_flux[(df_raw_flux["station_id"] == st_id) & (df_raw_flux["date"] == d_lag1)].iloc[0]["t2m_mean_c"]
    raw_sw_1 = df_raw_flux[(df_raw_flux["station_id"] == st_id) & (df_raw_flux["date"] == d_lag1)].iloc[0]["swvl1"]
    
    # Expected roll3 sum = r_1 + r_2 + r_3
    exp_roll3 = raw_r_1 + raw_r_2 + raw_r_3
    
    # Stage 3 values
    st3_lag1 = float(row["target_rain_lag_1d"])
    st3_lag2 = float(row["target_rain_lag_2d"])
    st3_lag3 = float(row["target_rain_lag_3d"])
    st3_roll3 = float(row["target_rain_roll_3d_sum"])
    st3_era_lag1 = float(row["era5_tp_lag_1d"])
    st3_t2m_lag1 = float(row["t2m_mean_lag_1d"])
    st3_swvl1_lag1 = float(row["swvl1_lag_1d"])
    
    # Check exact matches
    match_target = np.isclose(target_val, raw_r_t, atol=1e-3)
    match_lag1 = np.isclose(st3_lag1, raw_r_1, atol=1e-3)
    match_lag2 = np.isclose(st3_lag2, raw_r_2, atol=1e-3)
    match_roll3 = np.isclose(st3_roll3, exp_roll3, atol=1e-2)
    match_era1 = np.isclose(st3_era_lag1, raw_e_1, atol=1e-3)
    match_t1 = np.isclose(st3_t2m_lag1, raw_t_1, atol=1e-2)
    match_sw1 = np.isclose(st3_swvl1_lag1, raw_sw_1, atol=1e-3)
    
    all_matched = match_target and match_lag1 and match_lag2 and match_roll3 and match_era1 and match_t1 and match_sw1
    if not all_matched:
        discrepancy_count += 1
        status = "FAILED"
    else:
        status = "PASSED"
        
    issue_time_str = f"{d_lag1.strftime('%Y-%m-%d')} 03:00 UTC"
    target_window_str = f"{d_lag1.strftime('%Y-%m-%d')} 03:00 -> {d_target.strftime('%Y-%m-%d')} 03:00 UTC"
    
    reverification_results.append({
        "sample_id": len(reverification_results) + 1,
        "station_id": st_id,
        "district": dist,
        "subdivision": subdiv,
        "season": season,
        "target_date": d_target.strftime("%Y-%m-%d"),
        "prediction_issue_time": issue_time_str,
        "target_observation_window": target_window_str,
        "target_rainfall_mm": target_val,
        "lag1_target_rain_mm": st3_lag1,
        "raw_source_d_minus_1_rain_mm": raw_r_1,
        "lag1_era5_tp_mm": st3_era_lag1,
        "raw_source_d_minus_1_era5_mm": raw_e_1,
        "roll3_sum_mm": st3_roll3,
        "expected_roll3_sum_mm": round(float(exp_roll3), 4),
        "lag1_t2m_c": st3_t2m_lag1,
        "raw_source_d_minus_1_t2m_c": raw_t_1,
        "lag1_swvl1": st3_swvl1_lag1,
        "raw_source_d_minus_1_swvl1": raw_sw_1,
        "audit_verdict": status
    })

df_rever = pd.DataFrame(reverification_results)
out_rever = base_dir / "Stage3_1_Audit/data/feature_timestamp_reverification.csv"
df_rever.to_csv(out_rever, index=False)

print(f"Reverification completed: {len(df_rever)} samples tested.")
print(f"Discrepancies / Causality Violations: {discrepancy_count}")
print(f"Audit Pass Rate: {(len(df_rever) - discrepancy_count) / len(df_rever) * 100:.1f}%")
print("\nSample of First 5 Reverification Records:")
print(df_rever[["sample_id", "station_id", "target_date", "prediction_issue_time", "target_rainfall_mm", "lag1_target_rain_mm", "raw_source_d_minus_1_rain_mm", "audit_verdict"]].head(5).to_string())
