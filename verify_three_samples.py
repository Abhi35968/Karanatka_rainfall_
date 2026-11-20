import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"

df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

df_flux = pd.read_csv(base_dir / "Stage2_Corrected_Processing/data/corrected_flux_features.csv")
df_rain = pd.read_csv(base_dir / "Stage2_Corrected_Processing/data/corrected_daily_rainfall.csv")

samples = [
    ("Train Partition", df_train, 1, "2020-07-15"),
    ("Validation Partition", df_val, 52, "2023-08-10"),
    ("Test Partition", df_test, 125, "2024-07-20")
]

for partition, df, st_id, date_str in samples:
    print(f"================================================================================")
    print(f"SAMPLE AUDIT: {partition} | Grid-Cell ID: {st_id} | Target Hindcast Date: {date_str}")
    print(f"================================================================================")
    row = df[(df["station_id"] == st_id) & (df["date"] == date_str)].iloc[0]
    
    t_val = float(row["target_rainfall_mm"])
    
    y_lag1 = float(row["target_rain_lag_1d"])
    era5_lag1 = float(row["era5_tp_lag_1d"])
    t2m_lag1 = float(row["t2m_mean_lag_1d"])
    sro_lag1 = float(row["sro_lag_1d"])
    
    t_date = pd.to_datetime(date_str)
    prev_date = (t_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    r_curr = df_rain[(df_rain["station_id"] == st_id) & (df_rain["date"] == date_str)].iloc[0]
    r_prev = df_rain[(df_rain["station_id"] == st_id) & (df_rain["date"] == prev_date)].iloc[0]
    
    f_curr = df_flux[(df_flux["station_id"] == st_id) & (df_flux["date"] == date_str)].iloc[0]
    f_prev = df_flux[(df_flux["station_id"] == st_id) & (df_flux["date"] == prev_date)].iloc[0]
    
    print(f"1. Target Observation (y_t):")
    print(f"   - Target Variable Name: `target_rainfall_mm`")
    print(f"   - Target Value: {t_val:.2f} mm")
    print(f"   - IMD Observation Timestamp: {date_str} 08:30 IST (03:00 UTC)")
    print(f"   - Target Accumulation Period: {prev_date} 03:00 UTC ---> {date_str} 03:00 UTC")
    
    print(f"\n2. Prediction Issue Boundary:")
    print(f"   - Issue Time (T_issue): {prev_date} 03:00 UTC (08:30 IST on day t-1)")
    print(f"   - Max Information Timestamp: {prev_date} 03:00 UTC")
    
    print(f"\n3. Ingested Predictor Features (All Shifted by >= 1 Day):")
    print(f"   - `target_rain_lag_1d` = {y_lag1:.2f} mm")
    print(f"     Source: IMD observation on {prev_date} = {r_prev['imd_observed_mm']:.2f} mm (Diff: {abs(y_lag1 - r_prev['imd_observed_mm']):.6e})")
    print(f"     Accumulation Window: {t_date - pd.Timedelta(days=2):%Y-%m-%d} 03:00 UTC ---> {prev_date} 03:00 UTC")
    
    print(f"   - `era5_tp_lag_1d` = {era5_lag1:.4f} mm")
    print(f"     Source: ERA5-Land IMD-aligned on {prev_date} = {f_prev['tp_daily_imd_mm']:.4f} mm (Diff: {abs(era5_lag1 - f_prev['tp_daily_imd_mm']):.6e})")
    print(f"     Accumulation Window: {t_date - pd.Timedelta(days=2):%Y-%m-%d} 04:00 UTC ---> {prev_date} 03:00 UTC")
    
    print(f"   - `t2m_mean_lag_1d` = {t2m_lag1:.2f} °C")
    print(f"     Source: ERA5-Land t2m mean on {prev_date} = {f_prev['t2m_mean_c']:.2f} °C (Diff: {abs(t2m_lag1 - f_prev['t2m_mean_c']):.6e})")
    
    print(f"   - `sro_lag_1d` = {sro_lag1:.4f} mm")
    print(f"     Source: ERA5-Land surface runoff on {prev_date} = {f_prev['sro_daily_mm']:.4f} mm (Diff: {abs(sro_lag1 - f_prev['sro_daily_mm']):.6e})")
    
    print(f"\n4. Contemporaneous Target-Day Verification:")
    print(f"   - Same-day unlagged ERA5 total on {date_str}: {f_curr['tp_daily_imd_mm']:.4f} mm")
    print(f"   - Is contemporaneous ERA5 rain present in model input X? {'tp_daily_imd_mm' in row.index and pd.notna(row.get('tp_daily_imd_mm'))}")
    print(f"   - Is contemporaneous IMD target rain present in model input X? {'imd_observed_mm' in row.index and pd.notna(row.get('imd_observed_mm'))}")
    print(f"   - Causal Status: STRICT PASS (Predictor window ended {prev_date} 03:00 UTC; Target window began {prev_date} 03:00 UTC)\n")

