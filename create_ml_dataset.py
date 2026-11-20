"""
Create ML-Ready Dataset from Preprocessed Data (Scalable for Multi-Year Data)
-----------------------------------------------------------------------------
This script builds a clean, feature-engineered ML dataset from preprocessed data,
preserving ALL 30+ hydro-meteorological variables.

Optimized for scalability: Saves as both compressed Parquet (.parquet) and CSV (.csv).

Usage:
    python create_ml_dataset.py --output-file ML_Ready_Karnataka_Rainfall.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering across all variables:
    - Diurnal temperature range
    - Dewpoint depression
    - Cyclical time features
    - Spatial Lags & Rolling window precipitation accumulators
    - ML Targets (Next-day rainfall depth & classification category)
    """
    print("\n[Feature Engineering] Computing derived metrics, spatial lags, and targets...")

    time_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), "valid_time")
    lat_col = next((c for c in df.columns if "lat" in c.lower()), "latitude")
    lon_col = next((c for c in df.columns if "lon" in c.lower()), "longitude")
    precip_col = next((c for c in df.columns if "tp" in c.lower() or "precip" in c.lower()), "tp_daily_sum")
    temp_mean_col = next((c for c in df.columns if "t2m_daily_mean" in c.lower() or "temp_mean" in c.lower()), None)
    temp_max_col = next((c for c in df.columns if "t2m_daily_max" in c.lower() or "temp_max" in c.lower()), None)
    temp_min_col = next((c for c in df.columns if "t2m_daily_min" in c.lower() or "temp_min" in c.lower()), None)
    dew_col = next((c for c in df.columns if "d2m" in c.lower() or "dew" in c.lower()), None)

    # Sort deterministically by grid coordinates and time
    df = df.sort_values(by=[lat_col, lon_col, time_col]).reset_index(drop=True)

    # Downcast float64 -> float32 for memory efficiency
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # 1. Temperature Diurnal Range
    if temp_max_col and temp_min_col:
        df["temp_range_c"] = df[temp_max_col] - df[temp_min_col]

    # 2. Dewpoint Depression (T - Td)
    if temp_mean_col and dew_col:
        df["dewpoint_depression_c"] = df[temp_mean_col] - df[dew_col]

    # 3. Time & Seasonal Cyclical Features
    df[time_col] = pd.to_datetime(df[time_col], format="mixed")
    df["year"] = df[time_col].dt.year.astype("int16")
    df["month"] = df[time_col].dt.month.astype("int8")
    df["day_of_year"] = df[time_col].dt.dayofyear.astype("int16")

    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype("float32")
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype("float32")

    # Group by spatial grid cell for lag & rolling features
    grouped = df.groupby([lat_col, lon_col])

    # 4. Temporal Lag Features (Precipitation & Temperature history)
    df["precip_lag_1d"] = grouped[precip_col].shift(1).astype("float32")
    df["precip_lag_2d"] = grouped[precip_col].shift(2).astype("float32")
    df["precip_lag_3d"] = grouped[precip_col].shift(3).astype("float32")

    if temp_mean_col:
        df["temp_lag_1d"] = grouped[temp_mean_col].shift(1).astype("float32")

    # 5. Rolling Window Precipitation Accumulators
    df["precip_roll_3d_sum"] = grouped[precip_col].transform(lambda x: x.rolling(3, min_periods=1).sum()).astype("float32")
    df["precip_roll_7d_sum"] = grouped[precip_col].transform(lambda x: x.rolling(7, min_periods=1).sum()).astype("float32")

    # 6. ML Target 1: Next-Day Rainfall Depth (Regression Target)
    df["target_next_day_precip_mm"] = grouped[precip_col].shift(-1).astype("float32")

    val = df["target_next_day_precip_mm"].values
    conds = [val < 0.1, val < 10.0, val < 35.0]
    choices = [0, 1, 2]
    df["target_rain_class"] = np.select(conds, choices, default=3).astype("float32")

    # Clean boundary NaNs resulting from shifts
    df_clean = df.dropna(subset=["target_next_day_precip_mm", "precip_lag_1d", "precip_lag_3d"]).reset_index(drop=True)

    print(f"[OK] Feature engineering complete!")
    print(f"     Engineered {df_clean.shape[1]} total features across {df_clean.shape[0]:,} grid-day records.")
    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Build ML-Ready dataset from preprocessed data")
    parser.add_argument("--input-csv", type=str, default="CSVs/karnataka_rainfall_2019_2025_master.csv", help="Input master CSV file")
    parser.add_argument("--output-file", type=str, default="ML_Ready_Karnataka_Rainfall.csv", help="Output ML-ready CSV/Parquet file")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_file = Path(args.output_file)

    if not input_path.exists():
        input_path = Path("CSVs/karnataka_rainfall_2019_2025_master.csv")
    if not input_path.exists():
        input_path = Path("CSVs/karnataka_rainfall_3year_master.csv")

    if not input_path.exists():
        print(f"[Error] Input dataset missing at {input_path.resolve()}. Please run process_verify_and_plot.py first.")
        return


    print(f"Loading preprocessed dataset from: {input_path.name}...")
    df_input = pd.read_csv(input_path)
    print(f"Loaded {len(df_input):,} rows across {len(df_input.columns)} columns.")

    ml_df = engineer_features(df_input)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed Parquet (ultra-fast for ML model training)
    parquet_path = output_file.with_suffix(".parquet")
    try:
        ml_df.to_parquet(parquet_path, index=False, compression="snappy")
        print(f"\n[OK] Saved compressed Parquet: {parquet_path.resolve()} ({parquet_path.stat().st_size / (1024*1024):.1f} MB)")
    except Exception as e_pq:
        print(f"\n[Warning] Could not save Parquet ({e_pq}). Skipping Parquet export.")

    # Save as CSV in chunked mode for memory safety
    csv_path = output_file.with_suffix(".csv")
    ml_df.to_csv(csv_path, index=False, float_format="%.4f", chunksize=100000)
    print(f"[OK] Saved CSV: {csv_path.resolve()} ({csv_path.stat().st_size / (1024*1024):.1f} MB)")

    print(f"\n" + "=" * 70)
    print(f"[ML DATASET CREATED SUCCESSFULLY]")
    print(f"Total Rows: {len(ml_df):,}")
    print(f"Total Columns: {len(ml_df.columns)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
