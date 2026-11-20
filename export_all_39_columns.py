"""
Export All 30 Raw NetCDF Variables into 39-Column CSV files (Year-wise & Month-wise)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ERA5_Karnataka"
PREPROCESSED_DIR = BASE_DIR / "Preprocessed_Karnataka"
CSV_DIR = BASE_DIR / "CSVs"

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
MONTHS = list(range(1, 13))

accum_vars = ["tp", "e", "pev", "ro", "sro", "evabs", "evatc", "evavt"]
temp_vars = ["t2m", "d2m", "skt", "stl1", "stl2", "stl3"]

def process_and_export_all_columns():
    master_records = []

    for yr in YEARS:
        yr_dir = CSV_DIR / str(yr)
        yr_dir.mkdir(parents=True, exist_ok=True)
        yr_dfs = []

        print(f"\n================ Processing Year {yr} (Full 39 Columns) ================")

        for m in MONTHS:
            m_str = f"{m:02d}"
            nc_path = DATA_DIR / str(yr) / m_str / f"era5land_karnataka_{yr}_{m_str}.nc"

            if not nc_path.exists():
                print(f"  [Skip] Missing: {nc_path.name}")
                continue

            ds = xr.open_dataset(nc_path)
            ds_proc = ds.copy(deep=True)

            # 1. Temperature K -> deg C
            for v in temp_vars:
                if v in ds_proc:
                    if float(ds_proc[v].mean(skipna=True)) > 100.0:
                        ds_proc[v] = ds_proc[v] - 273.15
                        ds_proc[v].attrs["units"] = "deg_C"

            # 2. Precip, Evap, Runoff m -> mm
            for v in accum_vars + ["src"]:
                if v in ds_proc:
                    if v == "tp":
                        ds_proc[v] = xr.where(ds_proc[v] < 0, 0.0, ds_proc[v] * 1000.0)
                    else:
                        ds_proc[v] = ds_proc[v] * 1000.0
                    ds_proc[v].attrs["units"] = "mm"

            # 3. Pressure Pa -> hPa
            if "sp" in ds_proc:
                if float(ds_proc["sp"].mean(skipna=True)) > 50000.0:
                    ds_proc["sp"] = ds_proc["sp"] / 100.0
                    ds_proc["sp"].attrs["units"] = "hPa"

            # 4. Wind speed
            if "u10" in ds_proc and "v10" in ds_proc:
                ds_proc["wind_speed_ms"] = np.sqrt(ds_proc["u10"]**2 + ds_proc["v10"]**2)
                ds_proc["wind_speed_ms"].attrs["units"] = "m/s"

            # Save preprocessed NetCDF
            PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            prep_nc = PREPROCESSED_DIR / f"preprocessed_era5land_{yr}_{m_str}.nc"
            ds_proc.to_netcdf(prep_nc)

            # Daily aggregations preserving ALL variables
            daily_dict = {}
            for v in ds_proc.data_vars:
                if v in accum_vars:
                    daily_dict[f"{v}_daily_sum"] = ds_proc[v].resample(valid_time="1D").sum(dim="valid_time")
                elif any(t_kw in v for t_kw in ["t2m", "d2m", "skt", "stl"]):
                    daily_dict[f"{v}_daily_mean"] = ds_proc[v].resample(valid_time="1D").mean(dim="valid_time")
                    if v in ["t2m", "skt"]:
                        daily_dict[f"{v}_daily_max"] = ds_proc[v].resample(valid_time="1D").max(dim="valid_time")
                        daily_dict[f"{v}_daily_min"] = ds_proc[v].resample(valid_time="1D").min(dim="valid_time")
                else:
                    daily_dict[f"{v}_daily_mean"] = ds_proc[v].resample(valid_time="1D").mean(dim="valid_time")

            ds_daily = xr.Dataset(daily_dict)
            df_m = ds_daily.to_dataframe().reset_index()

            month_csv = yr_dir / f"karnataka_rainfall_{yr}_{m_str}.csv"
            df_m.to_csv(month_csv, index=False)
            print(f"  [OK] Exported Month {yr}-{m_str}: {month_csv.name} ({len(df_m):,} rows, {len(df_m.columns)} cols)")

            yr_dfs.append(df_m)
            ds.close()
            ds_proc.close()

        if yr_dfs:
            df_yr = pd.concat(yr_dfs, ignore_index=True)
            yr_csv = yr_dir / f"karnataka_rainfall_{yr}_yearly.csv"
            df_yr.to_csv(yr_csv, index=False)
            print(f"  [OK] Exported Yearly CSV {yr}: {yr_csv.name} ({len(df_yr):,} rows, {len(df_yr.columns)} cols)")
            master_records.append(df_yr)

    if master_records:
        df_master = pd.concat(master_records, ignore_index=True)
        master_csv = CSV_DIR / "karnataka_rainfall_2019_2025_master.csv"
        df_master.to_csv(master_csv, index=False)
        print("\n" + "=" * 70)
        print(f"[SUCCESS] Exported 7-Year Master CSV (2019-2025): {master_csv.name}")
        print(f"          Total Rows: {len(df_master):,}")
        print(f"          Total Columns: {len(df_master.columns)}")
        print(f"          Columns: {list(df_master.columns)}")
        print("=" * 70)



if __name__ == "__main__":
    process_and_export_all_columns()
