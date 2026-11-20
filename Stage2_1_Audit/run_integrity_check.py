import os
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

base = Path(r"d:\PROJECT\karnataka_rainfall_cds")
era_dir = base / "ERA5_Karnataka"

# 1. Check all 84 monthly NetCDF source files
nc_files = []
for yr in range(2019, 2026):
    for m in range(1, 13):
        p = era_dir / str(yr) / f"{m:02d}" / f"era5land_karnataka_{yr}_{m:02d}.nc"
        assert p.exists(), f"Missing raw NetCDF file: {p}"
        nc_files.append(p)
print(f"1. Raw NetCDF source files: {len(nc_files)} / 84 verified present.")

# 2. Check NetCDF dimensions and variables
nc_utc_p = base / "Stage2_Corrected_Processing/data/corrected_daily_era5_utc.nc"
nc_imd_p = base / "Stage2_Corrected_Processing/data/corrected_daily_era5_imd_aligned.nc"

ds_utc = xr.open_dataset(nc_utc_p)
ds_imd = xr.open_dataset(nc_imd_p)

utc_u = ds_utc.tp_corrected_utc.attrs.get("units")
imd_u = ds_imd.tp_corrected_imd_aligned.attrs.get("units")

print(f"2. NetCDF UTC dimensions: {dict(ds_utc.dims)}, coords: {list(ds_utc.coords.keys())}, units: {utc_u}")
print(f"   NetCDF IMD dimensions: {dict(ds_imd.dims)}, coords: {list(ds_imd.coords.keys())}, units: {imd_u}")

assert ds_utc.dims["time"] == 2557 and ds_utc.dims["latitude"] == 81 and ds_utc.dims["longitude"] == 51
assert ds_imd.dims["time"] == 2557 and ds_imd.dims["latitude"] == 81 and ds_imd.dims["longitude"] == 51

# 3. Verify Output CSV and NetCDF agreement on random test samples
df_rain = pd.read_csv(base / "Stage2_Corrected_Processing/data/corrected_daily_rainfall.csv")
df_rain["date"] = pd.to_datetime(df_rain["date"])

meta = pd.read_csv(base / "Stage2_Corrected_Processing/data/karnataka_252_stations_metadata.csv")

# Test interior station (e.g. station_id 10, 100, 200)
test_st_ids = [10, 100, 200]
print("\n3. Verifying CSV vs NetCDF consistency on test stations:")
for st_id in test_st_ids:
    st_meta = meta[meta["station_id"] == st_id].iloc[0]
    st_lat = st_meta["latitude"]
    st_lon = st_meta["longitude"]
    
    # Pick a specific peak monsoon date
    sample_date = "2022-07-15"
    csv_row = df_rain[(df_rain["station_id"] == st_id) & (df_rain["date"] == sample_date)].iloc[0]
    csv_utc_val = csv_row["corrected_era5_utc_mm"]
    csv_imd_val = csv_row["corrected_era5_imd_aligned_mm"]
    
    # NetCDF interpolated
    nc_utc_val = float(ds_utc["tp_corrected_utc"].sel(time=sample_date).interp(latitude=st_lat, longitude=st_lon).values)
    nc_imd_val = float(ds_imd["tp_corrected_imd_aligned"].sel(time=sample_date).interp(latitude=st_lat, longitude=st_lon).values)
    
    diff_u = abs(csv_utc_val - nc_utc_val)
    diff_i = abs(csv_imd_val - nc_imd_val)
    print(f"   Station {st_id} ({st_lat}, {st_lon}) on {sample_date}:")
    print(f"     UTC: CSV={csv_utc_val:.4f}, NetCDF={nc_utc_val:.4f}, diff={diff_u:.6f}")
    print(f"     IMD: CSV={csv_imd_val:.4f}, NetCDF={nc_imd_val:.4f}, diff={diff_i:.6f}")
    assert diff_u < 1e-3, f"UTC mismatch at station {st_id}: {diff_u}"
    assert diff_i < 1e-3, f"IMD mismatch at station {st_id}: {diff_i}"

ds_utc.close()
ds_imd.close()
print("\nAll reproducibility and integrity tests PASSED successfully!")

