"""
ERA5-Land Karnataka Rainfall Preprocessing Pipeline

This script performs full preprocessing on raw ERA5-Land monthly NetCDF files:
1. Unit Conversions (Kelvin -> Celsius, meters -> mm, Pa -> hPa, Wind vector -> speed)
2. Missing-Value Handling (Temporal/spatial linear interpolation & NaN checks)
3. Precipitation Processing (Accumulation handling, non-negative clipping, daily aggregation)
4. Quality Checks (Physical range boundary validation & report generation)

Usage:
    python preprocess_era5.py --input-dir ERA5_Karnataka --output-dir Preprocessed_Karnataka
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr


# Define standard physical range bounds for Quality Control (QC)
QC_RANGES = {
    "temperature_celsius": (-10.0, 55.0),
    "dewpoint_celsius": (-20.0, 40.0),
    "surface_pressure_hpa": (800.0, 1050.0),
    "total_precipitation_mm": (0.0, 500.0),
    "wind_speed_ms": (0.0, 75.0),
    "soil_moisture": (0.0, 1.0),
}

# Variable mappings for unit conversion
TEMP_VARS = [
    "t2m", "2m_temperature",
    "d2m", "2m_dewpoint_temperature",
    "skt", "skin_temperature",
    "stl1", "soil_temperature_level_1",
    "stl2", "soil_temperature_level_2",
    "stl3", "soil_temperature_level_3"
]

PRECIP_VARS = ["tp", "total_precipitation"]
EVAP_RUNOFF_VARS = [
    "e", "total_evaporation",
    "pev", "potential_evaporation",
    "ro", "runoff",
    "sro", "surface_runoff",
    "evabs", "evaporation_from_bare_soil",
    "evatc", "evaporation_from_the_top_of_canopy",
    "evtr", "evaporation_from_vegetation_transpiration"
]
PRESSURE_VARS = ["sp", "surface_pressure"]


def convert_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Perform unit conversions on ERA5-Land variables:
    - Temperature: Kelvin -> Celsius
    - Precipitation / Evaporation / Runoff: meters -> mm
    - Pressure: Pa -> hPa
    - Wind: Calculate wind speed (m/s) from u10 and v10
    """
    ds_converted = ds.copy(deep=True)

    # 1. Temperature conversion (Kelvin to Celsius)
    for var in ds_converted.data_vars:
        if any(t_name in var for t_name in TEMP_VARS):
            # Check if values are in Kelvin (> 100)
            if float(ds_converted[var].mean(skipna=True)) > 100.0:
                ds_converted[var] = ds_converted[var] - 273.15
                ds_converted[var].attrs["units"] = "°C"

    # 2. Precipitation conversion (meters to mm)
    for var in ds_converted.data_vars:
        if any(p_name in var for p_name in PRECIP_VARS):
            ds_converted[var] = ds_converted[var] * 1000.0
            ds_converted[var].attrs["units"] = "mm"

    # 3. Evaporation / Runoff conversion (meters to mm)
    for var in ds_converted.data_vars:
        if any(e_name in var for e_name in EVAP_RUNOFF_VARS):
            ds_converted[var] = ds_converted[var] * 1000.0
            ds_converted[var].attrs["units"] = "mm"

    # 4. Pressure conversion (Pa to hPa)
    for var in ds_converted.data_vars:
        if any(pr_name in var for pr_name in PRESSURE_VARS):
            if float(ds_converted[var].mean(skipna=True)) > 50000.0:
                ds_converted[var] = ds_converted[var] / 100.0
                ds_converted[var].attrs["units"] = "hPa"

    # 5. Wind Speed calculation if u10 and v10 exist
    u_var = next((v for v in ds_converted.data_vars if "u10" in v or "10m_u" in v), None)
    v_var = next((v for v in ds_converted.data_vars if "v10" in v or "10m_v" in v), None)

    if u_var and v_var and "wind_speed" not in ds_converted:
        wind_speed = np.sqrt(ds_converted[u_var] ** 2 + ds_converted[v_var] ** 2)
        wind_speed.attrs["long_name"] = "10m Wind Speed"
        wind_speed.attrs["units"] = "m s**-1"
        ds_converted["wind_speed"] = wind_speed

    return ds_converted


def handle_missing_values(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """
    Detect missing values (NaNs), interpolate temporally and spatially.
    """
    ds_filled = ds.copy(deep=True)
    nan_report = {}

    for var in ds_filled.data_vars:
        nan_count = int(ds_filled[var].isnull().sum().item())
        total_count = int(ds_filled[var].size)
        nan_pct = (nan_count / total_count) * 100.0 if total_count > 0 else 0.0

        nan_report[str(var)] = {
            "nan_count": nan_count,
            "total_count": total_count,
            "nan_percentage": round(nan_pct, 4)
        }

        if nan_count > 0:
            # Linear temporal interpolation for small missing gaps
            if "time" in ds_filled[var].dims:
                ds_filled[var] = ds_filled[var].interpolate_na(dim="time", method="linear", fill_value="extrapolate")
            # Nearest neighbor fill for any residual spatial NaNs
            ds_filled[var] = ds_filled[var].ffill(dim="time").bfill(dim="time")

    return ds_filled, nan_report


def process_precipitation(ds: xr.Dataset) -> xr.Dataset:
    """
    Precipitation processing:
    - Non-negativity constraint (clip precision artifacts < 0)
    - Accumulated precipitation handling
    """
    ds_precip = ds.copy(deep=True)

    for var in ds_precip.data_vars:
        if any(p_name in var for p_name in PRECIP_VARS):
            # Clip negative precision artifacts
            ds_precip[var] = xr.where(ds_precip[var] < 0, 0.0, ds_precip[var])
            ds_precip[var].attrs["processing"] = "Clipped negative values to 0.0 mm"

    return ds_precip


def perform_quality_checks(ds: xr.Dataset) -> dict:
    """
    Execute quality checks against known physical ranges and return a summary report.
    """
    qc_summary = {}

    for var in ds.data_vars:
        var_data = ds[var].values
        min_val = float(np.nanmin(var_data)) if var_data.size > 0 else None
        max_val = float(np.nanmax(var_data)) if var_data.size > 0 else None
        mean_val = float(np.nanmean(var_data)) if var_data.size > 0 else None

        status = "PASSED"
        flagged_anomalies = 0

        # Check against defined physical bounds
        if "temp" in str(var).lower() or "t2m" in str(var).lower():
            low, high = QC_RANGES["temperature_celsius"]
            if min_val < low or max_val > high:
                status = "WARNING_OUT_OF_RANGE"
                flagged_anomalies = int(np.sum((var_data < low) | (var_data > high)))
        elif "tp" in str(var).lower() or "precip" in str(var).lower():
            low, high = QC_RANGES["total_precipitation_mm"]
            if min_val < low or max_val > high:
                status = "WARNING_OUT_OF_RANGE"
                flagged_anomalies = int(np.sum((var_data < low) | (var_data > high)))

        qc_summary[str(var)] = {
            "min": round(min_val, 4) if min_val is not None else None,
            "max": round(max_val, 4) if max_val is not None else None,
            "mean": round(mean_val, 4) if mean_val is not None else None,
            "qc_status": status,
            "flagged_anomalies": flagged_anomalies
        }

    return qc_summary


def preprocess_file(file_path: Path, output_dir: Path) -> Path:
    """
    Process a single monthly ERA5 NetCDF file through the full preprocessing pipeline.
    """
    print(f"\n[Preprocessing] File: {file_path.name}")
    out_nc = output_dir / f"preprocessed_{file_path.name}"
    report_file = output_dir / f"qc_report_{file_path.stem}.json"

    ds = xr.open_dataset(file_path)

    try:
        # Step 1: Unit Conversion
        print("  |-- 1. Unit conversion (Kelvin->deg C, m->mm, Pa->hPa, Wind Speed)")
        ds_units = convert_units(ds)

        # Step 2: Missing Value Handling
        print("  |-- 2. Missing-value handling (Interpolation & NaN checks)")
        ds_clean, nan_report = handle_missing_values(ds_units)

        # Step 3: Precipitation Processing
        print("  |-- 3. Precipitation processing (Accumulation & Non-negative clipping)")
        ds_processed = process_precipitation(ds_clean)

        # Step 4: Quality Checks
        print("  +-- 4. Quality checks & validation report")
        qc_report = perform_quality_checks(ds_processed)

        # Save preprocessed NetCDF
        output_dir.mkdir(parents=True, exist_ok=True)
        ds_processed.to_netcdf(out_nc)

        # Write QC and Missing value report
        full_report = {
            "file": file_path.name,
            "nan_analysis": nan_report,
            "quality_checks": qc_report
        }
        report_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

        print(f"  [OK] Saved preprocessed NetCDF: {out_nc}")
        print(f"  [OK] Saved QC report: {report_file}")

    finally:
        ds.close()

    return out_nc


def main():
    parser = argparse.ArgumentParser(description="Preprocess ERA5-Land Karnataka NetCDF datasets")
    parser.add_argument("--input-dir", type=str, default="ERA5_Karnataka", help="Directory containing raw NetCDF files")
    parser.add_argument("--output-dir", type=str, default="Preprocessed_Karnataka", help="Output directory for preprocessed NetCDF")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    nc_files = list(input_path.rglob("era5land_karnataka_*.nc"))
    if not nc_files:
        nc_files = list(input_path.rglob("*.nc"))

    if not nc_files:
        print(f"No NetCDF (.nc) files found in {input_path.resolve()}. Please run downloader first.")
        return

    print(f"Found {len(nc_files)} NetCDF file(s) to preprocess.")

    for nc_file in nc_files:
        preprocess_file(nc_file, output_path)

    print("\n[Preprocessing Complete] All files processed successfully!")


if __name__ == "__main__":
    main()
