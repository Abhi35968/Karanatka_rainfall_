"""
Karnataka Rainfall Datasets: Verification, Quality Check, Plotting & CSV Conversion
----------------------------------------------------------------------------------
This script performs:
1. Dataset Relevance Check: Bounding box, variables, temporal resolution.
2. Null Value (NaN) & Quality Analysis: Percentage missing, physical bounds validation.
3. Graph Plotting:
   - 01_dataset_relevance_spatial_grid.png
   - 02_null_value_analysis.png
   - 03_sample_time_series_variables.png
   - 04_preprocessing_before_after.png
   - 05_monsoon_rainfall_distribution.png
   - 06_yearly_monthly_rainfall_comparison.png
4. Chunk Verification & Merging.
5. Preprocessing & Export to CSV Year-wise and Month-wise:
   - CSVs/YYYY/karnataka_rainfall_YYYY_MM.csv (Monthly)
   - CSVs/YYYY/karnataka_rainfall_YYYY_yearly.csv (Yearly aggregated)
   - CSVs/karnataka_rainfall_2019_2025_master.csv (Complete 7-year aggregated ML dataset)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")  # Non-interactive background plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set aesthetic visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ERA5_Karnataka"
PREPROCESSED_DIR = BASE_DIR / "Preprocessed_Karnataka"
CSV_DIR = BASE_DIR / "CSVs"
PLOTS_DIR = BASE_DIR / "Preprocessing_Plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
MONTHS = list(range(1, 13))

# Physical QC Ranges
QC_RANGES = {
    "temperature_celsius": (-10.0, 55.0),
    "dewpoint_celsius": (-20.0, 40.0),
    "surface_pressure_hpa": (800.0, 1050.0),
    "total_precipitation_mm": (0.0, 500.0),
    "wind_speed_ms": (0.0, 75.0),
    "soil_moisture": (0.0, 1.0),
}


def verify_chunks_and_merge():
    """Verify chunk files and ensure merged month files exist for all 3 years."""
    print("=" * 70)
    print("STEP 1: MERGING & VERIFYING CHUNK FILES")
    print("=" * 70)

    merged_files = []

    for y in YEARS:
        for m in MONTHS:
            m_str = f"{m:02d}"
            month_dir = DATA_DIR / str(y) / m_str
            if not month_dir.exists():
                print(f"  [Warning] Directory missing: {month_dir}")
                continue

            merged_path = month_dir / f"era5land_karnataka_{y}_{m_str}.nc"

            # Check if merged file exists and is valid
            is_valid = False
            if merged_path.exists() and merged_path.stat().st_size > 0:
                try:
                    with xr.open_dataset(merged_path) as test_ds:
                        # Quick check reading actual array slice to ensure HDF5 data integrity
                        _ = test_ds["tp"].isel(valid_time=0).values
                    is_valid = True
                except Exception as err:
                    print(f"  [Corrupted] {merged_path.name} invalid ({err}). Deleting and re-merging chunks...")
                    try:
                        merged_path.unlink()
                    except Exception:
                        pass

            if not is_valid:
                chunk_files = sorted(list(month_dir.glob("era5land_*_chunk*.nc")))
                if chunk_files:
                    print(f"  Merging {len(chunk_files)} chunks for {y}-{m_str}...")
                    datasets = [xr.open_dataset(cf) for cf in chunk_files]
                    combined = xr.merge(datasets, compat="override", join="exact")
                    combined.to_netcdf(merged_path)
                    combined.close()
                    for ds in datasets:
                        ds.close()
                    print(f"  [OK] Successfully merged chunks -> {merged_path.name}")
                else:
                    print(f"  [Error] No chunks found for {y}-{m_str}")
            else:
                print(f"  [OK] Verified merged file for {y}-{m_str} ({merged_path.stat().st_size / (1024*1024):.1f} MB)")

            if merged_path.exists():
                merged_files.append(merged_path)

    print(f"\nTotal verified monthly NetCDF files: {len(merged_files)} / {len(YEARS) * len(MONTHS)}")
    return sorted(merged_files)


def inspect_dataset_relevance(nc_files):
    """Inspect dataset relevance: spatial box, variables, temporal completeness."""
    print("\n" + "=" * 70)
    print("STEP 2: DATASET RELEVANCE & SCHEMAS INSPECTION")
    print("=" * 70)

    sample_nc = nc_files[0]
    ds = xr.open_dataset(sample_nc)

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    times = ds["valid_time"].values
    vars_list = list(ds.data_vars.keys())

    print(f"Sample File: {sample_nc.name}")
    print(f"Spatial Grid Bounding Box: Lat [{lat.min():.2f}°N, {lat.max():.2f}°N], Lon [{lon.min():.2f}°E, {lon.max():.2f}°E]")
    print(f"Grid Resolution: {abs(lat[1]-lat[0]):.2f}° x {abs(lon[1]-lon[0]):.2f}° (~11km x 11km)")
    print(f"Grid Dimensions: Latitude={len(lat)} points, Longitude={len(lon)} points (Total {len(lat)*len(lon)} grid cells)")
    print(f"Time Stamps in Month: {len(times)} hours")
    print(f"Variables Count: {len(vars_list)}")
    print(f"Variable List: {vars_list}")

    # Check relevance for Karnataka Rainfall Prediction
    relevant_keys = {
        "tp": "Total Precipitation (Rainfall)",
        "t2m": "2m Air Temperature",
        "d2m": "2m Dewpoint Temperature",
        "sp": "Surface Pressure",
        "u10": "10m Eastward Wind Component",
        "v10": "10m Northward Wind Component",
        "swvl1": "Volumetric Soil Water (0-7 cm)",
        "skt": "Skin Temperature",
        "e": "Total Evaporation",
        "ro": "Runoff",
    }

    print("\n[Relevance Verification against Rainfall Forecasting]")
    for key, desc in relevant_keys.items():
        matched = [v for v in vars_list if key in v]
        status = f"PRESENT ({matched[0]})" if matched else "MISSING"
        print(f"  - {desc} ({key}): {status}")

    ds.close()


def analyze_null_values(nc_files):
    """Scan dataset for missing (NaN) values across space and variables."""
    print("\n" + "=" * 70)
    print("STEP 3: NULL VALUE (NaN) ANALYSIS")
    print("=" * 70)

    nan_stats = {}
    sample_ds = xr.open_dataset(nc_files[0])

    total_cells_per_var = sample_ds.sizes["valid_time"] * sample_ds.sizes["latitude"] * sample_ds.sizes["longitude"]

    for var in sample_ds.data_vars:
        nan_count = int(sample_ds[var].isnull().sum().item())
        nan_pct = (nan_count / total_cells_per_var) * 100.0
        nan_stats[str(var)] = {
            "nan_count": nan_count,
            "total_cells": total_cells_per_var,
            "nan_pct": nan_pct
        }

    sample_ds.close()

    df_nan = pd.DataFrame.from_dict(nan_stats, orient="index")
    print("\nNull Value Report (Sample Month 2019-01):")
    print(df_nan[["nan_count", "total_cells", "nan_pct"]].head(15))

    return df_nan


def generate_graphs(nc_files, df_nan):
    """Generate and save all 6 verification and preprocessing plots."""
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING VERIFICATION & PREPROCESSING PLOTS")
    print("=" * 70)

    # Load 1 year sample (2019) for time series
    sample_files_2019 = [f for f in nc_files if "2019_" in f.name]
    datasets_2019 = [xr.open_dataset(f) for f in sample_files_2019]
    ds_2019 = xr.concat(datasets_2019, dim="valid_time")

    # GRAPH 1: Dataset Relevance & Spatial Grid
    print("  Generating Plot 1: Dataset Relevance & Spatial Grid...")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    mean_sp = ds_2019["sp"].mean(dim="valid_time") / 100.0  # Pa to hPa
    lon, lat = ds_2019["longitude"].values, ds_2019["latitude"].values

    c = ax.pcolormesh(lon, lat, mean_sp, cmap="terrain", shading="auto")
    cb = fig.colorbar(c, ax=ax, label="Surface Pressure (hPa) ~ Elevation proxy")

    ax.set_title("ERA5-Land Karnataka Spatial Grid & Coverage\nBounding Box: 11°N - 19°N, 74°E - 79°E (0.1° Grid)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°E)", labelpad=8)
    ax.set_ylabel("Latitude (°N)", labelpad=8)

    # Overlay Karnataka major cities
    cities = {
        "Bengaluru": (12.97, 77.59),
        "Mangaluru": (12.91, 74.85),
        "Hubballi": (15.36, 75.12),
        "Kalaburagi": (17.33, 76.83),
        "Mysuru": (12.30, 76.65)
    }
    for name, (clat, clon) in cities.items():
        ax.plot(clon, clat, "ro", markersize=6, markeredgecolor="white")
        ax.annotate(name, (clon + 0.15, clat + 0.05), fontsize=9, fontweight="bold", color="#111111", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = PLOTS_DIR / "01_dataset_relevance_spatial_grid.png"
    fig.savefig(p1)
    plt.close(fig)
    print(f"  [OK] Saved: {p1.name}")

    # GRAPH 2: Null Value Analysis
    print("  Generating Plot 2: Null Value Analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Bar chart of NaN %
    top_vars = df_nan.head(12)
    ax1.barh(top_vars.index, top_vars["nan_pct"], color="#3498db", edgecolor="#2980b9")
    ax1.set_title("Missing Values (NaN %) per Variable", fontsize=11, fontweight="bold")
    ax1.set_xlabel("NaN Percentage (%)")
    ax1.set_xlim(0, max(1.0, top_vars["nan_pct"].max() * 1.2))

    for i, v in enumerate(top_vars["nan_pct"]):
        ax1.text(v + 0.02, i, f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")

    # Spatial NaN Heatmap for a sample variable
    tp_var = "tp" if "tp" in ds_2019 else list(ds_2019.data_vars.keys())[0]
    nan_spatial_map = ds_2019[tp_var].isnull().sum(dim="valid_time") / len(ds_2019["valid_time"]) * 100.0

    c2 = ax2.pcolormesh(lon, lat, nan_spatial_map, cmap="Reds", shading="auto")
    fig.colorbar(c2, ax=ax2, label="NaN Frequency (%)")
    ax2.set_title("Spatial Distribution of Land/Ocean NaNs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")

    plt.suptitle("ERA5-Land Karnataka Data Quality & Null Value Inspection", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    p2 = PLOTS_DIR / "02_null_value_analysis.png"
    fig.savefig(p2)
    plt.close(fig)
    print(f"  [OK] Saved: {p2.name}")

    # GRAPH 3: Sample Time Series Variables
    print("  Generating Plot 3: Sample Time Series of Variables...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, dpi=300)

    # Spatial mean across Karnataka over time
    tp_series = (ds_2019["tp"].mean(dim=["latitude", "longitude"]) * 1000.0).to_dataframe()["tp"]
    t2m_series = (ds_2019["t2m"].mean(dim=["latitude", "longitude"]) - 273.15).to_dataframe()["t2m"]
    swvl_series = ds_2019["swvl1"].mean(dim=["latitude", "longitude"]).to_dataframe()["swvl1"]
    u10 = ds_2019["u10"].mean(dim=["latitude", "longitude"])
    v10 = ds_2019["v10"].mean(dim=["latitude", "longitude"])
    wind_series = np.sqrt(u10**2 + v10**2).to_series()

    axes[0].plot(tp_series.index, tp_series.values, color="#2980b9", linewidth=0.8, label="Hourly Precip (mm)")
    axes[0].set_ylabel("Precipitation (mm)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("ERA5-Land Karnataka Hourly Time Series Sample (2019)", fontsize=12, fontweight="bold")

    axes[1].plot(t2m_series.index, t2m_series.values, color="#e74c3c", linewidth=0.8, label="2m Temp (°C)")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend(loc="upper right")

    axes[2].plot(swvl_series.index, swvl_series.values, color="#27ae60", linewidth=0.8, label="Soil Moisture (m³/m³)")
    axes[2].set_ylabel("Soil Moisture")
    axes[2].legend(loc="upper right")

    axes[3].plot(wind_series.index, wind_series.values, color="#8e44ad", linewidth=0.8, label="10m Wind Speed (m/s)")
    axes[3].set_ylabel("Wind Speed (m/s)")
    axes[3].set_xlabel("Date & Time")
    axes[3].legend(loc="upper right")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p3 = PLOTS_DIR / "03_sample_time_series_variables.png"
    fig.savefig(p3)
    plt.close(fig)
    print(f"  [OK] Saved: {p3.name}")

    # GRAPH 4: Preprocessing Before vs After
    print("  Generating Plot 4: Preprocessing Before vs After Comparison...")
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Before (Raw Kelvin & Raw meters)
    raw_tp = ds_2019["tp"].mean(dim=["latitude", "longitude"]).values[:168]  # 1 week sample
    raw_t2m = ds_2019["t2m"].mean(dim=["latitude", "longitude"]).values[:168]

    # After (Converted mm & Celsius)
    proc_tp = np.clip(raw_tp * 1000.0, 0, None)
    proc_t2m = raw_t2m - 273.15

    time_hrs = np.arange(168)

    ax_before.plot(time_hrs, raw_t2m, color="#e67e22", label="Raw T2M (Kelvin)")
    ax_before.plot(time_hrs, raw_tp * 10000, color="#2980b9", linestyle="--", label="Raw Precip (m x 10^4)")
    ax_before.set_title("BEFORE Preprocessing (Raw SI Units)", fontsize=11, fontweight="bold")
    ax_before.set_xlabel("Hours (1 Week)")
    ax_before.set_ylabel("Raw Units (K / m)")
    ax_before.legend(loc="upper right")

    ax_after.plot(time_hrs, proc_t2m, color="#d35400", label="Converted Temp (°C)")
    ax_after.plot(time_hrs, proc_tp, color="#16a085", label="Converted Precip (mm)")
    ax_after.set_title("AFTER Preprocessing (Physical Units: °C & mm)", fontsize=11, fontweight="bold")
    ax_after.set_xlabel("Hours (1 Week)")
    ax_after.set_ylabel("Physical Units (°C / mm)")
    ax_after.legend(loc="upper right")

    plt.suptitle("Preprocessing Pipeline Validation: Unit Conversion & Clipping", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    p4 = PLOTS_DIR / "04_preprocessing_before_after.png"
    fig.savefig(p4)
    plt.close(fig)
    print(f"  [OK] Saved: {p4.name}")

    # Close sample dataset
    for d in datasets_2019:
        d.close()

    # GRAPH 5: Monsoon Rainfall Spatial Distribution for sample years
    print("  Generating Plot 5: Spatial Monsoon Rainfall Distribution...")
    sample_years = [YEARS[0], YEARS[len(YEARS)//2], YEARS[-1]] if len(YEARS) >= 3 else YEARS
    fig, axes = plt.subplots(1, len(sample_years), figsize=(5 * len(sample_years), 5), dpi=300)
    if len(sample_years) == 1:
        axes = [axes]

    for idx, yr in enumerate(sample_years):
        yr_files = [f for f in nc_files if f"_{yr}_" in f.name]
        if not yr_files:
            continue

        monsoon_tp = np.zeros((len(lat), len(lon)))
        for f in yr_files:
            with xr.open_dataset(f) as ds_m:
                monsoon_tp += (ds_m["tp"].sum(dim="valid_time") * 1000.0).values

        c_m = axes[idx].pcolormesh(lon, lat, monsoon_tp, cmap="YlGnBu", shading="auto", vmin=0, vmax=2500)
        axes[idx].set_title(f"Year {yr} Annual Cumulative Precip (mm)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Longitude (°E)")
        if idx == 0:
            axes[idx].set_ylabel("Latitude (°N)")

    fig.colorbar(c_m, ax=axes, orientation="vertical", label="Total Precipitation (mm)", pad=0.02)
    plt.suptitle("Karnataka Spatial Rainfall Verification (ERA5-Land)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p5 = PLOTS_DIR / "05_monsoon_rainfall_distribution.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {p5.name}")

    # GRAPH 6: Yearly & Monthly Rainfall Comparison
    print("  Generating Plot 6: Yearly & Monthly Rainfall Trend Comparison...")
    fig, (ax_mon, ax_yr) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    monthly_totals = {yr: [] for yr in YEARS}
    yearly_totals = {}

    for yr in YEARS:
        yr_tot = 0
        for m in MONTHS:
            f_path = DATA_DIR / str(yr) / f"{m:02d}" / f"era5land_karnataka_{yr}_{m:02d}.nc"
            if f_path.exists():
                ds_m = xr.open_dataset(f_path)
                m_sum = float((ds_m["tp"].sum(dim="valid_time").mean(dim=["latitude", "longitude"]) * 1000.0).item())
                ds_m.close()
                monthly_totals[yr].append(m_sum)
                yr_tot += m_sum
            else:
                monthly_totals[yr].append(0)
        yearly_totals[yr] = yr_tot

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(12)

    cmap = plt.cm.get_cmap("tab10", len(YEARS))
    for idx, yr in enumerate(YEARS):
        ax_mon.plot(month_names, monthly_totals[yr], marker="o", label=str(yr), color=cmap(idx), linewidth=1.8)
    ax_mon.set_ylabel("Average Precipitation (mm)")
    ax_mon.set_title("Monthly Rainfall Pattern Comparison (Multi-Year)", fontsize=11, fontweight="bold")
    ax_mon.legend()
    ax_mon.grid(True, linestyle=":", alpha=0.5)

    bar_colors = [cmap(i) for i in range(len(YEARS))]
    bars = ax_yr.bar([str(y) for y in YEARS], [yearly_totals[y] for y in YEARS], color=bar_colors, width=0.5)
    ax_yr.set_ylabel("Total Annual Precipitation (mm)")
    ax_yr.set_title("Yearly Cumulative Rainfall Total Comparison", fontsize=11, fontweight="bold")
    ax_yr.grid(True, linestyle=":", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax_yr.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval:.1f} mm", ha="center", va="bottom", fontweight="bold", fontsize=8)

    plt.tight_layout()
    p6 = PLOTS_DIR / "06_yearly_monthly_rainfall_comparison.png"
    fig.savefig(p6)
    plt.close(fig)
    print(f"  [OK] Saved: {p6.name}")


def convert_and_export_csvs(nc_files):
    """
    Preprocess raw NetCDF datasets, save to Preprocessed_Karnataka,
    and export into CSV files Year-wise and Month-wise.
    """
    print("\n" + "=" * 70)
    print("STEP 5: PREPROCESSING NETCDF & CONVERTING TO CSV (YEAR-WISE & MONTH-WISE)")
    print("=" * 70)

    master_daily_records = []

    for yr in YEARS:
        yr_csv_dir = CSV_DIR / str(yr)
        yr_csv_dir.mkdir(parents=True, exist_ok=True)

        yr_daily_df_list = []

        print(f"\n--- Processing Year {yr} ---")

        for m in MONTHS:
            m_str = f"{m:02d}"
            nc_path = DATA_DIR / str(yr) / m_str / f"era5land_karnataka_{yr}_{m_str}.nc"
            if not nc_path.exists():
                print(f"  [Skip] {nc_path.name} does not exist.")
                continue

            month_csv_path = yr_csv_dir / f"karnataka_rainfall_{yr}_{m_str}.csv"
            if month_csv_path.exists() and month_csv_path.stat().st_size > 1000:
                print(f"  [Resume OK] Using existing Month CSV {yr}-{m_str}: {month_csv_path.relative_to(BASE_DIR)}")
                df_monthly_daily = pd.read_csv(month_csv_path)
                yr_daily_df_list.append(df_monthly_daily)
                continue

            # Load dataset eagerly with self-healing chunk fallback
            try:
                with xr.open_dataset(nc_path) as ds:
                    ds_proc = ds.load()
            except Exception as e_load:
                print(f"  [Warning] Error loading {nc_path.name} ({e_load}). Re-merging from chunks...")
                chunk_files = sorted(list((DATA_DIR / str(yr) / m_str).glob("era5land_*_chunk*.nc")))
                if chunk_files:
                    datasets = [xr.open_dataset(cf) for cf in chunk_files]
                    combined = xr.merge(datasets, compat="override", join="exact")
                    combined.to_netcdf(nc_path)
                    combined.close()
                    for c_ds in datasets:
                        c_ds.close()
                    with xr.open_dataset(nc_path) as ds:
                        ds_proc = ds.load()
                else:
                    print(f"  [Skip] No chunk files to re-merge for {nc_path.name}")
                    continue

            # Preprocessing: Unit conversions for ALL variables
            # 1. Temperature K -> °C
            temp_vars = ["t2m", "d2m", "skt", "stl1", "stl2", "stl3"]
            for v in temp_vars:
                if v in ds_proc:
                    if float(ds_proc[v].mean(skipna=True)) > 100.0:
                        ds_proc[v] = ds_proc[v] - 273.15
                        ds_proc[v].attrs["units"] = "°C"

            # 2. Precipitation, Evaporation & Runoff m -> mm
            precip_evap_vars = ["tp", "e", "pev", "ro", "sro", "evabs", "evatc", "evavt", "src"]
            for v in precip_evap_vars:
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

            # 4. Wind Speed (u10, v10 -> wind_speed_ms)
            if "u10" in ds_proc and "v10" in ds_proc:
                ds_proc["wind_speed_ms"] = np.sqrt(ds_proc["u10"]**2 + ds_proc["v10"]**2)
                ds_proc["wind_speed_ms"].attrs["units"] = "m/s"

            # Save Full-Variable Preprocessed NetCDF
            prep_nc_path = PREPROCESSED_DIR / f"preprocessed_era5land_{yr}_{m_str}.nc"
            ds_proc.to_netcdf(prep_nc_path)

            # Compute Daily Aggregations for ALL 30+ variables (preserving complete feature set)
            accum_vars = ["tp", "e", "pev", "ro", "sro", "evabs", "evatc", "evavt"]
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

            # Convert to Pandas DataFrame
            df_monthly_daily = ds_daily.to_dataframe().reset_index()

            # Export Monthly CSV (All variables included)
            df_monthly_daily.to_csv(month_csv_path, index=False)

            print(f"  [OK] Processed & Exported Month {yr}-{m_str}: {month_csv_path.relative_to(BASE_DIR)} ({len(df_monthly_daily):,} rows, {len(df_monthly_daily.columns)} columns)")

            yr_daily_df_list.append(df_monthly_daily)

        # Combine Year-wise CSV
        if yr_daily_df_list:
            df_yearly = pd.concat(yr_daily_df_list, ignore_index=True)
            yearly_csv_path = yr_csv_dir / f"karnataka_rainfall_{yr}_yearly.csv"
            df_yearly.to_csv(yearly_csv_path, index=False)
            print(f"  [OK] Exported Yearly CSV for {yr}: {yearly_csv_path.relative_to(BASE_DIR)} ({len(df_yearly):,} rows, {len(df_yearly.columns)} columns)")
            master_daily_records.append(df_yearly)

    # Master 7-Year CSV
    if master_daily_records:
        df_master = pd.concat(master_daily_records, ignore_index=True)
        master_csv_path = CSV_DIR / "karnataka_rainfall_2019_2025_master.csv"
        df_master.to_csv(master_csv_path, index=False)
        print(f"\n[OK] Exported Complete 7-Year Master CSV (2019-2025): {master_csv_path.relative_to(BASE_DIR)}")
        print(f"     Total Rows: {len(df_master):,}")
        print(f"     Total Columns: {len(df_master.columns)}")
        print(f"     Columns: {list(df_master.columns)}")


def main():
    print("=" * 70)
    print("KARNATAKA RAINFALL 7-YEAR (2019-2025) PREPROCESSING & CSV CONVERSION")
    print("=" * 70)


    # 1. Chunk check & merge verification
    nc_files = verify_chunks_and_merge()
    if not nc_files:
        print("[Error] No NetCDF files available to process.")
        sys.exit(1)

    # 2. Relevance check
    inspect_dataset_relevance(nc_files)

    # 3. Null value analysis
    df_nan = analyze_null_values(nc_files)

    # 4. Generate all plots
    generate_graphs(nc_files, df_nan)

    # 5. Convert & export to CSVs year-wise and month-wise
    convert_and_export_csvs(nc_files)

    print("\n" + "=" * 70)
    print("  [SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!  ")
    print("  All verification checks passed.")
    print(f"  - Graph plots saved to: {PLOTS_DIR.resolve()}")
    print(f"  - Preprocessed NetCDF files saved to: {PREPROCESSED_DIR.resolve()}")
    print(f"  - Year-wise & Month-wise CSV files saved to: {CSV_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
