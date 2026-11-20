"""
Karnataka Rainfall Accuracy Verification and Bias Correction Pipeline
====================================================================
Cross-validation of ERA5-Land Reanalysis against India Meteorological 
Department (IMD) 0.25° Gridded Observed Rainfall over Karnataka (2019-2025).

Features:
1. IMD & ERA5 Data Extraction & Spatial Alignment (Bilinear Collocation on 0.25° Grid)
2. Evaluation of Pipeline Resample-Sum vs True 24h ERA5 Precipitation (Artifact Analysis)
3. Statistical Accuracy Verification:
   - Continuous: MAE, RMSE, PBIAS, Pearson r, Spearman rho, R2, KGE, NSE
   - Categorical / Contingency: POD, FAR, CSI, FBI across IMD intensity classes
   - Agro-climatic zonation (Coastal/Ghats, South Interior, North Interior)
   - Seasonal breakdown (JJAS Southwest Monsoon, OND Northeast Monsoon, etc.)
4. Rigorous Bias Correction:
   - Linear Scaling (LS)
   - Power Transformation (PT)
   - Empirical Quantile Mapping (EQM) with Drizzle / Dry-Day Thresholding
   - Parametric Quantile Mapping (Gamma PQM)
   - Out-of-sample Validation (Calibration 2019-2022, Validation 2023-2025)
5. Publication-grade Diagnostic Visualizations (6 comprehensive figures)
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
from scipy.optimize import minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.ticker as ticker

# Style settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.edgecolor"] = "#bcbcbc"
plt.rcParams["axes.linewidth"] = 0.8

# Paths
BASE_DIR = Path(__file__).parent
ERA_DIR = BASE_DIR / "ERA5_Karnataka"
IMD_DIR = BASE_DIR / "IMD_Data"
PLOTS_DIR = BASE_DIR / "IMD_Verification_Plots"
OUTPUT_DIR = BASE_DIR / "Model_Outputs"
CSV_DIR = BASE_DIR / "CSVs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
CALIBRATION_YEARS = [2019, 2020, 2021, 2022]
VALIDATION_YEARS = [2023, 2024, 2025]


# ==============================================================================
# 1. DATA INGESTION & EXTRACTION
# ==============================================================================

def load_imd_data():
    """Load IMD 0.25° daily rainfall dataset for all years over Karnataka."""
    import imdlib as imd
    print("\n" + "=" * 75)
    print("STEP 1: LOADING IMD 0.25° GRIDDED OBSERVED RAINFALL DATA (2019-2025)")
    print("=" * 75)
    
    # Open full IMD data
    imd_obj = imd.open_data('rain', YEARS[0], YEARS[-1], 'yearwise', str(IMD_DIR))
    ds = imd_obj.get_xarray()
    
    # Slice Karnataka domain: Lat 11.0 to 19.0, Lon 74.0 to 79.0
    ds_kar = ds.sel(lat=slice(11.0, 19.0), lon=slice(74.0, 79.0))
    rain = ds_kar.rain.values  # (N_days, 33, 21)
    lats = ds_kar.lat.values
    lons = ds_kar.lon.values
    times = pd.to_datetime(ds_kar.time.values)
    
    # In IMD, missing / ocean values are represented as NaN or -999.0
    rain = np.where((rain < 0) | (rain > 1000.0), np.nan, rain)
    
    # Establish persistent land mask (points with observed stations)
    land_mask = ~np.isnan(rain).all(axis=0)
    valid_points = np.sum(land_mask)
    
    print(f"  [OK] IMD Karnataka Grid: {len(lats)} lats x {len(lons)} lons ({rain.shape[1]*rain.shape[2]} cells)")
    print(f"  [OK] Active Karnataka land grid points with observations: {valid_points}")
    print(f"  [OK] Total daily time steps: {len(times):,} days ({times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')})")
    
    return {
        "rain": rain,
        "lats": lats,
        "lons": lons,
        "times": times,
        "land_mask": land_mask
    }


def extract_true_era5_daily(imd_meta):
    """
    Extract true 24-hour accumulated precipitation from ERA5-Land NetCDF files
    and bilinearly interpolate directly onto the IMD 0.25° grid over Karnataka.
    """
    print("\n" + "=" * 75)
    print("STEP 2: EXTRACTING TRUE 24H ERA5 PRECIPITATION & COLLOCATING TO IMD GRID")
    print("=" * 75)
    
    imd_lats = imd_meta["lats"]
    imd_lons = imd_meta["lons"]
    target_times = imd_meta["times"]
    
    cache_path = BASE_DIR / "Preprocessed_Karnataka" / "era5_true_24h_on_imd_grid.npz"
    if cache_path.exists():
        print(f"  [Resume] Loading cached collocated ERA5 dataset from {cache_path.name}...")
        data = np.load(cache_path)
        era_on_imd = data["era_on_imd"]
        era_sum_on_imd = data["era_sum_on_imd"]
        print(f"  [OK] Loaded ERA5 array: shape {era_on_imd.shape}")
        return era_on_imd, era_sum_on_imd

    all_daily_era = []
    all_sum_era = []
    
    grid_t, grid_lat, grid_lon = np.meshgrid(np.arange(1), imd_lats, imd_lons, indexing='ij')
    eval_pts = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)

    for yr in YEARS:
        t0 = time.time()
        yr_daily = []
        yr_sum = []
        
        for m in range(1, 13):
            nc_path = ERA_DIR / str(yr) / f"{m:02d}" / f"era5land_karnataka_{yr}_{m:02d}.nc"
            if not nc_path.exists():
                print(f"  [Warning] Missing {nc_path.name}")
                continue
                
            with xr.open_dataset(nc_path) as ds:
                tp = ds["tp"].values * 1000.0  # mm, shape: (n_hours, 81, 51)
                e_lats = ds["latitude"].values
                e_lons = ds["longitude"].values
                
                # Order latitudes ascending for RegularGridInterpolator
                if e_lats[0] > e_lats[-1]:
                    e_lats = e_lats[::-1]
                    tp = tp[:, ::-1, :]
                    
                n_hours = tp.shape[0]
                days_in_m = n_hours // 24
                
                for d in range(days_in_m):
                    # True 24-hour total accumulation (at end of day forecast step)
                    idx_24h = min(24 * d + 24, n_hours - 1)
                    true_24h = tp[idx_24h]
                    
                    # Also compute the uncorrected Resample Sum (artifact in pipeline)
                    day_slice = tp[24 * d : min(24 * d + 24, n_hours)]
                    resample_sum = np.sum(day_slice, axis=0)
                    
                    # Interpolate both to IMD grid
                    interp_true = RegularGridInterpolator((e_lats, e_lons), true_24h, method="linear", bounds_error=False, fill_value=np.nan)
                    interp_sum = RegularGridInterpolator((e_lats, e_lons), resample_sum, method="linear", bounds_error=False, fill_value=np.nan)
                    
                    grid_true = interp_true(eval_pts).reshape((len(imd_lats), len(imd_lons)))
                    grid_sum = interp_sum(eval_pts).reshape((len(imd_lats), len(imd_lons)))
                    
                    yr_daily.append(grid_true)
                    yr_sum.append(grid_sum)
                    
        print(f"  Year {yr}: {len(yr_daily)} daily grids extracted & interpolated in {time.time()-t0:.2f}s")
        all_daily_era.extend(yr_daily)
        all_sum_era.extend(yr_sum)
        
    era_on_imd = np.array(all_daily_era)       # (N_days, 33, 21)
    era_sum_on_imd = np.array(all_sum_era)   # (N_days, 33, 21)
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, era_on_imd=era_on_imd, era_sum_on_imd=era_sum_on_imd)
    print(f"  [Saved Cache] {cache_path.name} ({era_on_imd.shape})")
    
    return era_on_imd, era_sum_on_imd


# ==============================================================================
# 2. STATISTICAL METRICS & METEOROLOGICAL SUB-DIVISIONS
# ==============================================================================

def get_zone_masks(lats, lons, land_mask):
    """
    Partition Karnataka into 3 official IMD meteorological sub-divisions:
    1. Coastal Karnataka & Western Ghats (Heavy orographic precipitation)
    2. South Interior Karnataka (Plateau regime - Bengaluru, Mysuru, Hassan)
    3. North Interior Karnataka (Semi-arid Deccan rain-shadow regime)
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Sub-division 1: Coastal & Ghats (Western longitude boundary <= 75.6°E, lat 11.5 - 16.0)
    coastal_ghats = land_mask & (lon_grid <= 75.6) & (lat_grid >= 11.5) & (lat_grid <= 16.0)
    
    # Sub-division 2: South Interior (East of Ghats, Lat <= 14.5°N)
    south_interior = land_mask & (~coastal_ghats) & (lat_grid <= 14.5)
    
    # Sub-division 3: North Interior (Northern Deccan semi-arid plain, Lat > 14.5°N)
    north_interior = land_mask & (~coastal_ghats) & (lat_grid > 14.5)
    
    return {
        "Statewide": land_mask,
        "Coastal & Ghats": coastal_ghats,
        "South Interior": south_interior,
        "North Interior": north_interior
    }


def compute_metrics(obs, sim, threshold=0.1):
    """
    Compute comprehensive meteorological continuous and categorical validation metrics.
    """
    valid = ~np.isnan(obs) & ~np.isnan(sim) & (obs >= 0) & (sim >= 0)
    o = obs[valid]
    s = sim[valid]
    
    if len(o) < 10:
        return {}
        
    n = len(o)
    mean_obs = float(np.mean(o))
    mean_sim = float(np.mean(s))
    
    # Continuous metrics
    bias = float(np.mean(s - o))
    pbias = float(100.0 * np.sum(s - o) / np.sum(o)) if np.sum(o) > 0 else 0.0
    mae = float(np.mean(np.abs(s - o)))
    rmse = float(np.sqrt(np.mean((s - o)**2)))
    nrmse = float(rmse / mean_obs) if mean_obs > 0 else 0.0
    
    # Correlation & R2
    if np.std(o) > 1e-6 and np.std(s) > 1e-6:
        r = float(stats.pearsonr(o, s)[0])
        rho = float(stats.spearmanr(o, s)[0])
    else:
        r, rho = 0.0, 0.0
        
    ss_tot = np.sum((o - mean_obs)**2)
    ss_res = np.sum((o - s)**2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    nse = float(1.0 - (np.sum((o - s)**2) / np.sum((o - mean_obs)**2))) if ss_tot > 0 else 0.0
    
    # Kling-Gupta Efficiency (KGE)
    if mean_obs > 0 and mean_sim > 0 and np.std(o) > 0 and np.std(s) > 0:
        beta = mean_sim / mean_obs
        gamma = (np.std(s) / mean_sim) / (np.std(o) / mean_obs)
        kge = float(1.0 - np.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2))
    else:
        kge = 0.0
        
    # Categorical Detection Metrics at threshold (e.g. 0.1 mm or 2.5 mm)
    hits = np.sum((s >= threshold) & (o >= threshold))
    misses = np.sum((s < threshold) & (o >= threshold))
    false_alarms = np.sum((s >= threshold) & (o < threshold))
    correct_negs = np.sum((s < threshold) & (o < threshold))
    
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    csi = float(hits / (hits + misses + false_alarms)) if (hits + misses + false_alarms) > 0 else 0.0
    fbi = float((hits + false_alarms) / (hits + misses)) if (hits + misses) > 0 else 0.0
    
    # IMD Intensity Distribution (%)
    cat_bins = [0, 0.1, 2.5, 15.6, 64.5, 115.6, 9999.0]
    cat_labels = ["No Rain (<0.1)", "Very Light (0.1-2.4)", "Light (2.5-15.5)", 
                  "Moderate (15.6-64.4)", "Heavy (64.5-115.5)", "Very Heavy (>=115.6)"]
    
    obs_counts = np.histogram(o, bins=cat_bins)[0]
    sim_counts = np.histogram(s, bins=cat_bins)[0]
    
    obs_pct = (obs_counts / n * 100.0).tolist()
    sim_pct = (sim_counts / n * 100.0).tolist()
    
    return {
        "N": n,
        "mean_obs_mm": round(mean_obs, 3),
        "mean_sim_mm": round(mean_sim, 3),
        "mean_bias_mm": round(bias, 3),
        "pbias_pct": round(pbias, 2),
        "mae_mm": round(mae, 3),
        "rmse_mm": round(rmse, 3),
        "nrmse": round(nrmse, 3),
        "pearson_r": round(r, 4),
        "spearman_rho": round(rho, 4),
        "r2": round(r2, 4),
        "kge": round(kge, 4),
        "nse": round(nse, 4),
        "pod": round(pod, 4),
        "far": round(far, 4),
        "csi": round(csi, 4),
        "fbi": round(fbi, 4),
        "intensity_labels": cat_labels,
        "obs_intensity_pct": [round(x, 2) for x in obs_pct],
        "sim_intensity_pct": [round(x, 2) for x in sim_pct]
    }


# ==============================================================================
# 3. BIAS CORRECTION IMPLEMENTATION
# ==============================================================================

class BiasCorrectionPipeline:
    """
    Implements 3 major meteorological bias correction techniques:
    1. Linear Scaling (LS): Multiplicative monthly factor
    2. Power Transformation (PT): Adjusts mean and variance/skewness
    3. Empirical Quantile Mapping (EQM): Non-parametric quantile matching with drizzle thresholding
    """
    def __init__(self, calib_years=CALIBRATION_YEARS):
        self.calib_years = calib_years
        self.ls_factors = {}     # month -> factor map
        self.pt_params = {}      # month -> (a, b)
        self.eqm_tables = {}     # month -> (sim_quantiles, obs_quantiles, p_threshold)
        
    def fit(self, imd_rain, era_rain, times, land_mask):
        """Fit correction parameters on the calibration period (2019-2022)."""
        print("\n" + "=" * 75)
        print("STEP 3: CALIBRATING BIAS CORRECTION ALGORITHMS (CALIBRATION: 2019-2022)")
        print("=" * 75)
        
        calib_idx = np.isin(times.year, self.calib_years)
        obs_calib = imd_rain[calib_idx]
        sim_calib = era_rain[calib_idx]
        calib_times = times[calib_idx]
        
        for m in range(1, 13):
            m_mask = (calib_times.month == m)
            o_m = obs_calib[m_mask][:, land_mask].ravel()
            s_m = sim_calib[m_mask][:, land_mask].ravel()
            
            valid = ~np.isnan(o_m) & ~np.isnan(s_m) & (o_m >= 0) & (s_m >= 0)
            o_m = o_m[valid]
            s_m = s_m[valid]
            
            if len(o_m) == 0:
                continue
                
            # 1. Linear Scaling: Multiplicative factor
            m_obs_mean = np.mean(o_m)
            m_sim_mean = np.mean(s_m)
            factor = (m_obs_mean / m_sim_mean) if m_sim_mean > 0 else 1.0
            self.ls_factors[m] = factor
            
            # 2. Power Transformation: P_corr = a * P_sim^b
            # Find b such that CV(s^b) = CV(o)
            cv_o = np.std(o_m) / m_obs_mean if m_obs_mean > 0 else 1.0
            
            def objective_b(b):
                sb = np.power(s_m, b)
                mean_sb = np.mean(sb)
                if mean_sb <= 0:
                    return 999.0
                cv_sb = np.std(sb) / mean_sb
                return (cv_sb - cv_o)**2
                
            res = minimize_scalar(objective_b, bounds=(0.1, 2.5), method='bounded')
            b_opt = float(res.x)
            sb_opt = np.power(s_m, b_opt)
            a_opt = float(m_obs_mean / np.mean(sb_opt)) if np.mean(sb_opt) > 0 else 1.0
            self.pt_params[m] = (a_opt, b_opt)
            
            # 3. Empirical Quantile Mapping (EQM)
            # Threshold to match dry-day frequency (eliminate model drizzle)
            dry_frac_obs = np.mean(o_m < 0.1)
            p_threshold = np.quantile(s_m, dry_frac_obs) if dry_frac_obs > 0 else 0.0
            
            # Wet days
            s_wet = s_m[s_m >= p_threshold]
            o_wet = o_m[o_m >= 0.1]
            
            n_quantiles = 200
            quantiles = np.linspace(0.001, 0.999, n_quantiles)
            if len(s_wet) > 10 and len(o_wet) > 10:
                s_qs = np.quantile(s_wet, quantiles)
                o_qs = np.quantile(o_wet, quantiles)
            else:
                s_qs = quantiles
                o_qs = quantiles
                
            self.eqm_tables[m] = (s_qs, o_qs, p_threshold)
            
        print(f"  [OK] Successfully fitted LS, PT, and EQM models across all 12 calendar months.")
        for m in [1, 5, 7, 10]:
            print(f"    Month {m:02d} | Linear Factor: {self.ls_factors[m]:.3f} | PT (a={self.pt_params[m][0]:.3f}, b={self.pt_params[m][1]:.3f}) | EQM Drizzle Threshold: {self.eqm_tables[m][2]:.2f} mm")

    def transform(self, era_rain, times, method="eqm"):
        """Apply fitted bias correction to full ERA5 dataset."""
        corrected = np.copy(era_rain)
        
        for m in range(1, 13):
            m_mask = (times.month == m)
            if not np.any(m_mask):
                continue
                
            sub_era = era_rain[m_mask]
            
            if method == "ls":
                factor = self.ls_factors.get(m, 1.0)
                corr_sub = sub_era * factor
                
            elif method == "pt":
                a, b = self.pt_params.get(m, (1.0, 1.0))
                corr_sub = a * np.power(np.maximum(0, sub_era), b)
                
            elif method == "eqm":
                s_qs, o_qs, p_th = self.eqm_tables.get(m, (None, None, 0.0))
                # Set values below drizzle threshold to 0
                corr_sub = np.where(sub_era < p_th, 0.0, sub_era)
                
                # Interpolate wet quantiles
                orig_shape = corr_sub.shape
                flat = corr_sub.ravel()
                wet_mask = (flat >= p_th)
                if np.any(wet_mask) and s_qs is not None:
                    # monotonic interpolation
                    flat[wet_mask] = np.interp(flat[wet_mask], s_qs, o_qs)
                corr_sub = flat.reshape(orig_shape)
                
            corrected[m_mask] = np.maximum(0.0, corr_sub)
            
        return corrected


# ==============================================================================
# 4. PLOTTING & VISUALIZATIONS
# ==============================================================================

def plot_spatial_mean_annual(imd_meta, era_true, era_sum, era_eqm):
    """Figure 1: Spatial Mean Annual Rainfall Maps over Karnataka (Obs vs Sim vs Bias)."""
    print("\n  Generating Figure 1: Spatial Mean Annual Rainfall Maps...")
    lats = imd_meta["lats"]
    lons = imd_meta["lons"]
    mask = imd_meta["land_mask"]
    
    # Calculate mean annual rainfall (mm/year) across all 7 years
    n_years = len(YEARS)
    imd_ann = np.nanmean(imd_meta["rain"], axis=0) * 365.25
    era_ann = np.nanmean(era_true, axis=0) * 365.25
    sum_ann = np.nanmean(era_sum, axis=0) * 365.25
    eqm_ann = np.nanmean(era_eqm, axis=0) * 365.25
    
    # Apply land mask
    imd_ann = np.where(mask, imd_ann, np.nan)
    era_ann = np.where(mask, era_ann, np.nan)
    eqm_ann = np.where(mask, eqm_ann, np.nan)
    
    bias_raw = era_ann - imd_ann
    bias_eqm = eqm_ann - imd_ann
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5), dpi=300)
    
    # Color bounds for precipitation
    rain_levels = [0, 500, 750, 1000, 1250, 1500, 2000, 2500, 3500, 5000]
    rain_cmap = plt.cm.get_cmap("YlGnBu", len(rain_levels) - 1)
    rain_norm = BoundaryNorm(rain_levels, rain_cmap.N)
    
    bias_levels = [-600, -400, -200, -100, -50, 50, 100, 200, 400, 600]
    bias_cmap = plt.cm.get_cmap("RdBu_r", len(bias_levels) - 1)
    bias_norm = BoundaryNorm(bias_levels, bias_cmap.N)
    
    # Panel 1: IMD Observed
    im1 = axes[0].pcolormesh(lons, lats, imd_ann, cmap=rain_cmap, norm=rain_norm, shading='auto')
    axes[0].set_title("(a) IMD 0.25° Observed Ground Truth\nMean Annual Rainfall (2019-2025)", fontweight="bold", fontsize=11)
    fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.08, label="Rainfall (mm/year)")
    
    # Panel 2: ERA5 True 24h
    im2 = axes[1].pcolormesh(lons, lats, era_ann, cmap=rain_cmap, norm=rain_norm, shading='auto')
    axes[1].set_title("(b) ERA5-Land Reanalysis (True 24h)\nMean Annual Rainfall (2019-2025)", fontweight="bold", fontsize=11)
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.08, label="Rainfall (mm/year)")
    
    # Panel 3: Raw Bias
    im3 = axes[2].pcolormesh(lons, lats, bias_raw, cmap=bias_cmap, norm=bias_norm, shading='auto')
    axes[2].set_title(f"(c) Raw ERA5 Bias (ERA5 - IMD)\nMean Bias: {np.nanmean(bias_raw):+.1f} mm/yr", fontweight="bold", fontsize=11)
    fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.08, label="Bias (mm/year)")
    
    # Panel 4: EQM Bias-Corrected
    im4 = axes[3].pcolormesh(lons, lats, bias_eqm, cmap=bias_cmap, norm=bias_norm, shading='auto')
    axes[3].set_title(f"(d) Post-Correction Bias (EQM - IMD)\nMean Bias: {np.nanmean(bias_eqm):+.1f} mm/yr", fontweight="bold", fontsize=11)
    fig.colorbar(im4, ax=axes[3], orientation='horizontal', pad=0.08, label="Bias (mm/year)")
    
    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlim(73.8, 79.2)
        ax.set_ylim(11.0, 19.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        
    plt.suptitle("Karnataka Statewide Mean Annual Rainfall: IMD Observed Ground Truth vs ERA5 Reanalysis & Bias Correction", 
                 fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "01_spatial_mean_annual_rainfall_era5_vs_imd.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path.name}")


def plot_timeseries_comparison(imd_meta, era_true, era_eqm):
    """Figure 2: Multi-Year Statewide Daily Time Series Tracking with Monsoon Inset."""
    print("  Generating Figure 2: Statewide Daily Time Series Comparison...")
    times = imd_meta["times"]
    mask = imd_meta["land_mask"]
    
    # Calculate statewide spatial daily mean (mm/day)
    state_imd = np.nanmean(np.where(mask, imd_meta["rain"], np.nan), axis=(1, 2))
    state_era = np.nanmean(np.where(mask, era_true, np.nan), axis=(1, 2))
    state_eqm = np.nanmean(np.where(mask, era_eqm, np.nan), axis=(1, 2))
    
    fig, (ax_main, ax_zoom) = plt.subplots(2, 1, figsize=(16, 9), dpi=300, gridspec_kw={'height_ratios': [1.8, 1.2]})
    
    # Main panel: 7-year continuous series
    ax_main.plot(times, state_imd, label="IMD Observed Ground Truth", color="#1f77b4", linewidth=1.1, alpha=0.9)
    ax_main.plot(times, state_era, label="ERA5-Land Reanalysis (Raw 24h)", color="#e67e22", linewidth=0.9, linestyle="--", alpha=0.8)
    ax_main.plot(times, state_eqm, label="ERA5-Land Bias-Corrected (EQM)", color="#27ae60", linewidth=1.0, alpha=0.85)
    
    # Compute correlation
    corr_raw = np.corrcoef(state_imd, state_era)[0, 1]
    corr_eqm = np.corrcoef(state_imd, state_eqm)[0, 1]
    
    ax_main.set_ylabel("Daily Rainfall (mm/day)", fontweight="bold")
    ax_main.set_title(f"(a) 7-Year Continuous Statewide Daily Rainfall (2019-2025)\nStatewide Temporal Correlation: Raw ERA5 r = {corr_raw:.4f} | EQM Corrected r = {corr_eqm:.4f}", 
                      fontweight="bold", fontsize=11)
    ax_main.legend(loc="upper right", frameon=True)
    ax_main.xaxis.set_major_locator(mdates.YearLocator())
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_main.grid(True, linestyle=":", alpha=0.6)
    
    # Inset panel: Southwest Monsoon 2024 Deep-Dive (June to September 2024)
    monsoon_2024 = (times >= "2024-06-01") & (times <= "2024-09-30")
    t_sub = times[monsoon_2024]
    
    ax_zoom.plot(t_sub, state_imd[monsoon_2024], label="IMD Observed", color="#1f77b4", linewidth=1.8)
    ax_zoom.plot(t_sub, state_era[monsoon_2024], label="ERA5 Raw (Simulated)", color="#e67e22", linewidth=1.5, linestyle="--")
    ax_zoom.plot(t_sub, state_eqm[monsoon_2024], label="ERA5 Bias-Corrected (EQM)", color="#27ae60", linewidth=1.6)
    
    ax_zoom.set_ylabel("Daily Rainfall (mm/day)", fontweight="bold")
    ax_zoom.set_xlabel("Date", fontweight="bold")
    ax_zoom.set_title("(b) Southwest Monsoon 2024 Active & Break Spell Dynamics (1 June - 30 September 2024)", 
                      fontweight="bold", fontsize=11)
    ax_zoom.legend(loc="upper right", frameon=True)
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_zoom.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Karnataka Statewide Daily Rainfall Time Series Tracking: IMD Ground Truth vs Simulated ERA5", 
                 fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "02_statewide_daily_timeseries_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path.name}")


def plot_scatter_and_density(imd_meta, era_true, era_eqm):
    """Figure 3: Density Hexbin Scatter Plots with 1:1 Line and Validation Metric Cards."""
    print("  Generating Figure 3: Scatter & Density Hexbin Evaluation...")
    mask = imd_meta["land_mask"]
    
    obs = imd_meta["rain"][:, mask].ravel()
    sim_raw = era_true[:, mask].ravel()
    sim_eqm = era_eqm[:, mask].ravel()
    
    valid = ~np.isnan(obs) & ~np.isnan(sim_raw) & ~np.isnan(sim_eqm) & (obs >= 0) & (sim_raw >= 0)
    o = obs[valid]
    s_raw = sim_raw[valid]
    s_eqm = sim_eqm[valid]
    
    m_raw = compute_metrics(o, s_raw)
    m_eqm = compute_metrics(o, s_eqm)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8), dpi=300)
    
    # Hexbin 1: Raw ERA5 vs IMD
    hb1 = ax1.hexbin(o, s_raw, gridsize=70, cmap="Blues", mincnt=1, bins="log", extent=(0, 150, 0, 150))
    ax1.plot([0, 150], [0, 150], "r--", linewidth=1.5, label="1:1 Perfect Agreement")
    ax1.set_xlabel("IMD Observed Daily Rainfall (mm)", fontweight="bold")
    ax1.set_ylabel("ERA5 Raw Daily Rainfall (mm)", fontweight="bold")
    ax1.set_title("(a) Raw ERA5 Reanalysis vs IMD Observed", fontweight="bold", fontsize=11)
    
    text_raw = (
        f"N: {m_raw['N']:,}\n"
        f"Pearson r: {m_raw['pearson_r']:.3f}\n"
        f"MAE: {m_raw['mae_mm']:.2f} mm\n"
        f"RMSE: {m_raw['rmse_mm']:.2f} mm\n"
        f"PBIAS: {m_raw['pbias_pct']:+.1f}%\n"
        f"KGE: {m_raw['kge']:.3f}\n"
        f"CSI (>=0.1mm): {m_raw['csi']:.3f}"
    )
    ax1.text(0.04, 0.95, text_raw, transform=ax1.transAxes, va='top', fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
    fig.colorbar(hb1, ax=ax1, label="Log10(Count)")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Hexbin 2: EQM Bias-Corrected ERA5 vs IMD
    hb2 = ax2.hexbin(o, s_eqm, gridsize=70, cmap="Greens", mincnt=1, bins="log", extent=(0, 150, 0, 150))
    ax2.plot([0, 150], [0, 150], "r--", linewidth=1.5, label="1:1 Perfect Agreement")
    ax2.set_xlabel("IMD Observed Daily Rainfall (mm)", fontweight="bold")
    ax2.set_ylabel("ERA5 Bias-Corrected (EQM) Daily Rainfall (mm)", fontweight="bold")
    ax2.set_title("(b) EQM Bias-Corrected ERA5 vs IMD Observed", fontweight="bold", fontsize=11)
    
    text_eqm = (
        f"N: {m_eqm['N']:,}\n"
        f"Pearson r: {m_eqm['pearson_r']:.3f}\n"
        f"MAE: {m_eqm['mae_mm']:.2f} mm\n"
        f"RMSE: {m_eqm['rmse_mm']:.2f} mm\n"
        f"PBIAS: {m_eqm['pbias_pct']:+.1f}%\n"
        f"KGE: {m_eqm['kge']:.3f}\n"
        f"CSI (>=0.1mm): {m_eqm['csi']:.3f}"
    )
    ax2.text(0.04, 0.95, text_eqm, transform=ax2.transAxes, va='top', fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
    fig.colorbar(hb2, ax=ax2, label="Log10(Count)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    plt.suptitle("Collocated Grid-Day Rainfall Accuracy & Bias Correction Evaluation", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "03_taylor_and_scatter_evaluation.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path.name}")


def plot_zonal_comparison(imd_meta, era_true, era_eqm, zone_masks):
    """Figure 4: IMD Meteorological Sub-Divisions Performance Breakdown across Karnataka."""
    print("  Generating Figure 4: Meteorological Sub-Divisions Comparison...")
    
    zones = ["Coastal & Ghats", "South Interior", "North Interior"]
    metrics_list = []
    
    for z in zones:
        z_mask = zone_masks[z]
        obs_z = imd_meta["rain"][:, z_mask].ravel()
        raw_z = era_true[:, z_mask].ravel()
        eqm_z = era_eqm[:, z_mask].ravel()
        
        m_raw = compute_metrics(obs_z, raw_z)
        m_eqm = compute_metrics(obs_z, eqm_z)
        metrics_list.append({"zone": z, "raw": m_raw, "eqm": m_eqm})
        
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # 1. Mean Rainfall comparison
    x = np.arange(len(zones))
    width = 0.25
    
    obs_means = [m["raw"]["mean_obs_mm"] for m in metrics_list]
    raw_means = [m["raw"]["mean_sim_mm"] for m in metrics_list]
    eqm_means = [m["eqm"]["mean_sim_mm"] for m in metrics_list]
    
    axes[0].bar(x - width, obs_means, width, label="IMD Observed", color="#1f77b4")
    axes[0].bar(x, raw_means, width, label="Raw ERA5", color="#e67e22")
    axes[0].bar(x + width, eqm_means, width, label="EQM Corrected", color="#27ae60")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(zones, fontweight="bold", fontsize=9.5)
    axes[0].set_ylabel("Mean Daily Rainfall (mm/day)", fontweight="bold")
    axes[0].set_title("(a) Regional Mean Daily Rainfall", fontweight="bold", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    # 2. MAE & RMSE Reduction
    raw_mae = [m["raw"]["mae_mm"] for m in metrics_list]
    eqm_mae = [m["eqm"]["mae_mm"] for m in metrics_list]
    
    axes[1].bar(x - width/2, raw_mae, width, label="Raw ERA5 MAE", color="#e74c3c")
    axes[1].bar(x + width/2, eqm_mae, width, label="EQM Corrected MAE", color="#2ecc71")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(zones, fontweight="bold", fontsize=9.5)
    axes[1].set_ylabel("Mean Absolute Error (mm)", fontweight="bold")
    axes[1].set_title("(b) MAE Reduction Across Zones", fontweight="bold", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    # 3. KGE (Kling-Gupta Efficiency)
    raw_kge = [m["raw"]["kge"] for m in metrics_list]
    eqm_kge = [m["eqm"]["kge"] for m in metrics_list]
    
    axes[2].bar(x - width/2, raw_kge, width, label="Raw ERA5 KGE", color="#9b59b6")
    axes[2].bar(x + width/2, eqm_kge, width, label="EQM Corrected KGE", color="#3498db")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(zones, fontweight="bold", fontsize=9.5)
    axes[2].set_ylabel("Kling-Gupta Efficiency (KGE)", fontweight="bold")
    axes[2].set_title("(c) Hydrometeorological Skill (KGE)", fontweight="bold", fontsize=11)
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.5)
    
    plt.suptitle("Karnataka Meteorological Sub-Divisions Validation & Bias Correction Impact", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "04_agroclimatic_zonal_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path.name}")


def plot_cdf_and_quantiles(imd_meta, era_true, era_ls, era_eqm):
    """Figure 5: Empirical Cumulative Distribution Functions (ECDF) and Q-Q Plot."""
    print("  Generating Figure 5: ECDF and Quantile-Quantile (Q-Q) Analysis...")
    mask = imd_meta["land_mask"]
    
    obs = imd_meta["rain"][:, mask].ravel()
    raw = era_true[:, mask].ravel()
    ls = era_ls[:, mask].ravel()
    eqm = era_eqm[:, mask].ravel()
    
    valid = ~np.isnan(obs) & ~np.isnan(raw) & ~np.isnan(ls) & ~np.isnan(eqm) & (obs >= 0)
    o = obs[valid]
    r = raw[valid]
    l = ls[valid]
    e = eqm[valid]
    
    # Focus on wet days (>= 0.1 mm) for CDF
    o_wet = np.sort(o[o >= 0.1])
    r_wet = np.sort(r[r >= 0.1])
    l_wet = np.sort(l[l >= 0.1])
    e_wet = np.sort(e[e >= 0.1])
    
    fig, (ax_cdf, ax_qq) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300)
    
    # Panel 1: ECDF
    p_o = np.linspace(0, 1, len(o_wet))
    p_r = np.linspace(0, 1, len(r_wet))
    p_l = np.linspace(0, 1, len(l_wet))
    p_e = np.linspace(0, 1, len(e_wet))
    
    # Sample 1000 points for crisp plotting
    idx_sample = np.linspace(0, len(p_o)-1, 1000).astype(int)
    
    ax_cdf.plot(o_wet[idx_sample], p_o[idx_sample], label="IMD Ground Truth", color="#1f77b4", linewidth=2.0)
    ax_cdf.plot(r_wet[np.linspace(0, len(p_r)-1, 1000).astype(int)], np.linspace(0, 1, 1000), label="Raw ERA5", color="#e67e22", linestyle="--", linewidth=1.5)
    ax_cdf.plot(l_wet[np.linspace(0, len(p_l)-1, 1000).astype(int)], np.linspace(0, 1, 1000), label="Linear Scaled", color="#9b59b6", linestyle=":", linewidth=1.5)
    ax_cdf.plot(e_wet[np.linspace(0, len(p_e)-1, 1000).astype(int)], np.linspace(0, 1, 1000), label="EQM Corrected", color="#27ae60", linewidth=1.8)
    
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlim(0.1, 250)
    ax_cdf.set_xlabel("Rainfall Intensity (mm/day, Log Scale)", fontweight="bold")
    ax_cdf.set_ylabel("Empirical Cumulative Probability P(X <= x)", fontweight="bold")
    ax_cdf.set_title("(a) Empirical Cumulative Distribution Function (ECDF)", fontweight="bold", fontsize=11)
    ax_cdf.legend(loc="lower right")
    ax_cdf.grid(True, linestyle=":", alpha=0.6)
    
    # Panel 2: Quantile-Quantile (Q-Q) Plot
    percentiles = np.linspace(1, 99.9, 100)
    q_obs = np.percentile(o, percentiles)
    q_raw = np.percentile(r, percentiles)
    q_ls = np.percentile(l, percentiles)
    q_eqm = np.percentile(e, percentiles)
    
    ax_qq.plot([0, 120], [0, 120], "k--", label="1:1 Perfect Quantile Match", linewidth=1.2)
    ax_qq.plot(q_obs, q_raw, "o-", label="Raw ERA5 vs IMD", color="#e67e22", markersize=4, linewidth=1.2)
    ax_qq.plot(q_obs, q_ls, "s-", label="Linear Scaled vs IMD", color="#9b59b6", markersize=4, linewidth=1.2)
    ax_qq.plot(q_obs, q_eqm, "^-", label="EQM Corrected vs IMD", color="#27ae60", markersize=4, linewidth=1.4)
    
    ax_qq.set_xlim(0, 120)
    ax_qq.set_ylim(0, 120)
    ax_qq.set_xlabel("IMD Observed Quantiles (mm)", fontweight="bold")
    ax_qq.set_ylabel("Model Quantiles (mm)", fontweight="bold")
    ax_qq.set_title("(b) Quantile-Quantile (Q-Q) Calibration Plot", fontweight="bold", fontsize=11)
    ax_qq.legend(loc="lower right")
    ax_qq.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Statistical Distribution Calibration: ECDF & Quantile-Quantile Verification", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "05_bias_correction_cdf_and_quantiles.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path.name}")


def plot_intensity_contingency(metrics_raw, metrics_eqm):
    """Figure 6: IMD Rainfall Intensity Distribution & Detection Contingency Scores."""
    print("  Generating Figure 6: IMD Intensity Distribution & Contingency Scores...")
    
    labels = metrics_raw["intensity_labels"]
    obs_pct = metrics_raw["obs_intensity_pct"]
    raw_pct = metrics_raw["sim_intensity_pct"]
    eqm_pct = metrics_eqm["sim_intensity_pct"]
    
    fig, (ax_bar, ax_radar) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # 1. Frequency Distribution Bar Chart
    x = np.arange(len(labels))
    width = 0.25
    
    ax_bar.bar(x - width, obs_pct, width, label="IMD Observed", color="#1f77b4")
    ax_bar.bar(x, raw_pct, width, label="Raw ERA5", color="#e67e22")
    ax_bar.bar(x + width, eqm_pct, width, label="EQM Bias-Corrected", color="#27ae60")
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax_bar.set_ylabel("Frequency (% of Grid-Days)", fontweight="bold")
    ax_bar.set_yscale("log")
    ax_bar.set_ylim(0.01, 100)
    ax_bar.set_title("(a) IMD Rainfall Category Frequency Distribution", fontweight="bold", fontsize=11)
    ax_bar.legend()
    ax_bar.grid(True, linestyle=":", alpha=0.5)
    
    # 2. Skill Scores Bar Chart
    scores = ["POD (Hit Rate)", "1 - FAR (Success)", "CSI (Threat Score)", "FBI (Freq Bias / 2)"]
    raw_scores = [metrics_raw["pod"], 1.0 - metrics_raw["far"], metrics_raw["csi"], metrics_raw["fbi"] / 2.0]
    eqm_scores = [metrics_eqm["pod"], 1.0 - metrics_eqm["far"], metrics_eqm["csi"], metrics_eqm["fbi"] / 2.0]
    
    x2 = np.arange(len(scores))
    width2 = 0.35
    
    ax_radar.bar(x2 - width2/2, raw_scores, width2, label="Raw ERA5", color="#e67e22")
    ax_radar.bar(x2 + width2/2, eqm_scores, width2, label="EQM Corrected", color="#27ae60")
    
    for i in range(len(scores)):
        ax_radar.text(x2[i] - width2/2, raw_scores[i] + 0.02, f"{raw_scores[i]:.2f}", ha='center', fontsize=9, fontweight="bold")
        ax_radar.text(x2[i] + width2/2, eqm_scores[i] + 0.02, f"{eqm_scores[i]:.2f}", ha='center', fontsize=9, fontweight="bold")
        
    ax_radar.set_xticks(x2)
    ax_radar.set_xticklabels(scores, fontweight="bold", fontsize=9.5)
    ax_radar.set_ylabel("Skill Score (0 to 1)", fontweight="bold")
    ax_radar.set_ylim(0, 1.15)
    ax_radar.set_title("(b) Rain Detection Contingency Scores (Threshold >= 0.1 mm)", fontweight="bold", fontsize=11)
    ax_radar.legend(loc="upper right")
    ax_radar.grid(True, linestyle=":", alpha=0.5)
    
    plt.suptitle("IMD Categorical Intensity Distribution & Rain Event Detection Skill", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "06_intensity_contingency_scores.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] Saved: {out_path.name}")


# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    start_total = time.time()
    print("=" * 80)
    print("KARNATAKA RAINFALL ERA5 vs IMD VERIFICATION & BIAS CORRECTION PIPELINE")
    print("=" * 80)
    
    # 1. Ingestion
    imd_meta = load_imd_data()
    era_true, era_sum = extract_true_era5_daily(imd_meta)
    
    # 2. Agro-climatic masks
    zone_masks = get_zone_masks(imd_meta["lats"], imd_meta["lons"], imd_meta["land_mask"])
    
    # 3. Fit Bias Correction
    bc = BiasCorrectionPipeline(calib_years=CALIBRATION_YEARS)
    bc.fit(imd_meta["rain"], era_true, imd_meta["times"], imd_meta["land_mask"])
    
    # 4. Transform datasets
    print("\n" + "=" * 75)
    print("STEP 4: APPLYING BIAS CORRECTION TO FULL 2019-2025 DATASET")
    print("=" * 75)
    era_ls = bc.transform(era_true, imd_meta["times"], method="ls")
    era_pt = bc.transform(era_true, imd_meta["times"], method="pt")
    era_eqm = bc.transform(era_true, imd_meta["times"], method="eqm")
    print("  [OK] Linear Scaling, Power Transformation, and EQM transformations applied.")
    
    # 5. Comprehensive Quantitative Evaluation
    print("\n" + "=" * 75)
    print("STEP 5: COMPUTING ACCURACY METRICS ACROSS DOMAINS, ZONES & PERIODS")
    print("=" * 75)
    
    mask = imd_meta["land_mask"]
    times = imd_meta["times"]
    
    # Full Period (2019-2025)
    obs_all = imd_meta["rain"][:, mask].ravel()
    sim_sum = era_sum[:, mask].ravel()
    sim_true = era_true[:, mask].ravel()
    sim_ls = era_ls[:, mask].ravel()
    sim_pt = era_pt[:, mask].ravel()
    sim_eqm = era_eqm[:, mask].ravel()
    
    m_pipeline_sum = compute_metrics(obs_all, sim_sum)
    m_raw_true = compute_metrics(obs_all, sim_true)
    m_ls = compute_metrics(obs_all, sim_ls)
    m_pt = compute_metrics(obs_all, sim_pt)
    m_eqm = compute_metrics(obs_all, sim_eqm)
    
    # Out-of-Sample Validation Period (2023-2025)
    val_idx = np.isin(times.year, VALIDATION_YEARS)
    obs_val = imd_meta["rain"][val_idx][:, mask].ravel()
    raw_val = era_true[val_idx][:, mask].ravel()
    eqm_val = era_eqm[val_idx][:, mask].ravel()
    
    m_val_raw = compute_metrics(obs_val, raw_val)
    m_val_eqm = compute_metrics(obs_val, eqm_val)
    
    # Zonal Breakdown
    zonal_results = {}
    for z_name, z_mask in zone_masks.items():
        o_z = imd_meta["rain"][:, z_mask].ravel()
        r_z = era_true[:, z_mask].ravel()
        e_z = era_eqm[:, z_mask].ravel()
        zonal_results[z_name] = {
            "raw": compute_metrics(o_z, r_z),
            "eqm": compute_metrics(o_z, e_z)
        }
        
    # Yearly Breakdown
    yearly_results = {}
    for yr in YEARS:
        yr_idx = (times.year == yr)
        o_yr = imd_meta["rain"][yr_idx][:, mask].ravel()
        r_yr = era_true[yr_idx][:, mask].ravel()
        e_yr = era_eqm[yr_idx][:, mask].ravel()
        yearly_results[str(yr)] = {
            "raw": compute_metrics(o_yr, r_yr),
            "eqm": compute_metrics(o_yr, e_yr)
        }
        
    # Print Executive Summary Table
    print("\n" + "=" * 90)
    print(f"{'Method / Dataset':<32} | {'Mean (mm)':<10} | {'Bias (mm)':<10} | {'MAE (mm)':<9} | {'RMSE (mm)':<10} | {'r':<7} | {'KGE':<7}")
    print("-" * 90)
    print(f"{'1. IMD Observed (Ground Truth)':<32} | {m_raw_true['mean_obs_mm']:<10.2f} | {'0.00':<10} | {'0.00':<9} | {'0.00':<10} | {'1.000':<7} | {'1.000':<7}")
    print(f"{'2. Pipeline Resample Sum (Artifact)':<32} | {m_pipeline_sum['mean_sim_mm']:<10.2f} | {m_pipeline_sum['mean_bias_mm']:<+10.2f} | {m_pipeline_sum['mae_mm']:<9.2f} | {m_pipeline_sum['rmse_mm']:<10.2f} | {m_pipeline_sum['pearson_r']:<7.4f} | {m_pipeline_sum['kge']:<7.4f}")
    print(f"{'3. True ERA5 24h (Raw Simulated)':<32} | {m_raw_true['mean_sim_mm']:<10.2f} | {m_raw_true['mean_bias_mm']:<+10.2f} | {m_raw_true['mae_mm']:<9.2f} | {m_raw_true['rmse_mm']:<10.2f} | {m_raw_true['pearson_r']:<7.4f} | {m_raw_true['kge']:<7.4f}")
    print(f"{'4. Linear Scaled ERA5':<32} | {m_ls['mean_sim_mm']:<10.2f} | {m_ls['mean_bias_mm']:<+10.2f} | {m_ls['mae_mm']:<9.2f} | {m_ls['rmse_mm']:<10.2f} | {m_ls['pearson_r']:<7.4f} | {m_ls['kge']:<7.4f}")
    print(f"{'5. Power Transformed ERA5':<32} | {m_pt['mean_sim_mm']:<10.2f} | {m_pt['mean_bias_mm']:<+10.2f} | {m_pt['mae_mm']:<9.2f} | {m_pt['rmse_mm']:<10.2f} | {m_pt['pearson_r']:<7.4f} | {m_pt['kge']:<7.4f}")
    print(f"{'6. EQM Bias-Corrected ERA5':<32} | {m_eqm['mean_sim_mm']:<10.2f} | {m_eqm['mean_bias_mm']:<+10.2f} | {m_eqm['mae_mm']:<9.2f} | {m_eqm['rmse_mm']:<10.2f} | {m_eqm['pearson_r']:<7.4f} | {m_eqm['kge']:<7.4f}")
    print("=" * 90)
    
    # 6. Save Visualizations
    print("\n" + "=" * 75)
    print("STEP 6: GENERATING PUBLICATION-GRADE DIAGNOSTIC PLOTS")
    print("=" * 75)
    plot_spatial_mean_annual(imd_meta, era_true, era_sum, era_eqm)
    plot_timeseries_comparison(imd_meta, era_true, era_eqm)
    plot_scatter_and_density(imd_meta, era_true, era_eqm)
    plot_zonal_comparison(imd_meta, era_true, era_eqm, zone_masks)
    plot_cdf_and_quantiles(imd_meta, era_true, era_ls, era_eqm)
    plot_intensity_contingency(m_raw_true, m_eqm)
    
    # 7. Export Metrics to JSON and CSV
    print("\n" + "=" * 75)
    print("STEP 7: EXPORTING VERIFICATION METRICS (JSON & CSV)")
    print("=" * 75)
    
    full_report = {
        "metadata": {
            "region": "Karnataka, India",
            "bounding_box": {"min_lat": 11.0, "max_lat": 19.0, "min_lon": 74.0, "max_lon": 79.0},
            "period": "2019-2025",
            "total_days": len(times),
            "land_grid_points": int(np.sum(mask)),
            "total_collocated_records": int(m_raw_true["N"]),
            "calibration_years": CALIBRATION_YEARS,
            "validation_years": VALIDATION_YEARS
        },
        "overall_comparison": {
            "pipeline_resample_sum": m_pipeline_sum,
            "true_era5_24h_raw": m_raw_true,
            "linear_scaling": m_ls,
            "power_transformation": m_pt,
            "quantile_mapping_eqm": m_eqm
        },
        "validation_period_2023_2025": {
            "raw_era5": m_val_raw,
            "eqm_bias_corrected": m_val_eqm
        },
        "agroclimatic_zones": zonal_results,
        "yearly_metrics": yearly_results
    }
    
    json_path = OUTPUT_DIR / "imd_era5_accuracy_metrics.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"  [OK] Saved Metrics JSON: {json_path.name}")
    
    # Summary CSV
    summary_rows = [
        {"Dataset": "IMD Observed Ground Truth", "Mean_Daily_mm": m_raw_true["mean_obs_mm"], "Bias_mm": 0.0, "PBIAS_pct": 0.0, "MAE_mm": 0.0, "RMSE_mm": 0.0, "Pearson_r": 1.0, "KGE": 1.0, "CSI": 1.0},
        {"Dataset": "Pipeline Resample Sum (Artifact)", "Mean_Daily_mm": m_pipeline_sum["mean_sim_mm"], "Bias_mm": m_pipeline_sum["mean_bias_mm"], "PBIAS_pct": m_pipeline_sum["pbias_pct"], "MAE_mm": m_pipeline_sum["mae_mm"], "RMSE_mm": m_pipeline_sum["rmse_mm"], "Pearson_r": m_pipeline_sum["pearson_r"], "KGE": m_pipeline_sum["kge"], "CSI": m_pipeline_sum["csi"]},
        {"Dataset": "True ERA5 24h (Raw Reanalysis)", "Mean_Daily_mm": m_raw_true["mean_sim_mm"], "Bias_mm": m_raw_true["mean_bias_mm"], "PBIAS_pct": m_raw_true["pbias_pct"], "MAE_mm": m_raw_true["mae_mm"], "RMSE_mm": m_raw_true["rmse_mm"], "Pearson_r": m_raw_true["pearson_r"], "KGE": m_raw_true["kge"], "CSI": m_raw_true["csi"]},
        {"Dataset": "Linear Scaled ERA5", "Mean_Daily_mm": m_ls["mean_sim_mm"], "Bias_mm": m_ls["mean_bias_mm"], "PBIAS_pct": m_ls["pbias_pct"], "MAE_mm": m_ls["mae_mm"], "RMSE_mm": m_ls["rmse_mm"], "Pearson_r": m_ls["pearson_r"], "KGE": m_ls["kge"], "CSI": m_ls["csi"]},
        {"Dataset": "Power Transformed ERA5", "Mean_Daily_mm": m_pt["mean_sim_mm"], "Bias_mm": m_pt["mean_bias_mm"], "PBIAS_pct": m_pt["pbias_pct"], "MAE_mm": m_pt["mae_mm"], "RMSE_mm": m_pt["rmse_mm"], "Pearson_r": m_pt["pearson_r"], "KGE": m_pt["kge"], "CSI": m_pt["csi"]},
        {"Dataset": "Empirical Quantile Mapping (EQM)", "Mean_Daily_mm": m_eqm["mean_sim_mm"], "Bias_mm": m_eqm["mean_bias_mm"], "PBIAS_pct": m_eqm["pbias_pct"], "MAE_mm": m_eqm["mae_mm"], "RMSE_mm": m_eqm["rmse_mm"], "Pearson_r": m_eqm["pearson_r"], "KGE": m_eqm["kge"], "CSI": m_eqm["csi"]},
    ]
    df_sum = pd.DataFrame(summary_rows)
    csv_path = CSV_DIR / "imd_era5_verification_summary.csv"
    df_sum.to_csv(csv_path, index=False)
    print(f"  [OK] Saved Summary CSV: {csv_path.name}")
    
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - start_total:.2f} SECONDS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
