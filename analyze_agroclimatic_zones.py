"""
Karnataka Rainfall Project: 10 Agro-Climatic Zones Analysis & Seasonal Shifts
=============================================================================
This script performs:
1. Spatial delineation of the 10 Agro-Climatic Zones of Karnataka (ICAR / UAS Bangalore / UAS Dharwad standard).
2. Point-in-polygon mapping of all 0.25° IMD & ERA5 grid points to districts and zones.
3. Multi-year seasonal disaggregation (SW Monsoon JJAS vs NE Monsoon OND) from 2019 to 2025.
4. Evaluation of ERA5-Land reanalysis accuracy and bias against IMD ground truth per zone.
5. Detection of rainfall shifts, dry spell risks, and extreme event trends for agronomic impact.
6. Export of clean summary tables (JSON & CSV) and publication-quality diagnostic plots.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

BASE_DIR = Path(r"d:\PROJECT\karnataka_rainfall_cds")
GEOJSON_PATH = BASE_DIR / "karnataka_districts.geojson"
CACHE_PATH = BASE_DIR / "Preprocessed_Karnataka" / "era5_true_24h_on_imd_grid.npz"
IMD_DIR = BASE_DIR / "IMD_Data"
PLOTS_DIR = BASE_DIR / "IMD_Verification_Plots"
OUTPUT_DIR = BASE_DIR / "Model_Outputs"
CSV_DIR = BASE_DIR / "CSVs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Zone Metadata (Official UAS / ICAR Definitions)
AGRO_ZONES = {
    1: {
        "name": "North Eastern Transition Zone",
        "hq": "Bidar / Aland",
        "districts": ["Bidar", "Kalaburagi (North)"],
        "taluks": "Aurad, Basavakalyan, Bhalki, Bidar, Humnabad, Aland, Chincholi",
        "major_crops": "Red gram (tur), black gram, green gram, jowar, sugarcane",
        "color": "#e41a1c"
    },
    2: {
        "name": "North Eastern Dry Zone",
        "hq": "Raichur / Kalaburagi",
        "districts": ["Raichur", "Yadgir", "Kalaburagi (South)"],
        "taluks": "Gulbarga, Afzalpur, Chittapur, Jewargi, Sedam, Raichur, Manvi, Deodurga, Yadgir, Shahapur, Shorapur",
        "major_crops": "Rabi jowar, bajra, pigeonpea, sunflower, cotton, canal paddy",
        "color": "#377eb8"
    },
    3: {
        "name": "Northern Dry Zone",
        "hq": "Vijayapura / Bagalkote",
        "districts": ["Vijayapura", "Bagalkote", "Gadag", "Koppal", "Ballari", "Belagavi (East)"],
        "taluks": "Bijapur, Indi, Sindgi, Mudhol, Badami, Jamkhandi, Ron, Gangavathi, Hospet, Ballari, Athani, Gokak",
        "major_crops": "Rabi sorghum, chickpea (Bengal gram), maize, bajra, cotton, pomegranate",
        "color": "#4daf4a"
    },
    4: {
        "name": "Central Dry Zone",
        "hq": "Hiriyur / Chitradurga",
        "districts": ["Chitradurga", "Davanagere (East)", "Tumakuru (North)"],
        "taluks": "Challakere, Chitradurga, Hiriyur, Hosadurga, Holalkere, Jagalur, Pavagada, Sira, Madhugiri",
        "major_crops": "Groundnut, ragi (finger millet), maize, sunflower, onion, arecanut",
        "color": "#984ea3"
    },
    5: {
        "name": "Eastern Dry Zone",
        "hq": "Hebbal / Bengaluru",
        "districts": ["Bengaluru Urban", "Bengaluru Rural", "Ramanagara", "Kolar", "Chikkaballapura", "Tumakuru (South)"],
        "taluks": "Bangalore North/South, Devanahalli, Doddaballapura, Ramanagara, Channapatna, Kolar, Chikkaballapur, Tumkur",
        "major_crops": "Ragi, redgram, maize, vegetables, mulberry (sericulture), mango, grapes",
        "color": "#ff7f00"
    },
    6: {
        "name": "Southern Dry Zone",
        "hq": "Mandya",
        "districts": ["Mandya", "Mysuru (East)", "Chamarajanagara"],
        "taluks": "Mandya, Maddur, Malavalli, Pandavapura, Srirangapatna, Mysuru, Nanjangud, T. Narasipura, Kollegal, Gundlupet",
        "major_crops": "Sugarcane, paddy (Cauvery command), ragi, pulses, coconut",
        "color": "#a65628"
    },
    7: {
        "name": "Southern Transition Zone",
        "hq": "Navile / Shivamogga / Hassan",
        "districts": ["Hassan (East)", "Chikkamagaluru (East)", "Shivamogga (East)", "Davanagere (West)", "Mysuru (West)"],
        "taluks": "Hassan, Channarayapatna, Holenarasipura, Arkalgud, Belur, Hunsur, Periyapatna, Bhadravathi, Honnali, Tarikere",
        "major_crops": "Maize, ragi, tobacco, potato, ginger, pulses, paddy",
        "color": "#f781bf"
    },
    8: {
        "name": "Northern Transition Zone",
        "hq": "Dharwad",
        "districts": ["Dharwad", "Haveri", "Belagavi (West)"],
        "taluks": "Dharwad, Hubballi, Kalghatgi, Kundgol, Haveri, Byadgi, Ranebennur, Bailhongal, Chikkodi, Hukkeri",
        "major_crops": "Cotton, soybean, Byadgi chilli, maize, wheat, groundnut, sorghum",
        "color": "#999999"
    },
    9: {
        "name": "Hilly Zone (Malnad)",
        "hq": "Mudigere / Sirsi",
        "districts": ["Kodagu", "Hassan (West)", "Chikkamagaluru (West)", "Shivamogga (West)", "Uttara Kannada (Ghats)"],
        "taluks": "Madikeri, Somwarpet, Virajpet, Sakleshpur, Mudigere, Koppa, Sringeri, Sagar, Hosanagar, Thirthahalli, Sirsi, Yellapur",
        "major_crops": "Coffee, tea, black pepper, cardamom, arecanut, paddy, ginger, rubber",
        "color": "#66c2a5"
    },
    10: {
        "name": "Coastal Zone",
        "hq": "Brahmavara (Udupi)",
        "districts": ["Dakshina Kannada", "Udupi", "Uttara Kannada (Coast)"],
        "taluks": "Mangaluru, Bantwal, Puttur, Belthangady, Sullia, Udupi, Kundapura, Karkala, Karwar, Ankola, Kumta, Honnavar, Bhatkal",
        "major_crops": "Kharif paddy, cashew, coconut, arecanut, black pepper, fisheries",
        "color": "#1f78b4"
    }
}


def load_spatial_delineation(lats, lons, land_mask):
    """Map all grid points to Karnataka districts and 10 Agro-Climatic Zones."""
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    district_polygons = {}
    for feat in geojson_data["features"]:
        dname = feat["properties"]["district"]
        geom = feat["geometry"]
        paths = []
        if geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                paths.append(MplPath(ring))
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    paths.append(MplPath(ring))
        district_polygons[dname] = paths

    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if not land_mask[i, j]:
                continue
            pt = (lon, lat)
            found_dist = None
            for dname, paths in district_polygons.items():
                if any(p.contains_point(pt) for p in paths):
                    found_dist = dname
                    break
            if not found_dist:
                continue

            # Classify into 10 Agro-Climatic Zones
            zone_id = None
            if found_dist == "Bidar":
                zone_id = 1
            elif found_dist == "Kalaburagi":
                zone_id = 1 if lat >= 17.5 else 2
            elif found_dist in ["Yadgir", "Raichur"]:
                zone_id = 2
            elif found_dist in ["Vijayapura", "Bagalkote", "Gadag", "Koppal", "Ballari"]:
                zone_id = 3
            elif found_dist == "Belagavi":
                zone_id = 3 if lon >= 74.8 else 8
            elif found_dist == "Chitradurga":
                zone_id = 4
            elif found_dist == "Davanagere":
                zone_id = 4 if lon >= 75.8 else 7
            elif found_dist == "Tumakuru":
                zone_id = 4 if lat >= 13.5 else 5
            elif found_dist in ["Bengaluru Urban", "Bengaluru Rural", "Ramanagara", "Kolar", "Chikkaballapura"]:
                zone_id = 5
            elif found_dist in ["Mandya", "Chamarajanagara"]:
                zone_id = 6
            elif found_dist == "Mysuru":
                zone_id = 6 if lon >= 76.5 else 7
            elif found_dist == "Hassan":
                zone_id = 9 if lon < 75.8 else 7
            elif found_dist == "Chikkamagaluru":
                zone_id = 9 if lon < 75.8 else 7
            elif found_dist == "Shivamogga":
                zone_id = 9 if lon < 75.5 else 7
            elif found_dist in ["Dharwad", "Haveri"]:
                zone_id = 8
            elif found_dist == "Kodagu":
                zone_id = 9
            elif found_dist in ["Dakshina Kannada", "Udupi"]:
                zone_id = 10
            elif found_dist == "Uttara Kannada":
                zone_id = 10 if lon < 74.5 else 9

            records.append({
                "lat_idx": i,
                "lon_idx": j,
                "lat": float(lat),
                "lon": float(lon),
                "district": found_dist,
                "zone_id": zone_id,
                "zone_name": AGRO_ZONES[zone_id]["name"]
            })

    df_grid = pd.DataFrame(records)
    print(f"  [OK] Successfully classified {len(df_grid)} grid cells across Karnataka's 10 Agro-Climatic Zones.")
    return df_grid


def compute_hydrometeorological_metrics(obs, sim, threshold=0.1):
    """Compute verification statistics between observed and simulated rainfall."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[valid]
    s = sim[valid]
    if len(o) < 10:
        return {}

    n = len(o)
    mean_o = float(np.mean(o))
    mean_s = float(np.mean(s))
    bias = mean_s - mean_o
    pbias = (bias / mean_o * 100.0) if mean_o > 1e-4 else 0.0
    mae = float(np.mean(np.abs(s - o)))
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))

    # Pearson correlation
    std_o = np.std(o)
    std_s = np.std(s)
    if std_o > 1e-6 and std_s > 1e-6:
        r = float(np.corrcoef(o, s)[0, 1])
    else:
        r = 0.0

    # Kling-Gupta Efficiency (KGE)
    if mean_o > 1e-6 and std_o > 1e-6:
        beta = mean_s / mean_o
        gamma = (std_s / mean_s) / (std_o / mean_o) if mean_s > 1e-6 else 1.0
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))
    else:
        kge = -999.0

    return {
        "n": n,
        "mean_obs": mean_o,
        "mean_sim": mean_s,
        "bias": bias,
        "pbias_pct": pbias,
        "mae": mae,
        "rmse": rmse,
        "r": r,
        "kge": kge
    }


def analyze_seasonal_shifts(df_grid, imd_rain, era_true, times):
    """
    Compute SW Monsoon (JJAS) and NE Monsoon (OND) rainfall metrics,
    accuracy, and shift dynamics for each of the 10 zones.
    """
    print("\n" + "=" * 75)
    print("STEP: COMPUTING 10-ZONE SEASONAL DYNAMICS (SW JJAS vs NE OND: 2019–2025)")
    print("=" * 75)

    months = times.month
    years = times.year

    sw_mask = np.isin(months, [6, 7, 8, 9])
    ne_mask = np.isin(months, [10, 11, 12])

    zonal_summary = []
    zone_yearly_series = {}

    for zid in range(1, 11):
        zinfo = AGRO_ZONES[zid]
        zcells = df_grid[df_grid["zone_id"] == zid]
        lat_idx = zcells["lat_idx"].values
        lon_idx = zcells["lon_idx"].values
        n_cells = len(zcells)
        member_districts = ", ".join(sorted(zcells["district"].unique()))

        # Extract timeseries for member cells: (N_days, N_cells)
        obs_z = imd_rain[:, lat_idx, lon_idx]  # shape: (2557, n_cells)
        era_z = era_true[:, lat_idx, lon_idx]

        # Mean over zone cells per day: shape (2557,)
        obs_daily = np.nanmean(obs_z, axis=1)
        era_daily = np.nanmean(era_z, axis=1)

        # 1. Full Period Seasonal Metrics
        # SW Monsoon
        sw_obs = obs_daily[sw_mask]
        sw_era = era_daily[sw_mask]
        sw_metrics = compute_hydrometeorological_metrics(sw_obs, sw_era)
        sw_total_obs_mean = float(np.sum(sw_obs) / len(np.unique(years)))
        sw_total_era_mean = float(np.sum(sw_era) / len(np.unique(years)))

        # NE Monsoon
        ne_obs = obs_daily[ne_mask]
        ne_era = era_daily[ne_mask]
        ne_metrics = compute_hydrometeorological_metrics(ne_obs, ne_era)
        ne_total_obs_mean = float(np.sum(ne_obs) / len(np.unique(years)))
        ne_total_era_mean = float(np.sum(ne_era) / len(np.unique(years)))

        # Annual
        annual_obs_mean = float(np.sum(obs_daily) / len(np.unique(years)))
        annual_era_mean = float(np.sum(era_daily) / len(np.unique(years)))

        # 2. Inter-annual Shift Analysis: Early (2019-2021) vs Late (2022-2025)
        early_mask = np.isin(years, [2019, 2020, 2021])
        late_mask = np.isin(years, [2022, 2023, 2024, 2025])

        sw_early = np.sum(obs_daily[sw_mask & early_mask]) / 3.0
        sw_late = np.sum(obs_daily[sw_mask & late_mask]) / 4.0
        sw_shift_pct = float((sw_late - sw_early) / sw_early * 100.0) if sw_early > 0 else 0.0

        ne_early = np.sum(obs_daily[ne_mask & early_mask]) / 3.0
        ne_late = np.sum(obs_daily[ne_mask & late_mask]) / 4.0
        ne_shift_pct = float((ne_late - ne_early) / ne_early * 100.0) if ne_early > 0 else 0.0

        # 3. Agronomic Risk Indicators
        # Rainy days (>= 2.5 mm) per year in SW and NE
        sw_rainy_days = float(np.sum(obs_daily[sw_mask] >= 2.5) / len(np.unique(years)))
        ne_rainy_days = float(np.sum(obs_daily[ne_mask] >= 2.5) / len(np.unique(years)))

        # Heavy rain days (>= 64.5 mm) per year
        heavy_days = float(np.sum(obs_daily >= 64.5) / len(np.unique(years)))

        # Mid-Monsoon (July-August) Dry Spell Metric:
        # Max consecutive days with daily rain < 1.0 mm during July-August
        ja_mask = np.isin(months, [7, 8])
        ja_obs = obs_daily[ja_mask]
        dry_runs = []
        cur_run = 0
        for val in ja_obs:
            if val < 1.0:
                cur_run += 1
            else:
                if cur_run > 0:
                    dry_runs.append(cur_run)
                cur_run = 0
        if cur_run > 0:
            dry_runs.append(cur_run)
        max_dry_spell = max(dry_runs) if dry_runs else 0

        # Ratio of SW to NE rainfall (Kharif vs Rabi dependence)
        sw_to_ne_ratio = (sw_total_obs_mean / ne_total_obs_mean) if ne_total_obs_mean > 0 else 999.0

        zonal_summary.append({
            "zone_id": zid,
            "zone_name": zinfo["name"],
            "hq": zinfo["hq"],
            "grid_cells": n_cells,
            "districts": member_districts,
            "taluks_sample": zinfo["taluks"],
            "major_crops": zinfo["major_crops"],
            "annual_obs_mm": round(annual_obs_mean, 1),
            "annual_era_mm": round(annual_era_mean, 1),
            "annual_bias_mm": round(annual_era_mean - annual_obs_mean, 1),
            "sw_obs_mm": round(sw_total_obs_mean, 1),
            "sw_era_mm": round(sw_total_era_mean, 1),
            "sw_bias_mm": round(sw_total_era_mean - sw_total_obs_mean, 1),
            "sw_r": round(sw_metrics.get("r", 0.0), 3),
            "sw_kge": round(sw_metrics.get("kge", 0.0), 3),
            "sw_shift_pct": round(sw_shift_pct, 1),
            "sw_rainy_days": round(sw_rainy_days, 1),
            "ne_obs_mm": round(ne_total_obs_mean, 1),
            "ne_era_mm": round(ne_total_era_mean, 1),
            "ne_bias_mm": round(ne_total_era_mean - ne_total_obs_mean, 1),
            "ne_r": round(ne_metrics.get("r", 0.0), 3),
            "ne_kge": round(ne_metrics.get("kge", 0.0), 3),
            "ne_shift_pct": round(ne_shift_pct, 1),
            "ne_rainy_days": round(ne_rainy_days, 1),
            "sw_to_ne_ratio": round(sw_to_ne_ratio, 2),
            "max_dry_spell_days": int(max_dry_spell),
            "heavy_days_per_yr": round(heavy_days, 2)
        })

        # Year-by-year totals
        yr_records = []
        for yr in np.unique(years):
            yr_m = (years == yr)
            sw_yr = float(np.sum(obs_daily[yr_m & sw_mask]))
            ne_yr = float(np.sum(obs_daily[yr_m & ne_mask]))
            ann_yr = float(np.sum(obs_daily[yr_m]))
            yr_records.append({
                "year": int(yr),
                "annual_mm": round(ann_yr, 1),
                "sw_mm": round(sw_yr, 1),
                "ne_mm": round(ne_yr, 1)
            })
        zone_yearly_series[zid] = yr_records

    df_summary = pd.DataFrame(zonal_summary)

    # Export to CSV and JSON
    summary_csv = CSV_DIR / "karnataka_10_agroclimatic_zones_summary.csv"
    summary_json = OUTPUT_DIR / "karnataka_10_agroclimatic_zones_metrics.json"

    df_summary.to_csv(summary_csv, index=False)
    print(f"  [OK] Exported summary CSV: {summary_csv}")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "zones_summary": zonal_summary,
            "yearly_series": zone_yearly_series
        }, f, indent=2)
    print(f"  [OK] Exported summary JSON: {summary_json}")

    return df_summary, zone_yearly_series


def generate_agroclimatic_visualizations(df_grid, df_summary, zone_yearly_series):
    """Generate high-impact diagnostic figures for the 10 zones and seasonal shifts."""
    print("\n" + "=" * 75)
    print("STEP: GENERATING 10 AGRO-CLIMATIC ZONE VISUALIZATIONS")
    print("=" * 75)

    # -------------------------------------------------------------
    # Figure A: Spatial Map of 10 Agro-Climatic Zones over Karnataka
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 10), dpi=300)
    
    # Plot district boundaries
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    for feat in geojson_data["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                poly = np.array(ring)
                ax.plot(poly[:, 0], poly[:, 1], color="#7f8c8d", linewidth=0.7, linestyle="--")
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    p_arr = np.array(ring)
                    ax.plot(p_arr[:, 0], p_arr[:, 1], color="#7f8c8d", linewidth=0.7, linestyle="--")

    # Plot grid points colored by zone
    for zid in range(1, 11):
        zcells = df_grid[df_grid["zone_id"] == zid]
        zcolor = AGRO_ZONES[zid]["color"]
        zname = f"Zone {zid}: {AGRO_ZONES[zid]['name']}"
        ax.scatter(
            zcells["lon"], zcells["lat"],
            s=48, color=zcolor, edgecolors="#111111", linewidths=0.5,
            label=zname, zorder=5
        )

    ax.set_xlim(73.8, 78.8)
    ax.set_ylim(11.3, 18.8)
    ax.set_xlabel("Longitude (°E)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontweight="bold", fontsize=11)
    ax.set_title("Official 10 Agro-Climatic Zones of Karnataka (ICAR / UAS Standards)\nCollocated 0.25° Grid Verification Network (252 In-State Grid Stations)", fontweight="bold", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92, facecolor="#ffffff", edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig_map_path = PLOTS_DIR / "07_agroclimatic_10_zones_map.png"
    plt.tight_layout()
    fig.savefig(fig_map_path)
    plt.close(fig)
    print(f"  [OK] Saved Figure 7: {fig_map_path.name}")

    # -------------------------------------------------------------
    # Figure B: Seasonal Rainfall Breakdown (SW vs NE Monsoon across 10 Zones)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    zones_labels = [f"Z{z['zone_id']}: {z['zone_name'][:18]}..." for _, z in df_summary.iterrows()]
    x = np.arange(len(df_summary))
    w = 0.38

    # Left: SW vs NE Rainfall Magnitude
    axes[0].bar(x - w/2, df_summary["sw_obs_mm"], w, label="SW Monsoon (JJAS: Kharif)", color="#2980b9", edgecolor="#1c5980")
    axes[0].bar(x + w/2, df_summary["ne_obs_mm"], w, label="NE Monsoon (OND: Rabi)", color="#e67e22", edgecolor="#a85914")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(zones_labels, rotation=40, ha="right", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Mean Seasonal Rainfall (mm)", fontweight="bold", fontsize=10.5)
    axes[0].set_title("(a) Mean Seasonal Rainfall: SW (Kharif) vs NE (Rabi) Monsoons", fontweight="bold", fontsize=11.5)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Right: Shift in Rainfall Patterns (2019-2021 vs 2022-2025 % change)
    colors_sw = ["#27ae60" if v >= 0 else "#c0392b" for v in df_summary["sw_shift_pct"]]
    colors_ne = ["#2980b9" if v >= 0 else "#d35400" for v in df_summary["ne_shift_pct"]]

    axes[1].bar(x - w/2, df_summary["sw_shift_pct"], w, label="SW Monsoon Shift (%)", color="#27ae60", alpha=0.85)
    axes[1].bar(x + w/2, df_summary["ne_shift_pct"], w, label="NE Monsoon Shift (%)", color="#9b59b6", alpha=0.85)
    axes[1].axhline(0, color="#333333", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(zones_labels, rotation=40, ha="right", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Rainfall Shift (%) [2022–2025 vs 2019–2021]", fontweight="bold", fontsize=10.5)
    axes[1].set_title("(b) Recent Shift in Seasonal Precipitation (%)", fontweight="bold", fontsize=11.5)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Karnataka Agro-Climatic Zones: Seasonal Rainfall Dynamics & Recent Shifts (2019–2025)", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    fig_seasonal_path = PLOTS_DIR / "08_seasonal_shifts_sw_vs_ne_monsoon.png"
    fig.savefig(fig_seasonal_path)
    plt.close(fig)
    print(f"  [OK] Saved Figure 8: {fig_seasonal_path.name}")

    # -------------------------------------------------------------
    # Figure C: Model Skill & Agronomic Vulnerability Indicators
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)

    # Left: ERA5 Accuracy (KGE) in SW vs NE Monsoons
    axes[0].plot(x, df_summary["sw_kge"], marker="o", linewidth=2.2, color="#2980b9", label="SW Monsoon Skill (KGE)")
    axes[0].plot(x, df_summary["ne_kge"], marker="s", linewidth=2.2, color="#e67e22", label="NE Monsoon Skill (KGE)")
    axes[0].axhline(0.4, color="#e74c3c", linestyle=":", label="Satisfactory Hydrologic Threshold (0.4)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Z{i}" for i in range(1, 11)], fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Kling-Gupta Efficiency (KGE)", fontweight="bold", fontsize=10.5)
    axes[0].set_title("(a) ERA5-Land Hydrometeorological Skill Across Zones", fontweight="bold", fontsize=11.5)
    axes[0].legend(loc="lower left", fontsize=9.5)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Right: Mid-Monsoon Dry Spell Duration & Rainy Days
    ax2 = axes[1].twinx()
    b1 = axes[1].bar(x - w/2, df_summary["sw_rainy_days"], w, color="#3498db", label="SW Rainy Days/Yr", alpha=0.85)
    b2 = axes[1].bar(x + w/2, df_summary["ne_rainy_days"], w, color="#f39c12", label="NE Rainy Days/Yr", alpha=0.85)
    l1 = ax2.plot(x, df_summary["max_dry_spell_days"], color="#c0392b", marker="^", linewidth=2.5, label="Max July–Aug Dry Spell (Days)")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Z{i}" for i in range(1, 11)], fontweight="bold", fontsize=10)
    axes[1].set_ylabel("Average Rainy Days per Season (Days)", fontweight="bold", fontsize=10.5)
    ax2.set_ylabel("Max Consecutive Dry Spell (Days)", fontweight="bold", color="#c0392b", fontsize=10.5)
    axes[1].set_title("(b) Agronomic Moisture Risk: Rainy Days vs. Mid-Monsoon Dry Spells", fontweight="bold", fontsize=11.5)
    
    # Combined legend
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8.5)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("ERA5 Reanalysis Reliability & Agronomic Risk across Karnataka Agro-Climatic Zones", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    fig_risk_path = PLOTS_DIR / "09_agronomic_moisture_risk_and_model_skill.png"
    fig.savefig(fig_risk_path)
    plt.close(fig)
    print(f"  [OK] Saved Figure 9: {fig_risk_path.name}")


def main():
    print("=" * 75)
    print("STARTING 10 AGRO-CLIMATIC ZONAL & SEASONAL SHIFT PIPELINE")
    print("=" * 75)

    # Load data
    import imdlib as imd
    imd_obj = imd.open_data('rain', 2019, 2025, 'yearwise', str(IMD_DIR))
    ds = imd_obj.get_xarray()
    ds_kar = ds.sel(lat=slice(11.0, 19.0), lon=slice(74.0, 79.0))
    lats = ds_kar.lat.values
    lons = ds_kar.lon.values
    times = pd.to_datetime(ds_kar.time.values)
    rain_obs = ds_kar.rain.values
    rain_obs = np.where((rain_obs < 0) | (rain_obs > 1000.0), np.nan, rain_obs)
    land_mask = ~np.isnan(rain_obs).all(axis=0)

    # Load collocated ERA5
    era_data = np.load(CACHE_PATH)
    era_true = era_data["era_on_imd"]

    # 1. Spatial Delineation
    df_grid = load_spatial_delineation(lats, lons, land_mask)

    # 2. Seasonal Shifts and Accuracy
    df_summary, zone_yearly_series = analyze_seasonal_shifts(df_grid, rain_obs, era_true, times)

    # 3. Visualizations
    generate_agroclimatic_visualizations(df_grid, df_summary, zone_yearly_series)

    print("\n" + "=" * 75)
    print("10 AGRO-CLIMATIC ZONES PIPELINE SUCCESSFULLY COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    main()

