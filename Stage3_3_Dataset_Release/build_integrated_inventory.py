import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
p_int = base_dir / "Stage3_3_Dataset_Release/integrated_stage3_dataset.parquet"
df = pd.read_parquet(p_int)

dtypes = df.dtypes.to_dict()
rows = []

for col in df.columns:
    dt = str(dtypes[col])
    
    # 1. Identifiers
    if col in ["date", "station_id", "district", "zone_name"]:
        cat = "Identifier"
        src = "IMD Station Metadata & Calendar"
        avail = "Yes (Known a priori)"
        elig = "No"
        desc = "Administrative index / grouping label; non-generalizable spatial or temporal identifier."
        
    # 2. Protocol and Metadata Fields
    elif col in ["prediction_issue_time", "target_period"]:
        cat = "Protocol and Metadata Field"
        src = "Experimental Protocol Specification"
        avail = "Yes (Protocol definition)"
        elig = "No"
        desc = "Audit trail text describing target accumulation window and forecast issue time (03:00 UTC)."
        
    # 3. Supervised Targets
    elif col in ["target_rainfall_mm", "target_is_rain", "target_is_heavy_rain"]:
        cat = "Supervised Target"
        src = "IMD 0.25 deg Daily Gridded Rainfall"
        avail = "No (Future ground truth)"
        elig = "No"
        desc = "Prediction target variable; strictly quarantined from feature matrix X."
        
    # 4. Quality Flags
    elif col in ["is_coastal_fallback", "interpolation_method"]:
        cat = "Quality Flag"
        src = "Stage 2 Quality Control"
        avail = "Yes (Static grid property)"
        if col == "is_coastal_fallback":
            elig = "Yes (Optional Auxiliary Feature)"
            desc = "Binary flag (0/1) marking station grid cells interpolated via IDW due to ERA5 land-sea boundary masking."
        else:
            elig = "No"
            desc = "Categorical string redundant with binary is_coastal_fallback."
            
    # 5. Excluded Columns
    elif col in ["season"]:
        cat = "Excluded Column"
        src = "Meteorological Calendar Definition"
        avail = "Yes (Calendar lookup)"
        elig = "No"
        desc = "Season name string; redundant with month and cyclical harmonic ephemeris features."
        
    # 6. Pressure Proxies (Renamed)
    elif col in ["elevation_pressure_proxy_m", "slope_pressure_proxy_deg"]:
        cat = "Atmospheric Pressure Proxy"
        src = "ERA5-Land Surface Pressure Hypsometric Proxy"
        avail = "Yes (Observation window ending 03:00 UTC Day T-1)"
        elig = "No (Ablation / Auxiliary Only)"
        desc = "Hypsometric elevation/slope derived from multi-year mean surface pressure. Retained for transparency; excluded from primary model terrain features."
        
    # 7. Verified DEM Topography
    elif col in ["elevation_dem_m", "slope_dem_deg"]:
        cat = "Verified DEM Topography"
        src = "CGIAR-CSI SRTM v4.1 / Copernicus DEM GLO-90"
        avail = "Yes (Static station geography)"
        elig = "Yes"
        desc = "Satellite radar digital elevation model point elevation at 0.25 deg centroid and 0.25 deg regional macro-topographic slope in degrees."
        
    # 8. Categorical Predictors
    elif col in ["zone_id", "subdivision"]:
        cat = "Categorical Predictor"
        src = "IMD Agro-Climatic & Subdivision Cartography"
        avail = "Yes (Static station geography)"
        elig = "Yes"
        desc = "Macro-climatological grouping (Coastal, Malnad, North Interior, South Interior) and agro-climatic zones (1-10)."
        
    # 9. Numerical Spatial Geometry
    elif col in ["latitude", "longitude", "dist_to_coast_km", "dist_to_ghats_crest_km"]:
        cat = "Spatial Geometry"
        src = "Geographical Coordinates & Cartography"
        avail = "Yes (Static station geography)"
        elig = "Yes"
        desc = "Geographical coordinates and Euclidean geodesic distances to maritime boundary and Western Ghats crest."
        
    # 10. Temporal Ephemeris
    elif col in ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "is_monsoon"]:
        cat = "Temporal Ephemeris"
        src = "Astronomical / Solar Ephemeris Calendar"
        avail = "Yes (Known at forecast issue time)"
        elig = "Yes"
        desc = "Harmonic cyclical sinusoidal/cosinusoidal encodings of the annual solar cycle and monsoonal seasonality."
        
    # 11. Autoregressive IMD Rainfall Memory
    elif "target_rain_lag" in col or "target_rain_roll" in col or col == "api_14d":
        cat = "IMD Rainfall Memory"
        src = "IMD Observed Daily Rainfall (Retrospective)"
        avail = "Yes (Retrospective: finalized at 03:00 UTC Day T-1)"
        elig = "Yes"
        desc = "Autoregressive observed rainfall lags (1d, 2d, 3d, 7d, 14d), rolling sums/means/max, and 14-day antecedent precipitation index (decay k=0.85)."
        
    # 12. ERA5 Precipitation Memory
    elif "era5_tp" in col:
        cat = "ERA5 Precipitation Memory"
        src = "Corrected ERA5-Land Hourly Precipitation (IMD-Aligned)"
        avail = "Yes (Observation window ending 03:00 UTC Day T-1)"
        elig = "Yes"
        desc = "Physical numerical reanalysis precipitation lags (1d, 2d, 3d) and rolling aggregates (3d, 7d sums)."
        
    # 13. ERA5 Atmospheric State & Land Surface Fluxes
    else:
        cat = "Meteorological State / Flux"
        src = "Corrected ERA5-Land Hourly State & Fluxes"
        avail = "Yes (Observation window ending 03:00 UTC Day T-1)"
        elig = "Yes"
        desc = "Thermodynamic state (t2m, d2m, sp), wind field dynamics, soil moisture layers (swvl1-4), radiation, and surface water balance fluxes."
        
    rows.append({
        "column_name": col,
        "data_type": dt,
        "category": cat,
        "source": src,
        "available_at_issue_time": avail,
        "eligible_for_model_training": elig,
        "description": desc
    })

df_inv = pd.DataFrame(rows)
out_csv = base_dir / "Stage3_3_Dataset_Release/integrated_feature_inventory.csv"
df_inv.to_csv(out_csv, index=False)
print(f"Saved integrated_feature_inventory.csv with {len(df_inv)} columns.")
print("\nBreakdown by Category:")
print(df_inv["category"].value_counts())
print("\nBreakdown by Eligibility:")
print(df_inv["eligible_for_model_training"].value_counts())

