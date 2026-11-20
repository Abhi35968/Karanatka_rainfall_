import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_0_Pretraining_Verification"

df = pd.read_parquet(data_dir / "final_stage3_dataset.parquet")
columns = list(df.columns)
dtypes = df.dtypes.to_dict()

# Target columns to strictly exclude
target_cols = ["target_rainfall_mm", "target_is_rain", "target_is_heavy_rain", "target_rain_ge_35_5mm"]

# Identifiers
id_cols = ["date", "station_id", "district", "zone_name"]

# Protocol strings
protocol_cols = ["prediction_issue_time", "target_period"]

# Quality / Redundant strings
excluded_strings = ["interpolation_method", "season"]

# Pressure proxies
proxy_cols = ["elevation_pressure_proxy_m", "slope_pressure_proxy_deg"]

# Auxiliary feature
aux_cols = ["is_coastal_fallback"]

# Categorical predictors
cat_cols = ["subdivision", "zone_id"]

# Build primary predictors list
non_primary = set(target_cols + id_cols + protocol_cols + excluded_strings + proxy_cols + aux_cols)
primary_predictors = [c for c in columns if c not in non_primary]

print("--- Feature Matrix Reconstruction ---")
print(f"Total columns in Parquet: {len(columns)}")
print(f"Primary predictors count: {len(primary_predictors)}")
print(f"Auxiliary features count: {len(aux_cols)}")
print(f"Atmospheric proxies count: {len(proxy_cols)}")
print(f"Target columns count: {len(target_cols)}")
print(f"Identifier columns count: {len(id_cols)}")
print(f"Protocol strings count: {len(protocol_cols)}")
print(f"Excluded strings count: {len(excluded_strings)}")

assert len(primary_predictors) == 63, f"Expected 63 primary predictors, got {len(primary_predictors)}"
assert "elevation_dem_m" in primary_predictors, "elevation_dem_m must be in primary predictors!"
assert "slope_dem_deg" in primary_predictors, "slope_dem_deg must be in primary predictors!"
for t in target_cols:
    assert t not in primary_predictors, f"Target column {t} leaked into primary predictors!"
for p in proxy_cols:
    assert p not in primary_predictors, f"Pressure proxy {p} leaked into primary predictors!"

# Build detailed manifest table
manifest_rows = []
for c in columns:
    dt = str(dtypes[c])
    
    # Categorization and Selection
    if c in target_cols:
        status = "Excluded (Target Column)"
        cat = "Supervised Target"
        units = "mm or binary (0/1)"
        avail = "Future Ground Truth (03:00 UTC Day T)"
        rule = "Strictly quarantined from X. Used exclusively as supervised training/evaluation target."
        if c == "target_rainfall_mm":
            desc = "Continuous rainfall accumulation (mm) over 03:00-03:00 UTC window; regression target."
        elif c == "target_is_rain":
            desc = "Binary wet-day indicator (>=0.1 mm); Stage 1 Hurdle occurrence target."
        elif c == "target_is_heavy_rain":
            desc = "Official IMD heavy rainfall binary indicator (>=64.5 mm); extreme event target."
        else:
            desc = "Custom diagnostic intermediate rainfall binary indicator (>=35.5 mm)."
            
    elif c in id_cols:
        status = "Excluded (Identifier / Label)"
        cat = "Administrative Metadata"
        units = "Categorical / Datetime"
        avail = "Known a priori"
        rule = "Excluded from numerical feature matrix X. Used for grouping, indexing, and spatial mapping."
        desc = "Administrative or calendar index (date, station_id, district, zone_name)."
        
    elif c in protocol_cols:
        status = "Excluded (Protocol Metadata)"
        cat = "Administrative Metadata"
        units = "Text string"
        avail = "Protocol Specification"
        rule = "Excluded from X. Human-readable audit metadata describing issue cutoff and target window."
        desc = "Forecast window / issue cutoff description."
        
    elif c in excluded_strings:
        status = "Excluded (Redundant String)"
        cat = "Data Lineage / Temporal"
        units = "Text string"
        avail = "Known at Issue Time"
        rule = "Excluded from X. Redundant with numerical/cyclical features."
        desc = "Redundant text string (season or interpolation_method)."
        
    elif c in proxy_cols:
        status = "Excluded (Atmospheric Pressure Proxy)"
        cat = "Atmospheric Pressure Proxy"
        units = "m or degrees"
        avail = "Finalized 03:00 UTC Day T-1"
        rule = "Excluded from primary X. Reserved exclusively for diagnostic ablation studies."
        desc = "Hypsometric elevation or slope derived from surface pressure; not true terrain height."
        
    elif c in aux_cols:
        status = "Selected (Auxiliary Quality Feature)"
        cat = "Data Lineage Flag"
        units = "Binary (0/1)"
        avail = "Static Grid Property"
        rule = "Pass-through binary flag (0/1). Optional input for coastal land-sea sensitivity analysis."
        desc = "Indicator marking coastal cells requiring IDW interpolation due to land-sea masking."
        
    elif c in cat_cols:
        status = "Selected (Primary Predictor)"
        cat = "Topography & Climate Zone"
        units = "Categorical / Integer code"
        avail = "Static Grid Property"
        rule = "Categorical encoding (TargetEncoder or OneHotEncoder) fitted strictly on Training partition."
        desc = "Macro-climatological subdivision (4 classes) or agro-climatic zone ID (1-10)."
        
    elif c in ["elevation_dem_m", "slope_dem_deg"]:
        status = "Selected (Primary Predictor)"
        cat = "Verified DEM Topography"
        units = "m or degrees"
        avail = "Static Grid Property"
        rule = "Continuous numerical predictor. Standardized using StandardScaler fitted exclusively on Training partition."
        desc = "Copernicus GLO-90 DEM elevation at centroid (m) or regional macro-slope (degrees)."
        
    elif c in ["latitude", "longitude", "dist_to_coast_km", "dist_to_ghats_crest_km"]:
        status = "Selected (Primary Predictor)"
        cat = "Spatial Geometry"
        units = "degrees or km"
        avail = "Static Grid Property"
        rule = "Continuous spatial metric. Standardized using StandardScaler fitted on Training partition only."
        desc = "Spatial coordinates and geodesic distances to maritime boundary and Western Ghats crest."
        
    elif c in ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "is_monsoon"]:
        status = "Selected (Primary Predictor)"
        cat = "Temporal Ephemeris"
        units = "Cyclical harmonic / integer"
        avail = "Known at Issue Time"
        rule = "Direct numerical feature. No leakage risk; encapsulates seasonal solar ephemeris."
        desc = "Harmonic cyclical calendar feature modeling solar insolation cycle and monsoon season."
        
    elif "target_rain_lag" in c or "target_rain_roll" in c or c == "api_14d":
        status = "Selected (Primary Predictor)"
        cat = "IMD Rainfall Memory"
        units = "mm"
        avail = "Finalized 03:00 UTC Day T-1 (Retrospective)"
        rule = "Continuous rainfall memory. Power-transformed (log/Box-Cox) or scaled via StandardScaler on Train only."
        desc = "Autoregressive observed rainfall lags (1d, 2d, 3d, 7d, 14d), rolling sums/means/max, and API-14."
        
    elif "era5_tp" in c:
        status = "Selected (Primary Predictor)"
        cat = "ERA5 Precip Memory"
        units = "mm"
        avail = "Accumulation ending 03:00 UTC Day T-1"
        rule = "Continuous precipitation memory. Scaled using parameters fitted on Training partition only."
        desc = "Numerical reanalysis precipitation history (1d, 2d, 3d lags and 3d, 7d rolling sums)."
        
    elif any(s in c for s in ["swvl1", "swvl2", "swvl3"]):
        status = "Selected (Primary Predictor)"
        cat = "Soil Moisture Profile"
        units = "m3/m3"
        avail = "Observation window ending 03:00 UTC Day T-1"
        rule = "Continuous soil moisture. Standardized using StandardScaler fitted on Training partition only."
        if "swvl1" in c:
            desc = f"Volumetric soil water layer 1 (0-7 cm depth), {c.split('_')[-2]} {c.split('_')[-1]} (active surface layer)."
        elif "swvl2" in c:
            desc = "Volumetric soil water layer 2 (7-28 cm depth), lag 1 day (rooting zone)."
        else:
            desc = "Volumetric soil water layer 3 (28-100 cm depth), lag 1 day (deep root zone; layer 4 100-289 cm excluded)."
            
    else:
        status = "Selected (Primary Predictor)"
        cat = "Atmospheric State & Land Fluxes"
        units = "K, C, J/m2, m, or m/s"
        avail = "Observation window ending 03:00 UTC Day T-1"
        rule = "Continuous thermodynamic / kinematic state or flux. Standardized using StandardScaler fitted on Train only."
        desc = f"Corrected ERA5-Land state variable or surface flux ({c})."
        
    manifest_rows.append({
        "column_name": c,
        "data_type": dt,
        "selection_status": status,
        "predictor_category": cat,
        "units": units,
        "availability_issue_time": avail,
        "pipeline_handling": rule,
        "feature_description": desc
    })

df_manifest = pd.DataFrame(manifest_rows)
out_manifest = out_dir / "final_feature_matrix_manifest.csv"
df_manifest.to_csv(out_manifest, index=False)
print(f"Saved final_feature_matrix_manifest.csv with {len(df_manifest)} entries.")
print("\nBreakdown by Selection Status:")
print(df_manifest["selection_status"].value_counts().to_string())

