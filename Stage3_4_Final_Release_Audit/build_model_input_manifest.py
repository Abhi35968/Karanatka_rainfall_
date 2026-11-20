import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
p_full = base_dir / "Stage3_4_Final_Release_Audit/data/final_stage3_dataset.parquet"
df = pd.read_parquet(p_full)

dtypes = df.dtypes.to_dict()
manifest = []

for col in df.columns:
    dt = str(dtypes[col])
    
    # 1. Targets
    if col in ["target_rainfall_mm", "target_is_rain", "target_is_heavy_rain", "target_rain_ge_35_5mm"]:
        role = "Target Label"
        elig = "No"
        grp = "Target"
        avail = "Future Ground Truth (03:00 UTC Day T)"
        if col == "target_rainfall_mm":
            handling = "Continuous regression target y; strictly quarantined from feature matrix X."
        elif col == "target_is_rain":
            handling = "Binary classification target for wet-day occurrence (>=0.1 mm); Stage 1 Hurdle target."
        elif col == "target_is_heavy_rain":
            handling = "Official IMD heavy rainfall binary target (>=64.5 mm); extreme event evaluation metric."
        else:
            handling = "Custom intermediate diagnostic binary target (>=35.5 mm); preserved for auxiliary evaluation."
            
    # 2. Identifiers
    elif col in ["date", "station_id", "district", "zone_name"]:
        role = "Identifier"
        elig = "No"
        grp = "Administrative"
        avail = "Static / Known a priori"
        handling = "Non-generalizable index / administrative grouping; excluded from numerical feature matrix X."
        
    # 3. Protocol Strings
    elif col in ["prediction_issue_time", "target_period"]:
        role = "Protocol Metadata"
        elig = "No"
        grp = "Administrative"
        avail = "Protocol Specification"
        handling = "Audit trail text string; strictly excluded from feature matrix X."
        
    # 4. Excluded String
    elif col == "season":
        role = "Excluded String"
        elig = "No"
        grp = "Temporal Ephemeris"
        avail = "Calendar Lookup"
        handling = "Redundant season name string; excluded from X (represented numerically via cyclical harmonic encodings)."
        
    # 5. Quality Flags
    elif col in ["interpolation_method", "is_coastal_fallback"]:
        if col == "interpolation_method":
            role = "Quality Flag (String)"
            elig = "No"
            grp = "Data Lineage"
            avail = "Static Grid Property"
            handling = "String redundant with binary is_coastal_fallback; excluded from X."
        else:
            role = "Auxiliary Feature (Quality Flag)"
            elig = "Yes (Auxiliary Feature)"
            grp = "Data Lineage"
            avail = "Static Grid Property"
            handling = "Binary flag (0/1) for land-sea boundary masking; included in tree-based models as auxiliary predictor."
            
    # 6. Pressure Proxies (Renamed)
    elif col in ["elevation_pressure_proxy_m", "slope_pressure_proxy_deg"]:
        role = "Atmospheric Proxy (Ablation Only)"
        elig = "No (Ablation Only)"
        grp = "Atmospheric Pressure Proxy"
        avail = "Finalized 03:00 UTC Day T-1"
        handling = "Derived from multi-year surface pressure; excluded from primary feature set to prevent confusion with true DEM terrain."
        
    # 7. Verified DEM Topography
    elif col in ["elevation_dem_m", "slope_dem_deg"]:
        role = "Primary Predictor (Numerical)"
        elig = "Yes"
        grp = "Topography & Spatial"
        avail = "Static Grid Property"
        handling = "Standardized via StandardScaler fitted on Train partition only; represents physical ground height and 0.25 deg macro-gradient."
        
    # 8. Categorical Predictors
    elif col in ["zone_id", "subdivision"]:
        role = "Primary Predictor (Categorical)"
        elig = "Yes"
        grp = "Topography & Spatial"
        avail = "Static Grid Property"
        handling = "Target-encoded or one-hot encoded using statistics fitted exclusively on Train partition."
        
    # 9. Spatial Geometry
    elif col in ["latitude", "longitude", "dist_to_coast_km", "dist_to_ghats_crest_km"]:
        role = "Primary Predictor (Numerical)"
        elig = "Yes"
        grp = "Topography & Spatial"
        avail = "Static Grid Property"
        handling = "Continuous spatial coordinates/distances; standardized using StandardScaler fitted on Train only."
        
    # 10. Temporal Ephemeris
    elif col in ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "is_monsoon"]:
        role = "Primary Predictor (Numerical)"
        elig = "Yes"
        grp = "Temporal Ephemeris"
        avail = "Known at Issue Time"
        handling = "Smooth cyclical harmonic features; passed directly into tree/linear models without leakage."
        
    # 11. Autoregressive IMD Rainfall Memory
    elif "target_rain_lag" in col or "target_rain_roll" in col or col == "api_14d":
        role = "Primary Predictor (Numerical)"
        elig = "Yes"
        grp = "IMD Rainfall Memory"
        avail = "Finalized 03:00 UTC Day T-1 (Retrospective)"
        handling = "Autoregressive rainfall memory; power-transformed or standardized using parameters fitted on Train only."
        
    # 12. ERA5 Precip Memory
    elif "era5_tp" in col:
        role = "Primary Predictor (Numerical)"
        elig = "Yes"
        grp = "ERA5 Precip Memory"
        avail = "Accumulation ending 03:00 UTC Day T-1"
        handling = "Numerical reanalysis precipitation history; scaled using parameters fitted on Train only."
        
    # 13. ERA5 Atmospheric State & Land Surface Fluxes
    else:
        role = "Primary Predictor (Numerical)"
        elig = "Yes"
        grp = "Atmospheric State & Land Fluxes"
        avail = "Observation window ending 03:00 UTC Day T-1"
        handling = "Continuous meteorological fluxes; standardized using StandardScaler fitted on Train only."
        
    manifest.append({
        "column_name": col,
        "role_in_dataset": role,
        "eligible_for_X_primary": elig,
        "feature_group": grp,
        "data_type": dt,
        "availability_timestamp": avail,
        "pipeline_handling": handling
    })

df_man = pd.DataFrame(manifest)
out_csv = base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv"
df_man.to_csv(out_csv, index=False)
print(f"Saved final_model_input_manifest.csv with {len(df_man)} entries.")
print("\nBreakdown by Role:")
print(df_man["role_in_dataset"].value_counts())
print("\nBreakdown by Primary Eligibility:")
print(df_man["eligible_for_X_primary"].value_counts())

