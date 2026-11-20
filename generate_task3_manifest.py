import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
manifest_path = base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv"
feat_dict_path = base_dir / "Stage3_Feature_Engineering/feature_dictionary.csv"

df_man = pd.read_csv(manifest_path)
df_dict = pd.read_csv(feat_dict_path)

dict_lookup = {}
for _, row in df_dict.iterrows():
    dict_lookup[row["name"]] = row.to_dict()

out_rows = []

# 1. Primary Features (Index 1 to 63)
primary = df_man[df_man["eligible_for_X_primary"] == "Yes"].copy().reset_index(drop=True)
for idx, row in primary.iterrows():
    cname = row["column_name"]
    fgroup = row["feature_group"]
    dtype = row["data_type"]
    
    dinfo = dict_lookup.get(cname, {})
    desc = dinfo.get("description", row.get("pipeline_handling", ""))
    
    if "target_rain" in cname or cname == "api_14d":
        src_var = "IMD Gridded Daily Rainfall"
    elif "era5_tp" in cname:
        src_var = "ERA5-Land Total Precipitation (tp)"
    elif cname.startswith("t2m") or cname in ["d2m_mean_lag_1d", "temp_range_lag_1d", "dewpoint_depression_lag_1d"]:
        src_var = "ERA5-Land 2m Temperature / Dewpoint (t2m, d2m)"
    elif cname.startswith("sp"):
        src_var = "ERA5-Land Surface Pressure (sp)"
    elif cname.startswith("wind"):
        src_var = "ERA5-Land 10m Wind Speed (u10, v10)"
    elif cname.startswith("swvl"):
        src_var = "ERA5-Land Volumetric Soil Water (swvl1..3)"
    elif cname in ["ssrd_lag_1d", "strd_lag_1d"]:
        src_var = "ERA5-Land Surface Downward Radiation (ssrd, strd)"
    elif cname in ["slhf_lag_1d", "sshf_lag_1d"]:
        src_var = "ERA5-Land Turbulent Heat Fluxes (slhf, sshf)"
    elif cname in ["ro_lag_1d", "sro_lag_1d"]:
        src_var = "ERA5-Land Runoff Diagnostics (ro, sro)"
    elif cname in ["e_lag_1d", "pev_lag_1d", "evabs_lag_1d", "evatc_lag_1d", "evavt_lag_1d"]:
        src_var = "ERA5-Land Evaporation Diagnostics (e, pev, evabs, evatc, evavt)"
    elif cname in ["src_lag_1d", "fal_lag_1d"]:
        src_var = "ERA5-Land Skin Reservoir / Albedo (src, fal)"
    elif cname in ["latitude", "longitude", "dist_to_coast_km", "dist_to_ghats_crest_km", "zone_id", "subdivision"]:
        src_var = "Geographic Coordinates / Administrative Boundaries"
    elif cname in ["elevation_dem_m", "slope_dem_deg"]:
        src_var = "SRTM Digital Elevation Model (DEM 30m)"
    elif cname in ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "is_monsoon"]:
        src_var = "Astronomical Ephemeris / Calendar Date"
    else:
        src_var = "Derived Engineering"

    if "lag_1d" in cname:
        lag_win = "Day t-1 (Lag 1D: [t-2 03/04 UTC to t-1 03 UTC])"
    elif "lag_2d" in cname:
        lag_win = "Day t-2 (Lag 2D)"
    elif "lag_3d" in cname:
        lag_win = "Day t-3 (Lag 3D)"
    elif "lag_7d" in cname:
        lag_win = "Day t-7 (Lag 7D)"
    elif "lag_14d" in cname:
        lag_win = "Day t-14 (Lag 14D)"
    elif "roll_3d" in cname:
        lag_win = "3-Day Rolling Window (t-3 to t-1, shifted)"
    elif "roll_7d" in cname:
        lag_win = "7-Day Rolling Window (t-7 to t-1, shifted)"
    elif "roll_14d" in cname:
        lag_win = "14-Day Rolling Window (t-14 to t-1, shifted)"
    elif cname == "api_14d":
        lag_win = "14-Day Exponential Decay (k=0.85, t-14 to t-1)"
    elif cname in ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "is_monsoon"]:
        lag_win = "Day t (Deterministic Astronomical Ephemeris)"
    else:
        lag_win = "Static Invariant"

    if "rain" in cname or "tp" in cname or "swvl" in cname or "ro" in cname or "sro" in cname or "e_" in cname or "pev" in cname or cname == "api_14d":
        gating = "Precipitation / Hydrological Moisture State"
    elif "t2m" in cname or "d2m" in cname or "temp_range" in cname or "dewpoint" in cname or "sp" in cname or "wind" in cname:
        gating = "Atmospheric Thermodynamic / Kinematic State"
    elif "ssrd" in cname or "strd" in cname or "slhf" in cname or "sshf" in cname:
        gating = "Radiative & Turbulent Energy Balance"
    elif fgroup == "Temporal Ephemeris":
        gating = "Seasonal / Calendar Phenology"
    else:
        gating = "Topographic & Orographic Boundary"

    if cname in ["subdivision", "zone_id"]:
        model_dtype = "float32 (Target-encoded in X as subdivision_enc, zone_id_enc)"
    else:
        model_dtype = str(dtype)

    out_rows.append({
        "feature_index": str(idx + 1),
        "feature_name": cname,
        "feature_group": fgroup,
        "physical_source_variable": src_var,
        "temporal_lag_or_window": lag_win,
        "gating_dependency": gating,
        "is_coastal_fallback_flag": "No",
        "model_input_data_type": model_dtype,
        "description": desc
    })

# 2. Auxiliary Feature (is_coastal_fallback)
aux_row = df_man[df_man["column_name"] == "is_coastal_fallback"].iloc[0]
out_rows.append({
    "feature_index": "Aux-1",
    "feature_name": "is_coastal_fallback",
    "feature_group": "Data Lineage & Spatial QC",
    "physical_source_variable": "ECMWF Land-Sea Mask / Shoreline Boundary",
    "temporal_lag_or_window": "Static Invariant (Grid-Cell IDs: 52, 68, 125)",
    "gating_dependency": "Coastal Spatial-Fill Routing",
    "is_coastal_fallback_flag": "Yes (Binary Flag 0/1)",
    "model_input_data_type": "int8 (Excluded from X_primary; used for subgroup diagnostics)",
    "description": "Binary flag identifying 3 coastal margin cells requiring nearest-land ERA5 grid-point substitution."
})

# 3. Excluded Target, Administrative, and Proxy Columns (14 Columns)
excl_df = df_man[~df_man["column_name"].isin(primary["column_name"]) & (df_man["column_name"] != "is_coastal_fallback")]
for e_idx, (_, row) in enumerate(excl_df.iterrows(), 1):
    cname = row["column_name"]
    role = row["role_in_dataset"]
    fgroup = row["feature_group"]
    dtype = row["data_type"]
    desc = row.get("pipeline_handling", "")
    
    if "target" in cname:
        gating = "Ground-Truth Evaluation Label"
    elif role == "Identifier":
        gating = "Spatio-Temporal Indexing"
    elif "Proxy" in role:
        gating = "Excluded Pressure Proxy (Ablation Only)"
    else:
        gating = "Administrative / Metadata"

    out_rows.append({
        "feature_index": f"Excl-{e_idx}",
        "feature_name": cname,
        "feature_group": fgroup,
        "physical_source_variable": f"{role} ({fgroup})",
        "temporal_lag_or_window": "Target Day t or Metadata",
        "gating_dependency": gating,
        "is_coastal_fallback_flag": "No",
        "model_input_data_type": f"{dtype} (Excluded from model feature matrix X)",
        "description": f"{role}: {desc}"
    })

df_all_out = pd.DataFrame(out_rows)
out_csv = base_dir / "Stage9_2_Final_Author_Review/complete_63_feature_manifest.csv"
df_all_out.to_csv(out_csv, index=False)
print(f"Saved complete manifest ({len(df_all_out)} total rows: 63 primary, 1 auxiliary, 14 excluded) to:", out_csv)
