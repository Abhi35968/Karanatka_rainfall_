import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
pq_path = base_dir / "Stage3_Feature_Engineering/stage3_dataset.parquet"
schema = pq.read_schema(pq_path)
col_names = schema.names
col_types = {name: str(schema.field(name).type) for name in col_names}

reconciled = []
for col in col_names:
    col_type = col_types[col]
    
    # Category and Role
    if col in ["date", "station_id", "district", "zone_name"]:
        cat = "Identifier / Metadata"
        src = "GIS / Administrative"
        pred_elig = "No"
        targ_elig = "No"
        is_meta = "Yes"
        ts_avail = "Prior to experiment"
        excl_reason = "Administrative identifier / label; non-generalizable"
    elif col in ["target_period", "prediction_issue_time"]:
        cat = "Protocol Metadata"
        src = "Experimental Design"
        pred_elig = "No"
        targ_elig = "No"
        is_meta = "Yes"
        ts_avail = "Experiment definition"
        excl_reason = "Protocol description text; non-feature"
    elif col in ["target_rainfall_mm", "target_is_rain", "target_is_heavy_rain"]:
        cat = "Ground Truth Target"
        src = "IMD Gridded Observations"
        pred_elig = "No"
        targ_elig = "Yes"
        is_meta = "No"
        ts_avail = "Target day (d)"
        excl_reason = "Supervised learning prediction target"
    elif col in ["is_coastal_fallback"]:
        cat = "Quality Flag"
        src = "Stage 2 Coastal Audit"
        pred_elig = "Yes (Optional Feature / Subgroup Evaluation)"
        targ_elig = "No"
        is_meta = "Yes"
        ts_avail = "Static"
        excl_reason = "None (Useful as subgroup selector or model indicator)"
    elif col in ["interpolation_method"]:
        cat = "Quality Flag"
        src = "Stage 2 Coastal Audit"
        pred_elig = "No"
        targ_elig = "No"
        is_meta = "Yes"
        ts_avail = "Static"
        excl_reason = "String label redundant with binary is_coastal_fallback"
    elif col in ["season"]:
        cat = "Temporal"
        src = "Calendar Ephemeris"
        pred_elig = "No"
        targ_elig = "No"
        is_meta = "No"
        ts_avail = "Astronomical Ephemeris"
        excl_reason = "String label redundant with month and cyclical sin/cos encodings"
    elif col in ["latitude", "longitude", "elevation_m", "slope_deg", "dist_to_coast_km", "dist_to_ghats_crest_km", "zone_id", "subdivision"]:
        cat = "Spatial / Orographic"
        src = "GIS / DEM / ECMWF Topography"
        pred_elig = "Yes"
        targ_elig = "No"
        is_meta = "No"
        ts_avail = "Static prior to experiment"
        excl_reason = "None (Permitted spatial predictor)"
    elif col in ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year", "sin_month", "cos_month", "is_monsoon"]:
        cat = "Temporal Ephemeris"
        src = "Calendar Ephemeris"
        pred_elig = "Yes"
        targ_elig = "No"
        is_meta = "No"
        ts_avail = "Deterministic at issue time"
        excl_reason = "None (Permitted temporal predictor)"
    elif "target_rain_lag" in col or "target_rain_roll" in col or col == "api_14d":
        cat = "Target Autoregressive / Memory"
        src = "IMD Observed Rain (Antecedent)"
        pred_elig = "Yes"
        targ_elig = "No"
        is_meta = "No"
        ts_avail = "Observed on or before Day d-1 03:00 UTC"
        excl_reason = "None (Permitted causal dynamic predictor)"
    elif "era5_tp" in col:
        cat = "Reanalysis Precip Memory"
        src = "ERA5-Land (Antecedent)"
        pred_elig = "Yes"
        targ_elig = "No"
        is_meta = "No"
        ts_avail = "Observed on or before Day d-1 03:00 UTC"
        excl_reason = "None (Permitted causal dynamic predictor)"
    else:
        cat = "Meteorological State / Flux"
        src = "ERA5-Land Corrected (Antecedent)"
        pred_elig = "Yes"
        targ_elig = "No"
        is_meta = "No"
        ts_avail = "Observed on or before Day d-1 03:00 UTC"
        excl_reason = "None (Permitted causal dynamic predictor)"

    reconciled.append({
        "column_name": col,
        "column_type": col_type,
        "category": cat,
        "source": src,
        "predictor_eligibility": pred_elig,
        "target_eligibility": targ_elig,
        "is_metadata_or_id": is_meta,
        "timestamp_availability": ts_avail,
        "exclusion_reason": excl_reason
    })

df_rec = pd.DataFrame(reconciled)
out_p = base_dir / "Stage3_1_Audit/data/feature_schema_reconciliation.csv"
df_rec.to_csv(out_p, index=False)
print(f"Saved feature_schema_reconciliation.csv with {len(df_rec)} entries.")
print("\nReconciliation Breakdown by Category:")
print(df_rec["category"].value_counts())
print("\nPredictor Eligibility Breakdown:")
print(df_rec["predictor_eligibility"].value_counts())
