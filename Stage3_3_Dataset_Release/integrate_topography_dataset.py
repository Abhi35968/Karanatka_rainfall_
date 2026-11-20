import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
topo_path = base_dir / "Stage3_2_Final_Verification/station_topography_verified.csv"
df_topo = pd.read_csv(topo_path)

# 1. Inspect Topography Table
print("--- 1. Topography Source Check ---")
n_topo_stations = len(df_topo)
n_unique_topo = df_topo["station_id"].nunique()
print(f"Topography table stations: {n_topo_stations}, unique station_id: {n_unique_topo}")

# 2. Load Stage 3 Parquet
print("--- 2. Loading Stage 3 Datasets ---")
p_full = base_dir / "Stage3_Feature_Engineering/stage3_dataset.parquet"
p_train = base_dir / "Stage3_Feature_Engineering/train_dataset.parquet"
p_val = base_dir / "Stage3_Feature_Engineering/validation_dataset.parquet"
p_test = base_dir / "Stage3_Feature_Engineering/test_dataset.parquet"

df_full_s3 = pd.read_parquet(p_full)
df_train_s3 = pd.read_parquet(p_train)
df_val_s3 = pd.read_parquet(p_val)
df_test_s3 = pd.read_parquet(p_test)

print(f"Loaded: Full={len(df_full_s3):,}, Train={len(df_train_s3):,}, Val={len(df_val_s3):,}, Test={len(df_test_s3):,}")

# Prepare join mapping
# We take station_id, elevation_dem_m, slope_dem_deg
topo_features = df_topo[["station_id", "elevation_dem_m", "slope_dem_deg"]].copy()

def integrate_dataset(df_source, name):
    print(f"Integrating {name}...")
    df = df_source.copy()
    
    # Rename existing pressure proxy columns explicitly to prevent confusion with physical DEM
    rename_dict = {}
    if "elevation_m" in df.columns:
        rename_dict["elevation_m"] = "elevation_pressure_proxy_m"
    if "slope_deg" in df.columns:
        rename_dict["slope_deg"] = "slope_pressure_proxy_deg"
    df = df.rename(columns=rename_dict)
    
    # Merge on station_id
    df_merged = pd.merge(df, topo_features, on="station_id", how="left")
    
    # Position new DEM columns alongside orography features
    cols = list(df_merged.columns)
    # Reorder columns logically: place elevation_dem_m, slope_dem_deg right after slope_pressure_proxy_deg
    cols.remove("elevation_dem_m")
    cols.remove("slope_dem_deg")
    
    insert_idx = cols.index("slope_pressure_proxy_deg") + 1
    cols.insert(insert_idx, "elevation_dem_m")
    cols.insert(insert_idx + 1, "slope_dem_deg")
    
    df_merged = df_merged[cols]
    
    # Ensure float32 for continuous features to match stage 3 parquet convention
    df_merged["elevation_dem_m"] = df_merged["elevation_dem_m"].astype(np.float32)
    df_merged["slope_dem_deg"] = df_merged["slope_dem_deg"].astype(np.float32)
    
    return df_merged

df_full_int = integrate_dataset(df_full_s3, "Full Dataset")
df_train_int = integrate_dataset(df_train_s3, "Training Dataset")
df_val_int = integrate_dataset(df_val_s3, "Validation Dataset")
df_test_int = integrate_dataset(df_test_s3, "Test Dataset")

# 3. Comprehensive Join Integrity Audit
print("--- 3. Running Integrity Assertions ---")
integrity_checks = []

def add_check(cid, name, exp, act, cond, details=""):
    status = "PASSED" if cond else "FAILED"
    integrity_checks.append({
        "check_id": cid,
        "check_name": name,
        "expected_value": str(exp),
        "actual_value": str(act),
        "status": status,
        "details": details
    })
    print(f"Check {cid} [{name}]: {status} (Expected: {exp}, Actual: {act})")

# C1: Unique station count in topo
add_check("C01", "Unique Stations in Topography Source", 252, n_unique_topo, n_unique_topo == 252, "Must cover all in-state IMD stations.")

# C2: No duplicate keys in topo
add_check("C02", "No Duplicate Keys in Topography Table", 252, len(df_topo), len(df_topo) == 252, "Zero duplicate station_id rows.")

# C3: Full dataset row count
add_check("C03", "Full Integrated Dataset Row Count", 640836, len(df_full_int), len(df_full_int) == 640836, "Exact row count preservation.")

# C4: Train row count
add_check("C04", "Train Integrated Dataset Row Count", 364644, len(df_train_int), len(df_train_int) == 364644, "Exact train partition preservation.")

# C5: Validation row count
add_check("C05", "Validation Integrated Dataset Row Count", 91980, len(df_val_int), len(df_val_int) == 91980, "Exact validation partition preservation.")

# C6: Test row count
add_check("C06", "Test Integrated Dataset Row Count", 184212, len(df_test_int), len(df_test_int) == 184212, "Exact test partition preservation.")

# C7: Partition row sum
part_sum = len(df_train_int) + len(df_val_int) + len(df_test_int)
add_check("C07", "Partition Row Sum Equality", 640836, part_sum, part_sum == 640836, "Train + Val + Test = Full.")

# C8: No missing DEM elevation values
nan_elev = int(df_full_int["elevation_dem_m"].isna().sum())
add_check("C08", "No Missing Values in elevation_dem_m", 0, nan_elev, nan_elev == 0, "All 640,836 rows must have valid DEM elevation.")

# C9: No missing DEM slope values
nan_slope = int(df_full_int["slope_dem_deg"].isna().sum())
add_check("C09", "No Missing Values in slope_dem_deg", 0, nan_slope, nan_slope == 0, "All 640,836 rows must have valid DEM slope.")

# C10: Target identity check
tgt_match = np.allclose(df_full_int["target_rainfall_mm"], df_full_s3["target_rainfall_mm"], atol=1e-5)
add_check("C10", "Target Rainfall Unchanged Post-Join", True, bool(tgt_match), bool(tgt_match), "Target rainfall values must be bitwise identical.")

# C11: Date sequence identity check
date_match = (df_full_int["date"] == df_full_s3["date"]).all()
add_check("C11", "Date Sequence Unchanged Post-Join", True, bool(date_match), bool(date_match), "Calendar dates must match row-by-row.")

# C12: Station ID sequence identity check
st_match = (df_full_int["station_id"] == df_full_s3["station_id"]).all()
add_check("C12", "Station ID Sequence Unchanged Post-Join", True, bool(st_match), bool(st_match), "Station IDs must match row-by-row.")

# C13: DEM Value Fidelity
# Check on Station 1
st1_dem_expected = float(df_topo[df_topo["station_id"] == 1]["elevation_dem_m"].iloc[0])
st1_dem_actual = float(df_full_int[df_full_int["station_id"] == 1]["elevation_dem_m"].iloc[0])
dem_val_match = np.isclose(st1_dem_expected, st1_dem_actual, atol=1e-2)
add_check("C13", "DEM Value Fidelity for Station 1", st1_dem_expected, round(st1_dem_actual, 2), dem_val_match, "Station 1 DEM matches source CSV.")

# C14: Check max mountain summit DEM (Station 72)
st72_dem_expected = float(df_topo[df_topo["station_id"] == 72]["elevation_dem_m"].iloc[0])
st72_dem_actual = float(df_full_int[df_full_int["station_id"] == 72]["elevation_dem_m"].iloc[0])
dem_st72_match = np.isclose(st72_dem_expected, st72_dem_actual, atol=1e-2)
add_check("C14", "Summit Elevation Fidelity (Station 72)", st72_dem_expected, round(st72_dem_actual, 2), dem_st72_match, "Station 72 DEM matches 1599.54 m.")

# C15: Check coastal station DEM (Station 104)
st104_dem_expected = float(df_topo[df_topo["station_id"] == 104]["elevation_dem_m"].iloc[0])
st104_dem_actual = float(df_full_int[df_full_int["station_id"] == 104]["elevation_dem_m"].iloc[0])
dem_st104_match = np.isclose(st104_dem_expected, st104_dem_actual, atol=1e-2)
add_check("C15", "Coastal Elevation Fidelity (Station 104)", st104_dem_expected, round(st104_dem_actual, 2), dem_st104_match, "Station 104 DEM matches 1.12 m.")

# C16: Column count check (75 original + 2 added DEM - 0 dropped = 77)
add_check("C16", "Integrated Column Count", 77, len(df_full_int.columns), len(df_full_int.columns) == 77, "Renamed 2 proxy columns + added 2 DEM columns = 77 columns.")

df_integrity = pd.DataFrame(integrity_checks)
out_int_csv = base_dir / "Stage3_3_Dataset_Release/dataset_join_integrity.csv"
df_integrity.to_csv(out_int_csv, index=False)
print("Saved dataset_join_integrity.csv successfully.")

# 4. Save Final Parquet Datasets
out_dir = base_dir / "Stage3_3_Dataset_Release"
p_out_full = out_dir / "integrated_stage3_dataset.parquet"
p_out_train = out_dir / "integrated_train_dataset.parquet"
p_out_val = out_dir / "integrated_validation_dataset.parquet"
p_out_test = out_dir / "integrated_test_dataset.parquet"

print("Saving Parquet datasets...")
df_full_int.to_parquet(p_out_full, index=False, engine="pyarrow", compression="snappy")
df_train_int.to_parquet(p_out_train, index=False, engine="pyarrow", compression="snappy")
df_val_int.to_parquet(p_out_val, index=False, engine="pyarrow", compression="snappy")
df_test_int.to_parquet(p_out_test, index=False, engine="pyarrow", compression="snappy")
print("All Parquet datasets saved.")

# 5. Generate Manifest with SHA-256 Checksums
def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

manifest = []
files_to_manifest = [
    ("Integrated Full Dataset", p_out_full, len(df_full_int), len(df_full_int.columns), "2019-01-15", "2025-12-31"),
    ("Integrated Training Set", p_out_train, len(df_train_int), len(df_train_int.columns), "2019-01-15", "2022-12-31"),
    ("Integrated Validation Set", p_out_val, len(df_val_int), len(df_val_int.columns), "2023-01-01", "2023-12-31"),
    ("Integrated Test Set", p_out_test, len(df_test_int), len(df_test_int.columns), "2024-01-01", "2025-12-31"),
]

for name, fpath, n_rows, n_cols, d_start, d_end in files_to_manifest:
    f_size_bytes = fpath.stat().st_size
    f_size_mb = round(f_size_bytes / (1024 * 1024), 2)
    f_hash = compute_sha256(fpath)
    manifest.append({
        "dataset_name": name,
        "file_name": fpath.name,
        "row_count": n_rows,
        "column_count": n_cols,
        "date_start": d_start,
        "date_end": d_end,
        "file_size_bytes": f_size_bytes,
        "file_size_mb": f_size_mb,
        "sha256_checksum": f_hash
    })

df_manifest = pd.DataFrame(manifest)
out_manifest_csv = out_dir / "final_train_validation_test_manifest.csv"
df_manifest.to_csv(out_manifest_csv, index=False)
print("Saved final_train_validation_test_manifest.csv successfully.")
print(df_manifest[["dataset_name", "file_name", "row_count", "column_count", "file_size_mb", "sha256_checksum"]].to_string())

# 6. Save JSON Schema
schema_dict = {
    "dataset_title": "Karnataka Daily Rainfall Forecasting Integrated Spatiotemporal Dataset",
    "version": "Stage 3.3 Release Gate (Integrated)",
    "creation_timestamp": pd.Timestamp.now().isoformat(),
    "spatial_coverage": "Karnataka State, India (252 in-state 0.25 deg IMD grid cells)",
    "temporal_coverage": "2019-01-15 to 2025-12-31 (2,543 calendar days)",
    "total_records": len(df_full_int),
    "total_columns": len(df_full_int.columns),
    "partitions": {
        "training": {"rows": len(df_train_int), "dates": "2019-01-15 to 2022-12-31"},
        "validation": {"rows": len(df_val_int), "dates": "2023-01-01 to 2023-12-31"},
        "test": {"rows": len(df_test_int), "dates": "2024-01-01 to 2025-12-31"}
    },
    "columns": []
}

dtypes_int = df_full_int.dtypes.to_dict()
for c in df_full_int.columns:
    schema_dict["columns"].append({
        "column_name": c,
        "data_type": str(dtypes_int[c]),
        "sample_value": str(df_full_int[c].iloc[0])
    })

with open(out_dir / "final_dataset_schema.json", "w") as f:
    json.dump(schema_dict, f, indent=2)
print("Saved final_dataset_schema.json successfully.")

