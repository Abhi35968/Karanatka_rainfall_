import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
out_dir = base_dir / "Stage4_0_Pretraining_Verification"
out_dir.mkdir(parents=True, exist_ok=True)

print("--- 1. Loading Final Parquet Datasets ---")
df_full = pd.read_parquet(data_dir / "final_stage3_dataset.parquet")
df_train = pd.read_parquet(data_dir / "final_train_dataset.parquet")
df_val = pd.read_parquet(data_dir / "final_validation_dataset.parquet")
df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")

# Basic dimensions check
print(f"Full dataset: {len(df_full):,} rows, {len(df_full.columns)} columns")
print(f"Train partition: {len(df_train):,} rows")
print(f"Validation partition: {len(df_val):,} rows")
print(f"Test partition: {len(df_test):,} rows")

assert len(df_full) == 640836, f"Expected 640836, got {len(df_full)}"
assert len(df_full.columns) == 78, f"Expected 78 cols, got {len(df_full.columns)}"
assert len(df_train) == 364644, f"Expected 364644, got {len(df_train)}"
assert len(df_val) == 91980, f"Expected 91980, got {len(df_val)}"
assert len(df_test) == 184212, f"Expected 184212, got {len(df_test)}"
assert len(df_train) + len(df_val) + len(df_test) == len(df_full), "Partition row sum mismatch!"

# Unique cells check
unique_cells = df_full["station_id"].nunique()
print(f"Unique IMD grid cells: {unique_cells}")
assert unique_cells == 252, f"Expected 252 unique cells, got {unique_cells}"

# Duplicate cell/date check
n_dups = df_full.duplicated(subset=["station_id", "date"]).sum()
print(f"Duplicate grid-cell / date combinations: {n_dups}")
assert n_dups == 0, f"Found {n_dups} duplicates!"

# Negative target check
neg_tgt = (df_full["target_rainfall_mm"] < 0).sum()
print(f"Negative target rainfall values: {neg_tgt}")
assert neg_tgt == 0, f"Found {neg_tgt} negative rainfall values!"

# Load reference Copernicus extraction
df_comp = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/dem_source_harmonization_comparison.csv")

# Extract unique station terrain from Parquet
st_terrain = df_full[["station_id", "latitude", "longitude", "district", "subdivision", "elevation_dem_m", "slope_dem_deg"]].drop_duplicates().sort_values("station_id").reset_index(drop=True)
df_comp_sorted = df_comp.sort_values("grid_cell_id").reset_index(drop=True)

verification_rows = []
for i in range(len(st_terrain)):
    r_pq = st_terrain.iloc[i]
    r_src = df_comp_sorted.iloc[i]
    
    st_id = int(r_pq["station_id"])
    lat = float(r_pq["latitude"])
    lon = float(r_pq["longitude"])
    dist = r_pq["district"]
    subdiv = r_pq["subdivision"]
    
    pq_elev = float(r_pq["elevation_dem_m"])
    src_elev = float(r_src["copernicus_dem_glo90_m"])
    elev_diff = round(pq_elev - src_elev, 2)
    elev_pass = np.isclose(pq_elev, src_elev, atol=1e-2)
    
    pq_slope = float(r_pq["slope_dem_deg"])
    src_slope = float(r_src["copernicus_slope_deg"])
    slope_diff = round(pq_slope - src_slope, 4)
    slope_pass = np.isclose(pq_slope, src_slope, atol=1e-4)
    
    all_pass = elev_pass and slope_pass
    
    verification_rows.append({
        "grid_cell_id": st_id,
        "latitude": lat,
        "longitude": lon,
        "district": dist,
        "subdivision": subdiv,
        "parquet_elevation_dem_m": pq_elev,
        "copernicus_source_elevation_m": src_elev,
        "elevation_diff_m": elev_diff,
        "elevation_verification_status": "PASSED" if elev_pass else "FAILED",
        "parquet_slope_dem_deg": pq_slope,
        "calculated_slope_deg": src_slope,
        "slope_diff_deg": slope_diff,
        "slope_verification_status": "PASSED" if slope_pass else "FAILED",
        "overall_terrain_verdict": "PASSED" if all_pass else "FAILED"
    })

df_ver = pd.DataFrame(verification_rows)
out_csv = out_dir / "terrain_value_verification_252_cells.csv"
df_ver.to_csv(out_csv, index=False)
print(f"Saved terrain_value_verification_252_cells.csv successfully.")
print(f"Elevation verification pass rate: {(df_ver['elevation_verification_status'] == 'PASSED').sum()} / 252")
print(f"Slope verification pass rate:     {(df_ver['slope_verification_status'] == 'PASSED').sum()} / 252")
print(f"Overall terrain verification:     {(df_ver['overall_terrain_verdict'] == 'PASSED').sum()} / 252")

