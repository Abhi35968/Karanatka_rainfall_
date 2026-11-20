import urllib.request
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
out_dir = base_dir / "Stage3_4_Final_Release_Audit"
out_dir.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(base_dir / "Stage3_2_Final_Verification/station_topography_verified.csv")
prev_raw = pd.read_csv(base_dir / "Stage3_1_Audit/data/srtm_elevations_raw.csv")

# 1. Fetch Copernicus DEM GLO-90 for all 252 stations
print("Fetching Copernicus DEM GLO-90 for all 252 stations...")
copernicus_elev = []
batch_size = 50
for i in range(0, len(meta), batch_size):
    batch = meta.iloc[i:i+batch_size]
    lats = ",".join(str(r["latitude"]) for _, r in batch.iterrows())
    lons = ",".join(str(r["longitude"]) for _, r in batch.iterrows())
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        copernicus_elev.extend([float(e) for e in data["elevation"]])
    time.sleep(0.4)

meta["copernicus_dem_glo90_m"] = np.round(copernicus_elev, 2)
meta["open_elevation_m"] = np.round(prev_raw["dem_elevation_m"], 2)

# 2. Build terrain_source_inventory_252_cells.csv
inventory_rows = []
for idx, r in meta.iterrows():
    st_id = int(r["station_id"])
    lat = float(r["latitude"])
    lon = float(r["longitude"])
    dist = r["district"]
    subdiv = r["subdivision"]
    
    # In Stage 3.1: first 250 used open-elevation, last 2 used open-meteo
    if st_id <= 250:
        api_prov = "Open-Elevation (Public API)"
        api_end = "https://api.open-elevation.com/api/v1/lookup"
        ds_id = "None (Unspecified in API response payload)"
        src_dem = "Unverified SRTM derivative (claimed CGIAR-CSI SRTM v4.1 or 250m)"
        prov_status = "Unverified Third-Party Provenance"
        nat_res = "90 m / 250 m (unverified)"
        resamp_res = "250 m (reported in repo create-dataset script)"
        v_datum = "EGM96 (claimed by project)"
        model_type = "Digital Surface Model (DSM, radar canopy reflective)"
        extr_meth = "HTTP POST JSON batch query (point lookup at centroid)"
    else:
        api_prov = "Open-Meteo Elevation API"
        api_end = "https://api.open-meteo.com/v1/elevation"
        ds_id = "Copernicus DEM GLO-90"
        src_dem = "Copernicus DEM (GLO-90, 2021 Worldwide Release)"
        prov_status = "Verified Authoritative Copernicus DEM"
        nat_res = "90 m (3 arc-seconds)"
        resamp_res = "None (native 90m raster bilinear sample)"
        v_datum = "EGM2008 / EGM96"
        model_type = "Digital Surface Model (DSM, hydro-enforced / edited)"
        extr_meth = "HTTP GET URL query (bilinear lookup at centroid)"
        
    inventory_rows.append({
        "grid_cell_id": st_id,
        "latitude": lat,
        "longitude": lon,
        "district": dist,
        "subdivision": subdiv,
        "current_elevation_m": r["open_elevation_m"],
        "harmonized_copernicus_dem_m": r["copernicus_dem_glo90_m"],
        "api_provider": api_prov,
        "api_endpoint": api_end,
        "dataset_identifier_returned": ds_id,
        "source_dem_product": src_dem,
        "provenance_status": prov_status,
        "native_spatial_resolution": nat_res,
        "resampled_spatial_resolution": resamp_res,
        "vertical_datum": v_datum,
        "surface_model_type": model_type,
        "extraction_method": extr_meth
    })

df_inv = pd.DataFrame(inventory_rows)
out_inv = out_dir / "terrain_source_inventory_252_cells.csv"
df_inv.to_csv(out_inv, index=False)
print("Saved terrain_source_inventory_252_cells.csv")

# 3. Compute Slope with Rigorous Boundary Handling
# Let's define a rigorous function to calculate regional slope
coords_set = set(zip(meta["latitude"].round(2), meta["longitude"].round(2)))

def calculate_regional_slope(elev_dict):
    slopes = []
    dy_m = 27.75 * 1000.0  # 27,750 m
    
    for _, r in meta.iterrows():
        lat = round(r["latitude"], 2)
        lon = round(r["longitude"], 2)
        c_elev = elev_dict[(lat, lon)]
        
        has_n = (round(lat + 0.25, 2), lon) in coords_set
        has_s = (round(lat - 0.25, 2), lon) in coords_set
        
        if has_n and has_s:
            # Interior central difference
            dz_dy = (elev_dict[(round(lat + 0.25, 2), lon)] - elev_dict[(round(lat - 0.25, 2), lon)]) / (2.0 * dy_m)
        elif has_n:
            # North neighbor only (forward difference, 1 step)
            dz_dy = (elev_dict[(round(lat + 0.25, 2), lon)] - c_elev) / dy_m
        elif has_s:
            # South neighbor only (backward difference, 1 step)
            dz_dy = (c_elev - elev_dict[(round(lat - 0.25, 2), lon)]) / dy_m
        else:
            dz_dy = 0.0
            
        has_e = (lat, round(lon + 0.25, 2)) in coords_set
        has_w = (lat, round(lon - 0.25, 2)) in coords_set
        dx_m = 27.75 * np.cos(np.radians(lat)) * 1000.0
        
        if has_e and has_w:
            # Interior central difference
            dz_dx = (elev_dict[(lat, round(lon + 0.25, 2))] - elev_dict[(lat, round(lon - 0.25, 2))]) / (2.0 * dx_m)
        elif has_e:
            # East neighbor only (forward difference, 1 step)
            dz_dx = (elev_dict[(lat, round(lon + 0.25, 2))] - c_elev) / dx_m
        elif has_w:
            # West neighbor only (backward difference, 1 step)
            dz_dx = (c_elev - elev_dict[(lat, round(lon - 0.25, 2))]) / dx_m
        else:
            dz_dx = 0.0
            
        slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        slopes.append(round(slope_deg, 4))
        
    return slopes

oe_dict = {(round(r["latitude"], 2), round(r["longitude"], 2)): r["open_elevation_m"] for _, r in meta.iterrows()}
cop_dict = {(round(r["latitude"], 2), round(r["longitude"], 2)): r["copernicus_dem_glo90_m"] for _, r in meta.iterrows()}

meta["open_elevation_slope_deg"] = calculate_regional_slope(oe_dict)
meta["copernicus_slope_deg"] = calculate_regional_slope(cop_dict)

# 4. Build dem_source_harmonization_comparison.csv
meta["elevation_diff_cop_minus_oe_m"] = np.round(meta["copernicus_dem_glo90_m"] - meta["open_elevation_m"], 2)
meta["abs_elevation_diff_m"] = meta["elevation_diff_cop_minus_oe_m"].abs()
meta["slope_diff_deg"] = np.round(meta["copernicus_slope_deg"] - meta["open_elevation_slope_deg"], 4)
meta["abs_slope_diff_deg"] = meta["slope_diff_deg"].abs()

comp_cols = [
    "station_id", "latitude", "longitude", "district", "subdivision",
    "open_elevation_m", "copernicus_dem_glo90_m", "elevation_diff_cop_minus_oe_m", "abs_elevation_diff_m",
    "open_elevation_slope_deg", "copernicus_slope_deg", "slope_diff_deg", "abs_slope_diff_deg"
]

df_comp = meta[comp_cols].rename(columns={"station_id": "grid_cell_id"})
out_comp = out_dir / "dem_source_harmonization_comparison.csv"
df_comp.to_csv(out_comp, index=False)
print("Saved dem_source_harmonization_comparison.csv")

# Print Summary
print("\n--- Summary Statistics of Comparison (252 Grid Cells) ---")
print(f"Open-Elevation Min/Max/Mean:    {meta['open_elevation_m'].min():.1f} / {meta['open_elevation_m'].max():.1f} / {meta['open_elevation_m'].mean():.2f} m")
print(f"Copernicus DEM Min/Max/Mean:    {meta['copernicus_dem_glo90_m'].min():.1f} / {meta['copernicus_dem_glo90_m'].max():.1f} / {meta['copernicus_dem_glo90_m'].mean():.2f} m")
print(f"Elevation Pearson r:            {np.corrcoef(meta['copernicus_dem_glo90_m'], meta['open_elevation_m'])[0, 1]:.5f}")
print(f"Elevation Mean Difference:      {meta['elevation_diff_cop_minus_oe_m'].mean():+.2f} m")
print(f"Elevation MAE:                  {meta['abs_elevation_diff_m'].mean():.2f} m")
print(f"Elevation Max Absolute Diff:    {meta['abs_elevation_diff_m'].max():.2f} m (Station {meta.loc[meta['abs_elevation_diff_m'].idxmax(), 'station_id']})")
print(f"Slope Pearson r:                {np.corrcoef(meta['copernicus_slope_deg'], meta['open_elevation_slope_deg'])[0, 1]:.5f}")
print(f"Slope MAE:                      {meta['abs_slope_diff_deg'].mean():.4f} deg")
