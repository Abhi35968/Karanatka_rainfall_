import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
raw_dem = pd.read_csv(base_dir / "Stage3_1_Audit/data/srtm_elevations_raw.csv")
meta_enriched = pd.read_parquet(base_dir / "Stage3_Feature_Engineering/train_dataset.parquet", columns=[
    "station_id", "latitude", "longitude", "district", "zone_name", "subdivision", "elevation_m", "slope_deg"
]).drop_duplicates(subset=["station_id"]).sort_values("station_id").reset_index(drop=True)

# Merge
df_comp = pd.merge(raw_dem[["station_id", "dem_elevation_m"]], meta_enriched, on="station_id")
df_comp = df_comp.rename(columns={"elevation_m": "pressure_proxy_elevation_m", "slope_deg": "proxy_slope_deg"})

# Calculate true DEM-derived slope on 0.25 deg grid
lat_lon_dem = {(round(r["latitude"], 2), round(r["longitude"], 2)): r["dem_elevation_m"] for _, r in df_comp.iterrows()}
dem_slopes = []
for _, r in df_comp.iterrows():
    lat = round(r["latitude"], 2)
    lon = round(r["longitude"], 2)
    elev = r["dem_elevation_m"]
    
    e_north = lat_lon_dem.get((round(lat + 0.25, 2), lon), elev)
    e_south = lat_lon_dem.get((round(lat - 0.25, 2), lon), elev)
    dy_m = 27.75 * 1000.0
    dz_dy = (e_north - e_south) / (2.0 * dy_m)
    
    e_east = lat_lon_dem.get((lat, round(lon + 0.25, 2)), elev)
    e_west = lat_lon_dem.get((lat, round(lon - 0.25, 2)), elev)
    dx_m = 27.75 * np.cos(np.radians(lat)) * 1000.0
    dz_dx = (e_east - e_west) / (2.0 * dx_m)
    
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    dem_slopes.append(round(slope_deg, 3))

df_comp["dem_slope_deg"] = dem_slopes
df_comp["elevation_diff_m"] = np.round(df_comp["pressure_proxy_elevation_m"] - df_comp["dem_elevation_m"], 2)
df_comp["abs_elevation_diff_m"] = np.abs(df_comp["elevation_diff_m"])

print("--- DEM vs. Pressure Proxy Elevation Comparison (252 Stations) ---")
print(f"SRTM DEM Elevation Min / Max / Mean:     {df_comp['dem_elevation_m'].min():.1f} m / {df_comp['dem_elevation_m'].max():.1f} m / {df_comp['dem_elevation_m'].mean():.1f} m")
print(f"Pressure Proxy Elevation Min / Max / Mean: {df_comp['pressure_proxy_elevation_m'].min():.1f} m / {df_comp['pressure_proxy_elevation_m'].max():.1f} m / {df_comp['pressure_proxy_elevation_m'].mean():.1f} m")
print(f"Mean Error (Bias, Proxy - DEM):           {df_comp['elevation_diff_m'].mean():+.2f} m")
print(f"Mean Absolute Error (MAE):                {df_comp['abs_elevation_diff_m'].mean():.2f} m")
print(f"Root Mean Square Error (RMSE):            {np.sqrt((df_comp['elevation_diff_m']**2).mean()):.2f} m")
corr_elev = np.corrcoef(df_comp['dem_elevation_m'], df_comp['pressure_proxy_elevation_m'])[0, 1]
print(f"Pearson Correlation (r):                  {corr_elev:.4f}")

corr_slope = np.corrcoef(df_comp['dem_slope_deg'], df_comp['proxy_slope_deg'])[0, 1]
print(f"\nDEM Slope Min / Max / Mean:               {df_comp['dem_slope_deg'].min():.3f}° / {df_comp['dem_slope_deg'].max():.3f}° / {df_comp['dem_slope_deg'].mean():.3f}°")
print(f"Proxy Slope Min / Max / Mean:             {df_comp['proxy_slope_deg'].min():.3f}° / {df_comp['proxy_slope_deg'].max():.3f}° / {df_comp['proxy_slope_deg'].mean():.3f}°")
print(f"Slope Pearson Correlation (r):            {corr_slope:.4f}")

print("\n--- Sub-Division Breakdown ---")
sub_summary = df_comp.groupby("subdivision").agg({
    "dem_elevation_m": "mean",
    "pressure_proxy_elevation_m": "mean",
    "elevation_diff_m": "mean",
    "abs_elevation_diff_m": "mean",
    "dem_slope_deg": "mean",
    "proxy_slope_deg": "mean"
}).round(2)
print(sub_summary)

out_audit = base_dir / "Stage3_1_Audit/data/dem_elevation_slope_audit.csv"
df_comp.to_csv(out_audit, index=False)
print("\nSaved dem_elevation_slope_audit.csv to", out_audit)

