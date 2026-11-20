import urllib.request
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
meta_path = base_dir / "Stage2_Corrected_Processing/data/karnataka_252_stations_metadata.csv"
meta = pd.read_csv(meta_path)
print(f"Loaded {len(meta)} stations.")

# Prepare locations
locations = [{"latitude": float(r["latitude"]), "longitude": float(r["longitude"])} for _, r in meta.iterrows()]

# Fetch SRTM elevations via Open-Elevation in batches of 50
batch_size = 50
elevations = []

for i in range(0, len(locations), batch_size):
    batch = locations[i:i+batch_size]
    print(f"Fetching batch {i//batch_size + 1}/{(len(locations)-1)//batch_size + 1} ({len(batch)} points)...")
    payload = json.dumps({"locations": batch}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.open-elevation.com/api/v1/lookup",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    )
    success = False
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                for item in res["results"]:
                    elevations.append(float(item["elevation"]))
                success = True
                break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}. Retrying in 2s...")
            time.sleep(2)
    
    if not success:
        # Fallback to open-meteo elevation API for this batch
        print("  Falling back to open-meteo elevation API...")
        lats_str = ",".join(str(loc["latitude"]) for loc in batch)
        lons_str = ",".join(str(loc["longitude"]) for loc in batch)
        om_url = f"https://api.open-meteo.com/v1/elevation?latitude={lats_str}&longitude={lons_str}"
        om_req = urllib.request.Request(om_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(om_req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            elevations.extend([float(e) for e in res["elevation"]])

print(f"Total elevations retrieved: {len(elevations)} / {len(meta)}")
meta["dem_elevation_m"] = np.round(elevations, 2)

# Save intermediate
out_check = base_dir / "Stage3_1_Audit/data/srtm_elevations_raw.csv"
meta.to_csv(out_check, index=False)
print("Saved raw SRTM elevations to", out_check)

