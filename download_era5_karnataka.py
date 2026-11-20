"""
Download hourly ERA5-Land data for Karnataka from Copernicus CDS.

What this script does:
- Uses the ERA5-Land hourly dataset (0.1° grid, hourly).
- Downloads ~30 selected weather/land variables.
- Automatically splits each month into small requests so the CDS 12,000-field
  request limit is not exceeded.
- Saves one NetCDF file per request and one merged NetCDF per month.
- Optionally converts monthly NetCDF files into a single CSV.

Before running:
1. Create/activate a Python environment.
2. pip install "cdsapi>=0.7.7" xarray netCDF4 pandas numpy
3. Configure your CDS Personal Access Token in ~/.cdsapirc
   (Windows: %USERPROFILE%\\.cdsapirc)
4. Accept the ERA5-Land dataset Terms of Use on the CDS website.

Example:
    python download_era5_karnataka.py --start-year 2024 --end-year 2024 --months 6

For several months:
    python download_era5_karnataka.py --start-year 2019 --end-year 2024 --months 1 2 3 4 5 6 7 8 9 10 11 12

Notes:
- Area is [North, West, South, East] = [18, 74, 11, 79].
- This is a bounding box around Karnataka, not an exact state polygon.
- ERA5-Land itself is hourly and 0.1° in the CDS gridded product.
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from pathlib import Path

import cdsapi
import xarray as xr


# -----------------------------
# USER/PROJECT CONFIGURATION
# -----------------------------

DATASET = "reanalysis-era5-land"

# Karnataka bounding box:
# [North, West, South, East]
AREA = [19.0, 74.0, 11.0, 79.0]

# Keep below CDS's 12,000-field request limit.
# 14 variables x 31 days x 24 hours = 10,416 fields.
MAX_FIELDS = 11_000

# 30 selected hourly variables.
VARIABLES = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "skin_temperature",

    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",

    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",

    "surface_pressure",

    "10m_u_component_of_wind",
    "10m_v_component_of_wind",

    "total_precipitation",

    "forecast_albedo",

    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",

    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",

    "total_evaporation",
    "potential_evaporation",
    "runoff",
    "surface_runoff",

    "evaporation_from_bare_soil",
    "evaporation_from_the_top_of_canopy",
    "evaporation_from_vegetation_transpiration",

    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",

    "skin_reservoir_content",
]

HOURS = [f"{h:02d}:00" for h in range(24)]


def split_variables_for_month(year: int, month: int) -> list[list[str]]:
    """
    Split ERA5-Land variables into small requests.

    CDS request cost is not simply:
        number_of_variables × number_of_hours

    Therefore we deliberately use small chunks.
    """

    # CDS v2 strictly enforces cost limits. 5 variables per chunk passes CDS checks.
    CHUNK_SIZE = 5

    chunks = [
        VARIABLES[i:i + CHUNK_SIZE]
        for i in range(0, len(VARIABLES), CHUNK_SIZE)
    ]

    days = calendar.monthrange(year, month)[1]
    hours = days * 24

    print(
        f"{year}-{month:02d}: {days} days, "
        f"{hours} hourly timestamps, "
        f"{len(chunks)} requests"
    )

    return chunks

def download_chunk(
    client: cdsapi.Client,
    year: int,
    month: int,
    variables: list[str],
    chunk_id: int,
    output_dir: Path,
) -> Path:

    days = calendar.monthrange(year, month)[1]
    day_list = [f"{d:02d}" for d in range(1, days + 1)]

    out = output_dir / (
        f"era5land_{year}_{month:02d}_chunk{chunk_id:02d}.nc"
    )
    partial = out.with_suffix(out.suffix + ".part")

    if out.exists() and out.stat().st_size > 0:
        try:
            with xr.open_dataset(out):
                pass
            print(f"  SKIP existing: {out}")
            return out
        except Exception as exc:
            print(f"  Existing file is invalid; redownloading: {out}")
            print(f"  Validation error: {exc}")
            out.unlink()

    request = {
        "variable": variables,
        "year": str(year),
        "month": f"{month:02d}",
        "day": day_list,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }

    print()
    print(f"  Downloading chunk {chunk_id}")
    print(f"  Variables ({len(variables)}):")
    for v in variables:
        print(f"    - {v}")
    print(f"  Output: {out}")

    for attempt in range(1, 4):

        try:

            if partial.exists():
                partial.unlink()

            client.retrieve(
                DATASET,
                request,
                str(partial),
            )

            if partial.exists() and partial.stat().st_size > 0:
                with xr.open_dataset(partial):
                    pass
                partial.replace(out)
                print("  ✓ Download successful")
                return out

            raise RuntimeError(
                "CDS returned but output file is missing or empty."
            )

        except Exception as exc:

            if partial.exists():
                partial.unlink()

            error_text = str(exc)

            print(
                f"  Attempt {attempt}/3 failed:\n"
                f"  {error_text}"
            )

            if "cost limits exceeded" in error_text.lower():

                raise RuntimeError(
                    "\nCDS rejected this request because its cost is "
                    "too large.\n"
                    f"Variables in this request: {len(variables)}\n"
                    "Reduce CHUNK_SIZE in split_variables_for_month()."
                ) from exc

            if attempt == 3:
                raise

            wait_time = 30 * attempt

            print(
                f"  Waiting {wait_time} seconds before retry..."
            )

            time.sleep(wait_time)

    return out

def merge_month(
    year: int,
    month: int,
    files: list[Path],
    output_dir: Path
) -> Path:
    """
    Merge all downloaded variable chunks into one NetCDF file.
    """

    merged = output_dir / (
        f"era5land_karnataka_{year}_{month:02d}.nc"
    )

    if merged.exists() and merged.stat().st_size > 0:
        try:
            with xr.open_dataset(merged):
                pass
            print(f"  SKIP existing merged file: {merged}")
            return merged
        except Exception as exc:
            print(f"  Existing merged file is invalid; rebuilding: {merged}")
            print(f"  Validation error: {exc}")
            merged.unlink()

    print()
    print(f"  Merging {len(files)} chunks")
    print(f"  Output: {merged}")

    datasets = []

    try:
        for file in files:
            print(f"    Opening: {file.name}")
            datasets.append(xr.open_dataset(file))

        print("  Combining datasets...")

        combined = xr.merge(
            datasets,
            compat="override",
            join="exact"
        )

        print("  Saving merged NetCDF...")

        temporary_merged = merged.with_suffix(merged.suffix + ".part")
        if temporary_merged.exists():
            temporary_merged.unlink()
        combined.to_netcdf(temporary_merged)
        temporary_merged.replace(merged)

        combined.close()

    finally:
        for ds in datasets:
            ds.close()

    print(f"  ✓ Merged successfully: {merged}")

    return merged

def download_month(client: cdsapi.Client, year: int, month: int, root: Path) -> Path:
    month_dir = root / f"{year}" / f"{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_variables_for_month(year, month)
    files: list[Path] = []

    for chunk_id, variables in enumerate(chunks, start=1):
        path = download_chunk(
            client,
            year,
            month,
            variables,
            chunk_id,
            month_dir,
        )
        files.append(path)

    return merge_month(year, month, files, month_dir)


def export_monthly_csv(nc_file: Path) -> Path:
    """
    Export to CSV.

    WARNING: A Karnataka bounding box at 0.1° has many grid cells.
    A full month of hourly gridded data can therefore be very large.
    Use NetCDF for the main dataset; CSV is mainly for small tests/ML samples.
    """
    csv_file = nc_file.with_suffix(".csv")
    print(f"  Exporting CSV: {csv_file}")

    ds = xr.open_dataset(nc_file)
    try:
        df = ds.to_dataframe().reset_index()
        df.to_csv(csv_file, index=False)
    finally:
        ds.close()

    return csv_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=list(range(1, 13)),
        help="Months to download, e.g. --months 6 or --months 6 7 8",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ERA5_Karnataka",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export each merged monthly NetCDF to CSV. Not recommended for large runs.",
    )

    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")

    if any(m < 1 or m > 12 for m in args.months):
        raise ValueError("months must be between 1 and 12")

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    # Save a copy of the exact configuration used.
    config = {
        "dataset": DATASET,
        "area": AREA,
        "max_fields": MAX_FIELDS,
        "variables": VARIABLES,
        "resolution": "0.1 degree",
        "temporal_resolution": "hourly",
    }
    (root / "download_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nERA5-Land Karnataka downloader")
    print("=" * 60)
    print(f"Dataset: {DATASET}")
    print(f"Area: {AREA}")
    print(f"Variables: {len(VARIABLES)}")
    print(f"Output: {root.resolve()}")
    print("=" * 60)

    client = cdsapi.Client()

    for year in range(args.start_year, args.end_year + 1):
        for month in sorted(set(args.months)):
            print(f"\n>>> {year}-{month:02d}")
            merged = download_month(client, year, month, root)

            if args.csv:
                export_monthly_csv(merged)

    print("\nDONE!")
    print(f"Files are in: {root.resolve()}")


if __name__ == "__main__":
    main()
