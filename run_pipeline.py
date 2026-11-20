"""
Karnataka Rainfall End-to-End ERA5-Land Pipeline Orchestrator

Flow:
1. CDS Download / Sample Generation (5 years of ERA5-Land hourly data for Karnataka)
2. Preprocessing (Unit conversion, missing value handling, precipitation processing, quality checks)
3. ML-Ready Dataset Creation (Temporal/spatial aggregation, feature engineering, lag calculation, targets)
4. Model Training & Evaluation (XGBoost Regressor & Classifier, CSI, RMSE, Confusion Matrix, Plots)

Usage:
    # Full run with synthetic sample data for instant verification:
    python run_pipeline.py --mode demo

    # Full run with actual CDS download (Requires .cdsapirc):
    python run_pipeline.py --mode download --start-year 2019 --end-year 2023
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command_cmd(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"FAILED command: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Karnataka ERA5-Land Rainfall Pipeline Orchestrator")
    parser.add_argument("--mode", choices=["demo", "download"], default="demo", help="'demo' uses synthetic data for instant test; 'download' fetches real CDS data")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    args = parser.parse_args()

    python_exe = sys.executable

    print("\n" + "=" * 70)
    print("  KARNATAKA RAINFALL ERA5-LAND ML PIPELINE  ")
    print("=" * 70)

    # STEP 1: Data Acquisition
    if args.mode == "demo":
        print("\n[STEP 1/4] Generating Sample 5-Year ERA5 NetCDF files for Karnataka...")
        run_command_cmd([
            python_exe, "generate_sample_nc.py",
            "--output-dir", "ERA5_Karnataka",
            "--years", *[str(y) for y in range(args.start_year, args.end_year + 1)],
            "--months", "6", "7", "8"  # Monsoon months for sample
        ])
    else:
        print(f"\n[STEP 1/4] Downloading {args.start_year}-{args.end_year} ERA5-Land data from CDS...")
        run_command_cmd([
            python_exe, "download_era5_karnataka.py",
            "--start-year", str(args.start_year),
            "--end-year", str(args.end_year),
            "--months", *[str(m) for m in args.months],
            "--output", "ERA5_Karnataka"
        ])

    # STEP 2: Preprocessing, Plotting & CSV Export (39 Columns)
    print("\n[STEP 2/4] Preprocessing NetCDF, Verification, Plotting & 39-Column CSV Export...")
    run_command_cmd([
        python_exe, "process_verify_and_plot.py"
    ])

    # STEP 3: ML Dataset Generation
    print("\n[STEP 3/4] Creating Feature-Engineered ML-Ready Dataset...")
    run_command_cmd([
        python_exe, "create_ml_dataset.py",
        "--input-csv", "CSVs/karnataka_rainfall_2019_2025_master.csv",
        "--output-file", "ML_Ready_Karnataka_Rainfall.csv"
    ])


    # STEP 4: Machine Learning Model Training
    print("\n[STEP 4/4] Training & Evaluating ML Models (Regression & Classification)...")
    run_command_cmd([
        python_exe, "train_rainfall_model.py",
        "--input-file", "ML_Ready_Karnataka_Rainfall.csv",
        "--output-dir", "Model_Outputs"
    ])

    print("\n" + "=" * 70)
    print("  [OK] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!  ")
    print("  Check 'Model_Outputs/' directory for evaluation metrics and plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
