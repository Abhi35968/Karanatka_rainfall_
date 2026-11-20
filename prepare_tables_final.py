import os
import shutil
from pathlib import Path
import pandas as pd

src_tbl_dir = Path("d:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review/manuscript_tables")
out_tbl_dir = Path("d:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review_Final/manuscript_tables")
out_tbl_dir.mkdir(parents=True, exist_ok=True)

# Copy all CSVs from src to out
for f in src_tbl_dir.glob("*.csv"):
    shutil.copy2(f, out_tbl_dir / f.name)
    print(f"Copied table: {f.name}")

# Now apply specific refinements:

# 1. table4_heavy_rain_classification.csv
t4_path = out_tbl_dir / "table4_heavy_rain_classification.csv"
df_t4 = pd.read_csv(t4_path)

# Rename column from Threshold_Calibration_Source to Threshold_Selection_Basis
if "Threshold_Calibration_Source" in df_t4.columns:
    df_t4 = df_t4.rename(columns={"Threshold_Calibration_Source": "Threshold_Selection_Basis"})

# Update descriptions to use validation-selected decision threshold and grid-cell-specific
for col in ["Threshold_Selection_Basis", "Model_Name"]:
    if col in df_t4.columns:
        df_t4[col] = df_t4[col].astype(str)
        df_t4[col] = df_t4[col].str.replace("station-specific", "grid-cell-specific", regex=False)
        df_t4[col] = df_t4[col].str.replace("Station-Wise", "Grid-Cell", regex=False)
        df_t4[col] = df_t4[col].str.replace("Frozen threshold", "Validation-selected decision threshold", regex=False)
        df_t4[col] = df_t4[col].str.replace("tuned exclusively on 2023 Validation to maximize CSI", "chosen on 2023 Validation to maximize CSI", regex=False)

df_t4.to_csv(t4_path, index=False)
print("Updated table4_heavy_rain_classification.csv with Threshold_Selection_Basis and grid-cell terminology.")

# 2. table1_dataset_characteristics.csv
t1_path = out_tbl_dir / "table1_dataset_characteristics.csv"
df_t1 = pd.read_csv(t1_path)
for col in df_t1.columns:
    if df_t1[col].dtype == object:
        df_t1[col] = df_t1[col].str.replace("regional drought year", "validation partition", regex=False)
df_t1.to_csv(t1_path, index=False)
print("Verified table1_dataset_characteristics.csv neutral partition role.")

# 3. table8_rainfall_intensity_limitations.csv
t8_path = out_tbl_dir / "table8_rainfall_intensity_limitations.csv"
df_t8 = pd.read_csv(t8_path)
print("Table 8 rows:", len(df_t8))
df_t8.to_csv(t8_path, index=False)

print("Tables setup completed successfully.")

